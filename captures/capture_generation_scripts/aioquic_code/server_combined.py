import argparse
import asyncio
import logging
import os
from typing import Optional

from aioquic.asyncio import QuicConnectionProtocol, serve
from aioquic.h3.connection import H3_ALPN, H3Connection
from aioquic.h3.events import H3Event, HeadersReceived
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import QuicEvent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server configuration
SERVER_ADDR = ("127.0.0.1", 4433)
CERT_FILE = "ssl_cert.pem"
KEY_FILE = "ssl_key.pem"
ROOT_PATH = os.path.dirname(__file__) or "."

class UnifiedHttpServerProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs):
        # Pop custom arguments before calling the parent constructor
        self._slow_mode = kwargs.pop("slow", False)
        self._chunk_size = kwargs.pop("chunk_size", 100)  # Store chunk size
        super().__init__(*args, **kwargs)
        self._http: Optional[H3Connection] = H3Connection(self._quic)

    def quic_event_received(self, event: QuicEvent):
        if self._http is not None:
            for h3_event in self._http.handle_event(event):
                self.h3_event_received(h3_event)

    def h3_event_received(self, event: H3Event):
        if isinstance(event, HeadersReceived):
            headers = {k.decode(): v.decode() for k, v in event.headers}
            if headers.get(":method") == "GET" and headers.get(":path") == "/index.html":
                if self._slow_mode:
                    asyncio.create_task(self.send_slow_response(event.stream_id))
                else:
                    asyncio.create_task(self.send_fast_response(event.stream_id))

    async def send_fast_response(self, stream_id: int):
        """Serves the entire file at once."""
        logger.info(f"Client requested /index.html on stream {stream_id}. Sending response quickly...")
        try:
            filepath = os.path.join(ROOT_PATH, "index.html")
            with open(filepath, "rb") as f:
                body = f.read()
            
            self._http.send_headers(stream_id=stream_id, headers=[(b":status", b"200")])
            self._http.send_data(stream_id=stream_id, data=body, end_stream=True)
            self.transmit()
            logger.info(f"Fast response sent for stream {stream_id}")
        except FileNotFoundError:
            logger.warning(f"File not found: {filepath}")
            self._http.send_headers(stream_id=stream_id, headers=[(b":status", b"404")])
            self._http.send_data(stream_id=stream_id, data=b"Not Found", end_stream=True)
            self.transmit()

    async def send_slow_response(self, stream_id: int):
        """Serves the file in delayed chunks using the configured chunk size."""
        logger.info(f"Client requested /index.html on stream {stream_id}. Sending response slowly...")
        try:
            with open(os.path.join(ROOT_PATH, "index.html"), "rb") as f:
                content = f.read()
            
            self._http.send_headers(stream_id=stream_id, headers=[(b":status", b"200")])
            self.transmit()

            # Use the chunk size passed during initialization
            chunk_size = self._chunk_size
            logger.info(f"Using chunk size: {chunk_size} bytes")

            for i in range(0, len(content), chunk_size):
                chunk = content[i : i + chunk_size]
                is_last_chunk = (i + chunk_size) >= len(content)
                
                logger.info(f"Sending chunk {i//chunk_size + 1}...")
                self._http.send_data(stream_id=stream_id, data=chunk, end_stream=is_last_chunk)
                self.transmit()
                
                if not is_last_chunk:
                    await asyncio.sleep(0.15)

            logger.info("Finished sending slow response.")
        except Exception as e:
            logger.error(f"Error sending response: {e}")

async def main(slow: bool, chunk_size: int):
    configuration = QuicConfiguration(alpn_protocols=H3_ALPN, is_client=False)
    configuration.load_cert_chain(CERT_FILE, KEY_FILE)
    
    # Use a lambda to pass both 'slow' and 'chunk_size' arguments to the protocol factory
    protocol_factory = lambda *args, **kwargs: UnifiedHttpServerProtocol(
        *args, slow=slow, chunk_size=chunk_size, **kwargs
    )

    await serve(
        host=SERVER_ADDR[0],
        port=SERVER_ADDR[1],
        configuration=configuration,
        create_protocol=protocol_factory,
    )
    logger.info(f"Listening on {SERVER_ADDR} (Slow mode: {slow}, Chunk size: {chunk_size})")
    await asyncio.Future()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified QUIC HTTP/3 Server")
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Serve the file in slow, delayed chunks to simulate a slow network."
    )
    # Add the new chunk-size argument
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="The size of data chunks (in bytes) to send in slow mode."
    )
    args = parser.parse_args()

    # Ensure the dummy file exists
    with open("index.html", "w") as f:
        f.write("This is a sample file for testing aioquic connection migration during a slow transfer. " * 10)
    
    try:
        # Pass the new argument to main
        asyncio.run(main(slow=args.slow, chunk_size=args.chunk_size))
    except KeyboardInterrupt:
        logger.info("Server shutting down.")