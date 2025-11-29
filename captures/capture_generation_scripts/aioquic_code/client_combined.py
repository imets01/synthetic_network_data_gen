import argparse
import asyncio
import logging
from typing import Dict, Optional, cast

from aioquic.asyncio import QuicConnectionProtocol
from aioquic.h3.connection import H3_ALPN, H3Connection
from aioquic.h3.events import DataReceived, H3Event, HeadersReceived
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.connection import QuicConnection
from aioquic.quic.events import ConnectionTerminated, PingAcknowledged, QuicEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLIENT_IP_1 = "127.0.0.2"
CLIENT_IP_2 = "127.0.0.3"
SERVER_ADDR = ("127.0.0.1", 4433)

class MigratingHttpClient(QuicConnectionProtocol):
    def __init__(self, *, quic: QuicConnection):
        super().__init__(quic=quic)
        self._http: Optional[H3Connection] = H3Connection(self._quic)
        self._stream_buffers: Dict[int, bytearray] = {}
        self._stream_waiters: Dict[int, asyncio.Future[bytes]] = {}
        self._migration_waiter: Optional[asyncio.Event] = None

    def h3_event_received(self, event: H3Event):
        if event.stream_id in self._stream_waiters:
            waiter = self._stream_waiters[event.stream_id]
            if waiter.done(): return
            if isinstance(event, HeadersReceived):
                self._stream_buffers[event.stream_id] = bytearray()
            elif isinstance(event, DataReceived):
                self._stream_buffers[event.stream_id].extend(event.data)
                if event.stream_ended:
                    response_data = bytes(self._stream_buffers.pop(event.stream_id))
                    if not waiter.done(): waiter.set_result(response_data)

    def quic_event_received(self, event: QuicEvent):
        if isinstance(event, PingAcknowledged):
            if self._migration_waiter: self._migration_waiter.set()
        elif isinstance(event, ConnectionTerminated):
            for waiter in self._stream_waiters.values():
                if not waiter.done(): waiter.set_exception(ConnectionError("Connection terminated"))
        if self._http is not None:
            for h3_event in self._http.handle_event(event):
                self.h3_event_received(h3_event)

    async def get(self, url: str, migrate_during: bool = False) -> bytes:
        stream_id = self._quic.get_next_available_stream_id()
        waiter = asyncio.get_event_loop().create_future()
        self._stream_waiters[stream_id] = waiter

        self._http.send_headers(
            stream_id=stream_id,
            headers=[
                (b":method", b"GET"),
                (b":scheme", b"https"),
                (b":authority", b"localhost:4433"),
                (b":path", url.encode())
            ],
            end_stream=True,
        )
        # ------------------------------------
        
        self.transmit()
        if migrate_during:
            asyncio.create_task(self.trigger_migration_after_delay(0.05))
        return await asyncio.wait_for(waiter, timeout=10)

    async def trigger_migration_after_delay(self, delay: float):
        await asyncio.sleep(delay)
        await self.migrate()

    async def migrate(self):
        logger.info("\n--- Starting connection migration... ---\n")
        self._migration_waiter = asyncio.Event()
        loop = asyncio.get_running_loop()
        new_transport, _ = await loop.create_datagram_endpoint(lambda: self, local_addr=(CLIENT_IP_2, 0))
        self._transport = new_transport
        self._quic.change_connection_id()
        self._quic.send_ping(uid=1)
        self.transmit()
        await asyncio.wait_for(self._migration_waiter.wait(), timeout=5)
        self._migration_waiter = None
        logger.info("Path validation complete. Migration successful.")

async def main(scenario: str):
    keylog_file = open("sslkeylog.log", "a", buffering=1)
    configuration = QuicConfiguration(alpn_protocols=H3_ALPN, is_client=True, verify_mode=0)
    configuration.secrets_log_file = keylog_file
    connection = QuicConnection(configuration=configuration)
    
    transport, protocol = await asyncio.get_event_loop().create_datagram_endpoint(
        lambda: MigratingHttpClient(quic=connection), local_addr=(CLIENT_IP_1, 0)
    )
    client = cast(MigratingHttpClient, protocol)
    
    try:
        logger.info(f"--- RUNNING SCENARIO: {scenario.upper()} ---")
        client._quic.connect(SERVER_ADDR, now=asyncio.get_event_loop().time())
        client.transmit()
        await client.wait_connected()
        logger.info("QUIC connection established")

        if scenario == "before":
            await client.migrate()
            response_body = await client.get("/index.html")
            print(f"\nResponse received ({len(response_body)} bytes)")

        elif scenario == "during":
            response_body = await client.get("/index.html", migrate_during=True)
            print(f"\nResponse received ({len(response_body)} bytes)")

        elif scenario == "after":
            response_body = await client.get("/index.html")
            print(f"\nResponse received ({len(response_body)} bytes)")
            await client.migrate()
            await asyncio.sleep(1)

        logger.info(f"--- SCENARIO '{scenario.upper()}' TEST SUCCESSFUL ---")

    except Exception as e: logger.error(f"Execution failed: {e}", exc_info=True)
    finally:
        client.close()
        await client.wait_closed()
        keylog_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified QUIC Migration Client")
    parser.add_argument(
        "scenario",
        choices=["before", "during", "after"],
        help="The migration scenario to run."
    )
    args = parser.parse_args()
    asyncio.run(main(scenario=args.scenario))