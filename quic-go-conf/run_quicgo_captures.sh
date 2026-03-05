#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# CONFIG
# ----------------------------
NUM_RUNS=3
ID_OFFSET=5925      
INTERFACE="lo0"
PORT=6121

# BASE FOLDERS
BASE_DIR="$HOME/Desktop/synthetic_network_data_gen"
PCAP_DIR="$BASE_DIR/captures/pcap/quicgo"
KEYLOG_DIR="$BASE_DIR/keylog_files/quicgo"

mkdir -p "$PCAP_DIR" "$KEYLOG_DIR" 

# Single keylog file 
KEYLOG_FILE="$KEYLOG_DIR/all_quicgo_keys.log"

# Clear before run
: > "$KEYLOG_FILE"

sudo -v

# ----------------------------
# CLEAN SLATE 
# ----------------------------
echo "Cleaning up old processes..."
sudo killall tcpdump 2>/dev/null || true
sleep 1

# QUIC-GO repo
QUICGO="$HOME/Desktop/Masters_Project/quic-go"

echo "Starting QUIC-GO server..."
cd "$QUICGO"
go run ./example/main.go \
  -bind 127.0.0.1:6121 \
  -www ./www \
  -cert ./certs/server.crt \
  -key ./certs/server.key &
SERVER_PID=$!
sleep 5

# ---------------------------
# MAIN LOOP
# ---------------------------
for i in $(seq 1 "$NUM_RUNS"); do
  # Calculate the specific ID for this file
  CURRENT_FILE_ID=$((i + ID_OFFSET))

  echo "-------------------------"
  echo "RUN $i / $NUM_RUNS (File ID: $CURRENT_FILE_ID)"
  echo "-------------------------"

  PCAP_FILE="$PCAP_DIR/quicgo_${CURRENT_FILE_ID}.pcap"
  rm -f "$PCAP_FILE"

  echo "--> Starting tcpdump..."
  sudo tcpdump -U -s 0 -i "$INTERFACE" -w "$PCAP_FILE" "udp port $PORT" >/dev/null 2>&1 &
  
  sleep 2

  echo "--> Running QUIC client..."
  SSLKEYLOGFILE="$KEYLOG_FILE" go run ./example/client_migration/main.go \
    --insecure \
    --keylog "$KEYLOG_FILE" \
    --perform-migration \
    --source-ip 127.0.0.2 \
    --new-ip 127.0.0.3 \
    "https://127.0.0.1:${PORT}/"

  echo "--> Client finished. Waiting for trailing packets..."
  sleep 3

  echo "--> Stopping tcpdump..."
  sudo killall -2 tcpdump 2>/dev/null || true
  
  while pgrep -x "tcpdump" > /dev/null; do 
      echo "    ...waiting for tcpdump to exit..."
      sleep 1
  done
  
  echo "--> DONE RUN $i"
done

echo "Stopping server..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

# Fix permissions
sudo chown -R "$(whoami)" "$PCAP_DIR"
sudo chown -R "$(whoami)" "$KEYLOG_DIR"

echo "All done!"
echo "PCAP files → $PCAP_DIR"
echo "Keylog file → $KEYLOG_FILE"