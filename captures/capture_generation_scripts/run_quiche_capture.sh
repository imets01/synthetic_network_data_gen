#!/bin/bash
set -e

# === Paths ===
BASE_DIR="/mnt/c/Users/imets/UZH/Masters_project/synthetic_network_data_gen"
PCAP_DIR="$BASE_DIR/pcap_files/quiche"
JSON_DIR="$BASE_DIR/captures_json/quiche"
KEYLOG_FILE="$PCAP_DIR/sslkeylogfile.log"

# quiche client project path in WSL
QUICHE_SRC="/home/imets/uzh/masters_project/quiche"

# number of runs
NUM_RUNS=30

mkdir -p "$PCAP_DIR" "$JSON_DIR"

# loop through multiple captures
for i in $(seq 1 $NUM_RUNS); do
    PCAP_FILE="$PCAP_DIR/quiche_capture_${i}.pcap"
    JSON_FILE="$JSON_DIR/quiche_capture_${i}.json"

    echo ""
    echo "📡 Starting capture $i → $PCAP_FILE"
    sudo tcpdump -i any udp port 4433 -w "$PCAP_FILE" &
    TCPDUMP_PID=$!
    sleep 1  # small delay to ensure tcpdump starts

    echo "🚀 Running client $i..."
    (cd "$QUICHE_SRC" && SSLKEYLOGFILE="$KEYLOG_FILE" cargo run --quiet --bin quiche-client -- \
      https://127.0.0.1:4433/index.html \
      --no-verify \
      --enable-active-migration \
      --perform-migration \
      --source-ip 127.0.0.2 \
      --new-ip 127.0.0.3)


    sleep 1

    echo "🛑 Stopping capture $i..."
    sudo kill $TCPDUMP_PID
    wait $TCPDUMP_PID 2>/dev/null || true

    echo "📦 Converting to JSON..."
    tshark -r "$PCAP_FILE" -T json > "$JSON_FILE"

    echo "✅ Done: $PCAP_FILE → $JSON_FILE"
done

echo "🎉 All runs complete!"
