#!/bin/bash
# =============================================================================
# Bash script to automate QUIC capture generation with nginx and quiche-client
# =============================================================================

set -e

# --- Configuration ---
BASE_DIR="/mnt/c/Users/imets/UZH/Masters_project/synthetic_network_data_gen"
PCAP_DIR="$BASE_DIR/pcap_files/nginx-quiche"
JSON_DIR="$BASE_DIR/captures_json/nginx-quiche"
KEYLOG_DIR="$BASE_DIR/keylog_files/nginx-quiche"
MASTER_KEYLOG="$KEYLOG_DIR/master_sslkeylogfile.log"

# nginx configuration
NGINX_DIR="/mnt/c/Users/imets/UZH/Masters_project/nginx/nginx-1.29.3"
NGINX_BIN="$NGINX_DIR/objs/nginx"

# quiche client path
QUICHE_DIR="/mnt/c/Users/imets/UZH/Masters_project/quiche"
QUICHE_CLIENT="$QUICHE_DIR/target/debug/quiche-client"

# capture settings
NUM_RUNS=3000
START_RUN=8922

# --- Pre-run Checks ---
echo "🔍 Checking prerequisites..."

if [ ! -f "$NGINX_BIN" ]; then
    echo "❌ ERROR: nginx binary not found at $NGINX_BIN"
    exit 1
fi

if [ ! -d "$QUICHE_DIR" ]; then
    echo "❌ ERROR: quiche directory not found at $QUICHE_DIR"
    exit 1
fi

# Build quiche-client once if not already built
if [ ! -f "$QUICHE_CLIENT" ]; then
    echo "🔨 Building quiche-client (one-time setup)..."
    cd "$QUICHE_DIR"
    source $HOME/.cargo/env
    cargo build --bin quiche-client
    echo "✅ Build complete"
fi

# Check if nginx is already running
if pgrep -x "nginx" > /dev/null; then
    echo "✅ nginx is already running"
else
    echo "🚀 Starting nginx server..."
    cd "$NGINX_DIR"
    sudo ./objs/nginx
    sleep 2
    
    if pgrep -x "nginx" > /dev/null; then
        echo "✅ nginx started successfully"
    else
        echo "❌ ERROR: Failed to start nginx"
        exit 1
    fi
fi

# --- Setup IP aliases for migration ---
echo "🔧 Setting up IP aliases for connection migration..."
sudo ip addr add 127.0.0.2/8 dev lo 2>/dev/null || echo "   (127.0.0.2 already exists)"
sudo ip addr add 127.0.0.3/8 dev lo 2>/dev/null || echo "   (127.0.0.3 already exists)"

# --- Create directories ---
mkdir -p "$PCAP_DIR" "$JSON_DIR" "$KEYLOG_DIR"

# --- Create/Clear master keylog file ---
> "$MASTER_KEYLOG"
echo "📝 Using master keylog: $MASTER_KEYLOG"

# --- Main capture loop ---
echo ""
echo "🎯 Starting capture runs ($START_RUN to $((START_RUN + NUM_RUNS - 1)))..."
echo ""

for i in $(seq $START_RUN $((START_RUN + NUM_RUNS - 1))); do
    PCAP_FILE="$PCAP_DIR/${i}_nginx.pcap"
    JSON_FILE="$JSON_DIR/${i}_nginx.json"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Run $i of $((START_RUN + NUM_RUNS - 1))"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    echo "📡 Starting tcpdump capture → $PCAP_FILE"
    # Start tcpdump in the background and capture its PID directly
    sudo tcpdump -i lo -w "$PCAP_FILE" 'udp port 4433' > /dev/null 2>&1 &
    TCPDUMP_PID=$!
    
    sleep 0.5

    echo "🚀 Running quiche-client (using master keylog)..."
    
    # Run pre-built client directly - all runs append to same keylog
    SSLKEYLOGFILE="$MASTER_KEYLOG" "$QUICHE_CLIENT" \
        https://127.0.0.1:4433/index.html \
        --no-verify \
        --enable-active-migration \
        --perform-migration \
        --source-ip 127.0.0.2 \
        --new-ip 127.0.0.3 > /dev/null 2>&1

    # Small buffer to ensure trailing packets are caught
    sleep 0.5

    echo "🛑 Stopping tcpdump..."
    # Send SIGINT (Ctrl+C) to the specific tcpdump process
    sudo kill -2 "$TCPDUMP_PID" 2>/dev/null || true
    # Use 'wait' instead of a polling loop - this is much faster/cleaner
    wait "$TCPDUMP_PID" 2>/dev/null || true
    # Check if pcap file was created and has content
    # Check if pcap file exists
    if [ -f "$PCAP_FILE" ]; then
        # Check size quickly (Header is 24 bytes)
        FILE_SIZE=$(stat -c%s "$PCAP_FILE")
        
        if [ "$FILE_SIZE" -gt 24 ]; then
            # -T -m outputs CSV. 
            # tail -n 1 gets the last line (the data). 
            # cut -d ',' -f 2 gets the second column (the count).
            PACKET_COUNT=$(capinfos -T -m -c "$PCAP_FILE" 2>/dev/null | tail -n 1 | cut -d ',' -f 2)
            echo "📦 Captured $PACKET_COUNT packets"
        else
            echo "⚠️  WARNING: File created but contains 0 packets (Header only)"
        fi
    else
         echo "❌ ERROR: PCAP file was not created."
    fi

    echo "🔐 Converting to decrypted JSON with master keylog..."
    tshark -r "$PCAP_FILE" -o tls.keylog_file:"$MASTER_KEYLOG" -T json > "$JSON_FILE" 2>/dev/null

    echo "✅ Complete: $(basename $PCAP_FILE) → $(basename $JSON_FILE)"
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 All $NUM_RUNS captures completed successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 Output locations:"
echo "   PCAP files:   $PCAP_DIR"
echo "   JSON files:   $JSON_DIR"
echo "   Master keylog: $MASTER_KEYLOG"
echo ""

# Optional: Stop nginx if you want
# echo "🛑 Stopping nginx..."
# sudo pkill -x nginx
# echo "✅ nginx stopped"you want
# echo "🛑 Stopping nginx..."
# sudo pkill -x nginx
# echo "✅ nginx stopped"
