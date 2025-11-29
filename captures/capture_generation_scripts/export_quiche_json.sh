#!/bin/bash

# Define your paths (use Windows paths mounted under /mnt)
input_folder="/mnt/c/Users/imets/UZH/Masters_project/synthetic_network_data_gen/pcap_files/quiche"
output_folder="/mnt/c/Users/imets/UZH/Masters_project/synthetic_network_data_gen/captures_json/quiche"
keylog_file="/mnt/c/Users/imets/UZH/Masters_project/synthetic_network_data_gen/pcap_files/quiche/sslkeylogfile.log"

# Make sure output folder exists
mkdir -p "$output_folder"

# Check if keylog file exists
if [ ! -f "$keylog_file" ]; then
    echo "⚠️  Warning: SSL keylog file not found at $keylog_file"
    echo "Exporting without decryption - CRYPTO frames may not be visible"
fi

# Loop through all PCAPs named quiche_capture_*.pcap
for f in "$input_folder"/quiche_capture_*.pcap; do
    filename=$(basename "$f" .pcap)
    output_file="$output_folder/$filename.json"
    if [ -f "$output_file" ]; then
        echo "Skipping $filename.json (already exists)"
        continue
    fi
    echo "Exporting $filename.pcap → $output_file (with SSL decryption)"

    # Use tshark with a display filter for QUIC and the TLS keylog file.
    # The -Y "quic" filter ensures we only process packets dissected as QUIC.
    # Removed the unnecessary tls.desegment options for TCP.
    tshark -r "$f" \
           -o "tls.keylog_file:$keylog_file" \
           -T json > "$output_file"

done

echo "✅ All captures exported to $output_folder"