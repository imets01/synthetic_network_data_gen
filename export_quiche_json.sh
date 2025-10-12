#!/bin/bash

# Define your paths (use Windows paths mounted under /mnt)
input_folder="/mnt/c/Users/imets/UZH/Masters_project/synthetic_network_data_gen/pcap_files/quiche"
output_folder="/mnt/c/Users/imets/UZH/Masters_project/synthetic_network_data_gen/captures_json/quiche"

# Make sure output folder exists
mkdir -p "$output_folder"

# Loop through all PCAPs named quiche_capture_*.pcap
for f in "$input_folder"/quiche_capture_*.pcap; do
    filename=$(basename "$f" .pcap)
    output_file="$output_folder/$filename.json"
    if [ -f "$output_file" ]; then
        echo "Skipping $filename.json (already exists)"
        continue
    fi
    echo "Exporting $filename.pcap → $output_file"
    tshark -r "$f" -T json > "$output_file"
done


echo "✅ All captures exported to $output_folder"
