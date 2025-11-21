# =============================================================================
# QUICHE CLIENT AUTOMATED CAPTURE (UPGRADED)
# =============================================================================

# Stop script on first error for easier debugging
$ErrorActionPreference = 'Stop'

# --- Configuration ---
$QuicheProjectDir   = "C:\Users\imets\UZH\Masters_project\quiche"
$baseDir            = "C:\Users\imets\UZH\Masters_project\synthetic_network_data_gen"
# --- FIX: Set the correct capture interface. Find this with `dumpcap -D` ---
$CaptureInterface   = "\Device\NPF_Loopback"
$numRuns            = 30

# --- Paths (now using robust Join-Path) ---
$pcapDir    = Join-Path -Path $baseDir -ChildPath "pcap_files\quiche"
$jsonDir    = Join-Path -Path $baseDir -ChildPath "captures_json\quiche"
# --- NEW: A separate directory for individual keylog files ---
$keylogDir  = Join-Path -Path $baseDir -ChildPath "keylog_files\quiche"
$MasterKeylogFile = Join-Path -Path $baseDir -ChildPath "akos_master_sslkeylogfile.log"

# --- Executable Paths ---
$ServerExe = Join-Path -Path $QuicheProjectDir -ChildPath "target\debug\quiche-server.exe"
$ClientExe = Join-Path -Path $QuicheProjectDir -ChildPath "target\debug\quiche-client.exe"

# --- Pre-run Setup ---
# Create all necessary directories
New-Item -ItemType Directory -Force -Path $pcapDir, $jsonDir, $keylogDir | Out-Null

# --- Start the QUIC Server in the background ---
Write-Host "--> Starting the QUICHE server in the background..." -ForegroundColor Green
# Note: The server arguments assume the default quiche app structure. Adjust if needed.
$serverProcess = Start-Process -FilePath $ServerExe `
                               -ArgumentList "--listen 127.0.0.1:4433 --root apps/src/html --cert apps/src/bin/cert.crt --key apps/src/bin/cert.key" `
                               -WorkingDirectory $QuicheProjectDir `
                               -NoNewWindow -PassThru
Write-Host "--> Server started with PID: $($serverProcess.Id)"
Start-Sleep -Seconds 3 # Give the server a moment to initialize

# The try...finally block ensures the server is ALWAYS stopped, even if the script fails.
try {
    Write-Host "`n Starting $numRuns capture runs..."

    for ($i = 1; $i -le $numRuns; $i++) {
        $pcapFile = Join-Path -Path $pcapDir -ChildPath "quiche_capture_$i.pcap"
        $jsonFile = Join-Path -Path $jsonDir -ChildPath "quiche_capture_$i.json"
        # --- NEW: Create a unique keylog file for this specific run ---
        $keylogFile = Join-Path -Path $keylogDir -ChildPath "quiche_capture_$i.log"

        Write-Host "`n--- Starting capture $i → $pcapFile ---" -ForegroundColor Yellow

        # --- FIX: Start dumpcap on the correct loopback interface ---
        $dumpcapArgs = "-i `"$CaptureInterface`" -w `"$pcapFile`" -f `"udp port 4433`""
        $dumpcapProcess = Start-Process -FilePath "C:\Program Files\Wireshark\dumpcap.exe" -ArgumentList $dumpcapArgs -NoNewWindow -PassThru
        Start-Sleep -Seconds 1

        Write-Host " Running quiche-client.exe for run $i..."
        # --- FIX: Use the unique keylog file for this run ---
        $env:SSLKEYLOGFILE = $keylogFile
        
        & $ClientExe `
            "https://127.0.0.1:4433/index.html" `
            --no-verify `
            --enable-active-migration `
            --perform-migration `
            --source-ip 127.0.0.2 `
            --new-ip 127.0.0.3
        
        # Clean up the environment variable after use
        Remove-Item Env:SSLKEYLOGFILE
        Start-Sleep -Seconds 1

        Write-Host " Stopping capture..."
        Stop-Process -Id $dumpcapProcess.Id -Force
        Start-Sleep -Seconds 1
        
        Write-Host " Converting to decrypted JSON..."
        # --- FINAL WORKING COMMAND ---
        # This uses 'decode as' to tell tshark that the UDP port contains QUIC.
        # Tshark is then smart enough to see the 'h3' ALPN and dissect HTTP/3 automatically.
        $tsharkArgs = @(
            "-r", "`"$pcapFile`"",
            "-d", "udp.port==4433,quic",
            "-o", "tls.keylog_file:`"$keylogFile`"",
            "-T", "json"
        )

        & "C:\Program Files\Wireshark\tshark.exe" $tsharkArgs > $jsonFile

        Write-Host "Done: $pcapFile → $jsonFile"
    }
}
finally {
    # This code will run whether the loop finishes or fails, ensuring no orphaned processes.
    Write-Host "`n--> Stopping the QUICHE server (PID: $($serverProcess.Id))..." -ForegroundColor Red
    Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
    Write-Host "--> Server stopped."
}

# --- NEW: Combine all individual keylog files into one master file ---
Write-Host "`n--> Combining all keylog files into a single master file..." -ForegroundColor Cyan
$individualLogs = Get-ChildItem -Path $keylogDir -Filter "*.log"
# Check if any log files were actually created before trying to combine them
if ($individualLogs) {
    Get-Content -Path $individualLogs.FullName | Set-Content -Path $MasterKeylogFile
    Write-Host "--> Master keylog file created at: $MasterKeylogFile"
} else {
    Write-Warning "No individual keylog files were found to combine."
}


Write-Host "`n All runs complete! Files saved in $jsonDir"