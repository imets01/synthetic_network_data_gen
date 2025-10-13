# =============================================================================
# PowerShell script to automate QUIC capture generation with quiche on Windows.
# VERSION 6: Final - With Individual and Combined Keylogs for Decryption
# =============================================================================

# Stop script on first error
$ErrorActionPreference = 'Stop'

# --- Configuration ---
$QuicheProjectDir = "C:\Users\vassa\Desktop\UZH\Masters Project\quiche\quiche"
$BaseDir = "C:\Users\vassa\Desktop\UZH\Masters Project\synthetic_network_data_gen"
$CaptureInterface = "\Device\NPF_Loopback" 

# --- Script Paths (automatically generated) ---
$PcapDir = Join-Path -Path $BaseDir -ChildPath "pcap_files\quiche"
$JsonDir = Join-Path -Path $BaseDir -ChildPath "captures_json\quiche"
# --- NEW: Define a separate directory for individual keylogs ---
$KeylogDir = Join-Path -Path $BaseDir -ChildPath "keylog_files\quiche"
# --- NEW: Define the path for the final combined keylog file ---
$MasterKeylogFile = Join-Path -Path $BaseDir -ChildPath "master_sslkeylogfile.log"

# --- Define paths to the final compiled executables ---
$ServerExe = Join-Path -Path $QuicheProjectDir -ChildPath "target\debug\quiche-server.exe"
$ClientExe = Join-Path -Path $QuicheProjectDir -ChildPath "target\debug\quiche-client.exe"

# --- Script Settings ---
$StartRun = 31
$EndRun = 60

# --- Pre-run Checks ---
if (-not (Test-Path -Path $QuicheProjectDir -PathType Container)) {
    Write-Host "FATAL ERROR: The quiche project directory was not found at '$QuicheProjectDir'." -ForegroundColor Red
    exit 1
}

# --- Script Start ---
# --- MODIFIED: Added KeylogDir to the list of directories to create ---
New-Item -ItemType Directory -Force -Path $PcapDir, $JsonDir, $KeylogDir

# --- STEP 1: Build the client and server executables ONCE ---
Write-Host "--> Building quiche binaries (client and server)..." -ForegroundColor Cyan
Push-Location $QuicheProjectDir
cargo build --bin quiche-server --bin quiche-client
Pop-Location
Write-Host "--> Build complete."

Write-Host "--> Starting the QUICHE server in the background..." -ForegroundColor Green
# --- STEP 2: Run the already-built server executable ---
$serverProcess = Start-Process -FilePath $ServerExe -ArgumentList "--listen 127.0.0.1:4433 --root apps/src/html --cert apps/src/bin/cert.crt --key apps/src/bin/cert.key --enable-active-migration" -WorkingDirectory $QuicheProjectDir -NoNewWindow -PassThru -RedirectStandardOutput "$BaseDir\server.log" -RedirectStandardError "$BaseDir\server.err.log"
Write-Host "--> Server started with PID: $($serverProcess.Id)"
Start-Sleep -Seconds 3

try {
    for ($i = $StartRun; $i -le $EndRun; $i++) {
        $PcapFile = Join-Path -Path $PcapDir -ChildPath "quiche_capture_${i}.pcap"
        $JsonFile = Join-Path -Path $JsonDir -ChildPath "quiche_capture_${i}.json"
        # --- NEW: Create a unique keylog file path for each run ---
        $KeylogFile = Join-Path -Path $KeylogDir -ChildPath "quiche_capture_${i}.log"

        Write-Host ""
        Write-Host "--- Run $i of $EndRun ---" -ForegroundColor Yellow

        Write-Host "--> Starting capture: $PcapFile"
        $dumpcapArgs = "-i `"$CaptureInterface`" -w `"$PcapFile`" -f `"udp port 4433`""
        $dumpcapProcess = Start-Process dumpcap.exe -ArgumentList $dumpcapArgs -NoNewWindow -PassThru
        Start-Sleep -Seconds 1

        Write-Host "--> Running client (Keylog: $KeylogFile)"
        # --- MODIFIED: The environment variable now points to the unique keylog file for this run ---
        $env:SSLKEYLOGFILE = $KeylogFile
        
        # --- STEP 3: Run the already-built client executable ---
        & $ClientExe https://127.0.0.1:4433/index.html --no-verify --enable-active-migration --perform-migration --source-ip 127.0.0.2 --new-ip 127.0.0.3
        
        Remove-Item Env:SSLKEYLOGFILE
        Start-Sleep -Seconds 1

        Write-Host "--> Stopping capture..."
        Stop-Process -Id $dumpcapProcess.Id -Force

        Write-Host "--> Converting to Decrypted JSON..."
        # --- MODIFIED: tshark uses the unique keylog file for this specific capture ---
        & tshark -r $PcapFile -o tls.keylog_file:"$KeylogFile" -T json > $JsonFile
        
        Write-Host "--> Done: $JsonFile"
    }
}
finally {
    # This block will still run to ensure the server is always stopped
    Write-Host ""
    Write-Host "--> Stopping the QUICHE server (PID: $($serverProcess.Id))..." -ForegroundColor Red
    Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
    Write-Host "--> Server stopped."
}

# --- NEW STEP 4: Combine all individual keylog files into one master file ---
Write-Host ""
Write-Host "--> Combining all keylog files into a single master file..." -ForegroundColor Cyan
$individualLogs = Get-ChildItem -Path $KeylogDir -Filter "*.log"
Get-Content -Path $individualLogs.FullName | Set-Content -Path $MasterKeylogFile
Write-Host "--> Master keylog file created at: $MasterKeylogFile"

Write-Host ""
Write-Host "--- Script Finished ---" -ForegroundColor Green