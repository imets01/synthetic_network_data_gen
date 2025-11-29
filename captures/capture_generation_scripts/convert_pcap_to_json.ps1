$ErrorActionPreference = 'Stop'

$BaseDir = "C:\Users\vassa\Desktop\UZH\Masters Project\synthetic_network_data_gen\captures"

$PcapDir = Join-Path -Path $BaseDir -ChildPath "pcap_files\quicgo"
$JsonDir = Join-Path -Path $BaseDir -ChildPath "captures_json\quicgo"

$KeylogFile = Join-Path -Path $BaseDir -ChildPath "keylog_files\all_quicgo_keys_5922_7422.log"


if (-not (Test-Path -Path $PcapDir)) {
    Write-Host "FATAL ERROR: The PCAP directory was not found at '$PcapDir'." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path -Path $KeylogFile)) {
    Write-Host "FATAL ERROR: The Keylog file was not found at '$KeylogFile'." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path -Path $JsonDir)) {
    New-Item -ItemType Directory -Force -Path $JsonDir | Out-Null
    Write-Host "Created output directory: $JsonDir" -ForegroundColor Gray
}

$pcapFiles = Get-ChildItem -Path $PcapDir -Filter "*.pcap"

foreach ($file in $pcapFiles) {

	$currentPcap = $file.FullName
	$jsonName = $file.BaseName + ".json"
	$currentJson = Join-Path -Path $JsonDir -ChildPath $jsonName
	Write-Host "Processing: $($file.Name)" -NoNewline


try {
        # Run tshark
        # -r: Input file
        # -o tls.keylog_file: The master keylog (tshark looks up the correct key automatically)
        # -T json: Output format
        
        # Note: We use cmd /c to handle the redirection (>) reliably in all PS versions, 
        # though strictly PS redirection works too.
        $tsharkArgs = "-r `"$currentPcap`" -o tls.keylog_file:`"$KeylogFile`" -T json"
        
        # Execute tshark and redirect output to JSON file
        cmd /c "tshark $tsharkArgs > `"$currentJson`""
        
        Write-Host " -> Done" -ForegroundColor Green
    }
    catch {
        Write-Host " -> ERROR" -ForegroundColor Red
        Write-Error $_
    }
}
Write-Host ""
Write-Host "--- Conversion Finished ---" -ForegroundColor Green