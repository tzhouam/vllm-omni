# Default: verify only small metadata files. No model is launched without -Execute.
# Current user preference is to keep models unloaded; do not pass -Execute until changed.
[CmdletBinding()]
param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$rootPath = 'C:\Users\zhout\w2\strata-omni'
$launch = "$rootPath\runs\q4-image-execution-8g-20261013-v6\launch.json"
$freeze = "$rootPath\runs\q4-image-lifecycle-input-8g-20261013-v6\freeze.json"
$runner = "$rootPath\prototypes\image-lifecycle-combined-v2-20261012\run_strata_image_lifecycle.py"
$reference = "$rootPath\runs\ista-q2-visioncpu-v3\image-lifecycle-v1\reference.json"
$outputPath = "$rootPath\runs\q4-image-lifecycle-8g-20261013-v6"
$freezeSha = 'c213177e39c0dfe4a18cd181a9d30d49f71e17a15f4620acc95e4ffa48477f43'
$expected = @{
    $launch = '21970f20a9644b9f6b3feab6d848bdea32c9aeacd5b4209ba937170f872c3056'
    $freeze = $freezeSha
    $runner = '25b74a73ffd1ac57cbaa84a3271101976defc8d7171d0cf5d1a3777a2e85d5b9'
    $reference = 'ae90056198d16d861d4cc9139e4df68684533c74a38b55606f2ed267af19f3ed'
}
foreach ($entry in $expected.GetEnumerator()) {
    if ((Get-FileHash -LiteralPath $entry.Key -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) {
        throw "Input changed: $($entry.Key)"
    }
}
if (-not $Execute) {
    [ordered]@{ status = 'metadata_hashes_verified_not_executed'; model_executed = $false; files = $expected.Count; output_exists = (Test-Path -LiteralPath $outputPath) } | ConvertTo-Json -Compress
    return
}
if (Test-Path -LiteralPath $outputPath) { throw 'Fresh lifecycle output directory required; preserve existing evidence' }
# The runner additionally rechecks current source bindings, admission and process ownership.
& 'C:\Users\zhout\w2\omni029venv\Scripts\python.exe' -I -B -X utf8 $runner --launch $launch --reference $reference --freeze $freeze --expected-freeze-sha256 $freezeSha --output $outputPath
if ($LASTEXITCODE -ne 0) { throw "Lifecycle failed with exit code $LASTEXITCODE; retain output for diagnosis" }
