param(
    [Parameter(Mandatory)][string[]]$ChildArguments,
    [Parameter(Mandatory)][string]$LogBase,
    [int]$Seconds = 900,
    [double]$MaxMiB = 2048
)
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$logPath = [IO.Path]::GetFullPath($LogBase, $root)
$receiptPath = $logPath + '.launch.json'
foreach ($path in @($receiptPath, ($logPath + '.stdout'), ($logPath + '.stderr'))) {
    if (Test-Path -LiteralPath $path) { throw "One-shot artifact already exists: $path" }
}
$timer = [Diagnostics.Stopwatch]::StartNew()
$startedUnix = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
$report = [ordered]@{
    status = 'failed'; started_utc = [DateTime]::UtcNow.ToString('o')
    started_unix = $startedUnix; deadline_unix = $startedUnix + $Seconds
    command = @((Join-Path $root '.venv/Scripts/python.exe')) + $ChildArguments
    seconds_limit = $Seconds; tree_peak_sum_mib_limit = $MaxMiB
    memory_method = 'Maximum simultaneous sum of live owned process peak working sets, sampled every 500ms. Brief unobserved children may be missed.'
    launcher_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
}
$childProcess = $null
$streams = @()
$copies = @()
$observed = @{}
$treePeak = 0.0
try {
    # Verify the monitoring facility before launching any training.
    Get-CimInstance Win32_Process -Filter "ProcessId=$PID" -Property ProcessId | Out-Null
    $startInfo = [Diagnostics.ProcessStartInfo]::new($report.command[0])
    $startInfo.WorkingDirectory = $root
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.Environment['HUMANBRAIN_RUN_START_UNIX'] = $startedUnix.ToString([Globalization.CultureInfo]::InvariantCulture)
    $startInfo.Environment['HUMANBRAIN_RUN_DEADLINE_UNIX'] = ($startedUnix + $Seconds).ToString([Globalization.CultureInfo]::InvariantCulture)
    foreach ($argument in $ChildArguments) { $startInfo.ArgumentList.Add($argument) }
    $childProcess = [Diagnostics.Process]::Start($startInfo)
    $report.child_pid = $childProcess.Id
    $rootStarted = $childProcess.StartTime.ToUniversalTime()
    $streams = @([IO.File]::Open($logPath + '.stdout', 'CreateNew'), [IO.File]::Open($logPath + '.stderr', 'CreateNew'))
    $copies = @($childProcess.StandardOutput.BaseStream.CopyToAsync($streams[0]), $childProcess.StandardError.BaseStream.CopyToAsync($streams[1]))
    while (-not $childProcess.HasExited) {
        if ($timer.Elapsed.TotalSeconds -gt $Seconds) { throw 'Outer wall budget exhausted' }
        $table = @(Get-CimInstance Win32_Process -Property ProcessId,ParentProcessId,CreationDate)
        $owned = [Collections.Generic.HashSet[int]]::new()
        [void]$owned.Add($childProcess.Id)
        do {
            $added = $false
            foreach ($row in $table) {
                if ($row.CreationDate.ToUniversalTime() -ge $rootStarted -and $owned.Contains([int]$row.ParentProcessId)) {
                    if ($owned.Add([int]$row.ProcessId)) { $added = $true }
                }
            }
        } while ($added)
        $sum = 0.0
        foreach ($processId in $owned) {
            $entry = Get-Process -Id $processId -ErrorAction SilentlyContinue
            if ($null -eq $entry) { continue }
            try {
                $identity = "$processId/$($entry.StartTime.ToUniversalTime().Ticks)"
                $peak = $entry.PeakWorkingSet64 / 1MB
                $sum += $peak
                if (-not $observed.ContainsKey($identity) -or $peak -gt $observed[$identity]) { $observed[$identity] = $peak }
            } finally { $entry.Dispose() }
        }
        $treePeak = [Math]::Max($treePeak, $sum)
        if ($sum -gt $MaxMiB) { throw 'Observed process-tree peak sum budget exhausted' }
        [void]$childProcess.WaitForExit(500)
    }
    $report.exit_code = $childProcess.ExitCode
    if ($childProcess.ExitCode -ne 0) { throw "Experiment child exited $($childProcess.ExitCode)" }
    if (-not [Threading.Tasks.Task]::WhenAll([Threading.Tasks.Task[]]$copies).Wait(5000)) { throw 'Log finalization timed out' }
    if ($timer.Elapsed.TotalSeconds -gt $Seconds) { throw 'Outer wall budget exhausted during finalization' }
    $report.status = 'passed'
} catch {
    $report.error = $_.Exception.Message
    # Match process creation identities before cleanup; PID reuse must not select unrelated work.
    foreach ($identity in $observed.Keys) {
        $parts = $identity.Split('/')
        $entry = Get-Process -Id ([int]$parts[0]) -ErrorAction SilentlyContinue
        if ($null -eq $entry) { continue }
        try {
            if ($entry.StartTime.ToUniversalTime().Ticks -eq [long]$parts[1]) { $entry.Kill() }
        } catch [InvalidOperationException] {} finally { $entry.Dispose() }
    }
    if ($null -ne $childProcess -and -not $childProcess.HasExited) {
        $childProcess.Kill($true)
        [void]$childProcess.WaitForExit(5000)
    }
} finally {
    try { [void][Threading.Tasks.Task]::WhenAll([Threading.Tasks.Task[]]$copies).Wait(1000) } catch {}
    foreach ($stream in $streams) { $stream.Dispose() }
    $report.elapsed_seconds = $timer.Elapsed.TotalSeconds
    $report.finished_utc = [DateTime]::UtcNow.ToString('o')
    $report.observed_process_peaks_mib = $observed
    $report.observed_tree_peak_sum_mib = $treePeak
    if ($null -ne $childProcess) { $childProcess.Dispose() }
    [IO.File]::WriteAllText($receiptPath, ($report | ConvertTo-Json -Depth 5) + "`n", [Text.UTF8Encoding]::new($false))
    $report | ConvertTo-Json -Depth 5
}
if ($report.status -ne 'passed') { exit 1 }
