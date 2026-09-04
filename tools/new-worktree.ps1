[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Base,
    [Parameter(Mandatory)][string]$Branch,
    [Parameter(Mandatory)][string]$Destination,
    [Parameter(Mandatory)][string]$Owner,
    [Parameter(Mandatory)][string]$Scope
)

$ErrorActionPreference = 'Stop'
$sourceRoot = Split-Path -Parent $PSScriptRoot
Push-Location $sourceRoot
try {
    $gitRoot = & git rev-parse --show-toplevel
    if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve the source checkout.' }
    if ([IO.Path]::GetFullPath($gitRoot) -ne [IO.Path]::GetFullPath($sourceRoot)) {
        throw 'Run the helper from the Human Brain checkout.'
    }
    $baseCommit = & git rev-parse --verify --end-of-options "$Base^{commit}"
    if ($LASTEXITCODE -ne 0) { throw 'Base must resolve to an existing commit.' }
    if (-not $Branch.StartsWith('codex/')) { $Branch = "codex/$Branch" }
    & git check-ref-format --branch $Branch
    if ($LASTEXITCODE -ne 0) { throw 'Invalid branch name.' }
    if ([IO.Path]::IsPathRooted($Destination)) {
        $targetPath = [IO.Path]::GetFullPath($Destination)
    } else {
        $targetPath = [IO.Path]::GetFullPath((Join-Path $sourceRoot $Destination))
    }
    if (Test-Path -LiteralPath $targetPath) { throw "Destination already exists: $targetPath" }
    $dirty = & git status --porcelain --untracked-files=normal
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect source checkout status.' }
    Write-Host "Creating $Branch from committed base $baseCommit. Uncommitted files are excluded."
    & git worktree add -b $Branch -- $targetPath $baseCommit
    if ($LASTEXITCODE -ne 0) { throw 'Worktree creation failed; inspect git worktree list before retrying.' }
    $runtimePath = Join-Path $targetPath '.runtime'
    New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null
    @{
        source_checkout = $sourceRoot
        checkout = $targetPath
        base = $baseCommit
        branch = $Branch
        owner = $Owner
        scope = $Scope
        source_had_uncommitted_files = [bool]$dirty
        created_at = [DateTime]::UtcNow.ToString('o')
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtimePath 'worktree.json') -Encoding UTF8
    Write-Host "Worktree ready: $targetPath"
    Write-Host 'Run tools/setup.ps1 from that checkout to create its own environment.'
} finally {
    Pop-Location
}
