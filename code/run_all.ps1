# Windows version of run_all.sh: every experiment for every model, then all final statistics.
# Resumable: finished runs are skipped, so after an interruption just start it again.
# Usage (in an Anaconda / PowerShell window with the conda env active, from this folder):
#   powershell -ExecutionPolicy Bypass -File run_all.ps1          Logs: results\logs\<job>.log
# Hardware split: see run_all.sh (i9-14900, 9 parallel jobs x CPU_THREADS=4; override with $env:CPU_THREADS).
Set-Location $PSScriptRoot
$env:CUDA_VISIBLE_DEVICES = "-1"
if (-not $env:CPU_THREADS) { $env:CPU_THREADS = "4" }
New-Item -ItemType Directory -Force results\logs | Out-Null
$all = "tinycnn tinytcn tinyssm cnnlstm tinyeegnet tinytransformer musci2021"
$protocol = (python -c "import config; print(config.THRESHOLD_PROTOCOL)").Trim()
Write-Host "Threshold protocol: $protocol    CPU threads per job: $env:CPU_THREADS"

# 1. data self-check + data manifest (before the parallel jobs)
python overall.py check; if ($LASTEXITCODE -ne 0) { exit 1 }
python overall.py check --dataset sisfall; if ($LASTEXITCODE -ne 0) { exit 1 }

# 2. all training experiments in parallel
$jobs = @()
function Start-Job2($name, $command) {
    $p = Start-Process cmd.exe -ArgumentList "/c", "($command) > results\logs\$name.log 2>&1" -NoNewWindow -PassThru
    $null = $p.Handle                       # keeps the exit code readable after the process ends
    $script:jobs += [pscustomobject]@{ Name = $name; Process = $p }
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  started $name  (log: results\logs\$name.log)"
}
foreach ($m in "cnnlstm", "musci2021", "tinytransformer", "tinyssm", "tinytcn", "tinycnn", "tinyeegnet") {
    Start-Job2 "cv_$m" "python overall.py cv --models $m && python overall.py cv --dataset sisfall --models $m"
}
if ($protocol -eq "validation") {
    Start-Job2 "train" "python overall.py train --models $all --out_dir results\fixed_valthr"
} else {
    Start-Job2 "train" "python overall.py train --models $all && python overall.py deploy --models $all"
}
Start-Job2 "earlystop" "python overall.py earlystop"

$failed = $false
foreach ($j in $jobs) {
    $j.Process.WaitForExit()
    if ($j.Process.ExitCode -eq 0) { Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  finished $($j.Name)" }
    else { Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  FAILED $($j.Name), see results\logs\$($j.Name).log"; $failed = $true }
}
if ($failed) { Write-Host "Some jobs failed. Fix the problem and run run_all.ps1 again (finished runs are skipped)."; exit 1 }

# 3. final statistics over all saved runs
python overall.py report                   | Tee-Object results\logs\report_kfall.log
python overall.py report --dataset sisfall | Tee-Object results\logs\report_sisfall.log
python overall.py transfer                 | Tee-Object results\logs\transfer.log
python overall.py earlystop --report       | Tee-Object results\logs\earlystop_report.log
python overall.py analyze                  | Tee-Object results\logs\analyze.log
python plot_sa06.py                        | Tee-Object results\logs\plot_sa06.log
Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  ALL DONE."
