# League Draft Lab collector - Windows auto-start management.
#
# Installs/removes/controls a Task Scheduler job that runs the headless
# collector daemon hidden at every Windows sign-in (pythonw.exe), restarts it
# on failure, and applies no execution time limit (the daemon is meant to run
# for as long as the PC is on).
#
# Usage (also wrapped by install_/remove_/stop_/status_ .bat files):
#   powershell -NoProfile -ExecutionPolicy Bypass -File collector_autostart.ps1 -Action install
#   ... -Action remove | start | stop | status

param(
    [Parameter(Position = 0)]
    [ValidateSet("install", "remove", "start", "stop", "status")]
    [string]$Action = "status"
)

$ErrorActionPreference = "Stop"
$TaskName = "LeagueDraftLabCollector"
$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $AppDir ".venv\Scripts\pythonw.exe"
$DaemonScript = Join-Path $AppDir "collector_daemon.py"
$LogFile = Join-Path $AppDir "scraper.log"

function Assert-Setup {
    if (-not (Test-Path $PythonExe)) {
        throw "Virtual environment not found at $PythonExe. Run setup_windows.bat first."
    }
}

function Get-DaemonProcesses {
    Get-CimInstance Win32_Process -Filter "Name='pythonw.exe' OR Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*collector_daemon.py*" }
}

switch ($Action) {

    "install" {
        Assert-Setup
        $taskAction = New-ScheduledTaskAction -Execute $PythonExe -Argument "`"$DaemonScript`"" -WorkingDirectory $AppDir
        $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 10 -RestartInterval (New-TimeSpan -Minutes 1) -StartWhenAvailable -MultipleInstances IgnoreNew
        Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger -Settings $settings -Description "League Draft Lab headless Riot data collector. Runs hidden at sign-in; writes *.sync.zip bundles for the draft app." -Force | Out-Null
        Start-ScheduledTask -TaskName $TaskName
        Write-Host ""
        Write-Host "Auto-start installed, and the collector is running right now."
        Write-Host "  Task   : $TaskName (hidden, starts at every sign-in)"
        Write-Host "  Log    : $LogFile"
        Write-Host "Remove it any time with remove_autostart_windows.bat."
    }

    "remove" {
        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($task) {
            Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
            Write-Host "Auto-start task '$TaskName' removed and the collector stopped."
        }
        else {
            Write-Host "No task named '$TaskName' is installed."
        }
    }

    "start" {
        Assert-Setup
        Start-ScheduledTask -TaskName $TaskName
        Write-Host "Collector task started (hidden window). Log: $LogFile"
    }

    "stop" {
        $found = $false
        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($task) {
            Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
            $found = $true
        }
        # Also stop daemons started manually (e.g. a visible console window).
        Get-DaemonProcesses | ForEach-Object {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            $found = $true
        }
        if ($found) { Write-Host "Collector stopped." }
        else { Write-Host "The collector was not running." }
    }

    "status" {
        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($task) {
            $info = $task | Get-ScheduledTaskInfo
            Write-Host "Task        : $TaskName ($($task.State))"
            Write-Host "Last run    : $($info.LastRunTime)"
            Write-Host "Last result : $($info.LastTaskResult)"
        }
        else {
            Write-Host "Task        : $TaskName is NOT installed (run install_autostart_windows.bat)."
        }
        $procs = Get-DaemonProcesses
        if ($procs) {
            Write-Host "Daemon      : running (PID $($procs.ProcessId -join ', '))"
        }
        else {
            Write-Host "Daemon      : not running"
        }
        if (Test-Path $LogFile) {
            Write-Host ""
            Write-Host "Last log lines ($LogFile):"
            Get-Content $LogFile -Tail 5
        }
    }
}
