@echo off
setlocal
cd /d "%~dp0"
title Zhilv Qiling - Contract Review Workspace
if not "%~1"=="" goto launch
rem Normal double-click enters the verified, resident-model defense profile.
call "%~dp0start-defense-web.cmd"
exit /b %errorlevel%
:launch
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" "%~dp0start_web.py" %*
  goto done
)
if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" (
  "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" "%~dp0start_web.py" %*
  goto done
)
where py >nul 2>nul
if not errorlevel 1 (
  py -3 "%~dp0start_web.py" %*
  goto done
)
where python >nul 2>nul
if not errorlevel 1 (
  python "%~dp0start_web.py" %*
  goto done
)
echo Python 3.10 or newer is required. Install Python, then run this file again.
pause
exit /b 1
:done
if errorlevel 1 pause
endlocal
