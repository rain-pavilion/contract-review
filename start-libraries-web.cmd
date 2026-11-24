@echo off
setlocal
cd /d "%~dp0"
title Zhilv Qiling - Contract, Law and User Libraries
set "ZLQL_LIBRARY_ROOT=%~dp0data\libraries"
set "ZLQL_CONTRACT_LIBRARY_PATH=%~dp0data\libraries\contracts.sqlite"
set "ZLQL_LAW_LIBRARY_PATH=%~dp0data\libraries\laws.sqlite"
set "ZLQL_USER_LIBRARY_PATH=%~dp0data\libraries\users.sqlite"
set "ZLQL_STORE_PATH=%~dp0data\libraries\legacy-workspace.sqlite"
set "ZLQL_MODEL_AUDIT_PATH=%~dp0data\libraries\model-audit.sqlite"
set "ZLQL_SUBJECT_FACTS_PATH=%~dp0data\libraries\subject-facts.json"
set "LIBRARIES_MODEL_MODE=--without-model"
rem Explicit model flags replace the default; argparse makes them mutually exclusive.
:scan_model_mode
if "%~1"=="" goto launch
if /i "%~1"=="--with-model" set "LIBRARIES_MODEL_MODE="
if /i "%~1"=="--without-model" set "LIBRARIES_MODEL_MODE="
shift
goto scan_model_mode
:launch
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Missing .venv\Scripts\python.exe. Prepare the project environment first.
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0start_web.py" %LIBRARIES_MODEL_MODE% --port 8619 %*
set "LIBRARIES_EXIT_CODE=%errorlevel%"
endlocal & exit /b %LIBRARIES_EXIT_CODE%
