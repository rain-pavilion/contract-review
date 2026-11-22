@echo off
setlocal
set "ZLQL_STORE_PATH=%~dp0zhilv_qiling_v2\reports\workspace_core_final_20261121.sqlite"
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
call "%~dp0start-core-web.cmd" --port 8615 %*
endlocal
