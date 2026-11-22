@echo off
rem Explicit offline diagnostic mode; normal start-web.cmd uses the main model.
call "%~dp0start-web.cmd" --without-model %*
