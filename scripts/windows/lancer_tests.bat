@echo off
REM Lance les tests avec le Python du projet (.conda). Journal : runs\tests.log
cd /d "%~dp0..\.."
if not exist runs mkdir runs
"%CD%\.conda\python.exe" -m pytest -q > runs\tests.log 2>&1
type runs\tests.log
pause
