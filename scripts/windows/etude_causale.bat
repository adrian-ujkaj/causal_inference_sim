@echo off
REM Etude causale complete avec le Python du projet (.conda, cree par setup_windows.bat).
REM   Double-clic                       : etude complete (24 vols x 3 campagnes, 1 a 3 h)
REM   etude_causale.bat --runs 6        : version rapide
REM   etude_causale.bat --skip_sim      : analyses seules sur les vols deja simules
REM NB : pas de %%PY%% dans un bloc entre parentheses : un ")" dans le chemin du
REM dossier fermerait le bloc et cmd s'arreterait sans message.
cd /d "%~dp0..\.."
set "PY=%CD%\.conda\python.exe"
if not exist "%PY%" goto :nopython
if not exist runs mkdir runs
set "LOG=runs\etude_causale.log"
echo Journal : %LOG%> "%LOG%"

echo === 1/3 Bibliotheques de l'analyse causale ===
"%PY%" -c "import torch, networkx, statsmodels, sklearn" 2>nul
if not errorlevel 1 goto :tests
echo Installation de torch, networkx, statsmodels, scikit-learn (quelques minutes)...
"%PY%" -m pip install torch networkx statsmodels scikit-learn >> "%LOG%" 2>&1
if errorlevel 1 goto :fail

:tests
echo === 2/3 Tests ===
"%PY%" -m pytest tests\test_causal.py -q >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
echo Tests OK

echo === 3/3 Etude causale ===
"%PY%" analysis\run_causal_study.py %* >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
type "%LOG%"
echo.
echo Termine. Les 3 figures a regarder sont dans runs\causal\RESULTATS
pause
exit /b 0

:nopython
echo Python du projet introuvable : lance d'abord setup_windows.bat
pause
exit /b 1

:fail
type "%LOG%"
echo.
echo ECHEC, voir le journal ci-dessus (%LOG%)
pause
exit /b 1
