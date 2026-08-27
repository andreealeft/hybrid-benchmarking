@echo off
rem Double-click this once.  It installs the tool and puts an icon on the
rem Desktop that opens it.  Nothing has to be typed.
title Hybrid benchmarking

echo.
echo   Hybrid benchmarking
echo   ===================
echo.

rem ---------------------------------------------------------------- Python
set PY=
where py >nul 2>nul && set PY=py
if "%PY%"=="" (where python >nul 2>nul && set PY=python)
if "%PY%"=="" (
    echo   This needs Python, which is not on this computer yet.
    echo   Opening the download page now.
    echo.
    echo   Install it, tick "Add Python to PATH" in the installer,
    echo   then double-click this file again.
    start "" "https://www.python.org/downloads/windows/"
    echo.
    pause
    exit /b 0
)

rem ---------------------------------------------------------------- the tool
echo   Installing. This takes a minute or two the first time.
echo.
%PY% -m pip install --user --upgrade --quiet ^
    "https://github.com/andreealeft/hybrid-benchmarking/archive/refs/heads/main.zip"
if errorlevel 1 (
    echo.
    echo   The install did not finish. The message above says why.
    pause
    exit /b 1
)

rem ---------------------------------------------------------------- the icon
rem pythonw runs without a console window, so the icon behaves like an app
rem rather than opening a black box of text.
for /f "delims=" %%i in ('%PY% -c "import sys, os; print(os.path.join(os.path.dirname(sys.executable), 'pythonw.exe'))"') do set PYW=%%i

powershell -NoProfile -Command ^
  "$s = (New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop') + '\Hybrid benchmarking.lnk');" ^
  "$s.TargetPath = '%PYW%';" ^
  "$s.Arguments  = '-m hybrid_benchmarking.cli serve';" ^
  "$s.Description = 'Resource estimates for quantum algorithms';" ^
  "$s.Save()"

echo.
echo   Done.
echo.
echo   There is now an icon on your Desktop called Hybrid benchmarking.
echo   Double-click it whenever you want the tool: it opens in your browser.
echo   To stop it, close the browser tab and end it from the Task Manager.
echo.
pause
