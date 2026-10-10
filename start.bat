@echo off
setlocal
rem OpenFront AI launcher.
rem   start.bat             menu
rem   start.bat dashboard   control server + dashboard in the browser
rem   start.bat play        same, plus the local game server (localhost:9000)
rem   start.bat console     a console in train\ for the CLI tools

set "ROOT=%~dp0"
set "DASH_URL=http://127.0.0.1:8765/"

if /i "%~1"=="dashboard" goto dashboard
if /i "%~1"=="play" goto play
if /i "%~1"=="console" goto console

echo.
echo   OpenFront AI
echo.
echo   1  Dashboard
echo   2  Dashboard + local game server (for Play local / Watch replays)
echo   3  Console in train\
echo.
choice /c 123 /n /m "  Pick 1-3: "
if errorlevel 3 goto console
if errorlevel 2 goto play
goto dashboard

:play
call :port 9000
if errorlevel 1 (
  echo Starting the local game server in its own window...
  start "OpenFront game server" /d "%ROOT%OpenFrontIO" cmd /k npm run dev
) else (
  echo Local game server already running on port 9000.
)
goto dashboard

:dashboard
call :port 8765
if errorlevel 1 goto server_start
rem A server is up. One started before the code last changed serves the new
rem page but not its requests, so restart it unless it has jobs running.
set "STATE=%TEMP%\openfront_state.json"
curl -s -m 5 "%DASH_URL%api/state" > "%STATE%" 2>nul
findstr /c:"\"stale\": false" "%STATE%" >nul
if not errorlevel 1 (
  echo Control server already running on port 8765.
  goto open
)
findstr /c:"\"status\": \"running\"" "%STATE%" >nul
if not errorlevel 1 (
  echo The control server is running older code but has jobs running, so it was left alone.
  echo Restart it later: close its window, then run start.bat dashboard again.
  goto open
)
echo The control server is running older code; restarting it.
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:":8765 .*LISTENING"') do taskkill /pid %%p /f >nul 2>&1
ping -n 2 127.0.0.1 >nul

:server_start
echo Starting the control server in its own window; close that window to stop it.
start "OpenFront AI server" /d "%ROOT%train" cmd /k python control_server.py
ping -n 4 127.0.0.1 >nul

:open
start "" "%DASH_URL%"
goto :eof

:console
echo Console in train\. Try:
echo   python play_lobby.py --help
echo   python train.py --config configs/main.yaml
echo   python evaluate.py --help
start "OpenFront AI console" /d "%ROOT%train" cmd /k
goto :eof

rem errorlevel 0 when something is listening on port %1, 1 when free.
:port
netstat -ano | findstr /r /c:":%1 .*LISTENING" >nul
exit /b %errorlevel%
