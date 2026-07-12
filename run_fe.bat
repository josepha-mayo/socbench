@echo off
cd /d "%~dp0frontend"
"C:\Program Files\nodejs\node.exe" "node_modules\next\dist\bin\next" start -p 3000 >> "%~dp0fe.out" 2>&1