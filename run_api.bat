@echo off
cd /d "%~dp0backend"
C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe -u -m uvicorn socbench.api.app:app --host 127.0.0.1 --port 8000 >> "%~dp0api.out" 2>&1