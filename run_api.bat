@echo off
cd /d "%~dp0backend"
py -3 -u -m uvicorn socbench.api.app:app --host 127.0.0.1 --port 8000
