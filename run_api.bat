@echo off
cd /d C:\Users\USER\.vscode\vibe\backend
C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe -u -m uvicorn socbench.api.app:app --host 127.0.0.1 --port 8000 >> C:\Users\USER\.vscode\vibe\api.out 2>&1
