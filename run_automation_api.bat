@echo off
setlocal
title RVC Headless API Server

echo Starting Headless API Server...
echo API is available at: http://localhost:8000/run
echo Note: No GUI will be shown.

env\Scripts\python.exe automation\headless_server.py

pause
