@echo off
setlocal
title Vietnamese RVC By Anh

call env\Scripts\activate.bat
python main\app\app.py --open --allow_all_disk
echo.
pause
