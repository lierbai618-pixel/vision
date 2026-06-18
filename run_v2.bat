@echo off
cd /d "D:\ÏîÄ¿\yolov8-object-detection"
"D:\anaconda\envs\npy\python.exe" -m streamlit run app.py --server.port 8501 > run_v2.log 2>&1
