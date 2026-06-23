"""
依赖安装脚本 - 在 app.py 之前运行
用于确保所有必要的 Python 包已安装
"""
import subprocess
import sys

def install_if_missing(package_name, import_name=None):
    """如果包未安装则安装"""
    if import_name is None:
        import_name = package_name.replace("-", "_")
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        pass
    
    print(f"[deps] Installing {package_name}...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            package_name, 
            "--quiet",
            "--disable-pip-version-check",
            "--no-cache-dir"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 再次尝试导入
        try:
            __import__(import_name)
            print(f"[deps] {package_name} installed successfully")
            return True
        except ImportError:
            print(f"[deps] WARNING: {package_name} still not importable after install")
            return False
    except Exception as e:
        print(f"[deps] ERROR installing {package_name}: {e}")
        return False

if __name__ == "__main__":
    packages = [
        ("opencv-python-headless", "cv2"),
        ("ultralytics", "ultralytics"),
        ("mediapipe", "mediapipe"),
        ("streamlit", "streamlit"),
        ("streamlit-webrtc", "webrtc"),
        ("aiortc", "aiortc"),
        ("Pillow", "PIL"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("python-multipart", "multipart"),
        ("pyyaml", "yaml"),
        ("tqdm", "tqdm"),
        ("loguru", "loguru"),
    ]
    
    failed = []
    for pkg, imp in packages:
        if not install_if_missing(pkg, imp):
            failed.append(pkg)
    
    if failed:
        print(f"[deps] FAILED to install: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("[deps] All packages installed successfully!")