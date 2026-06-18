"""
智能视觉系统 - 统一版

集成目标检测、人脸识别、手势识别、口罩检测、实时监测、批量处理的完整智能分析平台
"""

import streamlit as st
# components import removed - using st.markdown instead
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import tempfile
import os
import time
import threading
from datetime import datetime, timedelta
# HTTPServer removed - using st.image directly


# ==================== 中文文字渲染 ====================

_FONT_CACHE_FILE = Path("cache/.font_path_cache.txt")

def _find_cjk_font() -> str:
    """查找中文字体（结果缓存到文件）"""
    if _FONT_CACHE_FILE.exists():
        cached = _FONT_CACHE_FILE.read_text().strip()
        if cached and Path(cached).exists():
            return cached
    candidates = [
        r'C:\Windows\Fonts\simhei.ttf', r'C:\Windows\Fonts\msyh.ttc',
        r'C:\Windows\Fonts\simsun.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/System/Library/Fonts/PingFang.ttc',
    ]
    for p in candidates:
        if Path(p).exists():
            _FONT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _FONT_CACHE_FILE.write_text(p)
            return p
    return ''

_CJK_FONT_PATH = _find_cjk_font()
_CJK_FONT_CACHE = {}

def _get_font(font_size):
    if font_size not in _CJK_FONT_CACHE:
        try:
            _CJK_FONT_CACHE[font_size] = ImageFont.truetype(_CJK_FONT_PATH, font_size) if _CJK_FONT_PATH else ImageFont.load_default()
        except Exception:
            _CJK_FONT_CACHE[font_size] = ImageFont.load_default()
    return _CJK_FONT_CACHE[font_size]

def batch_draw_texts(img, text_items):
    """批量绘制中文文字，只做一次 BGR->PIL->BGR 转换"""
    if not text_items:
        return img
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    for text, position, font_size, color_bgr in text_items:
        draw.text(position, text, font=_get_font(font_size), fill=color_bgr[::-1])
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def put_chinese_text(img, text, position, font_size=20, color=(0,255,0), thickness=2):
    return batch_draw_texts(img, [(text, position, font_size, color)])


# ==================== 实时监测（st.image 直接渲染，无需 MJPEG） ====================

class StreamServer:
    """摄像头抓帧 + 检测，通过 st.image 直接显示"""

    def __init__(self):
        self._cap = None
        self._running = False
        self._frame_rgb = None
        self._lock = threading.Lock()
        self._stats = {'fps': 0, 'objects': 0, 'faces': 0, 'masks': 0, 'frame_count': 0, 'start_time': 0}
        self._alerts = []
        self._last_ss_time = 0
        self._last_det_ss_time = 0
        self._det_model = None
        self._face_model = None
        self._mask_model = None
        self._cached_detections = {'text_items': []}
        self._cached_counts = {'obj': 0, 'face': 0, 'mask': 0}
        self._frame_idx = 0
        self._skip_frames = 2
        self._config = {
            'camera_id': 0, 'width': 640, 'height': 480,
            'enable_object': True, 'enable_face': True, 'enable_mask': False,
            'model_option': 'yolov8n.pt', 'conf_threshold': 0.35,
            'enable_alerts': True, 'alert_threshold': 3,
            'enable_auto_ss': False, 'ss_interval': 10, 'ss_on_detect': False,
        }

    def preload_models(self):
        """预加载模型，避免第一帧卡顿"""
        from ultralytics import YOLO
        cfg = self._config
        if cfg['enable_object'] and self._det_model is None:
            self._det_model = YOLO(cfg['model_option'])
        if cfg['enable_face'] and self._face_model is None:
            fp = Path(__file__).parent / "models" / "yolov8n-face.pt"
            self._face_model = YOLO(str(fp) if fp.exists() else "yolov8n.pt")
        if cfg['enable_mask'] and self._mask_model is None:
            self._mask_model = YOLO(safe_model_path("models/mask_detector.pt", "mask_detector.pt"))

    def start(self, **kwargs):
        if self._running:
            return
        self._config.update(kwargs)
        self._config['model_option'] = 'yolov8m.pt'
        cfg = self._config
        # 预加载模型
        self.preload_models()
        cap = None
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(cfg['camera_id'], backend)
            if cap.isOpened():
                break
            cap.release()
        if not cap or not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {cfg['camera_id']}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg['height'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self._running = True
        self._stats['start_time'] = time.time()
        self._stats['frame_count'] = 0

    def stop(self):
        self._running = False
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def read_and_detect(self):
        """读取一帧并执行检测，返回 RGB 图像用于 st.image"""
        if not self._running or not self._cap or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        t0 = time.time()
        cfg = self._config
        self._frame_idx += 1

        should_detect = (self._frame_idx % (self._skip_frames + 1)) == 0

        if should_detect:
            obj_count = 0
            face_count = 0
            mask_count = 0
            text_items = []

            if cfg['enable_object']:
                try:
                    if self._det_model is None:
                        from ultralytics import YOLO
                        self._det_model = YOLO(cfg['model_option'])
                    results = self._det_model(frame, conf=cfg['conf_threshold'], iou=0.5, agnostic_nms=True, imgsz=640, verbose=False)
                    if results[0].boxes is not None:
                        obj_count = len(results[0].boxes)
                        names = results[0].names
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            c = float(box.conf[0])
                            cls = int(box.cls[0])
                            name = COCO_CN.get(names[cls], names[cls])
                            # 人用绿色，其他目标用蓝色
                            if names[cls] == "person":
                                color = (0, 255, 0)
                            else:
                                color = (255, 0, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            text_items.append((f"{name} {c:.0%}", (x1, max(y1 - 18, 0)), 18, color))
                except Exception:
                    pass

            if cfg['enable_face']:
                try:
                    if self._face_model is None:
                        from ultralytics import YOLO
                        fp = Path(__file__).parent / "models" / "yolov8n-face.pt"
                        self._face_model = YOLO(str(fp) if fp.exists() else "yolov8n.pt")
                    f_res = self._face_model(frame, conf=0.5, imgsz=640, verbose=False)
                    if f_res[0].boxes is not None:
                        face_count = len(f_res[0].boxes)
                        for box in f_res[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            c = float(box.conf[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            text_items.append((f"人脸:{c:.0%}", (x1, max(y1 - 18, 0)), 18, (255, 0, 0)))
                except Exception:
                    pass

            if cfg['enable_mask']:
                try:
                    if self._mask_model is None:
                        from ultralytics import YOLO
                        self._mask_model = YOLO(safe_model_path("models/mask_detector.pt", "mask_detector.pt"))
                    m_res = self._mask_model(frame, conf=cfg['conf_threshold'], iou=0.5, agnostic_nms=True, imgsz=640, verbose=False)
                    if m_res[0].boxes is not None:
                        mask_count = len(m_res[0].boxes)
                        for box in m_res[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            c = float(box.conf[0])
                            cls = int(box.cls[0])
                            info = MASK_CLASSES.get(cls, {"name": "?", "color": (128, 128, 128)})
                            cv2.rectangle(frame, (x1, y1), (x2, y2), info['color'], 2)
                            text_items.append((f"{info['name']}:{c:.0%}", (x1, max(y1 - 18, 0)), 18, info['color']))
                except Exception:
                    pass

            self._cached_counts = {'obj': obj_count, 'face': face_count, 'mask': mask_count}
            self._cached_detections['text_items'] = text_items
        else:
            text_items = self._cached_detections.get('text_items', [])
            obj_count = self._cached_counts['obj']
            face_count = self._cached_counts['face']
            mask_count = self._cached_counts['mask']

        if text_items:
            frame = batch_draw_texts(frame, text_items)

        fps = 1.0 / max(time.time() - t0, 0.001)
        cv2.putText(frame, f"FPS:{fps:.0f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Obj:{obj_count} Face:{face_count} Mask:{mask_count}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        self._stats['fps'] = fps
        self._stats['objects'] = obj_count
        self._stats['faces'] = face_count
        self._stats['masks'] = mask_count
        self._stats['frame_count'] += 1

        if cfg['enable_alerts'] and obj_count >= cfg['alert_threshold']:
            a = {'time': datetime.now().strftime("%H:%M:%S"), 'msg': f"检测到 {obj_count} 个目标"}
            if a not in self._alerts:
                self._alerts.append(a)

        now = time.time()
        if cfg['enable_auto_ss'] and (now - self._last_ss_time) >= cfg['ss_interval']:
            self._last_ss_time = now
            Path("screenshots").mkdir(exist_ok=True)
            cv2.imwrite(f"screenshots/auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", frame)
        if cfg['ss_on_detect'] and obj_count > 0 and (now - self._last_det_ss_time) >= 3:
            self._last_det_ss_time = now
            Path("screenshots").mkdir(exist_ok=True)
            cv2.imwrite(f"screenshots/detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", frame)

        # BGR → RGB 用于 st.image 显示
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self._lock:
            self._frame_rgb = frame_rgb
        return frame_rgb

    def get_frame_rgb(self):
        with self._lock:
            return self._frame_rgb.copy() if self._frame_rgb is not None else None

    def get_stats(self):
        return dict(self._stats)

    def get_alerts(self):
        return list(self._alerts)

    def clear_alerts(self):
        self._alerts.clear()

    def screenshot(self):
        frame = self.get_frame_rgb()
        if frame is None:
            return None
        Path("screenshots").mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sp = f"screenshots/manual_{ts}.jpg"
        # 存储用 BGR 格式
        cv2.imwrite(sp, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return sp


# ==================== 页面配置 ====================

st.set_page_config(
    page_title="智能视觉系统",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    font-size: 2.5rem; font-weight: bold;
    background: linear-gradient(90deg, #1E88E5, #FF4B4B);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 0.5rem;
}
.sub-header { font-size: 1.1rem; color: #888; text-align: center; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ==================== 安全路径（MediaPipe不支持中文路径） ====================

def _ensure_safe_model_dir():
    """延迟创建安全模型目录（只在需要时创建）"""
    p = Path("C:/temp/models")
    p.mkdir(parents=True, exist_ok=True)
    return p

SAFE_MODEL_DIR = None  # 延迟初始化

# MediaPipe 段错误检测（缓存到文件，24小时内复用）
import subprocess
import sys as _sys

def _check_mediapipe_safe():
    """在子进程中测试 MediaPipe 是否安全可用（结果缓存24小时）"""
    _cache = Path("cache/.mediapipe_safe")
    try:
        if _cache.exists() and (time.time() - _cache.stat().st_mtime < 86400):
            return _cache.read_text().strip() == "OK"
        result = subprocess.run(
            [_sys.executable, "-c", "import mediapipe; print('OK')"],
            capture_output=True, text=True, timeout=10
        )
        ok = result.returncode == 0 and "OK" in result.stdout
        _cache.parent.mkdir(parents=True, exist_ok=True)
        _cache.write_text("OK" if ok else "FAIL")
        return ok
    except Exception:
        return _cache.exists() and _cache.read_text().strip() == "OK"

MEDIAPIPE_SAFE = _check_mediapipe_safe()


def safe_model_path(project_relative_path: str, safe_name: str) -> str:
    """将模型文件复制到无中文路径的目录，返回安全路径"""
    global SAFE_MODEL_DIR
    if SAFE_MODEL_DIR is None:
        SAFE_MODEL_DIR = _ensure_safe_model_dir()
    import shutil
    safe = SAFE_MODEL_DIR / safe_name
    if not safe.exists():
        src = Path(project_relative_path)
        if src.exists():
            shutil.copy2(str(src), str(safe))
    return str(safe)


# ==================== 口罩/手势类别配置 ====================

MASK_CLASSES = {
    0: {"name": "正确佩戴", "color": (0, 255, 0)},
    1: {"name": "佩戴不规范", "color": (0, 165, 255)},
    2: {"name": "未佩戴", "color": (0, 0, 255)},
}

# COCO 80类中文映射
COCO_CN = {
    "person": "人", "bicycle": "自行车", "car": "汽车", "motorcycle": "摩托车",
    "airplane": "飞机", "bus": "公交车", "train": "火车", "truck": "卡车",
    "boat": "船", "traffic light": "红绿灯", "fire hydrant": "消防栓",
    "stop sign": "停车标志", "parking meter": "停车计时器", "bench": "长椅",
    "bird": "鸟", "cat": "猫", "dog": "狗", "horse": "马", "sheep": "羊",
    "cow": "牛", "elephant": "大象", "bear": "熊", "zebra": "斑马",
    "giraffe": "长颈鹿", "backpack": "背包", "umbrella": "雨伞",
    "handbag": "手提包", "tie": "领带", "suitcase": "行李箱",
    "frisbee": "飞盘", "skis": "滑雪板", "snowboard": "单板滑雪",
    "sports ball": "运动球", "kite": "风筝", "baseball bat": "棒球棒",
    "baseball glove": "棒球手套", "skateboard": "滑板", "surfboard": "冲浪板",
    "tennis racket": "网球拍", "bottle": "瓶子", "wine glass": "酒杯",
    "cup": "杯子", "fork": "叉子", "knife": "刀", "spoon": "勺子",
    "bowl": "碗", "banana": "香蕉", "apple": "苹果", "sandwich": "三明治",
    "orange": "橙子", "broccoli": "西兰花", "carrot": "胡萝卜",
    "hot dog": "热狗", "pizza": "披萨", "donut": "甜甜圈", "cake": "蛋糕",
    "chair": "椅子", "couch": "沙发", "potted plant": "盆栽",
    "bed": "床", "dining table": "餐桌", "toilet": "马桶", "tv": "电视",
    "laptop": "笔记本电脑", "mouse": "鼠标", "remote": "遥控器",
    "keyboard": "键盘", "cell phone": "手机", "microwave": "微波炉",
    "oven": "烤箱", "toaster": "烤面包机", "sink": "水槽",
    "refrigerator": "冰箱", "book": "书", "clock": "时钟", "vase": "花瓶",
    "scissors": "剪刀", "teddy bear": "泰迪熊", "hair drier": "吹风机",
    "toothbrush": "牙刷",
}


def cn_name(en_name: str) -> str:
    """英文类别名转中文"""
    return COCO_CN.get(en_name, en_name)


# ==================== 全局状态初始化 ====================

def init_session_state():
    defaults = {
        'monitoring': False,
        'recording': False,
        'detection_history': [],
        'screenshots': [],
        'alerts': [],
        'current_frame': None,
        'frame_count': 0,
        'start_time': 0,
        'last_screenshot_time': 0,
        'last_detect_screenshot_time': 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if '_stream_server' not in st.session_state:
        st.session_state['_stream_server'] = StreamServer()


# ==================== 检测器加载（全部使用安全路径） ====================

@st.cache_resource
def load_object_detector(model_path='yolov8n.pt', conf=0.5):
    from src.detector import ObjectDetector
    return ObjectDetector(model_path=model_path, conf_threshold=conf)


@st.cache_resource
def load_yolo_face_model():
    """加载YOLOv8专用人脸检测模型"""
    from ultralytics import YOLO
    fp = Path(__file__).parent / "models" / "yolov8n-face.pt"
    if not fp.exists(): fp = Path("C:/temp/models/yolov8n-face.pt")
    if not fp.exists(): fp = Path("yolov8n.pt")
    return YOLO(str(fp))


@st.cache_resource
def load_mask_detector():
    from ultralytics import YOLO
    p = safe_model_path("models/mask_detector.pt", "mask_detector.pt")
    return YOLO(p)


@st.cache_resource
def load_gesture_detector():
    """加载手势识别器（MediaPipe HandLandmarker）"""
    if not MEDIAPIPE_SAFE:
        return None
    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        model_path = safe_model_path("models/hand_landmarker.task", "hand_landmarker.task")
        base = python.BaseOptions(model_asset_path=model_path)
        opts = vision.HandLandmarkerOptions(
            base_options=base,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        return vision.HandLandmarker.create_from_options(opts)
    except Exception as e:
        st.warning(f"MediaPipe 加载失败: {e}")
        return None


def classify_gesture(landmarks):
    """根据手部关键点简单判断手势"""
    # 指尖和指间关节的y坐标比较
    tips = [4, 8, 12, 16, 20]  # 拇指、食指、中指、无名指、小指的指尖
    pips = [3, 6, 10, 14, 18]  # 对应的指间关节
    fingers_up = []
    for tip, pip in zip(tips, pips):
        if tip == 4:  # 拇指特殊处理（水平方向）
            fingers_up.append(landmarks[tip].x < landmarks[pip].x)
        else:
            fingers_up.append(landmarks[tip].y < landmarks[pip].y)
    count = sum(fingers_up)
    if count == 0: return "拳头"
    if count == 5: return "张开手掌"
    if count == 1 and fingers_up[1]: return "食指指向前方"
    if count == 2 and fingers_up[1] and fingers_up[2]: return "耶/剪刀手"
    if count == 1 and fingers_up[0]: return "竖起拇指"
    return f"{count}根手指"


# ==================== 主界面 ====================

def main():
    init_session_state()

    st.markdown('<div class="main-header">👁️ 智能视觉系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">YOLOv8 · MediaPipe · 目标检测 · 人脸识别 · 手势识别 · 口罩检测 · 实时监测</div>', unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.markdown("## 🎯 功能导航")
        app_mode = st.selectbox("选择功能", [
            "📹 实时监测", "📷 目标检测", "👤 人脸识别", "✋ 手势识别",
            "😷 口罩检测", "🧠 模型训练", "ℹ️ 关于"
        ], index=0, key="app_mode")

        st.markdown("---")
        st.markdown("## ⚙️ 全局设置")
        model_option = st.selectbox("检测模型", [
            "yolov8m.pt", "yolov8l.pt", "yolov8x.pt", "yolov8s.pt", "yolov8n.pt"
        ], index=0, key="model_option", help="m/l 精度高，n/s 速度快")
        conf_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.35, 0.05, key="conf_threshold")

    route = {
        "📹 实时监测": lambda: show_realtime_monitor(model_option, conf_threshold),
        "📷 目标检测": lambda: show_object_detection(model_option, conf_threshold),
        "👤 人脸识别": show_face_detection,
        "✋ 手势识别": show_gesture_detection,
        "😷 口罩检测": lambda: show_mask_detection(conf_threshold),
        "🧠 模型训练": show_training,
        "ℹ️ 关于": show_about,
    }
    route[app_mode]()


# ==================== 手机摄像头实时检测 ====================

def show_mobile_camera(model_option, conf_threshold):
    """使用浏览器摄像头检测（支持手机）"""
    st.markdown("### 📱 手机摄像头检测")

    # 检测选项
    c1, c2, c3 = st.columns(3)
    with c1: enable_object = st.checkbox("🎯 目标检测", True, key="mob_obj")
    with c2: enable_face = st.checkbox("👤 人脸识别", True, key="mob_face")
    with c3: enable_mask = st.checkbox("😷 口罩检测", False, key="mob_mask")

    # 模式选择
    cam_mode = st.radio("选择模式", ["📹 实时视频流", "📸 拍照检测"], horizontal=True, key="mob_cam_mode")

    if cam_mode == "📸 拍照检测":
        show_camera_photo_mode(model_option, conf_threshold, enable_object, enable_face, enable_mask)
        return

    # 实时视频流模式
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
        import av
        import threading

        # 加载模型（使用缓存）
        @st.cache_resource
        def load_mob_models(model_opt):
            det_m = None
            face_m = None
            try:
                from ultralytics import YOLO
                det_m = YOLO(model_opt)
            except Exception:
                pass
            try:
                face_m = load_yolo_face_model()
            except Exception:
                pass
            return det_m, face_m

        det_model, face_model = load_mob_models(model_option)

        # 模型锁
        model_lock = threading.Lock()

        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            text_items = []

            with model_lock:
                # 目标检测
                if enable_object and det_model is not None:
                    try:
                        results = det_model(img, conf=conf_threshold, iou=0.5, agnostic_nms=True, imgsz=640, verbose=False)
                        if results[0].boxes is not None:
                            names = results[0].names
                            for box in results[0].boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                c = float(box.conf[0])
                                cls = int(box.cls[0])
                                name = COCO_CN.get(names[cls], names[cls])
                                color = (0, 255, 0) if names[cls] == "person" else (255, 0, 0)
                                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                                text_items.append((f"{name} {c:.0%}", (x1, max(y1 - 18, 0)), 16, color))
                    except Exception:
                        pass

                # 人脸识别
                if enable_face and face_model is not None:
                    try:
                        f_res = face_model(img, conf=0.25, imgsz=640, verbose=False)
                        if f_res[0].boxes is not None:
                            for box in f_res[0].boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                c = float(box.conf[0])
                                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                text_items.append((f"人脸 {c:.0%}", (x1, max(y1 - 18, 0)), 16, (255, 0, 0)))
                    except Exception:
                        pass

            # 绘制中文文字
            if text_items:
                img = batch_draw_texts(img, text_items)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        RTC_CONFIGURATION = RTCConfiguration(
            iceServers=[
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
            ]
        )

        webrtc_ctx = webrtc_streamer(
            key="mobile-realtime",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=False,
        )

        if webrtc_ctx.state.playing:
            st.success("📹 实时检测已启动！")
        else:
            st.info("👆 点击 Play 按钮启动摄像头")

    except ImportError:
        st.warning("实时视频流组件未安装，使用拍照模式")
        show_camera_photo_mode(model_option, conf_threshold, enable_object, enable_face, enable_mask)
    except Exception as e:
        st.error(f"实时视频流启动失败: {e}")
        st.info("请使用拍照模式")
        show_camera_photo_mode(model_option, conf_threshold, enable_object, enable_face, enable_mask)


def show_camera_photo_mode(model_option, conf_threshold, enable_object, enable_face, enable_mask):
    """拍照模式（备用）"""
    camera_photo = st.camera_input("📸 点击拍照", key="camera_input")

    if camera_photo is not None:
        image = Image.open(camera_photo)
        st.markdown("### 🔍 检测结果")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**原图**")
            st.image(image, width='stretch')

        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp, format='JPEG')
            tmp_path = tmp.name

        img = cv2.imread(tmp_path)
        os.unlink(tmp_path)

        text_items = []
        obj_count = 0
        face_count = 0
        mask_count = 0

        if enable_object:
            try:
                from ultralytics import YOLO
                det_model = YOLO(model_option)
                results = det_model(img, conf=conf_threshold, iou=0.5, agnostic_nms=True, imgsz=640, verbose=False)
                if results[0].boxes is not None:
                    obj_count = len(results[0].boxes)
                    names = results[0].names
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        c = float(box.conf[0])
                        cls = int(box.cls[0])
                        name = COCO_CN.get(names[cls], names[cls])
                        color = (0, 255, 0) if names[cls] == "person" else (255, 0, 0)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        text_items.append((f"{name} {c:.0%}", (x1, max(y1 - 18, 0)), 20, color))
            except Exception as e:
                st.warning(f"目标检测失败: {e}")

        if enable_face:
            try:
                face_model = load_yolo_face_model()
                f_res = face_model(img, conf=0.25, verbose=False)
                if f_res[0].boxes is not None:
                    face_count = len(f_res[0].boxes)
                    for box in f_res[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        c = float(box.conf[0])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        text_items.append((f"人脸 {c:.0%}", (x1, max(y1 - 18, 0)), 20, (255, 0, 0)))
            except Exception as e:
                st.warning(f"人脸识别失败: {e}")

        if enable_mask:
            try:
                mask_model = load_mask_detector()
                m_res = mask_model(img, conf=conf_threshold, verbose=False)
                if m_res[0].boxes is not None:
                    mask_count = len(m_res[0].boxes)
                    for box in m_res[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        c = float(box.conf[0])
                        cls = int(box.cls[0])
                        info = MASK_CLASSES.get(cls, {"name": "?", "color": (128, 128, 128)})
                        cv2.rectangle(img, (x1, y1), (x2, y2), info['color'], 2)
                        text_items.append((f"{info['name']}:{c:.0%}", (x1, max(y1 - 18, 0)), 20, info['color']))
            except Exception as e:
                st.warning(f"口罩检测失败: {e}")

        if text_items:
            img = batch_draw_texts(img, text_items)

        result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with c2:
            st.markdown("**检测结果**")
            st.image(result_img, width='stretch')

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("🎯 目标", obj_count)
        with c2: st.metric("👤 人脸", face_count)
        with c3: st.metric("😷 口罩", mask_count)


# ==================== 实时监测（st.image 直接渲染） ====================

def show_realtime_monitor(model_option, conf_threshold):
    st.markdown("## 📹 实时监测")
    st.markdown("摄像头实时检测 · 截图 · 告警")

    srv = st.session_state['_stream_server']

    # 检测是否在云端环境
    import socket
    try:
        # 尝试连接本地摄像头来判断是否在本地
        test_cap = cv2.VideoCapture(0)
        is_local = test_cap.isOpened()
        test_cap.release()
    except:
        is_local = False

    # 摄像头模式选择
    if is_local:
        camera_mode = st.radio("摄像头模式", ["🖥️ 本地摄像头", "📱 手机摄像头"], horizontal=True, key="cam_mode")
    else:
        camera_mode = "📱 手机摄像头"
        st.info("☁️ 云端环境，使用手机浏览器摄像头拍照检测")

    if camera_mode == "📱 手机摄像头":
        # 手机摄像头模式
        show_mobile_camera(model_option, conf_threshold)
        return

    with st.expander("⚙️ 监测配置", expanded=not srv._running):
        c1, c2, c3, c4 = st.columns(4)
        with c1: camera_id = st.selectbox("摄像头", [0, 1, 2], key="cam_id")
        with c2: enable_object = st.checkbox("目标检测", True, key="en_obj")
        with c3: enable_face = st.checkbox("人脸识别", True, key="en_face")
        with c4: enable_mask = st.checkbox("口罩检测", False, key="en_mask")

        c1, c2, c3, c4 = st.columns(4)
        with c1: resolution = st.selectbox("分辨率", ["640x480", "1280x720", "320x240"], key="res")
        with c2: enable_alerts = st.checkbox("启用告警", True, key="en_alert")
        with c3: alert_threshold = st.slider("告警阈值", 1, 10, 3, key="alert_th")
        with c4: enable_auto_ss = st.checkbox("自动截图", False, key="en_auto_ss")

        c1, c2, c3 = st.columns(3)
        with c1: ss_interval = st.slider("自动截图间隔(秒)", 5, 60, 10, key="ss_int")
        with c2: ss_on_detect = st.checkbox("检测到目标时截图", False, key="ss_on_det")
        with c3: skip_frames = st.slider("跳帧数(越大越快)", 0, 5, 2, key="skip_fr", help="每N帧检测1帧，0=每帧检测")

        width, height = map(int, resolution.split('x'))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("🚀 启动监测", width='stretch', type="primary", disabled=srv._running):
            try:
                srv._skip_frames = skip_frames
                srv.start(camera_id=camera_id, width=width, height=height,
                    enable_object=enable_object, enable_face=enable_face, enable_mask=enable_mask,
                    model_option=model_option, conf_threshold=conf_threshold,
                    enable_alerts=enable_alerts, alert_threshold=alert_threshold,
                    enable_auto_ss=enable_auto_ss, ss_interval=ss_interval, ss_on_detect=ss_on_detect)
                st.toast("✅ 监测已启动"); st.rerun()
            except Exception as e:
                st.error(f"❌ 启动失败: {e}")
    with c2:
        if st.button("⏹️ 停止监测", width='stretch', disabled=not srv._running):
            srv.stop(); st.toast("⏹️ 已停止"); st.rerun()
    with c3:
        if st.button("📸 手动截图", width='stretch'):
            sp = srv.screenshot()
            st.toast(f"📸 已保存: {sp}" if sp else "⚠️ 没有可用画面")
    with c4:
        if st.button("🗑️ 清除历史", width='stretch'):
            srv.clear_alerts()
            st.session_state['detection_history'] = []
            st.session_state['screenshots'] = []

    if srv._running:
        # st.image 直接渲染循环
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        alert_placeholder = st.empty()

        while srv._running:
            frame = srv.read_and_detect()
            if frame is not None:
                frame_placeholder.image(frame, channels="RGB", use_container_width=True)
            stats = srv.get_stats()
            total_t = time.time() - stats['start_time'] if stats['start_time'] else 1
            avg_fps = stats['frame_count'] / total_t if total_t > 0 else 0
            stats_placeholder.caption(f"📊 FPS: {stats['fps']:.0f} | 平均: {avg_fps:.0f} | 帧数: {stats['frame_count']} | 目标: {stats['objects']} | 人脸: {stats['faces']} | 口罩: {stats.get('masks', 0)}")
            alerts = srv.get_alerts()[-5:]
            if alerts:
                alert_placeholder.markdown("".join(f"🚨 **{a['time']}** - {a['msg']}  " for a in alerts))
            time.sleep(0.03)
    else:
        st.info("👆 点击 **启动监测** 开始实时检测")


def _show_monitor_statistics():
    st.markdown("## 📊 统计分析")
    history = st.session_state['detection_history']
    if not history:
        st.info("暂无统计数据"); return
    st.line_chart([h['fps'] for h in history[-50:]])
    c1, c2 = st.columns(2)
    with c1: st.metric("平均目标数", f"{sum(h['objects'] for h in history)/len(history):.1f}")
    with c2: st.metric("平均人脸数", f"{sum(h['faces'] for h in history)/len(history):.1f}")
    st.dataframe(history[-20:])


def _show_monitor_screenshots():
    st.markdown("## 🖼️ 截图管理")
    screenshots = st.session_state['screenshots']
    if not screenshots:
        st.info("暂无截图，点击「📸 手动截图」或启用自动截图"); return
    c1, c2 = st.columns(2)
    with c1: st.metric("截图总数", len(screenshots))
    with c2:
        if st.button("🗑️ 删除所有截图"):
            for s in screenshots:
                if Path(s['path']).exists(): os.remove(s['path'])
            st.session_state['screenshots'] = []; st.rerun()
    for i, ss in enumerate(screenshots):
        with st.expander(f"📸 {i+1} - {ss['time']}", expanded=False):
            c1, c2, c3 = st.columns([3, 2, 1])
            with c1:
                if Path(ss['path']).exists(): st.image(ss['path'], width='stretch')
            with c2:
                d = ss['detections']
                note = d.get('note', '')
                if note:
                    st.write(f"📝 {note}")
                else:
                    st.write(f"🎯 目标: {d.get('objects',0)} | 👤 人脸: {d.get('faces',0)}")
                st.code(ss['path'])
                if Path(ss['path']).exists():
                    with open(ss['path'], 'rb') as f:
                        st.download_button("📥 下载", f.read(), f"ss_{ss['time']}.jpg", "image/jpeg", key=f"dl_{i}")
            with c3:
                if st.button(f"🗑️", key=f"del_{i}"):
                    if Path(ss['path']).exists(): os.remove(ss['path'])
                    st.session_state['screenshots'].pop(i); st.rerun()


def _show_monitor_alerts():
    st.markdown("## 🚨 告警记录")
    alerts = st.session_state['alerts']
    if not alerts: st.info("暂无告警"); return
    st.metric("告警总数", len(alerts))
    for a in reversed(alerts[-20:]):
        color = "#f5576c" if a['level'] == 'warning' else "#ffa726"
        st.markdown(f'<div style="background:{color};padding:10px;border-radius:5px;color:white"><strong>{a["time"]}</strong> - {a["message"]}</div>', unsafe_allow_html=True)


# ==================== 临时文件工具 ====================

def _save_temp_image(image, suffix='.jpg'):
    """将 PIL Image 保存到临时文件，确保完全写入后返回路径"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        image.save(tmp, format='JPEG')
        tmp.flush()
        os.fsync(tmp.fileno())
        return tmp.name


def _safe_unlink(path):
    """安全删除临时文件"""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


# ==================== 目标检测 ====================

def show_object_detection(model_option, conf_threshold):
    st.markdown("## 📷 目标检测")
    st.markdown("上传一张或多张图片进行检测，支持批量处理")
    det = load_object_detector(model_option, conf_threshold)
    files = st.file_uploader("📤 上传图片（支持多张批量处理）", type=['jpg','jpeg','png','bmp'], accept_multiple_files=True, key='det_up')
    if files:
        for f in files:
            st.markdown(f"---\n**📷 {f.name}**")
            image = Image.open(f).convert("RGB")
            c1, c2 = st.columns(2)
            with c1: st.markdown("原图"); st.image(image, width='stretch')
            with st.spinner(f"检测 {f.name}..."):
                tp = _save_temp_image(image, '.jpg')
                try:
                    result_img = cv2.imread(tp)
                    det_model = det.model
                    raw = det_model(result_img, conf=conf_threshold, iou=0.4, agnostic_nms=True, imgsz=960, verbose=False)
                    cn_names = []
                    confs = []
                    if raw[0].boxes is not None:
                        text_items = []
                        for box in raw[0].boxes:
                            x1,y1,x2,y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0]); cls = int(box.cls[0])
                            name = cn_name(det_model.names[cls])
                            cn_names.append(name)
                            confs.append(conf)
                            # 人用绿色，其他目标用蓝色
                            if det_model.names[cls] == "person":
                                color = (0, 255, 0)
                            else:
                                color = (255, 0, 0)
                            cv2.rectangle(result_img, (x1,y1), (x2,y2), color, 2)
                            text_items.append((f"{name} {conf:.0%}", (x1, max(y1-18,0)), 20, color))
                        result_img = batch_draw_texts(result_img, text_items)
                    ri = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                finally:
                    _safe_unlink(tp)
            with c2: st.markdown("结果"); st.image(ri, width='stretch')
            c1,c2,c3 = st.columns(3)
            with c1: st.metric("数量", len(cn_names))
            with c2:
                if cn_names: st.metric("主要类别", max(set(cn_names), key=cn_names.count))
            with c3:
                if confs: st.metric("置信度", f"{np.mean(confs):.0%}")
            if cn_names:
                for i, (n, c) in enumerate(zip(cn_names, confs)):
                    st.write(f"- **{n}**: {c:.0%}")


# ==================== 人脸识别 ====================

def _detect_faces(img, conf=0.25):
    """YOLOv8专用人脸检测"""
    model = load_yolo_face_model()
    faces = []
    results = model(img, conf=conf, verbose=False)
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            c = float(box.conf[0])
            faces.append((x1, y1, x2-x1, y2-y1, c))
    return faces


def show_face_detection():
    st.markdown("## 👤 人脸识别（YOLOv8专用模型）")
    st.markdown("使用WIDER FACE数据集训练的YOLOv8人脸检测模型，精度远超MediaPipe")
    files = st.file_uploader("📤 上传图片（支持多张批量处理）", type=['jpg','jpeg','png','bmp'], accept_multiple_files=True, key='face_up')
    if files:
        for f in files:
            st.markdown(f"---\n**📷 {f.name}**")
            image = Image.open(f).convert("RGB")
            c1, c2 = st.columns(2)
            with c1: st.markdown("原图"); st.image(image, width='stretch')
            with st.spinner(f"检测 {f.name}..."):
                tp = _save_temp_image(image, '.jpg')
                try:
                    img = cv2.imread(tp)
                    faces = _detect_faces(img, conf=0.25)
                    face_texts = []
                    for (x, y, w, h, conf) in faces:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        face_texts.append((f"人脸 {conf:.0%}", (x, max(y-18,0)), 20, (255,0,0)))
                    img = batch_draw_texts(img, face_texts)
                    ri = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                finally:
                    _safe_unlink(tp)
            with c2:
                st.markdown("结果")
                if faces: st.image(ri, width='stretch')
                else: st.warning("未检测到人脸")
            st.metric("人脸数量", len(faces))


# ==================== 手势识别 ====================

def show_gesture_detection():
    st.markdown("## ✋ 手势识别")
    if not MEDIAPIPE_SAFE:
        st.error("⚠️ MediaPipe 在当前环境中不可用（段错误），手势识别功能已禁用。")
        st.info("建议：尝试降级 MediaPipe 版本 `pip install mediapipe==0.10.14` 或使用其他 Python 环境。")
        return
    st.markdown("上传一张或多张图片，支持批量处理")
    files = st.file_uploader("📤 上传图片（支持多张批量处理）", type=['jpg','jpeg','png','bmp'], accept_multiple_files=True, key='gesture_up')
    if files:
        for f in files:
            st.markdown(f"---\n**📷 {f.name}**")
            image = Image.open(f).convert("RGB")
            c1, c2 = st.columns(2)
            with c1: st.markdown("原图"); st.image(image, width='stretch')
            with st.spinner(f"检测 {f.name}..."):
                tp = _save_temp_image(image, '.jpg')
                try:
                    gd = load_gesture_detector()
                    img = cv2.imread(tp)
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    import mediapipe as mp
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = gd.detect(mp_img)
                    gestures = []
                    h, w = img.shape[:2]
                    if result.hand_landmarks:
                        for hand_lms in result.hand_landmarks:
                            gesture = classify_gesture(hand_lms)
                            gestures.append(gesture)
                            for lm in hand_lms:
                                cv2.circle(img, (int(lm.x*w), int(lm.y*h)), 4, (0,0,255), -1)
                            for (a,b) in mp.solutions.hands.HAND_CONNECTIONS:
                                ax,ay = int(hand_lms[a].x*w), int(hand_lms[a].y*h)
                                bx,by = int(hand_lms[b].x*w), int(hand_lms[b].y*h)
                                cv2.line(img, (ax,ay), (bx,by), (0,255,255), 2)
                            wrist = hand_lms[0]
                            cv2.putText(img, gesture, (int(wrist.x*w)-20, int(wrist.y*h)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                    ri = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                finally:
                    _safe_unlink(tp)
            with c2: st.markdown("结果"); st.image(ri, width='stretch')
            st.metric("手部数量", len(gestures))
            for i, g in enumerate(gestures):
                st.write(f"- **手 {i+1}**: {g}")


# ==================== 口罩检测 ====================

def show_mask_detection(conf_threshold):
    st.markdown("## 😷 口罩检测")
    st.markdown("上传一张或多张图片，支持批量处理")
    det = load_mask_detector()
    files = st.file_uploader("📤 上传图片（支持多张批量处理）", type=['jpg','jpeg','png','bmp'], accept_multiple_files=True, key='mask_up')
    if files:
        for f in files:
            st.markdown(f"---\n**📷 {f.name}**")
            image = Image.open(f).convert("RGB")
            c1, c2 = st.columns(2)
            with c1: st.markdown("原图"); st.image(image, width='stretch')
            with st.spinner(f"检测 {f.name}..."):
                tp = _save_temp_image(image, '.jpg')
                try:
                    results = det(tp, conf=conf_threshold, verbose=False)
                    ri = cv2.imread(tp)
                    counts = {0:0, 1:0, 2:0}
                    if results[0].boxes is not None:
                        mask_texts = []
                        for box in results[0].boxes:
                            x1,y1,x2,y2 = map(int, box.xyxy[0])
                            c = float(box.conf[0]); cls = int(box.cls[0])
                            info = MASK_CLASSES.get(cls, {"name":"?","color":(128,128,128)})
                            cv2.rectangle(ri, (x1,y1), (x2,y2), info['color'], 2)
                            mask_texts.append((f"{info['name']}:{c:.0%}", (x1, max(y1-18,0)), 20, info['color']))
                            counts[cls] = counts.get(cls, 0) + 1
                        ri = batch_draw_texts(ri, mask_texts)
                    ri = cv2.cvtColor(ri, cv2.COLOR_BGR2RGB)
                finally:
                    _safe_unlink(tp)
            with c2: st.markdown("结果"); st.image(ri, width='stretch')
            total = sum(counts.values())
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("总人数", total)
            with c2: st.metric("正确佩戴", counts[0])
            with c3: st.metric("不规范", counts[1])
            with c4: st.metric("未佩戴", counts[2])
            if total > 0: st.progress(counts[0]/total, text=f"合规率: {counts[0]/total:.0%}")


# ==================== 模型训练 ====================

def show_training():
    st.markdown("## 🧠 模型训练")
    st.markdown("下载数据集 · 训练自定义模型 · 提升检测精度")

    # 数据集选择
    st.markdown("### 📁 数据集")
    c1, c2 = st.columns(2)
    with c1:
        dataset_source = st.selectbox("数据集来源", [
            "COCO128 (本地)", "COCO128 (下载)", "VOC2012 (下载)", "自定义URL"
        ], key="ds_source")
    with c2:
        train_model = st.selectbox("预训练模型", [
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"
        ], index=2, key="train_model")

    # 自定义URL输入
    custom_url = ""
    if dataset_source == "自定义URL":
        custom_url = st.text_input("数据集URL", placeholder="https://example.com/dataset.zip", key="ds_url",
                                   help="支持 .zip 格式，目录结构须为 YOLO 格式 (images/train, labels/train)")

    # 训练参数
    st.markdown("### ⚙️ 训练参数")
    c1, c2, c3, c4 = st.columns(4)
    with c1: epochs = st.slider("训练轮数", 10, 300, 50, key="epochs")
    with c2: imgsz = st.selectbox("图片尺寸", [320, 416, 640], index=2, key="imgsz")
    with c3: batch = st.selectbox("批次大小", [4, 8, 16, 32], index=1, key="batch")
    with c4: lr = st.number_input("学习率", 0.0001, 0.1, 0.01, format="%.4f", key="lr")

    c1, c2, c3 = st.columns(3)
    with c1: device = st.selectbox("设备", ["自动", "CPU", "GPU"], key="device")
    with c2: patience = st.slider("早停耐心值", 10, 100, 30, key="patience")
    with c3: workers = st.slider("线程数", 0, 8, 4, key="workers")

    # 自定义类别
    st.markdown("### 🏷️ 类别 (可选)")
    custom_classes = st.text_input("自定义类别名", placeholder="person,car,dog", key="custom_classes",
                                   help="留空则使用数据集默认类别，多个类别用逗号分隔")

    # 训练按钮
    st.markdown("---")
    if st.button("🚀 开始训练", type="primary", width='stretch'):
        run_training(dataset_source, custom_url, train_model, epochs, imgsz, batch, lr, device, patience, workers, custom_classes)


def run_training(dataset_source, custom_url, train_model, epochs, imgsz, batch, lr, device, patience, workers, custom_classes):
    """执行训练流程"""
    import subprocess, shutil

    data_dir = Path("data/datasets")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. 准备数据集
    with st.status("📥 准备数据集...", expanded=True) as status:
        if dataset_source == "COCO128 (本地)":
            ds_path = data_dir / "coco128"
            if not ds_path.exists():
                st.error("❌ 本地 COCO128 不存在，请选择下载"); return
            st.write("✅ 使用本地 COCO128 数据集")
            data_yaml = Path("configs/coco128.yaml")

        elif dataset_source == "COCO128 (下载)":
            ds_path = data_dir / "coco128"
            if not ds_path.exists():
                st.write("⏳ 正在下载 COCO128...")
                import urllib.request, zipfile, io
                url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
                try:
                    resp = urllib.request.urlopen(url)
                    z = zipfile.ZipFile(io.BytesIO(resp.read()))
                    z.extractall(data_dir)
                    st.write("✅ COCO128 下载完成")
                except Exception as e:
                    st.error(f"❌ 下载失败: {e}"); return
            else:
                st.write("✅ COCO128 已存在，跳过下载")
            data_yaml = Path("configs/coco128.yaml")

        elif dataset_source == "VOC2012 (下载)":
            ds_path = data_dir / "VOC2012"
            if not ds_path.exists():
                st.write("⏳ 正在下载 VOC2012...")
                try:
                    from ultralytics.utils.downloads import download
                    download(f"https://github.com/ultralytics/assets/releases/download/v0.0.0/VOC2012.zip", dir=data_dir)
                    st.write("✅ VOC2012 下载完成")
                except Exception as e:
                    st.error(f"❌ 下载失败: {e}"); return
            else:
                st.write("✅ VOC2012 已存在，跳过下载")
            # 创建 VOC yaml
            voc_yaml = Path("configs/voc2012.yaml")
            voc_yaml.parent.mkdir(parents=True, exist_ok=True)
            voc_classes = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
                           "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
            with open(voc_yaml, 'w') as f:
                yaml_content = f"path: {ds_path.absolute()}\ntrain: images/train\nval: images/val\nnames:\n"
                for i, name in enumerate(voc_classes):
                    yaml_content += f"  {i}: {name}\n"
                f.write(yaml_content)
            data_yaml = voc_yaml

        elif dataset_source == "自定义URL":
            if not custom_url:
                st.error("❌ 请输入数据集URL"); return
            ds_name = "custom_" + custom_url.split("/")[-1].replace(".zip","")
            ds_path = data_dir / ds_name
            if not ds_path.exists():
                st.write(f"⏳ 正在下载 {custom_url}...")
                try:
                    import urllib.request, zipfile, io
                    resp = urllib.request.urlopen(custom_url)
                    z = zipfile.ZipFile(io.BytesIO(resp.read()))
                    z.extractall(ds_path)
                    st.write("✅ 下载完成")
                except Exception as e:
                    st.error(f"❌ 下载失败: {e}"); return
            else:
                st.write("✅ 数据集已存在，跳过下载")
            # 创建自定义 yaml
            custom_yaml = Path(f"configs/{ds_name}.yaml")
            custom_yaml.parent.mkdir(parents=True, exist_ok=True)
            classes = [c.strip() for c in custom_classes.split(",")] if custom_classes else []
            if not classes:
                # 自动扫描 labels 目录获取类别数
                label_dir = ds_path / "labels" / "train"
                if label_dir.exists():
                    max_cls = 0
                    for lf in label_dir.glob("*.txt"):
                        for line in lf.read_text().strip().split("\n"):
                            parts = line.split()
                            if parts:
                                max_cls = max(max_cls, int(parts[0]) + 1)
                    classes = [f"class_{i}" for i in range(max_cls)]
                else:
                    classes = ["object"]
            with open(custom_yaml, 'w') as f:
                content = f"path: {ds_path.absolute()}\ntrain: images/train\nval: images/val\nnames:\n"
                for i, name in enumerate(classes):
                    content += f"  {i}: {name}\n"
                f.write(content)
            data_yaml = custom_yaml

        status.update(label="✅ 数据集准备完成", state="complete")

    # 2. 开始训练
    st.markdown("### 🏋️ 训练日志")
    log_placeholder = st.empty()
    progress_bar = st.progress(0)
    log_lines = []

    device_arg = "" if device == "自动" else ("cpu" if device == "CPU" else "0")
    cmd = [
        _sys.executable, "train.py",
        "--data", str(data_yaml),
        "--model", train_model,
        "--epochs", str(epochs),
        "--imgsz", str(imgsz),
        "--batch", str(batch),
        "--lr", str(lr),
        "--patience", str(patience),
        "--workers", str(workers),
    ]
    if device_arg:
        cmd.extend(["--device", device_arg])

    with st.status("🏋️ 训练中...", expanded=True) as status:
        st.write(f"命令: `{' '.join(cmd)}`")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   cwd=str(Path(__file__).parent))
        current_epoch = 0
        for line in process.stdout:
            line = line.rstrip()
            log_lines.append(line)
            # 解析 epoch 进度
            if "Epoch" in line and "/" in line:
                try:
                    parts = line.split("Epoch")[1].strip().split("/")
                    current_epoch = int(parts[0].strip())
                    total_epochs = int(parts[1].split()[0].strip())
                    progress_bar.progress(min(current_epoch / total_epochs, 1.0))
                except:
                    pass
            # 显示最近20行日志
            log_placeholder.code("\n".join(log_lines[-20:]), language="bash")

        process.wait()
        if process.returncode == 0:
            status.update(label="✅ 训练完成!", state="complete")
        else:
            status.update(label="❌ 训练失败", state="error")
            st.error("训练出错，请检查日志"); return

    # 3. 显示结果
    best_model = Path("runs/train/custom_detect/weights/best.pt")
    # 搜索最新训练结果
    train_dir = Path("runs/train")
    if train_dir.exists():
        runs = sorted(train_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for r in runs:
            candidate = r / "weights" / "best.pt"
            if candidate.exists():
                best_model = candidate
                break

    if best_model.exists():
        st.markdown("### 📊 训练结果")
        st.success(f"✅ 最佳模型: `{best_model}`")

        # 复制到 models 目录
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        dest = models_dir / "yolov8_custom.pt"
        shutil.copy2(best_model, dest)
        st.info(f"📦 模型已复制到: `{dest}`")
        st.markdown("💡 **在左侧「检测模型」下拉中选择 `models/yolov8_custom.pt` 即可使用训练后的模型**")
    else:
        st.warning("⚠️ 未找到训练结果，请检查训练日志")


# ==================== 关于 ====================

def show_about():
    st.markdown("## ℹ️ 关于本项目")
    st.markdown("""
    ### 智能视觉系统

    | 功能 | 说明 |
    |------|------|
    | 📹 实时监测 | 非阻塞摄像头 · 手动截图 · 自动截图 · 告警 |
    | 📷 目标检测 | YOLOv8 检测 80+ 物体类别 · 支持批量处理 |
    | 👤 人脸识别 | YOLOv8 专用人脸检测模型 · 支持批量处理 |
    | ✋ 手势识别 | MediaPipe HandLandmarker · 支持批量处理 |
    | 😷 口罩检测 | YOLOv8 口罩佩戴检测 · 支持批量处理 |
    """)


if __name__ == '__main__':
    main()

