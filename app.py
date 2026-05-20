#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中药饮片智能鉴定系统 - Web 服务 (ResNet50 ONNX + TTA)
部署于 RK3568 开发板
功能：开机欢迎页 → 主菜单（拍照/传图）→ 鉴定 → 结果展示
"""

import os
import sys
import json
import base64
import io
import subprocess
import threading
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template_string, Response
import onnxruntime as ort

app = Flask(__name__)

# ==================== 配置 ====================
MODEL_PATH = '/opt/herb_ai/models/herb_resnet50.onnx'
CLASSES_PATH = '/opt/herb_ai/models/classes_129.txt'
DICT_PATH = '/opt/herb_ai/models/herb_dictionary.json'
INPUT_SIZE = 224
UPLOAD_FOLDER = '/tmp/herb_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALERT_THRESHOLD = 0.95

# Camera capture settings
CAM_DEV_PREVIEW = '/dev/video1'   # rkisp_selfpath: 预览用
CAM_DEV_SNAP = '/dev/video0'      # rkisp_mainpath: 拍照用
CAM_LOCK_SNAP = threading.Lock()
CAM_PREVIEW_W, CAM_PREVIEW_H = 640, 480
CAM_SNAP_W, CAM_SNAP_H = 1920, 1080

# ===================== MJPEG 实时流 =====================

class CameraStreamer:
    """后台 GStreamer 持续捕获，输出 MJPEG 帧到共享缓冲区"""
    def __init__(self, dev, width, height, quality=70, contrast=1.0, brightness=0.0, saturation=1.0, whitebalance=True):
        self.dev = dev
        self.width = width
        self.height = height
        self.quality = quality
        self.contrast = contrast
        self.brightness = brightness
        self.saturation = saturation
        self.whitebalance = whitebalance
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.proc = None
        self._thread = None
        self._start()

    def _white_balance(self, image):
        """灰度世界白平衡（numpy 加速，约 5-10ms@640x480）"""
        arr = np.array(image).astype(np.float32)
        r_avg = arr[:, :, 0].mean()
        g_avg = arr[:, :, 1].mean()
        b_avg = arr[:, :, 2].mean()
        gray = (r_avg + g_avg + b_avg) / 3.0
        if r_avg > 0 and g_avg > 0 and b_avg > 0:
            arr[:, :, 0] = np.clip(arr[:, :, 0] * (gray / r_avg), 0, 255)
            arr[:, :, 1] = np.clip(arr[:, :, 1] * (gray / g_avg), 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] * (gray / b_avg), 0, 255)
        return Image.fromarray(arr.astype(np.uint8))

    def _start(self):
        # GStreamer: v4l2src → UYVY → videoconvert → videobalance(对比度/亮度/饱和度) → jpegenc → stdout
        cmd = [
            'gst-launch-1.0', 'v4l2src', 'device={}'.format(self.dev),
            '!', 'video/x-raw,width={},height={},format=UYVY'.format(self.width, self.height),
            '!', 'videoconvert',
            '!', 'videobalance', 'contrast={}'.format(self.contrast),
            'brightness={}'.format(self.brightness), 'saturation={}'.format(self.saturation),
            '!', 'jpegenc', 'quality={}'.format(self.quality),
            '!', 'fdsink', 'fd=1'
        ]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self._thread = threading.Thread(target=self._read_frames, daemon=True)
            self._thread.start()
            print('[STREAM] Started {} {}x{} q={} wb={}'.format(self.dev, self.width, self.height, self.quality, self.whitebalance))
        except Exception as e:
            print('[STREAM] Failed to start {}: {}'.format(self.dev, e))

    def _read_frames(self):
        buf = b''
        while self.running:
            try:
                chunk = self.proc.stdout.read(16384)
                if not chunk:
                    break
                buf += chunk
                # 解析 JPEG 帧边界: FF D8 ... FF D9
                while True:
                    start = buf.find(b'\xff\xd8')
                    if start == -1:
                        buf = b''
                        break
                    end = buf.find(b'\xff\xd9', start + 2)
                    if end == -1:
                        buf = buf[start:]
                        break
                    frame = buf[start:end + 2]
                    # 白平衡校正
                    if self.whitebalance:
                        try:
                            img = Image.open(io.BytesIO(frame))
                            img = self._white_balance(img)
                            out = io.BytesIO()
                            img.save(out, format='JPEG', quality=self.quality)
                            frame = out.getvalue()
                        except Exception as e:
                            print('[STREAM] WB error:', e)
                    with self.lock:
                        self.latest_frame = frame
                    buf = buf[end + 2:]
            except Exception as e:
                print('[STREAM] read error:', e)
                break
        print('[STREAM] Reader thread exited for {}'.format(self.dev))

    def get_frame(self):
        with self.lock:
            return self.latest_frame

    def stop(self):
        self.running = False
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()

    def restart(self):
        self.stop()
        self.running = True
        self._start()

# 全局预览流实例 (video1, 640x480)
# 对比度1.2、饱和度1.2 增强药材纹理和颜色，启用灰度世界白平衡校正偏绿
PREVIEW_STREAMER = CameraStreamer(
    CAM_DEV_PREVIEW, CAM_PREVIEW_W, CAM_PREVIEW_H,
    quality=75, contrast=1.2, brightness=0.0, saturation=1.2, whitebalance=True
)

TTA_TRANSFORMS = [
    ('original', lambda img: img),
    ('hflip', lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)),
    ('vflip', lambda img: img.transpose(Image.FLIP_TOP_BOTTOM)),
    ('crop_left', lambda img: img.crop((0, 0, int(img.width*0.9), img.height)).resize((INPUT_SIZE, INPUT_SIZE))),
    ('crop_right', lambda img: img.crop((int(img.width*0.1), 0, img.width, img.height)).resize((INPUT_SIZE, INPUT_SIZE))),
]
# =============================================

with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
    CLASSES = [line.strip() for line in f if line.strip()]

HERB_DICT = {}
if os.path.exists(DICT_PATH):
    with open(DICT_PATH, 'r', encoding='utf-8') as f:
        HERB_DICT = json.load(f)
    print('[INFO] 电子药典加载成功 | {} 味药材'.format(len(HERB_DICT)))
else:
    print('[WARN] 未找到电子药典: {}'.format(DICT_PATH))

print('[INFO] 加载 ONNX 模型...')
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print('[INFO] ONNX Runtime 就绪 | 输入: {}, 输出: {}'.format(input_name, output_name))

# ============ 公共样式 ============
COMMON_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; -webkit-tap-highlight-color: transparent; user-select: none; }
html, body {
    width: 100%; height: 100%; overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans SC", "Microsoft YaHei", sans-serif;
    background: #1a1a2e; color: #eee;
}
#app { width: 100%; height: 100%; display: flex; flex-direction: column; }
.btn-primary {
    width: 100%; height: 64px; border: none; border-radius: 16px;
    background: linear-gradient(90deg, #e94560 0%, #ff6b6b 100%);
    color: #fff; font-size: 22px; font-weight: 700;
    cursor: pointer; letter-spacing: 3px;
    box-shadow: 0 6px 20px rgba(233, 69, 96, 0.35);
    transition: transform 0.15s, box-shadow 0.15s;
}
.btn-primary:active { transform: scale(0.97); box-shadow: 0 3px 10px rgba(233, 69, 96, 0.25); }
.btn-secondary {
    width: 100%; height: 64px; border: 2px solid #0f3460; border-radius: 16px;
    background: #16213e; color: #ccc; font-size: 22px; font-weight: 700;
    cursor: pointer; letter-spacing: 3px;
    transition: all 0.15s;
}
.btn-secondary:active { background: #1a2a4a; border-color: #e94560; }
.spinner {
    display: inline-block; width: 48px; height: 48px;
    border: 5px solid #333; border-top-color: #e94560;
    border-radius: 50%; animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
"""

# ============ 欢迎页 ============
WELCOME_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>中药饮片智能鉴定系统</title>
<style>
""" + COMMON_CSS + """
.welcome-container {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 30px; padding: 24px;
}
.welcome-icon { font-size: 120px; filter: drop-shadow(0 0 30px rgba(233,69,96,0.4)); }
.welcome-title { font-size: 32px; font-weight: 800; color: #fff; letter-spacing: 4px; text-align: center; }
.welcome-subtitle { font-size: 16px; color: #888; text-align: center; margin-top: -20px; }
.welcome-btn { max-width: 320px; margin-top: 20px; }
.version-tag { position: absolute; bottom: 20px; font-size: 12px; color: #555; }
</style></head>
<body>
<div id="app">
    <div class="welcome-container">
        <div class="welcome-icon">🌿</div>
        <div class="welcome-title">中药饮片<br>智能鉴定系统</div>
        <div class="welcome-subtitle">基于 ResNet50 + TTA | 129 类药材 | RK3568 离线运行</div>
        <button class="btn-primary welcome-btn" onclick="location.href='/main'">👉 点击进入系统</button>
        <div class="version-tag">Powered by RK3568 | ONNX Runtime CPU</div>
    </div>
</div>
</body></html>
"""

# ============ 主菜单 ============
MAIN_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>选择鉴定方式</title>
<style>
""" + COMMON_CSS + """
.header {
    flex-shrink: 0; height: 56px;
    background: linear-gradient(90deg, #16213e 0%, #0f3460 100%);
    display: flex; align-items: center; justify-content: center;
    border-bottom: 2px solid #e94560;
}
.header h1 { font-size: 20px; font-weight: 700; color: #fff; letter-spacing: 2px; }
.main { flex: 1; display: flex; flex-direction: column; padding: 20px; gap: 20px; justify-content: center; }
.menu-card {
    flex: 1; background: #16213e; border-radius: 20px; border: 2px dashed #0f3460;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 16px; cursor: pointer; transition: all 0.25s; max-height: 220px;
}
.menu-card:active { border-color: #e94560; background: #1a2a4a; transform: scale(0.98); }
.menu-icon { font-size: 64px; }
.menu-title { font-size: 24px; font-weight: 700; color: #fff; }
.menu-desc { font-size: 14px; color: #888; }
.back-btn { position: absolute; left: 16px; top: 14px; background: none; border: none; color: #fff; font-size: 24px; cursor: pointer; }
</style></head>
<body>
<div id="app">
    <div class="header">
        <button class="back-btn" onclick="location.href='/'">◀</button>
        <h1>📖 选择鉴定方式</h1>
    </div>
    <div class="main">
        <div class="menu-card" onclick="location.href='/camera'">
            <div class="menu-icon">📷</div>
            <div class="menu-title">拍照鉴定</div>
            <div class="menu-desc">调用摄像头实时拍摄药材</div>
        </div>
        <div class="menu-card" onclick="location.href='/upload'">
            <div class="menu-icon">📤</div>
            <div class="menu-title">上传图片</div>
            <div class="menu-desc">从本地选择图片文件</div>
        </div>
    </div>
</div>
</body></html>
"""

# ============ 拍照页 ============
CAMERA_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>拍照鉴定</title>
<style>
""" + COMMON_CSS + """
.header {
    flex-shrink: 0; height: 56px;
    background: linear-gradient(90deg, #16213e 0%, #0f3460 100%);
    display: flex; align-items: center; justify-content: center;
    border-bottom: 2px solid #e94560; position: relative;
}
.header h1 { font-size: 20px; font-weight: 700; color: #fff; letter-spacing: 2px; }
.back-btn { position: absolute; left: 16px; top: 14px; background: none; border: none; color: #fff; font-size: 24px; cursor: pointer; }
.cam-area { flex: 1; position: relative; background: #000; display: flex; align-items: center; justify-content: center; overflow: hidden; }
.cam-area img { max-width: 100%; max-height: 100%; object-fit: contain; }
.cam-overlay { position: absolute; top: 0; left: 0; right: 0; bottom: 0; border: 2px dashed rgba(233,69,96,0.5); margin: 40px; border-radius: 16px; pointer-events: none; }
.cam-hint { position: absolute; top: 12px; left: 0; right: 0; text-align: center; font-size: 14px; color: rgba(255,255,255,0.7); }
.controls {
    flex-shrink: 0; height: 100px; background: #0f0f1a;
    display: flex; align-items: center; justify-content: center; gap: 20px; padding: 0 24px;
}
.shutter-btn {
    width: 72px; height: 72px; border-radius: 50%; border: 4px solid #e94560;
    background: #fff; cursor: pointer; transition: transform 0.1s;
}
.shutter-btn:active { transform: scale(0.9); background: #e94560; }
.shutter-inner { width: 56px; height: 56px; border-radius: 50%; background: #e94560; margin: 4px; }
.ctrl-btn { height: 48px; padding: 0 24px; border-radius: 12px; border: none; font-size: 16px; font-weight: 700; cursor: pointer; }
.ctrl-btn.retake { background: #444; color: #fff; }
.ctrl-btn.confirm { background: linear-gradient(90deg, #e94560, #ff6b6b); color: #fff; }
.loading-box { position: absolute; top:0;left:0;right:0;bottom:0; background: rgba(0,0,0,0.8); display: none; flex-direction: column; align-items: center; justify-content: center; gap: 16px; z-index: 10; }
.loading-box.active { display: flex; }
</style></head>
<body>
<div id="app">
    <div class="header">
        <button class="back-btn" onclick="location.href='/main'">◀</button>
        <h1>📷 拍照鉴定</h1>
    </div>
    <div class="cam-area" id="camArea">
        <img id="liveImg" src="/stream" alt="摄像头预览" style="width:100%;height:100%;object-fit:cover;">
        <img id="previewImg" style="display:none;max-width:100%;max-height:100%;object-fit:contain;">
        <div class="cam-overlay" id="overlay"></div>
        <div class="cam-hint" id="camHint">正在启动摄像头...</div>
        <div class="loading-box" id="loading">
            <div class="spinner"></div>
            <div style="color:#aaa;font-size:16px;">分析中...</div>
        </div>
    </div>
    <div class="controls" id="controls">
        <button class="shutter-btn" id="shutterBtn" onclick="takePhoto()">
            <div class="shutter-inner"></div>
        </button>
    </div>
</div>
<script>
const liveImg = document.getElementById('liveImg');
const previewImg = document.getElementById('previewImg');
const overlay = document.getElementById('overlay');
const camHint = document.getElementById('camHint');
const controls = document.getElementById('controls');
const loading = document.getElementById('loading');
let capturedBlob = null;

liveImg.onload = () => {
    camHint.textContent = '请将药材放入框内，点击快门拍照';
    camHint.style.color = 'rgba(255,255,255,0.7)';
};
liveImg.onerror = () => {
    camHint.textContent = '摄像头连接失败，请检查硬件';
    camHint.style.color = '#ff6b6b';
};

async function takePhoto() {
    camHint.textContent = '正在拍照...';
    try {
        const res = await fetch('/camera_snap?t=' + Date.now());
        const data = await res.json();
        if (!data.success) throw new Error(data.error || '拍照失败');
        const byteChars = atob(data.image_b64);
        const byteNumbers = new Array(byteChars.length);
        for (let i = 0; i < byteChars.length; i++) byteNumbers[i] = byteChars.charCodeAt(i);
        const byteArray = new Uint8Array(byteNumbers);
        capturedBlob = new Blob([byteArray], { type: 'image/jpeg' });
        previewImg.src = URL.createObjectURL(capturedBlob);
        previewImg.style.display = 'block';
        liveImg.style.display = 'none';
        overlay.style.display = 'none';
        camHint.textContent = '确认照片或重新拍摄';
        controls.innerHTML = `
            <button class="ctrl-btn retake" onclick="retake()">🔄 重拍</button>
            <button class="ctrl-btn confirm" onclick="confirmPhoto()">✅ 开始鉴定</button>
        `;
    } catch (e) {
        camHint.textContent = '拍照失败: ' + e.message;
        camHint.style.color = '#ff6b6b';
    }
}

function retake() {
    capturedBlob = null;
    previewImg.style.display = 'none';
    liveImg.style.display = 'block';
    overlay.style.display = 'block';
    camHint.textContent = '请将药材放入框内，点击快门拍照';
    controls.innerHTML = `
        <button class="shutter-btn" id="shutterBtn" onclick="takePhoto()">
            <div class="shutter-inner"></div>
        </button>
    `;
}

async function confirmPhoto() {
    if (!capturedBlob) return;
    loading.classList.add('active');
    const formData = new FormData();
    formData.append('image', capturedBlob, 'capture.jpg');
    try {
        const res = await fetch('/infer', { method: 'POST', body: formData });
        const data = await res.json();
        sessionStorage.setItem('herbResult', JSON.stringify(data));
        sessionStorage.setItem('herbImage', previewImg.src);
        location.href = '/result';
    } catch (err) {
        loading.classList.remove('active');
        alert('鉴定失败: ' + err);
    }
}
</script>
</body></html>
"""

# ============ 传图页（现有功能升级版） ============
UPLOAD_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>上传图片鉴定</title>
<style>
""" + COMMON_CSS + """
.header {
    flex-shrink: 0; height: 56px;
    background: linear-gradient(90deg, #16213e 0%, #0f3460 100%);
    display: flex; align-items: center; justify-content: center;
    border-bottom: 2px solid #e94560; position: relative;
}
.header h1 { font-size: 20px; font-weight: 700; color: #fff; letter-spacing: 2px; }
.back-btn { position: absolute; left: 16px; top: 14px; background: none; border: none; color: #fff; font-size: 24px; cursor: pointer; }
.main { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 14px; }
.upload-card {
    background: #16213e; border-radius: 16px; border: 2px dashed #0f3460; padding: 20px;
    text-align: center; cursor: pointer; min-height: 180px;
    display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px;
}
.upload-card:active { border-color: #e94560; background: #1a2a4a; }
.upload-card.has-image { border-style: solid; border-color: #0f3460; }
.upload-icon { font-size: 56px; }
.upload-text { font-size: 18px; font-weight: 600; color: #ccc; }
.upload-hint { font-size: 14px; color: #888; }
.file-input { display: none; }
.preview-img { max-width: 100%; max-height: 200px; border-radius: 10px; object-fit: contain; }
.loading-box { display: none; text-align: center; padding: 20px; }
.loading-box.active { display: block; }
.status-bar { text-align: center; font-size: 12px; color: #666; padding: 4px; }
</style></head>
<body>
<div id="app">
    <div class="header">
        <button class="back-btn" onclick="location.href='/main'">◀</button>
        <h1>📤 上传图片鉴定</h1>
    </div>
    <div class="main" id="mainArea">
        <div class="upload-card" id="uploadArea" onclick="document.getElementById('fileInput').click()">
            <div class="upload-icon" id="uploadIcon">📷</div>
            <div class="upload-text" id="uploadText">点击上传药材图片</div>
            <div class="upload-hint" id="uploadHint">支持 JPG / PNG 格式</div>
            <input type="file" id="fileInput" class="file-input" accept="image/*" onchange="handleFile(this.files[0])">
        </div>
        <div id="previewContainer"></div>
        <button class="btn-primary" id="inferBtn" onclick="doInference()" disabled>🔍 开始鉴定</button>
        <div class="loading-box" id="loading">
            <div class="spinner"></div>
            <div class="loading-text">多角度分析中，请稍候...</div>
        </div>
        <div class="status-bar">ResNet50 + TTA | 129 类药材 | RK3568 离线运行</div>
    </div>
</div>
<script>
let currentFile = null;
const uploadArea = document.getElementById('uploadArea');
const uploadIcon = document.getElementById('uploadIcon');
const uploadText = document.getElementById('uploadText');
const uploadHint = document.getElementById('uploadHint');
const inferBtn = document.getElementById('inferBtn');
const loading = document.getElementById('loading');
const previewContainer = document.getElementById('previewContainer');

function handleFile(file) {
    if (!file) return;
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewContainer.innerHTML = `<img src="${e.target.result}" class="preview-img">`;
        uploadArea.classList.add('has-image');
        uploadIcon.textContent = '✅';
        uploadText.textContent = '已选择图片';
        uploadHint.textContent = file.name;
        inferBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

async function doInference() {
    if (!currentFile) return;
    inferBtn.disabled = true;
    loading.classList.add('active');
    const formData = new FormData();
    formData.append('image', currentFile);
    try {
        const res = await fetch('/infer', { method: 'POST', body: formData });
        const data = await res.json();
        const reader = new FileReader();
        reader.onload = (e) => {
            sessionStorage.setItem('herbResult', JSON.stringify(data));
            sessionStorage.setItem('herbImage', e.target.result);
            location.href = '/result';
        };
        reader.readAsDataURL(currentFile);
    } catch (err) {
        loading.classList.remove('active');
        inferBtn.disabled = false;
        alert('推理失败: ' + err);
    }
}
</script>
</body></html>
"""

# ============ 结果页 ============
RESULT_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>鉴定结果</title>
<style>
""" + COMMON_CSS + """
.header {
    flex-shrink: 0; height: 56px;
    background: linear-gradient(90deg, #16213e 0%, #0f3460 100%);
    display: flex; align-items: center; justify-content: center;
    border-bottom: 2px solid #e94560; position: relative;
}
.header h1 { font-size: 20px; font-weight: 700; color: #fff; letter-spacing: 2px; }
.back-btn { position: absolute; left: 16px; top: 14px; background: none; border: none; color: #fff; font-size: 24px; cursor: pointer; }
.main { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 14px; }
.preview-box { background: #16213e; border-radius: 16px; padding: 12px; text-align: center; border: 1px solid #0f3460; }
.preview-box img { max-width: 100%; max-height: 180px; border-radius: 10px; object-fit: contain; }
.alert-box { border-radius: 12px; padding: 14px 16px; display: flex; align-items: flex-start; gap: 12px; }
.alert-box.danger { background: rgba(233, 69, 96, 0.15); border: 2px solid #e94560; }
.alert-box.success { background: rgba(46, 204, 113, 0.12); border: 2px solid #2ecc71; }
.alert-icon { font-size: 28px; }
.alert-title { font-size: 17px; font-weight: 700; margin-bottom: 4px; }
.alert-box.danger .alert-title { color: #ff6b6b; }
.alert-box.success .alert-title { color: #2ecc71; }
.alert-desc { font-size: 13px; color: #bbb; line-height: 1.5; }
.top1-box {
    background: linear-gradient(90deg, #0f3460 0%, #16213e 100%);
    border-radius: 14px; padding: 18px; border-left: 5px solid #e94560; position: relative;
}
.top1-box.authentic { border-left-color: #2ecc71; }
.top1-name { font-size: 26px; font-weight: 800; color: #fff; margin-bottom: 6px; display: flex; align-items: center; gap: 10px; }
.top1-pinyin { font-size: 14px; color: #888; font-weight: 400; }
.top1-tag { display: inline-block; padding: 4px 12px; border-radius: 8px; font-size: 13px; font-weight: 700; }
.top1-tag.authentic { background: #2ecc71; color: #fff; }
.top1-tag.suspect { background: #e94560; color: #fff; }
.top1-confidence { font-size: 15px; color: #ccc; margin-top: 8px; }
.top1-confidence strong { color: #fff; font-size: 20px; }
.result-list { display: flex; flex-direction: column; gap: 8px; }
.result-row {
    display: flex; align-items: center; gap: 10px;
    background: #1a1a2e; border-radius: 10px; padding: 10px 12px;
}
.result-rank { width: 32px; height: 32px; border-radius: 50%; background: #0f3460; color: #fff; font-weight: 700; display: flex; align-items: center; justify-content: center; font-size: 14px; flex-shrink: 0; }
.result-rank.first { background: #e94560; }
.result-detail { flex: 1; min-width: 0; }
.result-name-small { font-size: 15px; font-weight: 600; color: #ddd; }
.result-bar-bg { background: #333; border-radius: 4px; height: 10px; margin-top: 6px; overflow: hidden; }
.result-bar { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #e94560, #ff6b6b); transition: width 0.6s ease; }
.result-bar.authentic { background: linear-gradient(90deg, #2ecc71, #27ae60); }
.result-score { font-size: 14px; color: #aaa; font-weight: 600; flex-shrink: 0; }
.pharma-card { background: #1a1a2e; border-radius: 14px; padding: 16px; border: 1px solid #333; }
.pharma-header { font-size: 18px; font-weight: 700; color: #fff; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 2px solid #0f3460; }
.pharma-grid { display: grid; grid-template-columns: 70px 1fr; gap: 8px 12px; font-size: 14px; line-height: 1.5; }
.pharma-label { color: #e94560; font-weight: 700; }
.pharma-value { color: #ccc; }
.pharma-value.warn { color: #ff6b6b; font-weight: 600; }
.pharma-divider { grid-column: 1 / -1; height: 1px; background: #333; margin: 4px 0; }
.review-box { background: rgba(255, 193, 7, 0.1); border: 2px solid #ffc107; border-radius: 12px; padding: 14px; margin-top: 12px; }
.review-title { font-size: 15px; font-weight: 700; color: #ffc107; margin-bottom: 8px; }
.review-text { font-size: 13px; color: #ddd; line-height: 1.6; }
.footer-bar { flex-shrink: 0; height: 32px; background: #0f0f1a; display: flex; align-items: center; justify-content: center; font-size: 11px; color: #555; }
.action-btns { display: flex; gap: 12px; margin-top: 10px; }
</style></head>
<body>
<div id="app">
    <div class="header">
        <button class="back-btn" onclick="location.href='/main'">◀</button>
        <h1>📋 鉴定结果</h1>
    </div>
    <div class="main" id="mainArea"></div>
    <div class="footer-bar">Powered by RK3568 | ONNX Runtime CPU</div>
</div>
<script>
const data = JSON.parse(sessionStorage.getItem('herbResult') || '{}');
const imgSrc = sessionStorage.getItem('herbImage') || '';
const main = document.getElementById('mainArea');

if (!data.success && !data.results) {
    main.innerHTML = '<div style="text-align:center;padding:40px;color:#888;">暂无结果，请返回重新鉴定</div>';
} else {
    const p = data.pharmacopoeia;
    const isAlert = data.alert;
    const confPct = (data.top1_confidence * 100).toFixed(1);

    let html = `<div class="preview-box"><img src="${imgSrc}"></div>`;

    html += isAlert
        ? `<div class="alert-box danger"><span class="alert-icon">⚠️</span><div><div class="alert-title">疑似伪品，建议人工复核</div><div class="alert-desc">Top-1 置信度 ${confPct}% &lt; 95%，样本可能为伪品或异常</div></div></div>`
        : `<div class="alert-box success"><span class="alert-icon">✅</span><div><div class="alert-title">系统判定为正品</div><div class="alert-desc">置信度 ${confPct}%，识别结果可信度高</div></div></div>`;

    html += `<div class="top1-box ${isAlert ? '' : 'authentic'}"><div class="top1-name">${p.chinese}<span class="top1-pinyin">${p.pinyin}</span><span class="top1-tag ${isAlert ? 'suspect' : 'authentic'}">${isAlert ? '疑似伪品' : '正品'}</span></div><div class="top1-confidence">置信度：<strong>${data.top1_confidence.toFixed(4)}</strong></div></div>`;

    html += '<div class="result-list">';
    data.results.forEach((item, i) => {
        const width = Math.round(item.confidence * 100);
        html += `<div class="result-row"><div class="result-rank ${i===0?'first':''}">${i+1}</div><div class="result-detail"><div class="result-name-small">${item.name}</div><div class="result-bar-bg"><div class="result-bar ${i===0 && !isAlert ? 'authentic' : ''}" style="width:${width}%"></div></div></div><div class="result-score">${item.confidence.toFixed(4)}</div></div>`;
    });
    html += '</div>';

    if (p) {
        html += `<div class="pharma-card"><div class="pharma-header">📖 电子药典</div><div class="pharma-grid"><div class="pharma-label">分类</div><div class="pharma-value">${p.category}</div><div class="pharma-label">性味</div><div class="pharma-value">${p.nature}</div><div class="pharma-label">归经</div><div class="pharma-value">${p.meridian}</div><div class="pharma-divider"></div><div class="pharma-label">功效</div><div class="pharma-value">${p.functions}</div><div class="pharma-divider"></div><div class="pharma-label">主治</div><div class="pharma-value">${p.indications}</div><div class="pharma-divider"></div><div class="pharma-label">注意</div><div class="pharma-value warn">${p.precautions}</div></div></div>`;
        if (isAlert) {
            html += `<div class="review-box"><div class="review-title">🔍 人工复核提示</div><div class="review-text">1. 对照《中国药典》性状描述核对样品外观<br>2. 注意常见伪品特征及鉴别要点<br>3. 必要时送检进行显微或理化鉴别</div></div>`;
        }
    }

    html += `<div class="action-btns"><button class="btn-secondary" onclick="location.href='/main'">🏠 返回首页</button><button class="btn-primary" onclick="history.back()">🔄 重新鉴定</button></div>`;
    main.innerHTML = html;
}
</script>
</body></html>
"""

# ===================== 后端路由 =====================

@app.route('/')
def index():
    return render_template_string(WELCOME_HTML)

@app.route('/main')
def main_menu():
    return render_template_string(MAIN_HTML)

@app.route('/camera')
def camera_page():
    return render_template_string(CAMERA_HTML)

@app.route('/upload')
def upload_page():
    return render_template_string(UPLOAD_HTML)

@app.route('/result')
def result_page():
    return render_template_string(RESULT_HTML)

# ===================== 推理逻辑 =====================

def preprocess(image):
    img = image.convert('RGB')
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr

def inference_single(image):
    input_data = preprocess(image)
    outputs = session.run([output_name], {input_name: input_data})
    logits = outputs[0][0]
    exp_x = np.exp(logits - np.max(logits))
    probs = exp_x / np.sum(exp_x)
    return probs

def inference_tta(image):
    all_probs = []
    for name, transform in TTA_TRANSFORMS:
        try:
            t_img = transform(image)
            probs = inference_single(t_img)
            all_probs.append(probs)
        except Exception as e:
            print('[TTA] {} failed: {}'.format(name, e))
    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs

# ===================== 摄像头捕获 =====================

# uyvy_to_jpeg 已废弃，GStreamer 直接输出 JPEG
def uyvy_to_jpeg(uyvy_data, width, height, quality=85):
    """[DEPRECATED] 保留兼容，GStreamer 直接输出 JPEG"""
    return uyvy_data

def capture_from_device(dev, width, height, quality=90, contrast=1.0, brightness=0.0, saturation=1.0):
    """用 GStreamer 从指定 V4L2 设备捕获单帧并返回 JPEG bytes（拍照用）"""
    import tempfile
    fd, jpeg_path = tempfile.mkstemp(suffix='.jpg', prefix='herb_cap_')
    os.close(fd)
    cmd = [
        'gst-launch-1.0', 'v4l2src', 'device={}'.format(dev), 'num-buffers=1',
        '!', 'video/x-raw,width={},height={},format=UYVY'.format(width, height),
        '!', 'videoconvert',
        '!', 'videobalance', 'contrast={}'.format(contrast),
        'brightness={}'.format(brightness), 'saturation={}'.format(saturation),
        '!', 'jpegenc', 'quality={}'.format(quality),
        '!', 'filesink', 'location={}'.format(jpeg_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            err = result.stderr.strip()[-150:] if result.stderr else 'unknown'
            print('[CAM] {} failed (code={}): {}'.format(dev, result.returncode, err))
            return None
        with open(jpeg_path, 'rb') as f:
            data = f.read()
        return data
    except Exception as e:
        print('[CAM] {} exception: {}'.format(dev, e))
        return None
    finally:
        try:
            os.unlink(jpeg_path)
        except Exception:
            pass

@app.route('/capture')
def capture_preview():
    """返回最新单帧 JPEG（兼容旧接口）"""
    frame = PREVIEW_STREAMER.get_frame()
    if frame is None:
        # fallback: 直接捕获
        jpeg = capture_from_device(CAM_DEV_PREVIEW, CAM_PREVIEW_W, CAM_PREVIEW_H, quality=75)
        if jpeg is None:
            return 'Camera error', 503
        return Response(jpeg, mimetype='image/jpeg')
    return Response(frame, mimetype='image/jpeg')

@app.route('/stream')
def video_stream():
    """MJPEG 实时流，前端 <img src=\"/stream\"> 直接播放"""
    def generate():
        while True:
            frame = PREVIEW_STREAMER.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n'
                       + frame + b'\r\n')
            # 控制帧率 ~15fps，避免CPU过载
            import time
            time.sleep(0.066)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_snap')
def camera_snap():
    """拍照：直接从预览流获取最新帧，避免与 streamer 冲突"""
    frame = PREVIEW_STREAMER.get_frame()
    if frame is None:
        return jsonify({'error': 'Camera capture failed'}), 503
    import base64
    return jsonify({'success': True, 'image_b64': base64.b64encode(frame).decode('utf-8')})

def get_herb_info(class_name):
    info = HERB_DICT.get(class_name, {})
    if not info:
        return {
            'chinese': class_name, 'pinyin': class_name, 'category': '未知',
            'nature': '待补充', 'meridian': '待补充', 'functions': '待补充',
            'indications': '待补充', 'precautions': '待补充'
        }
    return info

@app.route('/infer', methods=['POST'])
def infer():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    try:
        image = Image.open(file.stream)
        probs = inference_tta(image)
        top5_idx = np.argsort(probs)[-5:][::-1]
        results = []
        for idx in top5_idx:
            name = CLASSES[idx] if idx < len(CLASSES) else 'class_{}'.format(idx)
            results.append({'name': name, 'confidence': float(probs[idx])})

        # ===== Top-2 种子类混淆仲裁 =====
        SEED_CLASSES = {'maiya', 'guya', 'tusizi', 'laifuzi', 'yiyiren'}
        CN = {'maiya': '麦芽', 'guya': '谷芽', 'tusizi': '菟丝子',
              'laifuzi': '莱菔子', 'yiyiren': '薏苡仁'}
        arb_message = None
        if len(results) >= 2:
            r1, r2 = results[0], results[1]
            n1, c1, n2, c2 = r1['name'], r1['confidence'], r2['name'], r2['confidence']
            gap = c1 - c2
            # 策略1: 麦芽-谷芽互混
            if {n1, n2} == {'maiya', 'guya'} and gap < 0.20:
                arb_message = '疑似{}（与{}相似度{:.1%}），建议人工复核'.format(
                    CN.get(n1, n1), CN.get(n2, n2), c2)
            # 策略2: 谷芽被错分为其他种子类，Top-2是谷芽
            elif n2 == 'guya' and n1 in ('tusizi', 'yiyiren', 'laifuzi') and c1 < 0.85:
                arb_message = '疑似{}（可能为谷芽，相似度{:.1%}），建议人工复核'.format(
                    CN.get(n1, n1), c2)
        top1_conf = results[0]['confidence']
        top1_name = results[0]['name']
        alert = top1_conf < ALERT_THRESHOLD or arb_message is not None
        pharma = get_herb_info(top1_name)

        return jsonify({
            'success': True, 'results': results,
            'top1_confidence': top1_conf, 'top1_name': top1_name,
            'alert': alert, 'alert_message': arb_message if arb_message else ('疑似伪品，建议人工复核' if alert else None),
            'pharmacopoeia': pharma
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
