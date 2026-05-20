#!/usr/bin/env python3
"""
中药饮片智能鉴定系统 - v3.1 优化版 (RK3568 部署)
改进点：
  1. ONNX Runtime 图优化 + 4 线程
  2. 全局 Energy 阈值（One-Class）
  3. 启动时全局加载 JSON，避免每次请求 IO
  4. 移除 PatchCore（已验证增加误报）
  5. 药典数据 100% 完整（129 种）
  6. Top-2 种子类混淆仲裁（麦芽/谷芽/菟丝子/莱菔子/薏苡仁）
"""

import os
import sys
import json
import time
import numpy as np
from PIL import Image
from io import BytesIO
import onnxruntime as ort
from flask import Flask, request, jsonify, render_template_string

# ===================== 配置 =====================
MODEL_DIR = "/opt/herb_ai/models"
MODEL_PATH = os.path.join(MODEL_DIR, "herb_resnet50_phase3.onnx")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes_129.txt")
DICT_PATH = os.path.join(MODEL_DIR, "herb_dictionary.json")
ENERGY_THRESH_PATH = os.path.join(MODEL_DIR, "energy_thresholds_phase3.json")

GLOBAL_ENERGY_THRESHOLD = 21.5  # p95 of all authentic energies
INPUT_SIZE = 224
WARMUP_ITER = 3

app = Flask(__name__)

# ===================== 全局加载（启动时一次性） =====================
print("[INIT] Loading model and data...")

# 1. 类别列表
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    CLASSES = [line.strip() for line in f if line.strip()]
print(f"[INIT] Classes loaded: {len(CLASSES)}")

# 2. 药典数据
with open(DICT_PATH, "r", encoding="utf-8") as f:
    HERB_DICT = json.load(f)
print(f"[INIT] Dictionary loaded: {len(HERB_DICT)} herbs")

# 3. ONNX 模型（图优化 + 4 线程）
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = 4
opts.inter_op_num_threads = 4
opts.log_severity_level = 3

sess = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name
print(f"[INIT] ONNX model loaded: {os.path.basename(MODEL_PATH)}")

# 4. Warmup
dummy = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
for _ in range(WARMUP_ITER):
    sess.run([out_name], {inp_name: dummy})
print("[INIT] Warmup done. Service ready.")

# ===================== 工具函数 =====================
def preprocess(image_bytes):
    """预处理：bytes -> (1,3,224,224) numpy"""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((256, 256), Image.BILINEAR)
    img = img.crop((16, 16, 240, 240))  # center crop 224
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, 0)

def energy_score(logits, T=1.0):
    return -T * np.log(np.sum(np.exp(logits / T)))

def apply_seed_arbitration(results):
    """
    种子类混淆仲裁函数（内嵌简化版）
    results: [{'name': str, 'confidence': float}, ...]
    返回: {'arbitrated': bool, 'message': str|None}
    """
    if len(results) < 2:
        return {'arbitrated': False, 'message': None}
    
    r1, r2 = results[0], results[1]
    n1, c1 = r1['name'], r1['confidence']
    n2, c2 = r2['name'], r2['confidence']
    gap = c1 - c2
    
    CN = {'maiya': '麦芽', 'guya': '谷芽', 'tusizi': '菟丝子', 
          'laifuzi': '莱菔子', 'yiyiren': '薏苡仁'}
    SEED_CLASSES = {'maiya', 'guya', 'tusizi', 'laifuzi', 'yiyiren'}
    
    # 策略1: 麦芽-谷芽互混（核心策略）
    if {n1, n2} == {'maiya', 'guya'} and gap < 0.20:
        return {'arbitrated': True,
                'message': '疑似{}（与{}相似度{:.1%}），建议人工复核'.format(
                    CN.get(n1, n1), CN.get(n2, n2), c2)}
    
    # 策略2: 谷芽被错分为其他种子类，Top-2是谷芽时挽救提示
    if n2 == 'guya' and n1 in ('tusizi', 'yiyiren', 'laifuzi') and c1 < 0.85:
        return {'arbitrated': True,
                'message': '疑似{}（可能为谷芽，相似度{:.1%}），建议人工复核'.format(
                    CN.get(n1, n1), c2)}
    
    # 策略3: 通用种子类低置信度兜底
    if n1 in SEED_CLASSES and c1 < 0.80 and n2 in SEED_CLASSES:
        return {'arbitrated': True,
                'message': '疑似{}（与{}相似度{:.1%}），建议人工复核'.format(
                    CN.get(n1, n1), CN.get(n2, n2), c2)}
    
    return {'arbitrated': False, 'message': None}

# ===================== API =====================

@app.route("/")
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>中药饮片AI鉴定</title></head>
    <body>
        <h1>中药饮片智能鉴定系统 v3.1</h1>
        <p>状态: 运行中 | 模型: ResNet50 Phase 3 | 阈值: 全局 Energy 21.5</p>
        <p>功能: 品种识别 + Energy真伪鉴别 + Top-2种子类仲裁</p>
        <p>API: POST /identify (file=image)</p>
    </body>
    </html>
    """)

@app.route("/identify", methods=["POST"])
def identify():
    t_start = time.time()
    
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    image_bytes = file.read()
    
    # 预处理
    t0 = time.time()
    inp = preprocess(image_bytes)
    t_pre = time.time() - t0
    
    # 推理
    t0 = time.time()
    logits = sess.run([out_name], {inp_name: inp})[0][0]
    t_infer = time.time() - t0
    
    # 后处理
    t0 = time.time()
    top1_idx = int(np.argmax(logits))
    top1_class = CLASSES[top1_idx]
    top1_prob = float(np.exp(logits[top1_idx]) / np.sum(np.exp(logits)))
    
    # Energy Score 全局阈值判定
    e = energy_score(logits)
    is_alert = bool(e > GLOBAL_ENERGY_THRESHOLD)
    
    # Top-5
    top5_idx = np.argsort(logits)[-5:][::-1]
    top5 = [{"class": CLASSES[i], "prob": float(np.exp(logits[i]) / np.sum(np.exp(logits)))} for i in top5_idx]
    
    # ===== Top-2 种子类混淆仲裁（v2026-05-21 新增）=====
    arb_input = [{"name": item["class"], "confidence": item["prob"]} for item in top5]
    arb_result = apply_seed_arbitration(arb_input)
    
    # 如果仲裁触发，提升告警级别
    if arb_result['arbitrated']:
        is_alert = True
    
    # 药典数据
    info = HERB_DICT.get(top1_class, {})
    
    t_total = time.time() - t_start
    
    result = {
        "success": True,
        "predicted_class": top1_class,
        "predicted_name": info.get("chinese", top1_class),
        "confidence": round(top1_prob, 4),
        "energy_score": round(float(e), 2),
        "energy_threshold": GLOBAL_ENERGY_THRESHOLD,
        "authenticity_alert": is_alert,
        "authenticity_message": (
            arb_result['message'] if arb_result['arbitrated']
            else ("WARN 能量异常，建议人工复核" if is_alert else "OK 能量正常")
        ),
        "arbitration": {
            "triggered": arb_result['arbitrated'],
            "message": arb_result['message']
        },
        "category": info.get("category", ""),
        "nature": info.get("nature", ""),
        "meridian": info.get("meridian", ""),
        "functions": info.get("functions", ""),
        "indications": info.get("indications", ""),
        "precautions": info.get("precautions", ""),
        "top5": top5,
        "timing": {
            "preprocess_ms": round(t_pre * 1000, 1),
            "inference_ms": round(t_infer * 1000, 1),
            "postprocess_ms": round((t_total - t_pre - t_infer) * 1000, 1),
            "total_ms": round(t_total * 1000, 1)
        }
    }
    
    return jsonify(result)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "herb_resnet50_phase3", "classes": len(CLASSES),
                    "features": ["global_energy", "seed_arbitration"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
