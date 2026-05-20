#!/usr/bin/env python3
"""
Energy Score 阈值校准（增量保存版）
每类只取前 100 张，处理完一类立即保存，避免超时丢失
"""

import os
import json
import numpy as np
from PIL import Image
import onnxruntime as ort
from collections import defaultdict

MODEL_ONNX = r'D:\herb_ai_local\models\herb_resnet50_phase3.onnx'
CLASS_MAP = r'D:\herb_ai_local\resnet50_output\class_map.json'
DATA_DIR = r'D:\herb_ai_local\data_90\val'
OUTPUT_JSON = r'D:\herb_ai_local\energy_thresholds_phase3.json'
MAX_PER_CLASS = 100  # 每类最多取 100 张统计
T = 1.0

def preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))
    img = img.crop((16, 16, 240, 240))
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return arr

def energy_score(logits, T=1.0):
    return float(-T * np.log(np.sum(np.exp(logits / T))))

def main():
    print("[INFO] Energy Score 阈值校准（增量保存版）")
    
    with open(CLASS_MAP, 'r', encoding='utf-8') as f:
        class_map = json.load(f)
    
    session = ort.InferenceSession(MODEL_ONNX, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    thresholds = {}
    all_energies = []
    
    classes = sorted([c for c in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, c))])
    
    for idx, class_name in enumerate(classes, 1):
        class_dir = os.path.join(DATA_DIR, class_name)
        files = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
        files.sort()
        files = files[:MAX_PER_CLASS]
        
        energies = []
        for fname in files:
            img_path = os.path.join(class_dir, fname)
            try:
                inp = preprocess(img_path)
                logits = session.run([output_name], {input_name: inp})[0][0]
                energies.append(energy_score(logits, T))
            except Exception as e:
                print(f"  [WARN] {fname}: {e}")
        
        if len(energies) == 0:
            continue
        
        energies_arr = np.array(energies)
        mean = float(np.mean(energies_arr))
        std = float(np.std(energies_arr))
        p95 = float(np.percentile(energies_arr, 95))
        p99 = float(np.percentile(energies_arr, 99))
        
        thresholds[class_name] = {
            'mean': round(mean, 4), 'std': round(std, 4),
            'min': round(float(np.min(energies_arr)), 4),
            'max': round(float(np.max(energies_arr)), 4),
            'p95': round(p95, 4), 'p99': round(p99, 4),
            'threshold': round(p95, 4),
            'num_samples': len(energies)
        }
        all_energies.extend(energies)
        
        # 增量保存
        temp_output = {
            'method': 'energy_score', 'temperature': T,
            'progress': f'{idx}/{len(classes)}',
            'class_thresholds': thresholds
        }
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(temp_output, f, ensure_ascii=False, indent=2)
        
        print(f"[{idx:3d}/{len(classes)}] {class_name:<15s} | n={len(energies):3d} | mean={mean:7.2f} | std={std:6.2f} | p95={p95:7.2f}")
    
    # 最终保存
    all_arr = np.array(all_energies)
    final_output = {
        'method': 'energy_score',
        'temperature': T,
        'description': 'Energy = -T * log(sum(exp(logits/T))). threshold=p95 保守策略',
        'global_threshold': round(float(np.percentile(all_arr, 95)), 4),
        'global_mean': round(float(np.mean(all_arr)), 4),
        'global_std': round(float(np.std(all_arr)), 4),
        'num_classes': len(thresholds),
        'class_thresholds': thresholds
    }
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    print(f"\n[INFO] 完成！保存至: {OUTPUT_JSON}")
    print(f"[INFO] 全局阈值(p95): {final_output['global_threshold']:.2f}")

if __name__ == '__main__':
    main()
