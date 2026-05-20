#!/usr/bin/env python3
"""
遮挡敏感度热力图（Occlusion Sensitivity）- 纯 ONNX 实现
通过遮挡图像不同区域，观察分类概率变化，定位模型关注区域
"""

import os
import numpy as np
from PIL import Image
import onnxruntime as ort
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ONNX_PATH = r'D:\herb_ai_local\models\herb_resnet50_phase3.onnx'
CLASSES_PATH = r'D:\herb_ai_local\data_90_classes.txt'

with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
    CLASSES = [line.strip() for line in f if line.strip()]

opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = 4
opts.log_severity_level = 3
sess = ort.InferenceSession(ONNX_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
inp_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name

def preprocess_arr(arr_224):
    """将 (224,224,3) numpy 数组预处理为模型输入"""
    arr = arr_224.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, 0)

def get_prob(logits, target_idx):
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    return float(probs[target_idx])

def generate_occlusion(image_path, target_class, grid_size=14, patch_size=16):
    """
    遮挡敏感度分析
    grid_size: 14x14 grid, 每个 patch 16x16 pixels
    """
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_arr = np.array(img)
    
    # 基准概率（无遮挡）
    base_logits = sess.run([out_name], {inp_name: preprocess_arr(img_arr)})[0][0]
    target_idx = CLASSES.index(target_class)
    base_prob = get_prob(base_logits, target_idx)
    
    # 遮挡网格
    sensitivity = np.zeros((grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            occluded = img_arr.copy()
            y1, y2 = i * patch_size, (i + 1) * patch_size
            x1, x2 = j * patch_size, (j + 1) * patch_size
            occluded[y1:y2, x1:x2] = 128  # 灰色遮挡
            
            logits = sess.run([out_name], {inp_name: preprocess_arr(occluded)})[0][0]
            prob = get_prob(logits, target_idx)
            
            # 概率下降越多 = 该区域越重要
            sensitivity[i, j] = base_prob - prob
    
    return img_arr, sensitivity, base_prob

def plot_occlusion(image_path, target_class, save_path, grid_size=14):
    img_arr, sens, base_prob = generate_occlusion(image_path, target_class, grid_size)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(img_arr)
    axes[0].set_title(f'Original: {target_class}\nBase Prob: {base_prob:.3f}')
    axes[0].axis('off')
    
    # 遮挡敏感度热力图
    im1 = axes[1].imshow(sens, cmap='hot', interpolation='bilinear')
    axes[1].set_title('Occlusion Sensitivity\n(Red = Important)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # 叠加图
    from scipy.ndimage import zoom
    sens_up = zoom(sens, 224 / grid_size, order=1)
    axes[2].imshow(img_arr)
    axes[2].imshow(sens_up, cmap='hot', alpha=0.5, interpolation='bilinear')
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved: {save_path}')

# 对 4 个关键品类各生成一张
samples = [
    (r'D:\herb_ai_local\data_authenticity\danshen\val\authentic\丹参12.jpg', 'danshen'),
    (r'D:\herb_ai_local\data_authenticity\danshen\val\fake\云南丹参3.jpg', 'danshen'),
    (r'D:\herb_ai_local\data_authenticity\dangshen\val\authentic\党参17.jpg', 'dangshen'),
    (r'D:\herb_ai_local\data_authenticity\renshen\val\authentic\人参1.jpg', 'renshen'),
]

for path, cls in samples:
    if os.path.exists(path):
        save = path.rsplit('.', 1)[0] + '_occlusion.png'
        plot_occlusion(path, cls, save)
    else:
        print(f'Skip missing: {path}')

print('All occlusion maps generated.')
