#!/usr/bin/env python3
"""
Phase 3: 数据对齐 + Freeze Backbone 轻量微调（优化版）
改进：
- 移除 WeightedRandomSampler（避免 227K 样本权重计算开销）
- 每 epoch 只快速验证 data_authenticity（4 品类）
- 每 3 epoch 验证一次 data_90 val（避免 56K 验证开销）
- num_workers=0（Windows 兼容性）
"""

import os
import sys
import json
import time
import warnings
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image

warnings.filterwarnings('ignore', category=UserWarning)

def safe_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except (OSError, IOError) as e:
        print(f'[WARN] Bad image skipped: {path}')
        return Image.new('RGB', (224, 224), (0, 0, 0))

# ===================== 配置 =====================
DATA_DIR = r'D:\herb_ai_local\data_90'
AUTHENTICITY_DIR = r'D:\herb_ai_local\data_authenticity'
OUTPUT_DIR = r'D:\herb_ai_local\resnet50_output'
MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_model.pth')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 64  # 增大 batch size 提高吞吐量
NUM_WORKERS = 0  # Windows 兼容
INPUT_SIZE = 224
EPOCHS = 10
LR = 5e-5  # 更小学习率，更稳定
VAL_FREQ = 3  # 每 3 epoch 验证 data_90
# =============================================

class Tee:
    def __init__(self, filepath, mode='a'):
        self.file = open(filepath, mode, encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    def close(self):
        sys.stdout = self.stdout
        self.file.close()

class MergedDataset(Dataset):
    def __init__(self, base_dir, auth_dir, transform=None):
        self.transform = transform
        self.samples = []
        
        base_dataset = datasets.ImageFolder(os.path.join(base_dir, 'train'), loader=safe_loader)
        self.class_to_idx = base_dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        for path, label in base_dataset.samples:
            self.samples.append((path, label))
        
        auth_added = 0
        for herb_name in os.listdir(auth_dir):
            herb_path = os.path.join(auth_dir, herb_name)
            if not os.path.isdir(herb_path):
                continue
            authentic_dir = os.path.join(herb_path, 'train', 'authentic')
            if not os.path.exists(authentic_dir):
                continue
            if herb_name not in self.class_to_idx:
                continue
            label = self.class_to_idx[herb_name]
            for img_name in os.listdir(authentic_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
                    self.samples.append((os.path.join(authentic_dir, img_name), label))
                    auth_added += 1
        
        print(f'[INFO] Base: {len(base_dataset.samples)}, Added: {auth_added}, Total: {len(self.samples)}')
        added_counts = Counter([s[1] for s in self.samples[len(base_dataset.samples):]])
        for idx, count in added_counts.most_common():
            print(f'  + {self.idx_to_class[idx]:15s}: {count}')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = safe_loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label

def main():
    log_path = os.path.join(OUTPUT_DIR, 'phase3_output.log')
    tee = Tee(log_path, mode='w')
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'[INFO] GPU: {torch.cuda.get_device_name(0)}')
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.15),
        transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print('[INFO] Loading merged dataset...')
    train_dataset = MergedDataset(DATA_DIR, AUTHENTICITY_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_transform, loader=safe_loader)
    
    NUM_CLASSES = len(train_dataset.class_to_idx)
    print(f'[INFO] Classes: {NUM_CLASSES}, Val: {len(val_dataset)}')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True if DEVICE.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True if DEVICE.type == 'cuda' else False)
    
    print('[INFO] Building model...')
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, NUM_CLASSES))
    
    if os.path.exists(MODEL_PATH):
        print(f'[INFO] Loading Phase 2 model from {MODEL_PATH}')
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    else:
        print(f'[ERROR] Model not found: {MODEL_PATH}')
        return
    
    model = model.to(DEVICE)
    
    frozen = trainable = 0
    for name, param in model.named_parameters():
        if 'fc' in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    print(f'[INFO] Frozen: {frozen:,} ({frozen/(frozen+trainable)*100:.1f}%), Trainable: {trainable:,} ({trainable/(frozen+trainable)*100:.1f}%)')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    def train_epoch():
        model.train()
        total_loss = correct = total = 0
        num_batches = len(train_loader)
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
            if (batch_idx + 1) % 200 == 0 or batch_idx == 0:
                print(f'  [train] batch {batch_idx+1}/{num_batches} ({(batch_idx+1)/num_batches*100:.1f}%)', flush=True)
        return total_loss / total, correct / total
    
    def validate(loader):
        model.eval()
        total_loss = correct = total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                correct += outputs.argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        return total_loss / total, correct / total
    
    def validate_authenticity():
        model.eval()
        results = {}
        for herb_name in ['dangshen', 'danshen', 'renshen', 'suanzaoren']:
            val_auth_dir = os.path.join(AUTHENTICITY_DIR, herb_name, 'val', 'authentic')
            if not os.path.exists(val_auth_dir) or herb_name not in train_dataset.class_to_idx:
                continue
            true_label = train_dataset.class_to_idx[herb_name]
            auth_correct = auth_total = 0
            for img_name in os.listdir(val_auth_dir):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
                    continue
                img = safe_loader(os.path.join(val_auth_dir, img_name))
                img = val_transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    pred = model(img).argmax(1).item()
                auth_total += 1
                if pred == true_label:
                    auth_correct += 1
            results[herb_name] = {'acc': auth_correct / auth_total if auth_total else 0, 'correct': auth_correct, 'total': auth_total}
        return results
    
    print('\n========== Phase 3: Freeze backbone, fine-tune FC ==========')
    best_score = 0.0
    best_state = None
    
    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss, train_acc = train_epoch()
        
        # 每 epoch 快速验证 authenticity
        auth_results = validate_authenticity()
        avg_auth_acc = sum(v['acc'] for v in auth_results.values()) / len(auth_results) if auth_results else 0
        
        # 每 VAL_FREQ epoch 验证 data_90
        if (epoch + 1) % VAL_FREQ == 0 or epoch == EPOCHS - 1:
            val_loss, val_acc = validate(val_loader)
            val_str = f'Val Acc: {val_acc:.4f}'
        else:
            val_loss, val_acc = 0, 0
            val_str = 'Val Acc: (skipped)'
        
        scheduler.step()
        elapsed = time.time() - t0
        
        print(f'\nEpoch {epoch+1:2d}/{EPOCHS} | '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
              f'{val_str} | Auth Acc: {avg_auth_acc:.2%} | Time: {elapsed:.1f}s')
        for herb, res in auth_results.items():
            print(f'  {herb:12s}: {res["acc"]:.2%} ({res["correct"]}/{res["total"]})')
        
        # 保存标准：以 authenticity 识别率为主要指标
        score = avg_auth_acc
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f'  -> Saved best (auth_acc={avg_auth_acc:.4f})')
    
    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model_phase3.pth'))
    print(f'\n[INFO] Phase 3 done! Best auth accuracy: {best_score:.4f}')
    
    # 最终验证
    final_val_loss, final_val_acc = validate(val_loader)
    final_auth = validate_authenticity()
    print(f'\n========== Final ==========')
    print(f'data_90 val accuracy: {final_val_acc:.4f}')
    for herb, res in final_auth.items():
        print(f'{herb:12s} authentic accuracy: {res["acc"]:.2%} ({res["correct"]}/{res["total"]})')
    
    tee.close()

if __name__ == '__main__':
    main()
