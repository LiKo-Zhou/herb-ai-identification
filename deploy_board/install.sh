#!/bin/bash
# 中药饮片AI鉴定系统 v3.1 部署脚本 (RK3568)
set -e

DEPLOY_DIR="/opt/herb_ai"
MODEL_DIR="/opt/herb_ai/models"
BACKUP_DIR="/opt/herb_ai/backup_$(date +%Y%m%d_%H%M%S)"

echo "========================================"
echo "  Herb AI v3.1 Deployment Script"
echo "========================================"

# 1. 备份旧版本
echo "[1/6] 备份旧版本..."
mkdir -p "$BACKUP_DIR"
if [ -f "$DEPLOY_DIR/app.py" ]; then cp "$DEPLOY_DIR/app.py" "$BACKUP_DIR/"; fi
if [ -f "$DEPLOY_DIR/app_optimized.py" ]; then cp "$DEPLOY_DIR/app_optimized.py" "$BACKUP_DIR/"; fi
if [ -f "$MODEL_DIR/herb_resnet50.onnx" ]; then cp "$MODEL_DIR/herb_resnet50.onnx" "$BACKUP_DIR/"; fi
if [ -f "$MODEL_DIR/herb_dictionary.json" ]; then cp "$MODEL_DIR/herb_dictionary.json" "$BACKUP_DIR/"; fi
echo "  Backup -> $BACKUP_DIR"

# 2. 复制新文件
echo "[2/6] 复制新文件..."
cp herb_resnet50_phase3.onnx "$MODEL_DIR/"
cp herb_dictionary.json "$MODEL_DIR/"
cp energy_thresholds_phase3.json "$MODEL_DIR/"
cp classes_129.txt "$MODEL_DIR/"
cp app_optimized.py "$DEPLOY_DIR/"
cp herb-web-optimized.service /etc/systemd/system/
echo "  Done"

# 3. 检查 Python 环境
echo "[3/6] 检查依赖..."
if [ ! -d "$DEPLOY_DIR/venv" ]; then
    echo "  创建虚拟环境..."
    python3 -m venv "$DEPLOY_DIR/venv"
fi
source "$DEPLOY_DIR/venv/bin/activate"
pip install -q --upgrade pip
pip install -q flask onnxruntime numpy pillow
echo "  Done"

# 4. 停止旧服务
echo "[4/6] 停止旧服务..."
systemctl stop herb-web.service 2>/dev/null || true
systemctl stop herb-web-optimized.service 2>/dev/null || true

# 5. 启用新服务
echo "[5/6] 启用新服务..."
systemctl daemon-reload
systemctl enable herb-web-optimized.service
systemctl start herb-web-optimized.service
sleep 3
echo "  Service status:"
systemctl status herb-web-optimized.service --no-pager || true

# 6. 健康检查
echo "[6/6] 健康检查..."
for i in 1 2 3; do
    if curl -s http://127.0.0.1:5000/health | grep -q '"status": "ok"'; then
        echo "  ✅ Health check passed"
        echo ""
        echo "========================================"
        echo "  部署成功！"
        echo "  服务: http://127.0.0.1:5000"
        echo "  日志: journalctl -u herb-web-optimized -f"
        echo "  回滚: cp $BACKUP_DIR/* $DEPLOY_DIR/"
        echo "========================================"
        exit 0
    fi
    sleep 2
done

echo "  ❌ Health check failed. Check logs:"
echo "     journalctl -u herb-web-optimized -n 50"
exit 1
