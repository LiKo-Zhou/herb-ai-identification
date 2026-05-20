# 中药饮片智能鉴定系统 - 进度报告

## 当前部署状态（板端 RK3568）

**服务地址**: `http://192.168.1.9:5000` / `http://127.0.0.1:5000`

**当前模型**: ResNet50 ONNX + v3.0 分层防御（Energy + PatchCore）
- **输入尺寸**: 224x224
- **推理速度**: ~2.5s/张（ONNX Runtime CPU）
- **部署时间**: 2026-04-29

**硬件配置**:
- 主板: Firefly ROC-RK3568-PC, Debian 10, kernel 4.19.232
- 内存: 4GB（PatchCore 记忆库 230MB 完全可承载）
- 摄像头: OV8858 @ I2C-4 0x36, 4-lane MIPI CSI
- 显示屏: 7-inch USB 触摸屏 (WaveShare WS170120, VID:PID 0eef:0005) + HDMI
- 自动启动: systemd `herb-web.service` + Chromium kiosk mode

---

## 摄像头子系统（重大变更 v2.1）

### 架构变更: Browser getUserMedia → Python Backend Capture

**问题根源**: Chromium 91 的 WebRTC 不支持 RKISP 的 **multiplanar V4L2** (`V4L2_CAP_VIDEO_CAPTURE_MPLANE`)。getUserMedia 在 kiosk 模式下静默失败。

**解决方案**:
1. 放弃浏览器 `getUserMedia()`，改用 **Python 后端直接捕获**
2. 前端 camera 页面改为 **AJAX 轮询** (`/capture` 每 250ms 刷新预览图)
3. 拍照按钮调用 `/camera_snap` 获取 1920x1080 高质量 JPEG

**技术实现**:
- 后端: `v4l2-ctl --stream-mmap --stream-to=...` 捕获 UYVY → numpy YUV→RGB 转换 → PIL JPEG 编码
- 并发保护: `threading.Lock()` 确保串行访问 `/dev/video0`
- 预览: 640x480 @ ~4fps (足够拍照预览)
- 拍照: 1920x1080 @ quality=90

**ISP 参数**（systemd ExecStartPre 自动设置）:
```bash
v4l2-ctl -d /dev/video0 --set-ctrl analogue_gain=800 --set-ctrl exposure=1200
```

**验证状态**:
- [x] `v4l2-ctl --stream-to` 捕获 1920x1080 NV12 成功
- [x] UYVY→JPEG 转换正确 (test pattern 彩条验证)
- [x] `/capture` endpoint 返回 640x480 JPEG (200 OK)
- [x] `/camera_snap` endpoint 返回 1920x1080 base64 JPEG
- [x] Kiosk 模式下页面加载、轮询、拍照、鉴定全流程测试通过

---

## Chromium Kiosk 启动参数

```
/usr/bin/chromium --kiosk --app=http://127.0.0.1:5000 \
  --no-sandbox --disable-infobars --disable-session-crashed-bubble \
  --disable-features=TranslateUI --window-position=0,0 \
  --window-size=1024,600 --disable-gpu --start-fullscreen \
  --use-fake-ui-for-media-stream --autoplay-policy=no-user-gesture-required
```

---

## 服务管理命令

```bash
# 查看服务状态
systemctl status herb-web.service

# 重启服务
systemctl restart herb-web.service

# 查看日志
journalctl -u herb-web.service -f

# 手动测试摄像头捕获
curl -o /tmp/test.jpg http://127.0.0.1:5000/capture

# 手动测试拍照
curl -s http://127.0.0.1:5000/camera_snap | python3 -m json.tool

# 查看摄像头控制参数
v4l2-ctl -d /dev/video0 --list-ctrls
```

---

## 模型演进路线

| 版本 | 架构 | 品种鉴定准确率 | 真伪鉴别能力 | 状态 |
|------|------|:------------:|:------------:|------|
| v1.0 | 级联推理（二分类+度量学习） | - | 依赖主模型先识别 | 已废弃 |
| v2.0 | Energy Score 全局异常检测 | 97.30% | 单阈值，跨域失效 | 已废弃 |
| v2.5 | 品类自适应 Energy + PatchCore | 97.30% | 分层防御，但分布偏移导致失效 | 测试中 |
| **v3.1** | **Phase 3 数据对齐 + Freeze Backbone 微调** | **目标 ≥97%** | **目标 fake 检出 >80%, 误报 <20%** | **进行中** |

---

## Phase 3 计划（2026-05-19）

### 问题
v3.0 分层防御在 `data_authenticity` 验证集上失效：
- authentic 误报率 62-80%
- fake 检出率 0-60%
- 根因：训练集与验证集分布偏移，主模型 Top-1 识别率暴跌

### 解决方案
**数据对齐 + Freeze Backbone 轻量微调**

1. [ ] 合并 `data_authenticity` authentic 样本到训练集
2. [ ] 加载最佳模型，freeze backbone，训 FC 层（5-10 epochs）
3. [ ] 验证 4 品类 Top-1 识别率
4. [ ] 导出 ONNX
5. [ ] 重新校准 Energy Score
6. [ ] 重建 PatchCore 记忆库
7. [ ] v3.1 融合推理验证

### 理论支撑
- SHOT (ICML 2020): Source-Free Domain Adaptation
- Tent (ICLR 2021): Test-Time Adaptation
- Cao et al. (ICCV 2023): Anomaly Detection Under Distribution Shift

---

## 模型文件路径

板端: `/opt/herb_ai/models/`
- `herb_resnet50.onnx` - ResNet50 ONNX (90.6MB)
- `classes_129.txt` - 129 类药材名称
- `herb_dictionary.json` - 电子药典数据 (~60% 完整)
- `energy_thresholds.json` - Energy Score 自适应阈值
- `patchcore_memory.json` + `.npz` - PatchCore 记忆库

---

## 已知问题与待办

1. **图像亮度**: 当前模拟增益 800 + 曝光 1200。在非常暗的环境下图像仍偏暗，建议在明亮环境或补光条件下使用。
2. **预览帧率**: ~4fps，足够拍照预览，但非真正实时视频流。
3. **药典数据**: 仍有 ~40% 药材显示"待补充"，需持续完善。
4. **SSH 登录**: 当前只能通过 ADB 连接，SSH 待修复。
5. **v3.0 误报率过高**: Phase 3 数据对齐进行中。

---

**最后更新**: 2026-05-19
