# 板端部署补丁：Top-2 种子类混淆仲裁

## 适用版本
`app_optimized.py`（RK3568 板端部署版本）

## 插入位置
在 `/infer` 接口中，获取 `results`（Top-5 列表）后、`alert` 判定之前。

## 代码片段

```python
        # ===== Top-2 种子类混淆仲裁（v2026-05-20）=====
        SEED_CLASSES = {'maiya', 'guya', 'tusizi', 'laifuzi', 'yiyiren'}
        CN = {'maiya': '麦芽', 'guya': '谷芽', 'tusizi': '菟丝子',
              'laifuzi': '莱菔子', 'yiyiren': '薏苡仁'}
        arb_message = None
        
        if len(results) >= 2:
            r1, r2 = results[0], results[1]
            n1, c1 = r1['name'], r1['confidence']
            n2, c2 = r2['name'], r2['confidence']
            gap = c1 - c2
            
            # 策略1: 麦芽-谷芽互混（核心策略）
            if {n1, n2} == {'maiya', 'guya'} and gap < 0.20:
                arb_message = '疑似{}（与{}相似度{:.1%}），建议人工复核'.format(
                    CN.get(n1, n1), CN.get(n2, n2), c2)
            
            # 策略2: 谷芽被错分为其他种子类，Top-2是谷芽时挽救
            elif n2 == 'guya' and n1 in ('tusizi', 'yiyiren', 'laifuzi') and c1 < 0.85:
                arb_message = '疑似{}（可能为谷芽，相似度{:.1%}），建议人工复核'.format(
                    CN.get(n1, n1), c2)
        
        # 更新 alert 逻辑：仲裁触发时也提升警告
        top1_conf = results[0]['confidence']
        top1_name = results[0]['name']
        alert = top1_conf < ALERT_THRESHOLD or arb_message is not None
        pharma = get_herb_info(top1_name)
        
        return jsonify({
            'success': True, 'results': results,
            'top1_confidence': top1_conf, 'top1_name': top1_name,
            'alert': alert,
            'alert_message': arb_message if arb_message else ('疑似伪品，建议人工复核' if alert else None),
            'pharmacopoeia': pharma
        })
```

## 效果验证（基于PC端历史数据）

| 品类 | 原始Top-1 | 等效准确率（仲裁后）| 挽救率 | 误报率 |
|:---|:---|:---|:---|:---|
| 麦芽 | 89.7% | 93.5% | 37.5% | 5.8% |
| 谷芽 | 77.6% | 87.3% | 43.3% | 0.0% |

## 部署步骤
1. SSH 登录 RK3568：`ssh root@<板端IP>`
2. 备份原文件：`cp /opt/herb_ai/app_optimized.py /opt/herb_ai/app_optimized.py.bak`
3. 用 `vim` 或 `scp` 修改 `/opt/herb_ai/app_optimized.py`，在 `/infer` 接口中插入上述代码
4. 重启服务：`systemctl restart herb_ai` 或手动 `kill` + 重启 Python 进程
5. 验证：上传麦芽/谷芽测试图，检查返回中是否包含 `alert_message`

## 回滚
如出现问题，直接恢复备份：
```bash
cp /opt/herb_ai/app_optimized.py.bak /opt/herb_ai/app_optimized.py
systemctl restart herb_ai
```
