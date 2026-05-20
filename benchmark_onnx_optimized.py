#!/usr/bin/env python3
"""
ONNX Runtime 优化对比测试
对比：默认配置 vs 图优化 + 多线程
"""

import os
import time
import numpy as np
import onnxruntime as ort

ONNX_PATH = r'D:\herb_ai_local\models\herb_resnet50_phase3.onnx'
WARMUP = 5
ITERATIONS = 20
BATCH_SIZE = 1
INPUT_SHAPE = (1, 3, 224, 224)

def create_session(opt_level, intra_threads, inter_threads):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = opt_level
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = inter_threads
    # 禁用默认的优化信息输出
    opts.log_severity_level = 3
    return ort.InferenceSession(ONNX_PATH, sess_options=opts, providers=['CPUExecutionProvider'])

def benchmark(sess, name):
    inp_name = sess.get_inputs()[0].name
    dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    
    # Warmup
    for _ in range(WARMUP):
        sess.run(None, {inp_name: dummy})
    
    # Benchmark
    times = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        sess.run(None, {inp_name: dummy})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
    
    times = np.array(times)
    print(f"  {name:20s}: mean={times.mean():6.1f}ms, std={times.std():5.1f}ms, min={times.min():6.1f}ms, max={times.max():6.1f}ms")
    return times.mean()

print("=" * 60)
print("ONNX Runtime 优化对比测试")
print(f"模型: {ONNX_PATH}")
print(f"输入: {INPUT_SHAPE}, 轮次: {ITERATIONS}, 预热: {WARMUP}")
print(f"CPU: {os.cpu_count()} 逻辑核心")
print("=" * 60)

# 1. 默认配置（无优化，单线程）
print("\n1. 默认配置（ORT_ENABLE_BASIC, 单线程）")
sess_default = create_session(ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, 1, 1)
t_default = benchmark(sess_default, "default")

# 2. 图优化 + 单线程
print("\n2. 图优化（ORT_ENABLE_ALL, 单线程）")
sess_opt1 = create_session(ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 1, 1)
t_opt1 = benchmark(sess_opt1, "opt_all_1t")

# 3. 图优化 + 2线程
print("\n3. 图优化 + 2线程")
sess_opt2 = create_session(ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 2, 2)
t_opt2 = benchmark(sess_opt2, "opt_all_2t")

# 4. 图优化 + 4线程
print("\n4. 图优化 + 4线程")
sess_opt4 = create_session(ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 4, 4)
t_opt4 = benchmark(sess_opt4, "opt_all_4t")

# 5. 图优化 + 8线程（超线程）
print("\n5. 图优化 + 8线程")
sess_opt8 = create_session(ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 8, 8)
t_opt8 = benchmark(sess_opt8, "opt_all_8t")

# 汇总
print("\n" + "=" * 60)
print("加速比汇总（以默认配置为基准）")
print("=" * 60)
for name, t in [("default", t_default), ("opt_all_1t", t_opt1), 
                ("opt_all_2t", t_opt2), ("opt_all_4t", t_opt4), ("opt_all_8t", t_opt8)]:
    speedup = t_default / t
    print(f"  {name:20s}: {t:6.1f}ms  ({speedup:.2f}x)")

print("\n最佳配置推荐：")
best = min([("opt_all_1t", t_opt1), ("opt_all_2t", t_opt2), ("opt_all_4t", t_opt4), ("opt_all_8t", t_opt8)], key=lambda x: x[1])
print(f"  -> {best[0]} ({best[1]:.1f}ms, {t_default/best[1]:.2f}x 加速)")
