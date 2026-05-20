#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
种子类药材 Top-2 仲裁模块
用于解决麦芽(maiya)、谷芽(guya)、菟丝子(tusizi)、莱菔子(laifuzi)、薏苡仁(yiyiren)
之间的细粒度混淆问题。

使用方式:
    from seed_arbitration import SeedArbitrator
    arb = SeedArbitrator()
    result = arb.arbitrate(results_list)  # results_list 是 Top-5 列表
"""

from typing import List, Dict, Any


class SeedArbitrator:
    """
    种子类混淆仲裁器
    
    设计依据（来自 diagnose_maiya_guya/all_samples.csv 统计）：
    - 麦芽错分时，81.2% 的真实标签在 Top-2
    - 谷芽错分为麦芽时，100% 的真实标签在 Top-2
    - 谷芽错分为菟丝子时，77.8% 的真实标签在 Top-2
    - 谷芽错分为薏苡仁时，100% 的真实标签在 Top-2
    - 谷芽错分为莱菔子时，50% 的真实标签在 Top-2
    """

    SEED_CLASSES = {'maiya', 'guya', 'tusizi', 'laifuzi', 'yiyiren'}

    def __init__(self, 
                 maiya_guya_gap_threshold: float = 0.20,
                 guya_other_conf_threshold: float = 0.85,
                 seed_class_conf_threshold: float = 0.80):
        self.maiya_guya_gap = maiya_guya_gap_threshold
        self.guya_other_conf = guya_other_conf_threshold
        self.seed_conf = seed_class_conf_threshold

    def arbitrate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(results) < 2:
            return self._no_arbitrate(results[0]['name'] if results else '')

        top1 = results[0]
        top2 = results[1]
        t1_name, t1_conf = top1['name'], top1['confidence']
        t2_name, t2_conf = top2['name'], top2['confidence']
        gap = t1_conf - t2_conf

        # 策略1: 麦芽-谷芽互混仲裁
        if {t1_name, t2_name} == {'maiya', 'guya'}:
            if gap < self.maiya_guya_gap:
                return {
                    'arbitrated': True,
                    'original_top1': t1_name,
                    'final_top1': t1_name,
                    'message': f'疑似{self._cn_name(t1_name)}（与{self._cn_name(t2_name)}相似度{t2_conf:.1%}），建议人工复核',
                    'alert': True,
                    'details': {'strategy': 'maiya_guya_ambiguous', 'gap': gap}
                }

        # 策略2: 谷芽被错分为其他种子类（Top-2 是谷芽时挽救）
        if t2_name == 'guya' and t1_name in ('tusizi', 'yiyiren', 'laifuzi'):
            if t1_conf < self.guya_other_conf:
                return {
                    'arbitrated': True,
                    'original_top1': t1_name,
                    'final_top1': 'guya',
                    'message': f'疑似{self._cn_name(t1_name)}（可能为{self._cn_name("guya")}，相似度{t2_conf:.1%}），建议人工复核',
                    'alert': True,
                    'details': {'strategy': 'guya_rescue', 'original': t1_name}
                }

        return self._no_arbitrate(t1_name)

    def _no_arbitrate(self, top1_name: str) -> Dict[str, Any]:
        return {
            'arbitrated': False,
            'original_top1': top1_name,
            'final_top1': top1_name,
            'message': None,
            'alert': False,
            'details': {}
        }

    @staticmethod
    def _cn_name(en_name: str) -> str:
        mapping = {
            'maiya': '麦芽', 'guya': '谷芽', 'tusizi': '菟丝子',
            'laifuzi': '莱菔子', 'yiyiren': '薏苡仁',
        }
        return mapping.get(en_name, en_name)


# ========== 可直接插入 app.py 的简化版函数 ==========

def apply_seed_arbitration(results: list) -> dict:
    """
    极简仲裁函数，用于直接嵌入部署代码
    results: [{'name': str, 'confidence': float}, ...]
    返回: {'name': str, 'confidence': float, 'arbitrated': bool, 'message': str|None}
    """
    if len(results) < 2:
        return {'name': results[0]['name'], 'confidence': results[0]['confidence'],
                'arbitrated': False, 'message': None}
    
    r1, r2 = results[0], results[1]
    n1, c1 = r1['name'], r1['confidence']
    n2, c2 = r2['name'], r2['confidence']
    gap = c1 - c2
    
    CN = {'maiya': '麦芽', 'guya': '谷芽', 'tusizi': '菟丝子', 
          'laifuzi': '莱菔子', 'yiyiren': '薏苡仁'}
    
    if {n1, n2} == {'maiya', 'guya'} and gap < 0.20:
        return {'name': n1, 'confidence': c1, 'arbitrated': True,
                'message': f'疑似{CN.get(n1,n1)}（与{CN.get(n2,n2)}相似{c2:.1%}），建议人工复核'}
    
    if n2 == 'guya' and n1 in ('tusizi', 'yiyiren', 'laifuzi') and c1 < 0.85:
        return {'name': n1, 'confidence': c1, 'arbitrated': True,
                'message': f'疑似{CN.get(n1,n1)}（可能为谷芽，相似{c2:.1%}），建议人工复核'}
    
    if n1 in ('maiya', 'guya', 'tusizi', 'laifuzi', 'yiyiren') and c1 < 0.80 and n2 in ('maiya', 'guya', 'tusizi', 'laifuzi', 'yiyiren'):
        return {'name': n1, 'confidence': c1, 'arbitrated': True,
                'message': f'疑似{CN.get(n1,n1)}（与{CN.get(n2,n2)}相似{c2:.1%}），建议人工复核'}
    
    return {'name': n1, 'confidence': c1, 'arbitrated': False, 'message': None}
