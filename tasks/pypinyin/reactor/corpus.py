"""Deterministic gate corpus and timing workload for the pypinyin task."""
from __future__ import annotations

GATE_TEXTS = [
    "",
    "hello world 123",
    "今天天气很好。",
    "重庆市渝北区银行行长在长城饭店开会",
    "他背着背包去首都北京看降落的飞机",
    "中文ABC混合 test 123 标点，。！？",
    "一行白鹭上青天，窗含西岭千秋雪。",
    "利率下调了0.25个百分点，涨幅回落。",
    "单休论罪",
    "龘靐齉爩 灪麤彠",
    "а б в α β γ 混合",
    "🇨🇳 表情符号 test",
]

#: (style_name, attribute) — "default" means pypinyin.pinyin(text)
GATE_STYLES = [
    "default",
    "NORMAL",
    "INITIALS",
    "FINALS",
    "FIRST_LETTER",
    "TONE2",
    "TONE3",
]

_BASE = (
    "今天天气很好我们一起去公园散步吧这本书非常有意思他喜欢在周末的时候写代码调试程序"
    "北京市海淀区中关村大街一号楼的会议室里正在开会讨论明年的计划安排和预算分配方案"
    "人工智能技术的发展日新月异大语言模型在自然语言处理领域取得了令人瞩目的成就"
    "重庆市渝北区银行行长在长城饭店开会讨论利率下调对中小企业融资成本的影响"
)

def workload_texts() -> list[str]:
    """~50k characters: pinyin() (default style) then lazy_pinyin()."""
    return [_BASE * 280, _BASE * 200]
