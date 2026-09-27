"""用户画像：从问诊历史推断体质倾向、慢性症状与证型史，并给出生活习惯建议。

画像从数据推导（不单独存储）：扫描该用户全部档案的证型结论与可信症状，
按体质映射表加权统计。外感急性证（风寒/风热等）不计入体质统计。
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from . import reasoning
from .models import UserProfile

#: 证型名 -> 体质 映射（顺序敏感：特异在前，泛化在后）
_CONSTITUTION_RULES: list[tuple[str, str]] = [
    ("特禀", "特禀质"),
    ("阴虚", "阴虚质"),
    ("阳虚", "阳虚质"),
    ("虚寒", "阳虚质"),
    ("心脾两虚", "气虚质"),
    ("气血两虚", "气虚质"),
    ("气虚", "气虚质"),
    ("脾虚", "气虚质"),
    ("血虚", "气虚质"),
    ("中气下陷", "气虚质"),
    ("肝气郁", "气郁质"),
    ("气郁", "气郁质"),
    ("肝郁", "气郁质"),
    ("痰湿", "痰湿质"),
    ("湿热", "湿热质"),
    ("血瘀", "血瘀质"),
    ("瘀", "血瘀质"),
]

#: 外感/急性证标记（不计入体质，但保留在证型史）
_ACUTE_MARKERS = ("风寒", "风热", "表寒", "表热", "太阳", "卫分", "气分",
                  "暑湿", "外感", "感冒", "食积", "惊风")

#: 各体质生活习惯建议（饮食/起居/运动/情志/茶饮）
LIFESTYLE: dict[str, list[str]] = {
    "阴虚质": [
        "饮食宜滋阴润燥：银耳、百合、山药、梨、桑葚、鸭肉、枸杞；忌辛辣、油炸、烧烤、羊肉等温燥之品",
        "起居忌熬夜（23点前入睡），熬夜最伤阴血；中午可小憩15-30分钟养心阴",
        "运动宜柔和：八段锦、瑜伽、散步、游泳，避免大汗淋漓的高强度运动（汗为津液所化）",
        "情志宜静养：五心烦热易伴心烦急躁，可练习腹式呼吸、正念冥想",
        "茶饮参考：麦冬石斛茶、枸杞菊花茶、银耳羹；烟酒咖啡宜减",
    ],
    "阳虚质": [
        "饮食宜温补：羊肉、鸡肉、韭菜、生姜、桂圆、核桃；少食生冷瓜果、冷饮、绿茶",
        "起居注意保暖，尤其腰腹与下肢；可常晒太阳（背晒20分钟）；避免熬夜",
        "运动宜动养阳气：快走、慢跑、太极，以微微出汗为度；冬季不宜晨起冷水锻炼",
        "茶饮参考：生姜红枣茶、桂圆枸杞茶；可艾灸关元、足三里（须医师指导）",
    ],
    "气虚质": [
        "饮食宜健脾益气：山药、莲子、小米、南瓜、黄芪炖鸡、大枣；少食耗气之品与过度生冷",
        "起居宜规律，避免过劳与久言伤气；保证充足睡眠，不宜大汗后受风",
        "运动宜循序渐进：散步、八段锦（重点练'两手托天理三焦'），避免竞技性剧烈运动",
        "情志上避免思虑过度；过度思虑伤脾，劳逸结合",
        "茶饮参考：黄芪红枣茶、山药小米粥；感冒期间暂停进补",
    ],
    "气郁质": [
        "饮食宜行气解郁：陈皮、玫瑰花、佛手、萝卜、柑橘；少食收涩酸涩之品",
        "起居宜增加户外活动与社交，避免独处久坐；培养兴趣爱好疏解情志",
        "运动是疏肝良药：快走、慢跑、球类、登山等有一定强度的运动尤宜",
        "情志调摄最关键：倾诉、书写、正念；睡前不做易引发情绪波动的争执",
        "茶饮参考：玫瑰陈皮茶、佛手茶；经前乳胀者可用逍遥类方剂调理（须医师）",
    ],
    "痰湿质": [
        "饮食宜健脾化湿：薏苡仁、赤小豆、冬瓜、白萝卜、陈皮；忌肥甘厚味、甜食、酒、夜宵",
        "起居避免潮湿环境，雨季注意祛湿；晚餐宜早宜少，控制体重",
        "运动量宜偏大：快走、慢跑、游泳，坚持每日30分钟以上以动化湿",
        "茶饮参考：陈皮荷叶茶、赤小豆薏米水（孕期不宜大量薏米）",
    ],
    "湿热质": [
        "饮食宜清淡清热：绿豆、苦瓜、冬瓜、芹菜、莲藕；忌辛辣、酒、烧烤、油炸、芒果荔枝等热性水果",
        "起居避免熬夜与湿热环境（桑拿过频）；衣物透气，保持皮肤清洁",
        "运动宜中高强度且及时补水：跑步、球类；运动后勿立即冷水浴",
        "茶饮参考：荷叶茶、淡竹叶茶、绿豆汤；烟酒务必节制",
    ],
    "血瘀质": [
        "饮食宜活血行气：黑木耳、山楂、洋葱、玫瑰花、少量醋；少食油腻寒凉",
        "起居避免久坐久站，定时活动促进血行；注意保暖，寒凝血瘀者尤忌受凉",
        "运动宜促进气血运行：健身操、慢跑、太极；有出血倾向者运动强度宜温和",
        "情志：气行则血行，保持心情舒畅，避免长期郁怒",
        "茶饮参考：山楂玫瑰茶；长期服用抗凝药物者使用活血食材前须咨询医师",
    ],
    "特禀质": [
        "首要原则：查找并回避过敏原（尘螨、花粉、特定食物），居家勤除螨通风",
        "饮食宜清淡均衡，慎食已知过敏食物及'发物'（虾蟹、酒等）；记录饮食-症状日记",
        "起居：花粉季减少外出或戴口罩，新装修环境充分通风后再入住",
        "症状反复或出现胸闷、喉头紧缩感应及时就医，不可自行长期服抗过敏药",
    ],
    "平和质": [
        "饮食有节、起居有常、不妄作劳——继续保持规律作息与均衡饮食即可",
        "四季顺时养生：春养肝、夏养心、长夏健脾、秋润肺、冬补肾",
        "坚持适度运动与情志调摄，即为最好的保健",
    ],
}


def syndrome_to_constitution(syndrome_name: str) -> Optional[str]:
    """证型名 -> 体质；急性证或无法映射返回None。"""
    name = syndrome_name or ""
    if any(m in name for m in _ACUTE_MARKERS):
        return None
    for keyword, constitution in _CONSTITUTION_RULES:
        if keyword in name:
            return constitution
    return None


def lifestyle_advice(constitution: str) -> list[str]:
    """按体质返回生活习惯建议；未知体质返回空表。"""
    if not constitution:
        return []
    return list(LIFESTYLE.get(constitution, []))


def compute_profile(username: str, records: list[dict[str, Any]]) -> UserProfile:
    """从该用户全部问诊档案计算画像。"""
    constitution_counter: Counter[str] = Counter()
    symptom_counter: Counter[str] = Counter()
    symptom_sessions: dict[str, set[str]] = {}
    syndrome_history: list[dict[str, Any]] = []
    first_seen: Optional[float] = None
    last_active: Optional[float] = None

    for rec in records:
        sid = rec.get("id", "")
        updated = rec.get("updated_at") or 0
        created = rec.get("created_at") or updated
        if first_seen is None or created < first_seen:
            first_seen = created
        if last_active is None or updated > last_active:
            last_active = updated

        # 症状频次（规范化后统计，跨会话出现=慢性倾向）
        evidence = (rec.get("evidence") or {}).get("symptoms") or []
        for s in evidence:
            for canon in reasoning.normalize_symptoms([s]):
                symptom_counter[canon] += 1
                symptom_sessions.setdefault(canon, set()).add(sid)

        # 证型统计与历史（主证权重2、候选1）
        finals = rec.get("finals") or []
        if finals:
            last = finals[-1].get("result") or {}
            names = [last.get("primary_syndrome", "")] + \
                    [c.get("name", "") for c in (last.get("syndromes") or [])[1:3]]
            for weight, name in zip((2, 1, 1), names):
                constitution = syndrome_to_constitution(name)
                if constitution:
                    constitution_counter[constitution] += weight
            syndrome_history.append({
                "date": finals[-1].get("time"),
                "syndrome": last.get("primary_syndrome", ""),
                "treatment": last.get("treatment_principle", ""),
            })

    total = sum(constitution_counter.values())
    scores = {k: round(v / total, 3) for k, v in constitution_counter.most_common()} \
        if total else {}
    dominant = next(iter(scores), "")
    # 阈值：占比≥0.4才认定主导体质倾向，否则提示证据不足
    if dominant and scores[dominant] < 0.4:
        dominant = dominant  # 保留倾向但画像说明中标注证据分散

    chronic = [s for s, ses in symptom_sessions.items() if len(ses) >= 2]
    top_symptoms = [s for s, _ in symptom_counter.most_common(8)]

    profile = UserProfile(
        username=username,
        sessions=len(records),
        first_seen=first_seen or 0,
        last_active=last_active or 0,
        dominant_constitution=dominant,
        constitution_scores=scores,
        constitution_confident=bool(dominant and scores[dominant] >= 0.4),
        top_symptoms=top_symptoms,
        chronic_symptoms=chronic[:8],
        syndrome_history=syndrome_history,
    )
    profile.lifestyle_advice = lifestyle_advice(dominant)
    return profile
