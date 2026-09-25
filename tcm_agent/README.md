# tcm_agent —— 中医AI问诊与查询Agent

基于 **pydantic-ai** 框架与**分级Markdown知识库**（`../tcm_kb/`）构建的中医智能问诊与查询系统。

## 架构

```
用户输入（口语症状 / 知识问题）
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ pydantic-ai Agent 层（对话编排与解释）                │
│  ├─ query_agent  → QueryAnswer（知识问答）           │
│  └─ consult_agent→ ConsultationResult（病情推理）    │
│  LLM可选：TCM_MODEL 环境变量；测试用TestModel离线    │
└───────────────┬─────────────────────────────────────┘
                │ 调用工具（tools.py）
                ▼
┌─────────────────────────────────────────────────────┐
│ 确定性推理引擎（reasoning.py）★正确性基线            │
│  症状规范化（同义词/口语映射）                        │
│  → 红旗急症检测（red_flags.json规则）                │
│  → 证型辨证打分（syndromes.json，key×3+次症+舌脉）   │
│  → 方剂/中成药/调护关联（formulas/diseases.json）    │
│  → 十问歌追问生成 + 特殊人群提示                     │
└───────────────┬─────────────────────────────────────┘
                │ 检索接地（防幻觉）
                ▼
┌─────────────────────────────────────────────────────┐
│ 分级知识库（tcm_kb/）                                │
│  MD层：7大域约100个主题文件（带frontmatter）          │
│  JSON数据层：证型/中药/方剂/穴位/疾病/红旗/术语       │
│  检索：字符二元组加权打分（标题/标签/章节加权）        │
└─────────────────────────────────────────────────────┘
```

**设计原则**：推理结论以确定性引擎为准（不依赖LLM、可完全测试）；LLM只做
对话编排与解释性输出，且必须基于工具返回（知识库接地）——防止编造方剂配伍。
安全规则（红旗、配伍禁忌）为硬编码规则而非概率模型。

## 安装

```bash
cd tcm_agent
python -m venv ../.venv
../.venv/Scripts/python -m pip install -e ".[dev]"   # 或 pip install pydantic-ai pytest
```

## 使用

```bash
# 知识查询（离线知识库检索，无需任何凭据）
python -m tcm_agent query "什么是阴虚"
python -m tcm_agent query "感冒风寒风热怎么区别" -k 3

# 真实LLM问答（GLM，自动复用本机ZCode CLI登录凭据）
python -m tcm_agent ask "什么是阴虚？有什么表现？"
python -m tcm_agent ask "麻黄的功效和禁忌"

# 病情推理（离线确定性引擎）
python -m tcm_agent consult --symptoms "怕冷,发烧,无汗,头痛,清鼻涕" --tongue "舌淡红苔薄白" --pulse "浮紧"

# 真实LLM辨证（附离线引擎对照；第一阶段纯文字交互）
python -m tcm_agent consult --llm --symptoms "心慌,睡不着,多梦,健忘,疲乏,吃不下饭" --tongue "舌淡" --sex 女 --age 35
python -m tcm_agent consult --llm --symptoms "胸口像压了石头,胸痛放射到左臂,出冷汗"   # 急症红旗演示

# 交互式多轮问诊（LLM对话收集症状，/诊断 立即出结果；无凭据时自动离线问卷）
python -m tcm_agent chat
python -m tcm_agent chat --sex 女 --age 35

# 内置演示与统计
python -m tcm_agent demo
python -m tcm_agent info
```

### LLM凭据解析（`tcm_agent/llm.py`）

优先级：①环境变量 `TCM_LLM_KEY` / `TCM_LLM_BASE_URL` / `TCM_LLM_MODEL`；
②本机ZCode CLI登录凭据（`~/.zcode/v2/config.json` 的 bigmodel-coding-plan）。
默认接入 `GLM-5.3-Flash`（open.bigmodel.cn 的Anthropic兼容端点）。
凭据不可用时自动退回离线模式，全部离线功能不受影响。

## 测试

```bash
../.venv/Scripts/python -m pytest -v        # 离线套件：76项（数据/检索/推理/配伍/证据取证/会话状态机/接线）
../.venv/Scripts/python -m pytest -m llm -v # 真实LLM集成：11项（问答/辨证/多轮问诊/动态复辨/自诊质询，约10分钟）
```

覆盖：数据完整性 → 知识库检索准确性 → 确定性推理（经典辨证/红旗/追问/特殊人群）
→ 十八反十九畏配伍安全 → **证据可信度引擎**（分级/矛盾检测/口语识别/数值核验/维度解决）
→ **多轮会话状态机**（动态复辨/证型变化/矛盾质询/澄清替换/急症短路）
→ pydantic-ai Agent接线 → 真实GLM端到端（含"补充症状后证型从心血虚动态更新为心脾两虚"
与"自诊术语被质询且不污染辨证"两项行为验证）。

## 多轮问诊架构（`tcm_agent/chat_session.py` + `tcm_agent/evidence.py`）

```
用户输入 → ①确定性红旗检测（安全门禁，先于LLM）
         → ②LLM分类提取：症状原词 / 自诊术语 / 病史·体质 / 舌象 / 脉象(含来源) / 疑点
         → ③证据可信度引擎（不默认用户输入全真）：
             · 症状自述 → high（参与辨证）
             · 自诊术语（"我肾虚"）/病史 → medium（不参与推理，触发质询核实）
             · 舌象自述 → medium（参考参与）；脉象自述 → low（不采用，医师转述除外）
             · 同维度矛盾（无汗vs自汗、便秘vs泄泻、舌红vs舌淡、低烧vs实测39℃）
               → 双方disputed排除 + 生成质询；澄清后按最新陈述替换解决
         → ④动态复辨：每轮基于当前可信证据重新辨证
             · 证型变化时报告"心血虚证 → 心脾两虚证"
             · 存疑信息影响即时可见（置信度随证据 disputed/澄清而变化）
         → ⑤ready或/诊断 → 输出最终辨证（仅基于可信证据，附存疑说明）
```
- llm_fn可注入：生产=pydantic-ai chat agent（GLM），测试=脚本函数（全离线可测）
- LLM只负责提取与对话，可信度判定/矛盾检测/辨证结论全部由确定性引擎裁决

## 安全边界

- 红旗症状（心梗胸痛、中风、大出血等）命中即终止辨证，输出就医指引；
- 永不输出最终处方，方剂仅作"知识参考"并强制提示执业中医师；
- 孕妇/小儿/高龄自动附加特殊人群提示；
- 查询未命中知识库时明确说明"未收录"，禁止编造（系统提示词+工具降级双保险）。

## 目录

```
tcm_agent/
├── pyproject.toml
├── tcm_agent/
│   ├── config.py      # 路径与模型配置
│   ├── models.py      # pydantic输出模型
│   ├── data.py        # JSON数据层加载
│   ├── kb.py          # 分级MD知识库检索
│   ├── reasoning.py   # 确定性辨证推理引擎
│   ├── tools.py       # Agent工具
│   ├── agents.py      # pydantic-ai Agent定义
│   ├── cli.py         # 命令行入口
│   └── __main__.py
└── tests/             # pytest测试套件
```
