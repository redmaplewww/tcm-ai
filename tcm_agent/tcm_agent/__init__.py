"""tcm_agent —— 中医AI问诊与查询Agent包。

组成：
- config    路径与模型配置
- models    pydantic 输出模型（问诊结果、证型候选、红旗警示等）
- data      结构化数据加载（证型/中药/方剂/穴位/疾病/红旗/术语）
- kb        分级Markdown知识库加载与检索（RAG-lite）
- reasoning 确定性中医推理引擎（症状规范化、红旗检测、辨证打分、追问生成）
- tools     供LLM调用的知识查询与推理工具
- agents    pydantic-ai Agent 定义（查询Agent / 问诊Agent）
- cli       命令行入口（query / consult / demo / info）
"""

__version__ = "0.1.0"
