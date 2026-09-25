"""分级Markdown知识库结构与检索测试。"""

from pathlib import Path

from tcm_agent.config import KB_ROOT
from tcm_agent.kb import get_kb

DOMAIN_DIRS = [
    "01_基础理论", "02_诊断学", "03_中药", "04_方剂",
    "05_针灸外治", "06_临床各科", "07_科学证据",
]


def test_kb_tree_structure():
    """分级目录齐全，每域有_index.md，文件带frontmatter。"""
    assert KB_ROOT.exists(), "tcm_kb 根目录不存在"
    total = 0
    for d in DOMAIN_DIRS:
        p = KB_ROOT / d
        assert p.exists(), f"缺少域目录 {d}"
        mds = list(p.rglob("*.md"))
        assert len(mds) >= 6, f"{d} 文件过少：{len(mds)}"
        assert (p / "_index.md").exists(), f"{d} 缺少 _index.md"
        for f in mds:
            total += 1
            head = f.read_text(encoding="utf-8")[:200]
            assert head.startswith("---"), f"{f.name} 缺少frontmatter"
            assert "title:" in head, f"{f.name} frontmatter缺少title"
            assert "category:" in head, f"{f.name} frontmatter缺少category"
            lines = f.read_text(encoding="utf-8").splitlines()
            assert len(lines) <= 400, f"{f.name} 超过400行（{len(lines)}行），应继续拆分"
    assert total >= 70, f"知识库文件总数不足：{total}"


def test_kb_loaded_chunks():
    kb = get_kb()
    assert len(kb.files) >= 70
    assert len(kb.chunks) >= 200, f"知识块过少：{len(kb.chunks)}"


def test_search_yin_xu():
    kb = get_kb()
    hits = kb.search("阴虚的表现和调理", k=5)
    assert hits, "阴虚检索无结果"
    assert any(
        ("阴虚" in c.title) or ("阴虚" in " ".join(c.tags)) or ("阴虚" in c.section)
        for c in hits
    ), f"阴虚检索未命中相关章节: {[c.location for c in hits]}"


def test_search_domains():
    kb = get_kb()
    cases = {
        "痛经月经不调辨证": "06_临床各科",
        "舌诊舌苔": "02_诊断学",
        "麻黄的性味归经功效": "03_中药",
        "六味地黄丸": "04_方剂",
        "足三里": "05_针灸外治",
        "阴阳学说": "01_基础理论",
    }
    for q, domain in cases.items():
        hits = kb.search(q, k=5)
        assert hits, f"{q} 检索无结果"
        assert any(c.file.startswith(domain) for c in hits), \
            f"{q} 未命中域 {domain}: {[c.location for c in hits]}"


def test_search_no_result():
    kb = get_kb()
    assert kb.search("zzzqqq") == []
