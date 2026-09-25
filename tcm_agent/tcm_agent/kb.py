"""分级Markdown知识库的加载与检索（RAG-lite）。

知识库为树形目录：tcm_kb/<NN_域>/<主题>.md，每个文件带YAML frontmatter
(title/category/tags/source)。检索时按 ## 二级标题切块，对查询做字符二元组
匹配并按标题/标签/章节加权打分，返回top-k。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from .config import KB_ROOT

_PUNCT_RE = re.compile(r"[\s，。、；：？！,.;:?!()\[\]{}「」《》·—\-]+")


@dataclass
class Chunk:
    """知识库检索单元：一个文件的一个二级章节。"""
    file: str                 # 相对KB根的路径（正斜杠）
    title: str                # 文件title（frontmatter）
    category: str = ""
    section: str = ""         # 二级标题
    tags: list[str] = field(default_factory=list)
    text: str = ""

    @property
    def location(self) -> str:
        return f"{self.file}" + (f" :: {self.section}" if self.section else "")

    def excerpt(self, n: int = 280) -> str:
        t = self.text.strip().replace("\n", " ")
        return t[:n] + ("…" if len(t) > n else "")


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str, str]:
    """解析frontmatter，返回 (meta, title, body)。无frontmatter时优雅降级。"""
    lines = raw.splitlines()
    meta: dict[str, str] = {}
    if lines and lines[0].strip() == "---":
        try:
            end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
        except StopIteration:
            end = -1
        if end > 0:
            for ln in lines[1:end]:
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    meta[k.strip()] = v.strip()
            body = "\n".join(lines[end + 1:])
            title = meta.get("title", "")
            return meta, title, body
    title = ""
    for ln in lines:
        if ln.startswith("# "):
            title = ln[2:].strip()
            break
    return meta, title or "未命名", raw


def _split_sections(body: str) -> list[tuple[str, str]]:
    """按二级标题切块，返回 (章节名, 章节文本)。"""
    sections: list[tuple[str, str]] = []
    current_name = ""
    buf: list[str] = []
    for ln in body.splitlines():
        if ln.startswith("## ") or ln.startswith("##"):
            if buf or current_name:
                sections.append((current_name, "\n".join(buf)))
            current_name = ln.lstrip("#").strip()
            buf = []
        else:
            buf.append(ln)
    sections.append((current_name, "\n".join(buf)))
    # 过滤空块
    return [(name, txt) for name, txt in sections if txt.strip()]


class KnowledgeBase:
    """分级Markdown知识库。"""

    def __init__(self, root=KB_ROOT) -> None:
        self.root = root
        self.chunks: list[Chunk] = []
        self.files: list[str] = []
        self._load()

    # ------------------------------------------------------------ 加载

    def _load(self) -> None:
        if not self.root.exists():
            return
        for path in sorted(self.root.rglob("*.md")):
            rel = path.relative_to(self.root).as_posix()
            self.files.append(rel)
            try:
                raw = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            meta, title, body = _parse_frontmatter(raw)
            tags = [t.strip() for t in meta.get("tags", "").strip("[]").split(",") if t.strip()]
            category = meta.get("category", "")
            for sec_name, sec_text in _split_sections(body):
                self.chunks.append(
                    Chunk(
                        file=rel,
                        title=title,
                        category=category,
                        section=sec_name,
                        tags=tags,
                        text=sec_text,
                    )
                )

    # ------------------------------------------------------------ 检索

    @staticmethod
    def _bigrams(query: str) -> set[str]:
        """查询切词：中文串取二元组（短串整体保留），ASCII串取整词（小写）。

        分开处理可避免纯字母查询的二元组（如"zz"）在英文参考文献中误命中。
        """
        cleaned = _PUNCT_RE.sub(" ", query)
        terms: set[str] = set()
        for run in re.findall(r"[\u4e00-\u9fff]+", cleaned):
            if len(run) <= 4:
                terms.add(run)
            for i in range(len(run) - 1):
                terms.add(run[i:i + 2])
        for word in re.findall(r"[A-Za-z0-9]+", cleaned):
            if len(word) >= 2:
                terms.add(word.lower())
        return terms

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        """字符二元组/词 加权检索，返回top-k相关章节。"""
        grams = self._bigrams(query)
        if not grams:
            return []
        scored: list[tuple[float, Chunk]] = []
        for ch in self.chunks:
            score = 0.0
            title = ch.title.lower()
            section = ch.section.lower()
            tags_blob = " ".join(ch.tags).lower()
            body = ch.text.lower()
            for g in grams:
                if g in title:
                    score += 6.0
                if g in tags_blob:
                    score += 6.0
                if g in section:
                    score += 3.0
                hits = min(body.count(g), 3)
                score += hits * 1.2
            if score > 0:
                norm = score / (1.0 + len(body) / 2000.0)
                scored.append((norm, ch))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ch for _, ch in scored[:k]]

    # ------------------------------------------------------------ 统计

    def stats(self) -> dict[str, int]:
        by_dir: dict[str, int] = {}
        for rel in self.files:
            d = rel.split("/", 1)[0]
            by_dir[d] = by_dir.get(d, 0) + 1
        return {"files": len(self.files), "chunks": len(self.chunks), **by_dir}


@lru_cache(maxsize=1)
def get_kb() -> KnowledgeBase:
    """进程级单例知识库。"""
    return KnowledgeBase()


def format_search_results(chunks: list[Chunk]) -> str:
    """把检索结果格式化为供LLM阅读的文本。"""
    if not chunks:
        return "知识库中未检索到相关内容。"
    parts = []
    for i, ch in enumerate(chunks, 1):
        head = f"[{i}] {ch.title}" + (f" - {ch.section}" if ch.section else "") + f"（{ch.file}）"
        parts.append(f"{head}\n{ch.excerpt(500)}")
    return "\n\n".join(parts)
