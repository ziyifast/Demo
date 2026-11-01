"""知识入库：docx/md → Markdown → 切块 → embedding → pgvector。"""
import hashlib
import re
from pathlib import Path
import json 

import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from markdownify import markdownify
from sqlalchemy import text

from app.config import engine
from app.embeddings import embeddings


async def ingest_file(path: Path, doc_id: str | None = None) -> int:
    doc_id = doc_id or hashlib.sha256(path.name.encode()).hexdigest()[:12]
    md = _to_markdown(path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=80,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。"],
    )
    chunks = splitter.split_text(md)
    vectors = await embeddings.aembed_documents(chunks)

    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE knowledge_chunks SET status='superseded' WHERE doc_id=:d"),
            {"d": doc_id},
        )
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            await conn.execute(text("""
                INSERT INTO knowledge_chunks
                    (id, doc_id, version, title, content, embedding, doc_metadata, status)
                VALUES (
                    :id, :doc_id, 1, :title, :content,
                    CAST(:emb AS vector),
                    CAST(:meta AS jsonb),
                    'active'
                )
            """), {
                "id": f"{doc_id}-{i}", "doc_id": doc_id,
                "title": _extract_title(md, path.stem),
                "content": chunk,
                "emb": str(vec),
                "meta": json.dumps({"source": path.name}, ensure_ascii=False),
            })
    return len(chunks)


def _to_markdown(path: Path) -> str:
    if path.suffix.lower() == ".md":
        return path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".docx", ".doc"):
        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        raw_html = "\n".join(f"<p>{p}</p>" for p in paragraphs)
        for table in doc.tables:
            for row in table.rows:
                raw_html += "\n<p>" + " | ".join(c.text.strip() for c in row.cells) + "</p>"
        return markdownify(raw_html, heading_style="ATX")
    raise ValueError(f"不支持的格式: {path.suffix}")


def _extract_title(md: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", md, re.MULTILINE)
    return m.group(1).strip() if m else fallback