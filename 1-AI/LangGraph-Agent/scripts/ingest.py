"""python scripts/ingest.py docs/"""
import asyncio
import sys
from pathlib import Path

from app.db_init import init_db
from app.knowledge_pipeline import ingest_file


async def main(dir_path: str):
    await init_db()
    total = 0
    for p in Path(dir_path).glob("*"):
        if p.suffix.lower() in (".docx", ".md"):
            n = await ingest_file(p)
            total += n
            print(f"入库 {p.name}: {n} 块")
    print(f"完成，共 {total} 块")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "docs"))