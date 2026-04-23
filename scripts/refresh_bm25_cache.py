from __future__ import annotations

import argparse

from app.core.rag.public_kb_service import PublicKnowledgeService
from app.config.settings import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh BM25 cache for public KB")
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection name (default from settings)",
    )
    args = parser.parse_args()

    count = PublicKnowledgeService.refresh_cache(collection_name=args.collection)
    collection = args.collection or settings.PUBLIC_KB_COLLECTION
    print(f"BM25 cache refreshed: collection={collection} docs={count}")


if __name__ == "__main__":
    main()
