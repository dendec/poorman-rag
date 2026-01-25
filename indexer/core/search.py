import sqlite3
import os
import time 
import logging
from collections import defaultdict
from usearch.index import Index
from core.cfg import IndexingConfig
from core.embedder import Embedder

logger = logging.getLogger("indexer.search")

class Search:
    def __init__(self, config: IndexingConfig):
        if not os.path.exists(config.db_file) or not os.path.exists(config.index_file):
            logger.error(f"Missing search files: {config.db_file} or {config.index_file}")
            raise FileNotFoundError("Database or Index file missing!")

        # 1. SQLite (Read-Only)
        self.conn = sqlite3.connect(f"file:{config.db_file}?mode=ro", uri=True, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # 2. Vector Index
        try:
            self.index = Index.restore(config.index_file, view=True)
        except Exception as e:
            logger.warning(f"Index restore failed, trying manual view: {e}")
            self.index = Index(ndim=config.vector_dim, metric=config.index_metric, dtype=config.index_dtype)
            self.index.view(config.index_file)

        self.embed = Embedder(config.model_name, device="cpu", max_length=config.max_tokens)
        self.cfg = config
        logger.info(f"Search engine ready. Vectors: {len(self.index)}")

    def _search_vector(self, query: str, limit: int):
        try:
            prefixed_query = self.cfg.query_prefix + query
            vec = self.embed.embed([prefixed_query])[0].flatten()
            matches = self.index.search(vec, limit)
            return matches.keys.tolist()
        except Exception as e:
            logger.error(f"Vector search error: {e}", exc_info=True)
            return []

    def _search_fts(self, query: str, limit: int):
        try:
            safe_query = query.replace('"', '').replace("'", "")
            sql = f"""
                SELECT rowid
                FROM dataset_fts
                WHERE dataset_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """
            cursor = self.conn.execute(sql, (safe_query, limit))
            return [row[0] for row in cursor.fetchall()]
        except sqlite3.OperationalError as e:
            logger.error(f"FTS search error: {e}")
            return []

    def search_hybrid(self, query: str, k: int = None):
        top_k = k or self.cfg.top_k
        vector_ids = self._search_vector(query, self.cfg.limit_vector)
        fts_ids = self._search_fts(query, self.cfg.limit_fts)

        if not fts_ids:
            final_ids = vector_ids[:top_k]
        elif not vector_ids:
            final_ids = fts_ids[:top_k]
        else:
            scores = defaultdict(float)
            for rank, doc_id in enumerate(vector_ids):
                scores[doc_id] += 1 / (self.cfg.rrf_k + rank + 1)
            for rank, doc_id in enumerate(fts_ids):
                scores[doc_id] += 1 / (self.cfg.rrf_k + rank + 1)
            sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            final_ids = [doc_id for doc_id, score in sorted_docs[:top_k]]

        if not final_ids:
            return []

        placeholders = ','.join(['?'] * len(final_ids))
        sql = f"SELECT id, text FROM dataset WHERE id IN ({placeholders})"
        rows = self.conn.execute(sql, final_ids).fetchall()
        id_to_text = {row['id']: row['text'] for row in rows}

        results = []
        for doc_id in final_ids:
            if doc_id in id_to_text:
                results.append((doc_id, id_to_text[doc_id]))

        return results

    def close(self):
        self.conn.close()

    def loop(self):
        """CLI loop for manual testing."""
        logger.info("Entering search interactive loop (/exit to quit)")
        try:
            while True:
                try:
                    q = input("▶  query > ").strip()
                except (KeyboardInterrupt, EOFError): break
                if not q: continue
                if q == "/exit": break
                start = time.perf_counter()
                results = self.search_hybrid(q)
                elapsed = time.perf_counter() - start
                print(f"\nFound {len(results)} matches in {elapsed*1000:.1f} ms:")
                for doc_id, txt in results:
                    print(f"🔹 [ID:{doc_id}] {txt}")
                print("-" * 40 + "\n")
        finally:
            self.close()
            print("\n👋 Bye!")