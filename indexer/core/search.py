import logging
import os
import time
import lancedb
from core.cfg import IndexingConfig

logger = logging.getLogger("indexer.search")

class Search:
    def __init__(self, config: IndexingConfig):
        # We don't check for file existence like before because LanceDB handles connections gracefully
        # or raises on open_table.
        logger.info(f"Connecting to LanceDB at {config.lancedb_uri}...")
        try:
            self.db = lancedb.connect(config.lancedb_uri)
            self.table = self.db.open_table(config.table_name)
        except Exception as e:
             logger.error(f"Failed to open LancaDB table: {e}")
             raise e

        # No need to load embedding model separately if LanceDB does it? 
        # Actually LanceDB python client usually needs vectors/embeddings passed to it, 
        # unless we used the Embedding API of LanceDB (which is experimental/newer).
        # Poorman-rag manually embedded in `_search_vector`.
        # We should keep manual embedding for consistency with indexer configuration.
        # Wait, `poorman-rag` search.py used `core.embedder`.
        from core.embedder import Embedder
        self.embed = Embedder(config.model_name, device="cpu", max_length=config.max_tokens)
        self.cfg = config
        
        logger.info(f"Search engine ready. Rows: {self.table.count_rows()}")

    def search_hybrid(self, query: str, k: int = None):
        top_k = k or self.cfg.top_k
        
        # Embed query
        prefixed_query = self.cfg.query_prefix + query
        try:
            # Need strict error handling here
            query_vec = self.embed.embed([prefixed_query])[0]
            
            # If using int8 storage, we must scale the query vector to match
            if self.cfg.vector_dtype == "int8":
                import numpy as np
                query_vec = (query_vec * 127).clip(-128, 127).astype(np.int8)
            elif self.cfg.vector_dtype == "float16":
                import numpy as np
                query_vec = query_vec.astype(np.float16)
                
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []

        # LanceDB Hybrid Search
        # If FTS is not enabled in config, fallback to vector only?
        # But if we indexed with FTS, we can use it.
        # Let's try hybrid if FTS mode is not "none".
        
        try:
            # Note: The availability of "hybrid" search in python client depends on version.
            # Assuming standard .search() with vector and .text() is possible?
            # Or using .search(query_vec).limit(k)
            # Recent LanceDB: table.search(query_str, query_type="hybrid").vector(vec)
            
            # Let's check typical usage:
            # table.search(query, query_type="hybrid").limit(top_k) 
            # This requires LanceDB to generate embeddings itself OR we pass vector.
            # If we pass vector: table.search(vector).text(query).limit(top_k)?? No.
            
            # Safe bet: Manual RRF if detailed control needed, or use .search(vector)
            # The user code had hybrid RRF logic.
            # Let's implement RRF manually to match previous behavior 1:1 using LanceDB features.
            
            # 1. Vector Search
            # Debug types
            logger.debug(f"Query vector type: {query_vec.dtype}, shape: {query_vec.shape}")
            
            # Use explicit .vector() syntax which is more robust in recent LanceDB
            query_builder = self.table.search(query_vec, vector_column_name="vector")
            vec_results = query_builder.limit(self.cfg.limit_vector).to_arrow()
            
            # 2. FTS Search
            fts_ids = []
            if self.cfg.fts_mode != "none":
                # FTS in LanceDB usually via .search(query) with default query_type="vector" trying to parse? 
                # Or .search(query, query_type="fts")
                try:
                    fts_results = self.table.search(query, query_type="fts").limit(self.cfg.limit_fts).to_arrow()
                    fts_ids = fts_results["id"].to_pylist()
                except Exception as e:
                    logger.warning(f"FTS search failed (maybe not indexed?): {e}")

            # RRF Logic
            vector_ids = vec_results["id"].to_pylist()
            
            if not fts_ids:
                final_ids = vector_ids[:top_k]
            elif not vector_ids:
                final_ids = fts_ids[:top_k]
            else:
                from collections import defaultdict
                scores = defaultdict(float)
                for rank, doc_id in enumerate(vector_ids):
                    scores[doc_id] += 1 / (self.cfg.rrf_k + rank + 1)
                for rank, doc_id in enumerate(fts_ids):
                    scores[doc_id] += 1 / (self.cfg.rrf_k + rank + 1)
                sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                final_ids = [doc_id for doc_id, score in sorted_docs[:top_k]]
            
            if not final_ids:
                return []

            # Retrieve text for final IDs
            # LanceDB filtering
            # where id in [...]
            # id is int.
            id_list_str = ", ".join(map(str, final_ids))
            # Note: LanceDB SQL filtering might be limited. safer to filter by pyarrow or just fetch all?
            # Wait, we already have data in `vec_results` and `fts_results` but only subset.
            # We need to fetch specific IDs.
            # self.table.search().where(f"id IN ({id_list_str})") works in SQL usually.
            
            matches = self.table.search().where(f"id IN ({id_list_str})").limit(len(final_ids)).to_arrow()
            
            id_to_text = {}
            pylist = matches.to_pylist()
            for row in pylist:
                id_to_text[row['id']] = row['text']

            results = []
            for doc_id in final_ids:
                if doc_id in id_to_text:
                    results.append((doc_id, id_to_text[doc_id]))
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []

    def close(self):
        pass # Connections are managed by library

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