# memory_manager.py
import os
import pickle
from typing import List, Dict
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from threading import Lock, Thread
import logging # Keep this import for type hinting or if you need default logging somewhere else
from datetime import datetime
from queue import Queue, Empty
from tech_support_logger import TechSupportLogger # Import your custom logger

# --- Remove the old basic logging config and logger initialization ---
# logging.basicConfig(level=logging.INFO) # REMOVE OR COMMENT THIS LINE
# logger = logging.getLogger(__name__) # This will be replaced by our custom logger

# --- Initialize the custom logger for this module ---
memory_logger = TechSupportLogger(
    log_file_name="memory_manager.log", # Separate log for memory manager issues
    log_dir="data/logs",
    level=logging.INFO, # Set to INFO for production, DEBUG for verbose memory info
    max_bytes=10 * 1024 * 1024, # 10 MB
    backup_count=5,
    console_output=False # Typically, memory manager doesn't need to print to console
).get_logger()


# Pre-load models at module level for better performance
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = EMBEDDER.get_sentence_embedding_dimension()

class MemoryManager:
    def __init__(self,
                 max_tokens=2048,
                 memory_limit=1000,
                 index_path="faiss.index",
                 memory_path="memory.pkl"):

        self.max_tokens = max_tokens
        self.memory_limit = memory_limit
        self.recent_history: List[Dict] = []
        self.long_term_memory: List[Dict] = []
        self.vector_map = []

        self.index_path = index_path
        self.memory_path = memory_path
        self.lock = Lock()
        self.last_save = datetime.now()
        self.save_queue = Queue()

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(EMBED_DIM)

        # Start background saver thread
        self.saver_thread = Thread(target=self._background_saver, daemon=True)
        self.saver_thread.start()

        # Async load existing data
        self._load_memory_async()
        memory_logger.info(f"MemoryManager initialized. Index path: {self.index_path}, Memory path: {self.memory_path}")


    def _background_saver(self):
        """Dedicated thread for handling save operations"""
        memory_logger.info("Background saver thread started.")
        while True:
            try:
                # Wait for save commands with timeout
                save_type = self.save_queue.get(timeout=60.0)

                if save_type == "full":
                    self._safe_save()
                elif save_type == "index":
                    self._save_index_only()
                memory_logger.debug(f"Background saver processed '{save_type}' command.")

            except Empty:
                # Periodic flush even if no requests
                if (datetime.now() - self.last_save).total_seconds() > 300:  # 5 min
                    memory_logger.debug("Background saver: Periodic full save triggered.")
                    self._safe_save()
            except Exception as e:
                memory_logger.error(f"Background saver error: {e}", exc_info=True)

    def _safe_save(self):
        """Thread-safe full save operation"""
        try:
            with self.lock:
                memory_copy = self.long_term_memory.copy()
                vector_copy = self.vector_map.copy()
                index_copy = faiss.clone_index(self.index)

            # Save memory data
            temp_path = self.memory_path + ".tmp"
            with open(temp_path, "wb") as f:
                pickle.dump({
                    "long_term_memory": memory_copy,
                    "vector_map": vector_copy
                }, f)
            os.replace(temp_path, self.memory_path)

            # Save FAISS index
            temp_idx = self.index_path + ".tmp"
            faiss.write_index(index_copy, temp_idx)
            os.replace(temp_idx, self.index_path)

            self.last_save = datetime.now()
            memory_logger.info("Full memory state saved successfully.")

        except Exception as e:
            memory_logger.error(f"Failed to perform full save: {e}", exc_info=True)

    def _save_index_only(self):
        """Quick save of just the FAISS index"""
        try:
            with self.lock:
                index_copy = faiss.clone_index(self.index)

            temp_idx = self.index_path + ".tmp"
            faiss.write_index(index_copy, temp_idx)
            os.replace(temp_idx, self.index_path)

            memory_logger.debug("FAISS index saved.")
        except Exception as e:
            memory_logger.error(f"Failed to save FAISS index only: {e}", exc_info=True)

    def _load_memory_async(self):
        """Async memory loading"""
        memory_logger.info("Attempting to load existing memory data.")
        try:
            if os.path.exists(self.index_path):
                with self.lock:
                    self.index = faiss.read_index(self.index_path)
                memory_logger.info(f"FAISS index loaded from {self.index_path}. Total vectors: {self.index.ntotal}")
            else:
                memory_logger.info(f"No existing FAISS index found at {self.index_path}.")

            if os.path.exists(self.memory_path):
                with open(self.memory_path, "rb") as f:
                    data = pickle.load(f)
                    with self.lock:
                        self.long_term_memory = data.get("long_term_memory", [])
                        self.vector_map = data.get("vector_map", [])
                memory_logger.info(f"Long-term memory loaded from {self.memory_path}. Total messages: {len(self.long_term_memory)}")
            else:
                memory_logger.info(f"No existing memory data found at {self.memory_path}.")

        except Exception as e:
            memory_logger.error(f"Memory load failed: {e}", exc_info=True)

    def add_message(self, role: str, content: str):
        """Optimized message addition with async processing"""
        start_time = datetime.now()
        original_content_length = len(content)
        content = content[:self.max_tokens * 4] # Truncate very long messages before embedding

        try:
            # Generate embedding first (most time-consuming part)
            embedding = EMBEDDER.encode([content])[0]

            with self.lock:
                message = {"role": role, "content": content}
                self.recent_history.append(message)
                self.long_term_memory.append(message)

                # Add to FAISS index
                self.index.add(np.array([embedding]))
                self.vector_map.append(len(self.long_term_memory) - 1)

                self._trim_history()

            # Trigger background save if needed
            if (datetime.now() - self.last_save).total_seconds() > 60:
                self.save_queue.put("full")
                memory_logger.debug("Full save queued (time-based).")
            else:
                self.save_queue.put("index")  # Just save the index
                memory_logger.debug("Index save queued (incremental).")

        except Exception as e:
            memory_logger.error(f"Failed to add message (role: {role}, content_len: {original_content_length}): {e}", exc_info=True)
            raise

        duration = (datetime.now() - start_time).total_seconds()
        memory_logger.debug(f"Added message in {duration:.3f}s. Content length: {len(content)}")

    def _trim_history(self):
        """In-memory trimming optimized for performance"""
        # Token-based trimming for recent_history
        total_chars = sum(len(m["content"]) for m in self.recent_history)
        while total_chars > self.max_tokens * 4 and len(self.recent_history) > 1:
            removed_msg = self.recent_history.pop(0)
            total_chars -= len(removed_msg["content"])
            memory_logger.debug(f"Trimmed recent history. Removed message (role: {removed_msg['role']}, len: {len(removed_msg['content'])}). New recent history length: {len(self.recent_history)}")

        # Absolute limit enforcement for long_term_memory
        if len(self.long_term_memory) > self.memory_limit:
            excess = len(self.long_term_memory) - self.memory_limit
            self.long_term_memory = self.long_term_memory[excess:]
            self.vector_map = self.vector_map[excess:]
            memory_logger.info(f"Trimmed long-term memory. Removed {excess} messages. New length: {len(self.long_term_memory)}")
            self.rebuild_faiss_index()

    def rebuild_faiss_index(self):
        """Optimized index rebuilding"""
        memory_logger.info("Starting FAISS index rebuild.")
        try:
            new_index = faiss.IndexFlatL2(EMBED_DIM)
            embeddings = []

            for msg in self.long_term_memory:
                # Ensure content is not empty before embedding
                if msg["content"]:
                    embeddings.append(EMBEDDER.encode([msg["content"]])[0])
                else:
                    memory_logger.warning(f"Skipping empty content message during index rebuild: {msg}")


            if embeddings:
                new_index.add(np.array(embeddings))
                memory_logger.info(f"Rebuilt FAISS index with {len(embeddings)} vectors.")
            else:
                memory_logger.warning("No embeddings to add during index rebuild. Index remains empty.")

            with self.lock:
                self.index = new_index
                self.vector_map = list(range(len(self.long_term_memory)))

        except Exception as e:
            memory_logger.error(f"FAISS index rebuild failed: {e}", exc_info=True)
            raise

    def retrieve_relevant(self, query: str, top_k=3) -> List[Dict]:
        """Efficient retrieval with error handling"""
        if not self.long_term_memory or self.index.ntotal == 0:
            memory_logger.debug("No long-term memory or empty index, skipping retrieval.")
            return []

        try:
            query_vec = EMBEDDER.encode([query[:1000]])[0]  # Truncate long queries for embedding
            memory_logger.debug(f"Retrieving relevant memories for query: {query[:50]}...")

            with self.lock:
                D, I = self.index.search(np.array([query_vec]), top_k)
                # Filter out invalid indices (e.g., -1 if top_k is larger than index size)
                results = [self.long_term_memory[self.vector_map[i]]
                           for i in I[0] if i < len(self.vector_map) and i != -1]
                memory_logger.debug(f"Found {len(results)} relevant memories.")
                return results

        except Exception as e:
            memory_logger.error(f"Retrieval failed for query: {query[:50]}... Error: {e}", exc_info=True)
            return []

    def summarize_old(self) -> str:
        """Fast fallback summarization"""
        with self.lock:
            old_memory = self.long_term_memory[:-len(self.recent_history)] if self.recent_history else self.long_term_memory

            if not old_memory:
                memory_logger.debug("No old memory to summarize.")
                return "No previous context"

            combined = " ".join([m["content"] for m in old_memory])
            summary = f"Previous context: {combined[:500]}..."  # Simple truncation
            memory_logger.debug(f"Generated summary of old memory (length: {len(combined)}).")
            return summary

    def get_context(self, query: str) -> List[Dict]:
        """Safe context building with fallbacks"""
        try:
            summary = self.summarize_old()
            relevant = self.retrieve_relevant(query) or []
            context = [{"role": "system", "content": summary}] + relevant + self.recent_history
            memory_logger.debug(f"Context built. Summary len: {len(summary)}, Relevant count: {len(relevant)}, Recent history count: {len(self.recent_history)}")
            return context
        except Exception as e:
            memory_logger.error(f"Context build failed for query: {query[:50]}... Error: {e}", exc_info=True)
            # Fallback to recent messages
            fallback_context = self.recent_history[-10:]
            memory_logger.warning(f"Falling back to recent history for context. Count: {len(fallback_context)}")
            return fallback_context

    def __del__(self):
        """Cleanup on destruction - attempts to save final state."""
        memory_logger.info("MemoryManager instance is being deleted. Attempting final save.")
        self._safe_save()
        # It's good practice to join the saver thread if it's not a daemon,
        # but since it's daemon, it will exit with the main program.
        # If you wanted to ensure all queued saves are processed, you'd need
        # a more explicit shutdown mechanism (e.g., a sentinel value in the queue).
