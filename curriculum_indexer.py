# curriculum_indexer.py

import os
import pdfplumber
import numpy as np
import json
import faiss
import time
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import logging # Keep this import for type hinting or if you need default logging somewhere else
from tech_support_logger import TechSupportLogger # Import your custom logger

# --- Initialize the custom logger for this module ---
curriculum_logger = TechSupportLogger(
    log_file_name="curriculum_indexer.log", # Separate log for indexer specific issues
    log_dir="data/logs",
    level=logging.INFO, # Set to INFO for production, DEBUG for verbose indexing info
    max_bytes=10 * 1024 * 1024, # 10 MB
    backup_count=5,
    console_output=False # Typically, indexer doesn't need to print to console
).get_logger()


class FaissCache:
    def __init__(self, expiry_seconds=300):
        self.cache = {}
        self.expiry = expiry_seconds

    def get(self, query):
        item = self.cache.get(query)
        if item:
            ts, result = item
            if time.time() - ts < self.expiry:
                curriculum_logger.debug(f"Cache hit for query: {query[:50]}...")
                return result
            else:
                curriculum_logger.debug(f"Cache expired for query: {query[:50]}...")
                del self.cache[query]
        return None

    def set(self, query, result):
        self.cache[query] = (time.time(), result)
        curriculum_logger.debug(f"Cache set for query: {query[:50]}...")


class CurriculumIndexer:
    def __init__(
        self,
        pdf_folder: str,
        index_file: str = "data/pdf_faiss.index",
        metadata_file: str = "data/pdf_metadata.json",
        chunk_size: int = 200
    ):
        self.pdf_folder = pdf_folder
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.chunk_size = chunk_size
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache = FaissCache()
        self.index = None
        self.metadata = []

        curriculum_logger.info(f"Initializing CurriculumIndexer with PDF folder: {pdf_folder}")
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            curriculum_logger.info("Existing FAISS index and metadata found. Loading index.")
            self._load_index()
        else:
            curriculum_logger.info("No existing index found. Building new index from PDFs.")
            self.build_index()


    def _extract_text_from_pdf(self, pdf_path):
        """
        Extracts text from a PDF file.
        Optimization: Collects text from all pages into a list and then joins,
        which is more efficient than repeated string concatenation.
        """
        extracted_texts = []
        if not pdf_path.lower().endswith(".pdf"):
            curriculum_logger.warning(f"Skipping non-PDF file: {pdf_path}")
            return ""

        curriculum_logger.info(f"Extracting text from PDF: {pdf_path}")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        extracted_texts.append(text.strip())
            return "\n".join(extracted_texts) if extracted_texts else ""
        except Exception as e:
            curriculum_logger.error(f"Failed to extract text from PDF {pdf_path}: {e}", exc_info=True)
            return ""

    def _chunk_text(self, text):
        """Chunks text into smaller pieces based on sentence tokenization and chunk size."""
        sentences = sent_tokenize(text)
        chunks, current, length = [], [], 0
        for s in sentences:
            s_len = len(s.split())
            if length + s_len > self.chunk_size and current: # Ensure current is not empty before chunking
                chunks.append(" ".join(current))
                current, length = [s], s_len
            else:
                current.append(s)
                length += s_len
        if current:
            chunks.append(" ".join(current))
        curriculum_logger.debug(f"Chunked text into {len(chunks)} chunks.")
        return chunks

    def _get_embedding(self, text):
        """Generates a sentence embedding for the given text."""
        try:
            return self.embed_model.encode(text)
        except Exception as e:
            curriculum_logger.error(f"Failed to get embedding for text (first 50 chars): {text[:50]}... Error: {e}", exc_info=True)
            # Return a zero vector or handle as appropriate for your application
            return np.zeros(self.embed_model.get_sentence_embedding_dimension())


    def build_index(self):
        """
        Builds the FAISS index and metadata from PDF files in the specified folder.
        """
        embeddings = []
        metadata = []
        curriculum_logger.info(f"Starting to build index from PDFs in: {self.pdf_folder}")

        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.lower().endswith(".pdf")]
        if not pdf_files:
            curriculum_logger.warning(f"No PDF files found in {self.pdf_folder}. Index will be empty.")
            # It's better to raise an error here if an empty index is not desired
            # raise ValueError("No PDF content processed for indexing. No PDF files found.")

        for file in pdf_files:
            path = os.path.join(self.pdf_folder, file)
            text = self._extract_text_from_pdf(path)
            if not text:
                curriculum_logger.warning(f"No text extracted from {file}. Skipping this file.")
                continue

            chunks = self._chunk_text(text)
            if not chunks:
                curriculum_logger.warning(f"No chunks generated for {file}. Skipping this file.")
                continue

            for i, chunk in enumerate(chunks):
                emb = self._get_embedding(chunk)
                embeddings.append(emb)
                metadata.append({"file": file, "chunk_idx": i, "text": chunk})
            curriculum_logger.debug(f"Processed {len(chunks)} chunks from {file}.")

        if not embeddings:
            curriculum_logger.error("No embeddings generated. Index will be empty.")
            # Handle case where no valid content was processed
            self.index = faiss.IndexFlatL2(self.embed_model.get_sentence_embedding_dimension())
            self.metadata = []
            # Still attempt to save empty index/metadata to avoid re-building on next run
            os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
            faiss.write_index(self.index, self.index_file)
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f)
            return


        data_np = np.array(embeddings).astype('float32')
        dim = data_np.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(data_np)
        self.metadata = metadata

        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f)
        curriculum_logger.info(f"FAISS index built and saved with {len(embeddings)} embeddings.")

    def _load_index(self):
        """Loads the FAISS index and metadata from files."""
        try:
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
            curriculum_logger.info(f"FAISS index and metadata loaded successfully. Index size: {self.index.ntotal}")
        except Exception as e:
            curriculum_logger.critical(f"Failed to load FAISS index or metadata. Error: {e}", exc_info=True)
            curriculum_logger.warning("Attempting to rebuild index due to loading failure.")
            # Fallback: if loading fails, try to rebuild the index
            self.build_index()


    def search(self, query: str, top_k: int = 3):
        """Searches the FAISS index for relevant chunks based on a query."""
        cached = self.cache.get(query)
        if cached:
            curriculum_logger.debug(f"Returning cached search results for query: {query[:50]}...")
            return cached

        try:
            query_vec = np.array([self._get_embedding(query)]).astype('float32')
            if self.index.ntotal == 0:
                curriculum_logger.warning("FAISS index is empty. No search results will be returned.")
                return []

            D, I = self.index.search(query_vec, top_k)
            # Filter out invalid indices (e.g., -1 if top_k is larger than index size)
            results = [self.metadata[i]['text'] for i in I[0] if i < len(self.metadata) and i != -1]

            self.cache.set(query, results)
            curriculum_logger.info(f"Search completed for query: {query[:50]}... Found {len(results)} relevant chunks.")
            return results
        except Exception as e:
            curriculum_logger.error(f"Error during FAISS search for query: {query[:50]}... Error: {e}", exc_info=True)
            return []
