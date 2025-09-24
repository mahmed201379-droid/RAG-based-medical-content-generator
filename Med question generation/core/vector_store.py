import os
import logging
import pickle
import re
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from core.utils import get_file_hash
from core.processing import preprocess_document_content, extract_chapter_titles, extract_chapters_with_llm
from core import config
from typing import Dict, List, Tuple, Optional, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseChatModel
import asyncio
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# A version for the processing logic. Increment this to force reprocessing of all caches
# if you change how documents are chunked, preprocessed, or embedded.
CACHE_LOGIC_VERSION = "1.1"


def _create_or_load_retriever_for_book(
    pdf_file: str, embeddings: HuggingFaceEmbeddings, llm: BaseChatModel
) -> Tuple[Optional[EnsembleRetriever], List[str], List[Document], Optional[FAISS]]:
    """
    Creates or loads a chapter-aware hybrid retriever for a single PDF book.

    This function handles loading a PDF, preprocessing its content, chunking it, and
    setting up a hybrid search retriever (BM25 + FAISS). It persists the FAISS
    vector store and the processed documents to disk to avoid reprocessing.

    Args:
        pdf_file: The filename of the PDF in the PDF_DIRECTORY.
        embeddings: The embedding model to use for semantic search.
        llm: The language model, used for intelligent chapter extraction.

    Returns:
        A tuple containing:
        - An `EnsembleRetriever` instance if successful, otherwise `None`.
        - A list of extracted chapter titles for the book.
        - A list of all processed `Document` chunks for the book.
        - The `FAISS` vector store instance.
    """
    pdf_path = os.path.join(config.PDF_DIRECTORY, pdf_file)
    # Sanitize the model name to create a valid filename component
    model_name_safe = embeddings.model_name.replace("/", "_")
    base_name_raw = os.path.splitext(pdf_file)[0]
    # Sanitize the base_name to remove characters that are invalid in file paths.
    # This replaces any non-alphanumeric (except hyphen/underscore) with an underscore.
    base_name = re.sub(r'[^\w\-_.]', '_', base_name_raw)

    vector_store_path = os.path.join(
        config.VECTOR_STORE_DIRECTORY, f"{base_name}_{model_name_safe}.faiss"
    )
    docs_cache_path = os.path.join(
        config.VECTOR_STORE_DIRECTORY, f"{base_name}_{model_name_safe}_docs.pkl"
    )
    cache_info_path = os.path.join(
        config.VECTOR_STORE_DIRECTORY, f"{base_name}_{model_name_safe}_cache_info.json"
    )

    logger.info(f"Processing '{pdf_file}' for Hybrid Search...")

    # --- Cache Invalidation Logic ---
    current_pdf_hash = get_file_hash(pdf_path)
    cache_is_valid = False

    if os.path.exists(cache_info_path):
        try:
            with open(cache_info_path, "r") as f:
                cache_info = json.load(f)
            # Check if hash and logic version match, and that other required files exist
            if (cache_info.get("pdf_hash") == current_pdf_hash and
                cache_info.get("logic_version") == CACHE_LOGIC_VERSION and
                os.path.exists(vector_store_path) and
                os.path.exists(docs_cache_path)):
                logger.info(f"Valid cache found for '{pdf_file}' based on file hash and logic version.")
                cache_is_valid = True
            else:
                logger.info(f"Cache for '{pdf_file}' is outdated (hash or version mismatch). Reprocessing...")
        except (json.JSONDecodeError, KeyError, IOError) as e:
            logger.warning(f"Could not read or validate cache info for '{pdf_file}'. Reprocessing. Error: {e}")

    # 1. Load from cache or process from scratch
    if cache_is_valid:
        logger.info(f"Loading cached documents for '{pdf_file}'...")
        with open(docs_cache_path, "rb") as f:
            final_documents = pickle.load(f)

        logger.info(f"Loading existing FAISS vector store for '{pdf_file}'...")
        logger.warning(
            "Loading FAISS index with allow_dangerous_deserialization=True. "
            "This is a security risk if the index file is compromised."
        )
        vectors = FAISS.load_local(
            vector_store_path, embeddings, allow_dangerous_deserialization=True
        )

        # The chapter information is already in the metadata of the cached documents.
        # Deriving it from there is more reliable than re-extracting or relying on a separate cache file.
        if final_documents:
            # Use an ordered set to preserve chapter order from the document.
            seen_titles = set()
            chapter_titles = []
            for doc in final_documents:
                title = doc.metadata.get("chapter_title")
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    chapter_titles.append(title)
            if not chapter_titles:
                 chapter_titles = ["Full Document"]
            logger.info(f"Extracted {len(chapter_titles)} unique chapter titles from cached documents.")
        else:
            # Fallback for empty doc cache
            chapter_titles = []
            logger.warning(f"Cached document file for '{pdf_file}' is empty.")

    else:
        logger.info(f"No cache found. Processing '{pdf_file}' from scratch...")

        # 1. Attempt to extract chapters using LLM first as it's more accurate for structured TOCs.
        chapters = extract_chapters_with_llm(pdf_path, llm)

        # 2. If LLM fails or finds nothing, fall back to heuristic extraction.
        if not chapters:
            logger.info(f"LLM extraction failed or found no chapters. Falling back to heuristic extraction for '{pdf_file}'.")
            chapters = extract_chapter_titles(pdf_path)

        if not chapters:
            logger.warning(f"Could not extract any chapters from '{pdf_file}'. Processing as a single document.")
            chapters = [{"page_number": 1, "chapter_title": "Full Document"}]
        
        chapter_titles = [c["chapter_title"] for c in chapters]

        # 3. Load pages and group them by chapter
        logger.info(f"Loading and grouping pages by chapter for '{pdf_file}'...")
        loader = PyMuPDFLoader(pdf_path)
        docs_by_page = loader.load()
        if not docs_by_page:
            logger.warning(f"No documents found in '{pdf_file}'. Skipping.")
            return None, [], [], None

        # Create a mapping from page number to chapter title
        page_to_chapter_map: Dict[int, str] = {}
        chapter_iter = iter(chapters)
        current_chapter = next(chapter_iter, None)
        current_title = "Preface / Introduction" # Default for pages before the first chapter

        for i in range(len(docs_by_page)):
            page_number_1_based = i + 1
            if current_chapter and page_number_1_based >= current_chapter["page_number"]:
                current_title = current_chapter["chapter_title"]
                current_chapter = next(chapter_iter, None)
            page_to_chapter_map[i] = current_title

        # Group page documents by chapter
        chapter_to_pages: Dict[str, List[Document]] = {}
        for i, page_doc in enumerate(docs_by_page):
            chapter_title = page_to_chapter_map.get(i, "Unknown Chapter")
            if chapter_title not in chapter_to_pages:
                chapter_to_pages[chapter_title] = []
            chapter_to_pages[chapter_title].append(page_doc)

        # 3. Process and chunk each chapter's content
        final_documents = []
        text_splitter = SemanticChunker(embeddings)

        for chapter_title, page_docs in chapter_to_pages.items():
            logger.info(f"Chunking chapter: '{chapter_title}'")
            chapter_content = "\n\n".join(preprocess_document_content(doc.page_content) for doc in page_docs)
            if not chapter_content.strip():
                continue
            
            text_chunks = text_splitter.split_text(chapter_content)
            chapter_metadata = {"source": pdf_file, "chapter_title": chapter_title}
            for chunk in text_chunks:
                final_documents.append(Document(page_content=chunk, metadata=chapter_metadata))

        if not final_documents:
            logger.warning(f"Processing resulted in no documents for '{pdf_file}'. Skipping.")
            return None, [], [], None

        # Cache the processed documents
        logger.info(f"Saving processed documents to '{docs_cache_path}'...")
        with open(docs_cache_path, "wb") as f:
            pickle.dump(final_documents, f)

        # Create and save FAISS vector store
        logger.info(f"Creating and saving new FAISS vector store to '{vector_store_path}'...")
        vectors = FAISS.from_documents(final_documents, embeddings)
        vectors.save_local(vector_store_path)

        # After successfully creating all caches, write the cache info file.
        logger.info(f"Saving new cache metadata to '{cache_info_path}'...")
        cache_info_data = {
            "pdf_hash": current_pdf_hash,
            "logic_version": CACHE_LOGIC_VERSION,
            "source_pdf": pdf_file,
        }
        with open(cache_info_path, "w") as f:
            json.dump(cache_info_data, f, indent=4)

    # 2. Initialize Keyword-based BM25 Retriever
    bm25_retriever = BM25Retriever.from_documents(final_documents)
    bm25_retriever.k = config.BM25_K

    # 3. Initialize Semantic FAISS Retriever
    faiss_retriever = vectors.as_retriever(search_kwargs={"k": config.FAISS_K})

    # 4. Initialize Ensemble Retriever
    logger.info(f"Creating Ensemble Retriever for '{pdf_file}'...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=config.ENSEMBLE_WEIGHTS
    )
    logger.info(f"Ensemble retriever for '{pdf_file}' created with {len(final_documents)} chunks.")
    return ensemble_retriever, chapter_titles, final_documents, vectors

async def initialize_backend() -> Tuple[BaseChatModel, Dict[str, BaseRetriever], List[str], Dict[str, List[str]], Dict[str, List[Document]], Dict[str, FAISS]]:
    """
    Initializes all necessary components for the RAG backend.

    This function sets up the language model, embedding model, and processes
    all PDF files in the specified directory to create a dictionary of
    retrievers, one for each book.

    Returns:
        A tuple containing:
        - The initialized language model.
        - A dictionary mapping book filenames to their corresponding retriever.
        - A list of available PDF filenames.
        - A dictionary mapping book filenames to their list of chapter titles.
        - A dictionary mapping book filenames to their list of all processed chunks (for potential future use).
        - A dictionary mapping book filenames to their FAISS vector store instance.

    Raises:
        RuntimeError: If the PDF directory is missing, empty, or if any other
                      fatal error occurs during initialization.
    """
    try:
        llm = ChatGroq(
            temperature=0, 
            model_name=config.LLM_MODEL
        )
        if not os.path.exists(config.PDF_DIRECTORY) or not os.listdir(
            config.PDF_DIRECTORY
        ):
            raise FileNotFoundError(
                f"The directory '{config.PDF_DIRECTORY}' does not exist or is empty. "
                "Please add your PDF files."
            )

        os.makedirs(config.VECTOR_STORE_DIRECTORY, exist_ok=True)

        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

        retrievers: Dict[str, BaseRetriever] = {}
        chapters_by_book: Dict[str, List[str]] = {}
        docs_by_book: Dict[str, List[Document]] = {}
        vector_stores: Dict[str, FAISS] = {}
        pdf_files = [
            f for f in os.listdir(config.PDF_DIRECTORY) if f.lower().endswith(".pdf")
        ]

        async def process_book(pdf_file: str):
            """Helper to process a single book, suitable for asyncio."""
            retriever = None
            chapters = []
            documents = []
            vector_store_instance = None
            try:
                # Run synchronous, CPU/IO-bound code in a thread pool to avoid blocking
                retriever, chapters, documents, vector_store_instance = await asyncio.to_thread(
                    _create_or_load_retriever_for_book, pdf_file, embeddings, llm
                )
            except Exception as e:
                logger.error(
                    f"Failed to process and create retriever for '{pdf_file}': {e}",
                    exc_info=True,
                )
            return pdf_file, retriever, chapters, documents, vector_store_instance

        logger.info(f"Processing {len(pdf_files)} books in parallel...")
        tasks = [process_book(pdf_file) for pdf_file in pdf_files]
        results = await asyncio.gather(*tasks)

        for pdf_file, retriever, chapters, documents, vector_store_instance in results:
            if retriever and vector_store_instance:
                retrievers[pdf_file] = retriever
                chapters_by_book[pdf_file] = chapters
                vector_stores[pdf_file] = vector_store_instance
                if documents:
                    docs_by_book[pdf_file] = documents

        if not retrievers:
            logger.warning("No retrievers were successfully created. The API might not function as expected. Please check the PDF files and logs.")

        logger.info("Backend initialized successfully.")
        return llm, retrievers, pdf_files, chapters_by_book, docs_by_book, vector_stores
    except FileNotFoundError as e:
        logger.error(e)
        raise RuntimeError(str(e)) from e
    except Exception as e:
        logger.error(
            f"A critical error occurred during backend initialization: {e}",
            exc_info=True,
        )
        raise RuntimeError(
            "A critical error occurred during backend initialization."
        ) from e
