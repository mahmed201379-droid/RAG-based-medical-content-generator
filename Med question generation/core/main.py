import os
import time
import logging
import asyncio
import re
from typing import List
from langchain_core.documents import Document
from fastapi import FastAPI, HTTPException, Request
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from core.models import BookListResponse, ChapterListResponse, MCQRequest, MCQResponse, MCQList, MCQ, PresentationRequest, PresentationResponse, PresentationList
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from core import vector_store, config, processing
from core.processing import clean_chapter_title_for_query
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set this before any huggingface imports to avoid parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- CONFIGURATION ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if GROQ_API_KEY:
    os.environ['GROQ_API_KEY'] = GROQ_API_KEY
if HF_TOKEN:
    os.environ['HF_TOKEN'] = HF_TOKEN
if OPENROUTER_API_KEY:
    os.environ['OPENROUTER_API_KEY'] = OPENROUTER_API_KEY

# --- API SETUP ---
app = FastAPI(
    title="Medical MCQ Generation API",
    description="An API to generate Multiple-Choice Questions from medical documents.",
)

app.add_middleware(
    CORSMiddleware,
    # Allow all origins for development purposes.
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static Files and Frontend ---
# This will serve files from a 'static' directory at the project root
# and the main index.html file.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
static_dir = os.path.join(project_root, 'static')
os.makedirs(static_dir, exist_ok=True) # Ensure the static directory exists

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", include_in_schema=False)
async def read_index():
    """Serves the main index.html file."""
    index_path = os.path.join(project_root, 'index.html')
    return FileResponse(index_path)


# --- Backend Initialization State ---
# We use the app.state object to hold the initialization status and shared resources.
app.state.is_initialized = False
app.state.initialization_error = None
app.state.initialization_task = None

async def run_initialization():
    """The long-running initialization task that processes all books."""
    try:
        logger.info("Background initialization started...")
        (
            app.state.llm,
            app.state.retrievers,
            app.state.available_books,
            app.state.chapters_by_book,
            app.state.docs_by_book,
            app.state.vector_stores,
        ) = await vector_store.initialize_backend()
        app.state.is_initialized = True
        logger.info("Background initialization completed successfully.")
    except Exception as e:
        logger.error(f"Background initialization failed: {e}", exc_info=True)
        app.state.initialization_error = str(e)

@app.on_event("startup")
async def startup_event():
    """
    Schedules the backend initialization as a background task.
    This allows the API to be responsive immediately while processing happens.
    """
    logger.info("Scheduling backend initialization as a background task.")
    app.state.initialization_task = asyncio.create_task(run_initialization())

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanly cancel the initialization task on shutdown."""
    if app.state.initialization_task:
        logger.info("Cancelling initialization task on shutdown.")
        app.state.initialization_task.cancel()
        try:
            await app.state.initialization_task
        except asyncio.CancelledError:
            logger.info("Initialization task successfully cancelled.")


def _check_backend_readiness(request: Request):
    """
    A helper function to be called at the beginning of endpoints
    that rely on the initialized backend.
    """
    if request.app.state.initialization_error:
        raise HTTPException(
            status_code=500,
            detail=f"Backend initialization failed: {request.app.state.initialization_error}"
        )
    if not request.app.state.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Backend is still initializing. Please wait a moment and try again."
        )

@app.get("/list-books", response_model=BookListResponse)
async def list_books(request: Request):
    """
    Lists the PDF files available. If the backend is still initializing,
    it returns a 503 status, which the frontend should handle by retrying.
    """
    _check_backend_readiness(request)
    available_books = getattr(request.app.state, 'available_books', [])
    return BookListResponse(books=available_books)

@app.get("/list-chapters/{book_filename}", response_model=ChapterListResponse)
async def list_chapters(book_filename: str, request: Request):
    """Lists the extracted chapter titles for a given book."""
    _check_backend_readiness(request)
    chapters_by_book = getattr(request.app.state, 'chapters_by_book', {})
    chapters = chapters_by_book.get(book_filename)

    if chapters is None:
        raise HTTPException(
            status_code=404,
            detail=f"Book '{book_filename}' not found or no chapters were extracted."
        )
    return ChapterListResponse(chapters=chapters)

async def _get_context_for_chapter(request: Request, book_filename: str, chapter_title: str) -> List[Document]:
    """Helper to retrieve context for a given chapter with self-correction."""
    # Readiness is checked by the calling endpoint, so no need to check here.
    retriever = request.app.state.retrievers.get(book_filename)
    if retriever is None:
        raise HTTPException(
            status_code=404, 
            detail=f"Book '{book_filename}' not found or processed."
        )

    logger.info(f"Retrieving context for chapter: '{chapter_title}'")
    retrieved_docs = await retriever.ainvoke(chapter_title)

    # --- Self-Correction Mechanism ---
    if not retrieved_docs:
        logger.warning(
            f"Initial search for '{chapter_title}' yielded no results. "
            "Attempting self-correction."
        )
        # Attempt 1: Clean the chapter title and retry
        cleaned_title = clean_chapter_title_for_query(chapter_title)
        if cleaned_title and cleaned_title.lower() != chapter_title.lower():
            logger.info(f"Retrying with cleaned title: '{cleaned_title}'")
            retrieved_docs = await retriever.ainvoke(cleaned_title)

        # Attempt 2: If still no results, use metadata filtering as a fallback.
        if not retrieved_docs:
            logger.warning(f"Cleaned search for '{cleaned_title}' also failed. Falling back to metadata filter search.")
            
            faiss_store = request.app.state.vector_stores.get(book_filename)
            
            if faiss_store:
                search_query = cleaned_title if cleaned_title else chapter_title
                retrieved_docs = await asyncio.to_thread(
                    faiss_store.similarity_search,
                    query=search_query,
                    k=config.BM25_K + config.FAISS_K,
                    filter={"chapter_title": chapter_title},
                )
                
                if retrieved_docs:
                    logger.info(f"Metadata filter search found {len(retrieved_docs)} chunks for chapter '{chapter_title}'.")

    if not retrieved_docs:
        raise HTTPException(status_code=404, detail=f"Could not find any relevant context for the chapter '{chapter_title}' even after self-correction.")
    
    return retrieved_docs

def _clean_and_validate_mcqs(raw_mcq_list: List[dict], num_questions_to_collect: int, existing_mcqs: List[MCQ]) -> List[MCQ]:
    """
    Cleans and validates a list of MCQ dictionaries from the LLM.
    Handles various response formats and ensures data integrity.
    """
    final_mcqs = list(existing_mcqs) # Start with the ones we already have
    required_keys = {"question", "options", "correct_answer"}

    for i, mcq_data in enumerate(raw_mcq_list):
        if len(final_mcqs) >= num_questions_to_collect:
            break  # Stop if we've collected enough valid questions

        if not isinstance(mcq_data, dict) or not required_keys.issubset(mcq_data.keys()):
            logger.warning(f"Skipping malformed MCQ item #{i+1} in batch due to missing keys: {mcq_data}")
            continue

        if not isinstance(mcq_data.get("options"), list) or len(mcq_data["options"]) != 4:
            logger.warning(f"Skipping malformed MCQ item #{i+1} in batch due to invalid options: {mcq_data}")
            continue

        # Defensively handle correct_answer being a list
        if isinstance(mcq_data.get('correct_answer'), list):
            mcq_data['correct_answer'] = mcq_data['correct_answer'][0] if mcq_data['correct_answer'] else ""
        
        # Ensure correct_answer is a string before processing
        if not isinstance(mcq_data.get('correct_answer'), str):
            logger.warning(f"Skipping malformed MCQ item #{i+1} in batch due to invalid correct_answer type: {mcq_data}")
            continue

        # --- Defensively find the correct option text from various possible LLM response formats ---
        correct_answer_text = mcq_data['correct_answer']
        options = mcq_data['options']
        final_correct_answer = None

        # Attempt 1: The correct_answer is the exact text of an option
        if correct_answer_text in options:
            final_correct_answer = correct_answer_text
        
        # Attempt 2: The correct_answer is prefixed (e.g., "A. The answer"). Clean it and check.
        if not final_correct_answer:
            cleaned_answer = re.sub(r"^[A-D][.)]\s*", "", correct_answer_text, flags=re.IGNORECASE).strip()
            if cleaned_answer in options:
                final_correct_answer = cleaned_answer

        # Attempt 3: The correct_answer is just the letter (e.g., "B").
        if not final_correct_answer and re.fullmatch(r"[A-D]", correct_answer_text.strip(), re.IGNORECASE):
            letter = correct_answer_text.strip().upper()
            # First, check if any option starts with the letter prefix (e.g., "B. ...")
            for option in options:
                if re.match(rf"^{letter}[.)]\s*", option.strip(), re.IGNORECASE):
                    final_correct_answer = option
                    break
            # If not, fall back to assuming the letter is the index (A=0, B=1, etc.)
            if not final_correct_answer:
                option_index = ord(letter) - ord('A')
                if 0 <= option_index < len(options):
                    final_correct_answer = options[option_index]

        if final_correct_answer:
            mcq_data['correct_answer'] = final_correct_answer
            # Ensure we don't add duplicates
            if mcq_data not in final_mcqs:
                final_mcqs.append(MCQ(**mcq_data))
        else:
            logger.warning(f"Skipping malformed MCQ item #{i+1} in batch due to invalid correct_answer: '{correct_answer_text}' not in {options}")
            continue
    
    return final_mcqs

@app.post("/generate-mcq", response_model=MCQResponse)
async def handle_mcq_generation(mcq_request: MCQRequest, request: Request):
    """Receives a book, section name and generates MCQs based on relevant context."""
    _check_backend_readiness(request)

    start_time = time.time()
    try:
        # 1. Retrieve context for the section from the specific book's retriever
        retrieved_docs = await _get_context_for_chapter(request, mcq_request.book_filename, mcq_request.chapter_title)
        context_text_for_llm = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # 2. Create the MCQ generation chain with a JSON output parser
        parser = JsonOutputParser(pydantic_object=MCQList)
        
        mcq_prompt = ChatPromptTemplate.from_template(
            template=config.MCQ_PROMPT_TEMPLATE,
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        mcq_chain = mcq_prompt | request.app.state.llm | parser
        
        # 3. Invoke the chain with a retry loop to ensure the desired number of MCQs
        final_mcqs: List[MCQ] = []
        attempts = 0
        max_attempts = 3  # To prevent infinite loops

        while len(final_mcqs) < mcq_request.num_questions and attempts < max_attempts:
            needed = mcq_request.num_questions - len(final_mcqs)
            logger.info(f"Attempt {attempts + 1}/{max_attempts}: Generating {needed} more MCQs...")

            response_json = await mcq_chain.ainvoke({
                "context": context_text_for_llm,
                "num_questions": needed
            })

            # Defensively handle LLM output format variations.
            raw_mcq_list = []
            if isinstance(response_json, dict):
                raw_mcq_list = response_json.get("mcqs", [])
            elif isinstance(response_json, list):
                raw_mcq_list = response_json

            # Clean, validate, and append the newly generated MCQs
            final_mcqs = _clean_and_validate_mcqs(
                raw_mcq_list, mcq_request.num_questions, final_mcqs
            )
            attempts += 1

        end_time = time.time()

        if len(final_mcqs) < mcq_request.num_questions:
            logger.warning(
                f"Could only generate {len(final_mcqs)} out of {mcq_request.num_questions} "
                f"requested MCQs after {max_attempts} attempts."
            )

        if not final_mcqs:
            logger.error("LLM failed to generate any valid MCQs after all attempts.")
            raise HTTPException(
                status_code=500, 
                detail="The language model failed to generate any valid MCQs. Please try again or check the model's compatibility."
            )

        return MCQResponse(
            mcqs=final_mcqs,
            context=[doc.page_content for doc in retrieved_docs],
            response_time=end_time - start_time
        )
    except Exception as e:
        logger.error(f"Error during MCQ generation: {e}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail="An unexpected error occurred during MCQ generation.")

@app.post("/generate-presentation", response_model=PresentationResponse)
async def handle_presentation_generation(presentation_request: PresentationRequest, request: Request):
    """Receives a book and chapter title and generates a presentation."""
    _check_backend_readiness(request)

    start_time = time.time()
    try:
        # 1. Retrieve context
        retrieved_docs = await _get_context_for_chapter(
            request, presentation_request.book_filename, presentation_request.chapter_title
        )
        context_text_for_llm = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # 2. Create the presentation generation chain
        parser = JsonOutputParser(pydantic_object=PresentationList)
        
        presentation_prompt = ChatPromptTemplate.from_template(
            template=config.PRESENTATION_PROMPT_TEMPLATE,
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        presentation_chain = presentation_prompt | request.app.state.llm | parser
        
        # 3. Invoke the chain with a retry loop for robustness
        slides = []
        attempts = 0
        max_attempts = 3

        while not slides and attempts < max_attempts:
            attempts += 1
            logger.info(
                f"Attempt {attempts}/{max_attempts}: Generating presentation slides for "
                f"'{presentation_request.chapter_title}'..."
            )
            try:
                response_json = await presentation_chain.ainvoke({
                    "context": context_text_for_llm,
                })
                # Defensive handling of LLM output
                potential_slides = response_json.get("slides", []) if isinstance(response_json, dict) else []
                if potential_slides and isinstance(potential_slides, list):
                    slides = potential_slides
                else:
                    logger.warning(f"Attempt {attempts} failed to produce valid slides. Response: {response_json}")
            except Exception as chain_exc:
                logger.warning(f"Attempt {attempts} raised an exception during chain invocation: {chain_exc}")
                if attempts >= max_attempts:
                    raise  # Re-raise the last exception if all retries fail

        if not slides:
            logger.error("LLM failed to generate any valid presentation slides after all attempts.")
            raise HTTPException(
                status_code=500,
                detail="The language model failed to generate valid presentation slides after multiple attempts. Please try again."
            )

        end_time = time.time()

        return PresentationResponse(
            slides=slides,
            context=[doc.page_content for doc in retrieved_docs],
            response_time=end_time - start_time
        )
    except Exception as e:
        # Avoid logging full traceback for HTTPExceptions we raise intentionally
        logger.error(f"Error during presentation generation: {e}", exc_info=not isinstance(e, HTTPException))
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail="An unexpected error occurred during presentation generation.")
