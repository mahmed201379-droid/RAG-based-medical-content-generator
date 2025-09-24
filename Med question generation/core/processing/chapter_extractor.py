import os
import re
import logging
import fitz  # PyMuPDF
import statistics
import tiktoken
from collections import Counter
from typing import List, Dict, Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Maximum number of tokens to use for LLM-based TOC extraction.
MAX_TOC_TOKENS = 10000

# --- Pre-compiled Regex Patterns for Chapter Extraction ---
# These patterns help identify titles even if their font size is not distinct.
CHAPTER_NUM_PATTERN = re.compile(r"^\s*(chapter|part|section)\s*([IVXLCDM\d]+)\s*[:.\-–—]?\s*", re.IGNORECASE)
# e.g., "1. Introduction", "12. Advanced Topics"
SUBCHAPTER_L1_PATTERN = re.compile(r"^\s*(\d{1,2})\s+([A-Z].{5,})", re.IGNORECASE)
# e.g., "1.1 Background", "10.2 Methods"
SUBCHAPTER_L2_PATTERN = re.compile(r"^\s*(\d{1,2}\.\d{1,2})\s+([A-Z].{5,})", re.IGNORECASE)
# e.g., "1.1.1 Data Collection"
SUBCHAPTER_L3_PATTERN = re.compile(r"^\s*(\d{1,2}\.\d{1,2}\.\d{1,2})\s+([A-Z].{5,})", re.IGNORECASE)

def _get_title_level(line_text: str, line_size: float, is_bold: bool, median_size: float) -> int:
    """
    Determines the heading level of a line based on text patterns and font style.
    Returns a level from 1 (highest) to 3, or 0 if not a title.
    """
    # Titles should not start with a lowercase letter or be a sentence fragment.
    if not line_text or line_text[0].islower():
        return 0

    # Level 1: "Chapter 1", "Part I"
    if CHAPTER_NUM_PATTERN.match(line_text):
        return 1
    # Level 3: "1.2.3 Some Detail"
    if SUBCHAPTER_L3_PATTERN.match(line_text):
        return 3
    # Level 2: "1.1 Background"
    if SUBCHAPTER_L2_PATTERN.match(line_text):
        return 2
    # Level 1.5 (could be level 1 or 2): "1 Some Topic"
    if SUBCHAPTER_L1_PATTERN.match(line_text):
        return 2  # Classify as level 2, often a main section within a chapter.

    # If no numbering, rely on font size and style, with word count limits.
    word_count = len(line_text.split())
    if is_bold:
        if line_size >= median_size * 1.6 and word_count < 12:
            return 1  # Very large and bold is likely a main chapter.
        if line_size >= median_size * 1.3 and word_count < 15:
            return 2  # Large and bold is likely a sub-chapter.
    
    # Sub-headings might not always be bold, so check size alone for level 3.
    if line_size >= median_size * 1.15 and word_count < 12:
        return 3
        
    return 0  # Not a title.

def extract_chapter_titles(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts potential chapter and sub-chapter titles from a PDF using enhanced heuristics,
    identifying titles based on font size, boldness, and common numbering patterns.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Could not open or read PDF '{pdf_path}': {e}")
        return []

    try:
        all_candidates = []
        num_pages = len(doc)

        for page_num in range(num_pages):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            sizes = []
            potential_titles = []
            height = page.rect.height

            if not height > 0:
                continue

            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        line_max_size = 0
                        is_bold = False
                        min_y = height  # Initialize to bottom

                        for span in line["spans"]:
                            sizes.append(span["size"])
                            line_text += span["text"]
                            if span["size"] > line_max_size:
                                line_max_size = span["size"]
                            if "bold" in span["font"].lower():
                                is_bold = True
                            min_y = min(min_y, span["origin"][1])

                        line_text = line_text.strip()
                        if line_text:
                            potential_titles.append({"text": line_text, "size": line_max_size, "bold": is_bold, "y": min_y})

            if not sizes:
                continue

            median_size = statistics.median(sizes) if sizes else 0
            
            # A blocklist of common, non-content-related substrings to ignore.
            IGNORE_SUBSTRINGS = {
                "table of contents", "general information", "programme", "abbreviations",
                "bibliography", "index", "future events", "required reading",
                "scientific committee", "governing board", "welcome message",
                "university", "department", "e-mail", "@", "et al",
                "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
            }

            for p in potential_titles:
                level = _get_title_level(p["text"], p["size"], p["bold"], median_size)
                # Relaxed y-position (top 40%) and length constraints to catch more sub-headings.
                if level > 0 and p["y"] < height * 0.4 and 3 < len(p["text"]) < 150:
                    
                    lower_text = p["text"].lower()

                    # Rule 1: Skip if it contains ignored substrings.
                    if any(sub in lower_text for sub in IGNORE_SUBSTRINGS):
                        continue
                    
                    # Rule 2: Skip if it matches non-title patterns (citations, list items, captions).
                    if (re.search(r'\(p\s*[<=>]', lower_text) or 
                        re.match(r'^\s*([•*+-]|\d+\.\d+)', p["text"]) or
                        re.match(r'^\s*(figure|fig\.|table|tbl\.)\s*\d+', lower_text)):
                        continue

                    # Rule 3: Filter out single words that are not fully uppercase.
                    if len(p["text"].split()) == 1 and not p["text"].isupper():
                        continue

                    all_candidates.append({
                        "page_number": page_num + 1,
                        "chapter_title": p["text"],
                        "level": level,
                        "y": p["y"],
                    })

        if not all_candidates:
            return []

        # --- Post-processing to refine the list of titles ---
        # Filter out likely running headers by checking for high frequency
        title_counts = Counter(c['chapter_title'] for c in all_candidates)
        frequent_titles = {
            title for title, count in title_counts.items()
            if count > max(5, num_pages * 0.05)  # Appears on >5 pages or >5% of pages
        }

        # Sort all candidates by page number, then by vertical position
        all_candidates.sort(key=lambda x: (x["page_number"], x["y"]))

        final_titles = []
        processed_titles_lower = set()
        for cand in all_candidates:
            title_text = cand["chapter_title"]
            if title_text in frequent_titles or title_text.lower() in processed_titles_lower:
                continue

            final_titles.append(cand)
            processed_titles_lower.add(title_text.lower())

        return final_titles
    finally:
        doc.close()


# --- Pydantic Models for LLM-based TOC Extraction ---

class Chapter(BaseModel):
    title: str = Field(description="The full, clean title of the chapter or section.")
    page: int = Field(description="The starting page number of the chapter.")

class TableOfContents(BaseModel):
    chapters: List[Chapter]


# --- Prompt for LLM-based TOC Extraction ---

LLM_TOC_EXTRACTION_PROMPT = """
You are an expert assistant specialized in parsing PDF documents.
Your task is to analyze the text from the initial pages of a medical textbook and extract its Table of Contents (TOC).

Follow these rules STRICTLY:
1.  Identify the main Table of Contents. You MUST ignore lists of figures, tables, contributors, and any other preliminary lists.
2.  Extract ONLY main chapter titles or major section titles (e.g., "Part 1", "Section A").
3.  You MUST NOT extract subsections (e.g., "1.1 Introduction", "2.3.4 Methods"). Focus exclusively on the top-level entries.
4.  Clean each title thoroughly. Remove all formatting artifacts like trailing dots or dashes. For entries like "Chapter 1: The Beginning", the title should be "The Beginning".
5.  Extract the corresponding starting page number for each title.
6.  Ignore entries that are clearly not chapters, such as "Preface", "Introduction", "Index", or "Appendix".
7.  If you cannot find a clear, structured Table of Contents with distinct chapters, return an empty list.
8.  Your final output must be a single, valid JSON object that strictly follows the provided schema.

Here is the text from the beginning of the book:
---
{text_from_pdf}
---

{format_instructions}
"""

def extract_chapters_with_llm(pdf_path: str, llm: BaseChatModel) -> Optional[List[Dict[str, Any]]]:
    """
    Attempts to extract a table of contents from the first few pages of a PDF using an LLM.
    
    Returns:
        A list of chapter dictionaries [{'chapter_title': ..., 'page_number': ...}] if successful,
        otherwise None.
    """
    logger.info(f"Attempting LLM-based TOC extraction for '{os.path.basename(pdf_path)}'...")
    try:
        with fitz.open(pdf_path) as doc:
            # Extract text from the first 20 pages, where a TOC is likely to be.
            num_pages_to_scan = min(20, len(doc))
            if num_pages_to_scan == 0:
                return None
            
            toc_text = "\n\n".join(doc.load_page(i).get_text("text") for i in range(num_pages_to_scan))

        if not toc_text.strip():
            logger.warning("No text found in the first pages for LLM extraction.")
            return None

        # --- Token Limiting Logic ---
        try:
            # Use tiktoken to handle token counting for LLM context
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(toc_text)
            if len(tokens) > MAX_TOC_TOKENS:
                logger.warning(
                    f"TOC text for '{os.path.basename(pdf_path)}' is too long ({len(tokens)} tokens). "
                    f"Truncating to {MAX_TOC_TOKENS} tokens."
                )
                truncated_tokens = tokens[:MAX_TOC_TOKENS]
                toc_text = encoding.decode(truncated_tokens)
        except Exception as e:
            logger.error(f"Could not truncate text with tiktoken, proceeding with original text. Error: {e}")

        parser = JsonOutputParser(pydantic_object=TableOfContents)
        prompt = ChatPromptTemplate.from_template(
            template=LLM_TOC_EXTRACTION_PROMPT,
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = prompt | llm | parser

        response = chain.invoke({"text_from_pdf": toc_text})
        extracted_chapters = response.get('chapters', [])
        
        if not extracted_chapters:
            logger.warning("LLM did not find a usable Table of Contents.")
            return None
        
        # --- Post-processing and Filtering ---
        # 1. Basic validation and cleaning
        validated_chapters = []
        for item in extracted_chapters:
            title = item.get('title', '').strip()
            page = item.get('page')
            # Basic validation: title exists, is not just a number, and page is valid.
            if title and not title.isdigit() and len(title) > 3 and page is not None and page > 0:
                validated_chapters.append({'title': title, 'page': page})
        
        if not validated_chapters:
            logger.warning("LLM response contained no valid chapters after initial validation.")
            return None

        # 2. Sort by page number to process in order
        validated_chapters.sort(key=lambda x: x['page'])

        # 3. De-duplicate and filter out entries that are on the same page
        final_chapters = []
        seen_titles = set()
        last_page = -1

        for chapter in validated_chapters:
            title_lower = chapter['title'].lower()
            
            if title_lower in seen_titles:
                continue

            # Heuristic: if a "chapter" starts on the same page as the previous one,
            # it's almost certainly a subsection that the LLM failed to ignore.
            if chapter['page'] <= last_page and last_page != -1:
                logger.info(f"Filtering potential subsection '{chapter['title']}' (page {chapter['page']}) as it does not start on a new page after the previous chapter (page {last_page}).")
                continue

            final_chapters.append({"chapter_title": chapter['title'], "page_number": chapter['page']})
            seen_titles.add(title_lower)
            last_page = chapter['page']

        if not final_chapters:
            logger.warning("LLM output was filtered down to zero valid chapters after post-processing.")
            return None

        chapters = final_chapters
        logger.info(f"LLM successfully extracted and filtered {len(chapters)} chapters.")
        return chapters
    except Exception as e:
        logger.error(f"LLM-based TOC extraction failed: {e}. Will fall back to heuristic method.", exc_info=False)
        return None