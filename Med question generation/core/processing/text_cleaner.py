import re
import string
import logging

logger = logging.getLogger(__name__)

# --- Pre-compiled Regex Patterns for Preprocessing ---
# Compile regex patterns once at the module level for performance.

# 1. General patterns to remove
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

# 2. Patterns for splitting concatenated reference lists
REF_SPLIT_NUM_PATTERN = re.compile(r'(?<=\.)\s+(?=\d{1,3}\.\s)')
REF_SPLIT_AUTHOR_PATTERN = re.compile(r'(?<=\.)\s+(?=[A-Z][a-z]{2,},\s[A-Z]\.)')
REF_SPLIT_MEDICAL_PATTERN = re.compile(r'(?<=\.)\s+(?=[A-Z][a-z]{2,}\s[A-Z]\s)')

# 3. Patterns for identifying and removing sections/lines
REFERENCES_HEADER_PATTERN = re.compile(r"^\s*(references|reference list|bibliography|works cited)\s*$", re.IGNORECASE)
# This pattern is for TOCs with dot leaders, e.g., "Chapter 1 .......... 5"
TOC_DOT_LEADER_PATTERN = re.compile(r".{5,}[\. ]{5,}\s*([IVXLCDM]+|\d+)\s*$", re.IGNORECASE)
FOOTER_PATTERN = re.compile(r"^\s*([IVXLCDM]+|\d+)\s*$", re.IGNORECASE)
FOOTER_HEADER_PATTERN = re.compile(r"^\s*(continued|cont'd|page \d+|vol\. \d+|issue \d+)\s*$", re.IGNORECASE)
# New patterns for detecting other TOC formats, like those with author lists.
TOC_HEADING_PATTERN = re.compile(r'^\s*(contents|preface|index|contributors|dedication|appendix)\s*$', re.IGNORECASE)
TOC_ENTRY_WITH_AUTHORS_PATTERN = re.compile(
    r'^\s*\d{1,2}\s+'  # Starts with a chapter number (e.g., "1 ", "23 ")
    r'.*?'             # Non-greedy match for the title
    # Contains a common academic/medical credential, indicating an author list
    r'\b(MD|PhD|FACS|ScD|Hon|MS|MBA|RN|WOCN|FRCS)\b',
    re.IGNORECASE
)
# 4. Collection of patterns to identify a reference line
REFERENCE_LINE_PATTERNS = [
    # e.g., "... (2010); 23(8): 956-961."
    re.compile(r".*(\d{4})\s?;.*:\s?\d+([\-–]\d+)?\.?$", re.IGNORECASE),
    # Numbered with brackets: "[1] Author..."
    re.compile(r"^\s*\[\d{1,3}\]\s.*", re.IGNORECASE),
    # In-text style: "Author (2017)... 36-48." or "...(2013)... 187."
    re.compile(r".*\(\d{4}\).*\d+([\-–]\d+)?\.?$", re.IGNORECASE),
    # Numbered with dot: "22. Author..."
    re.compile(r"^\s*\d{1,3}\.\s.*", re.IGNORECASE),
    # Journal volume/issue: "... Year;vol(issue):pages."
    re.compile(r".*\d{44};\d+\(?\d*\)?\:\d+[\-–]\d+\.?$", re.IGNORECASE),
    # Lines with "et al." and year: common in citations
    re.compile(r".*et al\..*\(?\d{4}\)?.*", re.IGNORECASE),
    # DOI lines: "DOI: 10.XXXX/..."
    re.compile(r".*doi:\s*10\.\d{4}/.*", re.IGNORECASE),
    # PMID lines: "PMID: 12345678"
    re.compile(r".*pmid:\s*\d+.*", re.IGNORECASE),
    # PubMed or similar IDs
    re.compile(r".*pubmed:\s*\d+.*", re.IGNORECASE),
]

# 5. Pattern for collapsing multiple newlines
EXCESSIVE_NEWLINES_PATTERN = re.compile(r'\n{3,}')

# --- Pre-compiled Regex Patterns for Query Cleaning ---
# Removes common prefixes to create a better search query.
QUERY_CLEANING_PATTERN = re.compile(
    r"^\s*(chapter|part|section|no\.|figure|fig\.|table|tbl\.)\s*[\d.ivxclmd\s]*[:.\-–—]?\s*",
    re.IGNORECASE
)

def _is_reference_line(line: str) -> bool:
    """Check if a line matches any of the compiled reference patterns."""
    return any(p.match(line) for p in REFERENCE_LINE_PATTERNS)

def _is_toc_line(line: str) -> bool:
    """
    Checks if a line is likely part of a table of contents by matching against
    several common TOC formats.
    """
    # Format 1: "Title ........ 123"
    if TOC_DOT_LEADER_PATTERN.match(line):
        return True
    # Format 2: "Contents", "Preface", etc.
    if TOC_HEADING_PATTERN.match(line):
        return True
    # Format 3: "1. Chapter Title Author, MD"
    if TOC_ENTRY_WITH_AUTHORS_PATTERN.match(line):
        return True
    return False

def preprocess_document_content(text: str) -> str:
    """
    Cleans the extracted text from a PDF page to remove emails, URLs,
    reference sections, and table of contents pages.
    """
    # 0. Remove non-printable characters that can interfere with regex
    text = "".join(filter(lambda x: x in string.printable, text))

    # 1. Remove URLs and email addresses from the entire text first
    text = URL_PATTERN.sub('', text)
    text = EMAIL_PATTERN.sub('', text)

    # 2. Pre-emptively split reference lists that are concatenated.
    # Split on numbered references (e.g., ". 1. " or ". 23. ")
    text = REF_SPLIT_NUM_PATTERN.sub('\n', text)
    # Split on author-starting patterns (e.g., ". Name, I.")
    text = REF_SPLIT_AUTHOR_PATTERN.sub('\n', text)
    # Additional split for medical styles without dot (e.g., "Author A Year;vol:pages Author B")
    text = REF_SPLIT_MEDICAL_PATTERN.sub('\n', text)

    try:
        lines = text.split('\n')
        cleaned_lines = []
        in_references_section = False
        reference_density = 0  # Heuristic: track consecutive reference-like lines

        # Heuristic: If a high percentage of lines on a page look like TOC entries,
        # discard the entire page. This is more robust than a line-by-line check.
        toc_line_count = sum(1 for line in lines if _is_toc_line(line.strip()))
        if len(lines) > 5 and (toc_line_count / len(lines)) > 0.4:
            logger.info("Detected a page with high density of TOC-like lines. Discarding entire page content.")
            return ""

        for line in lines:
            stripped_line = line.strip()

            if not stripped_line:
                continue

            if REFERENCES_HEADER_PATTERN.match(stripped_line):
                in_references_section = True
                continue  # Skip the header itself

            is_ref = _is_reference_line(stripped_line)

            if is_ref:
                reference_density += 1
            else:
                reference_density = 0  # Reset if non-reference

            # Skip if in references, or matches patterns, or high density (e.g., >2 consecutive)
            if (in_references_section or
                _is_toc_line(stripped_line) or # Check individual lines as well
                is_ref or
                FOOTER_PATTERN.match(stripped_line) or
                FOOTER_HEADER_PATTERN.match(stripped_line) or
                reference_density > 2):  # Skip blocks of references even without header
                continue

            cleaned_lines.append(line)

        # Join the cleaned lines and remove excessive newlines
        cleaned_text = "\n".join(cleaned_lines)
        cleaned_text = EXCESSIVE_NEWLINES_PATTERN.sub('\n\n', cleaned_text)  # Collapse > 2 newlines
        
        return cleaned_text.strip()
    except Exception as e:
        logger.error(f"Error preprocessing document content: {e}")
        return text

def clean_chapter_title_for_query(title: str) -> str:
    """
    Cleans a chapter title to create a better search query by removing
    common non-descriptive prefixes.
    e.g., "Chapter 15: Advanced Surgical Techniques" -> "Advanced Surgical Techniques"
    e.g., "1.1. Introduction" -> "Introduction"
    """
    # Remove common prefixes like "Chapter 1", "Section 2.3", etc.
    cleaned_title = QUERY_CLEANING_PATTERN.sub("", title)
    
    # Remove leading/trailing whitespace and punctuation that might be left over
    cleaned_title = cleaned_title.strip(" .:;-–—")
    
    return cleaned_title