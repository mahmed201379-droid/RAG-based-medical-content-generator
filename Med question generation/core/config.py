import os

# --- PATHS ---
PDF_DIRECTORY = "books"
VECTOR_STORE_DIRECTORY = "vector_stores"

# --- MODELS ---
EMBEDDING_MODEL = "all-MiniLM-L12-v2"
# The model name must be one of the models available on Groq.
# "meta-llama/llama-4-scout-17b-16e-instruct" is not a valid Groq model.
LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


BM25_K = 3
FAISS_K = 3
ENSEMBLE_WEIGHTS = [0.5, 0.5]

# --- PROMPTS ---
# To improve reliability and reduce retries (which add latency), the prompt is very specific.
# A final instruction has been added to further emphasize the need for valid JSON output.
MCQ_PROMPT_TEMPLATE = """
You are an expert in creating multiple-choice questions from medical textbooks.
Based on the provided context, please generate {num_questions} multiple-choice questions.

Each mcq question must have:
1. A clear question.
2. Four options, formatted exactly like this:
   A. Option text
   B. Option text
   C. Option text
   D. Option text
3. The correct answer, which must be one of the provided options.

Context:
---
{context}
---

Ensure your output is a single, valid JSON object that strictly follows the format instructions.

{format_instructions}
"""

PRESENTATION_PROMPT_TEMPLATE = """
You are an expert in creating professional and pedagogically sound presentations from medical textbooks.
Based on the provided context, please generate a presentation with multiple slides that logically progress through the topic.

For each slide, you must generate:
1. A clear and concise title.
2. A list of 'bullets'. These should be brief and easy to read on a slide.
3. A set of 'speaker_notes'. These notes are for the presenter and should contain a more detailed explanation, additional context, or talking points related to the bullet points on the slide.

A good presentation structure often includes:
- An introduction or overview.
- Key concepts or mechanisms (e.g., pathophysiology).
- Clinical relevance (e.g., diagnosis, symptoms).
- Management or treatment.
- A concluding summary.

Please try to follow a logical flow.

Context:
---
{context}
---

Generate a presentation that covers the main topics in the context, following the structure and format described.
Ensure your output is a single, valid JSON object that strictly follows the format instructions.

{format_instructions}
"""
