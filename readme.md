# Medical Content Generation Tool

This web application helps medical students and educators by automatically generating Multiple Choice Questions (MCQs) and presentation slides from medical textbooks in PDF format.

## Overview

The tool provides a simple web interface where users can select a book and a specific chapter. Based on the selection, it can:
1.  Generate a specified number of MCQs with correct answers.
2.  Generate a structured presentation with titles, bullet points, and speaker notes.

The generated presentation can be downloaded directly as a `.pptx` file.

## Features

- **Book & Chapter Selection**: Dynamically loads available books and their chapters.
- **MCQ Generation**: Creates relevant multiple-choice questions from the text.
- **Presentation Generation**: Automatically creates presentation slides with key points and speaker notes.
- **Downloadable Presentations**: Download generated presentations in PowerPoint (.pptx) format.
- **Context Viewer**: Shows the source text used for generating the content, providing transparency and a way to verify the information.

## Tech Stack

- **Backend**: FastAPI
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **RAG Pipeline**: LangChain
- **Vector Store**: FAISS
- **Presentation Generation (Frontend)**: PptxGenJS
- **LLM**: llama-4-scout-17b-16e-instruct( via Groq inference api)
- **Embeddings**: Sentence Transformer (all-MiniLM-L12-v2) from Hugging Face

## Project Setup

Follow these instructions to get the project running on your local machine.

### Prerequisites

- Python 3.10

### 1. Clone the Repository

```bash
git clone https://github.com/mahmed201379-droid/RAG-based-medical-content-generator.git
```
```bash
cd RAG-based-medical-content-generator
```

### 2. Backend Setup

First, set up a Python virtual environment.

Open terminal into the project directory.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

Next, install the required Python packages. In terminal:

```bash
pip install -r requirements.txt
```

### 3. Environment Variables

The application  require environment variables . Create a `.env` file in the project root and add your keys.

**`.env`**:
```
HF_TOKEN="Your_api_key"
GROQ_API_KEY="Your_api_key"
```

### 4. Running the Application

1.  **Start the FastAPI Backend**:
    Open terminal in project directory and run the backend server.

    ```bash
    cd "Med question generation"
    ```

    ```bash
    uvicorn core.main:app --reload --port 8000
    ```

    The backend server will start, typically on `http://127.0.0.1:8000`.

2.  **Open the Frontend**:
    Open new terminal window:
    ```bash
    cd "Med question generation"
    ```
    then:

    ```bash
    python -m http.server 8080
    ```
    Then open the link: `http://localhost:8080` from any browser.



## Usage

1.  Open the application in your browser.
2.  The list of available books will load automatically. **Note**: The initial loading might take a few minutes if the backend is creating vector stores for the books for the first time.
3.  Select a book from the dropdown menu.
4.  Once a book is selected, the chapter dropdown will be populated. Select a chapter.
5.  **For MCQs**: Enter the number of questions you want and click "Generate MCQs".
6.  **For Presentations**: Click "Make Presentation".
7.  The results will be displayed on the page. For presentations, a "Download as PowerPoint" button will appear.
