// API URLs (using the same as in the FastAPI backend)
const API_BASE_URL = 'http://127.0.0.1:8000'; // Adjust if your backend is on a different host/port
const LIST_BOOKS_API_URL = `${API_BASE_URL}/list-books`;
const LIST_CHAPTERS_API_URL = `${API_BASE_URL}/list-chapters`;
const MCQ_API_URL = `${API_BASE_URL}/generate-mcq`;
const PRESENTATION_API_URL = `${API_BASE_URL}/generate-presentation`;

// Elements
const bookSelect = document.getElementById('bookSelect');
const chapterSelect = document.getElementById('chapterSelect');
const numQuestions = document.getElementById('numQuestions');
const generateMcqsBtn = document.getElementById('generateMcqsBtn');
const generatePresentationBtn = document.getElementById('generatePresentationBtn');
const spinner = document.getElementById('spinner');
const errorMessage = document.getElementById('errorMessage');
const resultsSection = document.getElementById('resultsSection');
const mcqResults = document.getElementById('mcqResults');
const presentationResults = document.getElementById('presentationResults');

// Load available books on page load
async function fetchWithTimeout(resource, options = {}, timeout = 300000) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);

    try {
        const response = await fetch(resource, {
            ...options,
            signal: controller.signal
        });
        return response;
    } catch (error) {
        if (error.name === 'AbortError') {
            throw new Error('Request timed out');
        }
        throw error;
    } finally {
        clearTimeout(id);
    }
}

async function loadBooks() {
    try {
        // Use a long timeout for the initial book loading, as the backend might be creating vector stores.
        const response = await fetchWithTimeout(LIST_BOOKS_API_URL, {}, 300000);
        if (!response.ok) {
            if (response.status === 503) {
                showError('Backend is initializing. Please wait a moment and refresh.');
            } else {
                throw new Error(`Failed to fetch books (Status: ${response.status})`);
            }
            return; // Stop execution if response is not ok
        }
        const data = await response.json();
        bookSelect.innerHTML = '<option value="">Choose a book...</option>';
        data.books.forEach(book => {
            const option = document.createElement('option');
            option.value = book;
            option.textContent = book;
            bookSelect.appendChild(option);
        });
    } catch (error) {
        showError('Could not fetch book list: ' + error.message);
    }
}

// Load chapters when book is selected
bookSelect.addEventListener('change', async () => {
    const book = bookSelect.value;
    chapterSelect.disabled = true;
    chapterSelect.innerHTML = '<option value="">Choose a chapter...</option>';
    if (!book) return;
    try {
        const response = await fetchWithTimeout(`${LIST_CHAPTERS_API_URL}/${book}`, {}, 30000);
        if (!response.ok) throw new Error('Failed to fetch chapters');
        const data = await response.json();
        data.chapters.forEach(chapter => {
            const option = document.createElement('option');
            option.value = chapter;
            option.textContent = chapter;
            chapterSelect.appendChild(option);
        });
        chapterSelect.disabled = false;
    } catch (error) {
        showError('Could not fetch chapters: ' + error.message);
    }
});

// Generate MCQs
generateMcqsBtn.addEventListener('click', async () => {
    const book = bookSelect.value;
    const chapter = chapterSelect.value;
    const num = parseInt(numQuestions.value);
    if (!book || !chapter) return showError('Please select a book and chapter.');
    clearResults();
    showSpinner(true);
    try {
        const response = await fetchWithTimeout(MCQ_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ book_filename: book, chapter_title: chapter, num_questions: num })
        });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to generate MCQs');
        }
        const data = await response.json();
        displayMcqs(data);
    } catch (error) {
        showError('Error generating MCQs: ' + error.message);
    } finally {
        showSpinner(false);
    }
});

// Generate Presentation
generatePresentationBtn.addEventListener('click', async () => {
    const book = bookSelect.value;
    const chapter = chapterSelect.value;
    if (!book || !chapter) return showError('Please select a book and chapter.');
    clearResults();
    showSpinner(true);
    try {
        const response = await fetchWithTimeout(PRESENTATION_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ book_filename: book, chapter_title: chapter })
        });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to generate presentation');
        }
        const data = await response.json();
        displayPresentation(data, book, chapter);
    } catch (error) {
        showError('Error generating presentation: ' + error.message);
    } finally {
        showSpinner(false);
    }
});

// Display MCQs
function displayMcqs(data) {
    resultsSection.style.display = 'block';
    mcqResults.innerHTML = `
        <div class="alert alert-success">
            Successfully generated ${data.mcqs.length} MCQs in ${data.response_time.toFixed(2)} seconds.
        </div>
        <h3>Generated Questions</h3>
    `;
    data.mcqs.forEach((mcq, i) => {
        const container = document.createElement('div');
        container.className = 'mcq-container';
        container.innerHTML = `
            <strong>Question ${i+1}:</strong> ${mcq.question}<br>
            ${mcq.options.join('<br>')}
            <div class="mt-2">
                <button class="btn btn-secondary btn-sm show-answer">Show Answer</button>
                <div class="answer mt-2">Correct Answer: ${mcq.correct_answer}</div>
            </div>
        `;
        mcqResults.appendChild(container);
        container.querySelector('.show-answer').addEventListener('click', () => {
            container.querySelector('.answer').style.display = 'block';
        });
    });
    // Context
    const contextExpander = createContextExpander(data.context, "mcq-context");
    mcqResults.appendChild(contextExpander);
}

// Display Presentation
function displayPresentation(data, book, chapter) {
    resultsSection.style.display = 'block';
    presentationResults.innerHTML = `
        <div class="alert alert-success">
            Successfully generated a presentation in ${data.response_time.toFixed(2)} seconds.
        </div>
        <h3>Generated Presentation</h3>
        <button id="downloadPptx" class="btn btn-primary mb-3">Download as PowerPoint (.pptx)</button>
    `;
    data.slides.forEach((slide, i) => {
        const container = document.createElement('div');
        container.className = 'slide-container';
        container.innerHTML = `
            <h5>Slide ${i + 1}: ${slide.title || 'Untitled'}</h5>
            <ul>${slide.bullets.map(b => `<li>${b}</li>`).join('')}</ul>
            ${slide.speaker_notes ? `
                <div class="mt-2">
                    <button class="btn btn-outline-secondary btn-sm" type="button" data-bs-toggle="collapse" data-bs-target="#notes-${i}" aria-expanded="false">
                        Show Speaker Notes
                    </button>
                    <div class="collapse mt-1" id="notes-${i}">
                        <div class="card card-body bg-light">
                            <small>${slide.speaker_notes}</small>
                        </div>
                    </div>
                </div>
            ` : ''}
        `;
        presentationResults.appendChild(container);
    });
    // Context
    const contextExpander = createContextExpander(data.context, "presentation-context");
    presentationResults.appendChild(contextExpander);

    // Download PPTX
    document.getElementById('downloadPptx').addEventListener('click', () => {
        createAndDownloadPptx(data.slides, book, chapter);
    });
}

// Create Context Expander
function createContextExpander(context, idSuffix) {
    const collapseId = `contextCollapse-${idSuffix}`;
    const expander = document.createElement('div');
    expander.className = 'context-expander';
    expander.innerHTML = `
        <button class="btn btn-info" type="button" data-bs-toggle="collapse" data-bs-target="#${collapseId}" aria-expanded="false" aria-controls="${collapseId}">
            View Context Used for Generation
        </button>
        <div class="collapse mt-2" id="${collapseId}"></div>
    `;
    const collapse = expander.querySelector(`#${collapseId}`);
    (context || []).forEach((doc, i) => {
        collapse.innerHTML += `
            <strong>Source Document ${i+1}</strong><br>
            ${doc}<hr>
        `;
    });
    return expander;
}



// Create and Download PPTX using PptxGenJS
function createAndDownloadPptx(slidesData, bookName, chapterName) {
    const pptx = new PptxGenJS();
    pptx.layout = 'LAYOUT_WIDE'; // 16:9

    const sanitizedBook = bookName.replace('.pdf', '');
    const sanitizedChapter = chapterName.replace(/[\\/*?:"<>|]/g, '');
    const fileName = `${sanitizedBook}_${sanitizedChapter}.pptx`.replace(/ /g, '_');

    // Title Slide
    const titleSlide = pptx.addSlide();
    titleSlide.addText(chapterName, {
        x: 0.5, y: 2.5, w: '90%', h: 1.5,
        fontSize: 44, bold: true, align: 'center',
    });
    titleSlide.addText(`Source: ${sanitizedBook}`, {
        x: 0.5, y: 4.0, w: '90%', h: 1,
        fontSize: 24, italic: true, align: 'center',
    });

    // Content Slides
    slidesData.forEach((slideData, i) => {
        const slide = pptx.addSlide();

        // Add speaker notes if they exist
        if (slideData.speaker_notes) {
            slide.addNotes(slideData.speaker_notes);
        }

        // Footer
        slide.addText(chapterName, {
            x: 0.5, y: 7.0, w: '85%', h: 0.4,
            fontSize: 11, color: '808080', align: 'left'
        });

        // Slide Number
        slide.addText(`${i + 1} / ${slidesData.length}`, {
            x: 12.0, y: 7.0, w: 1.0, h: 0.4,
            fontSize: 11, color: '808080', align: 'right'
        });

        // Title
        slide.addText(slideData.title || 'Untitled Slide', {
            x: 0.5, y: 0.25, w: '90%', h: 1.25,
            fontSize: 36, bold: true, align: 'left'
        });

        // Bullets
        const bullets = slideData.bullets || [];

        if (bullets.length > 0) {
            const bulletObjects = bullets.map(b => ({
                text: b,
                options: { bullet: true, fontSize: 20, color: "363636" }
            }));

            // ✅ Only one call, with array of objects as the first argument
            slide.addText(bulletObjects, {
                x: 0.75, y: 2, w: '85%', h: 5.0,
                valign: 'top'
                                
            });
        }
    });

    // Thank You Slide
    const thankYouSlide = pptx.addSlide();
    thankYouSlide.addText("Thank You", {
        x: 0, y: 0, w: '100%', h: '100%',
        align: 'center', valign: 'middle',
        fontSize: 44, bold: true
    });

    pptx.writeFile({ fileName });
}



// Utility Functions
function showSpinner(show) {
    spinner.style.display = show ? 'block' : 'none';
    generateMcqsBtn.disabled = show;
    generatePresentationBtn.disabled = show;
}

function showError(msg) {
    errorMessage.textContent = msg;
    errorMessage.style.display = 'block';
    setTimeout(() => { errorMessage.style.display = 'none'; }, 5000);
}

function clearResults() {
    mcqResults.innerHTML = '';
    presentationResults.innerHTML = '';
    resultsSection.style.display = 'none';
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    try {
        loadBooks();
    } catch (error) {
        showError("Failed to initialize the application: " + error.message);
    }
});