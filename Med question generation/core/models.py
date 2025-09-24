from pydantic import BaseModel, Field
from typing import List, Any

class MCQ(BaseModel):
    question: str = Field(description="The multiple-choice question.")
    options: List[str] = Field(description="A list of 4 possible answers.")
    correct_answer: str = Field(description="The correct answer from the options list.")

class MCQList(BaseModel):
    mcqs: List[MCQ]

class BookListResponse(BaseModel):
    books: List[str]

class ChapterListResponse(BaseModel):
    chapters: List[str]

class MCQRequest(BaseModel):
    book_filename: str
    chapter_title: str
    num_questions: int = 5

class MCQResponse(BaseModel):
    mcqs: List[MCQ]
    context: List[str]
    response_time: float

class Slide(BaseModel):
    title: str = Field(description="The title of the presentation slide.")
    bullets: List[str] = Field(description="A list of brief bullet points for the slide content.")
    speaker_notes: str = Field(default="", description="Concise speaker notes for the presenter, explaining the slide's content in more detail.")

class PresentationList(BaseModel):
    slides: List[Slide]

class PresentationRequest(BaseModel):
    book_filename: str
    chapter_title: str

class PresentationResponse(BaseModel):
    slides: List[Slide]
    context: List[str]
    response_time: float
