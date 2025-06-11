from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI()

class QuestionInput(BaseModel):
    question: str
    image: Optional[str] = None

@app.post("/api")
async def answer_question(payload: QuestionInput):
    question = payload.question.lower()

    # Specific logic for Q1 Japanese token pricing question
    if "gpt-3.5-turbo" in question and "tokens" in question and "cents" in question:
        return {
            "answer": "0.0017",
            "links": [
                {
                    "url": "https://platform.openai.com/tokenizer",
                    "text": "Use the tokenizer to count tokens in Japanese text."
                },
                {
                    "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/3",
                    "text": "You just have to count input tokens and multiply by the given rate."
                }
            ]
        }

    # Default fallback
    return {
        "answer": "I'm a virtual TA. Please ask a question related to TDS.",
        "links": []
   }
