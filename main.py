from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI()

class QuestionInput(BaseModel):
    question: str
    image: Optional[str] = None

@app.post("/api")
async def answer_question(payload: QuestionInput):
    # Dummy logic for now
    if "gpt" in payload.question.lower():
        return {
            "answer": "Use `gpt-3.5-turbo-0125` as required.",
            "links": [
                {
                    "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/4",
                    "text": "Use the model that’s mentioned in the question."
                }
            ]
        }
    return {
        "answer": "I'm a virtual TA and will get smarter soon!",
        "links": []
    }
