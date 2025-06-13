import os
import json
import sqlite3
import numpy as np
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import aiohttp
import asyncio
import logging
import traceback
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
import base64

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "knowledge_base.db"
SIMILARITY_THRESHOLD = 0.68
MAX_RESULTS = 10
MAX_CONTEXT_CHUNKS = 4

# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def cosine_similarity(vec1, vec2):
    try:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    except Exception as e:
        logger.error(f"Error in cosine_similarity: {e}")
        return 0.0

async def get_embedding(text, max_retries=3):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not set")
    retries = 0
    while retries < max_retries:
        try:
            async with aiohttp.ClientSession() as session:
                response = await session.post(
                    "https://aipipe.org/openai/v1/embeddings",
                    headers={"Authorization": API_KEY, "Content-Type": "application/json"},
                    json={"model": "text-embedding-3-small", "input": text}
                )
                if response.status == 200:
                    result = await response.json()
                    return result["data"][0]["embedding"]
                elif response.status == 429:
                    await asyncio.sleep(3 * (retries + 1))
                    retries += 1
                else:
                    raise HTTPException(status_code=response.status, detail=await response.text())
        except Exception as e:
            logger.error(str(e))
            retries += 1
    raise HTTPException(status_code=500, detail="Failed to get embedding")

async def find_similar_content(query_embedding, conn):
    results = []
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM discourse_chunks WHERE embedding IS NOT NULL")
    discourse_chunks = cursor.fetchall()

    for chunk in discourse_chunks:
        try:
            embedding = json.loads(chunk["embedding"])
            sim = cosine_similarity(query_embedding, embedding)
            if sim >= SIMILARITY_THRESHOLD:
                url = chunk["url"]
                if not url.startswith("http"):
                    url = f"https://discourse.onlinedegree.iitm.ac.in/t/{url}"
                results.append({
                    "source": "discourse",
                    "url": url,
                    "content": chunk["content"],
                    "similarity": sim
                })
        except:
            continue

    cursor.execute("SELECT * FROM markdown_chunks WHERE embedding IS NOT NULL")
    markdown_chunks = cursor.fetchall()

    for chunk in markdown_chunks:
        try:
            embedding = json.loads(chunk["embedding"])
            sim = cosine_similarity(query_embedding, embedding)
            if sim >= SIMILARITY_THRESHOLD:
                url = chunk["original_url"]
                if not url.startswith("http"):
                    url = f"https://docs.onlinedegree.iitm.ac.in/{chunk['doc_title']}"
                results.append({
                    "source": "markdown",
                    "url": url,
                    "content": chunk["content"],
                    "similarity": sim
                })
        except:
            continue

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:MAX_RESULTS]

async def generate_answer(question, relevant_results):
    context = ""
    for res in relevant_results:
        context += f"\n\nSource (URL: {res['url']}):\n{res['content'][:1500]}"
    prompt = f"""Answer based ONLY on the context. If not enough info, say you don't know.
Context: {context}
Question: {question}

Return in this format:
1. Answer
2. Sources:
1. URL: [url], Text: [summary]
2. URL: [url], Text: [summary]
"""
    async with aiohttp.ClientSession() as session:
        response = await session.post(
            "https://aipipe.org/openai/v1/chat/completions",
            headers={"Authorization": API_KEY, "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }
        )
        if response.status == 200:
            result = await response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise HTTPException(status_code=response.status, detail=await response.text())

def parse_llm_response(response):
    parts = response.split("Sources:")
    answer = parts[0].strip()
    links = []
    if len(parts) > 1:
        lines = parts[1].split("\n")
        for line in lines:
            match = re.search(r'URL:\s*(\S+),\s*Text:\s*(.+)', line)
            if match:
                links.append({
                    "url": match.group(1).strip(),
                    "text": match.group(2).strip()
                })
    return {"answer": answer, "links": links}

@app.post("/api/")
async def query(request: QueryRequest):
    try:
        conn = get_db_connection()
        query_embedding = await get_embedding(request.question)
        results = await find_similar_content(query_embedding, conn)
        llm_response = await generate_answer(request.question, results)
        parsed = parse_llm_response(llm_response)
        return parsed
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
async def health_check():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
        discourse_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
        markdown_count = cursor.fetchone()[0]
        return {
            "status": "healthy",
            "discourse_chunks": discourse_count,
            "markdown_chunks": markdown_count,
            "api_key_set": bool(API_KEY)
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# LOCAL RUN
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

# ✅ For Vercel: Mangum handler
from mangum import Mangum
handler = Mangum(app)
