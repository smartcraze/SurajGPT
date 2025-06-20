from fastapi import FastAPI, Request
from pydantic import BaseModel
from rag_chain import get_rag_chain
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Suraj's Portfolio RAG API", version="1.0")

origins = ["https://www.surajv.me",
            "https://surajv.me",
            "http://localhost:3000",
            "http://localhost:8000",
            ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

rag_chain = get_rag_chain()

@app.post("/ask")
async def ask_question(query: Query):
    result = rag_chain.invoke({"input": query.question})
    return {
        "question": query.question,
        "answer": result["answer"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
