from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel
from typing import List

app = FastAPI()
model = SentenceTransformer("paraphrase-MiniLM-L12-v2")

class CompareRequest(BaseModel):
    source: str
    candidates: List[str]

@app.post("/compare")
def compare(req: CompareRequest):
    source_embedding = model.encode(req.source, convert_to_tensor=True)
    candidate_embeddings = model.encode(req.candidates, convert_to_tensor=True)
    scores = util.cos_sim(source_embedding, candidate_embeddings)[0].tolist()
    return { "scores": scores }
