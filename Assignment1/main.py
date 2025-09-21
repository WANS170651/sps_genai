from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel
import spacy.cli
import spacy

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)

# spaCy model
spacy.cli.download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    input_word: str


class SimilarityRequest(BaseModel):
    word1: str
    word2: str


class SentenceEmbeddingRequest(BaseModel):
    sentence: str


class SentenceSimilarityRequest(BaseModel):
    sentence1: str
    sentence2: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length
    )
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    vector = nlp(request.input_word).vector
    return {"embedding": vector.tolist()}


@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    score = nlp(request.word1).similarity(nlp(request.word2))
    return {"similarity": float(score)}


@app.post("/sentence-embedding")
def get_sentence_embedding(request: SentenceEmbeddingRequest):
    vector = nlp(request.sentence).vector
    return {"embedding": vector.tolist()}


@app.post("/sentence-similarity")
def get_sentence_similarity(request: SentenceSimilarityRequest):
    score = nlp(request.sentence1).similarity(nlp(request.sentence2))
    return {"similarity": float(score)}
