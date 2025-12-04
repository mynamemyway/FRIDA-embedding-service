# app/models.py

from typing import List

from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    """
    Pydantic-модель для тела запроса на получение эмбеддингов.

    Ожидает JSON-объект с ключом 'texts', значением которого
    является список строк.
    """
    texts: List[str]


class EmbeddingResponse(BaseModel):
    """
    Pydantic-модель для тела ответа с эмбеддингами.

    Определяет структуру ответа, который содержит ключ 'embeddings'
    со списком векторов (списков float).
    """
    embeddings: List[List[float]]