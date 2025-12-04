# app/main.py

from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from .models import EmbeddingRequest, EmbeddingResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Менеджер жизненного цикла FastAPI.
    Выполняет загрузку модели при старте приложения.
    """
    print("INFO:     Загрузка модели ai-forever/FRIDA...")
    # Загружаем модель и сохраняем ее в состоянии приложения
    app.state.model = SentenceTransformer("ai-forever/FRIDA")
    print("INFO:     Модель успешно загружена.")
    yield
    # Код для очистки ресурсов при остановке (если необходимо)
    print("INFO:     Приложение останавливается.")


# Инициализируем FastAPI приложение с менеджером жизненного цикла
app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Проверяет работоспособность сервиса.
    """
    return {"status": "ok"}



@app.post("/embed", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest) -> Dict:
    """
    Принимает список текстов и возвращает их векторные представления.
    """
    # Получаем эмбеддинги для списка текстов
    embeddings = app.state.model.encode(request.texts)

    # Преобразуем результат в список и возвращаем в формате EmbeddingResponse
    response = {"embeddings": embeddings.tolist()}
    return response