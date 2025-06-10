import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

models_cache = {}
config = {}


def load_config():
    config_path = Path("api/config.json")
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)


def load_model(model_path: Path, model_type: str):
    try:
        if model_type == "catboost":
            model = CatBoostClassifier()
            cbm_file = model_path / "catboost_model.cbm"
            if cbm_file.exists():
                model.load_model(str(cbm_file))
                return model
            else:
                logger.error(f"Файл CatBoost модели не найден: {cbm_file}")
                return None
        elif model_type == "lgbm":
            pkl_file = model_path / "lightgbm_model.pkl"
            if pkl_file.exists():
                return joblib.load(pkl_file)
            else:
                logger.error(f"Файл LightGBM модели не найден: {pkl_file}")
                return None
        elif model_type == "rf":
            pkl_file = model_path / "random_forest_model.pkl"
            if pkl_file.exists():
                return joblib.load(pkl_file)
            else:
                logger.error(f"Файл Random Forest модели не найден: {pkl_file}")
                return None
        else:
            logger.error(f"Неизвестный тип модели: {model_type}")
            return None
    except Exception as e:
        logger.error(f"Ошибка загрузки модели {model_path}: {e}")
        return None


def load_models():
    global models_cache, config
    config = load_config()

    for model_name, model_base_path in config["model_paths"].items():
        models_cache[model_name] = {}

        for dataset in config["datasets"]:
            model_path = Path(model_base_path) / dataset

            if model_path.exists():
                model = load_model(model_path, model_name)
                if model is not None:
                    models_cache[model_name][dataset] = model
                    logger.info(f"Загружена модель: {model_name}/{dataset}")
                else:
                    logger.warning(f"Не удалось загрузить модель: {model_name}/{dataset}")
            else:
                logger.warning(f"Директория модели не найдена: {model_path}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Startup
    logger.info("Запуск API сервиса...")
    load_models()
    logger.info("API сервис готов к работе")
    yield
    # Shutdown
    logger.info("Завершение работы API сервиса...")
    models_cache.clear()
    logger.info("API сервис остановлен")


app = FastAPI(
    title="DNA Sequence Classification API",
    description="API для классификации ДНК-последовательностей с использованием различных ML моделей",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="Набор признаков")
    dataset: str = Field(
        ...,
        description="Тип датасета",
        pattern=r"^(enhancers|promoter_all|splice_sites_all|H3K9me3|H4K20me1)$",
    )


class PredictionResponse(BaseModel):
    prediction: List[float] = Field(..., description="Предсказание модели")
    dataset: str = Field(..., description="Тип датасета")
    model_version: str = Field(..., description="Версия модели")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Статус сервиса")
    loaded_models: Dict[str, List[str]] = Field(..., description="Загруженные модели")


class ModelInfo(BaseModel):
    model_version: str = Field(..., description="Версия модели")
    datasets: List[str] = Field(..., description="Доступные датасеты")
    description: str = Field(..., description="Описание модели")
    file_extension: str = Field(..., description="Расширение файла модели")


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "DNA Sequence Classification API",
        "version": "1.0.0",
        "available_endpoints": [
            "/health",
            "/predict",
            "/models",
            "/models/{model_version}/datasets",
            "/docs",
        ],
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    loaded_models = {}
    for version, datasets in models_cache.items():
        loaded_models[version] = list(datasets.keys())

    return HealthResponse(status="healthy", loaded_models=loaded_models)


def preprocess_features(features: List[float]):
    features_array = np.array(features).reshape(1, -1)
    l2_norm = np.linalg.norm(features_array, axis=1)
    features_with_norm = np.column_stack([features_array, l2_norm])
    feature_names = [f"emb_{i}" for i in range(len(features))]
    feature_names.append("l2_norm")

    return pd.DataFrame(features_with_norm, columns=feature_names)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    request: PredictionRequest,
    model_version: Optional[str] = Query(None, description="Версия модели (catboost, lgbm, rf)"),
):
    if model_version is None:
        model_version = config.get("default_model_version")

    if model_version not in models_cache:
        available_models = list(models_cache.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Версия модели '{model_version}' не найдена. Доступные модели: {available_models}",
        )

    if request.dataset not in models_cache[model_version]:
        available_datasets = list(models_cache[model_version].keys())
        raise HTTPException(
            status_code=404,
            detail=f"Модель для датасета '{request.dataset}' версии '{model_version}' не найдена. Доступные датасеты: {available_datasets}",
        )

    try:
        model = models_cache[model_version][request.dataset]
        features = preprocess_features(request.features)
        y_pred = model.predict(features)

        return PredictionResponse(
            prediction=y_pred, dataset=request.dataset, model_version=model_version
        )

    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")


@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models():
    models_info = []
    model_configs = config.get("model_configs", {})

    for model_version, datasets in models_cache.items():
        model_config = model_configs.get(model_version, {})
        models_info.append(
            ModelInfo(
                model_version=model_version,
                datasets=list(datasets.keys()),
                description=model_config.get("description", f"Модель {model_version}"),
                file_extension=model_config.get("file_extension", ".pkl"),
            )
        )

    return models_info


@app.get("/models/{model_version}/datasets", tags=["Models"])
async def list_model_datasets(model_version: str):
    if model_version not in models_cache:
        available_models = list(models_cache.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Версия модели '{model_version}' не найдена. Доступные модели: {available_models}",
        )

    model_config = config.get("model_configs", {}).get(model_version, {})
    dataset_metrics = config.get("dataset_metrics", {})

    datasets_info = []
    for dataset in models_cache[model_version].keys():
        datasets_info.append({"name": dataset, "metric": dataset_metrics.get(dataset, "Unknown")})

    return {
        "model_version": model_version,
        "description": model_config.get("description", f"Модель {model_version}"),
        "hyperparameters": model_config.get("hyperparameters", {}),
        "available_datasets": datasets_info,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
