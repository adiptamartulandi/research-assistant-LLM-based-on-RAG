from pydantic_settings import BaseSettings


class ModelConfig(BaseSettings):
    MODEL_EMBEDDINGS: str = "text-embedding-3-small"


MODEL_CONFIG = ModelConfig()