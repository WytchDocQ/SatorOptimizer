from __future__ import annotations

from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=".env",
        env_nested_delimiter=",",
    )
    # Auth & security (single key)
    api_key: Optional[str] = Field(default=None, alias="SATOR_API_KEY")
    rate_limit_per_min: int = Field(300, alias="SATOR_RATE_LIMIT_PER_MIN")
    ip_whitelist: List[str] = Field(default_factory=list, alias="SATOR_IP_WHITELIST")
    ip_blacklist: List[str] = Field(default_factory=list, alias="SATOR_IP_BLACKLIST")

    # Transports
    enable_http: bool = Field(True, alias="SATOR_ENABLE_HTTP")
    http_host: str = Field("0.0.0.0", alias="SATOR_HTTP_HOST")
    http_port: int = Field(8080, alias="SATOR_HTTP_PORT")
    enable_nats: bool = Field(False, alias="SATOR_ENABLE_NATS")
    nats_url: str = Field("nats://localhost:4222", alias="SATOR_NATS_URL")

    # Logging
    log_level: str = Field("info", alias="LOG_LEVEL")
    log_format: str = Field("human", alias="LOG_FORMAT")  # json|human
    log_to_file: bool = Field(False, alias="LOG_TO_FILE")
    log_file_path: Optional[str] = Field(None, alias="LOG_FILE_PATH")

    # Device
    device: str = Field("cpu", alias="SATOR_DEVICE")  # cpu|cuda
    cuda_device: int = Field(0, alias="SATOR_CUDA_DEVICE")

    # Runtime
    job_ttl_sec: int = Field(600, alias="SATOR_RESULT_TTL_SEC")
    job_timeout_sec: int = Field(300, alias="SATOR_JOB_TIMEOUT_SEC")
    concurrency: int = Field(4, alias="SATOR_CONCURRENCY")

    


def get_settings() -> Settings:
    return Settings()


