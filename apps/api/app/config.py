from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ZM_", env_file=str(REPO_ROOT / ".env"), extra="ignore")

    app_name: str = "ZeroMalaria API"
    database_url: str = f"sqlite:///{REPO_ROOT / 'apps' / 'api' / 'zeromalaria.db'}"
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]
    demo_today: str = "2026-09-30"
    overdue_hours: int = 24
    llm_enabled: bool = False
    synthetic_badge: str = "Synthetic demo data"
    demo_mode: bool = True
    demo_password: str = "demo1234"
    # Seed the DB with demo users/cases on startup if it's empty (for hosts with ephemeral disks).
    auto_seed: bool = False
    jwt_secret: str = "zeromalaria-demo-secret-change-in-production"
    jwt_expire_hours: int = 8
    # prompt = dismissible modal (default); enforce = cannot dismiss until password changed
    password_change_policy: str = "prompt"
    # AI
    gemini_api_key: str = ""
    groq_api_key: str = ""
    google_cloud_project: str = ""
    ai_provider_order: str = "gemini,groq,local"
    ai_timeout_seconds: float = 4.0
    # Pindo VoiceAI (server-side only; never expose through VITE_* variables)
    pindo_api_token: str = ""
    pindo_access_mode: Literal["public", "authenticated"] = "public"
    pindo_api_base_url: str = "https://api.pindo.io"
    pindo_timeout_seconds: float = 20.0
    app_version: str = "0.2.0"


settings = Settings()
