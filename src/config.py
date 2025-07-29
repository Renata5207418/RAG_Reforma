from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Final

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 1) Chave da OpenAI  → falha cedo se faltar
# ─────────────────────────────────────────────────────────────────────────────
API_KEY: Final[str] = os.getenv("OPENAI_API_KEY", "").strip()
if not API_KEY:
    raise RuntimeError("Variável de ambiente OPENAI_API_KEY não definida.")

# ─────────────────────────────────────────────────────────────────────────────
# 1) Chave da Qdrant
# ─────────────────────────────────────────────────────────────────────────────
QDRANT_API_KEY: Final[str] = os.getenv("QDRANT_API_KEY", "").strip()

# ─────────────────────────────────────────────────────────────────────────────
# 2) Modelos / parâmetros
#    (podem ser alterados via export LANGCHAIN_MODEL=gpt-4o-mini etc.)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL: Final[str] = os.getenv("LANGCHAIN_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL: Final[str] = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
TEMPERATURE: Final[float] = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS: Final[int] = int(os.getenv("MAX_TOKENS", "1024"))
TOP_P: Final[float] = float(os.getenv("TOP_P", "1.0"))

# ─────────────────────────────────────────────────────────────────────────────
# 3) Caminhos
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
LOG_DIR: Final[Path] = Path(os.getenv("LOG_DIR", PROJECT_ROOT / "logs"))

# Cria pastas se não existirem
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4) Outros ajustes
# ─────────────────────────────────────────────────────────────────────────────
# → desligar telemetria de Chroma (boa prática para produção)
CHROMA_TELEMETRY: Final[bool] = os.getenv("CHROMA_TELEMETRY", "false").lower() == "true"


class Settings:
    """
    Objeto imutável para importar em qualquer módulo.
    Agrupa configs de ambiente, modelos e caminhos.
    """
    api_key = API_KEY
    qdrant_api_key = QDRANT_API_KEY
    default_model = DEFAULT_MODEL
    embedding_model = EMBEDDING_MODEL
    temperature = TEMPERATURE
    max_tokens = MAX_TOKENS
    top_p = TOP_P
    log_dir = LOG_DIR
    chroma_telemetry = CHROMA_TELEMETRY


settings = Settings()
