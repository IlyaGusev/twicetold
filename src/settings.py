from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class OpenRouterRoute(BaseModel):
    id: str
    provider: dict | None = None
    strict: bool = True


class PassEfforts(BaseModel):
    pass1: str
    pass2: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    OPENAI_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    OPENREVIEW_USERNAME: str = ""
    OPENREVIEW_PASSWORD: str = ""

    DATA_DIR: Path = ROOT / "data"
    REPORTS_DIR: Path = ROOT / "reports"
    FIGURES_DIR: Path = ROOT / "paper" / "figs"
    TABLES_DIR: Path = ROOT / "paper" / "tables"
    PROMPTS_DIR: Path = ROOT / "src" / "prompts"
    RAW_ML_REVIEWS_DIR: Path = Path("/home/yallen/data/peer-review")
    RAW_JOURNAL_REVIEWS_DIR: Path = Path("/home/yallen/data/peer-review/other")

    @property
    def OTHER_REVIEWS_DIR(self):
        return self.DATA_DIR / "other"

    DEFAULT_MODEL: str = "gpt-6-luna-high"
    BLIND_SET_MODEL: str = "gpt-6-luna"
    SECOND_MODEL: str = "gpt-6.1-sol"
    AI_REFERENCE_MODEL: str = "gpt-6-luna"
    INSTRUMENT_MODELS: str = "gpt-6-luna,gpt-6.1-sol,glm-5.3-flash,deepseek-v4.1-flash"
    REVIEW_TEXT_MAX_CHARS: int = 14000

    EMBED_MODEL: str = "text-embedding-3-small"
    CLAIM_EMBED_MODEL: str = "text-embedding-3-large"
    COCODET_BASE_MODEL: str = "answerdotai/ModernBERT-base"
    COCODET_REPO: str = "khaaaaaan/CoCoDet"
    COCODET_WEIGHTS_FILE: str = "CoCoDet.bin"

    PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
        "gpt-6.1-sol": (2.00, 10.00),
        "gpt-6.1-terra": (1.00, 5.00),
        "gpt-6-luna": (0.10, 0.50),
        "gpt-6-luna-high": (0.10, 0.50),
        "glm-5.3-flash": (0.15, 0.50),
        "glm-zai": (0.15, 0.50),
        "deepseek-v4.1-flash": (0.15, 0.60),
        "sol-or": (2.00, 10.00),
        "opus-5.5": (4.00, 20.00),
        "kimi-k3.1-pro": (2.00, 10.00),
    }
    REASONING_EFFORTS: dict[str, PassEfforts] = {
        "gpt-6-luna": PassEfforts(pass1="low", pass2="medium"),
        "gpt-6.1-sol": PassEfforts(pass1="low", pass2="medium"),
        "glm-5.3-flash": PassEfforts(pass1="low", pass2="medium"),
        "deepseek-v4.1-flash": PassEfforts(pass1="low", pass2="medium"),
        "gpt-6-luna-high": PassEfforts(pass1="high", pass2="high"),
    }
    OPENROUTER_ROUTES: dict[str, OpenRouterRoute] = {
        "gpt-6-luna-high": OpenRouterRoute(id="openai/gpt-6-luna"),
        "glm-5.3-flash": OpenRouterRoute(id="z-ai/glm-5.3-flash"),
        "glm-zai": OpenRouterRoute(id="z-ai/glm-5.3-flash", provider={"order": ["Z.AI"], "allow_fallbacks": False}, strict=False),
        "deepseek-v4.1-flash": OpenRouterRoute(id="deepseek/deepseek-v4.1-flash-1017"),
        "sol-or": OpenRouterRoute(id="openai/gpt-6.1-sol"),
        "opus-5.5": OpenRouterRoute(id="anthropic/claude-opus-5.5"),
        "kimi-k3.1-pro": OpenRouterRoute(id="moonshotai/kimi-k3.1-pro"),
    }


settings = Settings()
