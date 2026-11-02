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
    GOLD_DIR: Path = ROOT / "data" / "gold"
    ANALYSIS_DIR: Path = ROOT / "data" / "overlap"
    ANALYSIS_REPORTS_DIR: Path = ROOT / "reports" / "overlap"
    FIGURES_DIR: Path = ROOT / "paper" / "figs"
    TABLES_DIR: Path = ROOT / "paper" / "tables"
    PROMPTS_DIR: Path = ROOT / "src" / "prompts"
    RAW_ML_REVIEWS_DIR: Path = Path("/home/yallen/data/peer-review")
    RAW_JOURNAL_REVIEWS_DIR: Path = Path("/home/yallen/data/peer-review/other")

    @property
    def OTHER_REVIEWS_DIR(self):
        return self.DATA_DIR / "other"

    @property
    def INTEL_DIR(self):
        return self.RAW_ML_REVIEWS_DIR / "intel"

    @property
    def ICLR_AREAS_DIR(self):
        return self.RAW_ML_REVIEWS_DIR / "iclr_areas"

    DEFAULT_MODEL: str = "gpt-6-luna-high"
    AI_REFERENCE_MODEL: str = "gpt-6-luna"
    GOLD_COUNCIL: list[str] = ["sol-or", "opus-5.5", "kimi-k3.1-pro"]
    GOLD_JUDGE_MODEL: str = "opus-5.5"
    REVIEW_TEXT_MAX_CHARS: int = 14000

    EMBED_MODEL: str = "text-embedding-3-small"
    COCODET_BASE_MODEL: str = "answerdotai/ModernBERT-base"
    COCODET_REPO: str = "khaaaaaan/CoCoDet"
    COCODET_WEIGHTS_FILE: str = "CoCoDet.bin"

    PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
        "gpt-6-luna": (0.10, 0.50),
        "gpt-6-luna-high": (0.10, 0.50),
        "sol-or": (2.00, 10.00),
        "opus-5.5": (4.00, 20.00),
        "kimi-k3.1-pro": (2.00, 10.00),
    }
    REASONING_EFFORTS: dict[str, PassEfforts] = {
        "gpt-6-luna": PassEfforts(pass1="low", pass2="medium"),
        "gpt-6-luna-high": PassEfforts(pass1="high", pass2="high"),
        "sol-or": PassEfforts(pass1="medium", pass2="medium"),
        "opus-5.5": PassEfforts(pass1="medium", pass2="medium"),
        "kimi-k3.1-pro": PassEfforts(pass1="medium", pass2="medium"),
    }
    OPENROUTER_ROUTES: dict[str, OpenRouterRoute] = {
        "gpt-6-luna-high": OpenRouterRoute(id="openai/gpt-6-luna"),
        "sol-or": OpenRouterRoute(id="openai/gpt-6.1-sol"),
        "opus-5.5": OpenRouterRoute(id="anthropic/claude-opus-5.5"),
        "kimi-k3.1-pro": OpenRouterRoute(id="moonshotai/kimi-k3.1-pro"),
    }


settings = Settings()
