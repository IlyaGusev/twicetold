from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


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
    AUDIT_DIR: Path = ROOT / "data" / "gold_audit"
    AUDIT_WRITTEN_DIR: Path = ROOT / "data" / "gold_audit_written"
    REWRITE_TEST_DIR: Path = ROOT / "data" / "rewrite_test"
    PAPER_REVIEW_DIR: Path = ROOT / "data" / "paper_reviews"
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
    REWRITE_SECOND_WRITER: str = "kimi-k3.1-pro"
    GOLD_COUNCIL: list[str] = ["sol-or", "opus-5.5", "kimi-k3.1-pro"]
    GOLD_JUDGE_MODEL: str = "opus-5.5"
    REVIEW_TEXT_MAX_CHARS: int = 14000
    PAPER_TEXT_MAX_CHARS: int = 60000
    PAPER_REVIEW_PAPERS: int = 100
    ARXIV_API: str = "https://export.arxiv.org/api/query"
    ARXIV_DELAY_SECONDS: float = 3.0

    EMBED_MODEL: str = "text-embedding-3-small"
    MELD_REPO: str = "anon-review-meld-2026/meld"
    MELD_REVISION: str = "ddfc9c789b8c8d0f92d3774265316d418c96dc67"
    MELD_FLAG_QUANTILE: float = 0.999
    MELD_CALIBRATION_LAST_YEAR: int = 2022

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
    OPENROUTER_ROUTES: dict[str, str] = {
        "gpt-6-luna-high": "openai/gpt-6-luna",
        "sol-or": "openai/gpt-6.1-sol",
        "opus-5.5": "anthropic/claude-opus-5.5",
        "kimi-k3.1-pro": "moonshotai/kimi-k3.1-pro",
    }


settings = Settings()
