from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class PassEfforts(BaseModel):
    pass1: str
    pass2: str


class RewriteCondition(BaseModel):
    writer: str
    task: str
    panel: str
    preserve: str
    text: str
    sources: tuple[str, ...] | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    OPENAI_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    OPENREVIEW_USERNAME: str = ""
    OPENREVIEW_PASSWORD: str = ""

    DATA_DIR: Path = ROOT / "data"
    REPORTS_DIR: Path = ROOT / "reports"
    GOLD_DIR: Path = ROOT / "data" / "gold"
    REWRITE_TEST_DIR: Path = ROOT / "data" / "rewrite_test"
    PAPER_REVIEW_DIR: Path = ROOT / "data" / "paper_reviews"
    ANALYSIS_DIR: Path = ROOT / "data" / "overlap"
    ANALYSIS_REPORTS_DIR: Path = ROOT / "reports" / "overlap"
    FIGURES_DIR: Path = ROOT / "paper" / "figs"
    TABLES_DIR: Path = ROOT / "paper" / "tables"
    PROMPTS_DIR: Path = ROOT / "src" / "prompts"
    RAW_ML_REVIEWS_DIR: Path = ROOT / "data" / "raw"
    RAW_JOURNAL_REVIEWS_DIR: Path = ROOT / "data" / "raw" / "other"
    RELEASE_DIR: Path = ROOT / "data" / "release"
    HF_DATASET_REPO: str = "IlyaGusev/twicetold"
    HF_DATASET_REVISION: str = "bad166fe177698c7e2aa877b2023d4c48aadcb26"
    HF_GOLD_DIR: str = "gold"
    HF_DATASET_LOCAL: bool = False
    KIM_DATASET_REPO: str = "prometheus-eval/peerreview-bench"
    KIM_DATASET_REVISION: str = "b782d6676fcece1ed33f1eb2c0e27241d19f9f63"
    CLAIMCHECK_REPO: str = "JHU-CLSP/CLAIMCHECK"
    CLAIMCHECK_REVISION: str = "788673e0ac214f0e3a5c7d013412fffc63fab9fe"

    @property
    def OTHER_REVIEWS_DIR(self):
        return self.DATA_DIR / "other"

    @property
    def INTEL_DIR(self):
        return self.RAW_ML_REVIEWS_DIR / "intel"

    @property
    def ICLR_AREAS_DIR(self):
        return self.RAW_ML_REVIEWS_DIR / "iclr_areas"

    @property
    def REWRITE_CONDITIONS(self):
        first, second = self.AI_REFERENCE_MODEL, self.REWRITE_SECOND_WRITER
        return {
            "outline": RewriteCondition(writer=first, task="rewrites", panel="rw2_", preserve="pres_", text="rw_"),
            "polish": RewriteCondition(writer=first, task="polishes", panel="rwp_", preserve="ppres_", text="pl_"),
            "outline_second": RewriteCondition(
                writer=second, task="rewrites", panel="rk2_", preserve="kpres_", text="rk_", sources=("ICLR",)
            ),
            "polish_second": RewriteCondition(
                writer=second, task="polishes", panel="rkp_", preserve="kppres_", text="pk_", sources=("ICLR",)
            ),
        }

    ICLR_FROM: int = 2020
    ICLR_LAST: int = 2027
    ML_SOURCES: list[str] = ["ICLR", "NeurIPS"]
    ML_VENUES: list[str] = ["ICLR", "NeurIPS", "ICML"]
    REWRITE_PAIR_PREFIX: str = "pair_"

    DEFAULT_MODEL: str = "gpt-6-luna-high"
    AI_REFERENCE_MODEL: str = "gpt-6-luna"
    REWRITE_SECOND_WRITER: str = "kimi-k3.1-pro"
    GOLD_COUNCIL: list[str] = ["sol-or", "deepseek-v4.1-flash", "kimi-k3.1-pro"]
    MODEL_LABELS: dict[str, str] = {
        "gpt-6-luna-high": "Luna",
        "sol-or": "Sol 6.1",
        "deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
        "kimi-k3.1-pro": "Kimi K3.1 Pro",
    }
    CONTRAST_YEARS: tuple[int, int] = (2025, 2026)
    REFERENCE_BOOTSTRAPS: int = 5000
    REFERENCE_BOOTSTRAP_SEED: int = 0
    REVIEW_TEXT_MAX_CHARS: int = 14000
    PAPER_TEXT_MAX_CHARS: int = 60000
    PAPER_REVIEW_PAPERS: int = 100
    ARXIV_API: str = "https://export.arxiv.org/api/query"
    ARXIV_DELAY_SECONDS: float = 3.0

    EMBED_MODEL: str = "text-embedding-3-small"
    CLAIM_EMBED_MODEL: str = "google/gemini-embedding-2"
    MELD_REPO: str = "anon-review-meld-2026/meld"
    MELD_REVISION: str = "ddfc9c789b8c8d0f92d3774265316d418c96dc67"
    MELD_FLAG_QUANTILE: float = 0.999
    MELD_CALIBRATION_LAST_YEAR: int = 2022
    MELD_WRITTEN_TASK: str = "generate"

    PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
        "gpt-6-luna": (0.10, 0.50),
        "gpt-6-luna-high": (0.10, 0.50),
        "sol-or": (2.00, 10.00),
        "deepseek-v4.1-flash": (0.30, 1.20),
        "kimi-k3.1-pro": (2.00, 10.00),
    }
    REASONING_EFFORTS: dict[str, PassEfforts] = {
        "gpt-6-luna": PassEfforts(pass1="low", pass2="medium"),
        "gpt-6-luna-high": PassEfforts(pass1="high", pass2="high"),
        "sol-or": PassEfforts(pass1="medium", pass2="medium"),
        "deepseek-v4.1-flash": PassEfforts(pass1="medium", pass2="medium"),
        "kimi-k3.1-pro": PassEfforts(pass1="medium", pass2="medium"),
    }
    OPENROUTER_PROVIDERS: dict[str, str] = {"kimi-k3.1-pro": "moonshotai", "deepseek-v4.1-flash": "fireworks"}
    OPENROUTER_ROUTES: dict[str, str] = {
        "gpt-6-luna-high": "openai/gpt-6-luna",
        "sol-or": "openai/gpt-6.1-sol",
        "deepseek-v4.1-flash": "deepseek/deepseek-v4.1-flash-1017",
        "kimi-k3.1-pro": "moonshotai/kimi-k3.1-pro",
    }


settings = Settings()
