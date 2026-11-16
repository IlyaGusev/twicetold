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
    ANALYSIS_REPORTS_DIR: Path = ROOT / "reports" / "overlap"
    FIGURES_DIR: Path = ROOT / "paper" / "figs"
    TABLES_DIR: Path = ROOT / "paper" / "tables"
    PAPER_TEX: Path = ROOT / "paper" / "main.tex"
    PROMPTS_TEX: Path = ROOT / "paper" / "prompts.tex"
    PROMPTS_DIR: Path = ROOT / "src" / "prompts"
    RAW_ML_REVIEWS_DIR: Path = ROOT / "data" / "raw" / "sources"
    RAW_JOURNAL_REVIEWS_DIR: Path = ROOT / "data" / "raw" / "sources" / "other"
    HF_DATASET_REPO: str = "IlyaGusev/twicetold"
    HF_DATASET_REVISION: str = "4b491a0a5c831c77c7a6fd4bb735bb085ba0bd2f"
    HF_PULL_PATTERNS: list[str] = ["final/**", "gold/**", "additional/**", "README.md", "stats.json"]
    HF_RAW_PATTERNS: list[str] = [
        "raw/*_raw.jsonl*",
        "raw/panels*.jsonl*",
        "raw/paper_areas.parquet*",
        "raw/other/stats.json*",
        "raw/meld_scores.jsonl*",
        "raw/editlens_scores.jsonl*",
    ]
    HF_PUSH_IGNORE: list[str] = ["additional/paper_reviews/raw/**"]
    KIM_DATASET_REPO: str = "prometheus-eval/peerreview-bench"
    KIM_DATASET_REVISION: str = "b782d6676fcece1ed33f1eb2c0e27241d19f9f63"
    CLAIMCHECK_REPO: str = "JHU-CLSP/CLAIMCHECK"
    CLAIMCHECK_REVISION: str = "788673e0ac214f0e3a5c7d013412fffc63fab9fe"

    @property
    def FINAL_DIR(self):
        return self.DATA_DIR / "final"

    @property
    def GOLD_DIR(self):
        return self.DATA_DIR / "gold"

    @property
    def RAW_DIR(self):
        return self.DATA_DIR / "raw"

    @property
    def ADDITIONAL_DIR(self):
        return self.DATA_DIR / "additional"

    @property
    def WORK_DIR(self):
        return self.DATA_DIR / "work"

    @property
    def ANALYSIS_DIR(self):
        return self.WORK_DIR / "overlap"

    @property
    def REWRITE_TEST_DIR(self):
        return self.ADDITIONAL_DIR / "rewrite_test"

    @property
    def PAPER_REVIEW_DIR(self):
        return self.ADDITIONAL_DIR / "paper_reviews"

    @property
    def OTHER_REVIEWS_DIR(self):
        return self.RAW_DIR / "other"

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
    ICLR_GAP_YEARS: tuple[int, int] = (2024, 2027)
    GAP_CELLS: dict[str, tuple[str, int, int]] = {
        "ICLR 2024–27": ("ICLR", 2024, 2027),
        "ICLR 2027": ("ICLR", 2027, 2027),
        "NeurIPS 2024–25": ("NeurIPS", 2024, 2025),
        "ICML 2025–26": ("ICML", 2025, 2026),
    }
    REWRITE_PAIR_PREFIX: str = "pair_"

    DEFAULT_MODEL: str = "gpt-6-luna-high"
    AI_REFERENCE_MODEL: str = "gpt-6-luna"
    REWRITE_SECOND_WRITER: str = "kimi-k3.1-pro"
    GOLD_COUNCIL: list[str] = ["sol-or", "deepseek-v4.1-flash", "kimi-k3.1-pro"]
    MODEL_LABELS: dict[str, str] = {
        "gpt-6-luna-high": "GPT-6 Luna",
        "sol-or": "Sol 6.1",
        "deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
        "kimi-k3.1-pro": "Kimi K3.1 Pro",
    }
    CONTRAST_YEARS: tuple[int, int] = (2025, 2026)
    ICLR_SNAPSHOT_YEARS: tuple[int, int] = (2026, 2027)
    MIN_PANEL_REVIEWS: int = 3
    ROW_GROUP_ROWS: int = 20000
    FLAG_FROM: int = 2024
    WITHDRAWN_CHECK: str = "Overlap, withdrawn papers excluded"
    REFERENCE_BOOTSTRAPS: int = 5000
    REFERENCE_BOOTSTRAP_SEED: int = 0
    REVIEW_TEXT_MAX_CHARS: int = 14000
    PAPER_TEXT_MAX_CHARS: int = 60000
    PAPER_REVIEW_PAPERS: int = 100
    ARXIV_API: str = "https://export.arxiv.org/api/query"
    ARXIV_DELAY_SECONDS: float = 3.0

    EMBED_MODEL: str = "text-embedding-3-small"
    CLAIM_EMBED_MODEL: str = "google/gemini-embedding-2"
    REVIEW_EMBED_MODEL: str = "google/gemini-embedding-2"
    REVIEW_EMBED_DIMENSIONS: int = 1536
    MELD_REPO: str = "anon-review-meld-2026/meld"
    MELD_REVISION: str = "ddfc9c789b8c8d0f92d3774265316d418c96dc67"
    MELD_FLAG_QUANTILE: float = 0.999
    MELD_CALIBRATION_LAST_YEAR: int = 2022
    MELD_WRITTEN_TASK: str = "generate"
    EDITLENS_REPO: str = "pangram/editlens_Llama-3.2-3B"
    EDITLENS_REVISION: str = "b4a297be1d4c36a2a02c7709ae12f3c6626fae58"
    EDITLENS_BASE_REPO: str = "unsloth/Llama-3.2-3B"
    EDITLENS_BASE_REVISION: str = "d4446454d87d51aa42e1fb174f25acc5f8762331"
    EDITLENS_BUCKETS: int = 4
    EDITLENS_MAX_LENGTH: int = 1024

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
