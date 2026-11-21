import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

INPUT_COST_PER_TOKEN = float(os.getenv("INPUT_COST_PER_TOKEN", "0.00000005"))
OUTPUT_COST_PER_TOKEN = float(os.getenv("OUTPUT_COST_PER_TOKEN", "0.00000008"))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_BASE_DIR = os.path.join(PROJECT_ROOT, "knowledge_base")
EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "eval_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

GOLDEN_DATASET_PATH = os.path.join(EVAL_DATA_DIR, "golden_dataset.json")
EVAL_RESULTS_PATH = os.path.join(RESULTS_DIR, "eval_results.json")
HISTORY_CSV_PATH = os.path.join(RESULTS_DIR, "history.csv")
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "3"))

ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.7"))
HALLUCINATION_THRESHOLD = float(os.getenv("HALLUCINATION_THRESHOLD", "0.2"))
LATENCY_THRESHOLD = float(os.getenv("LATENCY_THRESHOLD", "10.0"))
COST_THRESHOLD = float(os.getenv("COST_THRESHOLD", "0.05"))
FAILED_QUESTIONS_THRESHOLD = int(os.getenv("FAILED_QUESTIONS_THRESHOLD", "5"))

THRESHOLDS = {
    "accuracy": ACCURACY_THRESHOLD,
    "hallucination": HALLUCINATION_THRESHOLD,
    "latency": LATENCY_THRESHOLD,
    "cost": COST_THRESHOLD,
    "failed_questions": FAILED_QUESTIONS_THRESHOLD,
}
