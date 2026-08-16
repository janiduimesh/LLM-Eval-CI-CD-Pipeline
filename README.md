# 🧪 LLM Evaluation CI/CD Pipeline for RAG

A production-grade, custom-built evaluation pipeline that grades a Retrieval-Augmented Generation (RAG) chatbot on **5 key metrics**, enforces pass/fail thresholds via **GitHub Actions**, logs results to CSV/JSON, and displays historical trends on a **Streamlit dashboard**.

The RAG system answers questions based on university/academic rules across 4 domains: assignments, exams, registration, and payments.

---

## 📐 Architecture

```
┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Question   │────▶│  Retriever  │────▶│  LLM Client  │
│  (Golden DS) │     │  (TF-IDF)   │     │  (OpenAI)    │
└──────────────┘     └─────────────┘     └──────┬───────┘
                                                │
                                                ▼
                     ┌─────────────────────────────────────┐
                     │        Evaluation Engine            │
                     │  ┌─────────┐  ┌───────────────────┐ │
                     │  │ Metrics │  │ Threshold Checker  │ │
                     │  └─────────┘  └───────────────────┘ │
                     └────────────────┬────────────────────┘
                                      │
                          ┌───────────┼───────────┐
                          ▼           ▼           ▼
                     ┌─────────┐ ┌─────────┐ ┌──────────┐
                     │  JSON   │ │   CSV   │ │  GitHub  │
                     │ Report  │ │ History │ │ Actions  │
                     └─────────┘ └─────────┘ └──────────┘
                          │           │
                          └─────┬─────┘
                                ▼
                     ┌────────────────────┐
                     │ Streamlit Dashboard│
                     └────────────────────┘
```

---

## 📊 Evaluation Metrics

| # | Metric | Method | Threshold | Direction |
|---|--------|--------|-----------|-----------|
| 1 | **Accuracy** | SequenceMatcher + Keyword F1 vs. ground truth | ≥ 70% | Higher is better |
| 2 | **Hallucination Rate** | Sentence-level grounding check against context | ≤ 20% | Lower is better |
| 3 | **Latency** | Wall-clock time per LLM API call | ≤ 10s | Lower is better |
| 4 | **Cost** | Token usage × per-token rates | ≤ $0.05/query | Lower is better |
| 5 | **Failed Questions** | Count of questions breaching any threshold | ≤ 5 | Lower is better |

---

## 🗂 Directory Structure

```text
llm-eval-cicd/
│
├── knowledge_base/           # University rules (RAG knowledge)
│   ├── assignment_rules.txt
│   ├── exam_rules.txt
│   ├── registration_rules.txt
│   └── payment_rules.txt
│
├── eval_data/                # Benchmark dataset
│   └── golden_dataset.json   # 20 Q&A pairs (5 per domain)
│
├── src/                      # RAG chatbot source
│   ├── config.py             # Centralized configuration
│   ├── retriever.py          # TF-IDF + cosine similarity retriever
│   ├── llm_client.py         # OpenAI API wrapper
│   └── rag_chatbot.py        # RAG orchestrator
│
├── evaluation/               # Evaluation engine
│   ├── run_eval.py           # Main evaluation runner
│   ├── metrics.py            # 5 metric functions
│   ├── threshold_checker.py  # Pass/fail threshold logic
│   └── report_generator.py   # JSON/CSV/stdout reports
│
├── results/                  # Output (auto-generated)
│   ├── eval_results.json     # Per-question results
│   └── history.csv           # Historical run log
│
├── dashboard/                # Visualization
│   └── app.py                # Streamlit dashboard
│
├── .github/workflows/        # CI/CD
│   └── llm_eval.yml          # GitHub Actions workflow
│
├── requirements.txt
├── .env.example
├── main.py                   # Entry point
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/llm-eval-cicd.git
cd llm-eval-cicd
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY
```

### 3. Run Evaluation

```bash
python main.py
```

This will:
- Load the 20 golden questions
- Run each through the RAG pipeline (retrieve → LLM)
- Compute all 5 metrics per question
- Save results to `results/eval_results.json`
- Append a summary row to `results/history.csv`
- Print a formatted summary to stdout
- Exit with code **0** (pass) or **1** (fail)

### 4. View Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard shows:
- 📊 KPI cards with pass/fail colors
- 📈 Historical trend charts (accuracy, hallucination, latency, cost)
- 🔍 Per-question drill-down table
- ⚙️ Threshold configuration

---

## ⚙️ Configuration

All settings can be configured via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | **Required.** Your Google Gemini API key |
| `MODEL_NAME` | `gemini-2.0-flash` | Gemini model to use |
| `TEMPERATURE` | `0.2` | LLM temperature |
| `MAX_TOKENS` | `512` | Max tokens per response |
| `ACCURACY_THRESHOLD` | `0.7` | Minimum accuracy (70%) |
| `HALLUCINATION_THRESHOLD` | `0.2` | Maximum hallucination (20%) |
| `LATENCY_THRESHOLD` | `10.0` | Maximum latency (10s) |
| `COST_THRESHOLD` | `0.05` | Maximum cost ($0.05) |
| `FAILED_QUESTIONS_THRESHOLD` | `5` | Maximum failed questions |

---

## 🔄 CI/CD Integration

The GitHub Actions workflow (`.github/workflows/llm_eval.yml`) runs automatically on:
- **Pull requests** to `main`
- **Manual dispatch** via the Actions tab

### Setup

1. Add your `GEMINI_API_KEY` as a **GitHub Secret**:
   - Go to Settings → Secrets and variables → Actions → New repository secret
   - Name: `GEMINI_API_KEY`, Value: your key

2. (Optional) Set `MODEL_NAME` as a **GitHub Variable** to override the default model.

### What it does

1. ✅ Checks out code and installs dependencies
2. 🧪 Runs `python main.py`
3. 📎 Uploads `results/` as workflow artifacts
4. 💬 Posts an evaluation summary as a PR comment
5. 🚫 **Fails the workflow** if thresholds are breached → blocks merge

---

## 📝 Golden Dataset

The dataset contains 20 question/answer pairs across 4 domains:

| Domain | Questions | IDs |
|--------|-----------|-----|
| Assignment Rules | 5 | Q001–Q005 |
| Exam Rules | 5 | Q006–Q010 |
| Registration Rules | 5 | Q011–Q015 |
| Payment Rules | 5 | Q016–Q020 |

Each entry has:
- `id`: Unique question identifier
- `question`: The student's question
- `ground_truth`: The expected correct answer
- `category`: Domain classification

---

## 🛠 Tech Stack

- **LLM**: Google Gemini (2.0 Flash)
- **Embeddings**: Google text-embedding-004
- **Vector Store**: ChromaDB (persistent, local)
- **Evaluation**: Custom metrics (SequenceMatcher, keyword F1)
- **Dashboard**: Streamlit + Plotly
- **CI/CD**: GitHub Actions
- **Config**: python-dotenv

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.