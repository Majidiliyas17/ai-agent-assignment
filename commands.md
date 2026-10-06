# Commands

Every command used to set up, run and test this project. Run them from the
repository root after activating the virtual environment.

---

## 1. One-time setup

```bash
# clone
git clone https://github.com/MajidIliyas17/ai-agent-assignment.git
cd ai-agent-assignment

# virtual environment
python -m venv .venv
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
```

macOS / Linux:

```bash
source .venv/bin/activate
cp .env.example .env
```

```bash
# dependencies
pip install -r requirements.txt

# then edit .env and set GEMINI_API_KEY=<your key>
```

---

## 2. Start the mock CMS

```bash
uvicorn mock_cms.main:app --reload --port 8000
```

| URL | Purpose |
|---|---|
| <http://localhost:8000/docs> | Swagger UI |
| <http://localhost:8000/health> | Liveness check |
| <http://localhost:8000/records> | Stored records |

Leave this terminal running while the pipeline executes.

---

## 3. Run the pipeline

```bash
python -m app.main
```

```bash
python -m app.main --help                 # list all flags
python -m app.main --version              # print version
python -m app.main --provider offline     # keyless run (no API key needed)
python -m app.main --limit 5              # process only 5 records
python -m app.main --demo-failure         # inject failures for retry demo
```

---

## 4. CLI flags

| Flag | Effect |
|---|---|
| `--csv PATH` | Use a specific leads CSV |
| `--cms-url URL` | Override `CMS_BASE_URL` |
| `--fhir-count N` | Number of FHIR patients (min 15) |
| `--limit N` | Process at most N records |
| `--provider gemini\|offline` | Override the LLM provider |
| `--skip-csv` | Skip the CSV source |
| `--skip-fhir` | Skip the FHIR source |
| `--skip-cms` | Skip CMS delivery (offline analysis) |
| `--demo-failure` | Inject controlled failures |
| `--help` | Show help |
| `--version` | Show version |

---

## 5. Tests

```bash
python -m pytest                 # full suite
pytest                           # same (pytest.ini is configured)
pytest -q                        # quiet output
pytest tests/test_agent.py       # single file
pytest -k normalizer             # by keyword
```

Tests mock the LLM, so **no API key is required**.

---

## 6. Environment variables

Set these in `.env` (git-ignored). `.env.example` is the committed template.

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Gemini API key (required for real AI) |
| `GEMINI_MODEL` | `gemini-flash-latest` | Gemini model id |
| `LLM_PROVIDER` | `gemini` | `gemini` or `offline` |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature |
| `LLM_MAX_TOKENS` | `1024` | Max output tokens |
| `CMS_BASE_URL` | `http://localhost:8000` | Mock CMS base URL |
| `CMS_DB_PATH` | `./output/mock_cms.sqlite3` | CMS SQLite file |
| `FHIR_BASE_URL` | `https://r4.smarthealthit.org` | FHIR R4 sandbox |
| `FHIR_PATIENT_COUNT` | `20` | Patients to fetch (min 15) |
| `HTTP_TIMEOUT_SECONDS` | `20` | Per-request timeout |
| `MAX_RETRIES` | `3` | Bounded retry attempts |
| `RETRY_BACKOFF_SECONDS` | `1.0` | Exponential backoff base |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## 7. Git workflow

```bash
git status
git diff
git add commands.md
git commit -m "docs: add commands reference"
git push origin dev
```

---

## Typical session (2 terminals)

```text
Terminal 1:  uvicorn mock_cms.main:app --reload --port 8000
Terminal 2:  python -m app.main
```
