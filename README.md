# AI Agent for Lead & Patient Intake, Enrichment and Outreach

A backend-only take-home assignment: ingest two structurally different sources
(a LinkedIn-style CSV export and synthetic FHIR R4 patients from the public
SMART Health IT sandbox), normalize them into one schema, reason over each
record with an AI agent (Gemini by default), and deliver the structured
decisions to a local mock CMS over real HTTP POSTs.

> **No frontend.** Swagger UI (`/docs`) is the only UI.

---

## Quick start (after cloning)

```bash
git clone <your-repo-url>.git
cd ai-agent-assignment

python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1      macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env        # Windows: Copy-Item .env.example .env
# edit .env -> set GEMINI_API_KEY=<your key>

uvicorn mock_cms.main:app --reload --port 8000   # terminal 1
python -m app.main                               # terminal 2
```

Full detail in [§4 Setup instructions](#4-setup-instructions).

---

## 1. Project overview

```
CSV leads ──► Lead loader ──► Lead normalizer ──┐
                                               ├──► UnifiedRecord ──► AI Agent ──► AgentResult
FHIR sandbox ──► FHIR client ──► Patient normalizer ─┘                                    │
                                                                                          ▼
                                                                              Mock CMS (HTTP POST)
                                                                                          │
                                                                                          ▼
                                                              output/*.json + output/pipeline.log
```

Three stages plus a delivery stage:

| Stage | What it does | Where |
|---|---|---|
| 1. Ingest | Read `data/mock_leads.csv`; fetch 15+ synthetic `Patient` resources from `https://r4.smarthealthit.org` with bounded retries | `app/ingestion/` |
| 2. Normalize | Separate lead/patient normalizers produce one `UnifiedRecord` schema | `app/normalization/`, `app/models/` |
| 3. AI agent | Per-record loop: classify → action → channel/tone → message → rationale → validated JSON, with repair retries | `app/agent/` |
| 4. Deliver | Real `POST /records` to the mock CMS, structured payload (never a text blob), then write artifacts | `app/cms/`, `mock_cms/` |

**Key design rule:** no fake AI output. The Gemini API is called for every
record when configured. If the key is missing the pipeline fails fast with an
actionable error instead of inventing results.

---

## 2. Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full write-up and a Mermaid
diagram. In short:

* **Ingestion layer** — resilient CSV loader + async FHIR client (timeout,
  retries with exponential backoff, malformed-resource isolation).
* **Normalization layer** — two independent normalizers, one shared
  `UnifiedRecord` with a flexible `context` object instead of one giant schema.
* **Agent layer** — a real processing loop (not one generic prompt): context
  preparation → LLM call → JSON extraction → Pydantic validation → bounded
  repair prompt → decision logging.
* **LLM provider** — `BaseLLMProvider` abstraction; Gemini-specific code lives
  only in `app/agent/llm.py`. An Ollama provider can be added by registering
  one new class in `build_llm_provider`.
* **CMS integration** — async `httpx` client with retries and an explicit
  integration boundary (real HTTP, not a function call).

## 3. Technologies

| Concern | Choice |
|---|---|
| Language | Python 3.12+ |
| API framework | FastAPI + uvicorn |
| Validation | Pydantic v2 |
| HTTP | httpx (async) |
| LLM | Gemini `generateContent` REST API (default), optional `offline` rule-based provider for keyless demos |
| Config | python-dotenv |
| Storage (mock CMS) | SQLite via stdlib `sqlite3` |
| Tests | pytest + pytest-asyncio |
| Logging | stdlib logging (console + `output/pipeline.log`) |

Deliberately **not** used: LangChain, LangGraph, Kafka, Redis, Qdrant,
PostgreSQL, Kubernetes, any frontend framework.

## 4. Setup instructions

Follow these steps **after cloning the repository** to get the project running
on a fresh machine.

```bash
git clone <your-repo-url>.git
cd ai-agent-assignment
```

### 4.1 Python version

Python **3.12+** (developed and tested on 3.12/3.13).

```bash
python --version
```

### 4.2 Virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 4.3 Install dependencies

```bash
pip install -r requirements.txt
```

### 4.4 Create your `.env`

The repo ships only `.env.example` (secrets are git-ignored). Create the real
`.env` from it:

Windows (PowerShell):

```powershell
Copy-Item .env.example .env
```

macOS / Linux:

```bash
cp .env.example .env
```

Then open `.env` and paste your own Gemini key (see §6). Nothing else needs to
change for a first run.

### 4.5 Verify the install

```bash
pytest
```

You should see `113 passed`. Tests mock the LLM, so they pass **without** an
API key.

## 5. Environment variables

Copy the template (the repo ships only `.env.example`; `.env` is git-ignored):

```bash
cp .env.example .env
```

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | **Yes** (for real AI) | — | Gemini API key |
| `GEMINI_MODEL` | No | `gemini-flash-latest` | Any available Gemini Flash model id |
| `LLM_PROVIDER` | No | `gemini` | `gemini` or `offline` (keyless deterministic rules) |
| `LLM_TEMPERATURE` | No | `0.2` | Sampling temperature |
| `LLM_MAX_TOKENS` | No | `1024` | Max output tokens |
| `CMS_BASE_URL` | No | `http://localhost:8000` | Mock CMS base URL |
| `CMS_DB_PATH` | No | `./output/mock_cms.sqlite3` | CMS SQLite file |
| `LEADS_CSV_PATH` | No | auto-discover | Explicit path to the leads CSV |
| `FHIR_BASE_URL` | No | `https://r4.smarthealthit.org` | FHIR R4 sandbox |
| `FHIR_PATIENT_COUNT` | No | `20` | Patients to fetch (clamped to ≥ 15) |
| `FHIR_FETCH_CONDITIONS` | No | `false` | Also fetch `Condition` resources |
| `HTTP_TIMEOUT_SECONDS` | No | `20` | Per-request timeout |
| `MAX_RETRIES` | No | `3` | Bounded retry attempts |
| `RETRY_BACKOFF_SECONDS` | No | `1.0` | Exponential backoff base |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity |

`.env` is git-ignored; only `.env.example` is committed. **No secrets live in
the source code or the README.**

## 6. Gemini API key setup

1. Get a free key at <https://aistudio.google.com/apikey>.
2. Open `.env` and set:
   ```env
   GEMINI_API_KEY=your_key_here
   GEMINI_MODEL=gemini-flash-latest
   ```
3. Run the pipeline (see below).

If the key is missing or invalid the pipeline stops immediately with:

```text
[config error] GEMINI_API_KEY is missing. The pipeline cannot call the LLM.
  1. Get a free key at https://aistudio.google.com/apikey
  2. Copy .env.example to .env and set GEMINI_API_KEY=<your key>
  3. Re-run the pipeline.
For a fully offline demo without an LLM, set LLM_PROVIDER=offline.
```

There is **no** fallback that silently fabricates AI results.

### Free-tier quota behaviour (verified)

The Gemini free tier allows **20 `generateContent` requests per day, per
model, per project** (metric `GenerateRequestsPerDayPerProjectPerModel-FreeTier`;
verified live against `gemini-3.6-flash` and `gemini-flash-latest`). A run of
38 records therefore needs at least 38 free requests — more than one model's
daily allowance. When the quota runs out mid-run the pipeline:

1. classifies the 429 by its server `RetryInfo` hint — a hint longer than
   60 s (or daily-quota markers) is treated as hard quota exhaustion,
2. makes **no further API calls** for the rest of the run (per-run circuit),
3. logs `Record NOT processed by gemini ... quota` per record and reports
   `Gemini quota blocked ... N` in the summary,
4. **never** substitutes offline/hardcoded results for those records.

Short-hint 429s (per-minute rate limits) are still retried with exponential
backoff that honours the server's hint. To cover more than 20 records in one
day on the free tier, either split the run across two models (each has its
own daily 20) or wait for the quota window to reset — the error message
reports the exact wait (e.g. `retry after 4h46m42s`).

## 7. Running the mock CMS

```bash
uvicorn mock_cms.main:app --reload --port 8000
```

* Swagger UI: <http://localhost:8000/docs>
* Health: <http://localhost:8000/health>

Run it in a separate terminal and leave it running while the pipeline executes.

## 8. Running the pipeline

```bash
python -m app.main
```

Useful flags:

| Flag | Effect |
|---|---|
| `--csv PATH` | Use a specific leads CSV |
| `--cms-url URL` | Override `CMS_BASE_URL` |
| `--fhir-count N` | Number of FHIR patients (min 15) |
| `--limit N` | Process at most N records |
| `--provider gemini\|offline` | Override the LLM provider |
| `--skip-csv` / `--skip-fhir` / `--skip-cms` | Skip a stage |
| `--demo-failure` | Inject controlled failures (see §10) |

Example:

```bash
python -m app.main --provider gemini
python -m app.main --provider offline --limit 5     # quick keyless smoke run
python -m app.main --demo-failure                   # failure/retry demo
```

## 9. Running tests

```bash
pytest
```

Result in this workspace:

```text
113 passed in 1.54s
```

Tests cover CSV normalization, FHIR normalization, `UnifiedRecord`
validation, `AgentResult` validation, the mock CMS endpoints, the agent loop,
the CMS client and failure/retry behaviour. The LLM is **mocked** in unit
tests, so **no API key is required** to run them.

## 10. Example output

Run against the bundled data (18 CSV leads + 20 FHIR patients = 38 records).
The workspace snapshot below was produced with `--provider offline` because no
Gemini key was configured yet — with `GEMINI_API_KEY` set, the same run reports
`Provider ... gemini` and the messages/rationales come from the real model:

```text
==============================================================
 PIPELINE SUMMARY
==============================================================
 Total records ........... 38
 CSV leads loaded ........ 18
 FHIR patients fetched ... 20
 Normalized records ...... 38
 Successfully processed .. 38
 Failed .................. 0
 Malformed skipped ....... 0
 CMS delivered ........... 38
 CMS failed .............. 0
 Retries ................. 0
 LLM failures ............ 0
 Gemini quota blocked .... 0
 Provider ................ offline
 Duration ................ 5.14s
==============================================================
```

Artifacts written under `output/`:

* `processed_records.json` — record + `AgentResult` + delivery status per record
* `normalized_records.json` — every `UnifiedRecord` (raw payload preserved)
* `pipeline_summary.json` — the summary counters above
* `pipeline.log` — full structured run log
* `mock_cms.sqlite3` — downstream store

A downstream payload looks like this (structured JSON, never a text blob):

```json
{
  "record_id": "lead-001",
  "source": "linkedin",
  "record_type": "lead",
  "classification": "HOT",
  "priority": "HIGH",
  "action": "SEND_OUTREACH",
  "channel": "linkedin",
  "tone": "friendly",
  "message": "Hi Rebecca - saw your note about ...",
  "rationale": "High priority because the lead recently engaged with AI receptionist content and holds a healthcare operations role.",
  "processed_at": "2026-10-05T23:04:26+00:00",
  "model_provider": "gemini"
}
```

## 11. Failure / retry demonstration

One command exercises every failure path:

```bash
python -m app.main --demo-failure
```

What it injects and what you will see in the log:

| Injected failure | Handling | Log evidence |
|---|---|---|
| Malformed lead row | Skipped, run continues | `ERROR Record validation failed row=9999 ...` / `WARNING Skipping malformed record and continuing` |
| 2 malformed FHIR Patients | Skipped individually | `ERROR Record validation failed record_id=patient-...` |
| FHIR 503 on first attempt | Bounded retry + backoff, then success | `ERROR FHIR request failed attempt=1/3` → `WARNING Retrying request` → `INFO Request recovered on retry` |
| Invalid LLM JSON | Extract fails → repair prompt → second call | `WARNING LLM returned invalid structured output record_id=lead-001 attempt=1` → repaired on attempt 2 |
| Gemini 429 with short `RetryInfo` (per-minute rate limit) | Retried with backoff that honours the server hint | `WARNING Retrying request description=Gemini request sleep=...` → recovered |
| Gemini 429 with long `RetryInfo` (free-tier daily quota) | **No retries**; per-run circuit opens, remaining records fail loudly, no offline substitute | `ERROR Gemini quota exhausted - circuit opened ... retry after 4h46m42s` / `ERROR Record NOT processed by gemini - quota circuit open, API call skipped record_id=...` → summary `Gemini quota blocked .... N` |
| CMS 503 (transient) | Retried, then delivered | `ERROR CMS delivery failed attempt=1/3` → recovered |
| CMS 503 (persistent) | Exhausts 3 attempts, that record fails, run continues | `ERROR Giving up on CMS delivery after 3 attempts` / `WARNING Continuing with remaining records` |

Typical demo summary:

```text
 Total records ........... 38
 Successfully processed .. 38
 Malformed skipped ....... 3
 CMS delivered ........... 37
 CMS failed .............. 1
 Retries ................. 4
 LLM failures ............ 1
```

Errors are never silently swallowed: everything is logged, counted in the
summary and written to `output/pipeline_summary.json`.

## 12. API endpoints (mock CMS)

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service metadata + docs links |
| `GET` | `/health` | Liveness + stored record count |
| `POST` | `/records` | Store a validated `AgentResult`-like payload (201). Idempotent upsert by `record_id`. Header `X-Simulate-Error: once\|always` forces a 503 for demos |
| `GET` | `/records` | List stored records (`limit`, `classification` filters) |
| `GET` | `/records/{record_id}` | Fetch one record (404 if unknown) |

Interactive documentation: <http://localhost:8000/docs>

## 13. Project structure

```text
ai-agent-assignment/
├── app/
│   ├── main.py                 # CLI entry point + orchestration + artifacts
│   ├── config.py               # env-driven Settings (fail-fast validation)
│   ├── models/
│   │   ├── unified_record.py   # UnifiedRecord schema
│   │   └── agent_result.py     # AgentResult schema + contract validation
│   ├── ingestion/
│   │   ├── csv_loader.py       # robust CSV loader
│   │   └── fhir_client.py      # async FHIR R4 client w/ retries
│   ├── normalization/
│   │   ├── lead_normalizer.py
│   │   └── patient_normalizer.py
│   ├── agent/
│   │   ├── agent.py            # processing loop, JSON extraction, repair
│   │   ├── llm.py              # BaseLLMProvider + Gemini + offline + factory
│   │   └── prompts.py          # system/user/repair prompts
│   ├── cms/client.py           # async HTTP client to the mock CMS
│   └── utils/                  # logger, retry/backoff
├── mock_cms/                   # FastAPI mock CMS (main, db, schemas)
├── data/mock_leads.csv         # provided lead export (never modified)
├── output/                     # artifacts + SQLite + logs
├── tests/                      # 113 pytest tests
├── .env                        # your secrets (git-ignored)
├── .env.example
├── requirements.txt
├── ARCHITECTURE.md
└── README.md
```

## 14. Scaling considerations (≈10,000 records/day)

How this architecture would evolve — **not implemented here by design**:

* **Async → queue.** Replace the in-process loop with a worker queue
  (e.g. Celery/Arq/RQ over Redis or SQS). The agent loop is already
  `async` and per-record independent, so it maps directly onto tasks.
* **Horizontal workers.** 10k/day ≈ 7 records/minute average, but peaks
  matter: 8–16 workers with per-worker concurrency gives headroom without
  touching code beyond swapping `process_many` for queue consumers.
* **Rate limiting / batching.** Token-bucket per provider (Gemini free tier is
  ~10–15 RPM on some models) plus micro-batching (`generateContent` supports
  multiple candidates; batching also lowers cost).
* **Retries & dead-letter queue.** The existing bounded exponential backoff
  stays; after N attempts move the record to a DLQ with the full error context
  for replay instead of dropping it.
* **Idempotency.** `POST /records` already upserts by `record_id`; add an
  idempotency key + dedupe on `(record_id, model, prompt_version)` so replays
  never duplicate outreach.
* **Caching.** Cache FHIR pages and short-circuit identical record hashes to
  the previous `AgentResult` (prompt version in the key) — most re-runs are
  unchanged records.
* **Persistence.** Move from per-run JSON artifacts to PostgreSQL for
  records/results, object storage for raw payloads, and a time-series store
  for metrics.
* **Monitoring.** Structured (JSON) logs + OpenTelemetry traces around each
  stage; alerts on delivery failure rate, p95 latency, retry exhaustion and
  token spend.
* **LLM cost control.** Token accounting per run, per-record token budgets,
  prompt compression (send only `context`, never `raw_payload`), and a daily
  spend cap.
* **Model fallback.** Provider interface already isolates the model: add a
  fallback chain (Gemini Flash → local Ollama) with circuit breaking on 429/5xx.

## 15. Healthcare / PHI / HIPAA considerations

**This assignment uses 100% synthetic data** — SMART Health IT sandbox
patients, no real patient/PHI data, no real identifiers. **This demo is not
HIPAA compliant and does not claim to be.**

For a real healthcare deployment you would need:

* **HIPAA compliance program** — risk analysis, workforce training, policies.
* **BAA** with every vendor that could touch PHI (cloud host, FHIR vendor,
  and any LLM provider — many consumer LLM endpoints are *not* eligible).
* **Encryption** in transit (TLS 1.2+) and at rest (KMS-managed keys).
* **RBAC + minimum necessary access** — only the fields each role needs.
* **Audit logging** — who accessed which record, when, and why (immutable).
* **Secrets management** — vault/KMS, never `.env` in production.
* **PHI minimization / redaction** — ideally send the LLM a de-identified or
  field-subset payload; this codebase already sends only `context`/`person`
  to the model, never `raw_payload`, which is a useful first step.
* **Retention policies** — TTL on artifacts, logs and CMS rows; right to
  deletion.
* **Approved LLM architecture** — self-hosted or enterprise/BAA-covered
  endpoints, no-training guarantees, region pinning.

The system prompt explicitly forbids medical diagnosis or treatment advice:
patient outputs are administrative/workflow-only (`FOLLOW_UP`,
`SCHEDULE_OUTREACH`, `ADMIN_REVIEW`, `NO_ACTION`).

## 16. Security notes

* No API keys in source code; `.env` is git-ignored (`.env.example` only).
* Input validation with Pydantic on both the inbound CMS payload and the LLM
  output (`AgentResult` contract, enum enforcement).
* Timeouts and bounded retries on every outbound call.
* Safe logging: raw healthcare payloads are not dumped into logs; only ids,
  decisions and rationales.
* No real PHI, no LinkedIn scraping, no outbound calls other than the FHIR
  sandbox, Gemini API and the local CMS.
* The mock CMS binds to localhost by default.

## 17. Design trade-offs

| Decision | Why | Cost |
|---|---|---|
| `UnifiedRecord` with a flexible `context` object | Avoids one giant schema mixing LinkedIn and healthcare fields | Slightly weaker static typing inside `context` |
| Sequential per-record agent calls | Ordered logs, simple rate limiting, easy reasoning about failures | Lower throughput than concurrent calls (fine at this scale) |
| REST client instead of the Gemini SDK | Fewer dependencies, easy to swap/mock, explicit error mapping | We must maintain our own response parsing |
| SQLite for the mock CMS | Zero-ops, persistent, real SQL semantics | Single-writer, not for horizontal scale |
| `offline` provider | Lets tests/demos run with no key | Deterministic rules are **not** AI — loudly logged, never the default |
| Repair prompt instead of indefinite retries | Bounded cost, still recovers from transient model mistakes | A persistently bad output fails that one record |
| FHIR as a live dependency | Assignment requires the real public sandbox | Network hiccup → handled with retries, `--skip-fhir` available |

---

### Final self-check

`mock_leads.csv` ingestion · 15+ SMART FHIR patients · unified schema ·
lead + patient normalization · AI classification/action/channel/tone/message/
rationale · structured Pydantic output · real Gemini integration · error
handling · retry logic · deliberate failure case · logging · mock CMS · real
HTTP POST · structured downstream payload · 38 end-to-end records · output
artifacts · tests · README · architecture doc · scaling + PHI/HIPAA
discussion · `.env.example` · `.gitignore` · no secrets committed · Swagger ·
runnable from a clean environment.
