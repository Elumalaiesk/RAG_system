# Insurance RAG Service

A FastAPI service that ingests insurance policy PDFs and answers questions about
them, with every claim traced back to a page. Built as a learning project, so
the code is commented to explain *why* each step is the way it is, not just what
it does.

Embeddings run locally (free, offline, no key). Only answer generation calls a
provider API - **Gemini** or **Claude**, selected with one environment variable.

---

## How RAG works here

A language model does not know your policy documents. RAG fixes that by finding
the relevant clauses first and putting them in the prompt. Two pipelines:

**Ingestion** (once per document)

```
PDF -> parse -> chunk -> embed -> store
     page-aware  overlapping  384-dim   ChromaDB
                 + citable    vectors   on disk
```

**Query** (once per question)

```
question -> embed -> nearest-neighbour search -> filter weak hits
                                                        |
                        no hits -> decline (no API call, no cost)
                        hits    -> numbered context -> Claude -> cited answer
```

The chain is only as strong as its weakest link, and the weak link is almost
never the model. If the right clause is not retrieved, no prompt can save the
answer. That is why `scripts/evaluate.py` exists.

---

## Setup

```powershell
cd rag-insurance-service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env
```

Then set the provider and its key in `.env`:

```ini
# Gemini (free tier available)
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
GEMINI_API_KEY=AIza...        # https://aistudio.google.com/apikey

# ...or Claude
# LLM_PROVIDER=anthropic
# LLM_MODEL=claude-opus-5
# ANTHROPIC_API_KEY=sk-ant-...
```

**On Gemini's free tier, only Flash models work.** Verified against the live
endpoint: `gemini-3.1-pro-preview` and `gemini-pro-latest` return 429 with zero
quota, and `gemini-2.5-pro` returns 404 "no longer available to new users" while
still appearing in the model list. `gemini-2.5-flash` works without billing.

Switching providers changes nothing about retrieval - parsing, chunking,
embedding and search are identical and run locally. Only who writes the final
answer changes, which makes it easy to compare two models against the same index
and the same prompt. `/chat/ask` returns a `model` field so answers stay
attributable.

The first run downloads the embedding model (~90 MB) to your HuggingFace cache.
After that it works offline.

## Index your documents

```powershell
python scripts/ingest.py path\to\your\policies
python scripts/ingest.py path\to\your\policies --reset   # rebuild from scratch
```

Use `--reset` whenever you change `EMBEDDING_MODEL`. Vectors from two different
models are not comparable, and mixing them silently corrupts retrieval.

## Run

```powershell
uvicorn app.main:app --reload
```

- `http://127.0.0.1:8000/` - **test console** (upload PDFs, ask questions, inspect retrieved chunks and scores)
- `http://127.0.0.1:8000/docs` - interactive API docs
- `http://127.0.0.1:8000/health` - shows how many chunks are indexed

**Open `http://127.0.0.1:8000/` — do not double-click the HTML file.** Served
from the app, the console is same-origin with the API and its relative fetches
just work.

If you do need it elsewhere (opened from disk, or behind VS Code Live Server),
the **API box in the page header** takes an absolute base URL such as
`http://127.0.0.1:8000`, saved in `localStorage`. That origin must also appear in
`CORS_ORIGINS` in `.env`, or the browser blocks every request. The shipped list
covers localhost ports 8000/5500/3000 plus the `null` origin a `file://` page
sends.

If you set `API_KEY` in `.env`, run `localStorage.setItem("apiKey", "your-key")`
once in the browser devtools console so the page can authenticate.

## The API, and who each part is for

The chat request carries only what a client can actually supply. `top_k` and the
relevance floor are the server's decisions - a chat UI has no basis for choosing
5 over 12, and accepting it would let any caller quadruple the cost of every
request. The tuning knobs live on a separate, model-free surface instead.

**`POST /api/v1/chat/ask`** - for a chatbot:

```jsonc
// request - a question, and optionally which thread and which policy
{ "query": "What is the excess on a claim?",
  "conversation_id": null,     // omit on turn 1; the server mints one
  "document_id": null }        // omit to search everything

// response
{ "answer": "A compulsory excess of 5,000 applies [2].",
  "sources": [ { "number": 2, "citation": "policy.pdf p.12", "page_start": 12,
                 "score": 0.63, "excerpt": "..." } ],
  "conversation_id": "e4dbeec7-...",
  "grounded": true,
  "diagnostics": { "model": "gemini-2.5-flash", "input_tokens": 1749,
                   "output_tokens": 68, "top_k": 5, "min_similarity": 0.1,
                   "invalid_citations": [] } }
```

`sources` is user-facing - it is the reader's proof. Everything a chat UI can
ignore is grouped under `diagnostics`, so rendering an answer never requires
knowing what a cosine floor is.

**`POST /api/v1/retrieval/search`** - for tuning. Runs **no model**, so it costs
nothing:

```jsonc
{ "query": "Is it different?", "top_k": 10, "apply_floor": false }
```

Reach for this whenever an answer looks wrong. There are only two possible
causes - the right clause was never retrieved, or it was retrieved and the model
mishandled it - and they need completely different fixes. This tells you which,
for free, in one request. `apply_floor: false` is the key diagnostic: it shows
what `MIN_SIMILARITY` is throwing away, because a correct chunk sitting just
below the floor and an empty index look identical from `/chat/ask`.

## Test

```powershell
pytest
```

72 tests, no API calls, no cost. Generation is stubbed; retrieval is exercised
for real against a PDF generated at test time by `tests/pdf_fixture.py`.

---

## Tuning retrieval

This is the part worth spending time on.

```powershell
python scripts/evaluate.py            # score the current index
python scripts/evaluate.py --sweep    # compare chunk sizes
```

Write your own questions in `eval/questions.json` - one entry per question, with
substrings that must appear in a correctly retrieved chunk:

```json
{
  "question": "How long do I have to report an accident?",
  "expect_any": ["within 48", "48 hours"],
  "expect_pages": [3]
}
```

Twenty of these from your real policies turns "the answers feel wrong" into a
number you can move. The script reports:

- **hit@k** - how often the right chunk was retrieved at all
- **MRR** - whether it came back *first* (where the model actually reads it)
- **the lowest similarity among correct hits** - set `MIN_SIMILARITY` below this,
  or you will refuse questions your index can answer

---

## What each module does

| File | Role |
|---|---|
| `services/parser.py` | PDF -> per-page text. Detects scanned PDFs that need OCR. |
| `services/chunker.py` | Text -> overlapping chunks, sized in TOKENS, snapped to clause boundaries, carrying page numbers. |
| `services/embedder.py` | Chunks -> normalized 384-dim vectors, locally. |
| `services/vector_db.py` | ChromaDB persistence and nearest-neighbour search. |
| `services/pipeline.py` | Joins the four steps above into one ingest call. |
| `services/rag_engine.py` | Retrieve, guard, then ask the LLM for a cited answer; validates citation markers. |
| `services/llm.py` | Provider seam: `AnthropicLLM` and `OpenAICompatibleLLM` (Gemini, Ollama, vLLM). |
| `api/v1/endpoints/chat.py` | Chatbot surface: question in, cited answer out. No knobs. |
| `api/v1/endpoints/retrieval.py` | Developer surface: retrieval only, no model call, free. |
| `services/registry.py` | On-disk catalogue of what has been ingested. |
| `prompts/templates.py` | The grounding rules that keep answers in the document. |
| `static/index.html` | Browser test console, served at `/`. |

---

## Known limits

Honest list of what this does not do yet.

1. **No OCR.** Scanned policies are rejected with a clear message rather than
   silently indexing nothing. Add Tesseract to handle them.
2. **Ingestion is synchronous.** A large PDF blocks its request. Fine for
   learning; a production version needs a task queue.
3. **No multi-tenancy.** `document_id` scopes a search, but nothing enforces who
   may pass which id, so any caller can read any indexed policy. Retrieval scope
   is properly a property of *who is asking*, not of the request body. There is
   one place to fix it - `resolve_document_scope` in `api/deps.py` - which
   should derive the permitted ids from the authenticated principal and ignore
   the request entirely. Until then, do not put more than one party's documents
   behind this.
4. **No conversation memory - and follow-ups degrade silently.** The server
   mints a `conversation_id` and threads it, but it does not yet affect
   retrieval: each question is embedded on its own. Measured on the 50-page
   policy bond, the top similarity falls from 0.807 for "What is the grace
   period for premium payment?" to 0.407 for "What about for quarterly?" and
   0.217 for "Is it different?" - which retrieves *"the presence of bacterial
   infection in cerebrospinal fluid"*, clears the 0.10 floor, and is handed to
   the model as policy context. Rewriting it back to a standalone question
   recovers 0.785.

   The fix is not storage, it is **query rewriting**: an LLM call that turns
   (history + follow-up) into a standalone search query before retrieval. That
   costs an extra call per turn, which is why it is not in yet. Until then,
   treat each question as independent and phrase it fully.
5. **Dense retrieval only.** Exact identifiers (a policy number, "Section 4.2")
   are a known weakness of embeddings. Adding BM25 keyword search alongside and
   merging the results is the standard fix.
6. **The relevance floor barely separates signal from noise.** Measured on a
   50-page policy bond, correct chunks score as low as 0.153 while an unrelated
   question ("capital of France?") reaches 0.114 - a usable window only 0.039
   wide. `MIN_SIMILARITY` is set to 0.10, below both: it never falsely refuses a
   valid question, but an occasional junk query still reaches the model, where
   the grounding prompt is what declines it. The window narrows as the corpus
   grows, because more chunks mean more chances of a spurious match while
   correct-hit scores stay put. A stronger embedding model is the real fix if
   this matters; `scripts/evaluate.py` on YOUR questions is how you decide.
7. **Citation markers are validated, not guaranteed.** Gemini was observed
   citing `[5, 14]` when only 4 extracts were supplied - it had lifted those
   numbers from the policy's own clause numbering. The prompt now forbids it and
   the response carries an `invalid_citations` list when it happens anyway, which
   the console surfaces as "unverified citation". Detection, not prevention: a
   prompt is a request, not a guarantee.
8. **`core/security.py` is currently unused** - the API-key check lives in the
   middleware in `main.py`. Keep it if you want per-route enforcement later,
   otherwise delete it.

## Cost

Embedding and retrieval are free and run locally. Only `/chat/ask` and
`/chat/ask/stream` call a provider; a typical question sends roughly 1.3-1.6k
input tokens (measured on the 50-page policy bond at `top_k=4`-`5`) and returns
50-110 output tokens.

Gemini's free tier costs nothing but is rate-limited to a few requests per
minute. `LLM_EFFORT` applies to Anthropic only - the OpenAI-compatible layer
exposes no equivalent dial.

Note `LLM_MAX_TOKENS` is generous on purpose: on **both** providers reasoning
tokens are drawn from the same allowance, so too small a value returns an empty
answer rather than an error. The engine raises a clear message if that happens.

Every response reports its own `input_tokens` / `output_tokens`, and off-topic
questions cost **zero** - retrieval declines before any model is called.
