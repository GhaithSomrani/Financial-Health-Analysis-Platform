"""
Optimized SEC Company Data Fetcher
- Concurrent requests (ThreadPoolExecutor)
- Rate limiting (SEC allows ~10 req/sec)
- Automatic retries with exponential backoff
- Checkpoint/resume support (won't re-fetch already processed companies)
- Progress bar
"""

import requests
import pandas as pd
import time
import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Config ──────────────────────────────────────────────────────────────────
MAX_WORKERS      = 8       # Concurrent threads (keep ≤10 to respect SEC rate limit)
REQUESTS_PER_SEC = 8       # Stay just under SEC's 10 req/s limit
CHECKPOINT_FILE  = "./data/checkpoint.json"
OUTPUT_FILE      = "./data/company_tickers.csv"
LOG_FILE         = "./data/fetch.log"

# ── Logging ──────────────────────────────────────────────────────────────────
os.makedirs("./data", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ── Headers ──────────────────────────────────────────────────────────────────
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/145.0.0.0 Safari/537.36"

HEADERS_COMPANY = {
    "User-Agent": UA,
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate, br",
    "host": "www.sec.gov",
}
HEADERS_SUB = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "host": "data.sec.gov",
}

EXTRA_FIELDS = [
    "sicDescription",
    "exchanges",
    "operating",
    "ownerOrg",
    "insiderTransactionForOwnerExists",
    "insiderTransactionForIssuerExists",
]

# ── Rate limiter ─────────────────────────────────────────────────────────────
_semaphore = Semaphore(MAX_WORKERS)
_last_call_times: list[float] = []

def rate_limited_get(session: requests.Session, url: str, headers: dict) -> requests.Response:
    """Throttled GET — enforces REQUESTS_PER_SEC across all threads."""
    with _semaphore:
        # Simple token-bucket approach
        now = time.monotonic()
        _last_call_times[:] = [t for t in _last_call_times if now - t < 1.0]
        if len(_last_call_times) >= REQUESTS_PER_SEC:
            sleep_for = 1.0 - (now - _last_call_times[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        _last_call_times.append(time.monotonic())
        return session.get(url, headers=headers, timeout=30)

# ── Session with retry ───────────────────────────────────────────────────────
def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.5,          # 1.5s, 3s, 4.5s, 6s, 9s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session

# ── Checkpoint helpers ───────────────────────────────────────────────────────
def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}

def save_checkpoint(data: dict):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)

# ── Worker ───────────────────────────────────────────────────────────────────
def fetch_submission(session: requests.Session, index, cik: str) -> tuple:
    """Fetch submission data for one CIK. Returns (index, dict_of_fields)."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    result = {f: None for f in EXTRA_FIELDS}
    try:
        resp = rate_limited_get(session, url, HEADERS_SUB)
        resp.raise_for_status()
        sub = resp.json()
        for field in EXTRA_FIELDS:
            result[field] = sub.get(field)
    except requests.exceptions.HTTPError as e:
        log.warning(f"HTTP {e.response.status_code} for CIK {cik}")
    except Exception as e:
        log.warning(f"Failed CIK {cik}: {e}")
    return index, result

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # 1. Fetch company list
    log.info("Fetching company ticker list…")
    session = make_session()
    resp = session.get("https://data.sec.gov/files/company_tickers.json", headers=HEADERS_COMPANY, timeout=30)
    resp.raise_for_status()

    df = pd.DataFrame.from_dict(resp.json(), orient="index")
    df["cik_str"] = df["cik_str"].astype(str).str.zfill(10)
    for field in EXTRA_FIELDS:
        df[field] = None

    total = len(df)
    log.info(f"Total companies: {total:,}")

    # 2. Load checkpoint (resume support)
    checkpoint = load_checkpoint()
    already_done = set(checkpoint.keys())
    log.info(f"Already processed: {len(already_done):,} — resuming from checkpoint")

    # 3. Build work queue (skip already processed)
    work = [(idx, row["cik_str"]) for idx, row in df.iterrows() if row["cik_str"] not in already_done]
    log.info(f"Remaining to fetch: {len(work):,}")

    # 4. Apply checkpoint data back to df
    for cik, fields in checkpoint.items():
        matches = df[df["cik_str"] == cik].index
        for idx in matches:
            for field, val in fields.items():
                df.at[idx, field] = val

    # 5. Concurrent fetch with progress
    try:
        from tqdm import tqdm
        progress = tqdm(total=len(work), desc="Fetching submissions", unit="co")
    except ImportError:
        progress = None
        log.info("Install tqdm for a progress bar: pip install tqdm")

    SAVE_EVERY = 500  # Save checkpoint every N completions
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_submission, session, idx, cik): cik for idx, cik in work}

        for future in as_completed(futures):
            cik = futures[future]
            try:
                idx, fields = future.result()
                for field, val in fields.items():
                    df.at[idx, field] = val
                checkpoint[cik] = fields
                completed += 1

                if progress:
                    progress.update(1)

                # Periodic checkpoint save
                if completed % SAVE_EVERY == 0:
                    save_checkpoint(checkpoint)
                    df.to_csv(OUTPUT_FILE, index=False)
                    log.info(f"Checkpoint saved — {completed:,} done")

            except Exception as e:
                log.error(f"Unexpected error for CIK {cik}: {e}")

    if progress:
        progress.close()

    # 6. Final save
    save_checkpoint(checkpoint)
    df.to_csv(OUTPUT_FILE, index=False)
    log.info(f"Done! Saved {total:,} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()