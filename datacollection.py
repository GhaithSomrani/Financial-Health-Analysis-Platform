"""
Optimized SEC Company Data Fetcher
- Concurrent requests (ThreadPoolExecutor)
- Thread-safe rate limiting with RLock (fixes 429 race condition)
- Adaptive throttling: auto-slows on 429, auto-recovers after success streak
- Global pause: when any thread hits 429, ALL threads pause
- Automatic retries with exponential backoff
- Checkpoint/resume support
- Progress bar
"""

import requests
import pandas as pd
import time
import os
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import deque

# ── Config ───────────────────────────────────────────────────────────────────
MAX_WORKERS        = 5       # Start conservative — increase if no 429s
REQUESTS_PER_SEC   = 5.0     # Initial rate (SEC allows ~10/s but be safe)
MIN_RATE           = 1.0     # Floor: never go below 1 req/s
MAX_RATE           = 9.0     # Ceiling: never exceed 9 req/s
BACKOFF_FACTOR     = 0.5     # Multiply rate by this on 429  (5 → 2.5)
RECOVERY_FACTOR    = 1.1     # Multiply rate by this after N successes
RECOVERY_AFTER     = 50      # Successes before speeding back up
CHECKPOINT_FILE    = "./data/checkpoint.json"
OUTPUT_FILE        = "./data/company_tickers.csv"
LOG_FILE           = "./data/fetch.log"

EXTRA_FIELDS = [
    "sicDescription",
    "exchanges",
    "operating",
    "ownerOrg",
    "insiderTransactionForOwnerExists",
    "insiderTransactionForIssuerExists",
]

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
HEADERS_COMPANY = {"User-Agent": UA, "Accept": "application/json",        "Accept-Encoding": "gzip, deflate, br", "host": "www.sec.gov"}
HEADERS_SUB     = {"User-Agent": UA, "Accept": "application/xhtml+xml,*/*;q=0.8", "Accept-Encoding": "gzip, deflate, br", "host": "data.sec.gov"}


# ── Thread-safe adaptive rate limiter ────────────────────────────────────────
class AdaptiveRateLimiter:
    """
    Sliding-window token bucket with adaptive rate control.
    - Uses RLock so only ONE thread passes through at a time -> no race conditions.
    - On 429: immediately pauses all threads + halves the rate.
    - On sustained success: gradually speeds back up.
    """

    def __init__(self, initial_rate: float):
        self._lock           = threading.RLock()
        self._rate           = initial_rate
        self._timestamps     = deque()
        self._pause_until    = 0.0
        self._success_streak = 0

    def acquire(self):
        """Block until it's safe to make a request."""
        with self._lock:
            # 1. Respect global pause triggered by 429
            pause_remaining = self._pause_until - time.monotonic()
            if pause_remaining > 0:
                time.sleep(pause_remaining)

            # 2. Sliding-window throttle
            while True:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= 1.0:
                    self._timestamps.popleft()

                if len(self._timestamps) < self._rate:
                    break

                sleep_for = 1.0 - (now - self._timestamps[0]) + 0.001
                time.sleep(sleep_for)

            self._timestamps.append(time.monotonic())

    def on_success(self):
        with self._lock:
            self._success_streak += 1
            if self._success_streak >= RECOVERY_AFTER:
                new_rate = min(self._rate * RECOVERY_FACTOR, MAX_RATE)
                if new_rate > self._rate:
                    log.info(f"Rate increased -> {new_rate:.1f} req/s")
                self._rate = new_rate
                self._success_streak = 0

    def on_429(self, retry_after=None):
        with self._lock:
            self._success_streak = 0
            self._rate = max(self._rate * BACKOFF_FACTOR, MIN_RATE)
            wait = retry_after if retry_after else 60
            self._pause_until = time.monotonic() + wait
            log.warning(f"429 received — pausing {wait}s, new rate -> {self._rate:.1f} req/s")


# ── Session with retry ───────────────────────────────────────────────────────
def make_session():
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=2.0,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def get_with_429_retry(session, url, headers, max_retries=8):
    """GET with manual 429 retry — honors Retry-After header, falls back to exponential backoff."""
    for attempt in range(1, max_retries + 1):
        resp = session.get(url, headers=headers, timeout=30)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 0)) or min(30 * attempt, 300)
            log.warning(f"429 on {url} — waiting {wait}s (attempt {attempt}/{max_retries})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError(f"Giving up on {url} after {max_retries} attempts due to 429")


# ── Checkpoint helpers ───────────────────────────────────────────────────────
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}

def save_checkpoint(data):
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, CHECKPOINT_FILE)  # atomic write


# ── Worker ───────────────────────────────────────────────────────────────────
def fetch_submission(session, limiter, index, cik, max_retries=5):
    url    = f"https://data.sec.gov/submissions/CIK{cik}.json"
    result = {f: None for f in EXTRA_FIELDS}

    for attempt in range(1, max_retries + 1):
        limiter.acquire()
        try:
            resp = session.get(url, headers=HEADERS_SUB, timeout=30)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 0)) or None
                limiter.on_429(retry_after)
                continue  # retry after global pause

            resp.raise_for_status()
            sub = resp.json()
            for field in EXTRA_FIELDS:
                result[field] = sub.get(field)
            limiter.on_success()
            return index, result

        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            log.warning(f"HTTP {code} for CIK {cik} (attempt {attempt})")
            if attempt < max_retries:
                time.sleep(2 ** attempt)

        except Exception as e:
            log.warning(f"Error CIK {cik} attempt {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    log.error(f"Giving up on CIK {cik} after {max_retries} attempts")
    return index, result


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    log.info("Fetching company ticker list...")
    session = make_session()
    resp = get_with_429_retry(session, "https://data.sec.gov/files/company_tickers.json", HEADERS_COMPANY)

    df = pd.DataFrame.from_dict(resp.json(), orient="index")
    df["cik_str"] = df["cik_str"].astype(str).str.zfill(10)
    for field in EXTRA_FIELDS:
        df[field] = None

    total = len(df)
    log.info(f"Total companies: {total:,}")

    checkpoint   = load_checkpoint()
    already_done = set(checkpoint.keys())
    log.info(f"Already processed: {len(already_done):,} — resuming")

    for cik, fields in checkpoint.items():
        for idx in df[df["cik_str"] == cik].index:
            for field, val in fields.items():
                df.at[idx, field] = val

    work = [(idx, row["cik_str"]) for idx, row in df.iterrows() if row["cik_str"] not in already_done]
    log.info(f"Remaining: {len(work):,}")

    try:
        from tqdm import tqdm
        progress = tqdm(total=len(work), desc="Fetching", unit="co")
    except ImportError:
        progress = None

    limiter    = AdaptiveRateLimiter(initial_rate=REQUESTS_PER_SEC)
    SAVE_EVERY = 500
    completed  = 0
    lock       = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_submission, session, limiter, idx, cik): cik
            for idx, cik in work
        }

        for future in as_completed(futures):
            cik = futures[future]
            try:
                idx, fields = future.result()
                with lock:
                    for field, val in fields.items():
                        df.at[idx, field] = val
                    checkpoint[cik] = fields
                    completed += 1

                    if progress:
                        progress.update(1)

                    if completed % SAVE_EVERY == 0:
                        save_checkpoint(checkpoint)
                        df.to_csv(OUTPUT_FILE, index=False)
                        log.info(f"Checkpoint saved — {completed:,}/{len(work):,} done")

            except Exception as e:
                log.error(f"Unexpected error CIK {cik}: {e}")

    if progress:
        progress.close()

    save_checkpoint(checkpoint)
    df.to_csv(OUTPUT_FILE, index=False)
    log.info(f"Done! {total:,} rows -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()