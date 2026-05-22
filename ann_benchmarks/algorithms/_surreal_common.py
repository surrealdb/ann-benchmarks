"""Shared helpers for the SurrealDB ann-benchmarks modules.

Not an algorithm of its own — the algorithm discovery in
`ann_benchmarks/definitions.py` globs `*/config.yml`, so this file is invisible
to it. It only exists to be imported by `surreal_hnsw/module.py`,
`surreal_diskann/module.py`, etc.
"""

import os
import subprocess
import sys
import threading
import time
from multiprocessing.pool import ThreadPool

import requests
from requests.adapters import HTTPAdapter


# Default localhost endpoint and credentials used by every Surreal* module.
_HOST = "127.0.0.1"
_PORT = 8000
_SQL_URL = f"http://{_HOST}:{_PORT}/sql"
_HEALTH_URL = f"http://{_HOST}:{_PORT}/health"
_AUTH = ("ann", "ann")
_HEADERS = {
    "surreal-ns": "main",
    "surreal-db": "main",
    "Accept": "application/json",
}

# Where the rocksdb backend persists data. Relative path is fine — the
# benchmark always runs from the repo root.
DEFAULT_DB_DIR = "mydata/ann.db"


def _any_surreal_running() -> bool:
    """Return True iff `pgrep -x surreal` finds at least one process."""
    return subprocess.run(
        ["pgrep", "-x", "surreal"], capture_output=True
    ).returncode == 0


def stop_server(timeout: float = 10.0) -> None:
    """Send SIGTERM to any running `surreal`, wait for it to exit, escalate to SIGKILL.

    Best-effort: returns cleanly if nothing was running. Raises if a process
    refuses to die even after SIGKILL within the timeout.
    """
    if not _any_surreal_running():
        return
    subprocess.run(["pkill", "-TERM", "-x", "surreal"], check=False)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _any_surreal_running():
            return
        time.sleep(0.1)
    # Still alive after TERM — escalate.
    subprocess.run(["pkill", "-KILL", "-x", "surreal"], check=False)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _any_surreal_running():
            return
        time.sleep(0.1)
    raise RuntimeError("`surreal` process survived SIGTERM and SIGKILL")


def start_server(
    use_rocksdb: bool = True,
    db_dir: str = DEFAULT_DB_DIR,
    timeout: float = 30.0,
) -> subprocess.Popen:
    """Start a fresh SurrealDB server and block until it accepts requests.

    - Wipes any prior rocksdb directory so a new index won't collide with a
      leftover one from the previous algorithm definition.
    - Uses Popen (not `surreal start ... &`) so a failed bind / bad flag
      surfaces as a non-zero exit instead of being swallowed by the shell.
    - Polls `/health` until 200 (the server's defaults + credentials are
      initialised *before* the HTTP listener starts, so a healthy port means
      the ns / db / user are ready too).
    """
    if use_rocksdb:
        subprocess.run(["rm", "-rf", db_dir], check=True)
        backend = f"rocksdb:{db_dir}"
    else:
        backend = "memory"

    cmd = [
        "surreal", "start", "--allow-all",
        "--default-namespace", "main",
        "--default-database", "main",
        "-u", "ann", "-p", "ann",
        "-b", f"{_HOST}:{_PORT}",
        backend,
    ]
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # If the server bailed (port already bound, bad CLI, etc.) fail fast
        # instead of waiting out the whole timeout.
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(f"`surreal start` exited prematurely with code {rc}")
        try:
            r = requests.get(_HEALTH_URL, timeout=0.5)
            if r.status_code == 200:
                return proc
        except requests.RequestException:
            pass
        time.sleep(0.1)

    # Timed out — try not to leak the process before raising.
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    raise RuntimeError(f"SurrealDB did not become healthy within {timeout}s")


def _new_session() -> requests.Session:
    """Build a Session with auth/headers and a single keep-alive connection."""
    s = requests.Session()
    s.auth = _AUTH
    s.headers.update(_HEADERS)
    # One persistent TCP connection per session — sessions are per-thread, so
    # there is never any contention on the pool.
    adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=0)
    s.mount("http://", adapter)
    return s


class SurrealBatchMixin:
    """Add efficient `--batch` support to a Surreal* algorithm.

    Subclasses must implement `_build_query_sql(v, n)` returning a single
    SurrealQL statement string (terminated by `;`) for one query.

    The runner calls `prepare_batch_query` (untimed) then `run_batch_query`
    (timed) then `get_batch_results`. We fan out concurrent HTTP requests via a
    ThreadPool, with one Session per worker thread to avoid the default
    Session's bounded connection pool serializing requests.
    """

    # Thread-local storage holding one Session per worker thread.
    _tls = threading.local()

    # Cap the pool a little above cpu_count; localhost ANN queries are mostly
    # CPU-bound on the server side, so over-subscribing just adds noise.
    _max_workers = min(32, (os.cpu_count() or 8))

    def _build_query_sql(self, v, n):
        raise NotImplementedError(
            "Surreal* subclasses must implement _build_query_sql(v, n)"
        )

    def _thread_session(self) -> requests.Session:
        s = getattr(self._tls, "session", None)
        if s is None:
            s = _new_session()
            self._tls.session = s
        return s

    def prepare_batch_query(self, X, n) -> None:
        """Precompute SQL strings. Excluded from the timed window."""
        self._prepared = [self._build_query_sql(v.tolist(), n) for v in X]

    def run_batch_query(self) -> None:
        """Send all prepared queries concurrently. This is the timed window."""
        def one(q):
            s = self._thread_session()
            r = s.post(_SQL_URL, q)
            if r.status_code != 200:
                raise RuntimeError(r.text)
            res = r.json()
            block = res[0]
            if block.get("status") != "OK":
                raise RuntimeError(f"Error: {block}")
            # ids look like "items:1234" — strip the 6-char "items:" prefix.
            return [int(item["id"][6:]) for item in block["result"]]

        with ThreadPool(processes=self._max_workers) as pool:
            self._batch_results = pool.map(one, self._prepared)

    def get_batch_results(self):
        return self._batch_results
