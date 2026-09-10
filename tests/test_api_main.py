"""Tests for the application entry point: wiring, CORS, and the local runner.

API Layer Position:
    [MAIN.PY] → lifespan → RAGBackend + RunRegistry on app.state → routers

What concept it teaches:
    An entry point is a module like any other, and the parts of it that only
    run when you actually launch the process are exactly the parts nothing
    else covers. `python -m src.api.main` was documented in the README for
    months while doing nothing at all, because no test ever asked whether the
    module had a runner. These tests ask.

Why this approach over alternatives:
    We assert on the `__main__` guard's *shape* (it calls uvicorn.run with the
    configured host and port) rather than launching a real server. Binding a
    port in a unit test is slow, flaky under parallel runs, and tests uvicorn
    rather than this module.
"""

from __future__ import annotations

import runpy
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.config import API_HOST, API_PORT


class TestModuleRunner:
    """`python -m src.api.main` must actually start the server."""

    # WHY the filter: runpy re-executes an already-imported module, which is
    #     exactly what we want here and exactly what runpy warns about.
    @pytest.mark.filterwarnings("ignore:.*found in sys.modules.*:RuntimeWarning")
    def test_main_guard_runs_uvicorn_on_the_configured_address(self):
        # BUG FIX: this module previously had no `if __name__ == "__main__"`
        #          block. Running it imported the app and exited 0 — silently,
        #          so the documented dev command looked like it worked.
        with patch("uvicorn.run") as run:
            runpy.run_module("src.api.main", run_name="__main__")

        run.assert_called_once()
        args, kwargs = run.call_args
        assert args[0] == "src.api.main:app"
        assert kwargs["host"] == API_HOST
        assert kwargs["port"] == API_PORT


class TestLifespanWiring:
    """Entering the lifespan must populate everything the routes depend on."""

    def test_startup_populates_app_state(self):
        # The lifespan builds a *persistent* Chroma client and a real SQLite
        # file; conftest's session-wide `_isolate_app_state_dirs` points both
        # into tmp so no test writes to the developer's data/ directory.
        with TestClient(app) as client:
            assert client.app.state.backend is not None
            assert client.app.state.engine is not None
            assert client.app.state.run_registry is not None
            assert client.get("/health").json() == {"status": "healthy"}

    def test_health_is_reachable_without_lifespan_state(self):
        # /health must not depend on the backend — it is what a container
        # healthcheck hits, including while startup is still in progress.
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200

    def test_lifespan_never_opens_the_repository_data_directory(self):
        # BUG FIX: every test that entered `with TestClient(app)` ran the real
        #          lifespan against data/rag.db and data/chroma/ — the
        #          developer's own store. Confirmed by mtime: one route test
        #          rewrote data/chroma/chroma.sqlite3. Tests that swapped in a
        #          mock backend did so only after startup, too late to help.
        # WHY assert on the paths rather than on file mtimes: an mtime check is
        #      a race against anything else on the machine, and this states the
        #      invariant directly — the suite must never address data/.
        from src.api import main as api_main
        from src.config import DATA_DIR

        data_dir = str(DATA_DIR.resolve())
        assert not api_main.CHROMA_PATH.startswith(data_dir)
        assert data_dir not in api_main.SQLITE_URL


class TestCors:
    """CORS must be *configurable*, which is the part that used to be missing."""

    def test_cors_middleware_reads_the_configured_origins(self):
        # BEFORE: allow_origins=["*"] was hardcoded in this module, so the
        #         production image had no way to narrow it.
        # AFTER:  it comes from allowed_origins(), which reads $ALLOWED_ORIGINS.
        # WHY not assert it isn't "*": unset it deliberately still is, for local
        #      dev against a Vite server on another port. The deployed default
        #      is pinned in docker-compose.prod.yml, not here.
        from starlette.middleware.cors import CORSMiddleware

        from src.config import allowed_origins

        cors = [m for m in app.user_middleware if m.cls is CORSMiddleware]
        assert len(cors) == 1
        assert cors[0].kwargs["allow_origins"] == allowed_origins()

    def test_allowed_origins_parses_a_comma_separated_list(self, monkeypatch):
        from src.config import allowed_origins

        monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.example, https://b.example")
        assert allowed_origins() == ["https://a.example", "https://b.example"]

    def test_allowed_origins_falls_back_to_wildcard_when_unset(self, monkeypatch):
        from src.config import allowed_origins

        monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
        assert allowed_origins() == ["*"]
