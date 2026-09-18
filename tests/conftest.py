"""Test configuration.

The environment variables must be set BEFORE any app module is imported,
because `get_settings()` is `@lru_cache`d - the first call freezes the settings
for the whole process. pytest imports conftest.py ahead of the test modules,
which makes this the right place to do it.

Without this, running the test suite would write into the developer's real
storage/chroma directory and pollute their index.
"""

import os
import tempfile
from pathlib import Path

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="rag-tests-"))

os.environ["UPLOAD_DIR"] = str(_TEST_ROOT / "uploads")
os.environ["CHROMA_DIR"] = str(_TEST_ROOT / "chroma")
os.environ["API_KEY"] = ""  # disable the API-key middleware for tests
# A dummy key keeps the Anthropic client constructible offline. No test makes a
# real request - generation is always stubbed - so it is never used to auth.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

(_TEST_ROOT / "uploads").mkdir(parents=True, exist_ok=True)
(_TEST_ROOT / "chroma").mkdir(parents=True, exist_ok=True)
