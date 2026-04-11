```bash
# Install dependencies
uv sync

# Run all tests (do this before committing)
.venv/Scripts/python.exe -m pytest tests/ -v

# Fast feedback loop (unit tests only, ~8 s)
.venv/Scripts/python.exe -m pytest tests/unit/ -v

# Exclude slow tests (standard development loop)
.venv/Scripts/python.exe -m pytest tests/ -v -m "not slow"

# Targeted subsystem test (fastest, ~0.04 s)
.venv/Scripts/python.exe -m pytest tests/unit/rocket_engine/subsystems/test_injector.py -v

# CLI smoke test
thor --version
thor info

# Run characterization scripts (standalone, no pytest dependency)
.venv/Scripts/python.exe -u scripts/characterize_<name>.py
```

- Use ```hypothesis``` for property testing
