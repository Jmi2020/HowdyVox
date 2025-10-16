# Repository Guidelines

## Project Structure & Module Organization
- `voice_assistant/`: Core conversation loop, configuration, and wake-word handling. Adjust `config.py` for model, voice, and system prompt changes.
- `FastWhisperAPI/`: FastAPI speech-to-text microservice for local transcription.
- `ESP32/` and `ESP32P4_INTEGRATION.md`: Firmware, scripts, and docs for wireless microphone/LED integrations.
- Root launch scripts (`launch_howdy_ui.py`, `run_voice_assistant.py`, `launch_howdy_shell.py`) target different runtime profiles.
- Test utilities live next to features (`test_face_ui.py`, `run_tts_tests.py`, `Tests_Fixes/`) with assets in `test_audio/` and `test_output/`.

## Build, Test, and Development Commands
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python launch_howdy_ui.py
python run_voice_assistant.py
python test_fastapi.py
python run_tts_tests.py
```

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, and PEP 8 naming (`snake_case` functions, `PascalCase` classes).
- Prefer explicit module names that match hardware or speech domains (e.g., `microphone_manager.py`).
- Document public functions that touch audio buffers and concurrency-sensitive code.
- Before submitting, format touched files with `python -m black <file>`.

## Testing Guidelines
- Smoke-test speech loops with `python run_voice_assistant.py --dry-run` to verify pipeline stages.
- Validate voices via `python run_tts_tests.py --voices am_michael,am_bella`.
- For API regressions, run `uvicorn FastWhisperAPI.main:app --reload` and execute `python test_fastapi.py`.
- Name new test scripts `test_<feature>.py` and keep them beside their target module.

## Commit & Pull Request Guidelines
- Follow conventional prefixes observed in history (`fix:`, `feat:`, `chore:`, `docs:`), keeping subjects under 72 characters.
- One logical change per commit; include context in the body when touching audio timing or hardware behavior.
- Pull requests should summarize the user-facing impact, reference related issues or docs (e.g., `ECHOEAR_ENHANCEMENT_PLAN.md`), and attach logs or screenshots when UI changes are involved.
- Ensure all relevant diagnostics (`test_fastapi.py`, `run_tts_tests.py`) pass locally and note any skipped scenarios in the PR description.

## Security & Configuration Tips
- Store secrets in `.env` modeled after `example.env`; never commit actual keys.
- Validate microphone and speaker device indices with `python microphone_test.py` before enabling wake-word listening.
- Log ONNX upgrades in `VoiceModel/` so deployments remain reproducible.
