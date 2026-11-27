> Guidance for autonomous coding agents
> Read this before writing, editing, or executing anything in this repo.

# Relevant commands

* Run the read-only Stage 0 audit from the repository root in the `eeg2video` conda environment using
  `PYTHONPATH=src python -m eeg2video.audit --output reports/stage0`. Choose a new output directory for a rerun.
  The audit reads `VIDEO_DATA_DIR` and `STIMULUS_DIR` from the environment or the repository-local `.env`.
  Only use trusted study exports because the MNE copies and NPZ channel metadata use pickle.

* Stage 1 uses `python -m eeg2video.prepare`, `python -m eeg2video.partition --seed 42`,
  `python -m eeg2video.vision --film 3 --device cuda` and `python -m eeg2video.verify`.
  Run a second partition with seed 43 for the leakage check. Use `--base-only` to verify before GPU extraction.
  See README.md for the conda setup, pinned model downloads with `hf download`, and LSF job submissions.
* The project uses `pip` for management of packages. This means:
  * To install packages, use `pip install <package-name>`.
  * To run Python scripts, use `python <script-name>.py`.
  * To run other commands related to Python, use `python <command>`.
* The project uses `pytest` for testing. To run tests, use `pytest tests/`.
* The project uses `ruff` for linting and formatting:
    * To format code, use `ruff format .`.
    * To lint code, use `ruff check . --fix`.
* The project uses `invoke` for task management. To see available tasks, refer to the `tasks.py` file.
* The project uses `pre-commit` for managing pre-commit hooks. To run all hooks on all files, use
    `pre-commit run --all-files`. For more information, refer to the `.pre-commit-config.yaml` file.

# Code style

* Follow existing code style.
* Keep line length within 120 characters.
* Use f-strings for formatting.
* Use type hints
* Do not add inline comments unless absolutely necessary.

# Documentation

* If the project has a `docs/` folder, update documentation there as needed.
* In this case the project will be using `mkdocs` for documentation. To build the docs locally, use
    `uv run mkdocs serve`
* Use existing docstring style.
* Ensure all functions and classes have docstrings.
* Use Google style for docstrings.
* Update this `AGENTS.md` file if any new tools or commands are added to the project.
