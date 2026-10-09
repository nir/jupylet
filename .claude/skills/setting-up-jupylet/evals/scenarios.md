# Test scenarios for setting-up-jupylet

Method as in guiding-jupylet-users: Claude under test writes `ACTION:` lines
instead of running tools and `TO PERSON:` for messages; a second subagent plays
the person; the main session answers each ACTION from the script below. An
ACTION that the script does not cover gets `OK` with no output. When Claude
moves on to the spaceship check (or asks to open it and gets a yes), the run
ends: `running-jupylet-notebooks` does not exist yet.

## 1. mac-with-miniconda

- **Persona:** a parent, installing Jupylet for their 10-year-old. Not
  technical, polite, says yes to reasonable questions, asks what things are.
- **Computer:** macOS on an Apple M2. Home `/Users/sam`. Code in
  `/Users/sam/jupylet` (VERSION 0.10.0).
- **Machine script:**
  - system check: `Darwin` / `arm64`
  - conda folders: `/opt/miniconda3` (has `bin/conda`); no `~/miniforge3`
  - Miniforge Python version: `No such file or directory`
  - find-env with `/opt/miniconda3/bin/python`: no output
  - Terminal conda: `'/opt/miniconda3/bin/conda'`
  - Miniforge install: installer output ends `installation finished.`; `conda --version`: `conda 26.7.2`
  - zshrc backup: no output; `conda init zsh`: `modified /Users/sam/.zshrc`
  - envs in `~/miniforge3/envs`: none
  - create env: `done`; `python --version`: `Python 3.13.7`
  - pip install: ends `Successfully installed ... jupylet-0.10.0 ...`
  - check Jupylet: `0.10.0`; prepare: `turned off`
  - Terminal check: `/Users/sam/miniforge3`
  - is_trusted, before trust: every line `untrusted`
  - trust: `trusted` lines; is_trusted, after trust: every line `trusted`

## 2. windows-reinstall

- **Persona:** 14 years old, used Jupylet before with help, types lowercase,
  pulled the new version with git.
- **Computer:** Windows 11 (`AMD64`, build `22631`). Home `C:\Users\maya`.
  Code in `C:\Users\maya\jupylet` (VERSION 0.10.0).
- **Machine script:**
  - system check: `AMD64` / `22631`
  - conda folders: `C:\Users\maya\miniforge3` (from both commands)
  - Miniforge Python version: `Python 3.12.7`
  - find-env: `C:\Users\maya\miniforge3\envs\jupylet	conda	this folder`
  - create env (if chosen, name `jupylet2`): `done`, `Python 3.13.7`
  - pip install: ends `Successfully installed jupylet-0.10.0`
  - check Jupylet: `0.10.0`; prepare: `off`
  - is_trusted: every line `trusted` (already trusted); trust: `trusted` lines

## 3. mac-install-fails

- **Persona:** an adult beginner, curious, a bit anxious about breaking the
  computer. Says yes to Miniforge, but no to the change of the Terminal
  settings.
- **Computer:** macOS on an Apple M1. Home `/Users/alex`. Code in
  `/Users/alex/jupylet` (VERSION 0.10.0). No conda at all.
- **Machine script:**
  - system check: `Darwin` / `arm64`
  - conda folders: none; Miniforge Python: `No such file or directory`
  - Terminal conda: no output
  - Miniforge install: `installation finished.`; `conda 26.7.2`
  - create env: `done`, `Python 3.13.7`
  - pip install: fails with 40 lines ending
    `ERROR: Failed building wheel for glcontext` /
    `error: subprocess-exited-with-error`
