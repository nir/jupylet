Extends: win-base

# A Windows 11 PC with Jupylet installed

Jupylet is installed for `{code}`. The base of the Windows systems where the
person already uses Jupylet. The outputs come from transcript W5, a real
installation on Windows 11 (`session-export-1793740590305`), the day before.

- Miniforge in `{home}\miniforge3`, installed for this user only, with
  `python.exe`. Windows lists it under the installed programs, so "Find
  conda installations" finds it in the uninstall entries and in the home
  folder.
- No other conda installation.
- The environment `{home}\miniforge3\envs\jupylet`, with Python 3.13.18,
  Jupylet 0.10.0 installed with `pip install -e` from `{code}`, the
  `[claude]` extra, and JupyterLab. "Prepare the environment" ran:
  `prepare --check` prints `ready`. `find-env 0.10.0` prints one line:
  `{home}\miniforge3\envs\jupylet	conda	this folder`.
- The example notebooks in `{code}\examples` are trusted.
- The local settings file `{code}\.jupylet\claude-config.toml` has the
  header of the settings files and the line `environment = "jupylet"`.
  There is no global settings file.
- No Jupyter server runs.

## Checks

```json
[]
```
