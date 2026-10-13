Extends: mac-base

# A Mac with Jupylet installed

Jupylet is installed for `{code}`. The base of the systems where the
person already uses Jupylet.

- Miniforge in `{home}/miniforge3`, with Python 3.12.7 in `base`.
- The environment `{home}/miniforge3/envs/jupylet`, with Python 3.13,
  Jupylet 0.10.0 installed with `pip install -e` from `{code}`, the
  `[claude]` extra, and JupyterLab. "Prepare the environment" ran:
  `prepare --check` prints `ready`.
- The example notebooks in `{code}/examples` are trusted.
- File `{home}/.zshrc`: the `conda init zsh` block of Miniforge, with the
  path `{home}/miniforge3`. `"$SHELL" -ic 'conda info --base'` prints
  `{home}/miniforge3`.

## Checks

```json
[]
```
