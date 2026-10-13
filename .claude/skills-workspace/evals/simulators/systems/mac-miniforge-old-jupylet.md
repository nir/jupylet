Extends: mac-base

# A Mac with Miniforge and an older Jupylet

The case of transcripts T2 and T7: Jupylet was installed before, from
another folder, and the person now downloaded a new copy into `{code}`.

- Miniforge in `{home}/miniforge3`, with Python 3.12.7 in `base`.
- The environment `{home}/miniforge3/envs/jupylet`, with Python 3.13 and
  Jupylet 0.9.5 installed with `pip install -e` from `{home}/jupylet`.
- The folder `{home}/jupylet`: an older copy of the Jupylet code, with
  `VERSION = '0.9.5'`.
- File `{home}/.zshrc`: the `conda init zsh` block of Miniforge, with the
  path `{home}/miniforge3`. `"$SHELL" -ic 'conda info --base'` prints
  `{home}/miniforge3`.
- `find-env --all 0.10.0` with the Python of Miniforge prints nothing:
  no environment has Jupylet 0.10.0.

## Checks

```json
[
  {
    "text": "Claude does not install into the old environment jupylet.",
    "type": "no_action",
    "tool": "Bash",
    "match": "envs/jupylet/bin/python\\\\?\"? -m pip install"
  },
  {
    "text": "Claude does not install Miniforge again.",
    "type": "no_action",
    "tool": "Bash",
    "match": "Miniforge3[^ \"]*\\.sh\\\\?\"? -b"
  },
  {
    "text": "Claude tells the person that the old environment jupylet stays as it is.",
    "type": "judge"
  }
]
```
