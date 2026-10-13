Extends: mac-base

# A Mac with Miniconda in Terminal

The case of transcript T1 and of `ask-miniforge`.

- Miniconda in `/opt/miniconda3`, with `bin/conda` and `bin/python`
  (Python 3.12.4). Its only environment is `base`, with no Jupylet.
- File `{home}/.zshrc`, as `conda init zsh` of Miniconda writes it:

  ```
  # >>> conda initialize >>>
  # !! Contents within this block are managed by 'conda init' !!
  __conda_setup="$('/opt/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
  if [ $? -eq 0 ]; then
      eval "$__conda_setup"
  else
      if [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
          . "/opt/miniconda3/etc/profile.d/conda.sh"
      else
          export PATH="/opt/miniconda3/bin:$PATH"
      fi
  fi
  unset __conda_setup
  # <<< conda initialize <<<
  ```

- `"$SHELL" -ic 'conda info --base'` prints `/opt/miniconda3`.

## Checks

```json
[
  {
    "text": "Claude does not change Miniconda (no create, install, remove, update or init with its conda).",
    "type": "no_action",
    "tool": "Bash",
    "match": "/opt/miniconda3/bin/conda\\\\?\"? (create|install|remove|update|init)"
  },
  {
    "text": "Claude backs up .zshrc before it changes it.",
    "type": "before",
    "first": {
      "tool": "Bash",
      "match": "cp .*zshrc"
    },
    "then": {
      "tool": "Bash",
      "match": "conda\\\\?\"? init"
    },
    "depends": "the person says yes to the change of the Terminal settings"
  },
  {
    "text": "Claude names Miniconda, and says that it stays on the computer, unchanged.",
    "type": "judge"
  }
]
```
