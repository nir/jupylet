# Windows 11

Tested on one Windows 11 Home computer with the PowerShell tool of the
Claude app. Not tried: Windows 10, ARM, a folder with spaces or in OneDrive.

## Contents

- General changes
- Find the Jupyters that run, and find the environment
- Start Jupyter
- Call the tools
- Start Jupylet without Claude
- Stop Jupyter

## General changes

- Run each command with the PowerShell tool. Never use the Bash tool or
  `$SHELL -ic`, even if they exist.
- Put `&` in front of a command that starts with a program path in quotes,
  for example `& "{python}" -m jupylet.claude ...`.
- `{python}` is `{env}\python.exe`.
- Run `python -m jupylet.claude` from `{code}` (`Set-Location "{code}"`
  first). From another folder, Python may not find `jupylet.claude`. The
  PowerShell tool keeps the current folder between commands.
- Windows PowerShell 5.1 removes the double quotes inside an argument that
  it gives to a program. For a long `python -c` command, write a `.py` file
  into the scratchpad, and run that file.
- A free port prints `10061` in step 2 in `start-in-app.md`. Only `0` means
  that the port is in use.
- The monitoring commands run with `run_in_background`, from `{code}`:
  `Set-Location "{code}"; & "{python}" -m jupylet.claude wait-change ...`.
  `wait-change` is not tested on Windows yet.
- The browser pane is often hidden after `preview_start`.

## Find the Jupyters that run, and find the environment

Use the Python of Miniforge, by file path:

`& "$env:USERPROFILE\miniforge3\python.exe" "{code}\jupylet\claude.py" running`

`& "$env:USERPROFILE\miniforge3\python.exe" "{code}\jupylet\claude.py" find-env {version}`

If Miniforge does not exist, use the Python launcher:
`py "{code}\jupylet\claude.py" ...`.

For `base`, the environment is Miniforge's own folder.

## Start Jupyter

Write this file into the scratchpad as `start_jupyter.cmd`. For `base`, leave
out `{name}` after `activate.bat`.

```
@echo off
call "%USERPROFILE%\miniforge3\condabin\activate.bat" {name}
cd /d "{code}\examples"
jupyter lab --no-browser --port %2 --ServerApp.port_retries=0 --IdentityProvider.token=%1 "--JupyterMCPServerExtensionApp.allowed_jupyter_mcp_tools=notebook_run-all-cells,notebook_get-selected-cell,notebook_run-cell,notebook_move-cursor-down,notebook_move-cursor-up"
```

The file does these steps:

1. It activates the environment.
2. It goes to the folder `examples`.
3. It starts JupyterLab with the port and the token that it gets as
   arguments, without a browser window.
4. The last option lets you run one cell and move between cells.

A file is necessary, because PowerShell 5.1 breaks quotes inside quotes.
Do not use `conda init`: Windows blocks the profile scripts of PowerShell by
default.

Then start it detached, from `{code}`:

`& "{python}" -m jupylet.claude detach "{scratchpad}\jupyter.log" cmd /c {scratchpad}\start_jupyter.cmd {token} {port}`

`jupyter.log` must say `JupyterLab extension loaded from ...\envs\{name}\...`.
That shows that the environment is active.

Detach on Windows is not tested yet. If Jupyter stops when a background task
or the conversation ends, tell the person.

## Call the tools

Give the JSON on stdin, in a here-string with single quotes:

```
@'
{"notebook_name": "11-spaceship", "response_format": "detailed", "limit": 0}
'@ | & "{python}" -m jupylet.claude call {port} {token} read_notebook -
```

Or write the JSON into a file in the scratchpad, and give `@{file}`. Tools
without arguments, for example `notebook_run-all-cells`, need neither.

Tested on Windows: `use_notebook` (through `attach`), `read_notebook`,
`read_cell`, `execute_code`, `insert_cell`, `overwrite_cell_source`,
`notebook_run-all-cells`, `notebook_get-selected-cell`,
`notebook_move-cursor-down`, `notebook_move-cursor-up`, `notebook_run-cell`.
Not tried: `delete_cell`, `edit_cell_source`, `move_cell`,
`clear_cell_output`, `restart_notebook`, `replace-kernel`, `cleanup`.

The command `run-cell` is not tested on Windows yet.

## Start Jupylet without Claude

On Windows there is no Terminal setup: the Miniforge installer adds the
Miniforge Prompt to the Start menu. Tell the person, for example:

> Open the Start menu, type *Miniforge* and open **Miniforge Prompt**. It's a
> small window where you type commands. Type `conda activate {name}`, which
> switches it to Jupylet's environment. Then type `cd /d "{code}\examples"`
> to go to the example notebooks, and `jupyter lab`. Jupyter opens in your
> web browser.

## Stop Jupyter

Run `shutdown` from `{code}`. It finds the processes of your Jupyter by the
token in their command line: the launcher `cmd`, `jupyter`, `jupyter-lab` and
two `python`. Not tested on Windows yet.

If it prints `still running`, list the processes that remain:

`Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '{token}' } | Select-Object ProcessId, Name, CommandLine`

Check each command line. Stop only a process whose command line has your
token, by its id, never by its name:

`Stop-Process -Id {id} -Force`

Then tell the person that you closed a program that stayed open.

Windows keeps old records of servers that do not run any more, with their
tokens and process ids. A process id in such a record can now belong to a
different program. Never use those tokens or ids.
