# Commands of the notebook opener for Windows 11

The agent `notebook-opener` runs these commands by their names, for example:
run "Make a token". Run each command with the PowerShell
tool. Never use the Bash tool, even if it exists. Fill in the blanks in
`{braces}`: "Values" in the agent file defines them. Each command has a short
description of what it does, and its expected output.

Run each command exactly as this file shows it, alone in one call, with its
comment `# jupylet: {name}` at the end. The Jupylet system uses these comments to find each command and check it.
If you leave out the comment or change the command, this may cause
failures. Do not add `| Out-Null`,
`| Select-String`, `| Select-Object -First` or other commands to it: you
need its whole output to see an error.

General rules for these commands:



- Put `&` in front of a command that starts with a program path in quotes,
  for example `& "{python}" -m jupylet.claude ...`.
- The Python of a conda folder or environment is `python.exe` in it.
- Run each `python -m jupylet.claude` command from `{code}`: run
  `Set-Location "{code}"` first. From another folder, Python may not find
  `jupylet.claude`. The PowerShell tool keeps the current folder between
  commands.
- Windows PowerShell 5.1 deletes the double quotes inside an argument that it
  gives to a program. For a long `python -c` command, write a `.py` file into
  the scratchpad, and run that file.

*Developer note, not an instruction: tested on one Windows 11 Home computer.
Not tried: Windows 10, ARM, a folder with spaces or in OneDrive.*

## Read the version

`(Select-String -Path "{code}\jupylet\__init__.py" -Pattern '^VERSION' | Select-Object -First 1).Line  # jupylet: notebooks/read-version`

Prints the line `VERSION = '...'` of the Jupylet code in `{code}`. `{version}`
is the version between the quotes, for example `0.10.0`.

## Find conda installations

Run both commands:

`Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*','HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*' -ErrorAction SilentlyContinue | Where-Object { $_.DisplayName -match 'Anaconda|Miniconda|Miniforge|Mambaforge' } | ForEach-Object { Split-Path ($_.UninstallString -replace '"', '') } | Where-Object { Test-Path $_ } | Sort-Object -Unique  # jupylet: notebooks/find-conda-programs`

Prints the conda folders that Windows lists as installed programs.

`'miniforge3', 'mambaforge', 'miniconda3', 'anaconda3' | ForEach-Object { Join-Path $env:USERPROFILE $_ } | Where-Object { Test-Path $_ }  # jupylet: notebooks/find-conda-home`

Prints the usual conda folders in the home folder that exist.

Keep only a folder that has `python.exe`. Do not use `python`, `python3` or
`py` of the system: on a computer without Python, `python` and `python3` open
the Microsoft Store.

## List the Jupyter servers

`Set-Location "{code}"; & "{python}" -m jupylet.claude running  # jupylet: notebooks/list-servers`

Runs `claude.py` by its path, so it needs no Jupylet in that Python. Prints a
JSON list of the Jupyter servers that run, as the task "Find the Jupyter
servers that run" in the agent file describes.

## List the environments

`Set-Location "{code}"; & "{conda_python}" "{code}\jupylet\claude.py" find-env {version}  # jupylet: notebooks/list-envs`

Asks each conda environment on the computer which Jupylet it has. Prints one
line for each environment with the Jupylet version of the code in `{code}`,
best first, in three columns: the path, the kind (`conda` or `venv`), and the
source of its Jupylet: `this folder`, or its version and where it comes from.

## Check JupyterLab and the extra

`& "{python}" -c "import jupyterlab, jupyter_mcp_server"  # jupylet: notebooks/check-extra`

Expected: no output. An error names the part that is missing:
`jupyterlab` is JupyterLab, `jupyter_mcp_server` is the `[claude]` extra.

## Make a token

`& "{python}" -c "import secrets; print(secrets.token_hex(4))"  # jupylet: notebooks/make-token`

Prints a new token of eight characters.

## Check the environment

`Set-Location "{code}"; & "{python}" -m jupylet.claude prepare --check  # jupylet: notebooks/check-env`

Checks the Jupyter settings that "Prepare the environment" changes, and
changes nothing. Prints `ready` if the environment has all of them, or
`needs:` and the parts that "Prepare the environment" would change, for
example `needs: nbmodel`.

## Prepare the environment

`Set-Location "{code}"; & "{python}" -m jupylet.claude prepare  # jupylet: notebooks/prepare-env`

Changes the Jupyter settings of the environment that Jupylet needs in a
notebook with Claude:

- It turns off `jupyter_server_nbmodel`, which can make a cell stop
  responding.
- It keeps the live copy of each notebook while the Jupyter server runs, so
  that the web page shows the cells that Claude adds.
- It turns off the news pop-up of JupyterLab.

Prints `off` or `not installed` (nothing changed), `turned off` (it changed
the setting), or `still on: ...` (it could not change it).

## Start JupyterLab

Write this file into the scratchpad as `start_jupyter.cmd`. For the
environment `base`, leave out `{env name}` after `activate.bat`.

```
@echo off
call "{conda}\condabin\activate.bat" {env name}
cd /d "{folder}"
jupyter lab --no-browser --IdentityProvider.token=%1 "--JupyterMCPServerExtensionApp.allowed_jupyter_mcp_tools=notebook_run-all-cells,notebook_get-selected-cell,notebook_run-cell,notebook_move-cursor-down,notebook_move-cursor-up"
```

Then run it:

`Set-Location "{code}"; & "{python}" -m jupylet.claude detach "{scratchpad}\jupyter.log" cmd /c {scratchpad}\start_jupyter.cmd {token}  # jupylet: notebooks/start-jupyter`

The file does these steps:

1. It activates the environment.
2. It goes to the folder `{folder}`.
3. It starts JupyterLab with the token that it gets, without a browser
   window. JupyterLab uses port 8888, or the next free port if 8888 is in
   use.
4. The last option lets you run one cell and move between cells.

`detach` starts the file apart from your task, so that it continues after
your turn, until you stop it. It prints a process id, and returns at once.
Then wait until it is ready:

`Set-Location "{code}"; & "{python}" -m jupylet.claude wait "{scratchpad}\jupyter.log" {token}  # jupylet: notebooks/wait-ready`

It prints `ready` and the port, for example `ready 8889`, or `not ready
after 60 seconds`. The port is `{port}`.

`jupyter.log` must say `JupyterLab extension loaded from ...\envs\{env name}\...`:
that shows that the environment is active.

A file is necessary, because PowerShell 5.1 breaks quotes inside quotes. Do
not use `conda init`: Windows blocks the profile scripts of PowerShell by
default.

*Developer note, not an instruction: `detach` on Windows is not tested yet.*

## Collapse the JupyterLab side bar

Run this in the web page with `javascript_tool`:

```javascript
const stack = document.querySelector('#jp-left-stack');
const tab = document.querySelector('.jp-SideBar.jp-mod-left .lm-TabBar-tab.lm-mod-current');
if (stack && tab && !stack.classList.contains('lm-mod-hidden')) {
  const r = tab.getBoundingClientRect();
  const o = {bubbles: true, cancelable: true, button: 0, clientX: r.x + r.width / 2, clientY: r.y + r.height / 2};
  tab.dispatchEvent(new PointerEvent('pointerdown', o));
  tab.dispatchEvent(new PointerEvent('pointerup', o));
}
```

If the left side bar of JupyterLab shows a panel, for example the list of
files, it presses and releases the tab of that panel. That collapses the
panel. If the panel is already collapsed, it does nothing. A click event does
not work: the tab bar reacts only to pointer events. JupyterLab remembers the
collapsed panel for the next web page of this Jupyter server.

## Attach

`Set-Location "{code}"; & "{python}" -m jupylet.claude attach {port} {notebook}  # jupylet: notebooks/attach`

Connects the Jupyter MCP tools to the notebook. The notebook must be open in a
web page. Expected: the output contains `Successfully activate notebook`.
