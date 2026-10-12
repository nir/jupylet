# Commands for Windows 11

The tasks in `SKILL.md` run these commands by their names, for example: run
"Make a token" from the reference file. Run each command with the PowerShell
tool. Never use the Bash tool, even if it exists. Fill in the blanks in
`{braces}`: "Values" in `SKILL.md` defines them. Each command has a short
description of what it does, and its expected output.

Run each command exactly as this file shows it, alone in one call. Do not add
`| Out-Null`, `| Select-String`, `| Select-Object -First` or other commands
to it: you need its whole output to see an error.

Some blanks appear only here:

- `{tool}`, `{json arguments}`: a notebook tool and its arguments, from
  `jupyter-mcp-tools.md`.
- `{cell index}`: the place of a cell in the notebook, from 0.
- `{start of its source}`: the first words of the source of that cell.
- `{seconds}`: how long a monitoring command waits, from `waiting.md`.
- `{since}`, `{fingerprint}`: the fingerprint that the last "Wait for a run"
  printed.
- `{file}`: a file in the scratchpad.
- `{id}`: a process id, from "Stop JupyterLab".

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

## Contents

- Find conda installations
- List the Jupyter servers
- List the environments
- Check JupyterLab and the extra
- Check the Jupyter MCP tools
- Make a token
- Check a port
- Check the environment
- Prepare the environment
- Start JupyterLab
- Wait until it is ready
- Check the sign-in
- Collapse the JupyterLab side bar
- Open the default browser
- Attach
- Call a tool
- Run all cells
- Run all cells and wait
- Run one cell
- Wait for the page to open
- Wait for a run
- Replace the kernel
- Stop JupyterLab
- List the state files, Delete the state files

## Find conda installations

Run both commands:

`Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*','HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*' -ErrorAction SilentlyContinue | Where-Object { $_.DisplayName -match 'Anaconda|Miniconda|Miniforge|Mambaforge' } | ForEach-Object { Split-Path ($_.UninstallString -replace '"', '') } | Where-Object { Test-Path $_ } | Sort-Object -Unique`

Prints the conda folders that Windows lists as installed programs.

`'miniforge3', 'mambaforge', 'miniconda3', 'anaconda3' | ForEach-Object { Join-Path $env:USERPROFILE $_ } | Where-Object { Test-Path $_ }`

Prints the usual conda folders in the home folder that exist.

Keep only a folder that has `python.exe`. Do not use `python`, `python3` or
`py` of the system: on a computer without Python, `python` and `python3` open
the Microsoft Store.

## List the Jupyter servers

`& "{conda_python}" "{code}\jupylet\claude.py" running`

Runs `claude.py` by its path, so it needs no Jupylet in that Python. Prints a
JSON list of the Jupyter servers that run, as the task "Find the Jupyter
servers that run" in `SKILL.md` describes.

## List the environments

`& "{conda_python}" "{code}\jupylet\claude.py" find-env {version}`

Asks each conda environment on the computer which Jupylet it has. Prints one
line for each environment with the Jupylet version of this folder: the path,
the kind, and where its Jupylet comes from.

## Check JupyterLab and the extra

`& "{python}" -c "import jupyterlab, jupyter_mcp_server"`

Expected: no output. An error names the part that is missing:
`jupyterlab` is JupyterLab, `jupyter_mcp_server` is the `[claude]` extra.

## Check the Jupyter MCP tools

`Set-Location "{code}"; & "{python}" -m jupylet.claude tools {port} {token}`

Lists the Jupyter MCP tools that the Jupyter server offers. Expected: the list
contains `notebook_run-all-cells`.

## Make a token

`& "{python}" -c "import secrets; print(secrets.token_hex(4))"`

Prints a new token of eight characters. It is short, so that a kid can paste
or type it. The Jupyter server accepts connections from this computer only,
so a short token is enough.

## Check a port

`& "{python}" -c "import socket; print(socket.socket().connect_ex(('127.0.0.1', {port})))"`

Tries to connect to the port. `0` means that a program uses the port. Another
number, usually `10061`, means that the port is free.

## Check the environment

`Set-Location "{code}"; & "{python}" -m jupylet.claude prepare --check`

Checks the Jupyter settings that "Prepare the environment" changes, and
changes nothing. Prints `ready` if the environment has all of them, or
`needs:` and the parts that "Prepare the environment" would change, for
example `needs: nbmodel`.

## Prepare the environment

`Set-Location "{code}"; & "{python}" -m jupylet.claude prepare`

Changes the Jupyter settings of the environment that Jupylet needs in a
notebook with Claude:

- It turns off `jupyter_server_nbmodel`, which can make a cell stop
  responding.
- It keeps the live copy of each notebook while the Jupyter server runs, so
  that the page shows the cells that Claude adds.
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
jupyter lab --no-browser --port %2 --ServerApp.port_retries=0 --IdentityProvider.token=%1 "--JupyterMCPServerExtensionApp.allowed_jupyter_mcp_tools=notebook_run-all-cells,notebook_get-selected-cell,notebook_run-cell,notebook_move-cursor-down,notebook_move-cursor-up"
```

Then run it:

`Set-Location "{code}"; & "{python}" -m jupylet.claude detach "{scratchpad}\jupyter.log" cmd /c {scratchpad}\start_jupyter.cmd {token} {port}`

The file does these steps:

1. It activates the environment.
2. It goes to the folder `{folder}`.
3. It starts JupyterLab with the token and the port that it gets, without a
   browser window.
4. The last option lets you run one cell and move between cells.

`detach` starts the file apart from your task, so that it continues after
your turn, until you stop it. It prints a process id, and returns at once.
`jupyter.log` must say `JupyterLab extension loaded from ...\envs\{env name}\...`:
that shows that the environment is active.

A file is necessary, because PowerShell 5.1 breaks quotes inside quotes. Do
not use `conda init`: Windows blocks the profile scripts of PowerShell by
default.

*Developer note, not an instruction: `detach` on Windows is not tested yet.*

## Wait until it is ready

`Set-Location "{code}"; & "{python}" -m jupylet.claude wait {port} {token}`

Waits up to 60 seconds until the Jupyter server answers. Prints `ready`, or
`not ready after 60 seconds`.

## Check the sign-in

Run this in the page with `javascript_tool`:

`(await fetch('/api/status', {credentials: 'same-origin'})).status`

Prints `200` if the page signed in, or `403` if it did not.

## Collapse the JupyterLab side bar

Run this in the page with `javascript_tool`:

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
collapsed panel for the next page of this Jupyter server.

## Open the default browser

`Start-Process "http://localhost:{port}/lab/tree/{notebook}"`

Opens the notebook in the default web browser of the computer.

## Attach

`Set-Location "{code}"; & "{python}" -m jupylet.claude attach {port} {token} {notebook}`

Connects the Jupyter MCP tools to the notebook. The notebook must be open in a
page. Expected: the output contains `Successfully activate notebook`.

## Call a tool

Give the JSON on stdin, in a here-string with single quotes:

```
@'
{"notebook_name": "{notebook name}", "response_format": "detailed", "limit": 0}
'@ | & "{python}" -m jupylet.claude call {port} {token} read_notebook -
```

Or write the JSON into a file in the scratchpad, and give `@{file}` as the
last argument. Tools without arguments need neither. Run it from `{code}`.

Some tools need the id of the kernel:

`Set-Location "{code}"; & "{python}" -m jupylet.claude kernel {port} {token} {notebook}`

## Run all cells

`Set-Location "{code}"; & "{python}" -m jupylet.claude call {port} {token} notebook_run-all-cells`

Runs all cells, as Run > Run All Cells in the page does. Prints `True` after
a second or two. The cells can still run after that.

## Run all cells and wait

`Set-Location "{code}"; & "{python}" -m jupylet.claude run-all {port} {token} {notebook} {seconds}`

Run it in the background. It starts to watch the notebook, then runs all
cells, as Run > Run All Cells in the page does, and waits until the run ends.
Prints `ran {fingerprint}`, `timeout {fingerprint}`, or `error` and the
message of the tool. It must watch before the run starts: a fresh kernel
gives the same execution counts as the saved notebook, so a watch that starts
after the run sees no change.

## Run one cell

`Set-Location "{code}"; & "{python}" -m jupylet.claude run-cell {port} {token} {cell index} "{start of its source}"`

Moves the active cell of the page to the cell, one cell at a time. It checks
that the source of the cell starts with the text that you give, and only then
runs it. Prints `True`, or why it did not run. It moves the cursor of the
person.

*Developer note, not an instruction: `run-cell` is not tested on Windows
yet.*

## Wait for the page to open

`Set-Location "{code}"; & "{python}" -m jupylet.claude wait-open {port} {token} {notebook} {seconds}`

Run it in the background. Prints `open` when the notebook opens in a page,
or `timeout`.

## Wait for a run

`Set-Location "{code}"; & "{python}" -m jupylet.claude wait-change {port} {token} {notebook} {seconds} {since}`

Run it in the background. Prints `ran {fingerprint}` when the kernel ran
something and is idle again, or `timeout {fingerprint}`.

*Developer note, not an instruction: `wait-change` is not tested on Windows
yet.*

## Replace the kernel

`Set-Location "{code}"; & "{python}" -m jupylet.claude replace-kernel {port} {token} {notebook}`

Stops the kernel of the notebook, starts a new one, waits until it is ready
(about 10 seconds), and prints the new id. The variables of the old kernel
are lost.

## Stop JupyterLab

`Set-Location "{code}"; & "{python}" -m jupylet.claude shutdown {port} {token}`

Stops each kernel, then the Jupyter server, and waits until it ends. It finds
the processes of the Jupyter server by the token in their command line.
Prints `stopped`, `not running`, or `stopped after ending its process`.

**It prints `still running`:** tell the person that a part of Jupyter did not
stop, and ask for a yes to end it. Then list the processes that remain:

`Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '{token}' } | Select-Object ProcessId, Name, CommandLine`

Check each command line. Stop only a process whose command line has your
token, by its id, never by its name:

`Stop-Process -Id {id} -Force`

Then tell the person that you closed a program that stayed open.

Windows keeps old records of Jupyter servers that do not run any more, with
their tokens and process ids. A process id in such a record can now belong to
a different program. Never use those tokens or ids.

## List the state files, Delete the state files

List: `Set-Location "{code}"; & "{python}" -m jupylet.claude cleanup`

Delete: `Set-Location "{code}"; & "{python}" -m jupylet.claude cleanup --yes`

The state files are the files that Jupyter keeps for itself: the `.jupyter`
folders in `{code}` and `{code}\examples`, old `.jupyter_ystore.db` files,
and the cookie secret and the server records in the runtime folder of
Jupyter. Never a notebook. Deleting the cookie secret signs each page out.

*Developer note, not an instruction: `replace-kernel` and `cleanup` are not
tested on Windows yet.*
