# Commands for macOS

The tasks in `SKILL.md` run these commands by their names, for example: run
"Make a token" from the reference file. Run each command with the Bash tool.
Fill in the blanks in `{braces}`: "Values" in `SKILL.md` defines them. Each
command has a short description of what it does, and its expected output.

Run each command exactly as this file shows it, alone in one call, with its
comment `# jupylet: {name}` at the end. The Jupylet system uses these comments to find each command and check it.
If you leave out the comment or change the command, this may cause
failures. Do not add `>/dev/null`, `| grep`,
`| head` or other commands to it: you need its whole output to see an error.

Each `python -m jupylet.claude` command starts with `cd "{code}"`. In another
folder, a folder named `jupylet` can hide the installed Jupylet.

Some blanks appear only here:

- `{tool}`, `{json arguments}`: a notebook tool and its arguments, from
  `jupyter-mcp-tools.md`.
- `{cell index}`: the place of a cell in the notebook, from 0.
- `{start of its source}`: the first words of the source of that cell.
- `{seconds}`: how long a monitoring command waits, from `waiting.md`.
- `{since}`, `{fingerprint}`: the fingerprint that the last "Wait for a run"
  printed.
- `{file}`: a file in the scratchpad.


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
- Trust the notebooks
- Check the Terminal

## Find conda installations

`ls -d "$HOME/miniforge3/bin/python" "$HOME/mambaforge/bin/python" "$HOME/miniconda3/bin/python" "$HOME/anaconda3/bin/python" "$HOME/opt/miniconda3/bin/python" "$HOME/opt/anaconda3/bin/python" /opt/homebrew/Caskroom/miniforge/base/bin/python /opt/homebrew/Caskroom/miniconda/base/bin/python /opt/miniconda3/bin/python /opt/anaconda3/bin/python 2>/dev/null || true  # jupylet: notebooks/find-conda`

Prints the `bin/python` of each usual conda folder that has one: the Python
of that conda installation. The conda folder is the path before
`/bin/python`. `|| true` ends the command without an error code when some of
them do not exist.

## List the Jupyter servers

`"{conda_python}" "{code}/jupylet/claude.py" running  # jupylet: notebooks/list-servers`

Runs `claude.py` by its path, so it needs no Jupylet in that Python. Prints a
JSON list of the Jupyter servers that run, as the task "Find the Jupyter
servers that run" in `SKILL.md` describes.

## List the environments

`"{conda_python}" "{code}/jupylet/claude.py" find-env {version}  # jupylet: notebooks/list-envs`

Asks each conda environment on the computer which Jupylet it has. Prints one
line for each environment with the Jupylet version of this folder: the path,
the kind, and where its Jupylet comes from.

## Check JupyterLab and the extra

`"{python}" -c "import jupyterlab, jupyter_mcp_server"  # jupylet: notebooks/check-extra`

Expected: no output. An error names the part that is missing:
`jupyterlab` is JupyterLab, `jupyter_mcp_server` is the `[claude]` extra.

## Check the Jupyter MCP tools

`cd "{code}" && "{python}" -m jupylet.claude tools {port} {token}  # jupylet: notebooks/check-tools`

Lists the Jupyter MCP tools that the Jupyter server offers. Expected: the list
contains `notebook_run-all-cells`.

## Make a token

`"{python}" -c "import secrets; print(secrets.token_hex(4))"  # jupylet: notebooks/make-token`

Prints a new token of eight characters. It is short, so that a kid can paste
or type it. The Jupyter server accepts connections from this computer only,
so a short token is enough.

## Check a port

`"{python}" -c "import socket; print('in use' if socket.socket().connect_ex(('127.0.0.1', {port})) == 0 else 'free')"  # jupylet: notebooks/check-port`

Tries to connect to the port. Prints `in use` if a program uses the port,
or `free` if the port is free.

## Check the environment

`cd "{code}" && "{python}" -m jupylet.claude prepare --check  # jupylet: notebooks/check-env`

Checks the Jupyter settings that "Prepare the environment" changes, and
changes nothing. Prints `ready` if the environment has all of them, or
`needs:` and the parts that "Prepare the environment" would change, for
example `needs: nbmodel`.

## Prepare the environment

`cd "{code}" && "{python}" -m jupylet.claude prepare  # jupylet: notebooks/prepare-env`

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

Write this file into the scratchpad as `start_jupyter.sh`, with the Write tool:

```bash
#!/bin/bash
# Start JupyterLab for Jupylet. $1 is the token, $2 is the port.
source "{conda}/bin/activate" "{env}"
cd "{folder}"
exec jupyter lab --no-browser --port "$2" --ServerApp.port_retries=0 \
  --IdentityProvider.token="$1" \
  --JupyterMCPServerExtensionApp.allowed_jupyter_mcp_tools=notebook_run-all-cells,notebook_get-selected-cell,notebook_run-cell,notebook_move-cursor-down,notebook_move-cursor-up
```

Then run it:

`cd "{code}" && "{python}" -m jupylet.claude detach "{scratchpad}/jupyter.log" /bin/bash "{scratchpad}/start_jupyter.sh" {token} {port}  # jupylet: notebooks/start-jupyter`

The file does these steps:

1. It activates the environment with the activation script of its conda
   installation. It does not use the Terminal settings of the person.
2. It goes to the folder `{folder}`.
3. It starts JupyterLab with the token and the port that it gets, without a
   browser window.
4. The last option lets you run one cell and move between cells. Without it,
   you can only run all cells.

`detach` starts the file apart from your task, so that it continues after
your turn, until you stop it. It prints a process id, and returns at once.
The output of JupyterLab goes to `jupyter.log`.

*Developer note, not an instruction: the activation with `bin/activate`, the
file, and the last option are not tested on macOS yet.*

## Wait until it is ready

`cd "{code}" && "{python}" -m jupylet.claude wait {port} {token}  # jupylet: notebooks/wait-ready`

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

`open "http://localhost:{port}/lab/tree/{notebook}"  # jupylet: notebooks/open-browser`

Opens the notebook in the default web browser of the computer.

## Attach

`cd "{code}" && "{python}" -m jupylet.claude attach {port} {token} {notebook}  # jupylet: notebooks/attach`

Connects the Jupyter MCP tools to the notebook. The notebook must be open in a
page. Expected: the output contains `Successfully activate notebook`.

## Call a tool

`cd "{code}" && "{python}" -m jupylet.claude call {port} {token} {tool} '{json arguments}'  # jupylet: notebooks/call-tool`

Calls one notebook tool. Instead of the JSON, the last argument can be
`@{file}` (read the JSON from a file) or `-` (read it from stdin). Tools
without arguments need no last argument.

Some tools need the id of the kernel:

`cd "{code}" && "{python}" -m jupylet.claude kernel {port} {token} {notebook}  # jupylet: notebooks/kernel-id`

## Run all cells

`cd "{code}" && "{python}" -m jupylet.claude call {port} {token} notebook_run-all-cells  # jupylet: notebooks/run-all`

Runs all cells, as Run > Run All Cells in the page does. Prints `True` after
a second or two. The cells can still run after that.

## Run all cells and wait

`cd "{code}" && "{python}" -m jupylet.claude run-all {port} {token} {notebook} {seconds}  # jupylet: notebooks/run-all-wait`

Run it in the background. It starts to watch the notebook, then runs all
cells, as Run > Run All Cells in the page does, and waits until the run ends.
Prints `ran {fingerprint}`, `timeout {fingerprint}`, or `error` and the
message of the tool. It must watch before the run starts: a fresh kernel
gives the same execution counts as the saved notebook, so a watch that starts
after the run sees no change.

## Run one cell

`cd "{code}" && "{python}" -m jupylet.claude run-cell {port} {token} {cell index} "{start of its source}"  # jupylet: notebooks/run-cell`

Moves the active cell of the page to the cell, one cell at a time. It checks
that the source of the cell starts with the text that you give, and only then
runs it, because the wrong cell can restart a game. Prints `True`, or why it
did not run. It moves the cursor of the person.

## Wait for the page to open

`cd "{code}" && "{python}" -m jupylet.claude wait-open {port} {token} {notebook} {seconds}  # jupylet: notebooks/wait-open`

Run it in the background. Prints `open` when the notebook opens in a page,
or `timeout`.

## Wait for a run

`cd "{code}" && "{python}" -m jupylet.claude wait-change {port} {token} {notebook} {seconds} {since}  # jupylet: notebooks/wait-change`

Run it in the background. Prints `ran {fingerprint}` when the kernel ran
something and is idle again, or `timeout {fingerprint}`.

## Replace the kernel

`cd "{code}" && "{python}" -m jupylet.claude replace-kernel {port} {token} {notebook}  # jupylet: notebooks/replace-kernel`

Stops the kernel of the notebook, starts a new one, waits until it is ready
(about 10 seconds), and prints the new id. The variables of the old kernel
are lost.

## Stop JupyterLab

`cd "{code}" && "{python}" -m jupylet.claude shutdown {port} {token}  # jupylet: notebooks/stop-jupyter`

Stops each kernel, then the Jupyter server, and waits until it ends. Prints
`stopped`, `not running`, or `stopped after ending its process`.

## List the state files, Delete the state files

List: `cd "{code}" && "{python}" -m jupylet.claude cleanup  # jupylet: notebooks/list-state`

Delete: `cd "{code}" && "{python}" -m jupylet.claude cleanup --yes  # jupylet: notebooks/delete-state`

The state files are the files that Jupyter keeps for itself: the `.jupyter`
folders in `{code}` and `{code}/examples`, old `.jupyter_ystore.db` files,
and the cookie secret and the server records in the runtime folder of
Jupyter. Never a notebook. Deleting the cookie secret signs each page out.

## Trust the notebooks

`cd "{folder}" && "{python}" -m jupylet trust_notebooks  # jupylet: notebooks/trust-notebooks`

Marks each notebook in `{folder}` as trusted, so that the cells show the
canvas. Run it from inside `{folder}`. In the folder above `examples`, Python
finds the code folder, also named `jupylet`, instead of the installed
Jupylet, and the command fails.

## Check the Terminal

`"$SHELL" -ic 'conda info --base'  # jupylet: notebooks/check-terminal`

Starts the shell of the person as a new Terminal window does, and prints
which conda it uses. Expected: the last line is the full path of
`{miniforge}`.
