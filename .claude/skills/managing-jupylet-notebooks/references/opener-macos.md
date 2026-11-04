# Commands of the notebook opener for macOS

The agent `notebook-opener` runs these commands by their names, for example:
run "Make a token". Run each command with the Bash tool.
Fill in the blanks in `{braces}`: "Values" in the agent file defines them. Each
command has a short description of what it does, and its expected output.

Run each command exactly as this file shows it, alone in one call, with its
comment `# jupylet: {name}` at the end. The Jupylet system uses these comments to find each command and check it.
If you leave out the comment or change the command, this may cause
failures. Do not add `>/dev/null`, `| grep`,
`| head` or other commands to it: you need its whole output to see an error.

Each `python -m jupylet.claude` command starts with `cd "{code}"`. In another
folder, a folder named `jupylet` can hide the installed Jupylet.

## Read the version

`grep -m1 "^VERSION" "{code}/jupylet/__init__.py"  # jupylet: notebooks/read-version`

Prints the line `VERSION = '...'` of the Jupylet code in `{code}`. `{version}`
is the version between the quotes, for example `0.10.0`.

## Find conda installations

`ls -d "$HOME/miniforge3/bin/python" "$HOME/mambaforge/bin/python" "$HOME/miniconda3/bin/python" "$HOME/anaconda3/bin/python" "$HOME/opt/miniconda3/bin/python" "$HOME/opt/anaconda3/bin/python" /opt/homebrew/Caskroom/miniforge/base/bin/python /opt/homebrew/Caskroom/miniconda/base/bin/python /opt/miniconda3/bin/python /opt/anaconda3/bin/python 2>/dev/null || true  # jupylet: notebooks/find-conda`

Prints the `bin/python` of each usual conda folder that has one: the Python
of that conda installation. The conda folder is the path before
`/bin/python`. `|| true` ends the command without an error code when some of
them do not exist.

## List the Jupyter servers

`cd "{code}" && "{python}" -m jupylet.claude running  # jupylet: notebooks/list-servers`

Runs `claude.py` by its path, so it needs no Jupylet in that Python. Prints a
JSON list of the Jupyter servers that run, as the task "Find the Jupyter
servers that run" in the agent file describes.

## List the environments

`cd "{code}" && "{conda_python}" "{code}/jupylet/claude.py" find-env {version}  # jupylet: notebooks/list-envs`

Asks each conda environment on the computer which Jupylet it has. Prints one
line for each environment with the Jupylet version of the code in `{code}`,
best first, in three columns: the path, the kind (`conda` or `venv`), and the
source of its Jupylet: `this folder`, or its version and where it comes from.

## Check JupyterLab and the extra

`"{python}" -c "import jupyterlab, jupyter_mcp_server"  # jupylet: notebooks/check-extra`

Expected: no output. An error names the part that is missing:
`jupyterlab` is JupyterLab, `jupyter_mcp_server` is the `[claude]` extra.

## Make a token

`"{python}" -c "import secrets; print(secrets.token_hex(4))"  # jupylet: notebooks/make-token`

Prints a new token of eight characters.

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
  that the web page shows the cells that Claude adds.
- It turns off the news pop-up of JupyterLab.

Prints `off` or `not installed` (nothing changed), `turned off` (it changed
the setting), or `still on: ...` (it could not change it).

## Start JupyterLab

Write this file into the scratchpad as `start_jupyter.sh`, with the Write tool:

```bash
#!/bin/bash
# Start JupyterLab for Jupylet. $1 is the token.
source "{conda}/bin/activate" "{env}"
cd "{folder}"
exec jupyter lab --no-browser --IdentityProvider.token="$1" \
  --JupyterMCPServerExtensionApp.allowed_jupyter_mcp_tools=notebook_run-all-cells,notebook_get-selected-cell,notebook_run-cell,notebook_move-cursor-down,notebook_move-cursor-up
```

The file does these steps:

1. It activates the environment with the activation script of its conda
   installation. It does not use the Terminal settings of the person.
2. It goes to the folder `{folder}`.
3. It starts JupyterLab with the token that it gets, without a browser
   window. JupyterLab uses port 8888, or the next free port if 8888 is in
   use.
4. The last option lets you run one cell and move between cells. Without it,
   you can only run all cells.

If you have the tool `mcp__terminal__run_in_terminal`, start the file in a
tab of the Terminal panel of the Claude app. The tab stays hidden unless the
person opens the panel. The Jupyter server runs in it until you stop it, or
until the chat ends. Call `run_in_terminal` with the title `Jupyter` and this
command:

`/bin/bash "{scratchpad}/start_jupyter.sh" {token}`

Keep the `tab_id` that it returns: it is `{tab_id}`, and the main chat
needs it to stop the Jupyter server. Then
read the output of the tab with `mcp__terminal__read_terminal`, with the
`tab_id` and `wait_for_output_ms` 10000. Read again until a line shows the
address of the Jupyter server, for example
`http://localhost:8889/lab?token=...`. JupyterLab shows the address once it
is ready. The number after `localhost:` is `{port}`. If no address shows
after 60 seconds, the Jupyter server is not ready.

If you do not have that tool, as in Claude Code in a terminal, start the
file apart from your task, so that it continues after your turn:

`cd "{code}" && "{python}" -m jupylet.claude detach "{scratchpad}/jupyter.log" /bin/bash "{scratchpad}/start_jupyter.sh" {token}  # jupylet: notebooks/start-jupyter`

Then wait until it is ready:

`cd "{code}" && "{python}" -m jupylet.claude wait "{scratchpad}/jupyter.log" {token}  # jupylet: notebooks/wait-ready`

It prints `ready` and the port, for example `ready 8889`, or `not ready
after 60 seconds`. The port is `{port}`.

*Developer note, not an instruction: the activation with `bin/activate`, the
file, and the last option are not tested on macOS yet.*

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

`cd "{code}" && "{python}" -m jupylet.claude attach {port} {notebook}  # jupylet: notebooks/attach`

Connects the Jupyter MCP tools to the notebook. The notebook must be open in a
web page. Expected: the output contains `Successfully activate notebook`.
