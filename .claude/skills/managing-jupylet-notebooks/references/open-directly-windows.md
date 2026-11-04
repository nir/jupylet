# Commands to open a notebook directly, on Windows

The task "Open the notebook directly" in `SKILL.md` runs these commands by
their names. Fill in the blanks in `{braces}`: "Values" in `SKILL.md`
defines them. Run each PowerShell command exactly as this file shows it,
alone in one call, with its comment `# jupylet: {name}` at the end. The
Jupylet system uses these comments to find each command and check it.

## Start the Jupyter server

Call `mcp__terminal__run_in_terminal` with the title `Jupyter` and this
command. For a notebook that exists:

`& "$HOME\miniforge3\python.exe" "{code}\jupylet\claude.py" serve "{scratchpad}\jupyter.json" "{notebook path}"`

For a new notebook, with its short name, for example `my-game`:

`& "$HOME\miniforge3\python.exe" "{code}\jupylet\claude.py" serve "{scratchpad}\jupyter.json" --new {short name}`

A Terminal tab does not take the comment `# jupylet: ...`, so these commands
have none.

Runs `claude.py` with the Python of Miniforge, by its path. In one program,
it:

1. finds the environment of this Jupylet folder, by the local setting
   `environment`, or else by a search, which then saves the setting, and
   goes on with the Python of that environment;
2. checks that the environment has JupyterLab, the `[claude]` extra and the
   Jupylet of `{code}`;
3. prepares the Jupyter settings of the environment, if they need it;
4. with `--new`, creates the empty notebook in `{code}\examples` (with
   `-2`, `-3`... if the name is taken); without it, checks that no Jupyter
   server that runs has the notebook open;
5. reads the setting `hide-chat-list`, makes a token, and writes the state
   file `jupyter.json`;
6. activates the environment, and starts JupyterLab in the same tab,
   serving the folder of the notebook.

The Jupyter server runs in the tab until the tab closes, or the chat or the
Claude app ends.

## Wait for the Jupyter server

`& "$HOME\miniforge3\python.exe" "{code}\jupylet\claude.py" wait-ready "{scratchpad}\jupyter.json"  # jupylet: notebooks/wait-serve`

Waits, for up to 60 seconds, until the Jupyter server of "Start the Jupyter
server" answers. It can start in the same reply as "Start the Jupyter
server": it waits for the state file. Prints one line, for example:

`ready: 11-spaceship.ipynb, folder C:\Users\sam\jupylet\examples, port 8888, address http://localhost:8888/doc/tree/11-spaceship.ipynb?token=3f9a1c07, python C:\Users\sam\miniforge3\envs\jupylet\python.exe, hide chat list yes.`

`address` is the address of the web page of the notebook.

`hide chat list` gives the setting `hide-chat-list`: `yes`, `no`, or `no
setting`.

Sometimes `Tell the person:` follows, with one `- ` line for each thing to
tell the person. If the Jupyter server did not start, it prints `problem:`
and the reason.

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
panel. If the panel is already collapsed, it does nothing.

## Attach

`Set-Location "{code}"; & "{python}" -m jupylet.claude attach {port} {notebook}  # jupylet: notebooks/attach`

Connects the Jupyter MCP tools to the notebook. The notebook must be open in a
web page. Expected: the output contains `Successfully activate notebook`.
