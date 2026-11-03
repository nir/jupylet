# Commands to open a notebook directly, on macOS

The task "Open the notebook directly" in `SKILL.md` runs these commands by
their names. Fill in the blanks in `{braces}`: "Values" in `SKILL.md`
defines them. Run each Bash command exactly as this file shows it, alone in
one call, with its comment `# jupylet: {name}` at the end. The Jupylet
system uses these comments to find each command and check it.

## Start the Jupyter server

Call `mcp__terminal__run_in_terminal` with the title `Jupyter` and this
command. For a notebook that exists:

`"$HOME/miniforge3/bin/python" "{code}/jupylet/claude.py" serve "{scratchpad}/jupyter.json" "{notebook path}"`

For a new notebook, with its short name, for example `my-game`:

`"$HOME/miniforge3/bin/python" "{code}/jupylet/claude.py" serve "{scratchpad}/jupyter.json" --new {short name}`

A Terminal tab does not take the comment `# jupylet: ...`, so these commands
have none.

Runs `claude.py` with the Python of Miniforge, by its path. In one program,
it:

1. finds the environment of this Jupylet folder, by the local setting
   `environment`, or else by a search, which then saves the setting;
2. checks that the environment has JupyterLab, the `[claude]` extra and the
   Jupylet of `{code}`;
3. prepares the Jupyter settings of the environment, if they need it;
4. with `--new`, creates the empty notebook in `{code}/examples` (with
   `-2`, `-3`... if the name is taken);
5. reads the setting `hide-chat-list`, makes a token, and writes the state
   file `jupyter.json`;
6. becomes JupyterLab, serving the folder of the notebook.

The Jupyter server runs in the tab until the tab closes, or the chat or the
Claude app ends.

## Wait for the Jupyter server

`"$HOME/miniforge3/bin/python" "{code}/jupylet/claude.py" wait-ready "{scratchpad}/jupyter.json"  # jupylet: notebooks/wait-serve`

Waits, for up to 60 seconds, until the Jupyter server of "Start the Jupyter
server" answers. It can start in the same reply as "Start the Jupyter
server": it waits for the state file. Prints one line, for example:

`ready: 11-spaceship.ipynb, folder /Users/sam/jupylet/examples, port 8888, token 3f9a1c07, python /Users/sam/miniforge3/envs/jupylet/bin/python, hide chat list yes.`

`hide chat list` gives the setting `hide-chat-list`: `yes`, `no`, or `no
setting`.

Sometimes `Tell the person:` follows, with one `- ` line for each thing to
tell the person. If the Jupyter server did not start, it prints `problem:`
and the reason.

## Check the sign-in

Run this in the web page with `javascript_tool`:

`(await fetch('/api/status', {credentials: 'same-origin'})).status`

Prints `200` if the web page signed in, or `403` if it did not.

## Wait for the web page to open

`cd "{code}" && "{python}" -m jupylet.claude wait-open {port} {token} {notebook} {seconds}  # jupylet: notebooks/wait-open`

Run it in the background. Prints `open` when the notebook opens in a web page,
or `timeout`.

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

`cd "{code}" && "{python}" -m jupylet.claude attach {port} {token} {notebook}  # jupylet: notebooks/attach`

Connects the Jupyter MCP tools to the notebook. The notebook must be open in a
web page. Expected: the output contains `Successfully activate notebook`.
