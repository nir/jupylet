# Start Jupyter in the app

Follow these steps in order. The section names in quotes are in `SKILL.md`.

## Contents

- Step 1. Make a token
- Step 2. Choose a port, and prepare the environment
- Step 3. Start Jupyter
- Step 4. Wait until Jupyter is ready
- Step 5. Open the notebook in the browser pane
- Step 6. Sign in
- Step 7. Offer more room
- Step 8. Attach to the notebook
- Step 9. Run all cells

## Step 1. Make a token

`"{python}" -c "import secrets; print(secrets.token_hex(4))"`

The result is `{token}`. It has eight characters, so that a kid can paste or
type it. A short token is enough only while Jupyter accepts connections from
this computer alone, so never add `--ip` to the start command.

## Step 2. Choose a port, and prepare the environment

`"{python}" -c "import socket; print(socket.socket().connect_ex(('127.0.0.1', 8888)))"`

A number other than `0`: the port is free, and `{port}` is 8888. `0`: see
"Port 8888 is in use" in `troubleshooting.md`.

Then run, each time before Jupyter starts:

`"{python}" -m jupylet.claude prepare`

It changes the Jupyter settings that Jupylet needs in a notebook with you:
it turns off `jupyter_server_nbmodel`, which can make a cell stop
responding, and it makes the page show the cells that you add. Expected:
`off`, `turned off` or `not installed`. If it prints `still on: ...`, tell
the person that a notebook can stop responding after a few minutes, and
continue.

## Step 3. Start Jupyter

Start Jupyter detached:

`"{python}" -m jupylet.claude detach "{scratchpad}/jupyter.log" $SHELL -ic "conda activate {name} && cd \"{code}/examples\" && jupyter lab --no-browser --port {port} --ServerApp.port_retries=0 --IdentityProvider.token={token} --JupyterMCPServerExtensionApp.allowed_jupyter_mcp_tools=notebook_run-all-cells,notebook_get-selected-cell,notebook_run-cell,notebook_move-cursor-down,notebook_move-cursor-up"`

For `base`, leave out `conda activate {name} &&`.

The command does these steps:

1. It starts a new shell of the person, as a new Terminal window does.
2. It activates the environment.
3. It goes to the folder `examples`.
4. It starts JupyterLab on `{port}`, with `{token}`, without a browser
   window.
5. The last option lets you run one cell and move between cells. Without
   it, you can only run all cells. This option is not tested on macOS yet.

`detach` prints a process id, and returns at once. Jupyter writes its output
to `jupyter.log`, and runs until you stop it.

- Never start Jupyter as a background task (`run_in_background`). Jupyter
  stops when the task stops, and the notebooks stop with it.
- Never use the Terminal panel. It opens next to the browser pane, and
  confuses beginners.

Tell the person in one line, and use the time to explain, for example:
"Starting Jupyter now, one second. It runs here on your computer, and the
panel next to our chat shows it."

## Step 4. Wait until Jupyter is ready

`"{python}" -m jupylet.claude wait {port} {token}`

Expected: `ready`. If not, see "Jupyter does not become ready" in
`troubleshooting.md`.

## Step 5. Open the notebook in the browser pane

Use `preview_start` with the URL
`http://localhost:{port}/doc/tree/{notebook}?reset`.

Before you tell the person anything, do the sign-in check from step 6, and
call `tabs_context`. Then you can tell the person all they must do in one
message.

- **The pane is hidden, and the sign-in check says `200`:** tell the person,
  for example: "Please click the globe icon in the upper right corner of the
  app, so you can see the notebook." Then continue with step 7.
- **The pane is hidden, and the check says `403`:** start the monitoring command
  from step 6 first. Then tell the person to click the globe icon and to
  paste the token, in one message.
- **The pane is visible:** continue with step 6.

## Step 6. Sign in

Run this in the page with `javascript_tool`, if step 5 did not:

`(await fetch('/api/status', {credentials: 'same-origin'})).status`

- **`200`:** the person signed in before. Continue with step 7.
- **`403`:** start this monitoring command in the background, as in
  `waiting.md`:

  `"{python}" -m jupylet.claude wait-open {port} {token} {notebook} 90`

  Then tell the person, for example: "This page wants a special token just
  for this session. Please paste this in: `{token}`. The token shows Jupyter
  that it's really you, so nobody else can open your notebook. Ask me if you
  get stuck."

  When the command prints `open`, tell the person in one line, for example
  "You're signed in. Almost there, just getting the notebook ready...". When
  it prints `timeout`, start it again with a longer time, and ask how it
  goes.

## Step 7. Offer more room

The notebook is in a narrow pane next to the chat, and the canvas can be
cut off. If the tool `mcp__ccd_window__set_sidebar_collapsed` exists, ask,
for example (if the tool list shows only its name, load it with
`ToolSearch`):

> The notebook is a bit cramped. Want me to tuck the list of chats on the
> left out of the way, so there's more room for it?

After a yes, call the tool with `collapsed` set to true. Then tell the person
how to show the list again, for example:

> Done. You can bring it back any time by clicking the small panel icon in
> the top-left corner of the app, just right of the three lines. And if you
> want the notebook even bigger, the arrows icon at the top of the notebook
> panel makes it fill the window.

If the tool does not exist, or does not change anything, tell the person how
to do it themselves, with the same two hints. After a no, do not ask again.
Never show the list again yourself: it is a setting of the person.

## Step 8. Attach to the notebook

This can take up to a minute. If you did not just say so, tell the person in
one line first.

`"{python}" -m jupylet.claude attach {port} {token} {notebook}`

Expected: the output contains `Successfully activate notebook`. If it says
that there is no kernel, see "The notebook has no kernel" in
`troubleshooting.md`.

## Step 9. Run all cells

Tell the person how long it takes:

- **The first run after an installation in this conversation:** Jupylet
  prepares its code, and the page can show `[*]` for half a minute, with no
  visible change. Say, for example: "Running the notebook now. It can take up
  to a minute to get going, while Jupylet gets everything ready for the first
  time. After that it starts in seconds."
- **Any other run:** say, for example: "Running the notebook now, it takes a
  few seconds."

Start `wait-change` in the background first, with 90 seconds, as in
`waiting.md`. Then run all cells:

`"{python}" -m jupylet.claude call {port} {token} notebook_run-all-cells`

Expected: `True` after a second or two. The cells can still run. For
"Timeout waiting for result" or "Not Found", see
`troubleshooting.md`. Otherwise end your turn. You get a message
when the run ends.

Do not judge the run by the execution counts alone: the example notebooks are
saved with counts in them. Check that each code cell has a new execution
count, and that `read_cell` of the last one shows no error. Then tell the
person what to do next, for example for the spaceship: "The spaceship example
should be showing at the bottom of the notebook now. I ran every cell, top to
bottom, and the last one started it. Click the canvas, the area where it's
drawn, then steer the spaceship with the arrow keys."

After the first run that follows an installation in this conversation, offer
once to show how to start Jupylet without you, as in "Start Jupylet without
Claude".
