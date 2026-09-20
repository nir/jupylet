# CLAUDE.md - working with people learning jupylet

## About this file

This file is read-only: never edit it. It is the shared guide for helping
people learn to program with jupylet, and it stays the same for everyone.

Keep your own notes about this project in `CLAUDE.0.md` instead (create it
if it doesn't exist). If `CLAUDE.0.md` already exists, read it now, before
anything else.

Part 1 is a procedure: do the steps in order, exactly as written. Every
command in this file was tested by hand. If a step does not give the
expected result, look up the "Problem" it names in Part 5. If that does not
fix it, tell the person plainly what failed and stop. Don't improvise.

## How to talk to the person

The people you help are kids and beginners. If something breaks, they will
think the software is broken and give up. So:

- Never show them a raw error or a wall of output. Say in one plain sentence
  what happened and what you will try. Never say "broken".
- Never leave them with a problem and no next step. Most problems are fixed
  by starting over (Part 4), which takes about a minute and never touches
  their notebooks or code.
- Try one fix at a time, and don't loop. If two attempts fail, tell the
  person honestly that this needs a grown-up, and stop.

## Part 1: Start a live notebook

Words in `<angle brackets>` are values you fill in. `<folder>` is the folder
this file is in. The notebook is `11-spaceship.ipynb`, in `<folder>/examples`.

### Step 1. Ask permission

Ask, in a few plain words, for example:

> May I open a Jupyter notebook here so we can work on the game together?
> It runs only on this computer, and you'll see it right next to our chat.

Continue only after a clear yes.

### Step 2. Find the environment

Ask the person whether they created a conda environment for jupylet, and
what it is called. If they never made one, it is the default one, called
`base`. Call the name `<name>`.

Run this with the Bash tool. The last line of its output is the
environment's Python; call it `<python>`. For `base`, leave out
`conda activate <name> &&`.

`$SHELL -ic "conda activate <name> && python -c 'import sys; print(sys.executable)'"`

Then check that jupylet is installed:

`<python> -c "import jupylet, jupyterlab"`

Expected: no error. If either fails: Problem 14.

### Step 3. Make a token

`<python> -c "import secrets; print(secrets.token_hex(8))"`

Call the result `<token>`. Show it to the person: they will need it in step 8.

### Step 4. Check that port 8888 is free

`<python> -c "import socket; print(socket.socket().connect_ex(('127.0.0.1', 8888)))"`

Expected: a number other than `0`. If it prints `0`: Problem 11.

### Step 5. Start Jupyter

Run this with the Bash tool, with `run_in_background` set (leave out
`conda activate <name> &&` for `base`):

`$SHELL -ic "conda activate <name> && cd <folder>/examples && jupyter lab --no-browser --port 8888 --ServerApp.port_retries=0 --IdentityProvider.token=<token>"`

Never use the Terminal panel for this (Problem 5).

### Step 6. Wait until Jupyter is ready

`<python> -m jupylet.claude wait 8888 <token>`

Expected: `ready`. If not: Problem 15.

### Step 7. Open the notebook in the browser

Use `preview_start` with the URL

`http://localhost:8888/doc/tree/11-spaceship.ipynb?reset`

Use this one browser tab only. Then call `tabs_context`. If it says the
browser pane is hidden, tell the person: "Please click the globe icon in the
upper right corner of the app, so you can see the notebook."

### Step 8. Sign in

Run this in the page (`javascript_tool`):

`(await fetch('/api/status', {credentials: 'same-origin'})).status`

- `200`: already signed in. Go to step 9.
- `403`: the login page is showing. Tell the person: "Please paste this
  token into the login box: `<token>`". Wait until they say they are in, then
  run the check again. Never type the token yourself, and never put it in a
  URL.

### Step 9. Attach to the notebook

`<python> -m jupylet.claude attach 8888 <token> 11-spaceship.ipynb`

Expected: the output contains `Successfully activate notebook`. It waits up
to a minute for the notebook's kernel to be ready. If it says there is no
kernel: Problem 6.

### Step 10. Run all cells

`<python> -m jupylet.claude call 8888 <token> notebook_run-all-cells`

Expected: `True` after a second or two. The game is now running in the
notebook. If it says "Timeout waiting for result": Problem 1. If it says
"Not Found": Problem 2.

## Part 2: Working in the notebook

Every tool is called the same way, from any folder:

`<python> -m jupylet.claude call 8888 <token> <tool> '<json arguments>'`

`<python> -m jupylet.claude tools 8888 <token>` lists the tools.

- Change and run the notebook only through these tools.
- The kernel id is needed by some tools:
  `<python> -m jupylet.claude kernel 8888 <token> 11-spaceship.ipynb`
- Work directly in the person's notebook.
- Read cell outputs to find the real error when something fails.

Tested and working: `list_kernels`, `use_notebook`, `read_notebook` (after
`use_notebook`), `read_cell`, `execute_code` (runs code in the kernel, not
saved in the notebook; pass `kernel_id`), `delete_cell`,
`notebook_run-all-cells`.

Not working: `execute_cell` and `insert_execute_code_cell` (Problem 8).
Not tested: `insert_cell`, `edit_cell_source`, `overwrite_cell_source`,
`move_cell`.

If the person presses Restart Kernel, or run-all times out, replace the
kernel (Problem 1).

## Part 3: Stopping

1. `<python> -m jupylet.claude shutdown 8888 <token>` (expected: `stopped`).
2. Check that your Jupyter process is gone. Yours is the one whose command
   contains your token (macOS and Linux):
   `ps -axo pid,command | grep "token=<token>" | grep -v grep`
   Poll every few seconds for up to 30 seconds. If it is still there after
   that, stop that one process (Problem 10).
3. Close the browser page with `tabs_close`.

## Part 4: Starting over

Use this when something behaves strangely and a simple fix didn't help. Tell
the person first: "I'm going to reset the notebook setup. Your notebooks and
code are not touched."

1. Do Part 3.
2. From `<folder>`, list what would be deleted:
   `<python> -m jupylet.claude cleanup`
   It only lists Jupyter's own state files (the `.jupyter` folder in
   `examples`, the `.jupyter_ystore.db` files, and the cookie secret and
   stale `jpserver-*` / `kernel-*` files in Jupyter's runtime folder), never
   a notebook. Check the list, then delete with
   `<python> -m jupylet.claude cleanup --yes`.
   Deleting the cookie secret signs the browser out, so step 8 will ask for
   the token again.
3. Do Part 1 again, from step 3.

## Part 5: Problems and solutions

What we ran into while building this, and what to do. Each problem has a
number that the steps above refer to.

**1. Run-all times out.**
Symptom: "Error executing tool: Timeout waiting for result" after 30
seconds, and nothing ran (the game objects don't exist).
Cause: unknown. It happened every time after the kernel was restarted in
place (the Restart Kernel button; same kernel id), and never with a fresh
kernel. Kids will press that button. Hiding the browser pane and reloading
the page did not matter.
Do: replace the kernel:
`<python> -m jupylet.claude replace-kernel 8888 <token> 11-spaceship.ipynb`
It shuts the old kernel down, starts a new one, waits until it is ready
(about 10 seconds) and prints the new id. Then run all cells once more. Don't
retry in a loop.

**2. Run-all says "Error: Not Found".**
Cause: the kernel was just replaced and the page is not attached to it yet.
Do: wait ten seconds and try once more. (`replace-kernel` waits for this
itself.)

**3. "Unknown tool: notebook_run-all-cells".**
Cause: the server only knows the run-all tool after it was asked for its tool
list. Do: nothing. `jupylet.claude call` asks first, every time. Only your own
raw calls would hit this.

**4. `!pip` or `!python` in a cell uses the wrong Python.**
Cause: Jupyter was started without activating the environment (for example by
the full path of `jupyter-lab`), so the notebook's `PATH` starts with the
default environment.
Do: always start Jupyter as in step 5, through `$SHELL -ic` with the
environment activated.

**5. The Terminal panel opens and shares the screen with the browser.**
Cause: running a command with the terminal tool opens that panel, and it
leaves an idle tab behind every time. It confuses novices.
Do: start Jupyter as a background Bash process (step 5). Never use the
terminal tool.

**6. There is no kernel for the notebook.**
Cause: the notebook only gets a kernel once it is open in the browser page.
Do: check that step 7 opened the page and that the pane is not hidden; check
step 8 (signed in). Then run step 9 again.

**7. The login page shows.**
Cause: the page is not signed in. It always happens after the cookie secret
was deleted (Part 4) or with a new token.
Do: step 8. The sign-in otherwise survives restarts.

**8. Running a single cell hangs.**
Symptom: `execute_cell` and `insert_execute_code_cell` run until they time out
(minutes), even for `1+1`. Cause: unknown.
Do: use run-all, or `execute_code` for a quick check. Don't use those two.

**9. Run-all with a failing cell.**
Symptom: it stops at the failing cell and reports a vague "500 Internal
Server Error".
Do: read the cell outputs (`read_cell`, `read_notebook`) to find the real
error.

**10. Jupyter does not exit after the shutdown command.**
Symptom: `shutdown` prints `stopped` (the port is closed), but the process
is still there. Usually it exits within 1 to 10 seconds (kernels and
extensions are stopped first). Once, in our tests, it was still alive after
more than two minutes; its log ended at "Kernel shutdown" and nothing came
after. Cause: unknown.
Do: poll for up to 30 seconds (Part 3). Never force it earlier. If it is
still there, stop only your own process (the one with your token in its
command): `kill <pid>`, wait 5 seconds, and only then `kill -9 <pid>`. Then
`ps` again to confirm it is gone.

**11. Port 8888 is already in use, or a second Jupyter is running.**
Cause: another Jupyter is open, from an earlier session or the person's own
terminal. Do not start a second Jupyter on the same folder: once, a notebook
ended up with hundreds of empty cells in that situation (we never proved the
cause). Do: ask the person whether they have Jupyter open. If they do, use it
(its port and token), or ask them to close it. Never kill a process you did
not start.

**12. A notebook was corrupted (hundreds of empty cells, wrong content).**
It happened while the `.ipynb` file was edited on disk, and while Jupyter's
menus were clicked through the browser pane, with the notebook open. Cause
not proven. Do: change the notebook only through the tools (Part 2). If it
happens, tell the person; their last saved copy is in git or on disk.

**13. Claude Code's own MCP connection to Jupyter does not work reliably.**
We tried an `.mcp.json` with the stdio helper `jupyter-mcp-server`, and
with Jupyter's own HTTP endpoint. The stdio one connected once; after the
session was restarted while Jupyter was down, it failed for good with
"connection timed out after 30000ms", and a session only reads `.mcp.json`
when it starts. The HTTP entry never connected. Neither gives the run-all
tool natively.
Do: don't create an `.mcp.json`. `jupylet/claude.py` calls Jupyter's own
endpoint (`http://localhost:8888/mcp`) directly, and that always worked.

**14. jupylet or JupyterLab is not installed (step 2 fails).**
Explain in plain words what is missing, and guide the person through
installing it, following the "How to Install and Run Jupylet" section of
`README.md` step by step: one step at a time, waiting for them to finish and
confirm before the next, and answering their questions. You never install
software yourself; they run every command. Stop before the part where the
README starts Jupyter, because you do that in step 5. If the environment
name was wrong, ask again.

**15. Jupyter does not become ready (step 6).**
Do: read the background task's output file. If the port is taken, see
Problem 11. If it says `No module named`, see Problem 14. Otherwise tell the
person plainly and stop.

**16. A second browser tab.**
A second tab on the same notebook opens a different layout and can confuse
which page answers. Use one tab; close extra ones with `tabs_close`.
