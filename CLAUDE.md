# CLAUDE.md - working with people learning jupylet

## About this file

This file is read-only: never edit it. It is the shared guide for helping
people learn to program with jupylet, and it stays the same for everyone.

Keep private notes about one machine or one person (which environment they
use, which folder, what they like) in `CLAUDE.0.md` instead (create it if it
doesn't exist). If `CLAUDE.0.md` already exists, read it now, before anything
else. What you learn that would help other sessions goes in `EXPERIENCE.md`
(see the next section).

Part 1 is a procedure: do the steps in order, exactly as written. Every
command in this file was tested by hand. If a step does not give the
expected result, look up the "Problem" it names in Part 5. If that does not
fix it, tell the person plainly what failed and stop. Don't improvise.

## The rules

Everything else here is how. These are why none of it works without them,
whether the session is running a kid's notebook or doing Windows setup work.
Read them before anything else.

1. **`EXPERIENCE.md` is this file's growing memory: what earlier sessions
   learned about running this environment and about the people who use it.**
   Read it before you act. When you learn something worth keeping, write it
   there, not into a private note only you will read. The next session is not
   you; unwritten, it does not exist for whoever comes next.
2. **Looking costs nothing. Changing something does.** Read, check, compare,
   freely, without asking. The moment you would start, stop or delete
   something, or change a setting, ask first, in plain words, and wait for a
   clear yes.
3. **A raw error is a closed door: translate it.** Say in one plain sentence
   what happened and what you will try. Never leave the person with a problem
   and no next step.
4. **One thing at a time.** One question per message, one fix per attempt.
   Two failed attempts means stop and say so plainly: this needs a person, not
   a third guess.
5. **Do what was asked, first.** If you think something else should happen
   before or alongside it, say so and wait for a yes; never do it quietly and
   explain afterward.
6. **A guess is not a fact.** Say what you verified, what you read but never
   ran, and what you are guessing, and never blur the three together.
7. **When you are wrong, say so once, plainly, fix it, and go on.**

## About EXPERIENCE.md: your writable memory

Think of this file as a ROM: the basic operating instructions, fixed, tested,
the same for everyone, and never edited by a session. `EXPERIENCE.md`, next to
it, is writable memory: what earlier sessions learned by doing this work on real
computers, on macOS and on Windows 11, and by working with people. You start
every session without remembering the earlier ones; that file is the only way
they can pass on what they learned. It is expected to keep growing on each
person's computer, as you learn from your own mistakes and from the problems
you meet there, the way a person's know-how grows with practice. That is
intended: do not keep it short.

How to use it:

- **Read it at the start**, right after this file and `CLAUDE.0.md`: its "Read
  this first" list, its section on working with people, and the whole section
  for the platform you are on. Skim the other headings. When something goes
  wrong, search it for the words of the problem before you try anything else;
  the answer may already be there.
- **Trust it less than this file.** Each entry says on which platform and
  versions it was seen and how sure it is (verified, seen once, code reading,
  guess), and things change. Where it disagrees with this file, follow this
  file and write the disagreement into `EXPERIENCE.md`.
- **Write to it while you work**, not only at the end. When you make a mistake
  or hit a problem, write down what happened, why, and how to avoid or fix it
  next time. Also write down a step that failed, a fix that worked, a guess you
  confirmed or disproved, and anything about how people like to be helped. Put
  the entry where it belongs and tag it, as the rules at the top of
  `EXPERIENCE.md` say. Improve an entry when you learn more. Move an entry that
  turned out wrong to "Retired" with the reason. Never delete what is true
  just to save space.
- **Do it quietly.** The people you help are learning; do not narrate the file
  to them, and never write down their names, tokens, user names or paths.

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
- Steps and their numbers are for you, not the person: never say "Step N" to
  them, and never name a technical detail they have no use for (a port, an
  environment path, a process id). A step that is purely a technical check
  and succeeds needs no comment at all - go straight to the next one. Speak
  up only for something they must decide, something that failed, or, if a
  step is genuinely slow, one short line so silence does not look like a
  hang. They are not technical, but they are not simple either: say the real
  thing in plain words, don't hide that something is happening at all.
- **Exception, for developers only:** if the person asks you for verbose
  boxes, show the underlying command and its raw output in a plain code
  block for every step, before the plain sentence (the block is the
  computation that produced the result; the sentence, exactly as you would
  say it to a real person, is a summary of what the block showed). Do this
  only once they ask, for the rest of that session; never guess it on your
  own, and never let the block replace the sentence. If they never ask,
  proceed exactly as for a real person: no boxes.
- **Never narrate your own process or state to the person, boxes or not.**
  Not "continuing", not "no boxes requested", not which mode you are in, not
  your own reasoning about what to do next. This applies with boxes on too:
  the box is the computation; a remark that is only for whoever is reading
  along is a different thing again and needs its own clearly separate note,
  never loose text mixed in with the real script. When in doubt, say nothing
  at all rather than narrate yourself.

## Part 1: Start a live notebook

Before step 1, read `EXPERIENCE.md` (see "About EXPERIENCE.md").

On Windows 11, read Part 6 first: steps 2 and 5 are different there, and the
other steps have small differences.

Words in `<angle brackets>` are values you fill in. `<folder>` is the folder
this file is in. The notebook is `11-spaceship.ipynb`, in `<folder>/examples`.

### Step 1. Ask permission

Ask, in a few plain words, for example:

> May I open a Jupyter notebook here so we can work on the game together?
> It runs only on this computer, and you'll see it right next to our chat.

Continue only after a clear yes.

### Step 2. Find the environment

Read `<folder>/jupylet/__init__.py`; the line `VERSION = '<version>'` near
the top gives `<version>`. This is what every command below was tested
against, so the environment must have exactly this version, not just any
jupylet.

Miniforge's own Python (`$HOME/miniforge3/bin/python`) finds every
environment that has it installed, without needing to know in advance which
one that is (it runs `jupylet/claude.py` by file path on purpose: jupylet is
never installed into `base`, so `base`'s own Python cannot `import` it, but
running the file directly does not need to):

`$HOME/miniforge3/bin/python <folder>/jupylet/claude.py find-env <version>`

One path per line, most recently set up first.

- **No lines:** tell the person plainly that jupylet `<version>` is not
  installed in any environment on this computer, point them to Problem 14,
  and stop.
- **One line:** that is the environment. Call its path `<env>` and its
  python `<python>` (`<env>/bin/python`). Tell the person in one plain line,
  for example "Found it - using your jupylet setup." Do not ask; there is
  nothing to choose between.
- **More than one line:** the first is the most recently set up. Name the
  environments to the person (the last part of each path is its name, the one
  they picked when they set it up) and propose the first one explicitly, for
  example:

  > I found jupylet in more than one place on your computer: `jupylet`,
  > `jupylet2`. I'll use `jupylet2`, the most recently set up one - is that
  > right, or did you mean a different one?

  Continue only after a clear yes; if they name a different one, use that.

Call the last part of `<env>`'s path `<name>` (needed in step 5 to activate
it). If `<env>` is Miniforge's own folder itself, not a folder under `envs`,
it is the `base` environment, which has no separate name: leave out
`conda activate <name> &&` in step 5 instead, the same as on a Mac.

Then check that the environment also has jupyterlab:

`<python> -c "import jupyterlab"`

Expected: no error. If it fails: Problem 14.

Then check that the example notebooks are trusted (a Jupyter safety check; an
untrusted notebook shows its game picture as plain text instead):

`<python> -m jupylet is_trusted <folder>/examples`

Expected: every line says `trusted`. If any line says `NOT TRUSTED`, explain
and ask, for example:

> These notebooks aren't trusted on this computer yet, so the pictures
> won't show until they are. May I trust them?

After a clear yes:

`<python> -m jupylet trust_notebooks <folder>/examples`

Then run the check again to confirm every line says `trusted`, and tell the
person in one plain line, for example "The example notebooks are trusted
now." Never describe the check itself (lines, output, step numbers) to them.
If the person declines to trust them, tell them plainly that the game
picture may show as text instead, and go on anyway.

### Step 3. Make a token

`<python> -c "import secrets; print(secrets.token_hex(8))"`

Call the result `<token>`. Say nothing to the person yet; it is only
useful once the login page actually asks for it, in step 8.

### Step 4. Check that port 8888 is free

`<python> -c "import socket; print(socket.socket().connect_ex(('127.0.0.1', 8888)))"`

Expected: a number other than `0`. Say nothing to the person about this step;
it has nothing for them to act on unless it fails, and it should not feel
like a separate moment from starting Jupyter in step 5. If it prints `0`:
Problem 11.

### Step 5. Start Jupyter

Run this with the Bash tool, with `run_in_background` set (leave out
`conda activate <name> &&` for `base`):

`$SHELL -ic "conda activate <name> && cd <folder>/examples && jupyter lab --no-browser --port 8888 --ServerApp.port_retries=0 --IdentityProvider.token=<token>"`

Never use the Terminal panel for this (Problem 5). Starting takes a few
seconds; say so once, for example "Starting Jupyter now, one second...", so
the wait does not look like nothing is happening. Nothing else in steps 5 or
6 needs a comment; go straight to step 7 once it is ready.

### Step 6. Wait until Jupyter is ready

`<python> -m jupylet.claude wait 8888 <token>`

Expected: `ready`. If not: Problem 15.

### Step 7. Open the notebook in the browser

Use `preview_start` with the URL

`http://localhost:8888/doc/tree/11-spaceship.ipynb?reset`

Use this one browser tab only. Before saying anything to the person, also run
the sign-in check from step 8: it works whether or not the pane is visible to
them, so you can tell them everything they need in one message instead of
two. Then call `tabs_context`.

- Pane hidden, sign-in check says `200`: tell the person: "Please click the
  globe icon in the upper right corner of the app, so you can see the
  notebook."
- Pane hidden, sign-in check says `403`: tell them both at once, for example:
  "Please click the globe icon in the upper right corner of the app, so you
  can see the notebook. It will ask for a special token - please paste this
  in: `<token>`"
- Pane already visible: go straight to step 8.

### Step 8. Sign in

Run this in the page (`javascript_tool`) - skip it if you already have the
answer from step 7:

`(await fetch('/api/status', {credentials: 'same-origin'})).status`

- `200`: already signed in. Go to step 9.
- `403`: if you have not already told them (see step 7), tell the person, for
  example: "This page wants a special token just for this session - please
  paste this in: `<token>`". Never type the token yourself, and never put it
  in a URL, even though you can see the page and technically could: signing
  in stays in the person's hands, the same as any password or credential, no
  matter how low the stakes of this one particular token feel.

  Then check again yourself, a handful of times a few seconds apart, instead
  of leaving it to them to remember to tell you - most people are quick
  enough that this alone catches it. If it is still not `200` after that,
  say so and ask them to tell you once they are in, then wait for that
  instead of continuing to poll silently.

### Step 9. Attach to the notebook

This can take up to a minute; say so once first, for example "Almost
there, just getting the notebook ready...", so the wait does not look stuck:

`<python> -m jupylet.claude attach 8888 <token> 11-spaceship.ipynb`

Expected: the output contains `Successfully activate notebook`. If it says
there is no kernel: Problem 6.

### Step 10. Run all cells

`<python> -m jupylet.claude call 8888 <token> notebook_run-all-cells`

Expected: `True` after a second or two. Tell the person, for example "Your
game should be showing in the notebook now - take a look!" If it says
"Timeout waiting for result": Problem 1. If it says "Not Found": Problem 2.

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

On Windows 11, do not use step 1: see "Stopping on Windows 11" in Part 6.

1. `<python> -m jupylet.claude shutdown 8888 <token>`
   It ends every notebook and every kernel first (there can be kernels
   without a notebook), then shuts the server down, waits for it to exit, and
   only if the process lingers, stops it. It takes a few seconds. Expected:
   `stopped`. Also fine: `not running`, and `stopped after ending its
   process`. Anything else: Problem 10.
2. Close the browser page with `tabs_close`.
3. Add to `EXPERIENCE.md` what you learned in this session, if anything
   (see "About EXPERIENCE.md"), quietly.

Only ever do this for the Jupyter you started yourself: it ends every kernel
on that server.

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

**10. Jupyter does not stop cleanly.**
Symptom: before, the server stopped answering but its process stayed alive
for minutes, with the log ending at "Kernel shutdown". That happened when the
server was shut down while kernels were still running (a game in a notebook
kernel, or a kernel with no notebook). `shutdown` now ends every session and
kernel first, and then the server exits by itself within a second or two.
Do: nothing, in the normal case. `shutdown` also stops the process itself if
it lingers (the one with your token in its command; normal stop first, then
force) and prints `stopped after ending its process`. If it prints `a kernel
is still running: not shutting the server down`, or `still running`, tell the
person plainly and stop; don't kill anything yourself.

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
README starts Jupyter, because you do that in step 5.

**15. Jupyter does not become ready (step 6).**
Do: read the background task's output file. If the port is taken, see
Problem 11. If it says `No module named`, see Problem 14. Otherwise tell the
person plainly and stop.

**16. A second browser tab.**
A second tab on the same notebook opens a different layout and can confuse
which page answers. Use one tab; close extra ones with `tabs_close`.

Problems found later, each marked with its platform and how sure it is, are
in `EXPERIENCE.md`.

## Part 6: Windows 11

Added on 2026-09-22, from a session on one Windows 11 Home machine (Miniforge
in `C:\Users\<user>\miniforge3`, JupyterLab 4.6.3, jupyter-mcp-server 2.2.2,
jupyter_server_nbmodel 0.2.9, ipywidgets 8.1.9), driven with the PowerShell
tool of Claude Code Desktop. Everything below was run by hand there. Not
tried: Windows 10, ARM, folders with spaces or inside OneDrive, a fresh
Miniforge install, `replace-kernel`, `cleanup`.

What went wrong there, and why, is in `EXPERIENCE.md` under Windows 11.

Parts 1 to 5 apply unchanged, except where this part says otherwise. Use the
PowerShell tool for every command here. Do not use the Bash tool or
`$SHELL -ic` (Git may not be installed), and never `claude.py shutdown`.

### Step 2 on Windows 11: find the environment

Miniforge is normally in `C:\Users\<user>\miniforge3` (`<miniforge>`); ask if
it is elsewhere. Read `<folder>\jupylet\__init__.py`; the line
`VERSION = '<version>'` near the top gives `<version>`. This is what every
command below was tested against, so the environment must have exactly this
version, not just any jupylet.

The same idea as on a Mac, only the path convention differs, and deliberately
no PowerShell-specific syntax: Miniforge's own Python finds every environment
that has jupylet installed, without needing to know in advance which one that
is (it runs `jupylet\claude.py` by file path on purpose: jupylet is never
installed into `base`, so `base`'s own Python cannot `import` it, but running
the file directly does not need to):

`& "<miniforge>\python.exe" "<folder>\jupylet\claude.py" find-env <version>`

One path per line, most recently set up first.

- **No lines:** tell the person plainly that jupylet `<version>` is not
  installed in any environment on this computer, point them to Problem 14,
  and stop.
- **One line:** that is the environment. Call its path `<env>` and its
  python `<python>` (`<env>\python.exe`). Tell the person in one plain line,
  for example "Found it - using your jupylet setup." Do not ask; there is
  nothing to choose between.
- **More than one line:** the first is the most recently set up. Name the
  environments to the person (the last part of each path is its name, the one
  they picked when they set it up) and propose the first one explicitly, for
  example:

  > I found jupylet in more than one place on your computer: `jupylet`,
  > `jupylet2`. I'll use `jupylet2`, the most recently set up one - is that
  > right, or did you mean a different one?

  Continue only after a clear yes; if they name a different one, use that.

Call the last part of `<env>`'s path `<name>` (needed in step 5 to activate
it). If `<env>` is Miniforge's own folder itself, not a folder under `envs`,
it is the `base` environment, which has no separate name: leave out
`conda activate <name> &&` in step 5 instead, the same as on a Mac.

Then check that the environment also has jupyterlab:

`& "<python>" -c "import jupyterlab"`

Expected: no error. If it fails: Problem 14.

The helper commands (`wait`, `attach`, `call`, `tools`) are plain HTTP and
need no activation; only Jupyter itself does (step 5). Steps 3 and 4 are the
same, with `& "<python>"` in front. A free port prints `10061` (connection
refused): only `0` means it is taken. The trust check and fix are also the
same as in step 2, with `& "<python>"` in front: they resolve the notebook
files themselves, so there is nothing Windows-specific about them.

### Step 5 on Windows 11: start Jupyter

Write this file into the scratchpad as `start_jupyter.cmd` (a file, because
PowerShell 5.1 breaks nested quotes; see `EXPERIENCE.md`). For `base`, leave out
the name after `activate.bat`:

```
@echo off
call <miniforge>\condabin\activate.bat <name>
cd /d <folder>\examples
jupyter lab --no-browser --port 8888 --ServerApp.port_retries=0 --IdentityProvider.token=%1 "--JupyterMCPServerExtensionApp.allowed_jupyter_mcp_tools=notebook_run-all-cells,notebook_get-selected-cell,notebook_run-cell,notebook_move-cursor-down,notebook_move-cursor-up"
```

Run it with the PowerShell tool, with `run_in_background` set:

`cmd /c "<scratchpad>\start_jupyter.cmd <token>"`

The long last argument makes the MCP server offer the "run one cell" tools
(below); without it only run-all is offered. The background task's output
should say `JupyterLab extension loaded from ...\envs\<name>\...`: that shows
the environment was activated.

### Steps 6 to 10 on Windows 11

The same as in Part 1, with `& "<python>"` in front and, for
`python -m jupylet.claude`, the current folder set to `<folder>` first
(`No module named jupylet.claude` in `EXPERIENCE.md`). Nothing else here is
Windows-specific: Part 1's own step 7 and step 8 already cover a hidden pane
and checking sign-in before asking for the token. One confirmed fact worth
knowing: after a restart with a new token, the page was still already signed
in (status `200`), because the sign-in cookie survives a restart.

### Working in the notebook on Windows 11

Tested and working: `use_notebook` (through `attach`), `read_notebook`,
`read_cell`, `execute_code`, `insert_cell`, `overwrite_cell_source`,
`notebook_run-all-cells`, `notebook_get-selected-cell`,
`notebook_move-cursor-down`, `notebook_move-cursor-up`, `notebook_run-cell`.
`execute_cell` and `insert_execute_code_cell` time out (Problem 8). Not tried:
`delete_cell`, `edit_cell_source`, `move_cell`, `clear_cell_output`,
`restart_notebook`, `list_kernels`, `notebook_run-cell-and-select-next`,
`notebook_run-cell-and-insert-below`.

`insert_cell` puts a cell at the index you give (0-based) and does not run it.
`overwrite_cell_source` replaces a cell's source; the old output and execution
count stay until the cell is run. Cells you add or change are saved into the
person's notebook within seconds: remove your test cells, or tell the person
(more in `EXPERIENCE.md`).

Running one cell (needs the flag from step 5). `execute_code` runs code in the
kernel but puts nothing in a cell. To run a particular cell the way a person
would, select it and run the selection: `notebook_get-selected-cell` (answers
with `cellIndex` and `source`), `notebook_move-cursor-down` and
`notebook_move-cursor-up` (one cell per call), `notebook_run-cell` (answers
`True` at once; then `read_cell` shows the execution count and the output).
After a run-all the selection is on the last cell, so the script must be able
to move up as well as down. Check the selected cell's `source` before running:
the wrong cell can restart the game. It moves the person's cursor. Save this
into a `.py` file in the scratchpad, set the constants on the line that names
the notebook, the cell index and the exact source that cell should have, and
run it with `& "<python>" <file> <token>` (33 moves took under two seconds):

```python
import ast
import json
import re
import sys
import time
import urllib.request

PORT, TOKEN = 8888, sys.argv[1]
NOTEBOOK, TARGET, TARGET_SRC = '11-spaceship', 33, '3+3'


def rpc(body, timeout=45):
    req = urllib.request.Request(
        'http://localhost:%s/mcp' % PORT,
        data=json.dumps(body).encode(),
        headers={
            'Authorization': 'Bearer ' + TOKEN,
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',
        },
    )
    raw = urllib.request.urlopen(req, timeout=timeout).read().decode()
    return json.loads(re.search(r'\{.*\}', raw, re.S).group(0))


rpc({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'})


def call(name, args=None):
    out = rpc({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call',
               'params': {'name': name, 'arguments': args or {}}})

    if 'error' in out:
        return 'ERROR: ' + out['error']['message']

    return '\n'.join(c['text'] for c in out['result']['content'])


def selected():
    return ast.literal_eval(call('notebook_get-selected-cell'))


sel = selected()
print('selection starts at cell', sel['cellIndex'])

while sel['cellIndex'] != TARGET:
    step = 1 if sel['cellIndex'] < TARGET else -1
    call('notebook_move-cursor-down' if step == 1 else 'notebook_move-cursor-up')
    new = selected()

    if new['cellIndex'] != sel['cellIndex'] + step:
        sys.exit('unexpected move: %r -> %r' % (sel['cellIndex'], new['cellIndex']))

    sel = new

if sel['cellIndex'] != TARGET or sel.get('source', '').strip() != TARGET_SRC:
    sys.exit('NOT running: the selected cell is cell %r, not the one you meant' % sel['cellIndex'])

print('RUN-CELL:', call('notebook_run-cell'))
time.sleep(2)
print(call('read_cell', {
    'cell_index': TARGET,
    'include_outputs': True,
    'notebook_name': NOTEBOOK,
}))
```

### Stopping on Windows 11

Do not use step 1 of Part 3: `claude.py shutdown` looks for the process with
`ps`, which Windows does not have, so it would say `stopped` without checking,
and its force-stop uses signals Windows lacks. Instead:

1. Find your processes: the ones whose command line contains your token,
   `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '<token>' }`.
   Several carry it; all are yours (`EXPERIENCE.md`, old servers).
2. End every session and kernel and then ask the server to shut down. Save
   this as a `.py` file in the scratchpad, put the folder in the second line,
   and run it with `& "<python>" <file> <token>`. Expected output: `sessions
   and kernels ended`, `shutdown requested`, `port closed: True`:

```python
import sys
import urllib.error
import urllib.request

sys.path.insert(0, r'<folder>')

import jupylet.claude as c

port, tok = 8888, sys.argv[1]

if not c._answers(port):
    raise SystemExit('not running')

for s in c._api(port, tok, '/api/sessions'):
    try:
        c._api(port, tok, '/api/sessions/' + s['id'], 'DELETE')
    except urllib.error.HTTPError:
        pass

for k in c._api(port, tok, '/api/kernels'):
    try:
        c._api(port, tok, '/api/kernels/' + k['id'], 'DELETE')
    except urllib.error.HTTPError:
        pass

if not c._wait_until(lambda: not c._api(port, tok, '/api/kernels'), 30):
    raise SystemExit('a kernel is still running: not shutting the server down')

print('sessions and kernels ended')

req = urllib.request.Request(
    'http://localhost:%s/api/shutdown' % port,
    method='POST',
    headers={'Authorization': 'token ' + tok},
)
urllib.request.urlopen(req, timeout=30).read()

print('shutdown requested')
print('port closed:', c._wait_until(lambda: not c._answers(port), 30))
```

3. Check: no process carries your token any more, the port answers nothing
   (`connect_ex` is not `0`), and the background task ended with exit code 0.
   If a process is left, tell the person plainly and stop; do not stop other
   processes.
4. Close the browser page with `tabs_close`.

After stopping, and only when no Jupyter is running, delete
`<folder>\examples\.jupyter_ystore.db` and
`<folder>\examples\.jupyter\collaboration_sessions.json` (Jupyter's own state,
never a notebook). Stale collaboration state is the likely cause of cells added over MCP not
showing in the page (`EXPERIENCE.md`).
