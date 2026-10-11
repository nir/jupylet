# Known problems

What to do when a command does not give the expected result. Each problem
has the same form: what you see, why, and what to do. Task names in quotes
are tasks in `SKILL.md`. Command names in quotes are commands in the
reference file for this computer.

When a solution fails, follow rule 3 of `guiding-jupylet-users`: tell the
person, and ask before the next attempt.

## Contents

- Jupylet, JupyterLab or the extra is missing
- The Jupyter server does not become ready
- The notebook has no kernel
- The page shows the login page
- Run all says "Timeout waiting for result"
- Run all says "Not Found"
- Run all stops at a cell, with "500 Internal Server Error"
- A cell shows `[*]`, and does not finish
- Cells that you added do not show in the page
- The canvas shows as text
- The Jupyter server stopped by itself
- The Jupyter server does not stop correctly
- `!pip` or `!python` in a cell uses the wrong Python
- The "File Changed" message on each save
- A notebook has hundreds of empty cells
- Something behaves strangely: reset the Jupyter server

## Jupylet, JupyterLab or the extra is missing

**What you see:** "Check JupyterLab and the extra" prints an error, or the
log of the Jupyter server says `No module named`.

**Why:** the environment does not have JupyterLab, or the `[claude]` extra.

**What to do:** tell the person in plain words what is missing, for example:

> Jupylet's tools for working in the notebook with you aren't installed in
> this environment.

Offer to install it.

- **Yes:** use the skill `setting-up-jupylet`. When the installation is
  complete, do "Understand the request" in `SKILL.md` again, for what the
  person asked.
- **No:** tell the person what that means, for example:

  > No problem. Without them, I can't see or change your notebook while it's
  > open, or run it for you. I can still explain the code, write code for you
  > to paste into the notebook, and help with an error if you paste it here.
  > If you change your mind, just ask.

  **Only the extra is missing:** you can still start a Jupyter server and open
  a page, but do not attach. Tell the person how to run the cells: "Click into
  the first box of code and press Shift+Enter to run it. Each press runs one
  box and moves to the next."

## The Jupyter server does not become ready

**What you see:** "Wait until it is ready" prints `not ready after 60
seconds`.

**Why:** the log tells. Read `{scratchpad}/jupyter.log`.

**What to do:**

- **The port is in use:** do "Start a Jupyter server" again. It finds a free
  port.
- **`No module named`:** see "Jupylet, JupyterLab or the extra is missing".
- **Other errors:** tell the person, and stop.

## The notebook has no kernel

**What you see:** "Attach" says that there is no kernel.

**Why:** a notebook gets a kernel only when a page opens it.

**What to do:** check that a page shows the notebook, that the pane is
visible, and that the person signed in ("Check the sign-in" says `200`). Then
do "Attach to the notebook" again.

## The page shows the login page

**What you see:** the page asks for a token.

**Why:** the page is not signed in. This happens after a reset of the Jupyter
server, and with a browser that did not sign in to this Jupyter server
before. After a restart of a Jupyter server, a page usually stays signed in.

**What to do:** give the token to the person, as in "Open a page in the
Claude app".

## Run all says "Timeout waiting for result"

**What you see:** "Run all cells" stops after about 30 seconds with this
message.

**Why:** it happened after the person used Restart Kernel in the page. Once,
on Windows, the notebook ran anyway.

**What to do:** first check if the notebook ran:

- `read_cell` of the last cell: does it have a new execution count?
- The page: does the canvas show?
- `execute_code` with `print(app.is_running)`, if the notebook has `app`.

Then:

- **It ran:** continue.
- **Nothing ran:** ask for a yes to replace the kernel. Tell the person that
  the notebook starts fresh. After a yes, run "Replace the kernel", then "Run
  all cells" one more time. Do not try again in a loop.

## Run all says "Not Found"

**What you see:** "Run all cells" prints "Not Found".

**Why:** the kernel was just replaced, and the page is not attached to it
yet.

**What to do:** wait ten seconds, and try one more time.

## Run all stops at a cell, with "500 Internal Server Error"

**What you see:** "Run all cells" prints this error.

**Why:** a cell failed.

**What to do:** read the outputs with `read_cell` or `read_notebook` to find
the real error.

## A cell shows `[*]`, and does not finish

**What you see:** a cell stays at `[*]`, but the kernel is idle, and newer
cells wait behind it.

**Why:** `jupyter_server_nbmodel` is on. "Prepare the environment" turns it
off, but the change applies only to a Jupyter server that started after it.

**What to do,** in this order:

1. **"Prepare the environment" did not print `off` or `turned off`, or the
   Jupyter server started before it:** ask for a yes to stop and start the
   Jupyter server again. That solves it.
2. **The person wants to keep the variables of the kernel:** send the missing
   `idle` message of the stuck cell, with `execute_code`:

   `kernel = get_ipython().kernel; kernel.session.send(kernel.iopub_socket, "status", {"execution_state": "idle"}, parent={"header": h}, ident=kernel._topic("status"))`

   `h` is the header of the request of the stuck cell. A task that the cell
   started keeps it in `kernel._shell_parent`. Send it only for the stuck
   cell.
3. **Step 2 does not help:** "Replace the kernel" solves it, after a yes. The
   variables are lost. Then call `unuse_notebook`, and do "Attach to the
   notebook" again.

## Cells that you added do not show in the page

**What you see:** `read_notebook` and the saved file have the cell, but the
page shows fewer cells, or the cell as one empty line.

**Why:** the Jupyter server built its live copy of the notebook again from an
old log of changes. "Prepare the environment" sets the Jupyter server so that
this does not happen.

**What to do:**

1. Check the copy in the page, not only `read_cell`: count its cells, or
   scroll to the cell and read its text.
2. Ask for a yes to stop the Jupyter server, and to move two files of
   Jupyter out of `{folder}`. Then close the page, and do "Stop the Jupyter
   server".
3. Move `.jupyter_ystore.db` and `.jupyter/collaboration_sessions.json` from
   `{folder}` to a new folder in `{scratchpad}`. Keep them: they hold
   the log of changes, which can help to recover cells.
4. Do "Start a Jupyter server" again, with the same `{folder}` and token, so
   that the person does not need to sign in again.

## The canvas shows as text

**What you see:** the output of a cell is text like
`Image(value=b'\xff\xd8...`, but the game runs.

**Why:** the notebook, or that cell, is not trusted.

**What to do:** ask for a yes to trust the notebook, with
`"{python}" -m jupylet trust_notebooks` from `{folder}`. The cells
then show the canvas without a new run.

## The Jupyter server stopped by itself

**What you see:** "List the Jupyter servers" does not list it any
more.

**Why:** for example, the computer restarted, or a program stopped it. "Start
JupyterLab" starts it apart from your task, so the end of your task does not
stop it.

**What to do:** tell the person, and ask for a yes to start it again. Do
"Start a Jupyter server" with the same `{folder}` and token, so that the page
stays signed in. Then do "Open a page" (the old page shows an error) and
"Attach to the notebook".

## The Jupyter server does not stop correctly

**What you see:** "Stop JupyterLab" prints `a kernel is still running:
not shutting the server down`, or `still running`.

**What to do:**

- **macOS:** tell the person, and stop. Do not stop processes yourself.
- **Windows:** follow "Stop JupyterLab" in the reference file.

## `!pip` or `!python` in a cell uses the wrong Python

**What you see:** a `!pip` or `!python` line in a cell uses another
environment.

**Why:** the Jupyter server started without the environment activated.

**What to do:** stop it after a yes, and do "Start a Jupyter server", which
activates the environment.

## The "File Changed" message on each save

**What you see:** when the person presses Ctrl+S, JupyterLab asks
"Overwrite or Revert".

**Why:** the Jupyter server saves the notebook by itself a few seconds after
each change, so the file is newer than the last save of the page. Both have
the same content.

**What to do:** tell the person that "Overwrite" loses nothing, and that they
do not need to save by hand.

## A notebook has hundreds of empty cells

**What you see:** a notebook has hundreds of empty cells, or wrong content.

**Why:** it happened when a program changed the `.ipynb` file on the disk,
and when two Jupyter servers had the same notebook open.

**What to do:** tell the person, and do not change the notebook more. Ask
the person if they have a copy of it. The example notebooks are also in the
Jupylet code on GitHub.

## Something behaves strangely: reset the Jupyter server

**What you see:** something behaves strangely, and no other entry in this file
helps.

**What to do:** reset the Jupyter server. A reset deletes the state files
of Jupyter, and some of them belong to all Jupyter servers of the person. It
also deletes the old log of changes. **A notebook lost cells:** do "Cells that
you added do not show in the page" first, because it keeps that log.

First do "Find the Jupyter servers that run" in `SKILL.md`. **Another
Jupyter server runs, besides the one that you work with:** a reset would sign
out its pages. Ask the person if you can stop it first. **No:** do not reset,
and tell the person what is wrong in plain words.

Ask, for example:

> Something isn't working right. May I reset the notebook setup? It closes
> Jupyter and clears its saved state. Your notebooks and code are not
> touched.

After a yes:

1. Do "Stop the Jupyter server".
2. Run "List the state files" from the reference file. The list has only state
   files of Jupyter, never a notebook. Check the list. Then run "Delete the
   state files". This also signs the pages out, so the person must paste the
   token again.
3. Do "Start a Jupyter server", "Open a page" and "Attach to the notebook".
