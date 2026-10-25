# Known problems in the notebook

What to do when a cell, a run or the notebook does not behave as expected.
Each problem has the same form: what you see, why, and what to do. Command
names in quotes are commands in the reference file for this computer.

When a solution fails, follow rule 3 of `guiding-jupylet-users`: tell the
person, and ask before the next attempt.

## Contents

- Run all says "Timeout waiting for result"
- Run all says "Not Found"
- Run all stops at a cell, with "500 Internal Server Error"
- A cell shows `[*]`, and does not finish
- Cells that you added do not show in the web page
- The canvas shows as text
- The "File Changed" message on each save
- A notebook has hundreds of empty cells

## Run all says "Timeout waiting for result"

**What you see:** "Run all cells" stops after about 30 seconds with this
message.

**Why:** it happened after the person used Restart Kernel in the web page. Once,
on Windows, the notebook ran anyway.

**What to do:** first check if the notebook ran:

- `read_cell` of the last cell: does it have a new execution count?
- The web page: does the canvas show?
- `execute_code` with `print(app.is_running)`, if the notebook has `app`.

Then:

- **It ran:** continue.
- **Nothing ran:** ask for a yes to replace the kernel. Tell the person that
  the notebook starts fresh. After a yes, run "Replace the kernel", then "Run
  all cells" one more time. Do not try again in a loop.

## Run all says "Not Found"

**What you see:** "Run all cells" prints "Not Found".

**Why:** the kernel was just replaced, and the web page is not attached to it
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

**Why:** `jupyter_server_nbmodel` is on. The Jupylet settings of the
environment turn it off, but the change applies only to a Jupyter server that
started after it.

**What to do,** in this order:

1. **"Check the environment" does not print `ready`, or the Jupyter server
   started before the settings changed:** ask for a yes to stop and start
   the Jupyter server again. That solves it.
2. **The person wants to keep the variables of the kernel:** send the missing
   `idle` message of the stuck cell, with `execute_code`:

   `kernel = get_ipython().kernel; kernel.session.send(kernel.iopub_socket, "status", {"execution_state": "idle"}, parent={"header": h}, ident=kernel._topic("status"))`

   `h` is the header of the request of the stuck cell. A task that the cell
   started keeps it in `kernel._shell_parent`. Send it only for the stuck
   cell.
3. **Step 2 does not help:** "Replace the kernel" solves it, after a yes. The
   variables are lost. Then call `unuse_notebook`, and run "Attach" again.

## Cells that you added do not show in the web page

**What you see:** `read_notebook` and the saved file have the cell, but the
web page shows fewer cells, or the cell as one empty line.

**Why:** the Jupyter server built its live copy of the notebook again from an
old log of changes. The Jupylet settings of the environment prevent this.

**What to do:**

1. Check the copy in the web page, not only `read_cell`: count its cells, or
   scroll to the cell and read its text.
2. Ask for a yes to stop the Jupyter server, and to move two files of
   Jupyter out of `{folder}`. Then close the web page, and stop the Jupyter
   server.
3. Move `.jupyter_ystore.db` and `.jupyter/collaboration_sessions.json` from
   `{folder}` to a new folder in `{scratchpad}`. Keep them: they hold
   the log of changes, which can help to recover cells.
4. Start the Jupyter server again, with the same `{folder}` and token, so
   that the person does not need to sign in again.

## The canvas shows as text

**What you see:** the output of a cell is text like
`Image(value=b'\xff\xd8...`, but the game runs.

**Why:** the notebook, or that cell, is not trusted.

**What to do:** ask for a yes to trust the notebooks, with "Trust the
notebooks" from the reference file. The cells then show the canvas without a
new run.

## The "File Changed" message on each save

**What you see:** when the person presses Ctrl+S, JupyterLab asks
"Overwrite or Revert".

**Why:** the Jupyter server saves the notebook by itself a few seconds after
each change, so the file is newer than the last save of the web page. Both have
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
