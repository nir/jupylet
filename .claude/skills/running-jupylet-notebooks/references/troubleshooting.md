# Known problems

What to do when a command does not give the expected result. When a solution
fails, follow rule 3 of `guiding-jupylet-users`: tell the person, and ask
before the next attempt.

## Contents

- Jupylet, JupyterLab or the extra is missing
- Port 8888 is in use
- Jupyter does not become ready
- The notebook has no kernel
- The login page shows
- Run all says "Timeout waiting for result"
- Run all says "Not Found"
- Run all stops at a cell, with "500 Internal Server Error"
- A cell shows `[*]`, and does not finish
- Cells that you added do not show in the page
- The canvas shows as text
- Jupyter stopped by itself
- Jupyter does not stop correctly
- `!pip` or `!python` in a cell uses the wrong Python
- The "File Changed" message on each save
- A notebook has hundreds of empty cells

## Jupylet, JupyterLab or the extra is missing

A check in "Find the environment" fails. Tell the person in plain words what
is missing, for example "Jupylet's tools for working in the notebook with
you aren't installed in this environment." Offer to install it, and after a
yes, use the skill `setting-up-jupylet`.

After a no, tell the person what that means, for example:

> No problem. Without them, I can't see or change your notebook while it's
> open, or run it for you. I can still explain the code, write code for you
> to paste into the notebook, and help with an error if you paste it here.
> If you change your mind, just ask.

Then you can still start Jupyter, but skip steps 8 and 9 in
`start-in-app.md`, and tell the person how to run the cells: "Click into the first
box of code and press Shift+Enter to run it. Each press runs one box and
moves to the next."

## Port 8888 is in use

Another Jupyter runs, from an earlier session, or one of the person. Find it
with "Find the Jupyters that run".

**It serves `{code}/examples`:** do not start a second Jupyter. Ask to close
it, for example:

> Jupyter is already open on this computer, from earlier. May I close it?

After a yes, stop it as in "Stop Jupyter", with its port and token, and use
8888. After a no, tell the person that a second Jupyter cannot open the same
notebooks, and stop.

**It serves another folder:** ask, for example:

> Jupyter is already open on this computer, from earlier. May I close it? If
> you're still using it, just say no, and I'll start a separate one.

After a yes, stop it, and use 8888. After a no, use the first free port from
8889 up. Check each port as in step 2 in
`start-in-app.md`.

## Jupyter does not become ready

`wait` does not print `ready`. Read `{scratchpad}/jupyter.log`.

- The port is in use: see "Port 8888 is in use".
- `No module named`: see "Jupylet, JupyterLab or the extra is missing".
- Anything else: tell the person, and stop.

## The notebook has no kernel

A notebook gets a kernel only when the page opens it. Check that the page is
open, that the pane is visible, and that the person signed in (step 6 in
`start-in-app.md`). Then do step 8 again.

## The login page shows

The person did not sign in to the page yet. This happens after "Reset
Jupyter", and with a new token. Do step 6 in `start-in-app.md`. In other
cases, the sign-in stays after a restart of Jupyter.

## Run all says "Timeout waiting for result"

The call stops after about 30 seconds. First look: the notebook can have run
anyway.

- `read_cell` of the last cell: does it have a new execution count?
- The page: does the canvas show?
- `execute_code` with `print(app.is_running)`, if the notebook has `app`.

If it ran, continue. If nothing ran, this happened after Restart Kernel in
the page. Replace the kernel, after a yes, because the variables of the
person are lost:

`"{python}" -m jupylet.claude replace-kernel {port} {token} {notebook}`

It stops the old kernel, starts a new one, waits until it is ready (about 10
seconds), and prints the new id. Then run all cells one more time. Do not
try again in a loop.

## Run all says "Not Found"

The kernel was just replaced, and the page is not attached to it yet. Wait
ten seconds, and try one more time.

## Run all stops at a cell, with "500 Internal Server Error"

A cell failed. Read the outputs (`read_cell`, `read_notebook`) to find the
real error.

## A cell shows `[*]`, and does not finish

`jupyter_server_nbmodel` causes this, and `prepare` turns it off. If it
happens anyway, check that `prepare` printed `off` or `turned off`, and that
Jupyter started after that. If not, start Jupyter again after a yes.

To keep the variables of the kernel, there is a rescue: send the missing
`idle` message of the stuck cell, with `execute_code`:

`kernel.session.send(kernel.iopub_socket, "status", {"execution_state": "idle"}, parent={"header": h}, ident=kernel._topic("status"))`

`h` is the header of that cell's request (a task that a cell started keeps it
in `kernel._shell_parent`). Send it only for the stuck cell. Otherwise,
`replace-kernel` solves it, and the variables are lost. After
`replace-kernel`, call `unuse_notebook`, then `attach`.

## Cells that you added do not show in the page

`read_notebook` and the saved file have the cell, but the page shows fewer
cells, or the cell as one empty line. Jupyter built its live copy of the
notebook again from an old log of changes. `prepare` sets Jupyter so that
this does not happen.

If it happens anyway:

1. Check the copy of the page, not only `read_cell`: count its cells, or
   scroll to the cell and read its text.
2. Close the page, and stop Jupyter (after a yes).
3. Move `.jupyter_ystore.db` and `.jupyter/collaboration_sessions.json` from
   `examples` to a different folder. Keep them, do not delete them.
4. Start Jupyter again with the same token.

## The canvas shows as text

The output of a cell is text like `Image(value=b'\xff\xd8...`, but the game
runs. The notebook, or that cell, is not trusted. Check with
`"{python}" -m jupylet is_trusted` from `{code}/examples`. Trust it, after a
yes. The cells then show the canvas without a new run.

## Jupyter stopped by itself

A Jupyter that started as a background task stops when the conversation is
rewound, or at the time limit of the task. That is why step 3 in
`start-in-app.md` starts it detached. If Jupyter stopped anyway, ask, then
start it again with the same token, so that the sign-in of the page still
works. Then attach.

## Jupyter does not stop correctly

`shutdown` prints `a kernel is still running: not shutting the server down`,
or `still running`. Tell the person, and stop. Do not stop processes
yourself on macOS. On Windows, see `windows.md`.

## `!pip` or `!python` in a cell uses the wrong Python

Jupyter started without the environment activated. Always start it as in
step 3 in `start-in-app.md`.

## The "File Changed" message on each save

When the person presses Ctrl+S, JupyterLab asks "Overwrite or Revert".
Jupyter saves the notebook by itself a few seconds after each change, so the
file is newer than the last save of the page. Both have the same content:
"Overwrite" loses nothing. The person does not need to save by hand.

## A notebook has hundreds of empty cells

This happened when a program changed the `.ipynb` file on the disk, and when
two Jupyters had the same notebook open. Change notebooks only with the tools,
and never start a second Jupyter on the same folder. If it happens, tell the
person. The last saved copy is in git, or on the disk.
