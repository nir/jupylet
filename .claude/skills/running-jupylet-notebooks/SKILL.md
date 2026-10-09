---
name: running-jupylet-notebooks
description: Controls Jupyter and notebooks in a Jupylet folder, together with a person. Finds the Jupyters that run, connects to one that the person started, or starts Jupyter next to the chat. Opens a notebook, runs it, reads and edits its cells, reads outputs, and looks at the canvas. Watches what the person does in the notebook, and stays free to answer the chat. Also stops Jupyter, resets it, and solves known problems. Use this skill for each task that needs Jupyter or a notebook in a Jupylet folder. For example, the person asks to open, run or try a notebook, asks about an open notebook, or asks to stop Jupyter. setting-up-jupylet uses it to open the spaceship example. Use it together with guiding-jupylet-users.
---

# Running Jupylet notebooks

This skill is a set of tasks with Jupyter and notebooks. Do the task that the
situation needs, not all of them in sequence. Use the skill
`guiding-jupylet-users` for how to talk with the person and when to ask.

## Values

Values in `{braces}` are blanks that you fill in.

- `{code}`: the Jupylet folder. It is the folder that contains
  `.claude/skills/running-jupylet-notebooks`. Use its full path.
- `{version}`: the line `VERSION = '...'` near the top of
  `{code}/jupylet/__init__.py`.
- `{miniforge}`: the folder `miniforge3` in the home folder of the person.
- `{env}`: the folder of the environment that runs Jupylet.
- `{name}`: the name of that environment, the last part of `{env}`.
- `{python}`: the Python of `{env}`: `{env}/bin/python`.
- `{port}`, `{token}`: the port and the token of the Jupyter you work with.
- `{notebook}`: the file name of the notebook, for example
  `11-spaceship.ipynb`.
- `{notebook name}`: the file name without `.ipynb`, for example
  `11-spaceship`. The notebook tools use it.
- `{scratchpad}`: your scratchpad folder.

The commands below are for macOS. On Windows 11, read `references/windows.md`
first: each command needs small changes there, and some tasks are different.
Put every path in double quotes.

## Rules for this skill

- **Work with a Jupyter only after you know which one.** Never stop, restart
  or change a Jupyter that the person did not choose.
- **Stop Jupyter only when the person asks for it, or after a yes.**
- **Never start a second Jupyter on the same folder.** Two Jupyters on the
  same notebook can write over the changes of the other.
- **Change a notebook only with the notebook tools.** Do not edit an open
  `.ipynb` file on the disk. You can read the saved file.
- **Use one browser tab for the notebook.** A second tab on the same notebook
  shows a different layout. Close extra tabs with `tabs_close`.
- **Never type the token, and never put it in a URL.** The person signs in,
  as with a password.

## What to do when

| Situation | Section |
|---|---|
| The start of each task with Jupyter | Find the Jupyters that run |
| A Jupyter with Jupylet runs | Connect to a running Jupyter (`references/own-jupyter.md`) |
| No Jupyter runs, and the person wants a notebook | Find the environment, then Start Jupyter in the app (`references/start-in-app.md`) |
| The person names no notebook | Choose a notebook |
| Work in the notebook: read, edit, run, look | Work in the notebook, and `references/notebook-tools.md` |
| You ask the person to do something in the notebook | Wait for the person (`references/waiting.md`) |
| The person asks how to start Jupylet without you | Start Jupylet without Claude |
| The person asks to stop | Stop Jupyter |
| Something behaves strangely, and a simple fix did not help | Reset Jupyter |
| A command does not give the expected result | `references/troubleshooting.md` |

## Find the Jupyters that run

Run this before you start or connect to a Jupyter:

`"{miniforge}/bin/python" "{code}/jupylet/claude.py" running`

The script uses only the standard library, so any Python can run it. If
`{miniforge}` does not exist, use `python3`. If no Python can run it,
continue as if no Jupyter runs.

It prints one line for each Jupyter that runs, with six columns: the port,
the token, the folder that it serves, `jupylet` if that is a Jupylet folder,
the notebooks that are open in it, and its environment.

- **No line says `jupylet`:** go to "Find the environment".
- **One or more lines say `jupylet`:** go to "Connect to a running Jupyter".

## Connect to a running Jupyter

Read `references/own-jupyter.md`. In short: list the Jupyters with Jupylet,
suggest the best guess, and ask if that is the one. Connect to it if it can
take the connection. If it must start again, ask the person to save first.

## Find the environment

Get `{version}`. Then find each environment that has Jupylet:

`"{miniforge}/bin/python" "{code}/jupylet/claude.py" find-env {version}`

If `{miniforge}` does not exist, use `python3`. The script prints one line
for each environment, best first, with three columns: the path, the kind
(`conda` or `venv`), and where its Jupylet comes from. `this folder` means
that the environment runs the code in `{code}`. Use only `conda`
environments with `this folder`.

Call an environment by its name. Do not show paths or columns to the person.

- **One environment has `this folder`:** use it.
- **Several environments have `this folder`:** name them, suggest the most
  recently installed one, and ask which one to use. If `setting-up-jupylet`
  installed into one of them in this conversation, use that one without a
  question.
- **No environment has `this folder`:** tell the person that Jupylet is not
  installed for this folder yet, and offer to install it. After a yes, use the
  skill `setting-up-jupylet`. After a no, tell the person that the notebooks
  need Jupylet, and that they can ask at any time.

If `{env}` is `{miniforge}` itself, it is the environment `base`. For an
environment of another conda installation, `{name}` is its full path.

Then do two checks:

- JupyterLab: `"{python}" -c "import jupyterlab"`. Expected: no error.
- The `[claude]` extra, the tools that you use in the notebook:
  `"{python}" -c "import jupyter_mcp_server"`. Expected: no error.

If a check fails, see "Jupylet, JupyterLab or the extra is missing" in
`references/troubleshooting.md`.

## Start Jupyter in the app

Read `references/start-in-app.md`, and do its steps in order: make a token,
choose a port and prepare the environment, start Jupyter, wait until it is
ready, open the notebook in the browser pane, let the person sign in, offer
more room, attach to the notebook, and run all cells.

## Choose a notebook

If the person or another skill names a notebook, use it. If not, list the
example notebooks in `{code}/examples`, or ask what the person likes, for
example music, 2D graphics or 3D games. Then suggest a notebook. For
a first try, the spaceship (`11-spaceship.ipynb`) is a good choice.

If Jupyter and notebooks were not explained in this conversation yet, explain
them in one or two sentences when you ask, for example:

> May I open a Jupyter notebook for us? Jupyter is the program where you write
> and run your code, and a notebook is a page in it where you type code in
> small boxes, called cells, and run each one to see what it does.

## Work in the notebook

Read `references/notebook-tools.md` before the first tool call. It has the
call format, the tools that work, and how to edit cells without damage.

- **Look before you speak.** Before you tell the person what they see, check
  it. When you continue after a pause, check that the notebook is still open
  (`running` lists it), and that the browser pane is visible.
- **Look at the canvas only when the output is graphical.** After a run that
  changes the canvas or an image, take a screenshot of the page before you
  say what changed. For text output, read the output with `read_cell`.
- **Find the real error.** When a cell fails, read its output.
- **Delete your own tests.** A cell that you add stays in the
  notebook of the person, and Jupyter saves it in a few seconds. Delete your
  test cells, or tell the person about them. Code that you run in the kernel
  with `execute_code` can change settings of the person, for example the
  logging level. Undo such changes, and tell the person.
- **Watch the person only when the context needs it,** for example after you
  ask them to try something. Use "Wait for the person".

## Wait for the person

Read `references/waiting.md` before you ask the person to do something in
the notebook. In short: monitor the notebook with a command in the
background, end your turn, and answer the chat while the command runs.

## Start Jupylet without Claude

Offer this once, after the first run that follows an installation, for
example:

> By the way, you can also start Jupylet on your own, without me. Whenever
> you'd like, just ask and I'll show you how.

When the person asks, on macOS check first that Terminal finds Miniforge:

`"$SHELL" -ic 'conda info --base'`

It starts the shell of the person as a new Terminal window does, and prints
which conda it uses. Expected: the last line is the full path of
`{miniforge}`. If the check fails, tell the person
that Terminal cannot find Miniforge yet, and offer to prepare it, with step 4
of the skill `setting-up-jupylet` ("Prepare Terminal").

Then tell the person how to start Jupylet, for example:

> Open Terminal: press Cmd+Space, type *Terminal* and press Enter. It's a
> window where you type commands. Each line starts with `(base)`, the
> environment you are in. Type `conda activate {name}`, and it changes to
> `({name})`: now you are in Jupylet's environment. Then type
> `cd "{code}/examples"` to go to the example notebooks, and `jupyter lab`.
> Jupyter opens in your web browser.

On Windows, use the text in `references/windows.md`.

## Stop Jupyter

Stop Jupyter only when the person asks for it, or after a yes, and only a
Jupyter that you started, or that the person chose. `shutdown` stops each
kernel of that Jupyter.

1. Stop a monitoring command that still runs (`TaskStop`).
2. Tell the person in one line, for example "Closing Jupyter now. Your
   notebook is saved."
3. Close the browser tab with `tabs_close`. A page that stays open while
   Jupyter stops shows an error message, which can worry a beginner.
4. `"{python}" -m jupylet.claude shutdown {port} {token}`
   It takes a few seconds. Expected: `stopped`, `not running`, or
   `stopped after ending its process`. Anything else: see "Jupyter does not
   stop correctly" in `references/troubleshooting.md`.
5. Tell the person in one line, for example "All closed. Just ask when you
   want to open it again."

## Reset Jupyter

Use this when something behaves strangely, and a simple fix did not help.
Ask first, for example:

> Something isn't working right. May I reset the notebook setup? It closes
> Jupyter and clears its own saved state. Your notebooks and code are not
> touched.

After a yes:

1. Stop Jupyter, as in "Stop Jupyter".
2. List the files that would be deleted, from `{code}`:
   `"{python}" -m jupylet.claude cleanup`
   The list has only the state files of Jupyter, never a notebook. Check the
   list. Then delete them:
   `"{python}" -m jupylet.claude cleanup --yes`
   This also signs the browser out, so the person must paste the token
   again.
3. Start Jupyter again, as in "Start Jupyter in the app", from step 1.
