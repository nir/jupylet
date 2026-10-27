---
name: working-in-notebooks
description: Reads, changes and runs the cells of an open Jupyter notebook together with the person, and monitors what the person does in it. Use this skill as soon as a notebook is open in the browser pane of the Claude app and the Jupyter MCP tools are connected to it, whether the person asked only to open it, or to run it, work on its code, or try something in it. Also use it when you need to monitor what the person does in the attached notebook, to see how they progress and whether they need help. Use it together with guiding-jupylet-users.
---

# Working in notebooks

With this skill you work in a Jupyter notebook together with the person:
you read, change and run its cells, and you monitor what the person does
in it. The notebook is open in the Claude web browser, and you are connected
to it with the Jupyter MCP tools.

## Tasks

- "Work in the notebook": read and change cells together with the person.
- "Run the notebook": run all cells, and tell the person the result.
- "Monitor the notebook": monitor what the person does in the notebook, and
  react to it.

## Names and values

Values in `{braces}` are blanks that you fill in. You know them from when
the notebook was opened:

- `{code}`: the full path of the Jupylet folder, the folder that contains
  `.claude/skills/working-in-notebooks`.
- `{python}`: the Python of the environment that runs Jupylet.
- `{port}`, `{token}`: the port and the token of the Jupyter server.
- `{folder}`: the folder that the Jupyter server serves.
- `{notebook}`: the path of the notebook, from `{folder}`, for example
  `11-spaceship.ipynb`.
- `{notebook name}`: the file name without `.ipynb`, for example
  `11-spaceship`.
- `{scratchpad}`: your scratchpad folder.

Names in quotes:

- **Do "Name":** a task of this file, for example: do "Run the notebook".
- **Run "Name" from the reference file:** a command in the reference file
  for this computer, `references/macos.md` or `references/windows.md`.
- **The troubleshooting file:** `references/troubleshooting.md`, for
  problems with cells and runs.

## Rules for this skill

- **Run cells only when the person asks for it, or when another skill tells
  you to.**
- **Add, change, move or delete cells only when the person asks for it, or
  after a yes.** Do this only with the Jupyter MCP tools, never in the `.ipynb`
  file on the disk. You can read the saved file.
- **Work on notebooks together only in the Claude web browser.** Read, change
  and run cells only in a notebook that the Claude web browser shows. There
  you can monitor what the person sees, and check it. A web page in the default
  web browser is out of your reach. If the person works there and only asks
  for advice, answer, and read the saved notebook file when you need its code.

## Work in the notebook

In this task you read, edit and run the notebook together with the person.
Before the first tool call, read `references/jupyter-mcp-tools.md`. When you
use a Jupyter MCP tool, run it with "Call a tool" from the reference file.

- **Verify before you describe.** Never tell the person what is on their
  screen until you have confirmed it.
- **Check that the person still sees the notebook.** When you have reason
  to think that the web page or the browser pane closed, for example a
  notebook tool fails or the person says so, call `tabs_context` of the
  browser pane. If no tab has the origin `http://localhost:{port}`, or the
  pane does not show, tell the person, and continue once they see the
  notebook again.
- **Match the evidence to the output.** Graphical output, such as a canvas
  or an image, calls for a screenshot of the web page before you describe
  what changed. Text output calls for `read_cell`, and a screenshot adds
  nothing to it.
- **Diagnose from the full output.** When a cell fails, read its entire
  output and locate the underlying cause. The last line often shows only
  a symptom.
- **Clean up temporary code.** When you add code to verify something or
  to try an idea, clean up afterwards. This needs no yes. If you keep
  something, tell the person.
- **Monitor the notebook when you expect the person to act in it.**
  Whether you asked the person to do something there, or the person says
  that they are about to, do "Monitor the notebook".

## Run the notebook

This task runs every cell, waits for the run to end, and tells the person
how it went. Do it only when the person asks for it, or when another skill
tells you to.

1. Start the run. Tell the person, for example "Running the notebook
   now." Run "Run all cells and wait" from the reference file in the
   background, with 90 seconds, and end your turn. Its answer arrives as a
   message:
   - **If the answer is `finished`,** the run is over. Continue to step 2.
   - **If the answer is `still running`,** the run outlasted the wait.
     Tell the person that the notebook is still running, and extend the
     wait with "Wait for a run", with the fingerprint from the answer and
     times of 5 minutes, then 10 minutes. Once it prints `finished`,
     continue to step 2. If it never does, tell the person that a cell is
     still running, and stop here.
   - **If the answer is `not started`,** the run never began. Follow
     the troubleshooting file, and stop here.
2. Inspect the outcome. Read the whole notebook with the Jupyter tool
   `read_notebook`, through "Call a tool" from the reference file, with
   `{"notebook_name": "{notebook name}", "response_format": "detailed", "limit": 0}`.
   Each cell comes back with its source, its execution count and its
   output. For each cell that failed, trace the error to its cause.
3. Report. Tell the person in plain words what the notebook now shows, or
   what went wrong and why. If you are not sure what it should show, read
   its cells first.

## Monitor the notebook

Once in a conversation, when a notebook is open and you are attached to it,
offer the person a helper that follows along in the notebook. Also, when
you ask the person to do something in the notebook, or they say that they
are about to, start a `notebook-monitor` in the same turn. For both, follow
`references/monitoring.md`. The monitor reads the notebook in the background
and sends you notes, so that you remain free to talk with the person.
