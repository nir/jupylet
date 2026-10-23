---
name: working-in-notebooks
description: Reads, changes and runs the cells of an open Jupyter notebook together with the person, and monitors what the person does in it. Use this skill once a notebook is open in the browser pane of the Claude app and attached, for example by managing-jupylet-notebooks, and the person wants to run it, work on its code, or try something in it. Also use it when Claude expects the person to act in the notebook, for example after Claude asked them to try something, or when they say that they are about to. Use it together with guiding-jupylet-users.
---

# Working in notebooks

With this skill you work in a Jupyter notebook together with the person:
you read, change and run its cells, and you see what the person does in
it. The skill starts where `managing-jupylet-notebooks` ends: the notebook
is open in the Claude web browser, and you are attached to it. When the
work is done, or the person wants to stop Jupyter, go back to that skill.

## Tasks

- "Work in the notebook": read and change cells together with the person.
- "Run the notebook": run all cells, and tell the person the result.
- "Monitor the notebook": see what the person does in the notebook, and
  react to it.

## Names and values

This skill uses the values, the terms and the reference files of
`managing-jupylet-notebooks`, for example `{port}`, `{token}` and
`{notebook name}`:

- **Do "Name":** a task of this file, or of `managing-jupylet-notebooks`,
  for example "Find the Jupyter servers that run".
- **Run "Name" from the reference file:** a command in the reference file
  for this computer of `managing-jupylet-notebooks`:
  `references/macos.md` or `references/windows.md`.
- **The troubleshooting file:** `references/troubleshooting.md` of this
  skill, for problems with cells and runs. Problems with the Jupyter
  server, the web page or the sign-in are in the troubleshooting file of
  `managing-jupylet-notebooks`.

## Rules for this skill

- **Run cells only when the person asks for it, or when another skill tells
  you to.**
- **Add, change, move or delete cells only when the person asks for it, or
  after a yes.** Do this only with the Jupyter MCP tools, never in the `.ipynb`
  file on the disk. You can read the saved file.
- **Work on notebooks together only in the Claude web browser.** Read, change
  and run cells only in a notebook that the Claude web browser shows. There
  you can see what the person sees, and check it. A web page in the default
  web browser is out of your reach. If the person works there and only asks
  for advice, answer, and read the saved notebook file when you need its code.

## Work in the notebook

In this task you read, edit and run the notebook together with the person.
Before the first tool call, read `references/jupyter-mcp-tools.md`, and
invoke every tool through "Call a tool" from the reference file.

- **Verify before you describe.** Never tell the person what is on their
  screen until you have confirmed it.
- **Recheck that the notebook is still open.** If a notebook tool fails,
  or the person suggests that the notebook or its web page closed, for
  example "I'm back" or "it closed", use "Find the Jupyter servers that
  run" to confirm that the notebook is still open and that a web page
  shows it.
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

This task lets you see what the person does in the notebook, and stay
free to answer in the chat. Before you ask the person to do something in
the notebook, and before you start a monitoring command, read
`references/waiting.md`. It has the whole task.

*Developer note, not an instruction: the plan is a command in
`jupylet/claude.py` that waits for any change in the notebook, an edit or a
run, and returns a short diff of the changed lines. A `background-monitor`,
started with the skill `running-background-monitors` and a brief from this
skill, runs that command, judges each diff, and sends the main chat a note
only when something deserves a look.*
