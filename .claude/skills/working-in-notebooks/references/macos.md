# Commands for macOS

The tasks in `SKILL.md` and in the reference files of this skill run these
commands by their names, for example: run "Call a tool" from the reference
file. Run each command with the Bash tool. Fill in the blanks in `{braces}`:
"Names and values" in `SKILL.md` defines them. Each command has a short
description of what it does, and its expected output.

Run each command exactly as this file shows it, alone in one call, with its
comment `# jupylet: {name}` at the end. The Jupylet system uses these comments to find each command and check it.
If you leave out the comment or change the command, this may cause
failures. Do not add `>/dev/null`, `| grep`,
`| head` or other commands to it: you need its whole output to see an error.

Each `python -m jupylet.claude` command starts with `cd "{code}"`. In another
folder, a folder named `jupylet` can hide the installed Jupylet.

Some blanks appear only here:

- `{tool}`, `{json arguments}`: a notebook tool and its arguments, from
  `references/jupyter-mcp-tools.md`.
- `{cell index}`: the place of a cell in the notebook, from 0.
- `{start of its source}`: the first words of the source of that cell.
- `{seconds}`: how long a monitoring command waits, from the task that
  runs it.
- `{since}`, `{fingerprint}`: the fingerprint that the last "Wait for a run"
  printed.
- `{file}`: a file in the scratchpad.

## Contents

- Check the environment
- Attach
- Call a tool
- Run all cells
- Run all cells and wait
- Run one cell
- Wait for a run
- Replace the kernel
- Trust the notebooks

## Check the environment

`cd "{code}" && "{python}" -m jupylet.claude prepare --check  # jupylet: notebooks/check-env`

Checks the Jupyter settings that Jupylet needs in a notebook with Claude,
and changes nothing. Prints `ready` if the environment has all of them, or
`needs:` and the parts that are missing, for example `needs: nbmodel`.

## Attach

`cd "{code}" && "{python}" -m jupylet.claude attach {port} {token} {notebook}  # jupylet: notebooks/attach`

Connects the Jupyter MCP tools to the notebook. The notebook must be open in a
web page. Expected: the output contains `Successfully activate notebook`.

## Call a tool

`cd "{code}" && "{python}" -m jupylet.claude call {port} {token} {tool} '{json arguments}'  # jupylet: notebooks/call-tool`

Calls one notebook tool. Instead of the JSON, the last argument can be
`@{file}` (read the JSON from a file) or `-` (read it from stdin). Tools
without arguments need no last argument.

Some tools need the id of the kernel:

`cd "{code}" && "{python}" -m jupylet.claude kernel {port} {token} {notebook}  # jupylet: notebooks/kernel-id`

## Run all cells

`cd "{code}" && "{python}" -m jupylet.claude call {port} {token} notebook_run-all-cells  # jupylet: notebooks/run-all`

Runs all cells, as Run > Run All Cells in the web page does. Prints `True` after
a second or two. The cells can still run after that.

## Run all cells and wait

`cd "{code}" && "{python}" -m jupylet.claude run-all {port} {token} {notebook} {seconds}  # jupylet: notebooks/run-all-wait`

Run it in the background. It starts to watch the notebook, then runs all
cells, as Run > Run All Cells in the web page does, and waits until the run ends.
Prints `finished: ...` when the run is over, `still running: ...` when it
is not over after {seconds}, or `not started: ...` with the message of the
web page. The first two end with the fingerprint. When the web page says
"Timeout waiting for result", the run did start, and the command keeps
waiting. It must watch before the run starts: a fresh kernel
gives the same execution counts as the saved notebook, so a watch that starts
after the run sees no change.

## Run one cell

`cd "{code}" && "{python}" -m jupylet.claude run-cell {port} {token} {cell index} "{start of its source}"  # jupylet: notebooks/run-cell`

Moves the active cell of the web page to the cell, one cell at a time. It checks
that the source of the cell starts with the text that you give, and only then
runs it, because the wrong cell can restart a game. Prints `True`, or why it
did not run. It moves the cursor of the person.

## Wait for a run

`cd "{code}" && "{python}" -m jupylet.claude wait-change {port} {token} {notebook} {seconds} {since}  # jupylet: notebooks/wait-change`

Run it in the background. Prints `finished: ...` when the kernel ran
something and is idle again, or `not finished: ...` after {seconds}. Both
end with the fingerprint.

## Replace the kernel

`cd "{code}" && "{python}" -m jupylet.claude replace-kernel {port} {token} {notebook}  # jupylet: notebooks/replace-kernel`

Stops the kernel of the notebook, starts a new one, waits until it is ready
(about 10 seconds), and prints the new id. The variables of the old kernel
are lost.

## Trust the notebooks

`cd "{folder}" && "{python}" -m jupylet trust_notebooks  # jupylet: notebooks/trust-notebooks`

Marks each notebook in `{folder}` as trusted, so that the cells show the
canvas. Run it from inside `{folder}`. In the folder above `examples`, Python
finds the code folder, also named `jupylet`, instead of the installed
Jupylet, and the command fails.
