# Commands for Windows 11

The tasks in `SKILL.md` and in the reference files of this skill run these
commands by their names, for example: run "Call a tool" from the reference
file. Run each command with the PowerShell tool. Never use the Bash tool,
even if it exists. Fill in the blanks in `{braces}`: "Names and values" in
`SKILL.md` defines them. Each command has a short description of what it
does, and its expected output.

Run each command exactly as this file shows it, alone in one call, with its
comment `# jupylet: {name}` at the end. The Jupylet system uses these comments to find each command and check it.
If you leave out the comment or change the command, this may cause
failures. Do not add `| Out-Null`,
`| Select-String`, `| Select-Object -First` or other commands to it: you
need its whole output to see an error.

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

General rules for these commands:



- Put `&` in front of a command that starts with a program path in quotes,
  for example `& "{python}" -m jupylet.claude ...`.
- The Python of a conda folder or environment is `python.exe` in it.
- Run each `python -m jupylet.claude` command from `{code}`: run
  `Set-Location "{code}"` first. From another folder, Python may not find
  `jupylet.claude`. The PowerShell tool keeps the current folder between
  commands.
- Windows PowerShell 5.1 deletes the double quotes inside an argument that it
  gives to a program. For a long `python -c` command, write a `.py` file into
  the scratchpad, and run that file.

*Developer note, not an instruction: tested on one Windows 11 Home computer.
Not tried: Windows 10, ARM, a folder with spaces or in OneDrive.*

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

`Set-Location "{code}"; & "{python}" -m jupylet.claude prepare --check  # jupylet: notebooks/check-env`

Checks the Jupyter settings that Jupylet needs in a notebook with Claude,
and changes nothing. Prints `ready` if the environment has all of them, or
`needs:` and the parts that are missing, for example `needs: nbmodel`.

## Attach

`Set-Location "{code}"; & "{python}" -m jupylet.claude attach {port} {token} {notebook}  # jupylet: notebooks/attach`

Connects the Jupyter MCP tools to the notebook. The notebook must be open in a
web page. Expected: the output contains `Successfully activate notebook`.

## Call a tool

Give the JSON on stdin, in a here-string with single quotes:

```
@'
{"notebook_name": "{notebook name}", "response_format": "detailed", "limit": 0}
'@ | & "{python}" -m jupylet.claude call {port} {token} read_notebook -  # jupylet: notebooks/call-tool
```

Or write the JSON into a file in the scratchpad, and give `@{file}` as the
last argument. Tools without arguments need neither. Run it from `{code}`.

Some tools need the id of the kernel:

`Set-Location "{code}"; & "{python}" -m jupylet.claude kernel {port} {token} {notebook}  # jupylet: notebooks/kernel-id`

## Run all cells

`Set-Location "{code}"; & "{python}" -m jupylet.claude call {port} {token} notebook_run-all-cells  # jupylet: notebooks/run-all`

Runs all cells, as Run > Run All Cells in the web page does. Prints `True` after
a second or two. The cells can still run after that.

## Run all cells and wait

`Set-Location "{code}"; & "{python}" -m jupylet.claude run-all {port} {token} {notebook} {seconds}  # jupylet: notebooks/run-all-wait`

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

`Set-Location "{code}"; & "{python}" -m jupylet.claude run-cell {port} {token} {cell index} "{start of its source}"  # jupylet: notebooks/run-cell`

Moves the active cell of the web page to the cell, one cell at a time. It checks
that the source of the cell starts with the text that you give, and only then
runs it. Prints `True`, or why it did not run. It moves the cursor of the
person.

*Developer note, not an instruction: `run-cell` is not tested on Windows
yet.*

## Wait for a run

`Set-Location "{code}"; & "{python}" -m jupylet.claude wait-change {port} {token} {notebook} {seconds} {since}  # jupylet: notebooks/wait-change`

Run it in the background. Prints `finished: ...` when the kernel ran
something and is idle again, or `not finished: ...` after {seconds}. Both
end with the fingerprint.

*Developer note, not an instruction: `wait-change` is not tested on Windows
yet.*

## Replace the kernel

`Set-Location "{code}"; & "{python}" -m jupylet.claude replace-kernel {port} {token} {notebook}  # jupylet: notebooks/replace-kernel`

Stops the kernel of the notebook, starts a new one, waits until it is ready
(about 10 seconds), and prints the new id. The variables of the old kernel
are lost.

## Trust the notebooks

`Set-Location "{folder}"; & "{python}" -m jupylet trust_notebooks  # jupylet: notebooks/trust-notebooks`

Marks each notebook in `{folder}` as trusted, so that the cells show the
canvas. Run it from inside `{folder}`. In the folder above `examples`, Python
finds the code folder, also named `jupylet`, instead of the installed
Jupylet, and the command fails.
