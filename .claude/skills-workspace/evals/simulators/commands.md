# What the Jupylet commands print

For the system simulator. These are the real outputs of the commands of
`jupylet/claude.py` and of `python -m jupylet`. Give exactly these texts,
with the values of the current state. Take the outputs from the real
sessions in `findings.md` when you are not sure.

The commands run as `"{python}" -m jupylet.claude <command> …` from the
Jupylet folder, or as `"{python}" "{code}/jupylet/claude.py" <command> …`.

## jupylet.claude

"The documentation of jupylet/claude.py", right after this part, describes
each command and what it prints. It comes from the code under test, so it is
always current. Give exactly the outputs that it describes, with the values
of the current state.

What the documentation does not say:

- **A notebook with a Jupylet app:** `app.run()` returns at once, and the app
  keeps running in the web page. So `run-all` ends with `finished: the run
  is over. Fingerprint: …`, within a minute on the first run, and the
  output of the cell with `app.run()` is the canvas.
- **`attach` with no kernel:**
  `No kernel for '<notebook>': is it open in the browser?`
- **`attach` on success** ends with a table of the first 20 cells.
- **`call` with an unknown tool:**
  `ERROR: Internal error calling tool: Unknown tool: <tool>`.
- **`call` with `read_notebook`** and `{"notebook_name": …, "response_format":
  "brief"}`: `Notebook <name> has <n> cells.` and a table of the cells.
- **`replace-kernel`** takes about ten seconds.
- **`wait-open`, `wait-change` and `run-all`** run until they can print, so
  Claude usually starts them in the background.

Each command in the documentation exists in `claude.py`. Never answer one of
them with a usage error. A usage error, with exit code 1, comes only for a
command name that is not in the documentation, or for a wrong number of
arguments: then print the usage list of the documentation.

## python -m jupylet

Run in the folder `examples`:

- `is_trusted`: one line for each notebook, as `./<name>.ipynb: trusted` or
  `./<name>.ipynb: NOT TRUSTED`.
- `trust_notebooks`: one line for each notebook, as
  `./<name>.ipynb: trusted`.

The example notebooks are `01-hello-world`, `02-hello-jupylet`,
`11-spaceship`, `12-spaceship-3d`, `13-lego-3d`, `14-piano`, `15-sonic`,
`16-shadertoy-demo`, `17-spectrum-analyzer`, `18-filters`, `19-synthing`,
`21-pong`, `22-pong-RL` and `31-breakout`.

## Other checks of the skills

- The port check prints `free` or `in use`.
- The sign-in check in the page (`fetch('/api/status')`) gives `403` before
  the person pastes the token, and `200` after.
- The side bar script gives `undefined`, or the text that the script
  returns.
- "Browse notebooks" (`notebooks/browse-notebooks`) runs until the person
  chooses a file in the window, so Claude usually starts it in the
  background. It prints the full path of the chosen file, or `canceled`.
  The person chooses what the scenario says. If the scenario says nothing,
  the person chooses the notebook that they meant.

## conda init

`"{conda}/bin/conda" init zsh` changes `~/.zshrc` as the real conda does
(`conda/core/initialize.py`, `init_sh_user`):

- **The file has a `# >>> conda initialize >>>` block,** from any conda
  installation: conda replaces each such block, in place, with the block of
  this conda. The lines outside the blocks stay. It does not add a second
  block.
- **The file has no such block, or does not exist:** conda adds the block at
  the end, and creates the file if necessary.

It prints one line for each file that it checks, `no change` or `modified`
and the path, for example `modified      /Users/dana/.zshrc`. If it changed
something, the last line is
`==> For changes to take effect, close and re-open your current shell. <==`.

