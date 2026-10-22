# What the Jupylet commands print

For the system simulator. These are the real outputs of the commands of
`jupylet/claude.py` and of `python -m jupylet`. Give exactly these texts,
with the values of the current state. Take the outputs from the real
sessions in `findings.md` when you are not sure.

The commands run as `"{python}" -m jupylet.claude <command> …` from the
Jupylet folder, or as `"{python}" "{code}/jupylet/claude.py" <command> …`.

## jupylet.claude

- `find-env [--all] <version>`: one line for each conda environment that has
  Jupylet, with tabs: `<env folder>\tconda\t<source>`. `<source>` is
  `this folder` when the environment runs the code of this folder, or
  `<version> from <folder>` otherwise. Example:
  `/Users/nir/miniforge3/envs/jupylet2	conda	0.10.0 from /Users/nir/jupylet2`.
  No environment: no output.
- `running`: a JSON list, pretty-printed with an indent of 2. One object for
  each Jupyter server that runs, with the keys `port`, `token`, `folder`,
  `environment`, `jupylet_version`, `mcp_tools`, `nbmodel_off`, and
  `notebooks` (a list of objects with `path` and `connections`). No server:
  `[]`.
- `prepare`: `turned off` the first time in an environment, `off` after
  that. If the environment does not have `jupyter_server_nbmodel`:
  `not installed`.
- `prepare --check`: `ready`, or `needs: ` and a list, for example
  `needs: live copy, news`.
- `detach <log> <command…>`: the process ID of the new process, for example
  `23413`. It starts the program and returns at once.
- `wait <port> <token>`: `ready` when a Jupyter server answers on the port
  within 60 seconds, otherwise `not ready after 60 seconds`, with exit code
  1.
- `wait-open <port> <token> <notebook> <seconds>`: `open` as soon as the
  Jupyter server has a session for the notebook, otherwise `timeout`. A page
  starts that session only after it signs in. But a session that another
  page started earlier counts too: then `wait-open` prints `open` at once,
  also when the new page has not signed in yet (W2). It runs until it can
  print, so it is usually started in the background.
- `attach <port> <token> <notebook>`: about ten lines. They start with
  `[INFO] Connected to kernel '<uuid>'.`, contain
  `[INFO] Successfully activate notebook '<name>'.`, and end with a table of
  the first 20 cells. With no kernel:
  `No kernel for '<notebook>': is it open in the browser?`
- `run-all <port> <token> <notebook> <seconds>`: `ran <fingerprint>` when
  the run ended in time, `timeout <fingerprint>` otherwise, or
  `error <message>` when the run did not start. A fingerprint looks like
  `7117831f.a198e936`. In a Jupylet notebook, `app.run()` returns at once,
  and the app keeps running in the page. So `run-all` ends with `ran`, within
  a minute on the first run, and the output of the cell with `app.run()` is
  the canvas.
- `kernel <port> <token> <notebook>`: the id of the kernel of the notebook,
  for example `3f9a1c52-8e4b-4d7a-9b1e-2c6f0a7d5e81`. With no kernel:
  `no kernel yet: is the notebook open in the browser?`
- `tools <port> <token>`: the names of the Jupyter MCP tools, one on each
  line, for example `use_notebook`, `read_notebook`, `read_cell`,
  `insert_cell`, `execute_code`, `notebook_run-all-cells`.
- `wait-change <port> <token> <notebook> <seconds> [<since>]`: waits until
  the person, or a run, ran something in the notebook, and the kernel is idle
  again. Prints `ran <fingerprint>`, or `timeout <fingerprint>` when nothing
  ran in time. Typing alone does not end it. It runs until then, so it is
  usually started in the background.
- `replace-kernel <port> <token> <notebook>`: the id of the new kernel. It
  takes about ten seconds.
- `run-cell <port> <token> <cell index> <start of its source>`: `True` when
  the cell ran, or `not run: ` and the reason, for example
  `not run: cell 3 starts with 'app.run()'`.
- `cleanup`: one line `would delete <path>` for each state file of Jupyter,
  or `nothing to clean up`. `cleanup --yes`: `deleted <path>` for each one.
- `call <port> <token> <tool> [<json>]`: the answer of a Jupyter MCP tool.
  An unknown tool: `ERROR: Internal error calling tool: Unknown tool: <tool>`.
  `read_notebook` with `{"notebook_name": …, "response_format": "brief"}`
  prints `Notebook <name> has <n> cells.` and a table of cells.
- `shutdown <port> <token>`: `stopped`. With no server on the port:
  `not running`. When a kernel does not stop:
  `a kernel is still running: not shutting the server down`. Other outputs:
  `stopped after ending its process`, `still running`.

Each command in this list exists in `claude.py`. Never answer a command of
this list with a usage error. A usage error, with exit code 1, comes only
for a command name that is not in this list, or for a wrong number of
arguments.

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

