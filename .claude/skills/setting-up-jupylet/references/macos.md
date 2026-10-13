# Commands for macOS

Run each command with the Bash tool. Fill in the blanks in `{braces}`. Each
command has a short description of what it does.

Run each command exactly as this file shows it, alone in one call, with its
comment `# jupylet: {name}` at the end. The Jupylet system uses these comments to find each command and check it.
If you leave out the comment or change the command, this may cause
failures. Do not add `>/dev/null`, `| grep`,
`| head` or other commands to it: you need its whole output to see an error.

## Contents

- Check the system
- Find conda installations
- Get the Python version of Miniforge
- Find environments with Jupylet
- Find the conda of Terminal
- Install Miniforge
- Prepare Terminal
- Create the environment
- Install Jupylet, Check Jupylet, Prepare the environment
- Trust the notebooks, Check the trust
- Check the Terminal

## Check the system

`uname -s && uname -m  # jupylet: setup/check-system`

Prints the system name and the processor type. `Darwin` and `arm64` is a Mac
with an Apple chip. `Darwin` and `x86_64` is a Mac with an Intel chip.

## Find conda installations

`ls -d "$HOME/miniforge3/bin/conda" "$HOME/mambaforge/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "$HOME/opt/miniconda3/bin/conda" "$HOME/opt/anaconda3/bin/conda" /opt/homebrew/Caskroom/miniforge/base/bin/conda /opt/homebrew/Caskroom/miniconda/base/bin/conda /opt/miniconda3/bin/conda /opt/anaconda3/bin/conda 2>/dev/null || true  # jupylet: setup/find-conda`

Prints the `bin/conda` of each usual conda folder that has one. The conda
folder is the path before `/bin/conda`, for example `/opt/miniconda3`. A folder
without `bin/conda` is not a conda installation, for example a folder that
stays after an uninstall. `|| true` ends the command without an error code
when some of them do not exist.

## Get the Python version of Miniforge

`"$HOME/miniforge3/bin/python" --version  # jupylet: setup/miniforge-python`

Prints the Python version of the main environment of Miniforge, `base`.

## Find environments with Jupylet

`"{python}" "{code}/jupylet/claude.py" find-env --all {version}  # jupylet: setup/find-jupylet-envs`

`{python}` is `bin/python` in a conda folder from "Find conda installations".
Use `{miniforge}` if it exists.

The script asks each conda environment on the computer which Jupylet it has.
It prints one line for each environment with Jupylet: the path, the kind, and
where its Jupylet comes from. `this folder` means it runs the code in
`{code}`. The name of an environment is the last part of its path.

## Find the conda of Terminal

`grep -h -o "'[^']*/bin/conda'" "$HOME/.zshrc" "$HOME/.zprofile" 2>/dev/null || true  # jupylet: setup/terminal-conda`

Prints the conda that the Terminal settings start, if any. `|| true` ends the
command without an error code when it finds nothing.

- Nothing: Terminal uses no conda.
- `'{miniforge}/bin/conda'`: Terminal uses Miniforge.
- Another path: Terminal uses another conda.

## Install Miniforge

Run these two commands, each in its own call:

1. `curl -fL -o "{scratchpad}/Miniforge3.sh" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh  # jupylet: setup/download-miniforge`

   It downloads the latest Miniforge installer for Apple chips into the
   scratchpad, a temporary folder of this session.
2. Only if the download succeeded:
   `bash "{scratchpad}/Miniforge3.sh" -b -p "$HOME/miniforge3"  # jupylet: setup/run-miniforge-installer`

   It runs the installer without questions, into `~/miniforge3`.

The installer runs only if its file name ends in `.sh`. That is why the file
has the name `Miniforge3.sh`.

Check: `"$HOME/miniforge3/bin/conda" --version  # jupylet: setup/check-miniforge` prints a line like
`conda 26.x.x`.

## Prepare Terminal

`test -f "$HOME/.zshrc" && echo exists || echo missing  # jupylet: setup/zshrc-exists`

Prints `exists` if the Terminal settings file `~/.zshrc` exists, and `missing`
if it does not. A Mac has no `~/.zshrc` until a program or the person makes
one.

- **`exists`:** back it up:

  `cp -n "$HOME/.zshrc" "$HOME/.zshrc.before-miniforge" || true  # jupylet: setup/backup-terminal`

  Copies the Terminal settings to a backup file. If the backup exists already,
  from an earlier setup, it keeps it as it is. `|| true` ends the command
  without an error code in that case.
- **`missing`:** there is nothing to back up. Continue.

`"$HOME/miniforge3/bin/conda" init zsh  # jupylet: setup/conda-init`

Adds lines to the Terminal settings, so that new Terminal windows use
Miniforge.

To put the settings back, copy `~/.zshrc.before-miniforge` to `~/.zshrc`. Ask
the person first: this also deletes all changes in `~/.zshrc` after the
backup. If the backup does not exist, `~/.zshrc` did not exist before the
setup.

## Create the environment

`"$HOME/miniforge3/bin/conda" create -y -p "$HOME/miniforge3/envs/{env}" --override-channels -c conda-forge python=3.13 moderngl glcontext  # jupylet: setup/create-env`

Creates the environment `{env}` in Miniforge's `envs` folder, with Python 3.13
and the graphics libraries that Jupylet uses. `-p` with the full path puts the
environment in `envs` even if the conda settings of the person say otherwise.
`--override-channels -c conda-forge` uses only packages from conda-forge.

Check: `"$HOME/miniforge3/envs/{env}/bin/python" --version  # jupylet: setup/check-env-python` prints
`Python 3.13.x`.

For an existing environment that the person chose in step 5, skip this
command. Its Python is `bin/python` in its folder.

## Install Jupylet, Check Jupylet, Prepare the environment

`{env_python}` is `"$HOME/miniforge3/envs/{env}/bin/python"`, or the Python of
the environment that the person chose.

Install Jupylet:

`"{env_python}" -m pip install -e "{code}[claude]"  # jupylet: setup/install-jupylet`

Installs Jupylet from `{code}`, with the extra tools for Claude. `-e` means
that the environment uses the code in `{code}` directly.

Check Jupylet:

`"{env_python}" -I -c "import jupylet; print(jupylet.VERSION)"  # jupylet: setup/check-jupylet`

Prints the version of the Jupylet that the environment uses. `-I` stops
Python from finding a folder named `jupylet` in the current folder.

Prepare the environment:

`"{env_python}" -I -m jupylet.claude prepare  # jupylet: setup/prepare-env`

Changes the Jupyter settings of the environment:

- It turns off `jupyter_server_nbmodel`, which can make a cell that runs a
  game or a widget stop responding.
- It keeps the live copy of each notebook while Jupyter runs, so that the
  page shows the cells that Claude adds.
- It turns off the news pop-up of JupyterLab.

## Trust the notebooks, Check the trust

Trust the notebooks:

`cd "{code}/examples" && "{env_python}" -m jupylet trust_notebooks  # jupylet: setup/trust-notebooks`

Check the trust:

`cd "{code}/examples" && "{env_python}" -m jupylet is_trusted  # jupylet: setup/check-trust`

Run both from inside `examples`. In the folder above it, Python finds the
code folder, also named `jupylet`, instead of the installed Jupylet, and the
command fails.

## Check the Terminal

`"$SHELL" -ic 'conda info --base'  # jupylet: setup/check-terminal`

Starts the shell of the person as a new Terminal window does, and prints which
conda it uses.
