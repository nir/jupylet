# Commands for macOS

Run each command with the Bash tool. Fill in the blanks in `{braces}`. Each
command has a short description of what it does.

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

`uname -s && uname -m`

Prints the system name and the processor type. `Darwin` and `arm64` is a Mac
with an Apple chip. `Darwin` and `x86_64` is a Mac with an Intel chip.

## Find conda installations

`ls -d "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/opt/miniconda3" "$HOME/opt/anaconda3" /opt/homebrew/Caskroom/miniforge/base /opt/homebrew/Caskroom/miniconda/base /opt/miniconda3 /opt/anaconda3 2>/dev/null`

Prints each of the usual conda folders that exists. Keep only a folder that
contains `bin/conda`: a folder can stay after an uninstall.

## Get the Python version of Miniforge

`"$HOME/miniforge3/bin/python" --version`

Prints the Python version of the main environment of Miniforge, `base`.

## Find environments with Jupylet

`"{python}" "{code}/jupylet/claude.py" find-env --all {version}`

`{python}` is `bin/python` in a conda folder from "Find conda installations".
Use `{miniforge}` if it exists.

The script asks each conda environment on the computer which Jupylet it has.
It prints one line for each environment with Jupylet: the path, the kind, and
where its Jupylet comes from. `this folder` means it runs the code in
`{code}`. The name of an environment is the last part of its path.

## Find the conda of Terminal

`grep -h -o "'[^']*/bin/conda'" "$HOME/.zshrc" "$HOME/.zprofile" 2>/dev/null`

Prints the conda that the Terminal settings start, if any.

- Nothing: Terminal uses no conda.
- `'{miniforge}/bin/conda'`: Terminal uses Miniforge.
- Another path: Terminal uses another conda.

## Install Miniforge

`d="$(mktemp -d)" && curl -fL -o "$d/Miniforge3.sh" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh && bash "$d/Miniforge3.sh" -b -p "$HOME/miniforge3"; rm -rf "$d"`

The command does these steps:

1. It makes a new, private temporary folder.
2. It downloads the latest Miniforge installer for Apple chips into it.
3. It runs the installer without questions, into `~/miniforge3`.
4. It deletes the temporary folder.

The installer runs only if its file name ends in `.sh`. That is why the file
has the name `Miniforge3.sh`.

Check: `"$HOME/miniforge3/bin/conda" --version` prints a line like
`conda 26.x.x`.

## Prepare Terminal

`test -f "$HOME/.zshrc" && cp -n "$HOME/.zshrc" "$HOME/.zshrc.before-miniforge"`

Copies the Terminal settings to a backup file, if the settings file exists and
no backup exists yet.

`"$HOME/miniforge3/bin/conda" init zsh`

Adds lines to the Terminal settings, so that new Terminal windows use
Miniforge.

To put the settings back, copy `~/.zshrc.before-miniforge` to `~/.zshrc`. Ask
the person first: this also deletes all changes in `~/.zshrc` after the
backup. If the backup does not exist, `~/.zshrc` did not exist before the
setup.

## Create the environment

`"$HOME/miniforge3/bin/conda" create -y -p "$HOME/miniforge3/envs/{env}" --override-channels -c conda-forge python=3.13 moderngl glcontext`

Creates the environment `{env}` in Miniforge's `envs` folder, with Python 3.13
and the graphics libraries that Jupylet uses. `-p` with the full path puts the
environment in `envs` even if the conda settings of the person say otherwise.
`--override-channels -c conda-forge` uses only packages from conda-forge.

Check: `"$HOME/miniforge3/envs/{env}/bin/python" --version` prints
`Python 3.13.x`.

For an existing environment that the person chose in step 5, skip this
command. Its Python is `bin/python` in its folder.

## Install Jupylet, Check Jupylet, Prepare the environment

`{env_python}` is `"$HOME/miniforge3/envs/{env}/bin/python"`, or the Python of
the environment that the person chose.

Install Jupylet:

`"{env_python}" -m pip install -e "{code}[claude]"`

Installs Jupylet from `{code}`, with the extra tools for Claude. `-e` means
that the environment uses the code in `{code}` directly.

Check Jupylet:

`"{env_python}" -I -c "import jupylet; print(jupylet.VERSION)"`

Prints the version of the Jupylet that the environment uses. `-I` stops
Python from finding a folder named `jupylet` in the current folder.

Prepare the environment:

`"{env_python}" -I -m jupylet.claude prepare`

Changes the Jupyter settings of the environment:

- It turns off `jupyter_server_nbmodel`, which can make a cell that runs a
  game or a widget stop responding.
- It keeps the live copy of each notebook while Jupyter runs, so that the
  page shows the cells that Claude adds.
- It turns off the news pop-up of JupyterLab.

## Trust the notebooks, Check the trust

Trust the notebooks:

`cd "{code}/examples" && "{env_python}" -m jupylet trust_notebooks`

Check the trust:

`cd "{code}/examples" && "{env_python}" -m jupylet is_trusted`

Run both from inside `examples`. In the folder above it, Python finds the
code folder, also named `jupylet`, instead of the installed Jupylet, and the
command fails.

## Check the Terminal

`"$SHELL" -ic 'conda info --base'`

Starts the shell of the person as a new Terminal window does, and prints which
conda it uses.
