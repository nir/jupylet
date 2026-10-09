# Commands for Windows 11

Run each command with the PowerShell tool. Never use the Bash tool, even if it
exists. A command that starts with a program path in quotes needs `&` in
front, for example `& "{env_python}" -m pip ...`. Fill in the blanks in
`{braces}`. Each command has a short description of what it does.

Windows PowerShell 5.1 does not keep the double quotes inside an argument that
it gives to a program. For a long `python -c` command, write a `.py` file into the
scratchpad and run that file instead.

## Contents

- Check the system
- Find conda installations
- Get the Python version of Miniforge
- Find environments with Jupylet
- Install Miniforge
- Create the environment
- Install Jupylet, Check Jupylet, Prepare the environment
- Trust the notebooks, Check the trust

## Check the system

`$env:PROCESSOR_ARCHITECTURE; [Environment]::OSVersion.Version.Build`

Prints the processor type and the Windows build number. `AMD64` and a number
of `22000` or more is Windows 11 on Intel or AMD.

## Find conda installations

Run both commands.

`Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*','HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*' -ErrorAction SilentlyContinue | Where-Object { $_.DisplayName -match 'Anaconda|Miniconda|Miniforge|Mambaforge' } | ForEach-Object { Split-Path ($_.UninstallString -replace '"', '') } | Where-Object { Test-Path $_ } | Sort-Object -Unique`

Prints the folders of the conda installations that Windows lists as installed
programs (Settings > Apps).

`'miniforge3', 'mambaforge', 'miniconda3', 'anaconda3' | ForEach-Object { Join-Path $env:USERPROFILE $_ } | Where-Object { Test-Path $_ }`

Prints each of the usual conda folders in the home folder that exists, for an
installation that is not in the list of programs.

Keep only a folder that contains `Scripts\conda.exe`: the list of programs can
name a folder that no longer exists.

## Get the Python version of Miniforge

`& "$env:USERPROFILE\miniforge3\python.exe" --version`

Prints the Python version of the main environment of Miniforge, `base`.

## Find environments with Jupylet

`& "{python}" "{code}\jupylet\claude.py" find-env --all {version}`

`{python}` is `python.exe` in a conda folder from "Find conda
installations". Use `{miniforge}` if it exists.

The script asks each conda environment on the computer which Jupylet it has.
It prints one line for each environment with Jupylet: the path, the kind, and
where its Jupylet comes from. `this folder` means it runs the code in
`{code}`. The name of an environment is the last part of its path.

On Windows, there is no check of the conda of a terminal: each conda
installation has its own Prompt in the Start menu, and Miniforge changes none
of them.

## Install Miniforge

`$f = Join-Path $env:TEMP ('miniforge-' + [guid]::NewGuid() + '.exe'); curl.exe -fL -o $f https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe; if ($LASTEXITCODE -eq 0) { Start-Process -Wait -FilePath $f -ArgumentList '/S', '/InstallationType=JustMe', '/RegisterPython=0', '/AddToPath=0', "/D=$env:USERPROFILE\miniforge3" }; Remove-Item $f -ErrorAction SilentlyContinue`

The command does these steps:

1. It downloads the latest Miniforge installer to a temporary file with a
   unique name.
2. If the download succeeds, it runs the installer without windows or
   questions, into `miniforge3` in the home folder.
3. It deletes the installer file.

The installer options:

- `/InstallationType=JustMe`: install for this person only, without an
  administrator password.
- `/RegisterPython=0 /AddToPath=0`: do not change the other Python setups of
  the person.
- `/D=`: the installation folder. It must be the last option, without quotes.

Use `curl.exe`, not `curl`. In PowerShell, `curl` is a different command.

The installer also adds "Miniforge Prompt" to the Start menu.

Check: `& "$env:USERPROFILE\miniforge3\Scripts\conda.exe" --version` prints a
line like `conda 26.x.x`.

## Create the environment

`& "$env:USERPROFILE\miniforge3\Scripts\conda.exe" create -y -p "$env:USERPROFILE\miniforge3\envs\{env}" --override-channels -c conda-forge python=3.13 moderngl glcontext`

Creates the environment `{env}` in Miniforge's `envs` folder, with Python 3.13
and the graphics libraries that Jupylet uses. `-p` with the full path puts the
environment in `envs` even if the conda settings of the person say otherwise.
`--override-channels -c conda-forge` uses only packages from conda-forge.

Check: `& "$env:USERPROFILE\miniforge3\envs\{env}\python.exe" --version`
prints `Python 3.13.x`.

For an existing environment that the person chose in step 5, skip this
command. Its Python is `python.exe` in its folder.

## Install Jupylet, Check Jupylet, Prepare the environment

`{env_python}` is `"$env:USERPROFILE\miniforge3\envs\{env}\python.exe"`, or the
Python of the environment that the person chose.

Install Jupylet:

`& "{env_python}" -m pip install -e "{code}[claude]"`

Installs Jupylet from `{code}`, with the extra tools for Claude. `-e` means
that the environment uses the code in `{code}` directly.

Check Jupylet:

`& "{env_python}" -I -c "import jupylet; print(jupylet.VERSION)"`

Prints the version of the Jupylet that the environment uses. `-I` stops
Python from finding a folder named `jupylet` in the current folder.

Prepare the environment:

`& "{env_python}" -I -m jupylet.claude prepare`

Changes the Jupyter settings of the environment:

- It turns off `jupyter_server_nbmodel`, which can make a cell that runs a
  game or a widget stop responding.
- It keeps the live copy of each notebook while Jupyter runs, so that the
  page shows the cells that Claude adds.
- It turns off the news pop-up of JupyterLab.

## Trust the notebooks, Check the trust

Trust the notebooks:

`Set-Location "{code}\examples"; & "{env_python}" -m jupylet trust_notebooks`

Check the trust:

`Set-Location "{code}\examples"; & "{env_python}" -m jupylet is_trusted`

Run both from inside `examples`. In the folder above it, Python finds the
code folder, also named `jupylet`, instead of the installed Jupylet, and the
command fails.
