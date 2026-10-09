---
name: setting-up-jupylet
description: Installs Jupylet on the computer of a person, step by step, from the Jupylet code that is already on disk. Supports macOS with an Apple chip and Windows 11 on Intel or AMD. Use this skill when the person asks to install or set up Jupylet. Also use it when the person says that they downloaded Jupylet and asks what to do next. Also use it when CLAUDE_SETUP.md hands over after the download, or when running-jupylet-notebooks finds that Jupylet is not set up for this folder. Use it together with guiding-jupylet-users.
---

# Setting up Jupylet

This skill installs Jupylet from the Jupylet folder into a Miniforge
environment, checks the result, and ends with the spaceship example running.
Use the skill `guiding-jupylet-users` for how to talk with the person and when
to ask.

## Values

Values in `{braces}` are blanks that you fill in.

- `{code}`: the Jupylet folder. It is the folder that contains
  `.claude/skills/setting-up-jupylet`. Use its full path.
- `{version}`: the line `VERSION = '...'` near the top of
  `{code}/jupylet/__init__.py`.
- `{miniforge}`: the folder `miniforge3` in the home folder of the person.
- `{env}`: the name of the environment that Jupylet goes into.

The commands for each step are in `references/macos.md` and
`references/windows.md`. Read the file for this computer, if one exists,
before step 1. Put every path in double quotes.

## Rules for this skill

- **Install everything for this person only,** in the home folder of the
  person. Never use `sudo` or an administrator password.
- **Use only the Miniforge in `{miniforge}`.** Do not use or change another
  conda installation, for example Miniconda or Anaconda. Do not change an
  environment that the person did not choose for this installation.
- **Do not move or copy `{code}`.** The installation points to this folder,
  so the folder must stay where it is.
- **When an installation step fails, stop.** Tell the person that the
  installation
  did not finish, and that you changed none of their files. Tell the person
  that the README has manual instructions, in the section "No way, I want to
  do it myself!", at https://github.com/nir/jupylet. Do not show the error
  output, unless the person asks for it.

## Progress

Copy this checklist into your notes, and update it after each step:

```
- [ ] 1. Check the system
- [ ] 2. Find what is already on the computer
- [ ] 3. Install Miniforge, if necessary
- [ ] 4. Prepare Terminal (macOS only)
- [ ] 5. Choose the environment
- [ ] 6. Install Jupylet
- [ ] 7. Trust the example notebooks
- [ ] 8. Open the spaceship example
```

## Step 1. Check the system

Run the system check from the reference file.

- **macOS with an Apple chip, or Windows 11 on Intel or AMD:** tell the
  person in one line what system you found, and that Jupylet supports it.
  For example: "You have a Mac with an Apple chip, so Jupylet can run on it."
  Then continue.
- **macOS with an Intel chip:** tell the person that Jupylet needs a Mac with
  an Apple chip (M1 or newer). Then stop.
- **Linux:** tell the person that Jupylet runs on Linux, but that the
  installation with Claude is not available for Linux. Tell the person that
  the README has manual instructions, as in "Rules for this skill". Then
  stop.
- **Any other system:** tell the person in one line what system you found,
  and that Jupylet does not support it. Then stop.

## Step 2. Find what is already on the computer

These checks only read. Say at most one line about them, for example
"Checking what's already on your computer...". Do not tell the person the
results now. The next steps use them.

1. **Conda installations.** Find each Miniforge, Miniconda, Anaconda or
   Mambaforge folder on the computer.
2. **Miniforge in `{miniforge}`.** If it exists, get the Python version of
   its main environment, `base`. Jupylet needs Python 3.11 or newer.
3. **Environments with Jupylet.** If a conda installation exists, run
   `find-env --all {version}` with the Python of a conda installation. Use
   the Python of `{miniforge}` if it exists. The script prints one line for
   each environment that has Jupylet. `this folder` at the end of a line
   means that the environment runs the code in `{code}`.
4. **macOS only: the conda of Terminal.** Find which conda new Terminal
   windows use: none, the one in `{miniforge}`, or another one.

## Step 3. Install Miniforge, if necessary

Do only what step 2 found necessary.

**Miniforge is in `{miniforge}`, with Python 3.11 or newer:** tell the person
in one line, for example "You already have Miniforge, so there's nothing to
install there." Go to step 4.

**Miniforge has a Python older than 3.11:** tell the person that the Miniforge
on the computer is too old for Jupylet, and that they need to update it
first. When Miniforge is up to date, the person can ask you again to install
Jupylet. Then stop. Do not update Miniforge yourself: an update can affect
the other projects of the person.

**Miniforge is not in `{miniforge}`:** ask for a yes before you install it,
for example:

> To run Jupylet, your computer needs Miniforge, a free program that gives
> it Python and the tools Jupylet uses. It goes in your own home folder and
> doesn't need your administrator password. May I install it?

If step 2 found another conda installation, name it, and say that it stays on
the computer, unchanged.

After a clear yes, say one line, for example "Installing Miniforge now. This
may take a minute or two...". Then run "Install Miniforge" from the
reference file. Tell the person in one line, for example "Miniforge is
installed."

## Step 4. Prepare Terminal (macOS only)

On Windows, go to step 5.

To start Jupylet without you, the person types commands in Terminal. These
commands work only when Terminal can find Miniforge. This step changes the
Terminal settings, so that Terminal can find Miniforge.

**Step 2 found that Terminal uses `{miniforge}`:** go to step 5.

**Terminal uses no conda:** ask for a yes, for example:

> So that you can start Jupylet yourself from Terminal later, Terminal needs
> to find Miniforge. May I set that up? It adds a few lines to Terminal's
> settings. I keep a backup of them, so I can put them back if you ask.

**Terminal uses another conda:** name it, and ask for a yes, for example:

> Terminal now uses Miniconda. So that you can start Jupylet yourself from
> Terminal later, Terminal needs to use Miniforge instead. May I set that up?
> Miniconda stays on your computer, unchanged. It adds a few lines to
> Terminal's settings. I keep a backup of them, so I can put them back if you
> ask.

After a clear yes, run "Prepare Terminal" from the reference file, then "Check
the Terminal". Expected: the last line is the full path of `{miniforge}`.
Tell the person in one line, for example "Terminal is set up." If the check
fails, tell the person that Jupylet will work with you, but that Terminal
cannot start it yet. Then continue.

If the person says no, tell the person that it may be more difficult to start
Jupylet from Terminal, and that they can ask you to prepare Terminal at any
time. Then continue.

## Step 5. Choose the environment

**Step 2 found an environment with `this folder`:** tell the person its name,
and ask whether to install Jupylet again into that environment, or into a new
one. If several environments have `this folder`, name all of them, and add
each one as a choice. For example:

> You already have Jupylet set up for this folder, in a Miniforge environment
> called `jupylet`. Shall I install it again into that environment, which
> also brings it up to date with the code in the folder, or set up a new
> environment?

- **The same environment:** that environment is `{env}`. Go to step 6.
- **A new environment:** continue below.

**A new environment:** name it `jupylet`. If an environment with that name
already exists, try `jupylet2`, then `jupylet3`, and so on. Use the first name
that no environment in `{miniforge}/envs` has. That name is `{env}`.

If step 2 found Jupylet in other environments, name all of them, and say that
you leave them as they are. Then ask for a yes. For example:

> You already have Jupylet in two other environments, `jp145` and `jp14`.
> I'll leave them as they are. Next I'll set up a new Miniforge environment
> for Jupylet, called `{env}`. It's a separate space with its own copy of
> Python, so what you install there can't break anything else. Shall I go
> ahead?

After a clear yes, say one line, for example "Setting up the environment
now. This may take a minute...". Then run "Create the environment" from the
reference file. Tell the person in one line, for example "The environment
`{env}` is ready."

## Step 6. Install Jupylet

Before you start, tell the person what the Jupylet folder is, for example:

> I'll install Jupylet from `{code}`. That makes it your Jupylet folder: it
> holds the code Jupylet runs, and the example notebooks. Please keep it where
> it is.

Then say how long it takes, and explain Jupyter while the person waits, for
example:

> Installing Jupylet into the environment `{env}` now. This is the longest
> part, and may take a few minutes. It also installs Jupyter, the program
> where you'll write and run your code.

Run these from the reference file, in this order:

1. "Install Jupylet". It installs Jupylet from `{code}`, with the tools that
   you use to work in a notebook with the person (`[claude]`).
2. "Check Jupylet". Expected: `{version}`.
3. "Prepare the environment". Expected: `turned off` or `off`. It changes
   the Jupyter settings that Jupylet needs to run in a notebook with you:
   - It turns off `jupyter_server_nbmodel`, which can make a cell that runs
     a game or a widget stop responding.
   - It makes the page show the cells that you add to a notebook.

   If it prints anything else, the step failed. Find the cause. Tell the
   person in one sentence what happened, and how you want to solve the
   problem. Ask the person before each attempt. If you cannot solve it,
   stop, as in "Rules for this skill".

Then tell the person in one line, for example "Jupylet is installed."

## Step 7. Trust the example notebooks

First run "Check the trust" from the reference file. If each line says
`trusted`, go to step 8.

Otherwise, ask for a yes before you trust the example notebooks, for example:

> Jupyter treats notebooks you didn't create yourself as untrusted. To show
> the interactive parts of the example notebooks, including graphics and
> animations, I need to mark them as trusted first. May I do that?

After a clear yes, run "Trust the notebooks" from the reference file, then
"Check the trust". Expected: each line says `trusted`.

If the person says no, tell the person that the graphics and animations will
not show in the examples, and that they can ask you to trust the notebooks at
any time. Then continue.

## Step 8. Open the spaceship example

The installation is complete only when an example runs. Ask, for example:

> Now let's check that everything works, with the spaceship example: a small
> ship you steer with the arrow keys. It opens in a notebook right next to
> our chat. May I open it?

After a clear yes, use the skill `running-jupylet-notebooks` to open and run
`11-spaceship.ipynb` in `{code}/examples`, with the environment `{env}`.

If the person says no, tell the person that they can ask for it at any time.
