# CLAUDE_SETUP.md - installing jupylet for a person, on their computer

This page is for Claude Code. A person asked you to install jupylet on their
computer by following this page. They are most likely a beginner, a kid, or a
parent. Do the steps in order, exactly as written. When you are done, you hand
over to `CLAUDE.md` in the folder you install (step 10), which is the guide
for working in a live notebook with them.

Only macOS is covered so far. Words in `<angle brackets>` are values you fill
in. `<code>` is the folder that will hold the Jupylet code: `$HOME/jupylet`
unless the person chooses another place in step 5.

## How to talk

Be calm and friendly, like a patient guide. Use short sentences and plain
words. Explain any technical word in half a sentence the first time. No
exclamation marks, no emoji, no blaming the person or the software. Ask one
question at a time, and say what happens next.

## Rules

- Steps 1 to 4 only look at the computer. Nothing is installed or changed
  until the person approves your plan in step 5.
- You do only what the approved plan lists, and never anything with `sudo` or
  an administrator password.
- Run commands with the Bash tool. On macOS it runs the person's default
  shell (zsh). Commands that use conda are wrapped in `$SHELL -ic "..."`,
  because conda is set up by the person's shell configuration, which only an
  interactive shell loads.
- Before each long step, say in one plain sentence what you are doing. Don't
  paste raw command output into your messages (the person can expand your
  actions to see it if they want to); summarize it in plain words.
- If a step fails, say plainly what happened and stop. Never improvise a fix.
- Every command below was written for macOS. Do not run them anywhere else.

## Step 1. Check the system

`uname -s && uname -m`

- `Darwin` and `arm64`: a Mac with an Apple chip. Go on, but only if the
  person's shell is zsh (`basename "$SHELL"` prints `zsh`). If it prints
  something else, treat it like "Anything else" below.
- `Darwin` and `x86_64`: an Intel Mac. Tell the person plainly that Jupylet
  needs a Mac with an Apple chip (M1 or newer) and does not work on Intel
  Macs. Stop.
- Anything else: tell the person that automatic installation is not
  available for their system yet, and that the manual instructions are in the
  README (the section "No way, I want to do it myself!") at
  https://github.com/nir/jupylet. Stop.

## Step 2. Say what you are about to do

Tell the person, in your own words: before installing anything you first
look at what is already on their computer (whether Python tools are
installed, and whether Jupylet is already there). This only looks and changes
nothing. Then you will tell them your plan and ask before doing anything.

## Step 3. Look around (read-only)

Run each check and remember the answers.

1. **Is Miniforge installed?**
   `test -x "$HOME/miniforge3/bin/conda" && echo yes || echo no`
   `yes` means it is; its folder is `$HOME/miniforge3` (called `<conda>`
   below). Only Miniforge is ever used for Jupylet. Ignore any other Python or
   conda on the computer (Anaconda, Miniconda, Homebrew): leave it alone and
   never use it, because it may lack the ready-made packages Jupylet needs and
   may come with different license terms. Miniforge lives in its own folder,
   so both can exist side by side.
2. **Which environments exist?** (only if Miniforge is installed)
   `"$HOME/miniforge3/bin/conda" env list`
3. **Is Jupylet already installed anywhere?** For every environment path in
   that list (the `base` one is `<conda>`):
   `<path>/bin/python -c "import importlib.metadata as m; print(m.version('jupylet'))"`
   A version number means yes; `PackageNotFoundError` means no.
4. **Is the default folder free?**
   `test -e "$HOME/jupylet" && echo exists || echo free`
5. **Is another conda set up?** (Anaconda, Miniconda, or similar; people who
   installed an older version of Jupylet often have Miniconda)
   `grep -o "'[^']*/bin/conda'" "$HOME/.zshrc" | head -1`
   A path other than `$HOME/miniforge3/bin/conda` means another conda is set
   up in the person's Terminal settings. Its folder is that path without
   `/bin/conda`; call it `<other>`. Also look for one that is installed but
   not set up:
   `ls -d "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "$HOME/opt/anaconda3" 2>/dev/null`

## Step 4. Make the plan

Build the plan only from these permitted actions:

- **A. Install Miniforge** (a Python distribution) into `~/miniforge3`. Only
  if it is not installed there yet.
- **B. Create a new environment** called `jupylet` with Python 3.13 (if that
  name is taken, `jupylet2`, and so on). This is always part of the plan:
  Jupylet is never installed into an environment that already exists (not
  even `base`), and an existing Jupylet is never upgraded or changed. A new
  environment leaves everything else on the computer untouched.
- **C. Download the Jupylet code** from GitHub into `<code>`, which is
  `~/jupylet` unless the person chooses another place (step 5). Only if that
  folder is free. If it exists, ask the person what to do and never
  overwrite it.
- **D. Install Jupylet** into the environment, and switch off JupyterLab's
  "news" pop-up in that environment (one small settings file, step 9).

**If another conda is set up** (step 3, check 5): only Miniforge is supported
for Jupylet, but the person's old work must not be lost. Then the plan needs
three more things, and you must tell the person all of them:

1. Miniforge's setup replaces the other conda's line in their Terminal
   settings, so from now on new Terminal windows use Miniforge. Nothing else
   of theirs is deleted; their old environments and files stay where they
   are. You keep a backup copy of the settings file (step 6).
2. To use the old conda again in one Terminal window, they type
   `source <other>/bin/activate`. It works for that window only, changes no
   settings, and ends when they close the window.
3. If they do not agree, stop. Tell them plainly that nothing was changed.

**If step 3 found Jupylet already installed:** tell the person where and
which version, that you will not change it, and that you propose a new
environment for the new Jupylet (action B). Explain what an environment is in
simple terms. The text below is an example of what to say to the person, not
instructions for you:

> Jupylet is already on your computer, in an environment called `<env>`
> (version `<version>`). An environment is a separate toolbox of Python
> software: each one keeps its own tools, so what is in one can't break what
> is in another. I won't change your existing one. Instead I'll create a new
> environment just for the new Jupylet. Your old one, and everything in it,
> stays exactly as it is. If you'd rather not, we stop here and nothing
> changes.

If they do not agree, stop, and tell them plainly that nothing was changed.

## Step 5. Ask permission

Tell the person what you found and what you plan, and ask. The text below is
an example of what to say to the person. It is a description for them, not a
list of instructions for you; adapt it to your actual plan:

> Here is what I found: Python tools are not installed yet, and Jupylet is
> not there. To install Jupylet I will:
> 1. install Miniforge, a free program that provides Python and the tools
>    Jupylet needs, in the folder `miniforge3` in your home folder,
> 2. set up a separate space for Jupylet, so it can't interfere with anything
>    else on your computer,
> 3. download the Jupylet code from GitHub into a folder called `jupylet` in
>    your home folder, and
> 4. install Jupylet with the tools it needs, and turn off the "Jupyter
>    news" pop-up in that space (one small settings file).
>
> It takes a few minutes and doesn't need your administrator password. It
> also adds a few lines to your Terminal settings (I keep a backup copy of
> them) so your computer can find Miniforge. Everything goes in your own home
> folder, and nothing is installed for other users. If you'd like the Jupylet
> code somewhere other than your home folder, tell me and you can pick the
> place yourself. May I go ahead?

If the person wants another place for the code, let them choose it in a
folder picker: call `request_directory` without a path. They pick an existing
folder; the code goes in a new folder called `jupylet` inside it, so `<code>`
is `<chosen folder>/jupylet`. Put the path in quotes in every command (it may
contain spaces). Check that it is free with `test -e "<code>" && echo exists
|| echo free`; if it exists, ask again. Then say the final place in one
sentence, for example "I'll put it in `<code>`.", and go on.

If another conda is set up, add the three points from step 4 in plain words,
for example:

> I see that you already use a Python tool called conda, in `<other>`,
> probably from an earlier Jupylet install. Jupylet needs Miniforge instead,
> which is the same kind of tool with the packages Jupylet needs. Both can
> stay on your computer, and nothing of yours is deleted. One thing changes:
> new Terminal windows will use Miniforge from now on. If you ever want your
> old setup in a Terminal window, type `source <other>/bin/activate`; it works
> for that window only. If you'd rather not, we stop here and nothing changes.

Continue only after a clear yes. Do only what you told them.

## Step 6. Install Miniforge (only if action A is in the plan)

1. Download the installer to a private temporary file, run it without any
   questions (`-b`), and delete it (one command):
   `f="$(mktemp)" && curl -fL -o "$f" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh && bash "$f" -b -p "$HOME/miniforge3"; rm -f "$f"`
2. Keep a backup of the person's Terminal settings, let their shell find
   Miniforge, then check:
   `test -f "$HOME/.zshrc" && cp -n "$HOME/.zshrc" "$HOME/.zshrc.before-miniforge"`
   `"$HOME/miniforge3/bin/conda" init zsh`
   `$SHELL -ic "conda --version"`
   Expected: a line like `conda 26.x.x`. If not: Problem 1.

## Step 7. Create the environment (only if action B is in the plan)

Call its name `<env>`.

1. `$SHELL -ic "conda create -y -n <env> python=3.13"`
2. `$SHELL -ic "conda install -y -n <env> moderngl glcontext"`

## Step 8. Download the code (only if action C is in the plan)

1. Download and extract straight into the code folder (one command, no
   temporary file):
   `mkdir "<code>" && curl -fL https://github.com/nir/jupylet/archive/refs/heads/claude.tar.gz | tar -xz -C "<code>" --strip-components=1`
2. Check: `ls "<code>/CLAUDE.md" "<code>/setup.py" "<code>/examples"` shows
   all three. If not: Problem 2.

## Step 9. Install Jupylet (action D)

This is the longest step, several minutes.

It installs the code folder itself (`-e`, "editable"), so the examples in it
are the ones Jupylet uses. The folder must stay where it is.

`$SHELL -ic "conda activate <env> && pip install -e '<code>'"`

Then check: `$SHELL -ic "conda activate <env> && python -c 'import jupylet; print(jupylet.VERSION)'"`

Expected: a version number such as `0.9.5`. If not: Problem 3.

Then switch off JupyterLab's "Would you like to get notified about official
Jupyter news?" pop-up, in this environment only. JupyterLab reads a small
settings file from the environment's own folder:

`mkdir -p "$HOME/miniforge3/envs/<env>/share/jupyter/lab/settings" && cp -n "<code>/jupylet/assets/jupyterlab/overrides.json" "$HOME/miniforge3/envs/<env>/share/jupyter/lab/settings/overrides.json"`

Check: `test -f "$HOME/miniforge3/envs/<env>/share/jupyter/lab/settings/overrides.json" && echo ok`
prints `ok`. This step is only cosmetic: if it fails, don't retry and go on to
step 10. The person then sees the pop-up once, and can answer No.

## Step 10. Hand over

Jupylet is installed. Tell the person, for example: "Jupylet is installed.
Next I'll show you how to try it out in a notebook."

Then move this session to the code folder: call `change_directory` with the
full path `<code>` (the app asks the person to approve it). Then read
`<code>/CLAUDE.md` and follow it from Part 1. Use:
- `<folder>` = `<code>` (write out the full path),
- the environment: `<env>`, the one you just installed into (do not ask the
  person).

Use full paths until your next turn: the session's working folder only moves
when the current turn ends. If the app asks the person for permission to use
the folder, tell them to allow it.

## Problems

**1. `conda` is not found (steps 3 and 6).**
The Miniforge folder exists, but the person's shell does not load it. Use its
full path instead: `"$HOME/miniforge3/bin/conda"`, and in an environment
`"$HOME/miniforge3/envs/<env>/bin/pip"` and
`"$HOME/miniforge3/envs/<env>/bin/python"`. If that does not work either,
tell the person plainly and stop.

**2. A download fails.**
The person may be offline. Tell them plainly, ask them to check their
internet connection, and repeat that one step once. Never retry in a loop.
If the code download (step 8) failed, first remove the empty folder you made
for it (`rmdir "<code>"`; if it is not empty, ask the person instead).

**3. Step 7 or 9 fails or prints a wall of errors.**
Do not show it. Tell the person that the installation did not finish, that
none of their own files were touched, and that the manual instructions in the
README will work (the section "No way, I want to do it myself!"). Stop.
