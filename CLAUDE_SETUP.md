# CLAUDE_SETUP.md - installing jupylet for a person, on their computer

This page is for Claude Code. A person asked you to install jupylet on their
computer by following this page. They are most likely a beginner, a kid, or a
parent. Do the steps in order, exactly as written. When you are done, you hand
over to `CLAUDE.md` in the folder you install (step 10), which is the guide
for working in a live notebook with them.

It covers a Mac with an Apple chip, and Windows 11 on an Intel or AMD
processor. Words in `<angle brackets>` are values you fill in; "Paths and
commands" below lists them.

**Which code.** Install the branch named in the link the person gave you to
this page (for example `claude` in
`https://github.com/nir/jupylet/blob/claude/CLAUDE_SETUP.md`), or the branch
they name in the chat. If neither names one, use `master`. Call it
`<branch>`.

## How to talk to the person

The person is a beginner: a kid, a parent, someone new to programming. Not
simple, just new to this. Talk to them that way from your first message to
your last, whoever they seem to be. Even if they sound technical, or say they
work on Jupylet, this page is the beginner's path: someone testing it wants
to see exactly what a beginner would see.

- Be calm and friendly, like a patient guide. Use short sentences and plain
  words. Explain any technical word in half a sentence the first time: the
  first time you name Miniforge, say what it is and why Jupylet needs it
  ("Miniforge, the free program that gives your computer Python and the
  tools Jupylet uses"); the first time you say environment, what that is. No
  exclamation marks, no emoji, no blaming the person or the software.
- Not technical, but not simple either: call things by their real names,
  and explain a name once, the first time, rather than swapping it for a
  vague friendly word. Say "a Miniforge environment called `jp145`", not "a
  few places"; "Miniforge's main environment, called `base`", not "the main
  toolbox". They will meet these names again, in the Miniforge Prompt or
  Terminal and in `conda activate`, and a real name they understand is worth
  more than a cosy one they cannot connect to anything.
- What you find matters to them: that something is already installed is news
  worth one line, not a silent check.
- Ask one question per message, and say what happens next.
- Never show them a raw error or a wall of output. Say in one plain sentence
  what happened and what you will do. (They can expand your actions to see
  the details if they want to.)
- Steps, their numbers and this page are for you, not the person. Never say
  "step 3" or mention this page, and never name a technical detail they have
  no use for (a command, a registry, a version check, a path they did not
  choose). A check that succeeds needs no comment: go straight on. Speak up
  only for something they must decide, something that failed, or, if a step
  is slow, one short line so silence does not look like a hang.
- Never narrate your own process or state: not "following the procedure",
  not which part you are on, not notes about testing. Something meant only
  for a developer reading along does not belong in the conversation at all.
- **Exception, for developers only:** if the person asks you for verbose
  boxes, show each command and its raw output in a plain code block, before
  your plain sentence, for the rest of the session. Only once they ask; never
  guess it, and never let the box replace the sentence.

## Rules

- Looking costs nothing: read and check freely, without asking. Changing
  something does not: ask at the moment you are about to change it, in one
  short message, and wait for a clear yes. Never ask for everything at once
  up front.
- If they say no, stop, and tell them plainly what was already done (if
  anything) and that nothing else will change.
- Do only what this page says.
- Everything is installed for this person only, in their own home folder (or
  a folder they pick). Never for all users of the computer, never with `sudo`
  or an administrator password.
- If a step fails, say plainly what happened and stop. Never improvise a fix.

## Paths and commands

Each command is written once. Where macOS and Windows 11 need different
commands, both are given, labelled; run only the one for this computer.

- **macOS:** run commands with the Bash tool (the person's shell, zsh).
- **Windows 11:** run commands with the PowerShell tool, never the Bash tool,
  even if it exists. A command that starts with a quoted program path needs
  `&` in front: `& "<python>" -m pip ...`.

| | macOS | Windows 11 |
|---|---|---|
| `<home>` (the person's home folder) | `$HOME` | `$env:USERPROFILE` |
| the conda of a conda folder `<x>` | `<x>/bin/conda` | `<x>\Scripts\conda.exe` |
| the python of an environment folder `<x>` | `<x>/bin/python` | `<x>\python.exe` |
| path separator | `/` | `\` |

`<miniforge>` is the Miniforge folder, always `<home>/miniforge3`. `<conda>`
is its conda, `<base python>` its python, `<env>` the new environment's name,
and `<env python>` the python of `<miniforge>/envs/<env>`. `<code>` is the
new folder for the Jupylet code: the one step 3 proposes, or another place
the person picks in step 6. Put every path in double quotes.

Once Miniforge is installed, most work is done with its own Python, which
behaves the same on both systems, and a small helper script (at the end of
this page). Save it as `setup_helper.py` in your scratchpad folder when step 6
first needs it; `<helper>` is its full path.

## Step 1. Check the system

**macOS:** `uname -s && uname -m && basename "$SHELL"`

- `Darwin`, `arm64`, `zsh`: a Mac with an Apple chip. Go on.
- `Darwin` and `x86_64`: an Intel Mac. Tell the person plainly that Jupylet
  needs a Mac with an Apple chip (M1 or newer) and does not work on Intel
  Macs. Stop.
- `Darwin`, `arm64`, but a shell other than `zsh`: treat it like "Anything
  else" below.

**Windows 11:** `$env:PROCESSOR_ARCHITECTURE; [Environment]::OSVersion.Version.Build`

- `AMD64` and a number of `22000` or more: Windows 11 on an Intel or AMD
  processor. Go on.
- `ARM64`, or a number below `22000` (Windows 10): treat it like "Anything
  else" below.

**Anything else:** tell the person that automatic installation is not
available for their system yet, and that the manual instructions are in the
README (the section "No way, I want to do it myself!") at
https://github.com/nir/jupylet. Stop.

A supported system needs no comment.

## Step 2. Say hello

One short message, no question, then go straight on to step 3, for example:

> Hi. I'll set up Jupylet for you, one small step at a time, and I'll ask
> before I change anything on your computer. First, a quick look at what's
> already there.

## Step 3. Look around (read-only)

Run each check and remember the answers. Say nothing about the checks
themselves: what they found comes up in the next steps, where it matters.

1. **Which conda folders are there?** Conda is the tool that Miniforge,
   Miniconda and Anaconda all use to manage Python. Each one lives in a
   folder of its own.

   **macOS:**
   `ls -d "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/opt/miniconda3" "$HOME/opt/anaconda3" /opt/homebrew/Caskroom/miniforge/base /opt/homebrew/Caskroom/miniconda/base /opt/miniconda3 /opt/anaconda3 2>/dev/null`

   **Windows 11** (installed programs, then the usual folders):
   `Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*','HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*' -ErrorAction SilentlyContinue | Where-Object { $_.DisplayName -match 'Anaconda|Miniconda|Miniforge|Mambaforge' } | ForEach-Object { Split-Path ($_.UninstallString -replace '"', '') } | Where-Object { Test-Path $_ } | Sort-Object -Unique`
   `'miniforge3', 'mambaforge', 'miniconda3', 'anaconda3' | ForEach-Object { Join-Path $env:USERPROFILE $_ } | Where-Object { Test-Path $_ }`

   Keep only folders that have a conda (see the table). Installed-programs
   entries can be stale, which is why only folders that exist count.

2. **Is Miniforge already there, and new enough?** Only `<home>/miniforge3`
   counts: it is the one place Jupylet uses, and there is only ever one
   Miniforge, there. A Miniforge or Mambaforge in any other folder is not
   used; treat it like any other conda. If check 1 found
   `<home>/miniforge3`, check the Python version of its `base`:
   `"<python of <home>/miniforge3>" --version`
   Python 3.11 or newer (Miniforge 24.5.0, from July 2024, or later; earlier
   releases came with Python 3.10) is new enough. Jupylet needs Python 3.11
   or newer, and so does `base`: a beginner who types Python commands outside
   the Jupylet environment ends up in `base`, and an older Python there would
   fail in confusing ways.

3. **Is Jupylet already installed anywhere?** Collect environment folders
   from every conda folder: `"<its conda>" env list` (each path in the list,
   including the conda folder itself, which is its `base`). Add the lines of
   `<home>/.conda/environments.txt`, if that file exists (every conda writes
   its environments there). For each environment folder:
   `"<its python>" -I -c "import importlib.metadata as m; print(m.version('jupylet'))"`
   A version number means yes; an error means no.

4. **Which folder to propose for the code?** The first of
   `<home>/jupylet`, `<home>/jupylet2`, ... `<home>/jupylet9` that does not
   exist yet. It prints that folder; this is `<code>`, for now.
   **macOS:** `for n in "" 2 3 4 5 6 7 8 9; do test -e "$HOME/jupylet$n" || { echo "$HOME/jupylet$n"; break; }; done`
   **Windows 11:** `@('') + (2..9) | ForEach-Object { Join-Path $env:USERPROFILE "jupylet$_" } | Where-Object { -not (Test-Path $_) } | Select-Object -First 1`
   No output: all ten are taken, so the person picks a place in step 6.

5. **macOS only: which conda do new Terminal windows use?**
   `grep -h -o "'[^']*/bin/conda'" "$HOME/.zshrc" "$HOME/.zprofile" 2>/dev/null`
   `grep -h -E '^[^#]*export PATH=.*(conda|forge)' "$HOME/.zshrc" "$HOME/.zprofile" 2>/dev/null`
   No output: none. The conda of `<miniforge>`: Terminal is already set up.
   Anything else: another conda is set up in the person's Terminal settings;
   its folder is the path without `/bin/conda` (or the folder named in the
   `PATH` line, without `/bin`). Call it `<other>`. People who installed an
   older version of Jupylet often have Miniconda there.

   On Windows there is nothing to check here: every conda gets its own
   Prompt in the Start menu, and installing Miniforge changes none of them.

## Step 4. Miniforge

Do only what step 3 calls for. If `<home>/miniforge3` is there, new enough,
and (macOS) Terminal is already set up for it, tell the person in one line,
for example "You already have Miniforge, the free program that gives your
computer Python and the tools Jupylet uses, so there's nothing to install
there.", and go on to step 5.

**If Miniforge is not installed,** ask, for example:

> To run Jupylet, your computer needs Miniforge, a free program that
> provides Python and the tools Jupylet uses. May I install it? It goes in
> your own home folder and doesn't need your administrator password.

On macOS, add one sentence about Terminal, for example "It also adds a few
lines to your Terminal settings, so Terminal can find it. I keep a backup of
them." If another conda is set up in Terminal, ask about that instead (see
"Terminal on macOS" below), after this.

After a clear yes, say one line, for example "Installing Miniforge now, this
takes a minute or two...", then download the installer to a private
temporary file, run it without any questions or windows, for this person
only, and delete it (one command):

**macOS:**
`f="$(mktemp)" && curl -fL -o "$f" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh && bash "$f" -b -p "$HOME/miniforge3"; rm -f "$f"`

**Windows 11:**
`$f = Join-Path $env:TEMP ('miniforge-' + [guid]::NewGuid() + '.exe'); curl.exe -fL -o $f https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe; if ($LASTEXITCODE -eq 0) { Start-Process -Wait -FilePath $f -ArgumentList '/S', '/InstallationType=JustMe', '/RegisterPython=0', '/AddToPath=0', "/D=$env:USERPROFILE\miniforge3" }; Remove-Item $f -ErrorAction SilentlyContinue`

(`curl.exe`, not `curl`: in PowerShell `curl` is a different command.
`/InstallationType=JustMe` installs for this person only, without an
administrator password; `/RegisterPython=0 /AddToPath=0` leave their other
Python setups alone. `/D=` must be last and is never quoted.)

Check: `"<conda>" --version` prints a line like `conda 26.x.x`. If not:
Problem 1.

**If Miniforge is too old** (step 3, check 2): on Windows, automatic updating
is not available yet: Problem 5. On macOS, ask, for example:

> You already have Miniforge, but it's too old for Jupylet. May I update it?
> Your projects, your files and your Miniforge environments stay as they
> are. Only Miniforge's main environment, called `base`, gets new tools, so
> anything you installed into `base` yourself may need installing again.

If step 3 found Jupylet in `base`, add that this old Jupylet will stop
working, and that the new one replaces it. After a clear yes, say one line,
for example "Updating Miniforge now, this takes a few minutes...", then run
the same installer in its update mode (`-u`):
`f="$(mktemp)" && curl -fL -o "$f" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh && bash "$f" -b -u -p "$HOME/miniforge3"; rm -f "$f"`
Check: `"<base python>" --version` now prints Python 3.11 or newer. If not:
Problem 3.

**Terminal on macOS.** If step 3, check 5 did not find `<miniforge>` set up,
Terminal needs setting up so it finds Miniforge (new windows then show
`(base)`). If you just installed Miniforge with the person's yes, and no other
conda is set up, that yes covers it. Otherwise ask first: without another
conda, for example "May I set up Terminal so it finds Miniforge? It adds a
few lines to its settings, and I keep a backup of them." With another conda
(`<other>`), for example:

> You already have another Python setup, `<name of <other>>` (in
> `<other>`), probably from an earlier Jupylet. Jupylet needs Miniforge
> instead. Both can stay on your computer and nothing of yours is deleted,
> but new Terminal windows will use Miniforge from now on. You can still use
> `<name of <other>>` in a window by typing `source <other>/bin/activate`. Is
> that OK?

`<name of <other>>` is what the folder says it is, for example Miniconda for
`miniconda3`, Anaconda for `anaconda3`.

After a clear yes, keep a backup of the person's Terminal settings, then let
their shell find Miniforge:
`test -f "$HOME/.zshrc" && cp -n "$HOME/.zshrc" "$HOME/.zshrc.before-miniforge"`
`"<conda>" init zsh`
Step 9 checks that it worked.

On Windows, the installer adds "Miniforge Prompt" to the Start menu, and
changes nothing else; another conda keeps its own Prompt.

## Step 5. A new Miniforge environment for Jupylet

Jupylet always goes into a new environment, never one that already exists
(not even `base`), and an existing Jupylet is never upgraded or changed.
First choose the name: `jupylet`, or if any environment step 3 found already
has that name, `jupylet2`, and so on. Call it `<env>`. Then ask, for example:

> Next I'll install Jupylet in a new Miniforge environment, called `<env>`.
> A Miniforge environment is a separate space with its own copy of Python
> and its own tools, so what you install in one can't mix with or break
> anything in another. Shall I go ahead?

If step 3 found Jupylet already installed, start with one sentence that says
where, by its real name, for example "You already have an older Jupylet, in a
Miniforge environment called `jp145`. I'll leave it as it is." Other cases:
"in Miniforge's main environment, called `base`"; "in a Miniconda
environment called `<name>`"; for several, name one or two, "in Miniforge
environments called `jp145`, `jp144` and a few others".

If they ask why not in Miniforge's main environment, `base`, as the Jupylet
website says: its own environment keeps Jupylet and everything else on the
computer from getting in each other's way, and the only difference for them
is typing `conda activate <env>` first when they use Jupylet on their own.

After a clear yes, say one line, for example "Setting up the environment now,
this takes a minute...", then:

`"<conda>" create -y -p "<miniforge>/envs/<env>" --override-channels -c conda-forge python=3.13 moderngl glcontext`

(`-p` with the full path, not `-n`, so the environment ends up in
`<miniforge>/envs` even if the person's conda settings say otherwise;
`--override-channels -c conda-forge` so only conda-forge's packages are used,
whatever those settings say. It can still be activated by its name.)

Check: `"<env python>" --version` prints `Python 3.13.x`. If not: Problem 3.

## Step 6. Download the code

Say exactly where the code comes from and where it goes, and ask, for
example:

> Now I'll download the Jupylet code from GitHub, the website where it is
> kept: the `<branch>` branch (one version of the code) of
> https://github.com/nir/jupylet. I'll put it in a new folder, `<code>`. Is
> that OK, or would you like to pick another place for it?

If you were told to use a local archive instead of GitHub (to test this
page), say that instead, with the archive's full path, for example "I'll
unpack the Jupylet code from the archive `<full path of the archive>` into a
new folder, `<code>`." Its `file://` URL is then `<source>` below; otherwise
`<source>` is `<branch>`.

(If step 3 found no free name, say instead that the usual names are taken,
and go straight to the folder picker.) If they want another place, let them
choose it in a folder picker: call `request_directory` without a path. They
pick an existing folder; the code goes in a new folder inside it, the first
of `jupylet`, `jupylet2`, ... that does not exist there yet (check as in step
3, check 4, with the chosen folder instead of the home folder). That is
`<code>`. Say the final place in one sentence, for example "I'll put it in
`<code>`.", and go on. Continue only after a clear answer.

Then save the helper script (end of this page) as `setup_helper.py` in your
scratchpad folder, and download the code and extract it straight into the
code folder:

`"<base python>" "<helper>" download <source> "<code>"`

Expected: `ok`. It refuses a folder that exists and is not empty. If it
prints anything else: Problem 2.

## Step 7. Install Jupylet

The person already said yes in step 5. This is the longest part, several
minutes: say so in one line first, for example "Installing Jupylet now. This
is the longest part, a few minutes..."

It installs the code folder itself (`-e`, "editable"), so the examples in it
are the ones Jupylet uses. The folder must stay where it is.

`"<env python>" -m pip install -e "<code>"`

Then check: `"<env python>" -I -c "import jupylet; print(jupylet.VERSION)"`

Expected: a version number such as `0.9.5`. If not: Problem 3.

Then, without a word to the person, switch off JupyterLab's "Would you like
to get notified about official Jupyter news?" pop-up, in this environment
only. Always do this: `setup.py` lists the settings file under `data_files`,
but an editable install (`pip install -e`) never copies it. This copies it
into the environment's own folder, where JupyterLab reads it:

`"<env python>" "<helper>" overrides "<code>"`

Expected: `ok`. This is only cosmetic: if it fails, don't retry and go on.
The person then sees the pop-up once, and can answer No.

## Step 8. Trust the example notebooks

A downloaded notebook is untrusted until someone says it is safe to run its
interactive parts: a normal Jupyter safety check, not something specific to
Jupylet. Ask, for example:

> One last thing: Jupyter treats notebooks you didn't make yourself as
> untrusted, as a safety check, and then the game shows as text instead of a
> picture. May I mark the example notebooks as trusted?

After a clear yes:
`"<env python>" "<helper>" jupylet "<code>" trust_notebooks`

Check: `"<env python>" "<helper>" jupylet "<code>" is_trusted` prints
`trusted` for every file.

If they decline, tell them plainly that the example notebooks will not show
their pictures until they trust them (in JupyterLab: File > Trust Notebook),
and go on anyway.

## Step 9. Check the person's own way in

Without a word to the person, run what they will use later, the way they
will use it: a new Terminal window on macOS, the Miniforge Prompt on Windows.

`"<base python>" "<helper>" prompt "<miniforge>"`

Expected: `ok`. `mismatch <folder>` means their Terminal or Prompt uses
another conda; `no prompt` (Windows) means the Start menu has no Miniforge
Prompt. Either way: Problem 4.

## Step 10. Hand over

Tell the person it is done, with the one thing they need when using Jupylet
on their own, for example:

> Jupylet is installed. When you want to use it on your own later, open
> [macOS: a new Terminal window] [Windows: the Miniforge Prompt from your
> Start menu] and first type `conda activate <env>`, which switches to
> Jupylet's environment. Now let's try it out in a notebook.

Use only the bracketed part for this computer, without the brackets.

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

**1. Miniforge did not install (step 4).**
There is no conda in `<home>/miniforge3`. On Windows, security software may
have blocked the installer. Tell the person plainly that installing Miniforge
did not work and that nothing else was changed, and stop.

**2. A download fails.**
The person may be offline. Tell them plainly, ask them to check their
internet connection, and repeat that one step once. Never retry in a loop.
If the code download (step 6) left a folder behind, ask the person before
removing it.

**3. Updating Miniforge (step 4), creating the environment (step 5) or
installing Jupylet (step 7) fails or prints a wall of errors.**
Do not show it. Tell the person that the installation did not finish, that
none of their own files were touched, and that the manual instructions in the
README will work (the section "No way, I want to do it myself!"). Stop. If it
was the update of their Miniforge, say instead that updating Miniforge
did not finish, and do not promise that it is unchanged: you have not
checked that.

**4. The person's Terminal or Prompt does not find Miniforge (step 9).**
Jupylet is installed and works; this only affects using it on their own, not
the next part with you. Tell them plainly, go on with step 10, and write what
the helper printed into `EXPERIENCE.md` in `<code>`.

**5. Miniforge is too old, on Windows (step 4).**
Tell the person plainly that the Miniforge on their computer is too old for
Jupylet, and that updating it automatically is not available on Windows yet.
Nothing was changed. Stop.

## The helper script

Save this as `setup_helper.py` in your scratchpad folder. It only uses
Python's standard library, so it runs with any Miniforge Python.

```python
import glob
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

ARCHIVE = 'https://github.com/nir/jupylet/archive/refs/heads/%s.tar.gz'


def download(source, code):
    """Extract the code of a branch into an empty or new folder. For a test,
    source can also be the full URL of a .tar.gz (file:// included)."""
    code = os.path.abspath(code)

    if os.path.exists(code) and os.listdir(code):
        return 'not empty: ' + code

    fd, tmp = tempfile.mkstemp(suffix='.tar.gz')
    os.close(fd)

    try:
        url = source if '://' in source else ARCHIVE % source
        urllib.request.urlretrieve(url, tmp)
        os.makedirs(code, exist_ok=True)
        root = os.path.realpath(code)

        with tarfile.open(tmp) as tar:
            for member in tar.getmembers():
                # Every path starts with one top folder, like jupylet-<branch>/.
                rel = member.name.split('/', 1)[1] if '/' in member.name else ''

                if not rel:
                    continue

                target = os.path.realpath(os.path.join(code, rel))

                if not target.startswith(root + os.sep):
                    return 'unsafe path in the archive: ' + member.name

                if member.isdir():
                    os.makedirs(target, exist_ok=True)

                elif member.isfile():
                    os.makedirs(os.path.dirname(target), exist_ok=True)

                    with tar.extractfile(member) as src, open(target, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
    finally:
        os.remove(tmp)

    missing = [p for p in ('CLAUDE.md', 'setup.py', 'examples')
               if not os.path.exists(os.path.join(code, p))]

    return 'missing: ' + ', '.join(missing) if missing else 'ok'


def overrides(code):
    """Copy the settings file that turns off the news pop-up (run with the
    environment's python: sys.prefix is then the environment's folder)."""
    src = os.path.join(code, 'jupylet', 'assets', 'jupyterlab', 'overrides.json')
    folder = os.path.join(sys.prefix, 'share', 'jupyter', 'lab', 'settings')
    os.makedirs(folder, exist_ok=True)

    if not os.path.exists(os.path.join(folder, 'overrides.json')):
        shutil.copy(src, folder)

    return 'ok'


def jupylet(code, command):
    """Run python -m jupylet <command> inside the examples folder. Never
    from the folder above: a folder named jupylet there would be imported
    instead of the installed package."""
    out = subprocess.run([sys.executable, '-m', 'jupylet', command],
                         cwd=os.path.join(code, 'examples'))
    return None if out.returncode == 0 else 'failed'


def prompt(miniforge):
    """Ask the person's own Terminal or Prompt which conda it uses."""
    if sys.platform == 'win32':
        menu = os.path.join(os.environ['APPDATA'], 'Microsoft', 'Windows',
                            'Start Menu', 'Programs')

        if not glob.glob(os.path.join(menu, '**', 'Miniforge Prompt.lnk'), recursive=True):
            return 'no prompt'

        bat = os.path.join(miniforge, 'Scripts', 'activate.bat')
        cmd = 'cmd /c call "%s" "%s" && conda info --base' % (bat, miniforge)
        out = subprocess.run(cmd, capture_output=True, text=True)
    else:
        out = subprocess.run([os.environ['SHELL'], '-ic', 'conda info --base'],
                             capture_output=True, text=True)

    lines = out.stdout.strip().splitlines()
    base = lines[-1].strip() if lines else ''

    if base and os.path.normcase(os.path.realpath(base)) == \
            os.path.normcase(os.path.realpath(miniforge)):
        return 'ok'

    return 'mismatch ' + (base or out.stderr.strip())


COMMANDS = {'download': download, 'overrides': overrides,
            'jupylet': jupylet, 'prompt': prompt}

if __name__ == '__main__':
    result = COMMANDS[sys.argv[1]](*sys.argv[2:])

    if result:
        print(result)
```
