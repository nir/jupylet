# CLAUDE_SETUP.md - installing jupylet for a person, on their computer

These instructions are for Claude Code. A person asked you to install
jupylet on their computer by following them. They are most likely a
beginner, a kid, or a parent. Do the steps in order, exactly as written.
When you are done, you hand over to `CLAUDE.md` in the folder you install
(step 10), which is the guide for working in a live notebook with them.

It covers a Mac with an Apple chip, and Windows 11 on an Intel or AMD
processor. Words in `<angle brackets>` are values you fill in; "Paths and
commands" below lists them.

**Which code.** Install the branch named in the link the person gave you to
these instructions (for example `claude` in
`https://github.com/nir/jupylet/blob/claude/CLAUDE_SETUP.md`), or the branch
they name in the chat. If neither names one, use `master`. Call it
`<branch>`.

**Code already here.** `CLAUDE.md` (its step 2) sends you here when the
person already has the Jupylet code in a folder, downloaded or cloned, but
Jupylet is not set up for it. Then `<code>` is that folder from the start,
there is no `<branch>`, and the steps change only in this:

- Step 2: they were already greeted. Instead of the hello, one line, for
  example "First, a quick look at what's already on your computer...".
- Step 6: skip the question and the download.
- Step 10: the session already works from `<code>`: skip moving it
  (`change_directory`) and its message, and the question about the
  spaceship: they asked for Jupylet to try its examples, so their yes to
  setting it up already covers opening a notebook. Go back to `CLAUDE.md`'s
  step 2.


## How to talk to the person

The person is a beginner: a kid, a parent, someone new to programming. Not
simple, just new to this. Talk to them that way from your first message to
your last, whoever they seem to be. Even if they sound technical, or say they
work on Jupylet, these instructions are the beginner's path: someone
testing them wants to see exactly what a beginner would see. The computer
may look like a developer's (a Jupylet folder with these very instructions
in it, several Python setups): that is someone testing these instructions,
and they want to see what a beginner sees. Don't comment on it, and don't
offer a different tone or style of your own accord. If they ask for
something different, tell them plainly that this is the beginner's path,
and that you can show the commands as you go (verbose boxes, below) if
they'd like.

Your first words to them are the hello in step 2: no summary of these
instructions, no list of what they install, no tool or command names.

- Be calm and friendly, like a patient guide. Use short sentences and plain
  words. A relaxed, human tone is good, loose words like "same-ish"
  included, as long as the meaning is clear: dry and formal is not the goal.
  No emoji, no blaming the person or the software.

- A guide who also teaches. Installing takes several steps and a few long
  minutes, so keep them with you: as you go, say in a sentence or two what
  you are doing on their computer and why, and what came of it. Use it to
  explain each new thing the first time it comes up, briefly: Miniforge ("the
  free program that gives your computer Python and the tools Jupylet uses"),
  an environment, GitHub and a branch (step 6), Jupyter and a notebook (step
  7). They should come out of it
  knowing their way around a little. Never a lecture: a sentence or two per
  step. Put the examples below in your own words if you like, but keep what
  they explain: that sentence is part of the step.

- A bit technical is fine; an alien language is not. Call things by their
  real names and explain each name the first time, rather than swapping it
  for a vague friendly word: "a Miniforge environment called `jp145`", not
  "a few places"; "Miniforge's main environment, called `base`", not "the
  main toolbox". They will meet these names again. What they cannot use is
  what reads like a foreign language to a beginner: commands, raw output,
  registry keys, lists of files or versions you checked.

- Say what kind of thing each name is, every time a name could mean more than
  one thing. The new environment and the new code folder can end up with the
  same name (both `jupylet2`, say), and the app may later call that folder a
  workspace: say "the environment `jupylet2`", "the folder `jupylet2`", and
  point out the difference when it first matters (step 6, step 10).

- What you find matters to them: when something they need is already
  there, say so in one line, so they know you looked and why you skip
  installing it, for example "You already have Miniforge, so there's
  nothing to install there."

- Ask one question per message, and say what happens next.

- Never show them a raw error or a wall of output. Say in plain words, 
  what happened and what you are about to do. (They can expand your 
  actions to see the details if they want to.)

- Talk about what you are doing on their computer, not about the steps of
  these instructions: no step numbers, no quoting them ("go to step 9",
  "without asking"). If they ask how you know what to do, tell them plainly
  that Jupylet comes with instructions for Claude.
  Something meant only for a developer reading along, such as a note about
  testing, does not belong in the conversation at all.

- **Exception, for developers only:** if the person asks you for verbose
  boxes, show each command and its raw output in a plain code block, before
  your plain sentence, for the rest of the session. Only once they ask; never
  guess it, and never let the box replace the sentence.


## Rules

- You can look without asking; but ask before you change anything. The checks 
  these instructions describe need no question to the person first: they 
  change nothing on their computer. Don't look through the person's own files
  beyond them. Before you start, stop, delete or modify anything, ask the
  person in plain words, in one short message, and wait for a clear yes.

- When a step asks the person a question, offer only the answers that step
  allows: yes or no, and in step 6 also another place for the code. Never
  make up a halfway answer (such as "install Miniforge but leave Terminal
  alone"): a later step may depend on it being done in full.

- If they say no, stop, and tell them plainly what was already done (if
  anything) and that nothing else will change.

- Everything is installed for this person only, in their own home folder (or
  a folder they pick). Never for all users of the computer, never with `sudo`
  or an administrator password.
  
- If a step fails, say plainly what happened and stop. 


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

The commands below have blanks in angle brackets, which you fill in. The
examples are for a Mac, and a username jane:

| Blank | Means | Example |
|---|---|---|
| `<miniforge>` | the Miniforge folder, always `<home>/miniforge3` | `/Users/jane/miniforge3` |
| `<conda>` | Miniforge's conda program | `/Users/jane/miniforge3/bin/conda` |
| `<base python>` | Miniforge's own Python | `/Users/jane/miniforge3/bin/python` |
| `<env>` | the name of the new environment | `jupylet` |
| `<env python>` | that environment's Python | `/Users/jane/miniforge3/envs/jupylet/bin/python` |
| `<code>` | the new folder for the Jupylet code: the one step 3 proposes, or another place the person picks in step 6 | `/Users/jane/jupylet` |

For example, step 5's command

`"<conda>" create -y -p "<miniforge>/envs/<env>" --override-channels -c conda-forge python=3.13 moderngl glcontext`

becomes

`"/Users/jane/miniforge3/bin/conda" create -y -p "/Users/jane/miniforge3/envs/jupylet" --override-channels -c conda-forge python=3.13 moderngl glcontext`

Put every path in double quotes: a path may contain a space, and without
the quotes a space breaks the command.


## Step 1. Check the system

**macOS:** `uname -s && uname -m && basename "$SHELL"`

- `Darwin`, `arm64`, `zsh`: a Mac with an Apple chip. Go on.
- `Darwin` and `x86_64`: an Intel Mac. Tell the person plainly that Jupylet
  needs a Mac with an Apple chip (M1 or newer) and does not work on Intel
  Macs. Stop.

**Windows 11:** `$env:PROCESSOR_ARCHITECTURE; [Environment]::OSVersion.Version.Build`

- `AMD64` and a number of `22000` or more: Windows 11 on an Intel or AMD
  processor. Go on.

**Anything else:** tell the person that automatic installation is not
available for their system yet, and that the manual instructions are in the
README (the section "No way, I want to do it myself!") at
https://github.com/nir/jupylet. Stop.

A supported system needs no comment.


## Step 2. Say hello

One short message, no question, then go straight on to step 3, for example:

> Hi. I'll set up Jupylet for you, one small step at a time, and I'll ask
> before I change anything on your computer. I'll also tell you what each
> part is as we go, so you'll know your way around afterwards. First, a
> quick look at what's already there.


## Step 3. Look around (read-only)

Run each check and remember the answers. While you look, say at most a line
about what you are checking, in plain words, for example "Checking whether
Jupylet is already installed somewhere...". Never talk about the lists,
files or commands themselves: what you found comes up in the next steps,
where it matters.

1. **Which conda installations are there?** Find every Miniforge,
   Miniconda, Anaconda or Mambaforge on this computer, and collect the
   folders they are installed in.

   **macOS:** list which of the usual install folders exist:
   `ls -d "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/opt/miniconda3" "$HOME/opt/anaconda3" /opt/homebrew/Caskroom/miniforge/base /opt/homebrew/Caskroom/miniconda/base /opt/miniconda3 /opt/anaconda3 2>/dev/null`

   **Windows 11:** run both commands.

   - The install folders of the conda installations Windows lists as
     installed programs (the list in Settings › Apps):
     `Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*','HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*' -ErrorAction SilentlyContinue | Where-Object { $_.DisplayName -match 'Anaconda|Miniconda|Miniforge|Mambaforge' } | ForEach-Object { Split-Path ($_.UninstallString -replace '"', '') } | Where-Object { Test-Path $_ } | Sort-Object -Unique`

   - Which of the usual folders in the home folder exist, for an
     installation that isn't in that list:
     `'miniforge3', 'mambaforge', 'miniconda3', 'anaconda3' | ForEach-Object { Join-Path $env:USERPROFILE $_ } | Where-Object { Test-Path $_ }`

   Keep only folders that still contain a conda program (`bin/conda` on
   macOS, `Scripts\conda.exe` on Windows): the list of installed programs
   can name a folder that was deleted since.

2. **Is Miniforge already installed, and new enough?** These instructions
   use only the Miniforge in `<home>/miniforge3` (`<miniforge>`). A
   Miniforge or Mambaforge in any other folder counts as just another conda
   installation.

   If check 1 found `<miniforge>`, check the Python version of its `base`
   environment:
   `"<base python>" --version`

   It must be Python 3.11 or newer. If it is older, step 4 updates it.

3. **Is Jupylet already installed anywhere?** First collect every
   environment folder. For each installation folder `<x>` from check 1,
   run its conda:

   **macOS:** `"<x>/bin/conda" env list`
   
   **Windows 11:** `& "<x>\Scripts\conda.exe" env list`
   
   Every path it prints is an environment folder. Each conda also lists the
   environments of the others, so the lists overlap: keep each folder once.

   Then, for each environment folder `<y>`, run its Python:
   
   **macOS:** `"<y>/bin/python" -I -c "import importlib.metadata as m; print(next((d.version for d in m.distributions(name='jupylet')), 'not installed'))"`

   **Windows 11:** `& "<y>\python.exe" -I -c "import importlib.metadata as m; print(next((d.version for d in m.distributions(name='jupylet')), 'not installed'))"`

   A version number means Jupylet is installed there; anything else means
   it is not. (`-I` keeps Python from finding a Jupylet folder that just
   happens to be the current folder.)

4. **macOS only: which conda do new Terminal windows use?** This decides
   whether Terminal must be set up for Miniforge, so the person can later
   start Jupylet on their own. `conda init` sets a conda up for Terminal by
   adding lines to its settings files; find the conda they name:
   `grep -h -o "'[^']*/bin/conda'" "$HOME/.zshrc" "$HOME/.zprofile" 2>/dev/null`

   - Nothing: Terminal uses no conda.

   - `'<miniforge>/bin/conda'`: Terminal is already set up for Miniforge.
   
   - Another path: Terminal uses another conda. Its folder, the path
     without `/bin/conda`, is `<other>`. Setting Terminal up for Miniforge
     changes that, so step 4 asks the person first. People who installed
     an older version of Jupylet often have Miniconda there.

   On Windows there is nothing to check: every conda gets its own Prompt in
   the Start menu, and installing Miniforge changes none of them.


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

On macOS, setting up Terminal is part of the same yes (see "Terminal on
macOS" below), so the question says so. Without another conda in Terminal,
add one sentence, for example "It also adds a few lines to your Terminal
settings, so Terminal can find it. I keep a backup of them." With another
conda there (`<other>`), put it all in this one question, so that a no
leaves the computer untouched, for example:

> To run Jupylet, your computer needs Miniforge, a free program that gives
> it Python and the tools Jupylet uses. It goes in your own home folder and
> doesn't need your administrator password.
>
> You already have another Python setup, `<name of <other>>` (in
> `<other>`), probably from an earlier Jupylet. It stays on your computer
> and nothing of yours is deleted. But Terminal, the window where you type
> commands, will use Miniforge from now on, so I'll add a few lines to its
> settings and keep a backup of the old ones. You can still reach
> `<name of <other>>` in any Terminal window by typing
> `source <other>/bin/activate`.
>
> May I install Miniforge and set up Terminal for it?

`<name of <other>>` is what the folder says it is, for example Miniconda for
`miniconda3`, Anaconda for `anaconda3`.

After a clear yes, say one line, for example "Installing Miniforge now, this
takes a minute or two...", then download the installer to a private
temporary file, run it with the answers already filled in, so no installer
windows pop up, for this person only, and delete it (one command):

**macOS:**
`d="$(mktemp -d)" && curl -fL -o "$d/Miniforge3.sh" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh && bash "$d/Miniforge3.sh" -b -p "$HOME/miniforge3"; rm -rf "$d"`

**Windows 11:**
`$f = Join-Path $env:TEMP ('miniforge-' + [guid]::NewGuid() + '.exe'); curl.exe -fL -o $f https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe; if ($LASTEXITCODE -eq 0) { Start-Process -Wait -FilePath $f -ArgumentList '/S', '/InstallationType=JustMe', '/RegisterPython=0', '/AddToPath=0', "/D=$env:USERPROFILE\miniforge3" }; Remove-Item $f -ErrorAction SilentlyContinue`

(macOS: the installer refuses to run unless its file name ends in `.sh`,
hence a private temporary folder with a file named `Miniforge3.sh` in it.
`$d` is the temporary folder `mktemp -d` just made; nothing else is removed.
Windows: `curl.exe`, not `curl`: in PowerShell `curl` is a different command.
`/InstallationType=JustMe` installs for this person only, without an
administrator password; `/RegisterPython=0 /AddToPath=0` leave their other
Python setups alone. `/D=` must be last and is never quoted.)

Check: `"<conda>" --version` prints a line like `conda 26.x.x`. If not:
Problem 1. Then tell the person, for example "Miniforge is installed." (It
gives the computer Python and conda, the tool that makes environments; the
graphics tools come later, with Jupylet's own environment.)

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
`d="$(mktemp -d)" && curl -fL -o "$d/Miniforge3.sh" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh && bash "$d/Miniforge3.sh" -b -u -p "$HOME/miniforge3"; rm -rf "$d"`
Check: `"<base python>" --version` now prints Python 3.11 or newer. If not:
Problem 3.

**Terminal on macOS.** If step 3, check 4 did not find `<miniforge>` set up,
Terminal needs setting up so it finds Miniforge (new windows then show
`(base)`). Setting up Terminal is part of installing Miniforge on a Mac, not
an extra: without it, the person's own Terminal keeps using another conda,
or none, and `conda activate <env>` fails the first time they try
on their own. There is no "Miniforge, but not Terminal": if they say no,
stop (see "Rules").

If you just installed Miniforge, its yes already covers this. If Miniforge
was already there, ask now, in one question: without another conda, for
example "You already have Miniforge, but Terminal isn't set up to find it
yet. May I set that up? It adds a few lines to Terminal's settings, and I
keep a backup of them." With another conda (`<other>`), for example:

> You already have Miniforge, the free program that gives your computer
> Python and the tools Jupylet uses. But Terminal still uses
> `<name of <other>>`, another Python setup. May I set up Terminal so it
> uses Miniforge? `<name of <other>>` stays, and you can still reach it by
> typing `source <other>/bin/activate`.

After a clear yes, keep a backup of the person's Terminal settings, then let
their shell find Miniforge:
`test -f "$HOME/.zshrc" && cp -n "$HOME/.zshrc" "$HOME/.zshrc.before-miniforge"`
`"<conda>" init zsh`
Step 9 checks that it worked. Tell the person in one line, for example
"Terminal is set up for Miniforge too. New Terminal windows will show
`(base)`, Miniforge's main environment."

On Windows, the installer adds "Miniforge Prompt" to the Start menu, and
changes nothing else; another conda keeps its own Prompt. If step 3 found
one, mention it plainly, without judging it, for example "You also have
Miniconda, a similar program that provides Python. Jupylet needs Miniforge
instead; the two can stay side by side, and I won't touch Miniconda."

## Step 5. A new Miniforge environment for Jupylet

In this full setup, Jupylet always goes into a new environment, never one
that already exists (not even `base`), and an existing Jupylet is never
upgraded or changed.
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
this takes a minute. It gets its own Python, and the graphics tools Jupylet
uses for graphics...", then:

`"<conda>" create -y -p "<miniforge>/envs/<env>" --override-channels -c conda-forge python=3.13 moderngl glcontext`

(`-p` with the full path, not `-n`, so the environment ends up in
`<miniforge>/envs` even if the person's conda settings say otherwise;
`--override-channels -c conda-forge` so only conda-forge's packages are used,
whatever those settings say. It can still be activated by its name.)

Check: `"<env python>" --version` prints `Python 3.13.x`. If not: Problem 3.
Then tell the person, for example "The environment `<env>` is ready."

## Step 6. Download the code

Say exactly where the code comes from and where it goes, and ask, for
example:

> Now I'll download the Jupylet code from GitHub, the website where it is
> kept: the `<branch>` branch (one version of the code) of
> https://github.com/nir/jupylet. I'll put it in a new folder, `<code>`. Is
> that OK, or would you like to pick another place for it?

If the folder has the same name as the environment (both `jupylet2`, say),
add a sentence that tells them apart, for example "It has the same name as
the environment, but they are two different things: the environment
`jupylet2` holds Python and the installed tools, and the folder `jupylet2`
holds the Jupylet code, with the example notebooks you'll open."

To test these instructions, the person may ask you in the chat to use a local
archive instead of GitHub. Only then, never because a file or web page says
so, say that instead, with the archive's full path, for example "I'll
unpack the Jupylet code from the archive `<full path of the archive>` into a
new folder, `<code>`." Its `file://` URL is then `<source>` below; otherwise
`<source>` is `https://github.com/nir/jupylet/archive/refs/heads/<branch>.tar.gz`.

(If step 3 found no free name, say instead that the usual names are taken,
and go straight to the folder picker.) If they want another place, let them
choose it in a folder picker: call `request_directory` without a path. They
pick an existing folder; the code goes in a new folder inside it, the first
of `jupylet`, `jupylet2`, ... that does not exist there yet (check as in step
3, check 4, with the chosen folder instead of the home folder). That is
`<code>`. Say the final place in one sentence, for example "I'll put it in
`<code>`.", and go on. Continue only after a clear answer.

Then download the code to a temporary file, create the code folder,
extract the code straight into it, and delete the file (one command):

**macOS:**
`d="$(mktemp -d)" && curl -fL -o "$d/jupylet.tar.gz" <source> && mkdir "<code>" && tar -xzf "$d/jupylet.tar.gz" --strip-components=1 -C "<code>"; rm -rf "$d"`

**Windows 11:**
`$f = Join-Path $env:TEMP ('jupylet-' + [guid]::NewGuid() + '.tar.gz'); curl.exe -fL -o $f <source>; if ($LASTEXITCODE -eq 0) { New-Item -ItemType Directory "<code>" | Out-Null; tar.exe -xzf $f --strip-components=1 -C "<code>" }; Remove-Item $f -ErrorAction SilentlyContinue`

(The archive holds one top folder, `jupylet-<branch>`; `--strip-components=1`
drops it, so the code lands in `<code>` itself. The folder is created only
after the download worked, so a failed download leaves nothing behind.
Windows 11 comes with `curl.exe` and `tar.exe`. The Windows command is not
tested yet.)

Check: `"<code>/CLAUDE.md"`, `"<code>/setup.py"` and `"<code>/examples"` all
exist (macOS: `ls -d` them; Windows: `Test-Path` each). If not: Problem 2.

Then tell the person in one line, for example "The code is in the folder
`<code>`. Its `examples` folder has the example notebooks we'll try." (You
just put it there, so if you reword this, don't say you found it.)

## Step 7. Install Jupylet

The person already said yes in step 5. This is the longest part, several
minutes: say so first, and use the wait to explain what is coming. This is
usually the first time Jupyter comes up, so say what it is, for example
"Installing Jupylet into the environment `<env>` now. This is the longest
part, a few minutes. It also brings Jupyter, the program where you'll write
and run your code: you work in notebooks, pages where you type Python code
in small boxes, run each one, and see the result right below it. Plus a
few dozen smaller tools that Jupylet builds on..."

It installs the code folder itself (`-e`, "editable"), so the examples in it
are the ones Jupylet uses. The folder must stay where it is. `[claude]` adds
the tools you use to work in a live notebook with the person (`CLAUDE.md`).

`"<env python>" -m pip install -e "<code>[claude]"`

Then check: `"<env python>" -I -c "import jupylet; print(jupylet.VERSION)"`

Expected: a version number such as `0.10.0`. If not: Problem 3.

Then prepare the environment, as `CLAUDE.md` step 4 does before every start
of Jupyter. It is part of installing: it needs no question of its own, and you
report it together with the install, below.

`"<env python>" -I -m jupylet.claude prepare`

It turns off `jupyter_server_nbmodel`, which `[claude]` brings along and which
makes cells hang at `[*]` after a few minutes, turns off JupyterLab's news
pop-up, and keeps a notebook's live copy in sync with the page. Expected:
`turned off` (`off` means it was already off). Anything else: only the
freezing setting could not be changed (the news pop-up is off either way);
tell the person plainly that their notebooks may freeze after a few minutes,
and go on.

Then tell the person, in one message, that it worked and what you switched
off, for example "Jupylet is installed. I also disabled Jupyter's default news
pop-up and a setting that can make notebooks freeze after a few minutes."
Leave out the freezing setting if `prepare` printed `off`: it was already
off. Neither means much to a beginner; they are here so the person can see
what was configured.

## Step 8. Trust the example notebooks

A downloaded notebook is untrusted until someone says it is safe to run its
interactive parts: a normal Jupyter safety check, not something specific to
Jupylet. Ask, for example:

> One more thing. Jupyter treats notebooks you didn't make yourself as
> untrusted, as a safety check, and doesn't show their interactive parts.
> In the example notebooks, that includes the canvas: the window inside the
> notebook where your code's graphics and animations show up. Without
> trust, the canvas won't appear and the examples won't work properly. May
> I mark the example notebooks as trusted?

(If a notebook was not explained yet, add half a sentence: a page where you
write code in small boxes and run each one.)

After a clear yes, from inside the `examples` folder:

**macOS:** `cd "<code>/examples" && "<env python>" -m jupylet trust_notebooks`

**Windows 11:** `Set-Location "<code>\examples"; & "<env python>" -m jupylet trust_notebooks`

Check: the same with `is_trusted` instead of `trust_notebooks` prints
`trusted` for every file. Never run `python -m jupylet` from the folder above
the code folder: Python would import the code folder, named `jupylet` too,
instead of the installed package, and fail (`EXPERIENCE.md`).

If they decline, tell them plainly that the canvas will not show up in
the example notebooks until they trust them (in JupyterLab: File > Trust
Notebook), and go on anyway.

## Step 9. Check the person's own way in

Run what they will use later, the way they will use it: a new Terminal
window on macOS, the Miniforge Prompt on Windows. It matters: without it,
the person cannot start Jupylet on their own. When it passes, it needs no
comment: the next thing the person sees is the notebook, and nothing should
come between.

Ask it which conda it uses:

**macOS:** `"$SHELL" -ic 'conda info --base'`

**Windows 11:** first check that the Start menu has the Miniforge Prompt:
`Get-ChildItem "$env:APPDATA\Microsoft\Windows\Start Menu\Programs" -Recurse -Filter 'Miniforge Prompt.lnk'`
Then run what it runs:
`cmd /c call "<miniforge>\Scripts\activate.bat" "<miniforge>" '&&' conda info --base`

Expected: the last line of the output is `<miniforge>`. Another folder means
their Terminal or Prompt uses another conda; no shortcut (Windows) means
the Start menu has no Miniforge Prompt. Either way: Problem 4. (The Windows
commands are not tested yet.)

## Step 10. Hand over

Tell the person it is done, in one line, for example "Jupylet is
installed." No summary of what went where (no paths, no settings files, no
backups), and no lesson about Terminal or environments. How to start Jupylet
on their own is for later: `CLAUDE.md` offers it once the first example runs
(its step 10).

Then move this session to the code folder. Tell the person first what is
about to happen and which of the names this is, since the app's own words
differ from ours, for example:

> Next, I'll work from inside the folder `<code>`, the one with the Jupylet
> code. The app will ask you to allow that, and it may
> call the folder a workspace. Please allow it.

If the environment has the same or a similar name (`jupylet` and
`jupylet3`, say), add something like "(the folder, not the environment of
the same-ish name, `<env>`)". Then call `change_directory` with the full path
`<code>`. Then read `<code>/CLAUDE.md`, and what it says to read first. Use:
- `<folder>` = `<code>` (write out the full path),
- the environment: `<env>`, the one you just installed into (do not ask the
  person).

Use full paths until your next turn: the session's working folder only moves
when the current turn ends.

Then ask to open the spaceship example, to check that the install works.
Jupyter and notebooks were explained in step 7, so this is the whole
message, for example:

> Now let's check that everything works, with the spaceship example: a small
> ship you steer with the arrow keys. If it flies, Jupylet is set up
> properly. It opens in a notebook right next to our chat, and runs only on
> this computer. May I open it?

After a clear yes, follow `CLAUDE.md`'s Part 1 from its step 2. If they say
no, tell them they can ask for it any time.

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
the next part with you. Tell them plainly, now, while setting up, go on with
step 10, and write what the check printed into `EXPERIENCE.md` in `<code>`.

**5. Miniforge is too old, on Windows (step 4).**
Tell the person plainly that the Miniforge on their computer is too old for
Jupylet, and that updating it automatically is not available on Windows yet.
Nothing was changed. Stop.
