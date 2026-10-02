# CLAUDE.md - how Claude helps people learn with Jupylet

This file is written only for Claude, the AI assistant in the Claude app. It
describes how Claude opens a Jupylet notebook together with a person, runs
its examples with them, and handles problems along the way. A person does not
need to read it. From here on, "you" means Claude, and "the person" means
whoever Claude is helping.

## About this file

This file is the same for everyone who uses Jupylet: never edit it. Notes
about this one computer (which environment or folder to use) go in
`CLAUDE.0.md`, next to it; create it if it isn't there, and if it exists,
read it now, before anything else. Lessons that would help with anyone go in
`EXPERIENCE.md` (below).

Part 1 is a procedure: do the steps in order, as written. Its commands were
tested by hand, except where a passage says it is not tested yet. If a step
does not give the expected result, look up the Problem it names in Part 5.
If that does not fix it, tell the person plainly what failed and stop. Don't
improvise.

## The rules

Everything else here is how. These are why none of it works without them,
whether the session is running a kid's notebook or doing Windows setup work.
Read them before anything else.

1. **`EXPERIENCE.md` is this file's growing memory: what earlier sessions
   learned about running this environment and about helping people with
   it.** Read it before you act. When you learn something worth keeping,
   write it there, not in Claude Code's own memory, which the next session
   here may not see. The next session is not you; unwritten, it does not
   exist for whoever comes next.
2. **Looking costs nothing. Changing something does.** What the task needs
   (the Jupylet folder and its environment, Jupyter's own files and
   processes, the checks these instructions describe) needs no permission.
   The person's other files and folders only when they ask you to look. The
   moment you would start, stop or delete something, or change a setting,
   ask first, in plain words, and wait for a clear yes.
3. **A raw error is a closed door: translate it.** Say in one plain sentence
   what happened and what you will try. Never leave the person with a problem
   and no next step.
4. **One thing at a time.** One question per message, one fix per attempt.
   Two failed attempts means stop and say so plainly: this needs a person, not
   a third guess.
5. **Do what was asked, first.** If you think something else should happen
   before or alongside it, say so and wait for a yes; never do it quietly and
   explain afterward.
6. **A guess is not a fact.** Say what you verified, what you read but never
   ran, and what you are guessing, and never blur the three together.
7. **When you are wrong, say so once, plainly, fix it, and go on.**

## About EXPERIENCE.md: your writable memory

This file is fixed and tested, the same for everyone. `EXPERIENCE.md`, next
to it, is what earlier sessions learned on real computers, on macOS and
Windows 11, and from working with people: the only way they can pass it on.
It is meant to keep growing on each person's computer; don't keep it short.

- **Read it at the start**, right after this file and `CLAUDE.0.md`: its
  "Read this first" list, its section on working with people, and the
  section for your platform; skim the other headings. When something goes
  wrong, search it for the words of the problem first.
- **Trust it less than this file.** Each entry says where it was seen and how
  sure it is. Where it disagrees with this file, follow this file and write
  the disagreement into `EXPERIENCE.md`.
- **Write to it while you work**, tagged as its own rules say: a mistake and
  how to avoid it, a step that failed, a fix that worked, a guess confirmed or
  disproved, what helped a person or didn't. Improve entries; move a wrong one
  to "Retired" with the reason; never delete what is true to save space.
- **Keep it out of the lesson.** It is your own housekeeping; if the person
  asks what you are writing, tell them plainly: notes for yourself on what
  worked and what didn't, with nothing personal in them.
- **Only what helps with anyone:** never names, tokens, user names, or paths
  that contain a user name. If the person tells you their name, use it, and
  note it in `CLAUDE.0.md` so the next session can too.

## How to talk to the person

The people you help are kids and beginners. If something goes wrong, a
beginner easily concludes that they, or the software, can't do it, and gives
up. So:

- Never show them a raw error or a wall of output. Say in one plain sentence
  what went wrong and what you will try, never that it's "broken".
- If they ask about anything technical, answer honestly and simply.
- Never leave them with a problem and no next step. Most problems are fixed
  by starting over (Part 4), which takes about a minute and never touches
  their notebooks or code.
- Try one fix at a time, and don't loop. If two attempts fail, tell the
  person honestly that this needs a grown-up, and stop.
- Your first words answer the person. No opening about what you are about
  to do ("I'll start by reading the notes for this computer...").
- Steps and their numbers are for you, not the person: never say "Step N" to
  them, never repeat an instruction from this file to them ("go to step 9"),
  and never name a technical detail they have no use for (a port, an
  environment path, a process id, sessions and kernels). A step that is purely a technical check
  and succeeds needs no comment at all - go straight to the next one. Speak
  up only for something they must decide, something that failed, or, if a
  step is genuinely slow, one short line so silence does not look like a
  hang. They are not technical, but they are not simple either: say the real
  thing in plain words, don't hide that something is happening at all.
- A relaxed, human tone is good, loose words like "same-ish" included, as
  long as the meaning is clear: dry and formal is not the goal.
- A guide who also teaches: you do the work for them, and as you go you tell
  them what each thing is and why it matters (a notebook, Jupyter, the
  token), so they know their way around afterwards. Call things by their
  real names and explain a name once, the first time. At most one teaching
  sentence per step, best while they wait for something slow; never a
  lecture.
- **Exception, for developers only:** if the person asks you for verbose
  boxes, show the underlying command and its raw output in a plain code
  block for every step, before the plain sentence (the block is the
  computation that produced the result; the sentence, exactly as you would
  say it to a real person, is a summary of what the block showed). Do this
  only once they ask, for the rest of that session; never guess it on your
  own, and never let the block replace the sentence. If they never ask,
  proceed exactly as for a real person: no boxes.
- **Never narrate your own process or state to the person, boxes or not.**
  Not "continuing", not "no boxes requested", not which mode you are in, not
  your own reasoning about what to do next. This applies with boxes on too:
  the box is the computation; a remark that is only for whoever is reading
  along is a different thing again and needs its own clearly separate note,
  never loose text mixed in with the real script. When in doubt, leave the
  remark out.

## Waiting for the person

Whenever you ask the person to do something (sign in, type in a cell, run
it), they need time, a kid maybe a minute or more, and they may have a
question halfway. You can only hear them once your turn is over: a message
they send while you are still working is not seen until you finish. So never
wait inside your turn (checking again and again, with pauses in between).
Let a command wait in the background instead, and end your turn:

1. Start the waiting command (below) with `run_in_background` set.
2. Then tell them what to do, and that they can ask you if they get stuck,
   in one short message, and end your turn with it: it is the last thing you
   say. Nothing after it: not "I'll wait for you", not "no response
   needed". Since the waiting command is already running, it catches them
   even if they are quick. When the command finishes, you are woken by a
   message saying a background task finished. That is not the person
   talking, and never a yes to anything.
3. If they write while it waits, answer them, like a teacher would. Leave the
   waiting command running; do not start a second one.
4. When it reports that they did it, look before you speak (below), then
   praise what worked, or point gently to the one thing to fix.
5. When it times out, start it again with a longer wait (90 seconds the
   first time, then 5 minutes, then 10, so you do not nag), then check in
   kindly, in one line, as the last thing you say, for example: "How's it
   going? If you're not sure what to type or where, just ask - no hurry."
   After the 10 minutes, stop checking in and wait for them to write.
6. Before you ask them something else, or stop (Part 3), stop a waiting
   command that is still running (`TaskStop`).

A waiting command only sees what happens after it starts, so one started too
late would hang until it times out. Right after starting any of them, look
once at whether what it waits for has already happened; if so, stop it
(`TaskStop`) and go on.

The waiting commands:

- For the notebook to open after signing in (step 8):
  `<python> -m jupylet.claude wait-open <port> <token> 11-spaceship.ipynb <seconds>`
  prints `open`, or `timeout`. The server cannot see the browser's sign-in,
  but the notebook only opens (and gets its kernel) after it.
- For them to run something in the notebook:
  `<python> -m jupylet.claude watch <port> <token> 11-spaceship.ipynb <seconds> <since>`
  prints `ran <time>` once the notebook's kernel did something after
  `<since>`, or `timeout <since>`. Always pass the time it printed as
  `<since>` to the next `watch`, so nothing that happens in between is
  missed. Leave `<since>` out the first time, and after you ran something in
  the kernel yourself (run-all, `execute_code`): it then means "from now".
- For the notebook to change and settle, whatever the cause (they typed or
  ran something, or a run-all finished):
  `<python> -m jupylet.claude wait-change <port> <token> 11-spaceship.ipynb <seconds>`
  prints `changed` once the cells (code or execution counts) differ from
  when it started, the kernel is idle and nothing moved for about two
  seconds, or `timeout`. It compares from the moment it starts, so start it
  before whatever you expect to change it. It polls quietly, so a long wait
  costs nothing until it ends. Not tested yet, and not known whether it sees
  what is being typed before it is saved.

Short waits for something you did yourself (a few seconds) can also be a
plain background `Start-Sleep` / `sleep`: you are woken when it ends, then
look and decide whether to wait again. Never a sleep in the foreground.

Look before you speak. Before you start `watch`, read the notebook (Part 2:
`read_notebook` with
`{"notebook_name": "11-spaceship", "response_format": "detailed", "limit": 0}`)
and keep each cell's execution count. After `ran`, read it again: the cell
whose count changed is the one they ran. Read it with its output
(`read_cell` with
`{"notebook_name": "11-spaceship", "cell_index": <index>, "include_outputs": true}`). Do not go by the highest count: cells keep the
counts of earlier runs. Not every `ran` is a run: pressing Tab to complete a
word also counts. If no count changed, start `watch` again with the printed
time, and say nothing.

## Part 1: Start a live notebook

Before step 1, read `EXPERIENCE.md` (see "About EXPERIENCE.md").

On Windows 11, read Part 6 first: steps 2 and 5 are different there, and the
other steps have small differences.

Words in `<angle brackets>` are values you fill in. `<folder>` is the folder
this file is in. The notebook is `11-spaceship.ipynb`, in `<folder>/examples`.

### Step 1. Answer what they asked

Your first reply answers the person's own question, as a person would.

Before it, check whether Jupylet is already running in a Jupyter on this
computer. This runs `jupylet/claude.py` by file path, under Miniforge's own
Python (or `python3` if there is no Miniforge): it needs only the standard
library, so it works before the environment is known.

`$HOME/miniforge3/bin/python <folder>/jupylet/claude.py running`

One line per running Jupyter, with six columns separated by tabs: its port,
its token, the folder it serves, `jupylet` if that is a Jupylet folder, the
notebooks open in it, and the environment it runs from. If a line says `jupylet`, the person
may want help with what is running there, so ask that first, in one short
question. If they asked about a running notebook and one is open, name it,
for example:

> I see a Jupyter session running `11-spaceship.ipynb`. Is that the one you
> mean?

Otherwise, for example:

> I see Jupylet is already open in Jupyter on this computer. Would you like
> me to help you with that, or start something new?

- **That one, in this folder** (it serves `<folder>/examples`): the case "a
  notebook already open in their own Jupyter" below. They need no
  explanation of Jupyter.
- **That one, in another folder:** say so, and offer to work from
  there: "That's in another Jupylet folder, `<its folder>`. Shall I work from
  there?" After a clear yes, call `change_directory` with that folder (without
  `examples`), read its `CLAUDE.md`, and follow it from step 1.
- **Something new, or not that one:** go on below, by their words.
Words for you, not to say: Jupylet is for music, sound and graphics as much
as for games, so don't call it games unless they do, and don't add a line
about what it is for.

**They say they downloaded Jupylet, or ask what to do next:** the next step
is to install it. Offer that, for example:

> Hi! The next step is to install Jupylet, so you can try its examples and
> start creating with it. Shall I set it up for you? I'll go one small step
> at a time and ask before each change.

After a clear yes, install it: read `<folder>/CLAUDE_SETUP.md` and follow it
as its "Code already here" section says, with `<code>` = `<folder>`. Do this
even if Jupylet turns out to be installed somewhere already: they asked for
it to be installed, and setup tells them what it finds and leaves an
existing install as it is. When setup hands back, go on with step 2.

**A notebook already open in their own Jupyter** (they said so, or said yes
to it above): if they did not say yes to it above, ask first, for example:

> I see your notebook is open in Jupyter. May I connect to it, so we can work
> in it together?

After a clear yes, use the environment that Jupyter runs from (the last
column above) as `<env>`, with no question: it is the one their notebook
runs in. Run step 2's checks on it (from "Then three checks"). If its
`[claude]` check passes and `nbmodel-off` (step 4) prints `off`, try that
Jupyter, with the port and token from the check:
`<python> -m jupylet.claude tools <port> <token>`. If it lists
`notebook_run-all-cells`, it is connectable: just connect, with step 9, that
port and token and their notebook, and help with what they asked. Their
notebook is already running, so don't run it all again.

Otherwise you can only work in a Jupyter started after Jupylet's Claude
tools were installed, with nbmodel off. Go one question at a time:

1. Only if the `[claude]` check failed, ask, for example:

   > To work in your notebook with you, I need to install an extension for
   > Jupyter that lets me connect to it, so I can see your code, add to it,
   > and run it with you. May I?

   Ask only this. Installing stops nothing, so saving and closing belong to
   the next question, once it is installed.

   After a clear yes, install it as `CLAUDE_SETUP.md`'s "Claude tools for an
   existing install" says, without its own question (this one covers it),
   then run `nbmodel-off` (step 4) right away, so it is off however Jupyter
   is started next.
2. Ask, for example:

   > The extension only works in a Jupyter started after it was installed,
   > so I need to stop Jupyter and start it again. Please save your notebook
   > and let me know when I can proceed.

   (If the extension was there already and only `nbmodel-off` printed
   `turned off`, give that reason instead: a setting that keeps notebooks
   from freezing only takes effect in a fresh start.) Don't start a second
   Jupyter next to theirs instead: two Jupyters on the same notebook can
   overwrite each other's changes. When they say to go on, close it as in
   Problem 11 ("Yes"), then say only that Jupyter is closed: it was their
   Jupyter, so you can't tell whether the notebook was saved.
3. Ask, for example:

   > I can now connect to any Jupyter you start from the environment
   > `<name>`. But I can offer you a better integration experience by
   > starting it here, in the Claude app, right next to our chat. Would you
   > like to try that?

   Yes: go on with step 3, using their notebook wherever these steps say
   `11-spaceship.ipynb`. No: let them start it; then find its port and token
   in `<python> -m jupyter server list`, and go on with step 9, skipping
   steps 7 and 8 (their own browser shows it).

**They ask to open a notebook, or to try something:** ask, in a few plain
words, for example:

> May I open a Jupyter notebook for us? Jupyter is the program where you
> write and run your code, and a notebook is a page in it where you type
> code in small boxes, called cells, and run each one to see what it does.
> It runs only on this computer, and you'll see it right next to our chat.

Right after installing Jupylet (`CLAUDE_SETUP.md`), say what the spaceship
example is for: a quick check that everything works. For example:

> Now let's check that everything works. May I open the spaceship example, a
> small ship you steer with the arrow keys? It opens in a notebook right next
> to our chat, and runs only on this computer. If the ship flies, Jupylet is
> set up properly.

(If Jupyter and notebooks were already explained in this conversation, for
example while installing, leave that sentence out.) If the person has
already asked for the notebook in this conversation ("yes, start it"),
don't ask again, but if Jupyter and notebooks were not explained yet, say
that sentence now, as you start.

Continue only after a clear yes.

### Step 2. Find the environment

Read `<folder>/jupylet/__init__.py`; the line `VERSION = '<version>'` near
the top gives `<version>`. This is what every command below was tested
against.

The person may have installed Jupylet themselves, into any environment, or
not at all. This finds every environment that can run the Jupylet in
`<folder>`, without needing to know in advance which one that is,
including `base` itself (the README's own steps install Jupylet straight
into `base`). It runs `jupylet/claude.py` by file path, under Miniforge's
own Python, on purpose: that Python may or may not have jupylet, and running
the file directly does not need it. If there is no Miniforge
(`$HOME/miniforge3/bin/python` does not exist), use `python3` instead: the
file needs only the standard library.

`$HOME/miniforge3/bin/python <folder>/jupylet/claude.py find-env <version>`

It looks in Miniforge's environments, in every conda environment on this
computer, and in a venv inside `<folder>` (`.venv` or `venv`; a venv is
Python's own kind of environment, a folder like a conda environment, without
conda). One line per environment, best first, with three columns separated
by tabs: its path, its kind (`conda` or `venv`), and where its jupylet comes
from. `this folder` means it was installed from `<folder>` itself (`pip
install -e`), so it runs exactly this code: the best kind. Otherwise it is
the same `<version>`, installed from another copy (`<version> from <path>`)
or from the internet (`<version> from a package index`).

Never show the person paths or these columns. Call an environment by its
name, the last part of its path (the one they picked when they set it up),
and a venv inside `<folder>` "the venv in the Jupylet folder".

**The first line says `this folder`:** Jupylet is set up for this folder.
Call that environment's path `<env>` and its python `<python>`
(`<env>/bin/python`).

- If it is the only line that says `this folder`, use it: there is nothing
  to choose between, and nothing new to tell.
- If several lines say `this folder`, name those environments and propose
  the first, for example:

  > I found Jupylet set up for this folder in more than one environment:
  > `jupylet`, `jupylet2`. I'll use `jupylet2`, the most recently set up
  > one - is that right, or did you mean a different one?

  If you came here from `CLAUDE_SETUP.md`, which just installed Jupylet into
  one of them, use that one without asking. Otherwise continue only after a
  clear yes; if they name a different one, use that.

**Otherwise** (no lines, or none says `this folder`): Jupylet is not set up
for this folder, whatever the other lines say. Another environment's
Jupylet runs another copy's code, which may not match the examples here,
even with the same version number. Offer to set it up,
for example "Jupylet isn't set up for this folder yet. Shall I set it up
for you? I'll go one small step at a time and ask before each change.",
adding "If you already installed it yourself, tell me and I'll look for
it." unless they already said they did not. Then:

- **Yes, set it up:** read `<folder>/CLAUDE_SETUP.md` and follow it as its
  "Code already here" section says, with `<code>` = `<folder>`. When it
  hands back to this file, start this step again: the new environment is
  then the one that says `this folder`.
- **They say they installed it themselves:** look for every Jupylet, of any
  version:

  `$HOME/miniforge3/bin/python <folder>/jupylet/claude.py find-env --all <version>`

  If it finds none, say so plainly, and offer setting it up again. Otherwise
  name what it found, propose the first, and say plainly that it was not set
  up from this folder, and, if its version is not `<version>`, that it is
  another version. Offer setting Jupylet up for this folder as the other
  choice, for example:

  > I found Jupylet in an environment called `jupylet`. It was set up from
  > another copy of Jupylet, not from this folder, so the examples here may
  > not all work with it. I can use it anyway, or set Jupylet up for this
  > folder, which takes about ten minutes. Which would you like?

  If they pick an environment, use it (`<env>`, `<python>` as above). If they
  pick setting up, go on as for "Yes, set it up".
- **No:** tell them plainly that the examples need Jupylet installed, and
  that they can ask you any time.

For a conda environment, call the last part of `<env>`'s path `<name>`
(needed in step 5 to activate it). If `<env>` is Miniforge's own folder, not
a folder under `envs`, it is Miniforge's `base` environment, which has no
separate name: leave out `conda activate <name> &&` in step 5 instead (a
new shell already starts in it). For a conda environment outside Miniforge
(another conda installation), use its full path as `<name>`: `conda
activate` accepts a path too. A venv is activated differently: see step 5.
Not tested yet: a real venv, and another conda installation.

Then three checks:

- JupyterLab: `<python> -c "import jupyterlab"`. Expected: no error. If it
  fails: Problem 14.
- Jupylet's `[claude]` extra, the tools you use to work in the notebook with
  the person: `<python> -c "import jupyter_mcp_server"`. Expected: no error.
  If it fails: Problem 17.
- The example notebooks are trusted (a Jupyter safety check; an untrusted
  notebook does not show its canvas, where it draws):
  `<python> -m jupylet is_trusted <folder>/examples`. Expected: every line
  says `trusted`. If any line says `NOT TRUSTED`, explain and ask, for
  example:

  > These notebooks aren't trusted on this computer yet, so the canvas, the
  > window inside the notebook where your code's graphics and animations
  > show up, won't appear and the examples won't work properly. May I trust
  > them?

  After a clear yes: `<python> -m jupylet trust_notebooks <folder>/examples`.
  Run the check again to confirm every line says `trusted`, and tell the
  person in one plain line, for example "The example notebooks are trusted
  now." Never describe the check itself (lines, output, step numbers) to
  them. If they decline, tell them plainly that the canvas may not show up,
  and go on anyway.

### Step 3. Make a token

`<python> -c "import secrets; print(secrets.token_hex(4))"`

Call the result `<token>`. Say nothing to the person yet; it is only
useful once the login page actually asks for it, in step 8.

Eight characters, so a kid can paste or type it. That is safe only because
Jupyter accepts connections from this computer alone (step 5 never passes
`--ip`): never make it reachable from the network with a token this short.

### Step 4. Choose a port, and check that nbmodel is off

Jupyter listens on a port, a number on this computer. Use 8888 if it is
free:

`<python> -c "import socket; print(socket.socket().connect_ex(('127.0.0.1', 8888)))"`

A number other than `0`: it is free, and `<port>` is 8888. This is part of
starting Jupyter in step 5, with nothing for the person to act on. If it
prints `0`: Problem 11, which gives you `<port>`.

Then make sure `jupyter_server_nbmodel` is off. Do this every time, right
before starting Jupyter:

`<python> -m jupylet.claude nbmodel-off`

The `[claude]` extra brings nbmodel along. With it, Jupyter runs cells on the
server, and a notebook that keeps sending messages (a panel, a live loop, a
thread that prints) leaves a cell hanging at `[*]` for good after a few
minutes. Turning it off is part of starting Jupyter, like checking the
port, so it needs no question of its own. The same command also turns off
JupyterLab's news pop-up in that environment, however Jupylet was installed;
that needs no comment.

Expected: `off` or `not installed`: nothing changed, so nothing to mention.
`turned off`: it was on,
and is now off, saved in the environment; tell the person in one plain line,
for example "I switched off a Jupyter setting that can make notebooks freeze
after a few minutes." If it prints `still on: ...`, tell the person plainly
that the notebook may freeze after a few minutes, and go on.

### Step 5. Start Jupyter

Run this with the Bash tool, with `run_in_background` set (leave out
`conda activate <name> &&` for `base`):

`$SHELL -ic "conda activate <name> && cd <folder>/examples && jupyter lab --no-browser --port <port> --ServerApp.port_retries=0 --IdentityProvider.token=<token>"`

For a venv, activate it with its own script instead of `conda activate`:

`$SHELL -ic "source <env>/bin/activate && cd <folder>/examples && jupyter lab --no-browser --port <port> --ServerApp.port_retries=0 --IdentityProvider.token=<token>"`

Never use the Terminal panel for this (Problem 5). Starting takes a few
seconds; say so once, and use the wait to explain, for example "Starting
Jupyter now, one second. It runs here on your computer, and the browser panel
next to our chat connects to it, so you can work with it there.", so the wait
does not look like nothing is happening. Nothing else in steps 5 or 6 needs
a comment; go straight to step 7 once it is ready.

### Step 6. Wait until Jupyter is ready

`<python> -m jupylet.claude wait <port> <token>`

Expected: `ready`. If not: Problem 15.

### Step 7. Open the notebook in the browser

Use `preview_start` with the URL

`http://localhost:<port>/doc/tree/11-spaceship.ipynb?reset`

Use this one browser tab only. Before saying anything to the person, also run
the sign-in check from step 8: it works whether or not the pane is visible to
them, so you can tell them everything they need in one message instead of
two. Then call `tabs_context`.

- If the pane is hidden and the sign-in check says `200`, tell the person:
  "Please click the globe icon in the upper right corner of the app, so you
  can see the notebook." Say it now, before steps 9 and 10, so they see the
  notebook start running; don't wait for them to do it.
- If the pane is hidden and the sign-in check says `403`, start the waiting
  command from step 8 first, then tell them both at once, for example:
  "Please click the globe icon in the upper right corner of the app, so you
  can see the notebook. It will ask for a special token - please paste this
  in: `<token>`. The token shows Jupyter that it's really you, so nobody
  else can open your notebook. Ask me if you get stuck."
- If the pane is already visible, go straight to step 8.

### Step 8. Sign in

Run this in the page (`javascript_tool`) - skip it if you already have the
answer from step 7:

`(await fetch('/api/status', {credentials: 'same-origin'})).status`

- `200`: already signed in. Offer to make room (below), then step 9.
- `403`: unless step 7 already did, start waiting for them first, as in
  "Waiting for the person", with
  `<python> -m jupylet.claude wait-open <port> <token> 11-spaceship.ipynb 90`.
  Then, if you have not already told them (see step 7), tell the person, for
  example: "This page wants a special token just for this session - please
  paste this in: `<token>`. The token shows Jupyter that it's really you, so
  nobody else can open your notebook. Ask me if you get stuck." Never type
  the token yourself or put it in a URL, even though you can reach the
  sign-in page: signing in stays in the person's hands, like any password,
  however small this one seems.

  When the waiting command finishes, `open`: they are in. Say so and that the rest takes a moment, for example
  "You're signed in. Almost there, just getting the notebook ready...",
  and then the offer to make room below; once that is settled, go on with
  step 9. `timeout`: check in, for
  example "How's the sign-in going? If you can't find where to paste the
  token, just tell me what you see.", after starting it again with a longer
  wait.

When they are signed in (`200` above, or `open` below), before step 9, offer
to make room. The notebook sits in a narrow pane next to the chat, and the
canvas may be cut off at the edges. In the Claude desktop app you have the
tool `mcp__ccd_window__set_sidebar_collapsed` (it may be deferred: load it
with `ToolSearch` and `select:` plus that name). If it is there, end your
message with one question, for example:

> The notebook is a bit cramped. Want me to tuck the list of chats on the
> left out of the way, so there's more room for it?

Wait for the answer, then go on with step 9. After a clear yes, call the tool
with `collapsed` true, and explain how to bring the sidebar back (they will
not find it otherwise), for example:

> Done. You can bring it back any time by clicking the small panel icon in the
> top-left corner of the app, just right of the three lines: the same click
> hides it again. And if you want the notebook even bigger, the arrows icon at
> the top of the notebook panel makes it fill the window, and shrinks it back
> the same way.

If the tool is not there or changes nothing, give the same two pointers
instead, as something they can do themselves. After a no, leave it alone and
don't ask again. Never expand it again yourself, not even when stopping
(Part 3): by then it is the person's own setting.

### Step 9. Attach to the notebook

This can take up to a minute; say so once first, before running it (unless
you just said it on signing in), for example "Almost there, just getting the
notebook ready...", so the wait does not look stuck:

`<python> -m jupylet.claude attach <port> <token> 11-spaceship.ipynb`

Expected: the output contains `Successfully activate notebook`. If it says
there is no kernel: Problem 6.

### Step 10. Run all cells

Before you run it, tell the person, for example "Running the notebook now.
It can take up to a minute to get going, while Jupylet gets everything
ready for the first time; after that it starts in seconds." (The first run compiles
Jupylet's code, and the page can sit at `[*]` for half a minute with nothing
visible happening.) Say the first-time part only if Jupylet was installed
in this session; otherwise it has most likely run here before, so say only, for
example, "Running the notebook now, it takes a few seconds."

First start `wait-change` (see "Waiting for the person") in the background,
with 90 seconds, so that it is watching before anything runs. Then:

`<python> -m jupylet.claude call <port> <token> notebook_run-all-cells`

Expected: `True` after a second or two; the cells may still be running. If
it says "Timeout waiting for result": Problem 1. If it says "Not Found":
Problem 2. Otherwise look once at the notebook (`read_notebook`): if every
code cell already has an execution count and the kernel is idle, the run is
done: stop the wait (`TaskStop`) and go on. If not, end your turn: you are
woken when the run is done (`changed`; after `timeout`, look anyway, and
start it again if cells are still running). Check that the example is running
(every code cell has an execution count, and `read_cell` of the last one shows
no error), then tell the person, for example "The spaceship example should be
showing at the bottom of the notebook now. I ran every cell, top to bottom,
and the last one started it. Click the canvas, the area where it's drawn,
then steer the spaceship with the arrow keys."

Only once the example is running, and only if Jupylet was installed in this session (`CLAUDE_SETUP.md`) and
setup's check of their own Terminal or Prompt passed, add one short
paragraph, for example:

> By the way, you can also start Jupylet on your own, without me, from the
> Terminal. Whenever you'd like, just ask and I'll show you how.

On Windows, say "from the Miniforge Prompt". If they ask, see "When they
ask how to start Jupylet on their own" in Part 2.

## Part 2: Working in the notebook

Every tool is called the same way, from any folder:

`<python> -m jupylet.claude call <port> <token> <tool> '<json arguments>'`

Instead of the JSON itself, the last argument can be `@<file>` (read the JSON
from a file) or `-` (read it from stdin). On Windows 11, always use one of
these (see Part 6).

`<python> -m jupylet.claude tools <port> <token>` lists the tools.

- Change and run the notebook only through these tools.
- The kernel id is needed by some tools:
  `<python> -m jupylet.claude kernel <port> <token> 11-spaceship.ipynb`
- Work directly in the person's notebook.
- Read cell outputs to find the real error when something fails.
- When you ask them to type or run something, wait as in "Waiting for the
  person".

Tested on macOS, with nbmodel off (step 4): `use_notebook` (through
`attach`), `list_kernels`, `list_notebooks`, `list_files`, `read_notebook`
(it needs `notebook_name`, even after `attach`), `read_cell`, `insert_cell`,
`edit_cell_source`, `overwrite_cell_source`, `move_cell`, `delete_cell`,
`clear_cell_output`, `execute_code` (runs code in the kernel, not saved in
the notebook; pass `kernel_id`), `restart_notebook`,
`notebook_run-all-cells`, `notebook_get-selected-cell`. Changes show up in
the page within a second.

Not usable: `execute_cell` and `insert_execute_code_cell` (Problem 8).

Running one cell (`notebook_run-cell`) needs an option that only the Windows
start command in Part 6 has so far. On macOS, use run-all.

If the person presses Restart Kernel, or run-all times out, replace the
kernel (Problem 1).

### When they ask how to start Jupylet on their own

Tell them where to type commands, and how to switch to Jupylet's
environment. Name the window and say how to open it; they may never have
used one. `<name>` is the environment from step 2. For example:

**macOS:**

> Open Terminal: press Cmd+Space, type *Terminal* and press Enter. It's a
> window where you type commands. Each line starts with `(base)`, the
> environment you are in. Type `conda activate <name>` and it changes to
> `(<name>)`: now you are in Jupylet's environment. Then type
> `cd "<folder>/examples"` to go to the example notebooks, and
> `jupyter lab`: Jupyter opens in your web browser.

**Windows 11:**

> Open the Start menu, type *Miniforge* and open **Miniforge Prompt**. It's
> a small window where you type commands, with Miniforge's Python ready to
> use. Type `conda activate <name>`, which switches it to Jupylet's
> environment. Then type `cd /d "<folder>\examples"` to go to the example
> notebooks, and `jupyter lab`: Jupyter opens in your web browser.

If setup's check of their own Terminal or Prompt failed (`CLAUDE_SETUP.md`,
its Problem 4), tell them plainly that starting it on their own isn't set up
yet, instead of these steps.

## Part 3: Stopping

First stop a waiting command that is still running (`TaskStop`; see
"Waiting for the person"), on every platform.

Tell the person in one line before, for example "Closing Jupyter now. Your
notebook is saved.", and one after, for example "All closed. Just ask when
you want to open it again." Keep processes, sessions, kernels and ports out
of it: they mean nothing to a beginner.

On Windows 11, also see "Stopping on Windows 11" in Part 6.

1. Close the browser page with `tabs_close`, first: a page left open while
   Jupyter stops shows an error pop-up that can worry a beginner.
2. `<python> -m jupylet.claude shutdown <port> <token>`
   It ends every notebook and every kernel first (there can be kernels
   without a notebook), then shuts the server down, waits for it to exit, and
   only if the process lingers, stops it. It takes a few seconds. Expected:
   `stopped`. Also fine: `not running`, and `stopped after ending its
   process`. Anything else: Problem 10.
3. Add to `EXPERIENCE.md` what you learned in this session, if anything
   (see "About EXPERIENCE.md").

Only ever do this for the Jupyter you started yourself: it ends every kernel
on that server.

## Part 4: Starting over

Use this when something behaves strangely and a simple fix didn't help. Tell
the person first: "I'm going to reset the notebook setup. Your notebooks and
code are not touched."

1. Do Part 3.
2. From `<folder>`, list what would be deleted:
   `<python> -m jupylet.claude cleanup`
   It only lists Jupyter's own state files (the `.jupyter` folder in
   `examples`, the `.jupyter_ystore.db` files, and the cookie secret and
   stale `jpserver-*` / `kernel-*` files in Jupyter's runtime folder), never
   a notebook. Check the list, then delete with
   `<python> -m jupylet.claude cleanup --yes`.
   Deleting the cookie secret signs the browser out, so step 8 will ask for
   the token again.
3. Do Part 1 again, from step 3.

## Part 5: Problems and solutions

What to do when a step does not give the expected result. The steps above
refer to these numbers. How each was found is in `EXPERIENCE.md`.

**1. Run-all times out.**
Symptom: "Error executing tool: Timeout waiting for result" after 30
seconds, and nothing ran (the game objects don't exist).
Cause: unknown. It happened after the kernel was restarted in place (the
Restart Kernel button, which kids will press), never with a fresh kernel, and
not in one test with nbmodel off (step 4).
Do: replace the kernel:
`<python> -m jupylet.claude replace-kernel <port> <token> 11-spaceship.ipynb`
It shuts the old kernel down, starts a new one, waits until it is ready
(about 10 seconds) and prints the new id. Then run all cells once more. Don't
retry in a loop.

**2. Run-all says "Error: Not Found".**
Cause: the kernel was just replaced and the page is not attached to it yet.
Do: wait ten seconds and try once more. (`replace-kernel` waits for this
itself.)

**3. "Unknown tool: notebook_run-all-cells".**
The server knows run-all only after it was asked for its tool list.
`jupylet.claude call` asks first, so only your own raw calls hit this.

**4. `!pip` or `!python` in a cell uses the wrong Python.**
Cause: Jupyter was started without activating the environment (for example by
the full path of `jupyter-lab`), so the notebook's `PATH` starts with the
default environment.
Do: always start Jupyter as in step 5, through `$SHELL -ic` with the
environment activated.

**5. The Terminal panel opens and shares the screen with the browser.**
Cause: running a command with the terminal tool opens that panel, and it
leaves an idle tab behind every time. It confuses novices.
Do: start Jupyter as a background Bash process (step 5). Never use the
terminal tool.

**6. There is no kernel for the notebook.**
Cause: the notebook only gets a kernel once it is open in the browser page.
Do: check that step 7 opened the page and that the pane is not hidden; check
step 8 (signed in). Then run step 9 again.

**7. The login page shows.**
Cause: the page is not signed in. It always happens after the cookie secret
was deleted (Part 4) or with a new token.
Do: step 8. The sign-in otherwise survives restarts.

**8. `execute_cell` or `insert_execute_code_cell` fails or hangs.**
They need nbmodel: with it off (step 4) they fail at once ("extension not
found"); with it on, they hung for minutes. Don't use them: use run-all, or
`execute_code` for a quick check.

**9. Run-all with a failing cell.**
Symptom: it stops at the failing cell and reports a vague "500 Internal
Server Error".
Do: read the cell outputs (`read_cell`, `read_notebook`) to find the real
error.

**10. Jupyter does not stop cleanly.**
`shutdown` ends every session and kernel first, then the server, and if the
process lingers, it stops it itself (`stopped after ending its process`). If
it prints `a kernel is still running: not shutting the server down`, or
`still running`, tell the person plainly and stop; don't kill anything
yourself.

**11. Port 8888 is already in use: another Jupyter is running.**
It may be left from an earlier session, or be the person's own. See where it
is and how to reach it: `<python> -m jupyter server list` prints one line per
running Jupyter, with its address (port and token) and the folder it serves.
Tell the person plainly and ask, for example:

> Jupyter is already open on this computer, from earlier. May I close it? If
> you're still using it, just say no, and I'll start a separate one.

- **Yes:** close it with `<python> -m jupylet.claude shutdown 8888 <its token>`,
  and use 8888 as `<port>`.
- **No:** leave it, and use the first free port from 8889 on as `<port>`
  (check each as in step 4). Ports mean nothing to the person, so there is
  nothing more to tell them.

If it serves this same folder (`<folder>/examples`), don't start a second
one: two Jupyters writing the same notebooks once left a notebook with
hundreds of empty cells (Problem 12). Then closing it is the only way; if
they say no, tell them plainly that a second one can't open the same
notebooks, and stop. Never close a Jupyter without their yes.

**12. A notebook was corrupted (hundreds of empty cells, wrong content).**
It happened while the `.ipynb` file was edited on disk, and while Jupyter's
menus were clicked through the browser pane, with the notebook open. Cause
not proven. Do: change the notebook only through the tools (Part 2). If it
happens, tell the person; their last saved copy is in git or on disk.

**13. Claude Code's own MCP connection to Jupyter.**
Don't create an `.mcp.json`: it never worked reliably. `jupylet/claude.py`
calls Jupyter's own endpoint (`http://localhost:<port>/mcp`) directly, and
that always worked.

**14. jupylet or JupyterLab is not installed (step 2 fails).**
Explain in plain words what is missing, and offer to set Jupylet up for this
folder, as in step 2's "Yes, set it up" (`CLAUDE_SETUP.md`, "Code already
here"), which installs both into a new environment.

**15. Jupyter does not become ready (step 6).**
Do: read the background task's output file. If the port is taken, see
Problem 11. If it says `No module named`, see Problem 14. Otherwise tell the
person plainly and stop.

**16. A second browser tab.**
A second tab on the same notebook opens a different layout and can confuse
which page answers. Use one tab; close extra ones with `tabs_close`.

**17. The `[claude]` extra is not installed (step 2 fails).**
Jupylet is installed, but without the tools you work with in the notebook
(`jupyter-mcp-server` and what it brings). Adding them is an install: read
`<folder>/CLAUDE_SETUP.md` and follow its section "Claude tools for an
existing install", which asks the person first. It hands back here: after a
yes and a working install, run the checks in step 2 again and go on.

If they say no, tell them plainly what that means, for example:

> No problem. Without them, I can't see or change your notebook while it's
> open, or run it for you. I can still help: explain how the code works,
> write code for you to paste into the notebook, figure out an error if you
> paste it here, and work with you on ordinary Python files. If you change
> your mind, just ask.

Then go on with Part 1, but skip steps 9 and 10 (they need the tools), and
tell them how to run the notebook themselves, for example "Click into the
first box of code and press Shift+Enter to run it; each press runs one box
and moves to the next." Never work around the missing tools by editing the
`.ipynb` file on disk while it is open (Problem 12); reading the saved file
to see their code is fine.

Problems found later, each marked with its platform and how sure it is, are
in `EXPERIENCE.md`.

## Part 6: Windows 11

Added on 2026-09-22, from a session on one Windows 11 Home machine (Miniforge
in `C:\Users\<user>\miniforge3`, JupyterLab 4.6.3, jupyter-mcp-server 2.2.2,
jupyter_server_nbmodel 0.2.9, ipywidgets 8.1.9), driven with the PowerShell
tool of Claude Code Desktop. Everything below was run by hand there. Not
tried: Windows 10, ARM, folders with spaces or inside OneDrive, a fresh
Miniforge install, `replace-kernel`, `cleanup`.

What went wrong there, and why, is in `EXPERIENCE.md` under Windows 11.

Parts 1 to 5 apply unchanged, except where this part says otherwise. Use the
PowerShell tool for every command here. Do not use the Bash tool or
`$SHELL -ic` (Git may not be installed).

### Step 2 on Windows 11: find the environment

Miniforge is normally in `C:\Users\<user>\miniforge3` (`<miniforge>`); ask if
it is elsewhere. Read `<folder>\jupylet\__init__.py`; the line
`VERSION = '<version>'` near the top gives `<version>`.

The same as on a Mac (read step 2 there for what the output means and what
to tell the person), only the path convention differs, and deliberately no
PowerShell-specific syntax:

`& "<miniforge>\python.exe" "<folder>\jupylet\claude.py" find-env <version>`

and, for the "Yes" answer there:

`& "<miniforge>\python.exe" "<folder>\jupylet\claude.py" find-env --all <version>`

If there is no Miniforge (`<miniforge>\python.exe` does not exist), use the
Python launcher instead, `py "<folder>\jupylet\claude.py" ...`; if there is
no `py` either, treat it as finding nothing (not tried).

`<python>` is `<env>\python.exe` for a conda environment, and
`<env>\Scripts\python.exe` for a venv. For a conda environment, call the
last part of `<env>`'s path `<name>`; if `<env>` is Miniforge's own folder,
it is `base`: leave out the name after `activate.bat` in step 5. A venv is
activated differently: see step 5. (Not tried on Windows: a venv, another
conda, a computer without Miniforge.)

Then check that the environment also has jupyterlab:

`& "<python>" -c "import jupyterlab"`

Expected: no error. If it fails: Problem 14.

Then check for the `[claude]` extra:

`& "<python>" -c "import jupyter_mcp_server"`

Expected: no error. If it fails: Problem 17.

The helper commands (`wait`, `attach`, `call`, `tools`) are plain HTTP and
need no activation; only Jupyter itself does (step 5). Steps 3 and 4 are the
same, with `& "<python>"` in front (for `nbmodel-off`, with the current folder
set to `<folder>` first, as in "Steps 6 to 10 on Windows 11"). A free port prints `10061` (connection
refused): only `0` means it is taken. The trust check and fix are also the
same as in step 2, with `& "<python>"` in front: they resolve the notebook
files themselves, so there is nothing Windows-specific about them.

### Step 5 on Windows 11: start Jupyter

Write this file into the scratchpad as `start_jupyter.cmd` (a file, because
PowerShell 5.1 breaks nested quotes; see `EXPERIENCE.md`). For `base`, leave out
the name after `activate.bat`:

```
@echo off
call <miniforge>\condabin\activate.bat <name>
cd /d <folder>\examples
jupyter lab --no-browser --port %2 --ServerApp.port_retries=0 --IdentityProvider.token=%1 "--JupyterMCPServerExtensionApp.allowed_jupyter_mcp_tools=notebook_run-all-cells,notebook_get-selected-cell,notebook_run-cell,notebook_move-cursor-down,notebook_move-cursor-up"
```

For a venv, replace the `call ...activate.bat <name>` line with
`call <env>\Scripts\activate.bat` (not tried yet).

Run it with the PowerShell tool, with `run_in_background` set:

`cmd /c "<scratchpad>\start_jupyter.cmd <token> <port>"`

The long last argument makes the MCP server offer the "run one cell" tools
(below); without it only run-all is offered. The background task's output
should say `JupyterLab extension loaded from ...\envs\<name>\...`: that shows
the environment was activated.

### Steps 6 to 10 on Windows 11

The same as in Part 1, with `& "<python>"` in front and, for
`python -m jupylet.claude`, the current folder set to `<folder>` first
(`No module named jupylet.claude` in `EXPERIENCE.md`). Do not add a pause
(`sleep`) before step 6: `wait` does the waiting itself. Nothing else here is
Windows-specific: Part 1's own step 7 and step 8 already cover a hidden pane
and checking sign-in before asking for the token. One confirmed fact worth
knowing: after a restart with a new token, the page was still already signed
in (status `200`), because the sign-in cookie survives a restart.

The waiting commands ("Waiting for the person") run with the PowerShell tool
and `run_in_background` set, the same folder rule applying:
`Set-Location <folder>; & "<python>" -m jupylet.claude watch ...`. That is how
they were tested.

### Working in the notebook on Windows 11

Windows PowerShell 5.1 strips the double quotes from a JSON argument, so
`call ... '{"notebook_name": "11-spaceship"}'` fails with a JSONDecodeError.
Pass the JSON on stdin instead, in a single-quoted here-string:

```
@'
{"notebook_name": "11-spaceship", "response_format": "detailed", "limit": 0}
'@ | & "<python>" -m jupylet.claude call <port> <token> read_notebook -
```

or write it to a file in the scratchpad and pass `@<file>`. Tools without
arguments (`notebook_run-all-cells`) need neither.

Tested and working: `use_notebook` (through `attach`), `read_notebook`,
`read_cell`, `execute_code`, `insert_cell`, `overwrite_cell_source`,
`notebook_run-all-cells`, `notebook_get-selected-cell`,
`notebook_move-cursor-down`, `notebook_move-cursor-up`, `notebook_run-cell`.
`execute_cell` and `insert_execute_code_cell` are not usable (Problem 8). Not tried:
`delete_cell`, `edit_cell_source`, `move_cell`, `clear_cell_output`,
`restart_notebook`, `list_kernels`, `notebook_run-cell-and-select-next`,
`notebook_run-cell-and-insert-below`.

`insert_cell` puts a cell at the index you give (0-based) and does not run it.
`overwrite_cell_source` replaces a cell's source; the old output and execution
count stay until the cell is run. Cells you add or change are saved into the
person's notebook within seconds: remove your test cells, or tell the person
(more in `EXPERIENCE.md`).

Running one cell (needs the flag from step 5). `execute_code` runs code in the
kernel but puts nothing in a cell. To run a particular cell the way a person
would, with the current folder set to `<folder>`:

`& "<python>" -m jupylet.claude run-cell <port> <token> <cell index> "<start of its source>"`

It moves the page's selection to the cell, one cell at a time, checks that
its source starts with what you gave, and only then runs it (the wrong cell
could restart the game); it answers `True`, or why it did not run. It moves
the person's cursor. Then `read_cell` shows the execution count and the
output. (This command replaced a longer script on 2026-10-01, and is not
tested on Windows yet; the script is in git history if it fails.)

### Stopping on Windows 11

Part 3 applies: `shutdown` finds your Jupyter's processes on Windows too (the
ones whose command line carries your token: the launcher `cmd`, `jupyter`,
`jupyter-lab` and two `python`; all are yours). This replaced a manual
procedure on 2026-10-01 and is not tested on Windows yet. Run it with the
current folder set to `<folder>`. Then:

1. If it prints `still running`, list what is left with
   `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '<token>' } | Select-Object ProcessId, Name, CommandLine`,
   check each command line by eye (your own launcher and its children, with
   your token), and end each one by its id, never by name:
   `Stop-Process -Id <id> -Force`. Never stop a process without your token.
   The background task then reports `failed` (exit code 255): expected,
   because it was ended rather than asked to exit.
2. Now that no Jupyter is running, delete Jupyter's own state files (never
   a notebook), with the PowerShell tool like everything else here:
   `foreach ($f in "<folder>\examples\.jupyter_ystore.db", "<folder>\examples\.jupyter\collaboration_sessions.json") { if (Test-Path -LiteralPath $f) { Remove-Item -LiteralPath $f } }`
   Stale collaboration state is the likely cause of cells added over MCP not
   showing in the page (`EXPERIENCE.md`).
