# EXPERIENCE.md - what sessions have learned running jupylet with people

## What this file is

`CLAUDE.md` is the fixed, tested procedure, like a ROM: it does not change
during a session. This file is the writable memory next to it: what sessions
have learned about problems, workarounds and working with people, in whatever
shape is most useful. It serves macOS and Windows 11, and every entry says which
platform it belongs to and how sure it is. It is meant to keep growing on each
person's computer, from your own mistakes and from the problems you meet
there, as a person's know-how grows with practice; do not keep it short for
the sake of being short.

Read it right after `CLAUDE.md` (and `CLAUDE.0.md`, if it exists): "Read this
first", "Working with people", and the whole section for your platform; skim the
other headings. When something goes wrong, search this file for the words of the
problem before you improvise. If this file and `CLAUDE.md` disagree, `CLAUDE.md`
wins, and you write the disagreement down here. Nothing here lets you skip a
step there.

Sections: Read this first, Working with people, Any platform, macOS, Windows 11,
Retired, Unreviewed.

## How to write in it

- Start every entry with a tag: `[platform, evidence, date, versions]`.
  Platform: `macOS`, `Windows 11` or `any`. Evidence: `verified` (I ran it and
  saw the result), `seen once`, `code reading` (I read the source, did not run
  it) or `guess`.
- Write while you work, in the section where the entry belongs. Add a section
  when the material calls for it (a new platform, a new kind of problem). If you
  are unsure where an entry belongs or whether it is true, put it under
  "Unreviewed".
- The file grows. Do not delete an entry to save space. Improve it when you learn
  more (say what changed, and when). Merge entries about the same thing, so the
  file stays easy to search. Move an entry that turned out wrong or no longer
  true to "Retired", with the date and the reason, so the next session does not
  repeat the mistake. Keep "Read this first" short and current: every session
  reads it.
- Write what you saw, not what you hope. Write "unknown" when you do not know,
  and name the test that would settle it.
- Never write tokens, the names of people you help, user names or full paths.
  Use `<user>`, `<folder>`, `<env>`.
- Private notes about one machine or person go in `CLAUDE.0.md`, not here.
- **This is where a lesson goes.** If you keep any other memory of your own
  (a personal note file, anything outside this repo), check here first before
  writing to it. A lesson about running jupylet sessions or about working with
  people while doing it belongs in this file, in the section it fits, so every
  future session gets it too — not in a private place only you will read. It
  is easy to default to a habit of writing feedback to your own memory instead;
  when that happens here, it is a bug, not a style choice.

## Read this first

1. A timeout is not a failure. Look at the real state (execution counts, the
   canvas, the kernel) before you replace or restart anything.
2. Your Jupyter is the process whose command line has your token. Never trust
   old runtime files or process ids, and never touch anything that is not yours.
3. Before running or deleting something in the person's notebook, check that
   the target is what you think it is, and stop if it is not.
4. `execute_cell` hangs on both platforms. Use run-all, or the "run one cell"
   recipe (`CLAUDE.md`, Part 6).
5. Canvas shown as `Image(value=...)` text: check notebook/cell trust
   first (see "The canvas shows as text instead of a picture"). It is not
   about state files, position, the browser, or the executor.
6. Test cells you insert stay in the notebook. Remove them or say so.
7. One yes per action, one question at a time. Say how you read an unclear
   request, then do the smallest thing.
8. On Windows use the PowerShell tool, and script files instead of long
   `python -c` lines.
9. Never wait for the person inside your turn: you cannot hear them until it
   ends. Wait in the background (`CLAUDE.md`, "Waiting for the person").

## Working with people in a shared coding environment (any platform)

Written from sessions with the author, who wanted plain and honest answers.
Children need the same, put more gently (see `CLAUDE.md`).

- **Ask before you change things, and ask once.** Starting a server, editing a
  config, deleting files, adding cells: get a clear yes first. A yes covers that
  task, not the next one. Looking (reading files, checking status) needs none.
  Do not ask again what was already answered, and do not repeat back what the
  person just told you.
- **One question per message.** Several questions at once made a person say the
  session was tiring.
- **Say how you read an unclear word.** "Edit it" meant one particular cell. I
  picked the most recent one, did it, and said which cell I had picked and how
  to undo it.
- **Look before you conclude.** Compare two sources: the server's view and the
  page's view. Report what each said. Keep "verified", "code reading" and
  "guess" apart; a wrong guess stated as fact cost the most time.
- **Guard before you act in someone else's session.** Check the selected cell
  before running it; check the command line before stopping a process. A guard
  that refused once (wrong cell selected) prevented restarting the game.
- **Leave it as you found it.** Stop what you started, delete your scratch
  files, remove your test cells, and say what is left.
- **Say what you are doing during long work.** One short line. Silence looks
  like a hang.
- **When you got something wrong, say so once, plainly, fix it and go on.**
- **Do the literal task.** If a related fix seems useful, offer it in a
  sentence and wait.
- **Answer a question with words before acting on it.** A question such as
  "how would you remember this" gets an answer in words first, not a jump
  straight into actions that are supposed to speak for themselves.
- **When told to do something directly, do that thing first.** Report its
  result before anything else. If something else seems worth checking or
  fixing too, say so and ask before doing it — do not do it quietly in
  between the ask and the answer; a person waiting with no visibility into
  what is happening reads as wasted time, even when the extra work is useful.
- **A fix to your own behavior is not a fix until it is written down.**
  Correcting how you talk mid-session (dropping jargon, not narrating your
  own process) only helps this session; a future one, with none of this
  context, follows the document, not what you did here. After catching
  yourself, check whether the instructions actually say to do it that way -
  if not, the lesson does not exist yet, and the same mistake is exactly what
  a real session, facing a real student, will make.
- **Patch the whole script, not just the line someone flagged.** Fixing one
  person-facing line at a time, reactively, let small inconsistencies
  accumulate faster than they were being caught (a token shown with no
  explanation, a payoff line not actually marked as something to say, a step
  with no example at all). Once a few reactive fixes have piled up in one
  session, stop and read the entire person-facing sequence together, in the
  order a person would experience it, rather than trusting that each patch
  was locally enough.
- **A test run is only a test if you stay in character.** `[any, seen once,
  2026-09-23]` When the author asked for a trial run of `CLAUDE_SETUP.md`, I
  opened it with a note to them as the developer ("from here on I'm
  following the page, with you as the person...") and they stopped the run:
  once it starts, talk to whoever is there as a beginner, exactly as the
  page says, whoever they are. Anything for the developer goes before the
  run starts or after it ends, never in between. And after fixing something
  the author found in a test run, do not start the next run on your own:
  say it is fixed and that you are ready, and wait for them to start it.
- **Roll step by step; never present a whole plan to approve.** `[any, seen
  once, 2026-09-23]` `CLAUDE_SETUP.md` used to gather everything first and
  then ask once, in a long message listing every action, what it changes and
  why. The author called it "a long text to read like a wikipedia article":
  a beginner does not read it, so the yes means little. It now works like
  `CLAUDE.md`: one short question at the moment each change is about to
  happen, silent checks with no comment, one line before anything slow, and
  an explanation only where the person has a decision to make. Plain is not
  dumbed down: "in jp145 and a few other places" and "its own toolbox" were
  too vague. Name the real thing ("a Miniforge environment called `jp145`",
  "Miniforge's main environment, called `base`") and explain it once; the
  person will meet those names again. And news counts: that Miniforge was
  already there went unmentioned, because the page said to skip silently.
- **Waiting for the person: watch in the background, never inside your
  turn.** `[any, verified on Windows 11 with Opus 5.5, 2026-09-22]` Step 8 used
  to say "check a handful of times a few seconds apart" inside one turn. A
  message the person sends during that stretch is not seen until the turn
  ends, so a kid asking "what globe?" would get no answer, and when the
  checks ran out before the kid was done, the session just sat there. The
  person who raised it called it "not good". It is the same for anything you
  ask a person to do, not only signing in: you cannot both watch and listen
  inside one turn.

  What works (now `CLAUDE.md`, "Waiting for the person"): start a waiting
  command with `run_in_background`, end the turn, and let its finishing wake
  you. Tested in a small lesson (`7 * 6`, `print("hello")`, a typo with
  mismatched quotes, then the fix): each run woke the session within a few
  seconds, with no message from the person. A question asked mid-wait ("what
  does print do?") was answered at once while the watcher kept running.
  The 90-second timeout woke the session for a gentle check-in when nothing
  was run. The waking comes from Claude Code, not the model, so other models
  get it too. Whether they follow the pattern as reliably is not tested (the
  test would be the same lesson with Sonnet 5, picked in the app's model
  menu). Not tried on macOS. `wait-open` was tested to say `open` for an
  open notebook and to stay `timeout` for 15 seconds on the login page, and
  on 2026-09-23 end to end through a real sign-in: the person pasted the
  token, `wait-open` printed `open` and woke the session, which went on to
  attach and run-all without being told.

  Kids take their time: a 5-10 second wait is far too short. The person
  preferred a teacher-like check-in on a timeout ("How's it going? ...
  just ask") over silence, and backing off (90 seconds, 5 minutes, 10) so it
  does not nag.

## Any platform: how the tools work

These come from reading the source, so they do not depend on the operating
system. `[any, code reading and verified on Windows 11, 2026-09-22,
jupyter-mcp-server 2.2.2, jupyter_server_nbmodel 0.2.9, JupyterLab 4.6.3]`

- **`execute_cell` does not run the cell in the page.** It puts the cell (with
  its document and cell id) into the execution queue of `jupyter_server_nbmodel`
  and polls for the result. That never completed on Windows 11 (30 seconds, then
  "Execution timed out", no execution count), and `CLAUDE.md` Problem 8 says
  the same on the Mac. `execute_code` is a different path and works, but it
  puts nothing in a cell.
- **Page commands as tools.** The page registers about 433 JupyterLab commands
  with the server (the console says "Registered 433 tools"). The server offers
  only those on `allowed_jupyter_mcp_tools` (default: `notebook_run-all-cells`
  and `notebook_get-selected-cell`). A tool's name is the command id with `:`
  replaced by `_`. Add tools with the flag
  `--JupyterMCPServerExtensionApp.allowed_jupyter_mcp_tools=a,b,c` when
  starting Jupyter. Run on Windows 11 only; the flag is server-side and should
  behave the same on a Mac, which has not been tried.
- **Which plugins are switched off.** JupyterLab extensions can disable other
  plugins through `disabledExtensions` in their manifest. An extension that is
  itself disabled contributes nothing (`jupyterlab_server/config.py`, the
  "skip if the extension itself is disabled" check). Explicit entries in a
  `page_config.json` override the extensions' lists (same file, the final
  `update`). `jupyter labextension enable X` writes only when X is disabled in
  the static config (`jupyterlab/commands.py`), so for a plugin that an
  extension disabled it silently does nothing.
- **One shared notebook.** The page and the MCP tools edit the same shared
  document (`.jupyter_ystore.db` and the collaboration room). If the page's cell
  count differs from `read_notebook`'s, suspect stale state.
- **A healthy session looks like this:** `wait` says `ready`; the sign-in check
  says 200; `attach` says "Successfully activate notebook"; `read_notebook`
  and the page show the same number of cells; run-all answers `True`; the canvas
  image has the size of the app (for example 512x512); there is no
  `Image(value=` text on the page.
- **`python -m jupylet <command>` can silently run the wrong code.**
  `[any, verified, 2026-09-22]` `python -m package` puts the current directory
  first on `sys.path`, before the real installed package, even an editable
  one. A folder literally named `jupylet` sitting in that directory - the
  root of a freshly cloned or downloaded repo, for instance - is found first
  and shadows the real package, and since a repo's own root has no
  `__init__.py` (the real package is one level deeper, `jupylet/jupylet/`),
  this fails with `No module named jupylet.__main__`, not a wrong-version
  problem, which makes it confusing to diagnose from the error alone. Confirmed
  with a full, realistic clone layout, not just a toy reproduction. This is
  why `README.md`'s trust step has to run from inside `examples/`, never from
  the parent folder right after `download`/`git clone`, and why
  `CLAUDE_SETUP.md` step 8 runs `python -m jupylet` from inside
  `<code>/examples` (through its helper script). It is not specific to `is_trusted`/`trust_notebooks`/
  `download`: any future `python -m jupylet <command>` needs the same care
  about where it is documented to be run from.
- **An exit code says whether a check ran, not what it found.**
  `[any, verified, 2026-09-22]` `is_trusted` first exited non-zero whenever
  any notebook was untrusted, which is a normal, expected first-run result,
  not a failure - and it showed to whoever was watching the command run as a
  red "Failed" banner, for a check that had done exactly what it was asked.
  Fixed by splitting the two things a command can report: whether it could
  do its job (the exit code - non-zero only for a real error, a file that
  would not read, nothing found to check) and what it found (the printed
  text, `trusted` or `NOT TRUSTED` per line). A caller reads the text for the
  finding, the exit code only for whether the check itself worked. Worth
  remembering for any future `jupylet` CLI command with a normal outcome
  that is not simply success.
- **How `watch` and `wait-open` know what happened.** `[any, verified on
  Windows 11, 2026-09-22, JupyterLab 4.6.3]` `watch` reads the notebook
  kernel's `last_activity` and `execution_state` from `/api/sessions`. Any
  kernel request moves `last_activity`, a Tab completion too, so `ran` means
  "something happened", not "a cell ran". Over a few idle minutes with the
  page open, nothing moved it by itself. To find which cell they ran, compare
  execution counts before and after: the highest count is misleading, because
  cells keep counts from earlier kernels (two cells showed `16` next to a
  fresh `1`). `wait-open` waits for a session on the notebook to exist: with
  the browser signed out and on the login page, none appeared. It looks the
  kernel up through the session on every check, so it keeps working after
  `replace-kernel`.
- **Why the token is only 8 characters.** `[any, reasoning, not tested as
  an attack, 2026-09-23]` `CLAUDE.md` step 3 makes a 32-bit token
  (`token_hex(4)`), because the person has to paste or type it. That is
  enough for a server that listens on this computer alone. Other computers
  cannot reach it. A web page in the browser cannot read Jupyter's answers,
  and Jupyter refuses requests not addressed to localhost, so a page cannot
  try tokens. A program on the same computer does not need to guess: the
  token is in Jupyter's command line. A short token would only matter if
  Jupyter listened on the network (`--ip`), which the steps never do.

## macOS

Nothing new here yet: what was learned on the Mac is in `CLAUDE.md`, Part 5.
Things a Mac session could check, because the Windows 11 session found them but
could not test them there:

- Does the canvas show when cells are run by hand, with `jupyter-mcp-server`
  installed? (On Windows 11 it did not, without the config in the entry below.)
- Does the "run one cell" recipe work with the allowlist flag?
- Do `insert_cell` and `overwrite_cell_source` show up in the page at once?

## Windows 11

Seen on one Windows 11 Home machine: Miniforge in `C:\Users\<user>\miniforge3`,
JupyterLab 4.6.3, jupyter-mcp-server 2.2.2, jupyter_server_nbmodel 0.2.9,
ipywidgets 8.1.9, driven from Claude Code Desktop with the PowerShell tool. That
machine also had Git, so the Bash tool existed; a child's PC may not have it,
so everything was done in PowerShell on purpose.

### The canvas shows as text instead of a picture

`[any platform, verified, 2026-09-22]` The cell output is text like
`Image(value=b'\xff\xd8...`, and `ipywidgets.IntSlider()` prints
`IntSlider(value=0)`, or `app.get_logging_widget()` prints
`Output(layout=Layout(...))`. The game runs fine underneath (`app.is_running`
is `True`, VR worked); only the page does not draw the widget.

Cause, confirmed: the notebook (or the specific cell) is not trusted. Jupyter
does not render rich/interactive outputs (including ipywidgets' widget-view
mimetype) for untrusted content, and this is a per-cell check, not only a
whole-document one: a cell whose content/output lineage traces back to a
notebook that was never trusted can keep failing even after it is copied,
pasted, or moved into a different, trusted notebook, while a cell freshly
typed or copied from an always-trusted notebook works right next to it in the
very same document. This produced a long chain of false leads before it was
found (below), because none of them are the actual cause:

- Deleting `.jupyter_ystore.db` / `collaboration_sessions.json` (see "Sessions,
  stopping and state files"): no effect on trust.
- Cell position (which of several near-simultaneous `app.run()` calls comes
  first): no effect; moving the untrusted cell to a different position moved
  the failure with it.
- The browser: reproduced identically in two unrelated browser profiles.
- The specific cell's own stored metadata, output metadata, and `model_id`:
  verified byte-for-byte identical to working cells (the widget model is
  genuinely live and correct; the page simply refuses to render it).
- `jupyter_server_nbmodel` disabling JupyterLab's normal cell executor: this
  was the earlier, wrong theory below, now retired. A working environment with
  that executor still disabled (the default) rendered every widget correctly
  once the notebook was trusted.

Server log line to watch for, printed repeatedly while this is happening:
`Notebook <name>.ipynb is not trusted`. It was there from the very first
Windows 11 session that hit this symptom and was not connected to the cause
until much later.

Fix: trust the notebook (JupyterLab shows a banner/menu action for this, or it
happens automatically once the person opens and saves it normally). Once
trusted, cells that were already rendering as text render correctly without
re-running them.

Retired theory, superseded by the above: `[Windows 11, guess, 2026-09-22]`
disabling `jupyter_server_nbmodel`'s frontend (a `page_config.json` override,
see "Which plugins are switched off") appeared to fix this on one environment,
but was never isolated from a concurrent notebook-trust change, and a later,
controlled test (fresh environment, executor still disabled, trust fixed
instead) rendered every widget with no `page_config.json` override at all.
Left here so a future session does not try it again expecting it to matter.

### Run-all says "Timeout waiting for result", but the notebook ran

`[Windows 11, seen once, 2026-09-22]` `notebook_run-all-cells` failed after about
30 seconds, yet the kernel was idle, the cells had new execution counts and the
canvas was drawn. On the same notebook run-all had answered `True` at once
before. Cause unknown (a slow first run is a guess). Before replacing the kernel
(`CLAUDE.md` Problem 1) look: `read_cell` on `app.run()` for its execution
count, `execute_code` with `print(app.is_running)`, and the page for the canvas.
Replace the kernel only if nothing ran, because that destroys a running game.

### Cells added over MCP do not appear in the page

`[Windows 11, seen once; cause not proven, 2026-09-22]` `insert_cell` said it
worked and `read_notebook` showed one more cell, but the page kept the old
count for good. Compare `read_notebook` with the page's count (in the page:
`document.querySelectorAll('.jp-Notebook .jp-Cell').length`).

It happened on a server that started with a `.jupyter_ystore.db` left from
earlier sessions (the log said the file was "out-of-sync with the ystore"). On a
fresh server, after deleting the state files, inserted cells appeared in the
page within seconds, before and after running the notebook. So stale state is
the likely cause, but the failing notebook was a different one
(`12-spaceship-3d`), so that is a guess. `overwrite_cell_source` also showed up
in the page at once on the fresh server.

Do: stop adding cells, tell the person, stop Jupyter, delete the state files,
start again. Inserted cells are saved into the notebook file within seconds; the
old copy in the page did not overwrite the file.

### Starting Jupyter

`[Windows 11, verified, 2026-09-22]` The activation form that worked is a `.cmd`
file that calls `<miniforge>\condabin\activate.bat <name>`, then `cd /d`, then
`jupyter lab ...`, run in the background as `cmd /c "<file> <token>"`. The
background output showed JupyterLab loading from `envs\<name>`. Not tried:
`conda.bat run -n <name> --no-capture-output`, and `conda-hook.ps1`. Do not use
`conda init` (PowerShell profile scripts are blocked by default).

### `CLAUDE_SETUP.md` with Miniforge already installed

`[Windows 11, verified, 2026-09-23, Miniforge 26.7.2 with Python 3.14 in
base]` One full run, with the code from a local `git archive` tarball
(`download` given a `file://` URL) instead of GitHub:
- `conda create -y -p <miniforge>\envs\jupylet --override-channels -c
  conda-forge python=3.13 moderngl glcontext` worked, about a minute, and the
  environment is listed by name (`conda env list`), so `conda activate
  jupylet` works even though it was created with `-p`.
- `<env python> -m pip install -e <code>` worked without activating the
  environment, a few minutes.
- The helper's `download`, `overrides`, `jupylet ... trust_notebooks` /
  `is_trusted` and `prompt` all gave the expected answers.
- The handover into `CLAUDE.md` Part 1 worked: `find-env` listed the new
  environment first, and the game canvas showed as a 512x512 picture.

Not tried yet: installing Miniforge (the silent installer), updating an old
one, and a computer with another conda.

### PowerShell quoting

`[Windows 11, verified, 2026-09-22]` Windows PowerShell 5.1 strips double quotes
inside an argument it passes on: `python -c "...'x' ... "y" ..."` failed with
`'(' was never closed`. Single quotes inside `-c "..."` are fine. For anything
longer write a `.py` or `.cmd` file into the scratchpad. Scripts started by file
path do not search the current folder for modules: put
`sys.path.insert(0, r'<folder>')` at the top.

### `No module named jupylet.claude`

`[Windows 11, verified, 2026-09-22]` `jupylet` was installed with a regular
`pip install` (not `-e`), so the copy in the environment has no `claude.py`.
`python -m jupylet.claude` works from the folder that has the code and not from
its `examples` folder. The PowerShell tool keeps the current folder between
commands.

### Old servers and recycled process ids

`[Windows 11, verified, 2026-09-22]` `jupyter server list` and the
`jpserver-*.json` files in `%APPDATA%\jupyter\runtime` name two dozen servers
that no longer run, with their tokens. A process id in such a file can belong to
another program now (once it was the Claude app itself). Do not use those tokens
or ids. Your server is the set of processes whose command line contains your
token (`Get-CimInstance Win32_Process`): the launcher `cmd`, `jupyter`,
`jupyter-lab` and two `python`. All are yours; nothing else is.

### Sessions, stopping and state files

`[Windows 11, verified, 2026-09-22]` After `attach`, `/api/sessions` lists the
notebook twice with the same kernel (the page's session and the MCP server's).
Normal; ending both is part of stopping. `claude.py shutdown` was not run:
`_pids` uses `ps` and the force-stop uses `SIGKILL`, and Windows has neither, so
it would say `stopped` without checking. The stopping steps in `CLAUDE.md`
(Part 6) end sessions and kernels over HTTP, ask the server to shut down, then
check the token's processes and the port. A clean stop ended the background task
with exit code 0.

Jupyter's own state in the examples folder: `.jupyter_ystore.db` and
`.jupyter\collaboration_sessions.json`. They are recreated on the next start.
Delete them after stopping, when no Jupyter runs. `%APPDATA%\jupyter\file_id_manager.db`
and the old runtime files were left alone. `claude.py cleanup` only uses `glob`
and `shutil` and looks portable, but it was not run on Windows.

### The stop script says "port closed: True" but the process is still alive

`[Windows 11, seen once, 2026-09-22]` The stop script (`CLAUDE.md` Part 6,
"Stopping on Windows 11") ended sessions and kernels, requested `/api/shutdown`,
and printed `port closed: True` (`connect_ex` really did stop returning `0`).
The background task's log ended at `[I ...] YDocExtension] Deleting all rooms.`
with nothing after it — no further extension-shutdown lines, no process exit.
15+ seconds later, `Get-CimInstance` still listed all five of the launcher's
processes (`cmd.exe`, `jupyter.exe`, two `python.exe`, `jupyter-lab.exe`) alive
under the session's token, and the background task had not reported completion.
This is the same shape as `CLAUDE.md` Problem 10 (port stops answering, process
lingers), but that problem's fix (`claude.py shutdown`'s own force-stop) is
`ps`/`SIGKILL`-based and explicitly not used on Windows, and the Windows
stopping steps in Part 6 have no force-kill step of their own — they only say
to tell the person and stop if a process is left. Not tried: waiting longer
(a minute or more) before concluding it is truly stuck.

First try: told the person a process was left and stopped there, as Part 6
literally says. The person (an adult, in this case the project's own author)
pointed out the flaw: `CLAUDE.md`'s "tell the person and stop" line was
written assuming the person can do something about it, but most people this
tool is for - kids, beginners - have no way to find or end a stuck process
themselves, and "check your taskbar" leaves them stuck with a problem and no
real next step, which is exactly what rule 3 says never to do. So a stuck
Windows process after the normal stop is a case the session should resolve
itself, the same way it already resolves a stuck kernel (`replace-kernel`)
without waiting on the person - not a case to hand back to them. This is a
disagreement with `CLAUDE.md` as literally written (which this file is
supposed to record, per its own header); `CLAUDE.md` cannot be edited by a
session, so until it is updated by hand, treat this entry as the standing
correction and follow it instead of the literal "tell the person and stop"
line, for this specific situation only.

Do instead, the next time the Windows stop script reports `port closed: True`
(or otherwise finishes) but `Get-CimInstance` still lists processes carrying
the session's token a few seconds later: `[Windows 11, verified, 2026-09-22]`
list every process whose command line still carries the session's token with
`Get-CimInstance Win32_Process | Select ProcessId, Name, CommandLine`, check
each one's command line by eye against the exact launch command actually used
(confirm it is this session's own launcher and its children, never a process
found only by name), then `Stop-Process -Id <id> -Force` on each matching PID
by id (never by name - matching by name risks an unrelated process). All five
of the launcher's processes (`cmd.exe`, `jupyter.exe`, two `python.exe`,
`jupyter-lab.exe`) ended at once this way; the port closed (`connect_ex` back
to `10061`) immediately after. The background task the launcher ran in then
reported `failed` (exit code 255) instead of a clean exit - expected and fine
since it was killed rather than asked to exit; do not mistake that `failed`
status for something having gone wrong. The Claude Code Desktop app's own
PowerShell process did not appear in the token-matched list, so this targeted
approach never risked it. Still tell the person afterward, plainly and
without jargon, that a leftover program had to be closed and that it is done
- just do not stop and wait on them to act first.

### Other small facts

`[Windows 11, verified, 2026-09-22]`

- The browser pane is often hidden after `preview_start`; ask the person to click
  the globe icon.
- After a restart with a new token the page was already signed in (status 200):
  the cookie survives, and only a 403 needs the token typed by the person.
- A free port prints `10061` from `connect_ex` (connection refused).
- After run-all the selection is on the last cell. The run-one-cell script moves
  both up and down and checks the source each step; 33 moves took 1.6 seconds.
- The MCP tool call waits about 30 seconds for the page; run-all can be slower
  (see above).
- Not tried on Windows 11: Windows 10, ARM, a folder with spaces or in OneDrive,
  a fresh Miniforge install, `replace-kernel`, `delete_cell`, `edit_cell_source`,
  `move_cell`, `clear_cell_output`, `restart_notebook`.

## Retired

Entries that turned out wrong or no longer true, each with the date and why.
Keep them: they show what has already been tried.

- `[Windows 11, 2026-09-22]` The `page_config.json` / disable-nbmodel fix for
  "the canvas shows as text instead of a picture": kept inline under that
  entry (Windows 11 section) rather than moved here, since the correct cause
  belongs right next to it. See that entry for the retraction.

## Unreviewed

Entries you are unsure about (where they belong, or whether they are true), with
the tag from "How to write in it". Move them up, or to Retired, when you know.
