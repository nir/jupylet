# EXPERIENCE.md - Claude's notes from helping people with Jupylet

This file is written only for Claude. In it, Claude records what it learned
while helping people use Jupylet: problems it met on real computers and how
it solved them, and what helps when working with beginners. It is technical,
and a person does not need to read it. It never holds names or other
personal details.

From here on, "you" means Claude.

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
- Never write tokens, the names of people you help, user names, or paths
  that contain a user name. Use `<user>`, `<home>`, `<folder>`, `<env>`.
- Notes that only fit this one computer go in `CLAUDE.0.md` (see
  `CLAUDE.md`), not here.
- **This is where a lesson goes.** Claude Code also has its own memory,
  outside this folder. A lesson about running Jupylet or helping people with
  it belongs here instead, in the section it fits, so that every future
  session gets it. It is easy to default to a habit of writing feedback to
  Claude Code's own memory instead; when that happens here, it is a bug, not
  a style choice.

## Read this first

1. A timeout is not a failure. Look at the real state (execution counts, the
   canvas, the kernel) before you replace or restart anything.
2. Your Jupyter is the process whose command line has your token. Never trust
   old runtime files or process ids, and never touch anything that is not yours.
3. Before running or deleting something in the person's notebook, check that
   the target is what you think it is, and stop if it is not.
4. `execute_cell` needs nbmodel, which `CLAUDE.md` step 4 turns off. Use
   run-all, or `claude.py run-cell` (`CLAUDE.md`, Part 6).
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
10. Look for the simplest, most elegant solution. See "Simple beats clever"
   below.

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
- **A test run is only a test if you stay in character.** `[any, seen
  twice, 2026-09-23]` Once a trial run of `CLAUDE_SETUP.md` starts, talk to
  whoever is there as a beginner, exactly as the page says, even if they are
  the author or the computer looks like a developer's. Notes for the
  developer go before the run or after it, never in between; after fixing
  something the author found, say it is ready and let them start the next
  run. A session that broke character also opened with a summary of the
  page instead of the hello, invented an in-between choice the page never
  offered ("install Miniforge but leave Terminal alone", which would have
  broken `conda activate` later), dropped the page's explanations, and ended
  with a list of paths; the page now says each of these outright. One thing
  it did better, asking about Miniforge and Terminal in one question so a no
  leaves the computer untouched, became the page's way.
- **Roll step by step; never present a whole plan to approve.** `[any, seen
  once, 2026-09-23]` A long message listing every action up front read "like
  a wikipedia article": a beginner does not read it, so the yes means little.
  Ask one short question at the moment each change is about to happen. Plain
  is not dumbed down: name the real thing ("a Miniforge environment called
  `jp145`", "Miniforge's main environment, called `base`") and explain it
  once, since the person will meet those names again; news counts (say that
  Miniforge was already there). Be a teacher as well as a guide: say how to
  open a window they have never used. Keep them with you during long
  installs with a sentence on what you are doing and why, never the
  instructions themselves ("go to step 9"). When one name means two things
  (the environment `jupylet2`, the folder `jupylet2`, which the app calls a
  workspace), say which every time. Short technical status lines and loose,
  human wording ("same-ish") were fine: clear and warm beats dry and formal.
- **Waiting for the person: watch in the background, never inside your
  turn.** `[any, verified on Windows 11 with Opus 5.5, 2026-09-22]` Checking
  again and again inside one turn meant a kid's question ("what globe?") went
  unheard until the checks ran out. Now in `CLAUDE.md`, "Waiting for the
  person", and tested: each run woke the session within seconds, a question
  asked mid-wait was answered at once, and the timeout gave a gentle
  check-in. Kids take their time, so waits are long and back off (90 s, 5
  min, 10 min). Start the waiter first and make the instruction the last
  thing you say: with the order reversed, sessions kept adding "I'll wait for
  you" after it `[Windows 11, seen twice, 2026-09-23]`. Not tried with other
  models than Opus 5.5.
- **Simple beats clever.** `[any platform, recurring, Opus 5.5, 2026-09-25]`
  Claude's first proposals and code tend to be more complex than the problem
  needs, with extra state, flags, handles or layers where a few plain lines
  would do it more elegantly. With the author, many designs had to be
  simplified substantially before they went into jupylet. A beginner cannot
  do that simplifying, and cannot find the subtle bugs, such as timing
  errors, that extra machinery brings in. So avoid creating the complexity in
  the first place. Before writing code, look for the simplest solution that
  fits jupylet's existing ideas, and prefer it. When moving or generalizing
  code that works, keep its logic unless there is a reason to change it.
  Extra machinery is justified by a real problem. So when you suspect one,
  check that it exists (measure, read the code, try it) before building a
  fix, and say what you found. When the person asks for something simpler,
  cut; don't restructure.

- **Understand an explanation fully before you give it, then give only what
  it is for.** `[any platform, recurring, Opus 5.5, 2026-09-28]` Writing a
  lesson with the author, most of the rounds went into explanations that
  sounded right but were not: a made-up cause, a claim about a real circuit
  when the equations described only the code, two opposite effects presented
  as one. Fluent wording hid the gaps; the author found each by asking "why?"
  and "is that true of this, exactly?". A child or a student cannot do that
  checking, and will simply learn the wrong thing. So before explaining, lay
  the explanation out as a plain chain of claims and question each one as a
  sharp beginner would: what exactly is it about (the code, the math, a real
  device), how do I know (computed, read, or assumed), why is it so, does it
  follow, and has the person been told everything it rests on. Compute or
  drop what fails, before writing a word. Then, separately, decide what the
  explanation is for, and give only the detail that serves it: understanding
  every step yourself does not mean telling every step. A short answer that
  names the causes and their effect usually teaches better than a full
  derivation. Rule lists ("one idea per sentence", "change the least") do not
  replace this; followed literally, they make new problems.
- **True is not the same as understandable: write from where the learner
  stands.** `[any platform, recurring, Opus 5.5, 2026-09-29]` A lesson
  section whose every claim had been checked still took many rounds with its
  author. The problem was not an error but a point of view: it was written
  from inside the writer's own understanding, so it only had to make sense
  to someone who already knew. It jumped to the mechanism in compact
  notation, used words and symbols the reader had never met, and skipped the
  questions any newcomer would ask, which the author then had to ask one by
  one ("what is that 1?", "why may we write it like that?", "why does the
  subtraction work if the output is shifted?"). The writer had never asked
  them, because the writer never stood where the reader stands.
  The remedy is a way of thinking, not a procedure: before and while
  writing, actually take the place of someone who does not know, as a good
  teacher sees the question on a student's face before it is asked. Walk
  the path from what they already have to the new idea yourself, as if for
  the first time, and notice where you would stumble, wonder or lose
  interest. What to explain, in what order, and how much, then follows from
  that. Beware of turning this into a fixed recipe (a set sequence of
  questions, a template for every section, "define every term first"):
  followed mechanically, a recipe produces its own bad text, stiff and
  formulaic, answering questions nobody asked, and it replaces the one thing
  that matters, looking through the reader's eyes, with ticking boxes.

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
- **`read_cell` returns the outputs too, even with `include_outputs` false.**
  `[macOS, verified, Opus 5.5, 2026-09-26]` Its text is the source followed by
  the cell's outputs as text (for example `<IPython.core.display.Image
  object>`, printed lines, a traceback). A script that read a cell with it,
  changed a word and wrote it back with `overwrite_cell_source` wrote that
  output text into the source of every cell that had been run, which then
  failed with a `SyntaxError`. To edit a cell's source, take the source from
  the saved `.ipynb` on disk (read only; Jupyter saves it within seconds), or
  only edit cells that have never been run, and check the result.
- **Which jupylet does a notebook import?** `[macOS, seen once, Opus 5.5,
  2026-09-26]` The environment can have jupylet installed in editable mode
  from a different checkout than the one you are working in. The example
  notebooks that need the repo's own code start with
  `sys.path.insert(0, os.path.abspath('./..'))`; a new example notebook needs
  that line too, or it imports the other checkout and fails on anything new
  (`cannot import name ...`). Check with
  `python -c "import jupylet; print(jupylet.__file__)"`. When testing a
  notebook outside Jupyter (for example with `nbconvert --execute`), do not
  set `PYTHONPATH` to the repo: that hides exactly this mistake.
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
- **Why the token is only 8 characters.** `[any, reasoning, 2026-09-23]` `CLAUDE.md` step 3 makes a 32-bit token
  (`token_hex(4)`), because the person has to paste or type it. That is
  enough for a server that listens on this computer alone. Other computers
  cannot reach it. A web page in the browser cannot read Jupyter's answers,
  and Jupyter refuses requests not addressed to localhost, so a page cannot
  try tokens. A program on the same computer does not need to guess: the
  token is in Jupyter's command line. A short token would only matter if
  Jupyter listened on the network (`--ip`), which the steps never do.
- **`read_notebook` needs `notebook_name`.** `[any, verified on macOS,
  2026-09-23, jupyter-mcp-server 2.2.2]` Even after `attach` (`use_notebook`),
  calling it with only `response_format` and `limit` failed with "Field
  required ... notebook_name". `CLAUDE.md` now gives the full arguments.

## macOS

What was learned on the Mac before 2026-09-23 is in `CLAUDE.md`, Part 5.

- **Test code in the person's kernel can leave side effects.** `[macOS, seen
  once, Opus 5.5, 2026-09-25]` Calling jupylet's `get_logging_widget()` in a
  test via `execute_code` added a handler to the root logger and set the stderr
  handler to ERROR, so the person's later `logger.info` messages never showed.
  Test in a separate process where possible; if it must be the kernel, undo
  global changes (loggers, settings, tempo) and say so.
- **"It does not run cells": the page is stuck, not the kernel.** `[macOS, seen
  once, Opus 5.5, 2026-09-24, JupyterLab 4.6.4]` After a long session with many
  cells added over MCP, ipywidgets and a sonic live loop, a cell showed `[*]`
  while the server said the kernel was idle and `execute_code` ran fine; every
  newer cell queued behind it. The console had repeated "Cannot read properties
  of null (reading 'stateChanged')" and "CodeMirrorEditor already set". Check
  the kernel first (sessions API, `execute_code`), then reload the page with
  `navigate` to the notebook's URL: the kernel and its variables survive. That
  reload did NOT fix it: the `[*]` survived the reload (it is
  in the shared document), and Jupyter's log showed no "Executed cell" from
  `jupyter_server_nb_model` since the stuck cell, i.e. the server-side
  execution queue was stuck, not the page (the stuck cell was `app.stop(...)`
  on a sonic live loop, after a kernel interrupt). What fixed it: `replace-kernel`
  (with the person's yes; the variables are lost), then `unuse_notebook` and
  `attach`. The queued cell ran at once on the new kernel. The old `[*]` mark
  stays on the stuck cell until it is run again. An earlier `Error saving file
  ... IndexError: Array index out of range` in `jupyter_ydoc` may be related.
  The cause was found later: the nbmodel hang, in "A cell stays `[*]` forever
  while the kernel is idle" below (2026-10-01).
- **After the person restarts or replaces their kernel, `attach` keeps the old
  one.** `[macOS, verified, Opus 5.5, 2026-09-24, jupyter-mcp-server 2.2.2]`
  The person restarted the notebook's kernel a few times from the page; the
  session had a new kernel id, but `attach` answered "already connected ... runs
  on execution backend '<old id>' ... not applied". Fix that worked:
  `call ... unuse_notebook '{"notebook_name": "<name>"}'`, then `attach` again;
  it then connected to the new kernel and `execute_code` saw the person's
  variables.
- **Your own `execute_code` wakes a running `watch`.** `[macOS, verified, Opus
  5.5, 2026-09-24]` While a `watch` waited for the person, each of my own test
  runs in their kernel (`execute_code`) ended it with `ran`, and no cell count
  had changed. That's harmless if you re-read the counts, as `CLAUDE.md` says,
  but it's noise. Better: stop the waiting command (`TaskStop`) before your own
  kernel runs, then start it again afterwards without `<since>`. With a
  developer who also asks questions between runs, the 90 s / 5 / 10 min
  back-off check-ins aren't wanted: they are actively in the conversation.

- **`CLAUDE_SETUP.md` on a Mac with Miniconda.** `[macOS, verified, Sonnet
  5, 2026-09-23, Apple chip, Miniforge 26.7.2-0]` A full run from the
  `claude` branch on GitHub, with Miniconda in `/opt/miniconda3` set up in
  `.zshrc` and a Jupylet folder already in `~/jupylet`. Miniforge installed
  into `~/miniforge3` in about 20 seconds, `conda init zsh` switched
  Terminal from Miniconda to Miniforge (backup kept), the environment `jupylet` took 10
  seconds, the code went into `~/jupylet2`, `pip install -e` took 40
  seconds, and the Terminal check (`prompt`) said `ok`. Then the handover:
  sign-in noticed by `wait-open` 25 seconds after the token was given,
  attach, run-all, the game, and a clean `shutdown` (`stopped`) after the
  page was closed first. About 12 minutes from the first message to the
  game, answers included.

  One thing failed: the Miniforge installer refused to run, with `Please run
  using "bash"/"dash"/"sh"/"zsh", but not "." or "source".`, although it was
  run with `bash`. The installer (built by constructor 3.16.1) first checks
  `echo "$0" | grep '\.sh$'`, so its file name must end in `.sh`, and the
  page downloaded it to a plain `mktemp` name. The session improvised a
  `.sh` name and it worked. Fixed in `CLAUDE_SETUP.md` step 4 (install and
  `-u` update): the file is now `Miniforge3.sh` in a `mktemp -d` folder.
  That exact command is not yet run on a Mac; the `.sh` name is what
  mattered.
- **`overwrite_cell_source` works on macOS.** `[macOS, verified, 2026-09-23,
  jupyter-mcp-server 2.2.2, JupyterLab 4.6.4]` The diff it reports matched
  a `read_cell` afterwards, and the person saw the new text in the page.

Things a Mac session could still check, because the Windows 11 session found
them but could not test them there:

- Does the canvas show when cells are run by hand, with `jupyter-mcp-server`
  installed? (On Windows 11 it did not, without the config in the entry below.)
- Does the "run one cell" recipe work with the allowlist flag? Yes:
  `[macOS, verified, Opus 5.5, 2026-10-01]` with nbmodel off and the flag in
  the start command, selecting a cell and `notebook_run-cell` ran it, a
  syntax error included.
- Does `insert_cell` show up in the page at once? Yes: `[macOS, verified,
  Opus 5.5, 2026-09-23]` 17 `insert_cell` (index 0 and -1) and two
  `overwrite_cell_source` calls into a notebook the person had just created
  in the page all showed there within seconds (checked by counting
  `.jp-Cell` elements with `javascript_tool`; `find` does not see cell
  text, because the editor is not in the accessibility tree). `attach`
  works for any notebook name, not only `11-spaceship.ipynb`.

### How some of `CLAUDE.md`'s Problems were found

`[macOS, 2026-09-22 to 2026-09-24]` Moved here from `CLAUDE.md` Part 5 on
2026-10-01, which now keeps only what to do.
- Problem 1 (run-all timeout): seen every time after Restart Kernel (same
  kernel id), never with a fresh kernel; hiding the browser pane and reloading
  the page did not matter. With nbmodel off, `restart_notebook` then run-all
  worked once (2026-10-01).
- Problem 10 (not stopping cleanly): the server used to stop answering while
  its process stayed alive for minutes, the log ending at "Kernel shutdown",
  whenever kernels were still running at shutdown (a game in a notebook
  kernel, or a kernel with no notebook). `shutdown` now ends sessions and
  kernels first, and the server then exits within a second or two.
- Problem 13 (`.mcp.json`): an `.mcp.json` with the stdio helper
  `jupyter-mcp-server` connected once, and after the session was restarted
  while Jupyter was down, failed for good ("connection timed out after
  30000ms"); a session reads `.mcp.json` only when it starts. An HTTP entry
  never connected. Neither offered the run-all tool natively.

### A cell stays `[*]` forever while the kernel is idle

`[macOS, seen 4 times, fix verified once, Opus 5.5, 2026-09-25]` After a
live loop or a self-refreshing widget had been running for a while with no
cell run, the next cell (always the "stop" cell) hung at `[*]`, with the
cells after it queued. The kernel had run it (it is in `In`) and was idle.
`GET /api/kernels/<id>/execute` showed the request `running`, and
`DELETE .../requests/<rid>` did not help. Likely cause (a strong guess, no
clean repro yet): `jupyter_server_nbmodel` keeps one kernel client open and
reads IOPub only while a cell runs. Background output piles up in its
ZeroMQ receive queue (1000 messages by default), newer messages are
dropped, and so is that cell's `idle` status, which `execute_interactive`
waits for with no timeout.
Fix without losing variables: have the kernel send the missing `idle`. The
request id is `<session>_<server pid>_<n>`. Tasks and `call_later` handles
started from notebook cells carry their cell's header in
`kernel._shell_parent` (`task.get_context()[var]`, `handle._context[var]`),
which gives the session, the pid and a recent `n`. Then, with
`execute_code`, call `kernel.session.send(kernel.iopub_socket, "status",
{"execution_state": "idle"}, parent={"header": h}, ident=kernel._topic("status"))`
for a range of `n`. The executor finished the cell at once, because the
reply had been waiting on the shell channel. Mistake made once: do not send
`idle` for numbers past the stuck one. The executor keeps those, so every
later cell up to that number finishes early and shows no output (the log
says `outputs=0`). To fix it, use up the numbers by posting `{"code": "pass"}`
to `POST /api/kernels/<id>/execute` once per number; after that, output came
back (verified).
Otherwise `replace-kernel` fixes it (variables are lost).

`[macOS, verified, Opus 5.5, 2026-10-01, JupyterLab 4.6.4,
jupyter_server_nbmodel 0.2.9]` Seen again with a jupylet `Panel` displayed:
its 0.5s refresh, while a live loop moved a knob, gave about 8 IOPub messages
a second (slider `update`, the page's `echo_update`, busy/idle). Related
symptom: with nbmodel, output a thread prints after its cell finished never
shows in the page (plain JupyterLab without nbmodel shows it).
Without nbmodel, everything we use still works (verified): start Jupyter with
`JUPYTER_CONFIG_PATH=<dir>`, where `<dir>/jupyter_server_config.json` has
`{"ServerApp": {"jpserver_extensions": {"jupyter_server_nbmodel": false}}}`
and `<dir>/labconfig/page_config.json` has
`{"disabledExtensions": {"@datalayer/jupyter-server-nbmodel": true}}`. The
page's `serverSideExecution` is then `false`, and the collaboration
extension's cell executor falls back to running cells in the page. Worked:
`attach`, `read_notebook`, `read_cell`, `insert_cell`, `edit_cell_source`,
`overwrite_cell_source`, `move_cell`, `delete_cell`, `clear_cell_output`,
`execute_code`, `restart_notebook`, run-all (also right after
`restart_notebook`), thread output live in the page, and one cell with
`notebook_run-cell` (needs `allowed_jupyter_mcp_tools`, as on Windows).
Lost: only `execute_cell` and `insert_execute_code_cell` (error "extension
not found"), which hang anyway (`CLAUDE.md` Problem 8). Watch: run-all goes
to whichever page the tools pick; a page loaded before the restart still uses
the server executor and gets 404s, so reload or close old pages.

### Overwriting a cell by index hit the person's new cells

`[macOS, seen once, Opus 5.5, 2026-09-25]` Cell indices shift whenever the
person inserts or deletes cells, even between two of your own calls. A batch
of `overwrite_cell_source` calls, using indices read a few minutes earlier,
replaced three cells the person had just inserted. The script printed each
cell's first line but did not stop when it didn't match. Before each
overwrite, read the cell and check its content (or its `id` in the saved
`.ipynb`), and stop if it isn't the expected cell. Recovery: the
collaboration store `examples/.jupyter_ystore.db` (SQLite, table `yupdates`,
per-notebook `path`) holds every edit. Open it read-only, replay the updates
in `rowid` order into a `pycrdt.Doc` (`doc.get('cells', type=Array)`), and
keep each cell's source as it changes: this recovered the overwritten text.

### Rewriting a section: address cells by id, move them, and check the saved file

`[macOS, verified, Opus 5.5, 2026-09-27, jupyter-mcp-server as installed]`
`overwrite_cell_source`, `clear_cell_output` and `move_cell` accept a
`cell_id` (the `id` field in the saved `.ipynb`) instead of an index, which
stays true while cells are inserted around it. To rewrite and reorder a whole
section (55 cells in, 74 out), this worked in one pass: first overwrite every
changed source by id; then walk the target order, calling `insert_cell` (by
index) for new cells and `move_cell(source_index, target_index)` for existing
ones, while keeping a local list of ids in step with every call (`move_cell`
pops and inserts: the cell ends up at `target_index`). Moving an existing
cell keeps its output with it, while overwriting a code cell in another
position would leave a stale output under the wrong code. Check afterwards
against the saved file, which the collaboration extension writes within a few
seconds. Run every new code cell first in a separate Python, with
`sounddevice.play` replaced by a stub, so nothing plays aloud.
`delete_cell` takes `cell_ids_to_delete` (a list), not `cell_id` or
`cell_ids`. Two edits to the same cell in one script must build on each
other: overwriting twice from the same saved copy silently undid the first
edit (seen 2026-09-27).

### Markdown in notebook cells: boxes, code blocks and math

`[macOS, verified, Opus 5.5, 2026-09-27, JupyterLab 4.6]` Rendered in
JupyterLab: a Markdown blockquote (`>` on each line) shows as an indented
block with a gray bar; `<div class="alert alert-block alert-info">` as a teal
box whose tinted text clashes with inline code; a `<div style="...">` keeps
its inline background style; `<details>` collapses; GitHub's `> [!NOTE]` is
not supported (it shows the literal text). Inside a blockquote, a fenced code
block gets no top or bottom margin (outside it gets 24px), so the next
paragraph sticks to it. `$$...$$` math keeps its spacing there, but a
multi-line `$$` block inside a blockquote breaks (the `>` marks end up inside
the equation): write it on one line, e.g.
`> $$\begin{aligned} a &= b \\ c &= d \end{aligned}$$`. To see how a cell
renders without touching the person's page, write a throwaway notebook, open
it in a background tab (`tabs_create`), click "No Kernel", look, then close
the tab and delete the file.

### The "File Changed" dialog on almost every save

`[macOS, code reading and the server log, Opus 5.5, 2026-09-27, JupyterLab
4.6]` With real-time collaboration on, the server keeps each open notebook as
a shared document and writes it to disk by itself a few seconds after each
change (the log shows `YDocExtension] Saving file: <notebook>` again and
again). When the person presses Ctrl+S, the page compares the file's time on
disk with its own last save, finds the server's newer write, and asks
"Overwrite or Revert". Both hold the same document, so Overwrite loses
nothing, and saving by hand is not needed at all. Unknown: whether turning
off the page's own autosave setting makes the dialog go away.

### Rewinding the conversation stops Jupyter

`[macOS, seen once, Opus 5.5, 2026-09-25]` Jupyter runs as a background
Bash task of the Claude session. When the person rewound to an earlier
question in the Claude app, the session restarted and its background tasks
ended, so Jupyter shut down cleanly with its kernel ("Shutting down 1
kernel" at the end of its log). The next message then reported the task as
stopped. The notebook was saved, but the kernel's variables were lost. Fix:
ask, then start Jupyter again (step 5) with the same token, so the page's
sign-in still works, and `attach`. Worth telling a person before they
rewind while a kernel holds work they care about.

## Windows 11

Seen on one Windows 11 Home machine: Miniforge in `C:\Users\<user>\miniforge3`,
JupyterLab 4.6.3, jupyter-mcp-server 2.2.2, jupyter_server_nbmodel 0.2.9,
ipywidgets 8.1.9, driven from Claude Code Desktop with the PowerShell tool. That
machine also had Git, so the Bash tool existed; a child's PC may not have it,
so everything was done in PowerShell on purpose.

### The canvas shows as text instead of a picture

`[any platform, verified, 2026-09-22]` The cell output is text like
`Image(value=b'\xff\xd8...` (or `IntSlider(value=0)`, `Output(layout=...)`),
while the game runs fine underneath. Cause: the notebook, or that cell, is
not trusted, and Jupyter does not render widgets for untrusted content. The
check is per cell: a cell that came from a never-trusted notebook can keep
failing after it is copied into a trusted one. The server log says `Notebook
<name>.ipynb is not trusted`, repeatedly. Fix: trust the notebook; cells
already shown as text then render without being run again.

False leads, each tested and ruled out: deleting the state files, the cell's
position, the browser (two profiles), the cell's metadata and `model_id`
(identical to working cells), and nbmodel disabling JupyterLab's cell
executor. That last one was an earlier theory: a `page_config.json` override
seemed to fix it once, but a controlled test later rendered every widget
without it once the notebook was trusted. Don't try it again for this.

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

`[macOS, seen once, Opus 5.5, 2026-09-28, JupyterLab 4.6, jupyter-collaboration
3.0.4]` Seen again, worse: a notebook whose room held over 22,000 edits showed
only its first 140 of 163 cells in the page, and kept doing so after a page
reload, closing and reopening the tab, restarting the kernel, and "reload from
disk". `read_notebook`, the file on disk, and the room replayed read-only
with `pycrdt` (as above) all had 163. The person saw code "gone" that the tools
said was there: count the page's cells before telling a person something is in
the notebook (scroll the windowed list; `data-windowed-list-index` of the last
cell + 1, since off-screen cells are not in the page). Stopping Jupyter, moving
`.jupyter_ystore.db` and `collaboration_sessions.json` aside (kept, not
deleted), and starting again fixed it: the page loaded all 163 from the file.

### Starting Jupyter

`[Windows 11, verified, 2026-09-22]` The activation form that worked is a `.cmd`
file that calls `<miniforge>\condabin\activate.bat <name>`, then `cd /d`, then
`jupyter lab ...`, run in the background as `cmd /c "<file> <token>"`. The
background output showed JupyterLab loading from `envs\<name>`. Not tried:
`conda.bat run -n <name> --no-capture-output`, and `conda-hook.ps1`. Do not use
`conda init` (PowerShell profile scripts are blocked by default).

### `CLAUDE_SETUP.md` on Windows 11

`[Windows 11, verified, 2026-09-23, Miniforge 26.7.2]` Three full runs:
- With Miniforge already there, and the code from a local `git archive`
  (`download` with a `file://` URL): `conda create -p ...` took about a
  minute, and the environment is still listed by name, so `conda activate
  jupylet` works; `pip install -e` worked without activating it; every helper
  command gave its expected answer; `find-env` listed the new environment
  first, and the canvas showed as a picture.
- With no Miniforge but Miniconda and an old Jupylet in `jp13` (the typical
  returning user): the installer, with its answers given up front, installed
  Miniforge in about 45 seconds, with no administrator password and no
  window, and created "Miniforge Prompt" (the `prompt` check needs it). Step
  3 found Miniconda both ways, and nothing of it was touched.
- A fresh session (Sonnet) from the `claude` branch on GitHub, through
  `CLAUDE.md` to a clean stop. What it got wrong was what it said, and the
  pages now address each point: it repeated an instruction to the person
  ("go straight to step 9"), described internals yet said nothing before a
  minute-long wait, added "I'll wait for you" after the waiter, skipped
  deleting the state files after stopping, did not tell apart the three
  things called `jupylet2` (environment, folder, "workspace"), and ran
  `sleep 5` before `wait`.

Not tried yet: updating an old Miniforge (macOS only), and a user name with
spaces or non-English letters.

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

`[Windows 11, verified, 2026-09-22]` After `attach`, `/api/sessions` lists
the notebook twice with the same kernel (the page's and the MCP server's):
normal, and ending both is part of stopping. Jupyter's own state in the
examples folder, `.jupyter_ystore.db` and `.jupyter\collaboration_sessions.json`,
is recreated on the next start: delete it after stopping, when no Jupyter
runs. `%APPDATA%\jupyter\file_id_manager.db` and old runtime files were left
alone.

Until 2026-10-01, `claude.py shutdown` could not be used on Windows (it used
`ps` and `SIGKILL`), so `CLAUDE.md` Part 6 had a manual procedure: end
sessions and kernels over HTTP, ask for `/api/shutdown`, then check the
token's processes. Since then `shutdown` looks for the processes with
PowerShell on Windows; not tested there yet.

### The port closes but the processes stay

`[Windows 11, seen in two of three runs, 2026-09-22 and 2026-09-23]` After
the shutdown request, the port closed and the log ended at `YDocExtension]
Deleting all rooms.`, but all five of the launcher's processes (`cmd.exe`,
`jupyter.exe`, two `python.exe`, `jupyter-lab.exe`) stayed alive, 15 seconds
and more later. (In the third run they exited by themselves.) Not tried:
waiting a minute or more.

First the session told the person a process was left and stopped there, as
the page then said. The author pointed out the flaw: a kid has no way to find
or end a stuck process, so that left them with a problem and no next step.
A stuck process after the normal stop is for the session to resolve, the way
it resolves a stuck kernel. What worked `[verified, 2026-09-22]`: list the
processes whose command line carries the session's token, check each command
line by eye against the launch command, and end each by its id (never by
name). All five ended at once and the port stayed closed. The background
task then reported `failed` (exit code 255): expected, since it was ended
rather than asked to exit. The app's own PowerShell process was never in the
list. Afterwards tell the person plainly that a leftover program had to be
closed and that it is done. This is `CLAUDE.md` Part 6, "Stopping on Windows
11", and what `shutdown` now does by itself.

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
