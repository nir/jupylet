# Changes of the simulators

Each entry: the version, the file, what changed, why (the run that showed
the problem), and the run that shows the effect.

## Version 1 (2026-10-12)

- `system.md`, `human.md`, `commands.md`: first versions, from the real
  transcripts in `findings.md`. No runs yet.

## Version 2 (2026-10-12)

- `system.md`: the simulator keeps the state in its own context. The seed is
  in its system prompt ("The computer at the start"), each request has only
  the action, and each answer only the output. At the end, `<final/>` asks
  for the state once, for the grader. Before, every request sent the whole
  state, and every answer wrote it out again.
- Why: in `ask-miniforge-6`, the system simulator made 23% of the tokens
  and 17k output tokens in 5 calls, mostly by rewriting the state. Nir: the
  sessions are a few dozen rounds, and a shorter context recalls better.
- Also, in the harness: the system and human simulators run with
  `--effort low` (less thinking).
- Effect: to be shown by `ask-miniforge-8` and `smoke-remember-1`.

## System version 3, human version 2 (2026-10-13)

- `commands.md`: how `conda init zsh` changes `~/.zshrc`, from the conda
  source (`init_sh_user`): it replaces each existing `conda initialize`
  block in place, and adds a block only when there is none. Why: in
  `ask-miniforge-13`, the simulator added a second block below the Miniconda
  block. Real conda replaces it.
- `human.md`: "Follow the persona": give the answers that the persona fixes,
  also when Claude words or groups the proposal differently; ask only the
  questions of the persona; end where the persona says. Why: in
  `smoke-hello-3`, the person asked a new question instead of ending; in
  `ask-miniforge-12`, Dana skipped the Terminal change and said yes to the
  environment, against her persona; in `ask-miniforge-13`, Dana asked
  question after question instead of "not now", and the run reached the time
  cap.
- Effect: to be shown by the next `smoke-hello` and `ask-miniforge` runs.

## System version 4 (2026-10-13)

- `system.md`: "What the computer at the start does not name does not
  exist", and the browser pane is closed unless the start says otherwise.
  Each part of the description adds to the parts above it.
- `systems/mac-base.md`, `systems/mac-jupylet-installed.md`: no "No …"
  lines any more; only what the systems add. `library.py`: the line between
  the parts says "adds to", not "changes".
- Why: in `ask-miniforge-18`, the base said "No `~/.zshrc`", the system above
  it added a `.zshrc` with Miniconda, and the simulator followed the base. The
  scenario additions of `smoke-stop` (two servers) and `smoke-remember` (a
  settings file) had the same conflict with `mac-jupylet-installed`.
- Effect: to be shown by the next `ask-miniforge` run.

## Human version 3 (2026-10-13)

- `human.md`: a fixed answer comes only when Claude makes the proposal, not
  before; and "Stay the person": never talk about the persona, the script,
  the eval or the session, and never name the person in the third person.
- Why: in `ask-miniforge-19`, Dana agreed to the Terminal change before Claude
  asked, and her last message said "Dana's request was only to install
  Miniforge…, and I said I'd stop there… So I'm stopping here."
- Effect: to be shown by the next `ask-miniforge` run.

## System version 5 (2026-10-13)

- `commands.md`: the missing commands of `claude.py`: `kernel`, `tools`,
  `wait-change`, `replace-kernel`, `run-cell`, `cleanup`, the other outputs
  of `shutdown`, and the exit code of `wait`. The outputs are from the code.
  A rule: never a usage error for a command of the list.
- Harness: the seed now ends with the list of files of the worker's folder,
  so the simulator knows, for example, that `header.md` exists. And all agents
  run with `--setting-sources project`.
- Why: in `full-install-miniconda-1`, the simulator answered `wait-change`
  with a usage error, and `cp -n` of `header.md` with "No such file". The
  worker had loaded a plugin skill of Nir's (`built-in-browser`) and then
  used the default browser instead of the browser pane.
- Effect: to be shown by `full-install-miniconda-2`.



## Human version 4, harness (2026-10-13)

- Harness: the worker's folder is a copy of the whole Jupylet code (the
  files that git tracks, without media and 3D files), and Claude reads the
  real files. A command that names a notebook brings its real cells to the
  system simulator. When Claude opens a page with a notebook, the human
  simulator gets its cells in a `<page>` part, in the percent format of
  jupytext.
- `human.md`: what the `<page>` part is.
- Why: in `open-spaceship-1` and `-2`, the system simulator made up the code
  of the spaceship notebook, Claude built a "game window" on it, and the
  human simulator, which saw nothing, agreed. In a real session, Claude can
  read the code and the person sees the page.
- Effect: to be shown by `open-spaceship-3`.

## Human version 5 (2026-10-13)

- `human.md`: "Answer each question of Claude", do not wait in silence.
- `scenarios/open-spaceship.json`: when Claude asks whether to run the
  notebook, Sam says yes; `max_turns` 14.
- Why: in `open-spaceship-3`, Claude asked whether to run the notebook. The
  persona did not say, so the human simulator waited for four turns, and the
  run reached the turn limit before the result.
- Effect: to be shown by `open-spaceship-4`.

## System version 6 (2026-10-13)

- `commands.md`: in a Jupylet notebook, `app.run()` returns at once (its
  docstring in `jupylet/app.py`), so `run-all` ends with `ran`, and the cell's
  output is the canvas.
- Why: in `open-spaceship-2` to `-4`, the simulator answered `timeout` with the
  last cell still at `[*]`, and Claude explained it with a game window.
- Effect: to be shown by `open-spaceship-5`.

## System version 7 (2026-10-13)

- `system.md`: the `<examples>` part. Harness: `recordings.py` collects the
  real outputs of tagged commands and of the app tools from exported
  sessions into `simulators/recordings.json`, and the harness adds the
  outputs of the same command to each request (few-shot). An error output is
  shown only next to a normal one.
- Source: T8, a clean install with Sonnet 5.5 on Nir's Mac
  (`session-export-1791923120091`): 46 outputs for 43 commands and tools.
- Why: the simulator gave short or invented outputs for the installer,
  `conda init`, `preview_start` and the "Tab Context" of the browser tools.
- Effect: to be shown by the next runs.

## Harness: background commands as in the real app (2026-10-13)

- A background command gives its ID and the path of its output file; the
  notification has only a status line; Claude reads the output file, which
  the harness serves. `grade.py` allows reading such a file.
- Why: T8 shows the real form. Before, the harness put the output into the
  notification, so a run never showed the step of reading the file.


## System version 8, Windows systems (2026-10-14)

- `system.md`: three phrases that named only the Mac or Bash now say "the
  real computer" and "Bash and PowerShell". Nothing else changed, so the Mac
  runs get the same rules.
- New systems `win-base` (with `Platform: windows`) and `win-miniconda`,
  from W1, a real installation on Windows 11 with Sonnet 5.5
  (`session-export-1791990888138`). `recordings.py` added its 45 outputs to
  `recordings.json`, marked `windows`. The system simulator gets only the
  examples of its own platform.
- Harness: a system's platform sets the home folder, the scratchpad and the
  shell tool. On Windows the worker gets a simulated `PowerShell` tool
  instead of `Bash`, and a note that the Jupylet folder has the form of a
  macOS path (the worker runs on the Mac). `run_eval.py --system NAME` runs
  a scenario on another system. The new scenario `full-install` names its
  steps by command tags, so it serves both platforms.
- Why: Nir: Haiku can play a Windows computer as well as a Mac, and the
  skills already have the Windows commands.
- Checked: the Mac worker prompt is byte for byte the same, the Mac
  recordings are the same, and the 34 earlier runs grade the same. Fake runs
  `fake-mac-1` and `fake-win-1` pass.
- Effect: to be shown by `full-install-win-1`.
- Also from Windows: W2 (`session-export-1791995312126`, a server already
  runs, with the notebook open in another browser) added 12 outputs, and W3
  (`session-export-1791995717388`, open the spaceship) 6. Among them:
  `running` with a server in its JSON, `attach` to a notebook that is
  already connected, and the sign-in check with `403` and `200`.

## Commands: wait-open as it really is (2026-10-14)

- `commands.md`: `wait-open` prints `open` as soon as the Jupyter server has
  a session for the notebook, also a session of another page that signed in
  earlier. Before, it said "when a page shows the notebook and has signed
  in", which is what the command is meant to do, not what it does.
- Why: W2 (Windows, a server already runs with the notebook open in the web
  browser): `wait-open` printed `open` at once, before the new page in the
  app signed in. The simulator must show the real behavior, so that a
  scenario can test how Claude handles it.
- New scenario `open-spaceship-running`, on `mac-jupyter-in-browser` with the
  setting `hide-chat-list = yes`.
- Effect: to be shown by `open-spaceship-running-1`.

## Harness: the text of header.md in the seed; Sam knows the setting (2026-10-14)

- The seed of the system simulator now has the text of
  `configuring-jupylet/header.md`, as the header of each settings file.
- Why: in `open-spaceship-7` (after `cp -n "…/header.md"`) and in
  `open-spaceship-running-2` (a settings file "with the header" in the
  seed), the simulator made up another header.
- `open-spaceship-running`: the request tells Sam that, in an earlier
  session, Sam asked Claude to hide the list of chats and to remember it.
  Why: in `open-spaceship-running-2`, the setting was only in the seed,
  which the human simulator does not get, so Sam objected when Claude hid
  the list. The request is the note of the director to the actor.
- Effect: to be shown by `open-spaceship-running-3`.

## System version 9 (2026-10-14)

- `system.md`: `tabs_context` adds the line that says whether the person
  sees the pane, in the real wording, and a page that `preview_start` opens
  is in a hidden pane until the person shows it.
- Source: a live test in Nir's Claude app (desktop, macOS): `preview_start`
  opened a page in a hidden pane ("The Browser pane is currently hidden.");
  after Nir showed the pane, "The Browser pane is currently displayed."
- Why: in `open-spaceship-running-2`, the simulated `tabs_context` had no
  such line, so Claude could not do what the skill says: check whether the
  pane is visible.
- Effect: to be shown by `open-spaceship-running-3`.
