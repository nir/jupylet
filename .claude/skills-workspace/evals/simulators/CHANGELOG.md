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

