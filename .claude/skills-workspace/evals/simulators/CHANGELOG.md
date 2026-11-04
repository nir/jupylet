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

## Commands: wait-open waits for a new web page (2026-10-21)

- `commands.md`: `wait-open` prints `open` only when a new web page opens
  the notebook: its first session, or one more connection to its kernel
  than at the start. This follows the fix of `wait_open()` in
  `jupylet/claude.py`. The entry of 2026-10-14 described the old behavior
  (W2), which printed `open` at once when another web page had the notebook.

## Commands: claude.py's own documentation (2026-10-21)

- The harness adds the USAGE text of `jupylet/claude.py`, from the code
  under test, to the prompt of the system simulator, as "The documentation
  of jupylet/claude.py". `commands.md` no longer describes the commands of
  `claude.py` itself: only what the documentation does not say (how a
  Jupylet app runs, a few outputs of the Jupyter MCP tools, the rule against
  usage errors).
- Why: Nir: one source for the commands' outputs, so that the simulator
  cannot drift from the code. The docstring now documents every command.
- Effect: to be shown by the next run.

## Commands: Browse notebooks (2026-10-21)

- `commands.md`: the output of the new command "Browse notebooks"
  (`notebooks/browse-notebooks`): the full path of the chosen file, or
  `canceled`.
- Why: the running skill now opens a file window when it cannot find the
  notebook that the person names. Without this line, the simulator would
  make up an output.
- Effect: to be shown by a run that uses the command.

## System version 10: the Terminal panel (2026-10-22)

- `system.md`: the tools of the Terminal panel, `run_in_terminal`,
  `read_terminal`, `list_terminal_tabs` and `stop_terminal_tab`, with the
  real outputs of a test tab, and the real start output of JupyterLab from a
  test server.
- Also, in the harness: a simulated server `terminal` with these tools, and
  the grader takes the start and the read of the Jupyter tab as the steps
  `notebooks/start-jupyter` and `notebooks/wait-ready`. The checks of
  `notebooks/check-port` are gone with the command.
- Why: in the Claude app, the skills now start Jupyter in a hidden tab of
  the Terminal panel, so that it lives as long as the chat (Nir tested it),
  and read the port from its output. Jupyter picks a free port itself.
- Effect: to be shown by the next run of `open-spaceship`.

## Commands: plain answers of run-all and wait-change (2026-10-23)

- `commands.md`: `run-all` ends with `finished: the run is over.
  Fingerprint: …`, not `ran …`. The other answers come from `claude.py`'s
  own documentation, which has them: `still running`, `not started`, and
  for `wait-change` `finished` and `not finished`.
- Why: Nir asked for answers that read clearly. Also, `run-all` no longer
  ends with `error` when the web page says "Timeout waiting for result":
  the run did start (seen 2026-10-22, first run in a fresh environment).
- The run-all recordings in `recordings.json` (T8, W1, W2, W3) have the
  old answers. They are real outputs, so they stay, marked `outdated`, and
  `recordings.examples()` leaves out each outdated recording.
- Effect: to be shown by the next run of `open-spaceship`.

## Recordings: T9, with the notebook-opener (2026-11-01)

- `recordings.json`: 29 outputs of T9 (`session-export-1793568223431`), a
  fresh install on Nir's Mac with Sonnet 5.5 and the Haiku notebook-opener:
  the download, the setup, the opener's commands, its Terminal tab, the
  real wait for the person (403, then the script cut off by the sign-in
  reload, then 200), and the fold of the JupyterLab side bar.
- Harness: `recordings.py` also reads the subagents' transcripts of an
  export (the opener's calls are there, not in the main chat), also keeps
  the `mcp__terminal__…` tools, and leaves out the app's task-list tools
  (`mcp__ccd_turn__…`). A newer session now replaces the oldest outputs of
  a full key; within one session, its first outputs are kept. The
  JavaScript tool keeps 6 outputs, not 3, since it runs several different
  scripts.
- Why: before, `add` would have skipped all of the opener's calls, and the
  first sessions (T8, W1) kept their places, so the examples could not
  follow the skills.
- Replaced: the 403 and the 200 of T8 in `tool:mcp__Claude_Browser__javascript_tool`.
- Effect: to be shown by the next run of `smoke-open-new` or `open-spaceship`.

## Recordings: T13, T15 and W4, the direct open and Windows 11 (2026-11-03)

- `recordings.json`: 7 outputs of T13 (`session-export-1793703772534`) and 6
  of T15 (`session-export-1793704313650`), on Nir's Mac with the new task
  "Open the notebook directly": `notebooks/wait-serve`, the sign-in check
  (200 in T13, 403 in T15), `preview_start`, `tabs_context`, attach, and
  `list_terminal_tabs`. 32 outputs of W4 (`session-export-1793724063238`), a
  fresh install on Windows 11 with the notebook-opener: the download, the
  setup, the environment, trust, the opener's commands, and run-all.
- Left out: the two failed `download/get-archive` of W4 (`bad output glob`):
  the scratchpad path was 282 characters, over the limit of Windows, and the
  download now goes to `$env:TEMP`.
- Not recorded: the `run_in_terminal` call of `serve` (its output has the
  same form as T9's), and the `wait-open` of T15.
- Effect: to be shown by the next run of `smoke-open-new` on macOS, and of a
  Windows scenario.

## Commands: serve and wait-ready (2026-11-03)

- `commands.md`: what `serve` does to the state (the setting, the new
  notebook, the check for a server that has the notebook open, the new
  server in the Terminal tab), that its output goes to the tab, how soon
  `wait-ready` answers, the exit code 1 of a `wait-open` timeout, and that a
  new server on a port of an earlier one is often already signed in.
- Why: the documentation of `claude.py` says what the two commands print,
  but not what they change in the state, and the new task "Open the notebook
  directly" uses them. T12 and T13 were signed in at once.
- Effect: to be shown by the next run of `smoke-open-new` on macOS.

## Settings: claude.py config, no header.md (2026-11-04)

- `recordings.json`: removed the 10 outputs of `config/make-folder`,
  `config/copy-header` and `config/add-line` (T8, W1, W4). The skill
  `configuring-jupylet` now reads and writes the settings with `claude.py
  config`, `config set` and `config delete`, and these commands are gone.
- Harness: the seed gives the header of the settings files from
  `CONFIG_HEADER` in `claude.py`, since `header.md` is deleted.
- Effect: to be shown by the next run of `smoke-remember`.

## Systems: win-jupylet-installed (2026-11-04)

- `systems/win-jupylet-installed.md`: a Windows 11 PC where Jupylet is
  installed, from W5: Miniforge, the environment `jupylet` prepared, the
  examples trusted, the local setting `environment`, and no global settings.
- Why: on macOS Claude opens a notebook directly, so the notebook-opener
  runs only on Windows, and no Windows system had Jupylet installed. It
  tests the opener's read of the settings with `claude.py config`.
- Effect: to be shown by `open-spaceship --system win-jupylet-installed`.

## The web page opens with the token in its address (2026-11-04)

- Skills: Claude opens the notebook at its address with `?token=`, as
  JupyterLab prints it, and the web page is ready at once. The checks of
  `smoke-open-new`, `open-spaceship`, `open-spaceship-running`,
  `full-install` and `full-install-miniconda` now check for that address.
- `system.md`, `human.md`, `commands.md`, the harness: the examples and the
  notes of the old steps for the person are replaced.
- `recordings.json`: removed 8 outputs of the old page scripts (T9, T13,
  T15, W4).
- Why: T16 (`session-export-1793781614434`), on Nir's Mac: Claude opened
  the address that JupyterLab printed, and the notebook was ready in 27 s.
- Effect: to be shown by the next run of `smoke-open-new`.

## Harness: the open page is a fact of the state (2026-11-04)

- `run_eval.py`, `page_fact()`: with a `preview_start` or `navigate` to a
  notebook address, the harness tells the system simulator that the web page
  now shows the notebook and is connected to its kernel, also in a hidden
  pane, so that `attach` and `kernel` follow from the state.
- Why: in `smoke-open-new-5`, the simulator took the page in a hidden pane
  for a page that is not open.
- `smoke-open-new-6` then showed a real problem, not one of the simulator:
  `wait-open` does not count a web page that was open before the wait
  started, and with the token in its address the page is open at once. So
  the skills now attach right after the page opens: `attach` waits up to a
  minute for the kernel. Its `wait-open` timeout was right.
- Effect: to be shown by `smoke-open-new-7`.

## Commands take the port; no wait-open, no background monitors (2026-11-04)

- `claude.py`: the commands that talk to a Jupyter server take only its
  port, and find its token in Jupyter's records. `wait-ready` prints the
  address of the web page with the token. `shutdown` names the processes
  that remain (`still running: processes ...`). `wait-open` is removed.
- Skills: no `{token}` in the commands of Claude. The opener still makes the
  token for its start script and the address of the web page. The skill
  `running-background-monitors`, the agent `background-monitor` and
  `briefs.md` are removed: nothing started a monitor any more.
- `commands.md`, `system.md`: the notes of `wait-open` are replaced, and a
  note says how the commands with a port find the token.
- `recordings.json`: removed `notebooks/make-token` and `notebooks/wait-open`,
  and the token after the port in 17 inputs.
- Effect: to be shown by `smoke-open-new-8`.
- `smoke-open-new-8`: the simulator gave the address of `wait-ready` as
  `/lab/tree/...`. The usage text of `claude.py` now shows its form,
  `http://localhost:<port>/doc/tree/<notebook>?token=<token>`.
- `smoke-open-new-9`: the simulator copied the `ready:` line of the T13
  recording, with `token` and no `address`, and with `hide chat list yes`
  from T13, not from the state. Removed the recordings of
  `notebooks/wait-serve`: the usage text of `claude.py` gives the form.
- Harness, `recordings.py`: each output of a `claude.py` command now keeps
  a fingerprint of the command's part of the usage text (`doc`), and
  `examples()` leaves an output out when that part differs in the code
  under test. So a change of a command makes its old outputs stale by
  itself, without a hand edit. `recordings.py stamp` gave the current
  fingerprints to the 28 outputs that match the current code.

## Settings files in TOML (2026-11-04)

- `claude.py`: the settings are in `.jupylet/claude-config.toml`, one
  `key = "value"` line for each, under a header of comments. A
  `claude-config.md` of an earlier version still gives its settings until
  the first write, which moves them into the TOML file and removes it.
- Seeds and checks: `smoke-remember` and `open-spaceship-running` start with
  a TOML file. The state checks look for `key = "value"`, and the checks of
  the Write and Edit tools name both file names.
- Systems: `mac-jupyter-in-browser` and `win-jupylet-installed` have the
  TOML file.
- Effect: to be shown by `smoke-remember-3`.

## The environment's Python (2026-11-04)

- `claude.py`: `serve` goes on in the environment's Python as soon as it
  finds the environment, so its checks and `running` run there. Only
  `find-env`, `config` and `wait-ready` run with the Python of Miniforge.
- Skills: "List the Jupyter servers" runs with `{python}`. The opener finds
  the environment before it lists the servers.
- Why: W6 (Windows 11): `running` failed in Miniforge's base, from a second
  `_runtime_dir()` without a fallback.
