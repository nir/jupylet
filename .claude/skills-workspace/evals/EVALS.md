# Evals for the Jupylet skills

This file describes the eval machine for the Jupylet Claude skills: what it
is for, how it works, how to run it, the decisions behind it, and its state.
Read it before you change or run anything in this folder.

## What we want

The Jupylet skills (`guiding-jupylet-users`, `setting-up-jupylet`,
`managing-jupylet-notebooks`, `working-in-notebooks`,
`running-background-monitors`, `configuring-jupylet`, and `CLAUDE.md` with
`CLAUDE_SETUP.md`) guide Claude when it installs Jupylet and works in
notebooks with a beginner. Until now, Nir tested them by hand: a real
session, an export, and a review. Each test took long, changed the
computer, and tested one path.

The eval machine tests the skills in simulated sessions instead:

- **Many cheap runs.** All agents run on Haiku 5.5. A session of a few
  turns costs about 1 to 2 cents at API prices.
- **Realistic sessions.** The failures that matter appear only over many
  turns: Claude forgets an instruction, drifts, or collects small errors.
  So each agent is one long session that grows, as in a real chat. Claude
  gets the real Claude Code system prompt, the real `CLAUDE.md`, the real
  skills through the real Skill tool, and real tool calls.
- **Checks from real failures.** `findings.md` lists what went wrong in
  real sessions. Each eval checks some of these points.
- **The skill-creator loop.** Evals, paired runs against an older version,
  grading, a benchmark, the viewer, and Nir's feedback.

A simulation cannot test some parts of the Claude app: when skills appear in
the list after a folder change, the file box, rewind, the browser pane
itself. Real sessions stay the test for those. `findings.md` marks them as
"real only".

## The three agents

Each run has three agents. All three run on Haiku 5.5
(`claude-haiku-5-5`). Do not use Sonnet or Opus for them. `grade.py` checks
the model of every call.

1. **The worker** plays Claude in the Code tab of the Claude app. It runs
   as a `claude -p` session in a temporary copy of the Jupylet code under
   test: every file that git tracks, as a person downloads it, and the whole
   folder `.claude/skills`, also its files that git does not track yet. Media
   and 3D files (images, sounds, video, `.blend`, `.bin`, `.gltf`, fonts) are
   not copied: Claude does not read them. So the CLI loads `CLAUDE.md` and
   the skills exactly as in a real session, and Claude can read the real
   notebooks and the real Jupylet code. The seed of the system simulator
   lists all these files, the media files included. It has:
   - the real Claude Code system prompt, plus `worker.md` appended (the tool
     names, the home folder of the imagined person, the scratchpad);
   - the real Skill tool, and no other built-in tool;
   - tools of our MCP server (`sim_mcp.py`): `mcp__computer__Bash`,
     `mcp__computer__Read`, `mcp__computer__Write`, `mcp__computer__Edit`,
     the browser pane tools as `mcp__browser__…`, and the app tools with
     their real names (`mcp__ccd_window__set_sidebar_collapsed`,
     `mcp__ccd_directory__change_directory`, and the Terminal panel tools
     `mcp__terminal__…`). The CLI keeps the server name
     `Claude_Browser` for the real pane, and drops another server with that
     name. Until run `open-spaceship-1`, the worker had no browser tools at
     all, and used the default browser. Now `worker.md` maps the names, and
     the harness records each call under the real name,
     `mcp__Claude_Browser__…`, for the checks and the system simulator.
2. **The system simulator** plays the computer and the Claude app. It
   answers each tool call with the output that the real tool gives
   (`simulators/system.md`, `simulators/commands.md`). The seed of the eval,
   "The computer at the start", is in its system prompt. It keeps track of
   the state in its own context. At the end, `<final/>` asks it for the
   state once, for the grader (`state.md`).
3. **The human simulator** plays the person (`simulators/human.md` and the
   persona of the eval). It sees only Claude's chat text. It answers with
   `<say>` (chat), `<do>` (an action on the computer, for example "paste the
   token into the page"), `<wait/>` or `<end/>`.

All three agents run with `--effort medium` (`EFFORT` in `run_eval.py`), since run `ask-miniforge-22`. Before, the simulators ran at `low`, and the system simulator lost the Miniconda part of its seed in run 20. The worker had no effort option, so it had the CLI default.

Nothing in a run touches the real computer. The imagined home folder (for
example `/Users/dana`) exists only as text in the seed. The Jupylet folder is
the real temporary copy, because the CLI needs it, and the seed refers to it
as `{code}`.

## How a run works

`run_eval.py` is the harness. For each eval:

1. It copies the files under test into a new temporary folder named
   `jupylet`: from the working tree, from `--skills <folder>`, or from a git
   commit with `--ref <commit>` (for a baseline).
2. It starts a Unix socket. The worker's CLI starts `sim_mcp.py` (one process
   for each MCP server name). For each tool call, `sim_mcp.py` asks the
   harness through the socket, and returns the answer as the tool's result.
3. The harness answers each tool call:
   - **A Read or `cat` of a file of the copy** (skills, `CLAUDE.md`,
     notebooks, Jupylet's code): the real file. A Read of a notebook gives
     its cells in the percent format of jupytext, not the JSON, as the real
     Read tool also gives a notebook as cells. A `cat` gives the JSON. After a command of the run
     names the file or one of its folders, for example the trust command
     for `examples/`, the file is state, and the system simulator answers.
   - **A Jupylet command that names a notebook** (`attach`, `call
     read_notebook`, `run-all` and so on): the harness adds the real cells
     of that notebook to the request, so the system simulator gives the
     real sources and only makes up the run state. Before run
     `open-spaceship-3`, it made up the code of the spaceship notebook.
   - **A tagged command, or an app tool:** before the system simulator
     answers, the harness adds the real outputs of the same command or tool
     from real sessions (`simulators/recordings.json`), as examples. To add
     the outputs of a new export: `python3 recordings.py add <export.zip>
     <name>`, for example `T9`. `recordings.py list` shows what is there.
     Each output has its platform, macOS or Windows (`W1`), and the system
     simulator gets only the outputs of its own platform.
   - **Known limitation:** no command runs for real. When a command uses
     the content of a real file of the copy to make an imagined one, for
     example `cp -n "…/header.md" ~/.jupylet/claude-config.md`, the system
     simulator does not know that content, and makes it up. In
     `open-spaceship-7`, Claude then saw a header that differs from
     `header.md`. Treat such a difference as an artifact of the simulator,
     not as a failure of Claude. Fixed for `header.md`, which came back in
     `open-spaceship-running-2`: the seed now has its text. For other files,
     fix it only if it comes back often.
   - **A background command** (`run_in_background`): as the real Bash
     tool, an ID and the path of an output file at once. The command stays
     pending. When the system simulator ends it, the notification to Claude
     has only a status line, with the exit code, and the output goes into
     the file. Claude reads it with Read or `cat`, and the harness serves it,
     with `[exited with code N]` at the end. Before T8, the notification had
     the output in it.
   - **Any other tool call:** the system simulator's answer.
4. A turn of the person: the human simulator writes a message. The person's
   `<do>` actions go to the system simulator as events. Then the system
   simulator says which pending background commands end now, and the worker
   gets their task notifications.
5. A turn of Claude: one `claude -p --resume` call of the worker. The CLI
   runs Claude's whole turn, with all its tool calls through our MCP server,
   and returns at the end of the turn. The harness reads the turn (Claude's
   texts, the Skill calls, the tool calls) from the worker's session file,
   and sends Claude's text to the human simulator.
6. This repeats until the person writes `<end/>`, or a limit is reached:
   `max_turns`, `max_actions`, or `max_seconds` (a safety cap, 120 seconds
   by default, for a run that hangs).
7. At the end, the harness asks the system simulator for the final state,
   keeps the worker's session file in the run folder, and deletes what the
   run left outside it: the copy, the socket, the agents' temporary folders
   and their CLI session files.

The system and human simulators run in empty temporary folders. The CLI
loads `CLAUDE.md` and `.claude/` of the folders above its working folder.
Inside the repository, that gave every agent the Jupylet skills and
Claude's memory of this project.

A run writes these files into its folder:

- `conversation.md`: the session in order (`PERSON:`, `PERSON DOES:`,
  `CLAUDE:`, `ACTION`, `MACHINE`, `NOTIFICATION`). A served Jupylet file
  shows as `MACHINE (real file, N lines)`.
- `events.jsonl`: the same, as data, for the grader.
- `state.md`: the final state of the computer, from `<final/>`.
- `timing.json`: tokens, cost and time of each agent.
- `run.json`: the eval, how the run ended, the time, the version of the
  skills and of the simulators, the digest of each library file used
  (`library`), and `git`: the commit of the files under test, and those of
  them (skills, CLAUDE files, `jupylet/claude.py`) that differ from it.
  Compare only runs whose `git`, `simulators` and `library` match, or say
  what differs.
- `tags.json`: each command tag in the files under test, with its command
  lines. `grade.py` fails a command whose tag is not in it: a tag that Claude
  made up.
- `eval.json`: the eval as it ran, with the library files resolved and their
  checks added. `grade.py` grades these checks.
- `agents/`: the prompts and the raw messages of each agent, and the
  worker's CLI session file (`worker.session.jsonl`).
- `grading.json`: written by `grade.py`.

## How to run

**Where.** The commands that Claude runs with its Bash tool get no login
from the Claude app, on purpose. So Claude starts each run in the Terminal
panel of the app, with the tool `mcp__terminal__run_in_terminal`, with the
working folder `.claude/skills-workspace/evals`. Each call opens a new tab.
Claude does not close the tabs: Nir closes them when he wishes. One run per
tab: the run and its grade, and nothing else. Never chain several runs in
one command line, so that each tab shows one run, easy to read. Claude reads
the tab with `mcp__terminal__read_terminal`, and the results in the run
folder. While a run goes on, `run_eval.py` prints one line for each message
of the person, each message of Claude, each tool call and its first output
line, and the cost of each agent after each turn of Claude.

**Login.** The CLI in the Terminal panel is signed in to Nir's Claude
account, on his subscription, not API billing. This was done once, on
2026-10-12, with:

```bash
"$HOME/Library/Application Support/Claude/claude-code/2.1.295/9d6405864fdc/claude.app/Contents/MacOS/claude" auth login
```

The sign-in happens in the browser (Nir uses Google). If a run fails with
"Not logged in", Nir must run this again. Claude does not sign in for him.
The harness finds the newest bundled CLI by itself (or `CLAUDE_BIN`).

**Permission.** Start a run only after Nir says yes to it. Runs use his
quota.

**Commands,** from `.claude/skills-workspace/evals`:

```bash
python3 run_eval.py smoke.json ask-miniforge runs/ask-miniforge-12
python3 grade.py smoke.json runs/ask-miniforge-12
python3 token_report.py summary runs/ask-miniforge-12
python3 token_report.py calls runs/ask-miniforge-12
```

Each run in its own tab, with its grade:

```bash
python3 run_eval.py smoke.json ask-miniforge runs/ask-miniforge-14 --live && python3 grade.py smoke.json runs/ask-miniforge-14
```

Options of `run_eval.py`:

- `--live`: the whole transcript in the terminal while the run goes on:
  each message in full, and each command output, up to 40 lines. Claude's
  messages show while its turn runs, from its CLI session file. Without
  `--live`, the terminal shows one short line for each event.
- `--fake`: scripted stand-ins for the three agents, no model calls, no MCP.
  It tests the harness itself, for free.
- `--ref <commit>`: the skills of an older commit, for a baseline run.
- `--skills <folder>`: the skills of another Jupylet folder.
- `--system <name>`: run the scenario on another system of the library, for
  example `win-miniconda`. The platform of the system (macOS or Windows)
  sets the home folder, the scratchpad and the shell tool of the worker:
  Bash or PowerShell.

```bash
python3 run_eval.py scenarios/full-install.json full-install runs/full-install-win-1 --system win-miniconda --live
```

Run folders are numbered: do not reuse a name, because the old results are
the comparison.

## Grade a real session

An exported real session can be graded with the checks of a scenario:

```bash
python3 import_export.py ~/Downloads/session-export-….zip scenarios/full-install.json runs/real-T8
python3 grade.py scenarios/full-install.json runs/real-T8
```

For a session on Windows, add the system with `--system win-miniconda`, so
that the checks of a Windows system apply. `run.json` gets the platform of
the session (Windows when it used the PowerShell tool).

`import_export.py` writes a run folder, with `"real": true` in `run.json`.
The grader skips what only a simulated run has: the final state, the models
of the agents, the skills that the worker saw, and the served files. It
also skips the preamble: what Claude did before its first Jupylet step,
for example the download of `CLAUDE_SETUP.md` from the address that the
person gave. A check about the system of the scenario only, for example
its Miniconda, does not apply to a computer that does not have it.

Grading costs nothing: it calls no model.

## Test the machine first

Never start a long eval (about 20 minutes) on a machine that is not tested.

1. **Fake run, free,** after each change of `run_eval.py` or `grade.py`:
   `python3 run_eval.py smoke.json smoke-hello <scratch folder> --fake`.
   It tests the turns, background commands, notifications, the output files
   and the grader. It does not test MCP.
2. **Smoke runs.** The sessions in `smoke.json` are tiny by design: one small
   request and a few turns, so each one ends by itself in 20 to 30 seconds
   (`expected_seconds`). `loaded_skills` starts a session with skills
   already loaded, as after a compaction. After each smoke run, read
   `conversation.md` and `state.md`, and report what happened.
3. **Longer tests**, such as `ask-miniforge` (about 70 to 100 seconds).
4. **Long evals**, only after the shorter ones behave.

Each run checks the machine too:

- "Every agent call used claude-haiku-5-5."
- "The harness served every Jupylet file that Claude read (none made up)."
  If this fails, the run says nothing about the skills.

## Evals and checks

An eval is an object in a JSON file (`smoke.json` now, the `evals.json` of a
skill later):

- `id`, `home` (the imagined home folder).
- `persona`: who the person is, what they want, and when they end.
- `seed`: the state of the computer at the start. `{code}` is the Jupylet
  folder. Take outputs from the real transcripts where possible.
- `loaded_skills`, `max_turns`, `max_actions`, `expected_seconds`.
- `checks`: what a correct session does. Each check names its finding in
  `findings.md` when it has one.

All evals now use the library in `simulators/LIBRARY.md`, in
`smoke.json` and in `scenarios/*.json` (one eval per file). Inline
`persona` and `seed` texts, as above, still work, but the library is the
rule:

- `persona`: the name of a file in `simulators/personas/`. `request` adds
  what the person wants in this session, the answers that the scenario
  fixes, and when the person ends.
- `system`: the name of a file in `simulators/systems/`, with the systems
  that it extends. An optional `seed` adds to it.
- `user`, `home`: fill the blanks `{user}` and `{home}` of the files.
- The checks of the persona and the system files are added to `checks`.
  Each check gets `source`: `scenario`, `persona …` or `system …`.

`library.py` does this. Run a scenario as any eval, with its file:
`python3 run_eval.py scenarios/full-install.json full-install runs/full-install-mac-1`.

**Only Claude is graded.** The simulators can deviate from the persona
or from a real computer, for example a person who says no where the
persona says yes. That is not a failure of Claude. A check that applies only
when a simulator does something has `depends`, with that condition, for
example `"depends": "Dana says yes to the change of the Terminal settings"`.
When such a check fails, `grade.py` marks it `REVW` (to review), not `FAIL`.
The review finds out whether the simulator did what the check needs:

- **It did not:** the simulator deviated. Note it, and propose a small
  improvement of the simulator (see "Improve the simulators").
- **It did:** it is a real failure of Claude.

`grade.py` grades the checks in the `eval.json` of the run. After a change of
the checks, grade again with `--current`.

**Commands from the skills.** Each shell command in the skills ends with a
comment, `# jupylet: {group}/{name}`, for example
`# jupylet: setup/find-conda`. The groups are `download` (`CLAUDE_SETUP.md`),
`setup`, `notebooks` and `config`. `grade.py` fails a run when a Bash or
PowerShell command of Claude has no such comment, or a tag that is not in
the skills (in `tags.json` of the run): Claude wrote the command itself
(finding 5). An eval can allow other commands with `allowed_commands`, a list
of patterns. The same comments make the commands easy to find in real
transcripts.

`grade.py` also checks that each tagged command is the command of the skills
with that tag: the same text, with only the blanks in `{braces}` filled in.
A command that Claude changed and kept the comment fails, for example a
`printf … >` instead of `cp -n`, or `sleep 5;` added in front.

**The same checks on macOS and Windows.** The macOS and the Windows command
of a step have the same tag. So a check names a step by its tag, not by its
command text: `{"type": "action", "tag": "setup/create-env"}`, or in
`before`, `"first": {"tag": "setup/check-system"}`. A tag is a pattern:
`setup/find-conda.*` matches the conda search of both platforms. A check
for one platform only has `"platform": "macos"` or `"windows"`, for
example the Terminal steps, and is not graded on the other.

**A problem that comes back becomes a check.** When a run shows a problem
that earlier runs showed too, add a PASS/FAIL check for it, without
`depends` when the skill asks for the behavior in each case. For example,
after `open-spaceship-2`: "Claude offers to hide the list of chats".

Check types of `grade.py`:

- `action`, `no_action`: an action of a tool whose input matches a pattern,
  or a shell command with a tag (`tag`).
- `before`: the first match of one pattern comes before the first match of
  another.
- `not_in_turn`: no matching action in a turn of the conversation.
- `state`: the final state matches a pattern.
- `text`: some message of Claude matches a pattern.
- `no_text`: no message of Claude matches a pattern.
- `questions`: at most N question marks in each turn of Claude. This is too
  crude: a heading that quotes the person's question counts too. Replace it
  with a judge check.
- `judge`: needs judgment. Not graded yet: a Haiku grader is to be written.

## Paired runs and the benchmark

Each long eval runs with the current skills and with a baseline, the skills
of an older commit (`--ref`). Both runs use the same version of the
simulator files. Otherwise a change of the score can come from the
simulators and not from the skills. Run each side more than once: Haiku
varies from run to run. Then `aggregate_benchmark` and `generate_review.py`
of skill-creator show the results to Nir.

## Improve the simulators

After each run, compare the simulators with the real sessions:

- **The system simulator** must give real outputs, not helpful ones. Errors
  to look for: a command succeeds that fails on a real Mac, a file is
  missing or wrong, a Jupylet command prints the wrong text.
- **The human simulator** must act like the persona. Errors to look for: it
  knows what only the computer knows, it follows the skill steps by itself,
  it does not end where the persona ends.

Propose each change to Nir, as a small change of `simulators/system.md` or
`simulators/human.md`. Make it only after his yes. A change must make a
simulator more like the real sessions, never easier for the skills. Raise
the version number in the file, and write the change into
`simulators/CHANGELOG.md`, with the run that showed the problem and the run
that shows the effect.

## Tokens and cost

`python3 token_report.py summary|calls <run folders>` reports tokens and
cost, per run and per call. `reports/` keeps the reports.

- **Count from the last call of each agent.** In a resumed session, the CLI
  reports the totals of the whole session so far (`modelUsage`,
  `total_cost_usd`), over all model requests. Summing them over calls counts
  early calls many times: run 8 showed 1.2M tokens instead of 360k.
  `usage` covers only the last model request of a call.
- **Input has three parts:** cache read (the old part of the context, at
  about a tenth of the input price), cache write (the new part of each
  request, slightly above the input price), and uncached input (only what
  follows the last cache marker: about 2 tokens). So the real new content of
  a round is in "cache write".
- **Typical numbers** (`ask-miniforge`, runs 9 to 11): 360k to 480k tokens,
  over 85% cache reads, $0.014 to $0.017. The worker makes about two thirds
  of the cost: each of its model requests reads its whole context, about
  12k for the Claude Code base, about 7k for the loaded skills and the
  reference file, and the conversation. Thinking is about 40 to 50% of the
  output.
- **Haiku 5.5 prices:** $0.10 per million input tokens and $0.50 per million
  output tokens, for requests whose prompt (all input tokens of the one
  request, cached ones included) is up to 100k tokens. Above that, $0.50 and
  $2.50. Our largest requests are about 30k.

## Decisions and why

- **Haiku for all agents.** Runs must be cheap enough to repeat often.
- **Separate agents for the computer and the person.** In one agent, the
  person would know what only the computer knows, and the two roles pull
  against each other. Separate contexts are also smaller.
- **One growing session per agent, not a fresh call for each step.** A fresh
  worker reads the session as a log, with its instructions fresh at the top.
  A real Claude drifts as its context grows, and that drift is what we want
  to see.
- **A Python harness, not subagents.** Tested on 2026-10-12:
  - Subagents can start subagents, but continuing one (SendMessage) is
    asynchronous. The answer arrives in a later turn.
  - A subagent that ends its turn to wait hands back to the main session,
    so the main session (Opus) is woken in every round. Even with a
    background child, it handed back early.
  - In the test, the agents also sent messages to each other outside the
    relay, and mixed up IDs.
  - The Claude Code docs describe no way for a subagent to block until an
    answer arrives. Workflows return one result, but each `agent()` call is
    one-shot. Agent teams are experimental.
  - `claude -p --resume` in a script is the documented synchronous way.
- **The real Claude Code context for the worker.** In `ask-miniforge-1`, with
  `CLAUDE.md` pasted into our own prompt, Claude loaded no skill at all. With
  the real mechanism (the copy of the files as the working folder), it loads
  the guiding skill first, as in real sessions.
- **MCP tools for the computer, not text blocks.** First the worker wrote its
  tool calls as text blocks, and had the real Skill tool. After one real
  Skill call, Haiku made its next calls as real calls too, to Bash, which was
  off, and then told the person that it cannot run commands
  (`ask-miniforge-2` and `-3`). The docs offer no way to give the native Bash
  tool simulated results (hooks cannot supply a result, a custom tool cannot
  take a built-in's name), but MCP tools are the documented way to add tools.
  So the computer's tools are MCP tools, and all calls are real calls.
- **The harness serves the Jupylet files.** The system simulator does not
  know them, and makes them up. The copy's path is resolved with
  `realpath`: on macOS, the temporary folder `/var/…` is a link to
  `/private/var/…`, and the CLI gives Claude the `/private/var` path. Before
  this fix, `macos.md` was not served in runs 5 to 8, and the simulator made
  up a different file in each run.
- **The simulator keeps the state in its own context** (simulator version
  2). Version 1 sent the whole state with each request and wrote it out in
  each answer. Nir: the sessions are a few dozen rounds, and a shorter
  context recalls better. In `ask-miniforge-8`, version 2 stayed consistent
  (it changed exactly the right line of `~/.zshrc`, and gave an accurate
  final state), at about a third of the output tokens per call.
- **Empty working folders outside the repository** for the simulators. See
  "How a run works".

## State, 2026-10-12

Built: `run_eval.py`, `sim_mcp.py`, `grade.py`, `token_report.py`,
`worker.md`, simulators version 7 (system) and 5 (human), `smoke.json`, and
`findings.md` with the findings from 7 real sessions and 8 proposed evals.

Runs, in `runs/`:

| Runs | What | Result |
|---|---|---|
| `smoke-hello-1`, `-2` | the first live runs | worked. In `-2`, the human did not end where the persona ends. |
| `ask-miniforge-1` | `CLAUDE.md` in our own prompt | no skill loaded |
| `ask-miniforge-2`, `-3` | real Skill tool, other tools as text blocks | real calls to the off Bash tool. Not valid. |
| `ask-miniforge-4` | MCP tools | works. Claude loaded the guiding skill, but not the setup skill, and improvised the install (`rm -rf "$DL"`). |
| `ask-miniforge-5` to `-7` | after the skill change for partial installs (below) | the setup skill loads (3 of 3). But `macos.md` was made up (the `/var` bug). Not valid for the reference commands. |
| `ask-miniforge-8` | system simulator version 2 | the simulator behaved well. `macos.md` still made up. |
| `smoke-remember-1` | "always hide the chat list" | Claude asked "Shall I remember?" again, and loaded no configuring skill. Nothing was written. |
| `ask-miniforge-9` to `-11` | after the `/var` fix | 13, 13 and 12 of 13 checks. Claude runs the reference commands exactly, each alone. No `rm -rf`. The one failure is the crude question counter. |

Skill changes from these runs (in the repository, staged by Nir):

- `CLAUDE.md`: `setting-up-jupylet` is for "install Jupylet, or a part of
  the installation, for example Miniforge or the environment of Jupylet".
- `setting-up-jupylet`: the description names partial installs, and the
  intro says to do only the steps of the part that the person asked for,
  with the checks of steps 1 and 2 first.

Next:

1. A Haiku grader for `judge` checks, and the question check as a judge
   check.
2. The `smoke-remember` finding: a direct request for a setting.
3. `human.md` version 2 ("End where the persona says"), proposed, not yet
   approved.
4. The long evals from `findings.md`, with paired baseline runs.

## Files

- `EVALS.md`: this file.
- `findings.md`: the failures from real sessions, and the proposed evals.
- `worker.md`: appended to the worker's Claude Code prompt. The harness
  fills in the home folder, the scratchpad and the platform, and adds
  `loaded_skills`.
- `simulators/system.md`, `simulators/human.md`: the simulators, with a
  version number.
- `simulators/commands.md`: what each Jupylet command prints.
- `simulators/CHANGELOG.md`: each change of the simulators, and its reason.
- `run_eval.py`: the harness.
- `sim_mcp.py`: the MCP server of the worker's tools.
- `grade.py`: the script grader.
- `token_report.py`: the token and cost report.
- `smoke.json`: the short evals and `ask-miniforge`.
- `library.py`: builds an eval from a scenario and the library.
- `recordings.py`, `simulators/recordings.json`: the real outputs of
  commands and tools, from exported sessions, for the system simulator.
- `simulators/LIBRARY.md`, `simulators/personas/`, `simulators/systems/`:
  the persona and system library, and how to write its files.
- `scenarios/`: the scenarios, one file each.
- `runs/`: the runs. `reports/`: token reports.
- `iteration-N/`: the runs of one round of long evals, as in skill-creator.
