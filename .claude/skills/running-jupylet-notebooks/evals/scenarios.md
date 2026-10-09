# Test scenarios for running-jupylet-notebooks

Method as in the other skills:

- **Claude under test** gets a copy of the skill with only `SKILL.md` and
  `references/` (no `evals/`, no `intent.md`), and is told to read only that
  copy. It writes `ACTION:` lines instead of running tools, and `TO PERSON:`
  for messages.
- **The machine** is the main session. It answers each ACTION from the
  script below. An ACTION that the script does not cover gets `OK` with no
  output.
- **The simulated person** is a second subagent. It plays the persona and
  answers what Claude actually said. It knows nothing about the skill.
- A background command (`run_in_background`) ends when the script says so.
  The machine then sends Claude the message "background task finished" with
  its output, as Claude Code does.
- After the runs, scan the transcripts of the test subagents for any read of
  `evals/` or `scenarios.md`.

## 1. after-setup-signed-out

- **Persona:** a parent with their 10-year-old, on macOS. Polite, not
  technical. Pastes the token when asked, after about a minute.
- **Setup for Claude:** in this conversation, `setting-up-jupylet` has just
  installed Jupylet into the new environment `jupylet2` in
  `/Users/sam/miniforge3/envs`, and the person said yes to opening the
  spaceship example. Code in `/Users/sam/jupylet` (VERSION 0.10.0).
- **Machine script:**
  - `running`: no output
  - `find-env 0.10.0`: two lines, `/Users/sam/miniforge3/envs/jupylet2	conda	this folder`
    and `/Users/sam/miniforge3/envs/jupylet	conda	this folder`
  - `import jupyterlab`, `import jupyter_mcp_server`: no output
  - token: `3f9a1c07`; port check: `61`
  - `prepare`: `off`
  - `detach`: `48213`
  - `wait`: `ready`
  - `preview_start`: OK; sign-in check: `403`; `tabs_context`: pane visible
  - `wait-open` (background): ends with `open` after the person pastes the
    token
  - `mcp__ccd_window__set_sidebar_collapsed` exists; calling it: OK
  - `attach`: `... Successfully activate notebook ...`
  - `wait-change` (background): ends with `ran 7c1e` 40 seconds after
    run-all
  - `call ... notebook_run-all-cells`: `True`
  - `read_notebook` / `read_cell`: every code cell has a new execution
    count, and the last one shows no error
- **Person:** says yes to the sidebar offer.

## 2. restart-kernel-timeout

- **Persona:** 12 years old, types in lowercase. The spaceship ran earlier in
  this session. They pressed Restart Kernel in the page, and now nothing
  moves.
- **Setup for Claude:** Claude started Jupyter in the app earlier in this
  conversation: environment `jupylet`, port 8888, token `5b0d22e1`, notebook
  `11-spaceship.ipynb`, attached.
- **Person's message:** "the ship stopped and nothing works now. can u run it
  again"
- **Machine script:**
  - `call ... notebook_run-all-cells`: `Error executing tool: Timeout waiting
    for result` (after 30 seconds)
  - `read_cell` of the last cell: the old execution count, no output
  - `execute_code` with `print(app.is_running)`: `NameError: name 'app' is
    not defined`
  - screenshot: no canvas
  - `replace-kernel`: `new kernel 9e41...`
  - second run-all: `True`; `wait-change`: `ran 2d4f`; every code cell has a
    new execution count
- **Person:** says yes to replacing the kernel.

## 3. own-jupyter-needs-restart

- **Persona:** an adult beginner who started Jupyter in their own browser
  from the Terminal, as Claude showed them last week.
- **Setup for Claude:** Code in `/Users/alex/jupylet` (VERSION 0.10.0). No
  Jupyter started by Claude in this conversation.
- **Person's message:** "I have the spaceship open in my browser. Can you help
  me change the color of the ship?"
- **Machine script:**
  - `running`: one line, `8888	a1b2c3d4e5f6	/Users/alex/jupylet/examples	jupylet	11-spaceship.ipynb	/Users/alex/miniforge3/envs/jupylet`
  - `import jupyterlab`, `import jupyter_mcp_server`: no output
  - `prepare`: `turned off`
  - `tools`: a list with `notebook_run-all-cells`
  - `shutdown 8888 a1b2c3d4e5f6`: `stopped`
- **Person:** yes, that is the one. Then: "ok, saved it". Then: chooses to
  start Jupyter in the app. The run ends when Claude starts Jupyter (step 3
  in `start-in-app.md`), or asks to.
