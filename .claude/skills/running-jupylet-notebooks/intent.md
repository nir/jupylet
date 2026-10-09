# running-jupylet-notebooks: intent and interview decisions

## Captured intent (confirmed)

1. **What:** a set of capabilities for Jupyter and notebooks in a Jupylet
   folder, chosen by the situation, not one fixed sequence. For example:
   - find the Jupyters that run now, and connect to one that the person
     started in their own browser
   - start Jupyter in the app, next to the chat
   - open, run and edit a notebook: read and edit cells, move between cells,
     run one cell or all cells, read outputs and errors, take a screenshot of
     the canvas
   - watch what the person does in the notebook, and stay free to answer the
     chat
   - stop Jupyter, start over, recover from known problems
2. **When:** any task that needs Jupyter or a notebook in a Jupylet folder,
   from starting it, through working in it with the person, to stopping it.
   `setting-up-jupylet` uses this skill in its step 8 to open and run the
   spaceship example: it does not hand over control.
3. **Output:** no file. The state of Jupyter and the notebook as the person
   sees it, with short plain messages, as `guiding-jupylet-users` describes.
4. **Tests:** simulated conversations, as for the other skills. Write the
   scenarios after the draft. Run them only after an explicit go. Real
   Jupyter behavior (timing, the browser pane, hangs) needs tests by hand.

## Interview decisions

- **Shape:** `SKILL.md` is a short map from situation to section. Each
  capability is a section or a reference file.
- **The `[claude]` extra missing:** an install, so it belongs to
  `setting-up-jupylet`. This skill tells the person and offers setup.
- **Environments:** any conda environment that `find-env` reports as
  `this folder`, `base` and other conda installations included. No venv for
  now (never tested, and setup never makes one). If several environments
  have `this folder`, list them and suggest the most recently set up one.
  Exception: right after setup installed into an environment in this
  conversation, use that one without a question.
- **No environment has `this folder`:** tell the person that Jupylet is not
  set up for this folder, and offer to set it up. Only after a yes, use
  `setting-up-jupylet`.
- **A Jupyter the person started:** run `running`. If one or more Jupyters
  with Jupylet run, list them by notebook, folder and environment, suggest
  the best guess, and ask if that is the one. If it can take the connection,
  connect. If it needs a restart, explain why, ask the person to save and to
  say when to continue, then close it. Never start a second Jupyter on the
  same folder. Then offer to start Jupyter in the app, or let the person
  start it and connect to that.
- **Start in the app:** keep the tested procedure (token, port and
  `prepare`, start detached, wait, open the page, the person signs in,
  attach, run all). Rewrite only the wording.
- **Sidebar:** keep the offer to hide the list of chats after sign-in.
- **Which notebook:** the one the person or setup names. Otherwise list the
  examples, or ask what the person is interested in (music, 2D graphics,
  3D). Keep this part loose for now.
- **Waiting for the person:** the reviewed text, now in
  `references/waiting.md`. It may be tuned later.
- **Start Jupylet on their own:** keep the offer after the first run, and
  the instructions when the person asks. On macOS, first check that Terminal
  finds Miniforge. If not, say so, explain that Terminal must be set up, and
  offer to do it with step 4 of `setting-up-jupylet` ("Prepare Terminal").
- **Stopping:** stop Jupyter only when the person asks for it, or after a
  yes. Close the browser tab first.
- **Starting over:** keep it (stop, list the state files, delete them after
  a yes, start again). Notebooks are never touched.
- **Known problems:** all in `references/troubleshooting.md`. Drop Problem 3
  (raw calls only) and Problem 13 (`.mcp.json`). Problems 14 and 17
  (Jupylet, JupyterLab or the extra missing): tell the person and offer
  setup.
- **Windows:** `references/windows.md`, as in setup. Keep a "not tested yet"
  note only where it changes what Claude does.
- **Running one cell on macOS:** add the allowlist option to the macOS start
  command, marked "not tested on macOS yet".
- **Teaching** (first lesson, red error box, a typo with no error): left for
  `teaching-jupylet`. Notebook control stays here.
- **Screenshots:** only when a cell's output is graphical (the canvas, an
  image). Never for text output.
- **Watching the person:** only when the context calls for it, for example
  after you ask them to try something. Not all the time.
