# setting-up-jupylet: intent and interview decisions

## Captured intent (confirmed)

1. **What:** install Jupylet on the person's computer (macOS with an Apple
   chip, or Windows 11 on Intel or AMD), from Jupylet code that is already on
   disk. Steps:
   - check the system
   - look for existing conda installations and Jupylet
   - install Miniforge (or update it, on macOS), and set up Terminal on macOS
   - create a new environment
   - `pip install -e {code}[claude]`, then `prepare`
   - trust the example notebooks
   - check the person's own Terminal or Miniforge Prompt
   - open the spaceship example to check that everything works
2. **When:** CLAUDE_SETUP.md hands over after the download; the person asks
   to install Jupylet, or says they downloaded it and asks what now; the
   notebook skill finds Jupylet not set up for this folder.
3. **Output:** a working Jupylet environment, the person told in plain words
   at each step, and the spaceship running.
4. **Tests:** multi-agent simulations with a scripted machine: a fresh Mac, a
   Mac with Miniconda in Terminal, Windows with Miniforge, an Intel Mac, a
   failed download.

## Interview decisions

- **Existing setups:** always use our own Miniforge in `~/miniforge3`, and
  always a new environment. Leave Miniconda, Anaconda and older Jupylet
  environments as they are. On macOS, switch Terminal to Miniforge, with a
  question first and a backup of the settings.
- **End of the skill:** the spaceship check is part of the install. The last
  step uses the skill `running-jupylet-notebooks` to open and run
  `11-spaceship.ipynb` (dependency by instruction).
- **Failures:** when an install step fails, stop. Say that none of the
  person's files were touched, and point to the manual instructions in the
  README. (guiding-jupylet-users rule 3 was changed to allow this: "If you
  have no idea, or the task says to stop, stop.")
- **Code folder:** install from the folder the code is in, wherever it is.
  Do not move it or copy it. Tell the person clearly that this folder is
  their Jupylet folder from now on, so it must stay where it is.
- **Do not offer to do only part of a task** (moved from guiding-jupylet-users):
  when a task cannot be done fully, explain the problem and stop, because a
  later step can need the full task. Example: Miniforge without Terminal
  setup breaks `conda activate` later.
- **Old Miniforge (Python older than 3.11), on any platform:** do not update
  or reinstall it. Tell the person that the Miniforge found is too old for
  Jupylet, and that they should update it or install the latest version
  themselves. Then stop. Reason: rare for a beginner, and an update can affect
  the person's other projects.
- **An environment already runs this folder's code** (installed with
  `pip install -e` from this folder): the person asked to install, so
  install. Tell the person about that environment, by its name, and ask: install
  again into that environment (which also brings it up to date with the code
  in the folder), or create a new environment.
