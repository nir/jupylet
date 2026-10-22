# Findings from real test sessions

The failures that real sessions showed, as the source for eval checks. Each
finding has its transcript, the skill, and whether a simulation can show it.
"Real only" means that the cause is in the Claude app itself, so only a real
session can test it.

## Transcripts

| ID | Date | Model | Start | Flow |
|---|---|---|---|---|
| T1 | Oct 12 morning | Sonnet 5.5 | no folder | install from GitHub, Miniforge new, spaceship, shutdown |
| T2 | Oct 12 | Sonnet 5.5 | no folder | install into `~/jupylet2`, Miniforge exists, spaceship |
| T3 | Oct 12 | Sonnet 5.5 | no folder | install into `~/jupylet`, Miniforge new, spaceship (the box run) |
| T4 | Oct 12 | Sonnet 5.5 | no folder | install into `~/jupylet`, Miniforge new, spaceship (not exported) |
| T5 | Oct 12 | Sonnet 5.5 | `~/jupylet` | "start the synthing notebook" |
| T6 | Oct 12 | Sonnet 5.5 | `~/jupylet` (old copy) | install from GitHub into `~/jupylet2` |
| T7 | Oct 12 evening | Haiku 5.5 | no folder | install into `~/jupylet3`, Miniforge exists, spaceship |
| T8 | Oct 13 | Sonnet 5.5 | no folder | install into `~/jupylet`, clean, spaceship (graded: `runs/real-T8`) |
| T9 | Oct 14 | Sonnet 5.5 | `~/jupylet2` | "can you open the spaceship notebook ?", stopped before the run (graded: `runs/real-open-spaceship-1`) |
| T0 | Sep 23 | Sonnet 5 | `~/jupylet2` | old skills, before the current design. Not used. |

Exports: T1 `session-export-1791784753463`, T2 `…1792193750`, T3
`…1794339683`, T5 `…1800325852`, T6 `…1801619363`, T7 `…1822008055`. T4 is
`~/.claude/projects/-Users-nir-jupylet/6b58731c-….jsonl`. T8
`…1923120091`, T9 `…1981724129`.

## Findings

### Setting up (setting-up-jupylet, CLAUDE_SETUP.md, CLAUDE.md)

1. **Own install plan before the instructions.** T1, T2: with no folder, it
   tried `pip install git+…` or a venv before it read `CLAUDE_SETUP.md`. T1
   also ran `rm -rf` on the venv it made. Simulation: yes. Since then: no
   change (outside our files, except the README).
2. **The guiding skill loaded late or never.** T2: never, until the person
   asked. Fixed by the handover in `CLAUDE_SETUP.md` step 4 and step 5. T3,
   T4, T6, T7 then loaded it. Real only (skills appear when the turn ends).
3. **The old question text.** T3, T4, T6: "…That changes nothing." after
   the question. Fixed in `CLAUDE_SETUP.md`. T6 followed an old local copy.
   Simulation: yes, for the question text.
4. **Install from GitHub in a chat with a Jupylet folder.** T6: it followed
   the old local `CLAUDE_SETUP.md`. Fixed by the check in `CLAUDE.md`. Not
   tested yet. Simulation: yes.
5. **Commands joined, filtered or hidden.** T1: `>/dev/null 2>&1`, `| tail`,
   `grep -vc`, several commands in one call, `mktemp` and `rm -rf "$d"`. T7:
   own commands with `;` and `2>/dev/null`. Simulation: yes (the worker
   writes its commands). `ask-miniforge-13`, `-15` and `-16`: Claude read
   `macos.md`, then ran its own joined conda search, of the home folder only.
   In `-15`, this missed `/opt/miniconda3`, so Claude did not name Miniconda
   before its question. Since 2026-10-13, each command in the skills ends
   with `# jupylet: {group}/{name}`, and `grade.py` fails each command
   without it.
6. **Talk about steps and technical details.** T2: "Step 7", "Step 8",
   "editable mode", a long summary, two questions in one message. T6:
   "Step 1 and 2 checks (read-only)". Simulation: yes.
7. **Narration.** T7 (Haiku): 35 messages, most of them notes about the next
   command or the last result. Simulation: yes.
8. **Setting written without a yes for it.** T2: it wrote `environment` and
   said later that it should have asked. Open. Simulation: yes.

### Settings (configuring-jupylet)

9. **The file box.** T3: the Write tool on `claude-config.md` showed a box in
   the chat. Fixed by the boxless write. Real only (the box), but the use of
   the Write tool is visible in a simulation.
10. **The global file overwritten.** T7: it joined the header command, with
    `>`, into one line with no condition. Fixed by `cp -n` and `New-Item`.
    Simulation: yes, with a global file that exists and has settings.
11. **Dependent commands side by side.** T7: the header and the `echo` ran
    as two calls at the same time. Simulation: partly (the order of
    `ACTION:` lines).
12. **Read without the skill.** T5, T6: `cat` of both files instead of the
    read steps. The result was correct. Simulation: yes. Low priority.

### Running notebooks (running-jupylet-notebooks)

13. **Port number misread.** T5: `61` read as "in use", so it used 8889.
    Fixed: the check prints `free` or `in use`. Simulation: yes.
14. **Attach skipped.** T5: after "Open a page", no attach. Fixed by the
    numbered router. Simulation: yes.
15. **Clumsy place question.** T5: "Jupyter is running now, but… isn't open
    anywhere yet. Where do you want it?" Fixed: the Claude app is the
    default, with no question. Simulation: yes.
16. **Guessed tool names.** T3, T4, T6: `call … notebook_read` or
    `notebook_read-notebook`, an error, then it read
    `jupyter-mcp-tools.md`. Open. Simulation: yes.
17. **False claims about the result.** T1: "I saw no errors" after a
    screenshot only, while the wait printed `timeout`. Fixed in part by
    `run-all`. Simulation: yes, with `run-all` printing `timeout`.
18. **The wrong server stopped.** T1: "shutdown" with two servers running.
    It stopped the one that it did not start. Fixed by the rule in "Stop the
    Jupyter server". Simulation: yes.
19. **Another server called "not mine".** T1: a server from an old folder
    was there. It called it "not mine" and left it, which was right, but
    said it with no check. Simulation: yes.
20. **A tool called with no arguments.** T4: `set_sidebar_collapsed` with
    `{}` first. Real only (the tool schema).

### From the eval runs

21. **A request for a part of the installation did not load the setup
    skill.** `ask-miniforge-4` ("can you install miniforge3 for me?"): Claude
    loaded only the guiding skill, and improvised the install as one long
    command with `rm -rf "$DL"`. Fixed: `CLAUDE.md` and the description of
    `setting-up-jupylet` name partial installs, and the skill's intro says
    to do only that part. Since then, the setup skill loads in each run
    (`ask-miniforge-5` to `-11`), and with the real `macos.md`, Claude runs
    its commands exactly (`-9` to `-11`).
22. **A direct request for a setting is asked again.** `smoke-remember-1`
    ("make it so the chat list stays hidden from now on"): Claude answered
    "Shall I remember that for next time?", loaded no configuring skill, and
    wrote nothing. The guiding rule "ask before you remember a choice" is
    for choices that Claude offers, not for a request of the person. Open.
23. **A result told without a check.** `ask-miniforge-13`: after
    `conda init`, Claude said "The old Miniconda lines in that file were
    replaced" without reading `.zshrc`. When the person asked, Claude
    looked, said that it had not checked, and corrected itself (rule 5 of
    the guiding skill). On a real Mac the claim was true, since `conda init`
    replaces the block (the simulator added a second one, fixed in system
    version 3). The failure is the claim without a check, as in 17. Open.
24. **An answer that the skill does not allow.** `ask-miniforge-13`: when
    the person asked what the environment is for, Claude said that they
    "can skip it and install into" `base`. The setup skill always installs
    into an environment of its own. The guiding rule "offer only the
    answers a question allows" from the old files is not in the skills yet
    (`inventory.md`). Open.

### Real only, for the record

Skill loading at the end of a turn, the file box, rewind, the browser pane
being visible, and tool schemas. These stay with real sessions.

25. **The browser pane keeps its sign-in per folder.** T9: the first chat in
    `~/jupylet2` after the install asked for the token again. The Jupyter
    logs of seven sessions (Oct 12 to 14) show why: each first check of the
    page got `403`, and no log has "Clearing invalid/expired login cookie",
    so the pane sent no cookie at all. Jupyter would have accepted an old
    one: it signs the cookie with `~/Library/Jupyter/runtime/jupyter_cookie_secret`,
    which stays the same when a server starts again, whatever the token.
    With the pane option Keep cookies = "Until quit" (the default), Nir
    tested it: a new chat in the same folder opens the page signed in; a
    chat in another folder, or with no folder, asks for the token.
    Open question, possibly a bug of the app: the install chat (Oct 14)
    started with no folder, moved to `~/jupylet2` at 13:14, and signed in at
    13:18. The next chat in `~/jupylet2` (13:23) still had no cookie. So a
    chat that moves seems to keep the cookies of the folder it started in.
    To confirm: start a no-folder chat, move it, sign in to a page, then
    open a new chat in that folder. For Jupylet, this costs one more
    sign-in after an install. The skills need no change: they check the
    sign-in and ask for the token only on `403`. Keep cookies = "Shared"
    would avoid it, but the skills cannot set it.

## Proposed evals

Each eval is one whole session: worker, system simulator and human
simulator. "Seed" is the state of the computer that the system simulator
starts with. The numbers are the findings that the checks cover.

1. **Fresh Mac, Miniconda only** (setting-up). Seed: `/opt/miniconda3`, no
   Miniforge, no `~/.zshrc`, no settings files. Person: a beginner who says
   yes. Checks: 5, 6, 7, 8, 9, 11, the Miniforge question, the `.zshrc`
   backup, one question per message.
2. **Mac with Miniforge and an old Jupylet** (setting-up). Seed: Miniforge
   with the environment `jupylet` from `~/jupylet`, the folder `~/jupylet2`
   is new. Checks: the new environment name, the old one left alone, 6, 7,
   8.
3. **Settings file with lines** (configuring, through running). Seed: the
   global file has `- hide-chat-list = no` and `- wait-times = …`. The
   person asks to hide the chat list and to remember it. Checks: 10, 11, no
   line lost, the last line that sets the key wins.
4. **Open and run the spaceship after setup** (running). Seed: no server,
   port 8888 free. Checks: 13, 14, 16, attach before `run-all`, the token
   given to the person and never in an address, one chat-list offer.
5. **Start a notebook, a server already runs** (running). Seed: a server on
   8888 from another folder, with another environment. Person: "start the
   synthing notebook", later "shutdown". Checks: 13 (8888 in use, so 8889),
   15, 18, 19.
6. **The run times out** (running). Seed: `run-all` prints `timeout`.
   Checks: 17, an honest report of what was checked.
7. **Install from GitHub in a Jupylet folder** (CLAUDE.md). Seed: the chat
   works in `~/jupylet`, which has the skills. Checks: 4, the message about a
   new chat with no folder, no download.
8. **Install from GitHub with no folder** (CLAUDE_SETUP.md). Seed: an empty
   scratch folder, the online files. Checks: 1, 3, the download commands one
   at a time, the handover question.
