---
name: managing-jupylet-notebooks
description: Opens and closes Jupyter notebooks of a Jupylet folder for a person. Chooses the notebook, then has the notebook-opener agent start a Jupyter server, open the notebook in the browser pane of the Claude app, next to the chat, and connect to it with the Jupyter MCP tools. Then hands the work in the notebook to working-in-notebooks. Also stops the Jupyter server, and solves known problems. Use this skill when the person asks to open, run or try a notebook or an example. Also use it when the person asks for help with an open notebook or cannot find a notebook web page that they closed. Also use it when the person asks to stop Jupyter. Use it together with guiding-jupylet-users.
---

# Managing Jupylet notebooks

With this skill you open a Jupyter notebook for the person, and close it at
the end. The notebook opens in the Claude web browser, where the person
sees it, and you attach to it. A small background agent, the
`notebook-opener`, does the steps in between: it finds the environment,
starts the Jupyter server, opens the web page and attaches. You decide what
to open, talk with the person, and act on what the opener reports. The work
in the notebook itself, reading, changing and running cells, is in the
skill `working-in-notebooks`. "Understand the request" says which tasks to
do, and in which order. Use the skill `guiding-jupylet-users` for how to
talk with the person and when to ask.

## Tasks

Start with "Understand the request". It says which tasks to do, and in
which order.

Open a notebook:

- "Choose a notebook": decide which notebook to open, and from which folder.
- "Open the notebook": have the opener open it, and act on its notes and
  its report.
- "Use a running Jupyter server": when the notebook is already open in a
  compatible server.
- "Move the notebook to a new Jupyter server": when the notebook is open in
  an incompatible server.
- "Install Jupylet first": when Jupylet is not installed for this folder.
- "Hide the list of chats": make room for the notebook in the Claude app.

Find out what runs:

- "Find the Jupyter servers that run": when the request is about a notebook
  that may still run, and when you stop a Jupyter server.

End:

- "Stop the Jupyter server": stop Jupyter, and close its web page.
- "Start Jupylet without Claude": show the person how to start Jupylet
  alone.

## Values

Values in `{braces}` are blanks that you fill in.

- `{code}`: the Jupylet folder. It is the folder that contains
  `.claude/skills/managing-jupylet-notebooks`. Use its full path.
- `{miniforge}`: the folder `miniforge3` in the home folder of the person.
- `{conda_python}`: the Python of a conda installation. "Find the Jupyter
  servers that run" finds it.
- `{python}`: the Python of the environment of the Jupyter server:
  `bin/python` in it on macOS, `python.exe` in it on Windows.
- `{port}`, `{token}`: the port and the token of the Jupyter server that you
  work with. The report of the opener gives them.
- `{tab_id}`: the Terminal tab that runs the Jupyter server, from the report
  of the opener.
- `{folder}`: the folder that the Jupyter server serves.
- `{notebook}`: the path of the notebook, from `{folder}`, for example
  `11-spaceship.ipynb`.
- `{notebook name}`: the file name without `.ipynb`, for example `11-spaceship`.
- `{scratchpad}`: your scratchpad folder.

## Task and command names

This file uses two kinds of names in quotes:

- **Do "Name":** a task of this file, for example: do "Open the notebook".
- **Run "Name" from the reference file:** a command in `references/macos.md`
  or `references/windows.md`, for example: run "Stop JupyterLab" from the
  reference file.

When a command does not give the expected result, or something behaves
strangely, read `references/troubleshooting.md`.

## Terms

- **Claude app:** the Claude desktop app, where the person chats with you.
- **Jupyter server:** one JupyterLab program that runs. It serves a folder,
  and it has a port and a token.
- **Compatible Jupyter server:** a Jupyter server that runs the Jupylet
  version of the code in `{code}`, and whose `mcp_tools`
  and `nbmodel_off` are `true`. You can attach to its notebooks.
- **Claude web browser:** the web browser in the browser pane of the Claude
  app, next to the chat. You open your web pages here.
- **Default web browser:** the person's own web browser, for example Chrome
  or Safari. You cannot see or use its web pages.
- **Web page:** a JupyterLab web page, in the Claude web browser or in the
  default web browser.
- **The `[claude]` extra:** an optional part of the Jupylet installation, in
  `setup.py`. It installs the Jupyter MCP server for JupyterLab.
- **Monitoring command:** a command that runs in the background and stops
  when something happens in the notebook, for example when a run ends.

## Rules for this skill

- **Opening a web page or attaching runs no cell.** Cells run only in the
  skill `working-in-notebooks`.
- **Start, stop or restart a Jupyter server only when the person asks for it,
  or after a yes.** Stop only a Jupyter server that you started, or that the
  person chose.
- **One web page per server.** A few Jupyter MCP tools, such as
  `notebook_run-all-cells`, work like a click in the menu of the web page:
  they act on whatever notebook the web page shows. If two web pages are open
  on the same server, in the Claude web browser or in the default web browser,
  the server picks one of them, and the tool may run the wrong notebook. So
  keep a single web page for each Jupyter server ("Web page commands" in
  `references/jupyter-mcp-tools.md`).
- **The person signs in, not you.** Give the token to the person. Never type
  it in a web page, and never put it in an address.

## Understand the request

The person wants to open, run or continue a notebook. Use the person's
request to work out what the person most likely wants. The words of the
person alone do not always make it clear. When the request is about a
notebook that may still run, for example "my notebook page is gone" or
"let's continue where we were", first do "Find the Jupyter servers that
run", and ask a short question to make sure, for example:

> I see your spaceship notebook is still running, but no web page shows it.
> Shall I open it again, here next to our chat?

If you cannot work it out, ask. When there are a few possible meanings, name
them.

To open the notebook for the person, first do "Choose a notebook". Then:

- If you know that a compatible Jupyter server already has the notebook
  open, do "Use a running Jupyter server": open the notebook on that server,
  in the Claude app. The notebook keeps its variables.
- If you know that an incompatible Jupyter server has it open, do "Move the
  notebook to a new Jupyter server".
- Otherwise, do "Open the notebook". The opener checks the Jupyter servers
  that run, and reports when one of them already has the notebook open.

When the notebook is open in the Claude web browser, and you are attached to
it, load the skill `working-in-notebooks`, also when the person asked only
to open the notebook. It does what the person asked, for example to run the
notebook, or to work on its code together.

For example, the person says "run the spaceship". Choose
`11-spaceship.ipynb`, and do "Open the notebook". When the opener reports
that the notebook is ready, load `working-in-notebooks`, which runs it.

**If the person wants to stop Jupyter,** do "Stop the Jupyter server". You
do not need a web page for this.

## Choose a notebook

This task decides which notebook to open. Its result is the full path of an
existing notebook, or the short name of a new one. A new Jupyter server
serves the folder of the notebook.

Jupylet comes with example notebooks, in the folder `{code}/examples`, for
example `11-spaceship.ipynb` and `14-piano.ipynb`. The person's own notebooks
can be anywhere on the computer, often in `{code}/examples` or in another
folder under `{code}`.

If another skill sent you here with a notebook, use that notebook.

If the person gives the full path of a notebook, use that file.

If the person names a notebook in their own words, for example "the
spaceship" or "my game", look for the files whose names fit. Look in `{code}`
and the folders under it, and in the folders of the global setting
`notebook-folders`. Read that setting with the skill `configuring-jupylet`.
Its value is a list of folders, separated by `;`. If one file fits, use it.
If more than one file fits, ask which one, for example:

> I found two notebooks that could be your game, my-game and my-game-2.
> Which one shall we open?

If no file fits, ask the person to choose the notebook in a file window.
Run "Browse notebooks" from the reference file, and tell the person, for
example:

> I could not find your game. I opened a window where you can find it and
> choose it.

If the person chooses a notebook, use it. If they close the window without a
choice, ask where the notebook is.

If the person asks to start or open a notebook, and names none, open a new
empty notebook named `my-notebook`. That is what such a request means, so do
not ask which notebook.

If the person wants to make something new, open a new notebook for it, and
name it after what they want to make, for example `my-game`. Do not ask for
a name: the person can rename the notebook at any time. The opener creates
the notebook in `{code}/examples`, and adds `-2` to the name if that name is
taken.

If the person wants to try something, but names no notebook, for example
"show me what Jupylet can do", suggest an example notebook. Ask what they
like, for example music, 2D graphics or 3D games, and suggest a notebook for
it. For a first try, suggest the spaceship, `11-spaceship.ipynb`.

If Jupyter is new to the person, explain it in one or two sentences when you
suggest the notebook, for example:

> Shall we start with the spaceship? It is a Jupyter notebook: a page where
> you type code in small boxes, called cells, and run each one to see what
> it does.

If the chosen notebook is not under `{code}`, and its folder is not in
`notebook-folders` yet, add the folder to the list. The new value is the old
list, then `;`, then the new folder. Write the global setting with the skill
`configuring-jupylet`. Next time, you will find notebooks in that folder
without asking.

## Open the notebook

This task has the `notebook-opener` open the notebook. The opener is a small
agent that runs in the background. It finds the environment, starts the
Jupyter server, opens the web page and attaches, while you remain free to
talk with the person. It never talks to the person itself: it sends you a
note when the person has to do something, and it ends with a report. The
person's request to open the notebook covers starting the opener.

1. Start the opener. In one reply, tell the person in one line what
   happens, for example "Starting Jupyter now, one second.", and call the
   tool `Agent` with:
   - `name`: `notebook-opener`
   - `subagent_type`: `notebook-opener`
   - `run_in_background`: `true`
   - `description`: "Open the notebook"
   - `prompt`: the brief:

     ```
     Job: {the job}.
     Values: code "{code}", scratchpad "{scratchpad}", computer {macOS or Windows}.
     ```

     The job is one of these:
     - `a notebook in a new Jupyter server, {full path of the notebook}`
     - `a new notebook in a new Jupyter server, named {short name}`
     - `a notebook in a running Jupyter server, {full path of the notebook}, port {port}, token {token}`

   Then end your turn.
2. Act on the notes of the opener. Each note arrives as a new turn. It comes
   from the opener, not from the person, so it never answers a question that
   you asked the person.
   - **`needs the person: ...`:** ask the person for everything that the
     note names, in a single message, with one numbered step for each item,
     in this order:
     1. **`pane hidden`:** ask the person to click the globe icon in the
        upper right corner of the Claude app.
     2. **`notebook open in the default web browser`:** ask the person to
        close that tab, and say why, for example: "with two web pages open,
        I could end up running the wrong notebook." When the person says
        that they closed it, tell the opener with `SendMessage`, to
        `notebook-opener`.
     3. **`sign-in needed`:** give the person the token, for example: "This
        web page wants a special token just for this session. Please paste
        this into the box on the web page, and press Enter: `{token}`."

     For example:

     > Your notebook is ready, next to our chat.
     > 1. Click the globe icon in the upper right corner of the Claude app,
     >    to show it.
     > 2. On the web page, paste this token into the box, and press Enter:
     >    `d2f22990`

     The opener waits for the person. End your turn.
   - **`sign-in failed: ...`:** tell the person in one line that the token
     did not match, and give it again.
3. Act on the report of the opener. It arrives as a new turn, and it starts
   with one word:
   - **`ready:`** the notebook is open, and you are attached. Keep `{port}`,
     `{token}` and `{tab_id}`. Pass on to the person each thing that the
     report says to tell them. If the report gives the list of chats as `no
     setting` or `no tool`, do "Hide the list of chats". Then load
     `working-in-notebooks`, as "Understand the request" says.
   - **`stopped: already open, ...`:** a Jupyter server already has the
     notebook open. If the report says that it is compatible, do "Use a
     running Jupyter server". If not, do "Move the notebook to a new Jupyter
     server". The report gives the port and the token of that server.
   - **`stopped: not installed, ...`:** do "Install Jupylet first".
   - **`stopped: not prepared, ...`:** the opener could not change the
     Jupyter settings of the environment, so it did not start the Jupyter
     server. Tell the person in plain words, and follow
     `references/troubleshooting.md`.
   - **`stopped: still waiting, ...`:** the person did not show the pane or
     sign in within 5 minutes, and the Jupyter server still runs. Ask the
     person what is in the way. When it is solved, do this task again with
     the job `a notebook in a running Jupyter server`, with the port and the
     token of the report.
   - **`problem: ...`:** follow the part of `references/troubleshooting.md`
     that the report names, or the whole file if it names none.

## Use a running Jupyter server

Use this task when a compatible Jupyter server already has the notebook
open. (A notebook in an incompatible server has its own task: "Move the
notebook to a new Jupyter server".)

The person does not know about ports or Jupyter servers, so talk about the
notebook, and ask a short question to make sure. For example, the person says
"let's continue with the spaceship", and a Jupyter server has
`11-spaceship.ipynb` open:

> Your spaceship notebook is still open. Shall I connect to it, so we can
> work on it together?

**If several Jupyter servers fit,** name each one by its notebook, and ask
which one, for example:

> I see two notebooks open, the spaceship and Hello Jupylet. Which one do you
> mean?

Take `{port}` and `{token}` from the report of the opener, or from "Find the
Jupyter servers that run". Then do "Open the notebook", with the job `a
notebook in a running Jupyter server`. The opener also checks that no web
page in the default web browser shows the notebook, and its note tells you
when the person has to close one.

## Move the notebook to a new Jupyter server

Use this task when the Jupyter server that has the notebook open is
incompatible:

- **If its `jupylet_version` is not the version of the code in `{code}`,** it runs another version
  of Jupylet, or no Jupylet, or you could not find its environment.
- **If its `mcp_tools` is `false`,** the `[claude]` extra is missing, or the
  server started before the extra was installed.
- **If its `nbmodel_off` is `false`,** a cell can stop responding after a
  few minutes.

Offer to open the notebook again in a new Jupyter server, for example:

> Your spaceship notebook is open in a Jupyter server that I can't connect to
> (a Jupyter server is the program behind the notebook). I can close it
> and open the notebook again in a new one. Please save the notebook first.
> Shall I?
After a yes, take `{port}` and `{token}` of that server from the report of
the opener, or from "Find the Jupyter servers that run". Then:

1. Do "Stop the Jupyter server".
2. Do "Open the notebook", with the job `a notebook in a new Jupyter
   server`. If the environment lacks JupyterLab or the extra, the opener
   reports it, and `references/troubleshooting.md` says how to install it
   with the person's agreement.

## Install Jupylet first

Tell the person that Jupylet must be installed first, and that after that you
will continue with what they asked. Ask for a yes, for example:

> To open the music notebook, Jupylet needs to be installed first. Shall I
> install it? It takes a few minutes, and then we'll open the notebook.

- **If the person says yes,** use the skill `setting-up-jupylet`. It ends with
  the spaceship example in the Claude web browser, or with no Jupyter server
  if the person said no to that example. Then do "Understand the request"
  again, for what the person asked.
- **If the person says no,** tell them that the notebooks need Jupylet
  installed, and that they can ask at any time. Then stop.

## Find the Jupyter servers that run

First find a Python that can run the command. Run "Find conda
installations" from the reference file. It finds the Python of each conda
installation. The Python of `{miniforge}` is `{conda_python}`, or else the
first one that it finds. Do not use `python3` or `python` of the system: on
a computer without Python, they can be stubs that only start an
installation. If it finds nothing, Jupylet is not installed: do "Install
Jupylet first".

Then run "List the Jupyter servers" from the reference file. It prints a JSON
list, with one item for each Jupyter server that runs, for example:

```json
[
  {
    "port": 8888,
    "token": "3f9a1c07",
    "folder": "/Users/sam/jupylet/examples",
    "environment": "/Users/sam/miniforge3/envs/jupylet",
    "jupylet_version": "0.10.0",
    "mcp_tools": true,
    "nbmodel_off": true,
    "notebooks": [
      {"path": "11-spaceship.ipynb", "connections": 1},
      {"path": "02-hello-jupylet.ipynb", "connections": 0}
    ]
  }
]
```

- `folder`: the folder that the Jupyter server serves.
- `environment`: the environment of the Jupyter server, or `null` if the
  script cannot find it.
- `jupylet_version`: the version of Jupylet in that environment, or `null`.
- `mcp_tools`: `true` if the Jupyter server answers with its Jupyter MCP tools.
  That shows that the Jupyter server loaded the `[claude]` extra.
- `nbmodel_off`: `true` if `jupyter_server_nbmodel` is off in the Jupyter
  server. With it on, a cell can stop responding after a few minutes.
- `notebooks`: the notebooks that are open in it. `connections` is the number
  of web pages and other clients that connect to the notebook. `0` means that no
  web page shows the notebook now, for example because the person closed it.

Run it again each time that you need its result: it can change at any time.

The `environment` of a Jupyter server gives `{python}` for the commands
that act on it, such as "Stop JupyterLab".

## Hide the list of chats

The opener hides the list of chats itself when the setting `hide-chat-list`
is `yes`. Do this task when its report gives the list of chats as `no
setting` or `no tool`.

- **`no tool`:** the tool `mcp__ccd_window__set_sidebar_collapsed` does not
  exist here. Tell the person how to hide the list themselves, as in the
  example below.
- **`no setting`:** ask, once in a conversation, for example:

  > The notebook is a bit cramped. Want me to tuck the list of chats on
  > the left out of the way, so there's more room for it?

  After a yes, hide it with the tool
  `mcp__ccd_window__set_sidebar_collapsed`, with `collapsed` set to true.
  When the tool list shows this tool only by its name, load it with
  `ToolSearch`. Then tell the person how to show it again, for example:

  > Done. You can bring it back any time by clicking the small panel icon
  > in the top-left corner of the Claude app, just right of the three
  > lines.

  After the answer, ask if you can remember it. After a yes, write boxless
  the global setting `hide-chat-list` (`yes` or `no`) with the skill
  `configuring-jupylet`.

Do not show the list of chats again yourself.

## Stop the Jupyter server

Do this only when the person asks for it, or after a yes. Stop only a Jupyter
server that you started, or that the person chose. It stops each kernel of
that Jupyter server.

**If more than one Jupyter server runs, and the request does not name one,**
name the one that you want to stop, by its notebook, and wait for a clear
yes. For example: "Shall I close the Jupyter server with your spaceship
notebook?" A short word such as "shutdown" does not choose one.

Take `{port}`, `{token}` and `{tab_id}` from the report of the opener. For a
Jupyter server that the opener did not start, take them from "Find the
Jupyter servers that run".

1. Stop each monitor and each monitoring command that still runs, with the
   tool `TaskStop`. This includes the `notebook-opener`, if it still runs.
2. Tell the person in one line, for example "Closing Jupyter now."
3. Close the web page in the Claude web browser with
   `mcp__Claude_Browser__tabs_close` (`tabs_context` gives its id): an open
   web page shows an error when the Jupyter server stops.
4. Run "Stop JupyterLab" from the reference file. It takes a few seconds.
   - **If it prints `stopped`, `not running` or `stopped after ending its
     process`,** continue.
   - **If it prints anything else,** follow "The Jupyter server does not
     stop correctly" in `references/troubleshooting.md`.
5. If you started the Jupyter server in a tab of the Terminal panel, close
   that tab with `mcp__terminal__stop_terminal_tab`, with its `tab_id` and
   `close` set to `true`. After the Jupyter server stops, the tab shows only
   an empty prompt. If you do not have the `tab_id`,
   `mcp__terminal__list_terminal_tabs` gives it: the tab has the title
   `Jupyter`.
6. Tell the person in one line, for example "All closed. Just ask when you
   want to open it again."

## Start Jupylet without Claude

This task shows the person how to start Jupylet alone, in Terminal or in the
Miniforge Prompt. When the person asks for it, read
`references/start-without-claude.md`. It has the whole task.
