---
name: managing-jupylet-notebooks
description: Opens and closes Jupyter notebooks of a Jupylet folder for a person. Finds the environment, starts a Jupyter server, opens the notebook in the browser pane of the Claude app, next to the chat, and connects to it with the Jupyter MCP tools. Then hands the work in the notebook to working-in-notebooks. Also stops the Jupyter server, and solves known problems. Use this skill when the person asks to open, run or try a notebook or an example. Also use it when the person asks for help with an open notebook or cannot find a notebook web page that they closed. Also use it when the person asks to stop Jupyter. Use it together with guiding-jupylet-users.
---

# Managing Jupylet notebooks

With this skill you open a Jupyter notebook for the person, and close it at
the end. You open the notebook in the Claude web browser, where the person
sees it, and attach to it. The work in the notebook itself, reading,
changing and running cells, is in the skill `working-in-notebooks`.
"Understand the request" says which tasks to do, and in which order. Use
the skill `guiding-jupylet-users` for how to talk with the person and when
to ask.

## Tasks

Start with "Understand the request". It says which tasks to do, and in
which order.

Find out what is on the computer:

- "Find a Python": find a Python that can run the first commands.
- "Find the Jupyter servers that run": find any Jupyter servers that
  already run.
- "Find the environment": choose the environment for a new server.

Open a notebook:

- "Use a running Jupyter server": when the notebook is already open in a
  compatible server.
- "Move the notebook to a new Jupyter server": when the notebook is open in
  an incompatible server.
- "Choose a notebook": decide which notebook to open, and from which folder.
- "Start a Jupyter server": start a new server for the folder of the
  notebook.
- "Open a web page": open the notebook in the Claude web browser, next to the
  chat.
- "Hide the list of chats": make room for the notebook in the Claude app.
- "Attach to the notebook": connect to the notebook, so that you can read
  and run its cells.

End:

- "Stop the Jupyter server": stop Jupyter, and close its web page.
- "Start Jupylet without Claude": show the person how to start Jupylet
  alone.

## Values

Values in `{braces}` are blanks that you fill in.

- `{code}`: the Jupylet folder. It is the folder that contains
  `.claude/skills/managing-jupylet-notebooks`. Use its full path.
- `{version}`: the Jupylet version of the code in `{code}`.
- `{miniforge}`: the folder `miniforge3` in the home folder of the person.
- `{conda_python}`: the Python of a conda installation. "Find a Python" finds
  it.
- `{env}`: the folder of the environment that runs Jupylet.
- `{env name}`: the name of that environment, the last part of `{env}`. For
  the main environment of a conda installation, it is `base`.
- `{conda}`: the folder of the conda installation that has `{env}`: the folder
  above `envs`, or `{env}` itself for `base`.
- `{python}`: the Python of `{env}`: `bin/python` in it on macOS, `python.exe`
  in it on Windows.
- `{port}`, `{token}`: the port and the token of the Jupyter server that you
  work with.
- `{folder}`: the folder that the Jupyter server serves. For a Jupyter server
  that runs, it is its `folder`. For a new one, "Choose a notebook" gives it.
- `{notebook}`: the path of the notebook, from `{folder}`, for example
  `11-spaceship.ipynb`.
- `{notebook name}`: the file name without `.ipynb`, for example `11-spaceship`.
- `{scratchpad}`: your scratchpad folder.

## Task and command names

This file uses two kinds of names in quotes:

- **Do "Name":** a task of this file, for example: do "Attach to the notebook".
- **Run "Name" from the reference file:** a command in `references/macos.md`
  or `references/windows.md`, for example: run "Make a token" from the
  reference file.

When a command does not give the expected result, or something behaves
strangely, read `references/troubleshooting.md`.

## Terms

- **Claude app:** the Claude desktop app, where the person chats with you.
- **Jupyter server:** one JupyterLab program that runs. It serves a folder,
  and it has a port and a token.
- **Compatible Jupyter server:** a Jupyter server that runs Jupylet
  `{version}`, the version of the code in `{code}`, and whose `mcp_tools`
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

The person wants to open, run or continue a notebook. First, run "Read the
version" from the reference file to get `{version}`. Then do "Find a Python"
and "Find the Jupyter servers that run". Then use the person's request
and what you found to work out what the person most likely wants. The words of
the person alone do not always make it clear. Ask a short question to make
sure, for example:

> I see your spaceship notebook is still running, but no web page shows it.
> Shall I open it again, here next to our chat?

If you cannot work it out, ask. When there are a few possible meanings, name
them.

To open the notebook for the person, first do "Choose a notebook". Then:

- If a compatible Jupyter server already has the notebook open, do "Use a
  running Jupyter server": open the notebook on that server, in the Claude
  app. The notebook keeps its variables.
- If an incompatible Jupyter server has it open, do "Move the notebook to a
  new Jupyter server".
- If no Jupyter server has it open, do "Find the environment", "Start a
  Jupyter server", "Open a web page" and "Attach to the notebook", in this
  order.

When the notebook is open in the Claude web browser, and you are attached to
it, use the skill `working-in-notebooks` for what the person asked, for
example to run the notebook, or to work on its code together.

For example, the person says "run the spaceship" and no Jupyter server runs.
Choose `11-spaceship.ipynb`, find the environment, start a Jupyter server,
open a web page, attach, and then run the notebook.

**If the person wants to stop Jupyter,** do "Stop the Jupyter server". You
do not need a web page for this.

## Find a Python

The first commands need only Python 3, so the Python of each conda installation
can run them. Do not use `python3` or `python` of the system: on a computer
without Python, they can be stubs that only start an installation (the
Microsoft Store on Windows, or the Command Line Tools on macOS).

Run "Find conda installations" from the reference file. It finds the Python
of each conda installation.

- **If it finds the Python of `{miniforge}`,** that Python is
  `{conda_python}`.
- **If it finds only other conda installations,** for example Anaconda, the
  first Python that it finds is `{conda_python}`.
- **If it finds nothing,** the computer has no conda installation, and
  Jupylet is not installed in a way that this skill can use. Do "Install
  Jupylet first", below.

### Install Jupylet first

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

Run "List the Jupyter servers" from the reference file. It prints a JSON
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

Take `{port}`, `{token}`, `{env}` and `{folder}` from "Find the Jupyter servers
that run". Then:

1. Make sure that no web page in the default web browser shows the
   notebook. Call `mcp__Claude_Browser__tabs_context`. If the notebook has
   a connection (`connections` is more than `0`), but no tab of the Claude
   web browser shows this server (`localhost:{port}`), a web page in the
   default web browser shows it. Ask the person to close it, and say why,
   for example:

   > Your notebook is also open in your web browser. Please close that tab:
   > with two web pages open, I could end up running the wrong notebook.

   Continue when the person says that they closed it.
2. Do "Open a web page".
3. Do "Attach to the notebook".

## Move the notebook to a new Jupyter server

Use this task when the Jupyter server that has the notebook open is
incompatible:

- **If its `jupylet_version` is not `{version}`,** it runs another version
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

After a yes, take `{port}`, `{token}` and `{folder}` from "Find the Jupyter
servers that run". Then:

1. Do "Find the environment". It checks the extra, and installs it if it is
   missing and the person agrees.
2. Do "Stop the Jupyter server".
3. Do "Start a Jupyter server".
4. Do "Open a web page".
5. Do "Attach to the notebook".

## Find the environment

This task chooses the conda environment for a new Jupyter server: one that
runs the Jupylet code in `{code}`, with JupyterLab and the `[claude]` extra.

Run "List the environments" from the reference file. The suitable
environments are those whose kind is `conda` and whose source is `this
folder`. The others run another copy of Jupylet, or are a venv, which this
skill does not use.

If no environment is suitable, Jupylet is not installed for the Jupylet
folder yet. Do "Install Jupylet first", in "Find a Python".

Otherwise, read the setting `environment` with the skill
`configuring-jupylet`. It names the environment chosen for the Jupylet
folder. Use that environment if it is suitable, or else the first suitable
one in the list, which is the best. Its path is `{env}`. When you talk to
the person, call it by its name.

Then run "Check JupyterLab and the extra" from the reference file. If it
fails, follow "Jupylet, JupyterLab or the extra is missing" in
`references/troubleshooting.md`.

## Choose a notebook

This task decides which notebook file to open. Its result is the full path of
the file. Two values come from that path: `{folder}`, the folder that has the
file, and `{notebook}`, the file name. A new Jupyter server serves `{folder}`.

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

If the person wants to make something new, create a blank notebook for it.
Ask for a short name, or suggest one from what they want to make, for
example `my-game.ipynb`. Create it in `{code}/examples`. If a file with that
name is already there, choose another name with the person. Write the file
with the Write tool, as a notebook with no cells. The `kernelspec` names the
Python kernel of the environment, so that JupyterLab does not ask the person
to choose a kernel:

```json
{"cells": [], "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}}, "nbformat": 4, "nbformat_minor": 5}
```

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

## Start a Jupyter server

This task starts a new Jupyter server for the notebook. The Jupyter server
runs in the conda environment `{env}` and serves the folder `{folder}`. It
keeps running after your turn, until you stop it or the chat ends. A Jupyter
server cannot change its folder. To serve another folder, do "Stop the
Jupyter server", and then this task again.

This task gives two values that later tasks need: `{token}`, the password of
the Jupyter server, and `{port}`, the port where the Jupyter server answers.
If you do not have them later, "Find the Jupyter servers that run" gives
them again.

1. Make the token. Run "Make a token" from the reference file.
2. Prepare the environment. Jupylet needs a few Jupyter settings in the
   environment. Without them, a cell can stop responding after a few
   minutes, so the notebook cannot work with Claude. The installation
   makes these settings, but a later change to the environment can undo
   them. To put them back is a part of starting the Jupyter server, so tell
   the person, but do not ask. Run "Check the environment". If it prints
   `ready`, the settings are there. If it prints `needs: ...`, tell the
   person in one line, for example "Fixing a Jupyter setting first, so the
   notebook doesn't freeze.", and run "Prepare the environment". If it could
   not change the settings (`still on: ...`), tell the person, and do not
   start the Jupyter server.
3. Tell the person in one line, for example: "Starting Jupyter now, one
   second."
4. Start the Jupyter server. Run "Start JupyterLab" from the reference file.
   It gives `{port}` once the Jupyter server is ready. If it gives no port,
   see "The Jupyter server does not become ready" in
   `references/troubleshooting.md`.

## Open a web page

This task shows the notebook in the Claude web browser, next to the chat,
and signs the person in. The address of the notebook is
`http://localhost:{port}/doc/tree/{notebook}`. The `doc` address shows only
the notebook, which fits the narrow pane. The browser tools below are
`mcp__Claude_Browser__*`, by their short names.

1. Open the address. Call `tabs_context`.
   - **If a tab already shows this Jupyter server** (`localhost:{port}`),
     open the address in that tab with `navigate`, to keep one web page
     for each Jupyter server.
   - **If no tab does,** open it with `preview_start`.
2. Show the pane. A web page that you open is often in a hidden pane, and
   only the person can unhide it. Call `tabs_context` again: its last line
   says whether the pane is displayed or hidden.
   - **If it is displayed,** continue to step 3.
   - **If it is hidden,** ask the person to click the globe icon in the
     upper right corner of the Claude app. Then monitor the pane with the
     skill `running-background-monitors`, with "The pane brief" in
     `references/briefs.md`, and end your turn. While the pane is hidden,
     the monitor sends a note every minute. On each note, offer the person
     help. If the monitor reports that the pane is displayed, continue to
     step 3. If it reports that the pane is still hidden after 5 minutes,
     ask the person what is in the way.
3. Sign in. Run "Check the sign-in" from the reference file in the web
   page, with `javascript_tool`.
   - **If it prints `200`,** the web page is signed in. Continue to step 4.
   - **If it prints `403`,** the web page waits for the token. Give the
     person the token, for example: "This web page wants a special token
     just for this session. Please paste this into the box on the web
     page, and press Enter: `{token}`." Then monitor the sign-in with the
     skill `running-background-monitors`, with "The sign-in brief" in
     `references/briefs.md`, and end your turn. While the web page is not
     signed in, the monitor sends a note every minute. On each note, offer the
     person help. If the monitor reports that the web page is signed in,
     tell the person "You're signed in.", and continue to step 4. If it
     reports that the web page is still not signed in after 5 minutes,
     ask the person what is in the way.
   - **If it prints anything else,** the web page did not load. See "The
     Jupyter server stopped by itself" in `references/troubleshooting.md`,
     and stop here.
4. Collapse the JupyterLab side bar. The pane is narrow, and the side bar
   takes much of it. Run "Collapse the JupyterLab side bar" from the
   reference file in the web page, with `javascript_tool`, each time that
   you open a new web page.
5. Do "Hide the list of chats". The list of chats of the Claude app takes
   the most room.

## Hide the list of chats

This task makes room for the notebook in the browser pane: it hides the
list of chats of the Claude app, if the person wants it. To hide the list,
use the tool `mcp__ccd_window__set_sidebar_collapsed`, with `collapsed` set
to true. When the tool list shows this tool only by its name, load it with
`ToolSearch`. When this tool does not exist, tell the person how to hide
the list themselves, as in the example below.

Read the setting `hide-chat-list` with the skill `configuring-jupylet`, and
act on its value:

- **If it is `yes`,** hide the list of chats. Tell the person in one line,
  for example "Hiding the list of chats to make room for the notebook, as
  you asked last time."
- **If it is `no`,** do not hide it.
- **If it has no value,** ask, once in a conversation, for example:

  > The notebook is a bit cramped. Want me to tuck the list of chats on
  > the left out of the way, so there's more room for it?

  After a yes, hide it, and tell the person how to show it again, for
  example:

  > Done. You can bring it back any time by clicking the small panel icon
  > in the top-left corner of the Claude app, just right of the three
  > lines.

  After the answer, ask if you can remember it. After a yes, write boxless
  the global setting `hide-chat-list` (`yes` or `no`) with the skill
  `configuring-jupylet`.

Do not show the list of chats again yourself.

## Attach to the notebook

This task connects the Jupyter MCP tools to the notebook. It runs no cell. A
web page must show the notebook.

It can take up to a minute. Tell the person in one line first, for example:

> Almost there, just getting the notebook ready.

Run "Attach" from the reference file.

- **If the output contains `Successfully activate notebook`,** the task is
  complete.
- **If the output says that there is no kernel,** follow "The notebook has
  no kernel" in `references/troubleshooting.md`.
- **If it says anything else,** see `references/troubleshooting.md`.

## Stop the Jupyter server

Do this only when the person asks for it, or after a yes. Stop only a Jupyter
server that you started, or that the person chose. It stops each kernel of
that Jupyter server.

**If more than one Jupyter server runs, and the request does not name one,**
name the one that you want to stop, by its notebook, and wait for a clear
yes. For example: "Shall I close the Jupyter server with your spaceship
notebook?" A short word such as "shutdown" does not choose one.

1. Stop each monitor and each monitoring command that still runs, with the
   tool `TaskStop`.
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
