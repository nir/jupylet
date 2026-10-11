---
name: running-jupylet-notebooks
description: Opens, runs and closes Jupyter notebooks of a Jupylet folder, together with a person. Starts a Jupyter server, and opens the notebook in the browser pane of the Claude app or in the default browser. Connects to the notebook with the Jupyter MCP tools, to read, change and run cells with the person. Also stops the Jupyter server, and solves known problems. Use this skill when the person asks to open, run or try a notebook or an example. Also use it when the person asks for help with an open notebook or cannot find a notebook page that they closed. Also use it when the person asks to stop Jupyter. Use it together with guiding-jupylet-users.
---

# Running Jupylet notebooks

With this skill you work together with the person in a Jupyter notebook. You
open the notebook in a page that the person sees, attach to the notebook, and
then read, change and run its cells with the person. This file has the tasks
for that. "Understand the request" says which tasks to do, and in which order.
Use the skill `guiding-jupylet-users` for how to talk with the person and when
to ask.

## Values

Values in `{braces}` are blanks that you fill in.

- `{code}`: the Jupylet folder. It is the folder that contains
  `.claude/skills/running-jupylet-notebooks`. Use its full path.
- `{version}`: the line `VERSION = '...'` near the top of
  `{code}/jupylet/__init__.py`.
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

## Tasks and commands

This file uses two kinds of names in quotes:

- **Do "Name":** a task of this file, for example: do "Attach to the notebook".
- **Run "Name" from the reference file:** a command in `references/macos.md`
  or `references/windows.md`, for example: run "Make a token" from the
  reference file.

Read the reference file for this computer, `references/macos.md` or
`references/windows.md`, before the first command.

When a command does not give the expected result, or something behaves
strangely, read `references/troubleshooting.md`.

## Terms

- **Claude app:** the Claude desktop app, where the person chats with you.
- **Default browser:** the default web browser of the computer.
- **Jupyter server:** one JupyterLab program that runs. It serves a folder,
  and it has a port and a token.
- **Page:** a JupyterLab page in the browser pane of the Claude app, or in the
  default browser.
- **The `[claude]` extra:** an optional part of the Jupylet installation, in
  `setup.py`. It installs the Jupyter MCP server for JupyterLab.
- **Monitoring command:** a command that runs in the background and stops
  when something happens in the notebook, for example when a run ends.

## Rules for this skill

- **Run cells only when the person asks for it, or when another skill tells
  you to.** Opening a page or attaching runs nothing.
- **Add, change, move or delete cells only when the person asks for it, or
  after a yes.** Do this only with the Jupyter MCP tools, never in the `.ipynb`
  file on the disk. You can read the saved file.
- **Start, stop or restart a Jupyter server only when the person asks for it,
  or after a yes.** Stop only a Jupyter server that you started, or that the
  person chose.
- **Keep one page open on a Jupyter server.** Some Jupyter MCP tools act
  through a page. With two pages, the Jupyter server chooses one of them, and
  you cannot know which one ("Page commands" in
  `references/jupyter-mcp-tools.md`).
- **The person signs in, not you.** Give the token to the person. Never type
  it in a page, and never put it in an address.

## Understand the request

The person wants to open, run or continue a notebook. First do "Find a Python"
and "Find the Jupyter servers that run". Then use the person's request
and what you found to work out what the person most likely wants. The words of
the person alone do not always make it clear. Ask a short question to make
sure, for example:

> I see your spaceship notebook is still running, but no page shows it. Do
> you want me to open it again, here next to our chat or in your browser?

If you cannot work it out, ask. When there are a few possible meanings, name
them, for example:

> Do you mean the notebook that's open in your browser, or should I open a new
> one here?

When the person asks to open a notebook, to run it, or to work on it
together, the notebook can be in one of these states:

- **It is open in a Jupyter server that is ready for you**
  (`jupylet_version` is `{version}`, `mcp_tools` is `true` and `nbmodel_off`
  is `true`): do "Use a running Jupyter server". The person keeps their page
  and their variables.
- **It is open in another Jupyter server:** do "Move the notebook to a new
  Jupyter server".
- **No Jupyter server has it open:** do "Find the environment", "Choose a
  notebook", "Start a Jupyter server", "Open a page" and "Attach to the
  notebook".

When the notebook is open in a page, and you are attached to it, do what the
person asked:

- **To run the notebook:** do "Run the notebook".
- **To work on the code together:** do "Work in the notebook".

For example, the person says "run the spaceship" and no Jupyter server runs.
Find the environment, choose `11-spaceship.ipynb`, start a Jupyter server,
open a page, attach, and then run the notebook.

**The person wants to stop Jupyter:** do "Stop the Jupyter server". You do
not need a page for this.

## Find a Python

The first commands need only Python 3, so the Python of each conda installation
can run them. Do not use `python3` or `python` of the system: on a computer
without Python, they can be stubs that only start an installation (the
Microsoft Store on Windows, or the Command Line Tools on macOS).

**`{miniforge}` has a Python** (`bin/python` on macOS, `python.exe` on
Windows): that Python is `{conda_python}`.

**`{miniforge}` does not exist, or has no Python:** Miniforge is not installed
there. The person can have another conda installation, for example Anaconda.
Run "Find conda installations" from the reference file. `{conda_python}` is
the Python of a folder that it prints.

**The computer has no conda installation:** Jupylet is not installed in a
way that this skill can use. Do "Install Jupylet first", below.

### Install Jupylet first

Tell the person that Jupylet must be installed first, and that after that you
will continue with what they asked. Ask for a yes, for example:

> To open the music notebook, Jupylet needs to be installed first. Shall I
> install it? It takes a few minutes, and then we'll open the notebook.

- **Yes:** use the skill `setting-up-jupylet`. It ends with the spaceship
  example in a page in the Claude app, or with no Jupyter server if the
  person said no to that example. Then do "Understand the request" again,
  for what the person asked.
- **No:** tell the person that the notebooks need Jupylet installed, and that
  they can ask at any time.

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
  of pages and other clients that connect to the notebook. `0` means that no
  page shows the notebook now, for example because the person closed it.

Run it again each time that you need its result: it can change at any time.

## Use a running Jupyter server

Use this task for a notebook that is open in a Jupyter server whose
`jupylet_version` is `{version}`, the Jupylet version of this folder, and
whose `mcp_tools` and `nbmodel_off` are `true`. (A notebook in another Jupyter
server has its own task: "Move the notebook to a new Jupyter server".)

The person does not know about ports or Jupyter servers, so talk about the
notebook, and ask a short question to make sure. For example, the person says
"let's continue with the spaceship", and a Jupyter server has
`11-spaceship.ipynb` open:

> Your spaceship notebook is still open. Shall I connect to it, so we can
> work on it together?

**Several Jupyter servers fit:** name each one by its notebook, and ask which
one, for example:

> I see two notebooks open, the spaceship and Hello Jupylet. Which one do you
> mean?

Take `{port}`, `{token}`, `{env}` and `{folder}` from "Find the Jupyter servers
that run". Then:

1. Make sure that a page shows the notebook. **No page shows it**
   (`connections` is `0`): offer to open it again. After a yes, do "Open a
   page".
2. Do "Attach to the notebook".

## Move the notebook to a new Jupyter server

You cannot work with the Jupyter server that has the notebook open:

- **`jupylet_version` is not `{version}`:** it runs another version of
  Jupylet, or no Jupylet, or its environment is unknown.
- **`mcp_tools` is `false`:** the `[claude]` extra is missing, or the Jupyter
  server started before its installation.
- **`nbmodel_off` is `false`:** a cell can stop responding after a few
  minutes.

Offer to open the notebook again in a new Jupyter server, for example:

> Your spaceship notebook is open in a Jupyter server that I can't connect to
> (a Jupyter server is the program behind the notebook page). I can close it
> and open the notebook again in a new one. Please save the notebook first.
> Shall I?

After a yes, take `{port}`, `{token}` and `{folder}` from "Find the Jupyter
servers that run". Do "Find the environment" first: it checks the extra, and
installs it if it is missing and the person agrees. Then do "Stop the Jupyter
server", "Start a Jupyter server", "Open a page" and "Attach to the notebook".

## Find the environment

A new Jupyter server runs in a conda environment. That environment must have
Jupylet `{version}`, the version of the Jupylet code in `{code}`. It must also
have JupyterLab and the `[claude]` extra. "Start a Jupyter server" changes the
Jupyter settings to what Jupylet needs, so this task does not check them.

Run "List the environments" from the reference file. It prints one line for
each environment with the Jupylet version of this folder, best first, with
three columns: the path, the kind (`conda` or `venv`), and where its Jupylet
comes from. Use only `conda` environments: the skill does not support a venv.
The path is `{env}`. Call an environment by its name. Do not show paths or
columns to the person.

**One conda environment:** use it.

**Several conda environments:** use the first case that applies.

1. **The setting `environment` names one of them:** read the setting with the
   skill `configuring-jupylet`. It names the environment of this folder:
   `setting-up-jupylet` writes it after an installation, and you write it when
   the person chooses. If that environment is in the list, use it.
2. **Other cases:** ask the person which one to use, and suggest the first one
   in the list, which is the best one. For example: "You have two Jupylet
   environments, `jupylet` and `jp146`. Shall I use `jupylet`?" After the
   person chooses, ask if you can remember the choice. After a yes, write the
   local setting `environment` with the skill `configuring-jupylet`.

**No conda environment:** Jupylet is not installed for this folder yet. Do
"Install Jupylet first", in "Find a Python".

Then run "Check JupyterLab and the extra" from the reference file. **It
fails:** see "Jupylet, JupyterLab or the extra is missing" in
`references/troubleshooting.md`.

## Choose a notebook

**The person or another skill names a notebook:** use it. `{folder}` is the
folder of the notebook, or `{code}/examples` for an example notebook.

**Nobody names a notebook:** list the example notebooks in `{code}/examples`,
or ask what the person likes, for example music, 2D graphics or 3D games.
Then suggest a notebook. For a first try, the spaceship (`11-spaceship.ipynb`)
is a good choice. `{folder}` is `{code}/examples`.

**Jupyter and notebooks are new in this conversation:** explain them in one
or two sentences when you ask, for example:

> May I open a Jupyter notebook for us? Jupyter is the program where you write
> and run your code, and a notebook is a page in it where you type code in
> small boxes, called cells, and run each one to see what it does.

## Start a Jupyter server

This task starts a new Jupyter server in the conda environment `{env}`. The
Jupyter server serves the folder `{folder}`. The task makes a token, finds a
free port, prepares the environment, and waits until the Jupyter server is
ready. The Jupyter server continues to run after your turn, until you stop it.

This task opens no page and runs no cell: "Open a page" and "Attach to the
notebook" do that. To change the folder of a Jupyter server, do "Stop the
Jupyter server", and then this task with the new `{folder}`.

1. Run "Make a token" from the reference file. The result is `{token}`. **You
   start a Jupyter server again for a page that signed in before:** use the
   token of the old Jupyter server instead, so that the page stays signed in.
   The token is the password of the Jupyter server. Later tasks need it, and
   the person pastes it into a new page to sign in. If you do not have it
   later, take it from "Find the Jupyter servers that run".
2. Run "Check a port" for port 8888.
   - **The port is free:** `{port}` is 8888.
   - **The port is in use:** check 8889, 8890, and so on. `{port}` is the
     first free one.

   Later tasks need `{port}` too. If you do not have it later, take it from
   "Find the Jupyter servers that run".
3. Ask for a yes to change the Jupyter settings of the environment, for
   example: "May I change a Jupyter setting in your environment? It stops
   notebooks from freezing after a few minutes." After a yes, run "Prepare the
   environment". It turns off `jupyter_server_nbmodel`, a Jupyter extension
   that can make a cell stop responding, and prints what it found:
   - **`off`, `not installed` or `turned off`:** the extension is off.
     Continue.
   - **`still on: ...`, or the person said no:** tell the person that a cell
     can stop responding after a few minutes, and continue.
4. Tell the person in one line, for example: "Starting Jupyter now, one
   second."
5. Run "Start JupyterLab". Do not start it as a background task
   (`run_in_background`): the Jupyter server stops when the task stops. Do
   not use the Terminal panel of the Claude app.
6. Run "Wait until it is ready".
   - **`ready`:** the task is complete.
   - **Other output:** see "The Jupyter server does not become ready" in
     `references/troubleshooting.md`.

## Open a page

This task opens the notebook in a browser page, either in the browser pane of
the Claude app, next to the chat, or in the default browser. If a page with
the notebook is already open, the task uses that page. If the notebook has no
kernel yet, the page starts it. "Attach to the notebook" needs that kernel.

1. **A page with the notebook is already open** (`connections` is more than
   `0`): call the tool `mcp__Claude_Browser__tabs_context`. It lists the tabs
   of the browser pane, with their addresses, and says whether the pane is
   visible.
   - **A tab has the address `localhost:{port}`:** use that page. If the pane
     is hidden, ask the person to click the globe icon in the upper right
     corner of the Claude app. End the task.
   - **Other cases:** the page is probably in the default browser. Ask the
     person to look for it, for example: "Your notebook is already open in
     your web browser. Can you find a tab called 11-spaceship?" **Yes:** use
     that page, and end the task. **No:** tell the person that two open pages
     can cause trouble, and that it is better to close the old one if they
     find it later. Then continue with step 2.
2. Read the setting `open-notebooks-in` (`claude-app` or `default-browser`)
   with the skill `configuring-jupylet`. **It has a value:** suggest that
   place. **No value:** ask which place the person prefers. After the person
   chooses, ask if you can remember the choice. After a yes, write the global
   setting `open-notebooks-in` with the skill `configuring-jupylet`.
3. Do "Open a page in the Claude app" or "Open a page in the default browser".

## Open a page in the Claude app

This task shows the notebook in the browser pane of the Claude app, next to
the chat. The browser tools are `mcp__Claude_Browser__preview_start`,
`mcp__Claude_Browser__tabs_context`, `mcp__Claude_Browser__navigate` and
`mcp__Claude_Browser__javascript_tool`. Below, they have their short names.

1. Call `tabs_context`, and open the notebook at the address
   `http://localhost:{port}/doc/tree/{notebook}`. The `doc` address shows only
   the notebook, which fits the narrow pane.
   - **A tab has the address `localhost:{port}`,** for example with another
     notebook: open the address in that tab, with `navigate`. Do not open a
     second tab: keep one page on a Jupyter server.
   - **No such tab:** open the address with `preview_start`.
2. Run "Check the sign-in" from the reference file in the page, with
   `javascript_tool`. Call `tabs_context` to see if the pane is visible.
   - **`200`, and the pane is visible:** continue with step 3.
   - **`200`, and the pane is hidden:** tell the person, for example:
     "Please click the globe icon in the upper right corner of the Claude
     app, so you can see the notebook." Continue with step 3.
   - **`403`:** start "Wait for the page to open" from the reference file in
     the background, as `references/waiting.md` says. Then give the person the
     token, for example: "This page wants a special token just for this
     session. Please paste this into the box on the page, and press Enter:
     `{token}`. The token shows Jupyter that it's really you." If the pane is
     hidden, ask the person first to click the globe icon.
     - **"Wait for the page to open" prints `open`:** tell the person in one
       line, for example "You're signed in." Continue with step 3.
     - **It prints `timeout`:** start it again with a longer time, and ask
       how it goes.
   - **Other results:** the page did not load. Tell the person, and see "The
     Jupyter server stopped by itself" in `references/troubleshooting.md`.
3. Offer more room. The pane is narrow, and it can cut off the canvas. This
   needs the tool `mcp__ccd_window__set_sidebar_collapsed`. If the tool list
   shows it only by its name, load it with `ToolSearch`. Read the setting
   `hide-chat-list` with the skill `configuring-jupylet`:
   - **`yes`:** hide the list of chats (`collapsed` set to true). Tell the
     person in one line, for example "Hiding the list of chats to make room
     for the notebook, as you asked last time."
   - **`no`:** the task is complete.
   - **No value:** ask, for example:

     > The notebook is a bit cramped. Want me to tuck the list of chats on
     > the left out of the way, so there's more room for it?

     After a yes, hide it, and tell the person how to show it again, for
     example:

     > Done. You can bring it back any time by clicking the small panel icon
     > in the top-left corner of the Claude app, just right of the three
     > lines.

     After the answer, ask if you can remember it. After a yes, write the
     global setting `hide-chat-list` (`yes` or `no`) with the skill
     `configuring-jupylet`.
   - **The tool does not exist:** tell the person how to hide the list
     themselves, as in the example above.

   Ask about the list of chats only once in a conversation. Do not show the
   list again yourself.

## Open a page in the default browser

This task opens the notebook in the default browser. Run "Open the default
browser" from the reference file. It opens the full JupyterLab page, with the
list of files.

- **The page shows the notebook:** the task is complete.
- **The page asks for the token:** give the token to the person, as in "Open
  a page in the Claude app". A browser that signed in to this Jupyter
  server before does not ask.

## Attach to the notebook

This task connects the Jupyter MCP tools to the notebook. It runs no cell. A
page must show the notebook.

It can take up to a minute. Tell the person in one line first, for example:

> Almost there, just getting the notebook ready.

Run "Attach" from the reference file.

- **The output contains `Successfully activate notebook`:** the task is
  complete.
- **The output says that there is no kernel:** see "The notebook has no
  kernel" in `references/troubleshooting.md`.
- **Other output:** see `references/troubleshooting.md`.

## Work in the notebook

Read `references/jupyter-mcp-tools.md` before the first tool call. Call each
tool with "Call a tool" from the reference file.

- **Check before you speak.** Before you tell the person what they see, check
  it. When you continue after the person was away, check with "Find the Jupyter
  servers that run" that the notebook is still open, and that a page shows it.
- **Look at the canvas only when the output is graphical.** After a run that
  changes the canvas or an image, take a screenshot of the page in the browser
  pane before you say what changed. For text output, read the output with
  `read_cell`. You cannot see a page in the default browser: ask the person
  what they see.
- **Find the real error.** When a cell fails, read its output.
- **Delete your own test cells.** This needs no yes, as an exception to the
  rule "Add, change, move or delete cells" in "Rules for this skill". A cell
  that you add stays in the notebook, and the Jupyter server saves it in a few
  seconds. If you keep one, tell the person. Code that you run in the kernel
  with `execute_code` can change settings of the person, for example the
  logging level. Undo such changes, and tell the person.
- **Follow the person when you ask them to try something:** do "Wait for the
  person".

## Run the notebook

This task runs all cells of the notebook. Do it only when the person asks for
it, or when a path tells you to.

1. Tell the person in one line, for example: "Running the notebook now."
2. Read the notebook with `read_notebook`, and keep the execution count of
   each cell.
3. Start "Wait for a run" from the reference file in the background, with 90
   seconds, as `references/waiting.md` says.
4. Run "Run all cells" from the reference file.
   - **`True`, after a second or two:** the cells can still run. End your
     turn. "Wait for a run" sends you a message when the run ends.
   - **"Timeout waiting for result" or "Not Found":** see
     `references/troubleshooting.md`.
5. When "Wait for a run" stops, read the notebook again.
   - **Each code cell has a new execution count, and `read_cell` of the last
     code cell shows no error:** the run worked.
   - **A cell shows an error:** read its output, to find the real error.
   - **`timeout`, and cells still show `[*]`:** a cell can run for a long
     time. Start "Wait for a run" again, with the times in
     `references/waiting.md`. After the last time, tell the person that a
     cell still runs, and stop waiting.
6. Tell the person the result.

## Wait for the person

This task lets you follow the person in the notebook, and stay free to
answer. Before you ask the person to do something in the notebook, and before
you start a monitoring command, read `references/waiting.md`.

## Stop the Jupyter server

Do this only when the person asks for it, or after a yes. Stop only a Jupyter
server that you started, or that the person chose. It stops each kernel of
that Jupyter server.

1. Stop each monitoring command that still runs, with the tool `TaskStop`.
2. Tell the person in one line, for example "Closing Jupyter now."
3. Close the page in the browser pane with `mcp__Claude_Browser__tabs_close`
   (`tabs_context` gives its id): an open page shows an error when the Jupyter
   server stops. **A page in the default browser:** ask the person to close it.
4. Run "Stop JupyterLab" from the reference file. It takes a few seconds.
   - **`stopped`, `not running` or `stopped after ending its process`:**
     continue.
   - **Other output:** see "The Jupyter server does not stop correctly" in
     `references/troubleshooting.md`.
5. Tell the person in one line, for example "All closed. Just ask when you
   want to open it again."

## Start Jupylet without Claude

This task shows the person how to start Jupylet alone, in Terminal or in the
Miniforge Prompt. When the person asks for it, read
`references/start-without-claude.md`. It has the whole task.
