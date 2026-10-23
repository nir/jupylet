# Known problems

What to do when a command does not give the expected result. Each problem
has the same form: what you see, why, and what to do. Task names in quotes
are tasks in `SKILL.md`. Command names in quotes are commands in the
reference file for this computer. For problems with cells and runs in the
notebook, see `references/troubleshooting.md` of the skill
`working-in-notebooks`.

When a solution fails, follow rule 3 of `guiding-jupylet-users`: tell the
person, and ask before the next attempt.

## Contents

- Jupylet, JupyterLab or the extra is missing
- The Jupyter server does not become ready
- The notebook has no kernel
- The web page shows the login page
- The Jupyter server stopped by itself
- The Jupyter server does not stop correctly
- `!pip` or `!python` in a cell uses the wrong Python
- Something behaves strangely: reset the Jupyter server

## Jupylet, JupyterLab or the extra is missing

**What you see:** "Check JupyterLab and the extra" prints an error, or the
log of the Jupyter server says `No module named`.

**Why:** the environment does not have JupyterLab, or the `[claude]` extra.

**What to do:** tell the person in plain words what is missing, for example:

> Jupylet's tools for working in the notebook with you aren't installed in
> this environment.

Offer to install it.

- **Yes:** use the skill `setting-up-jupylet`. When the installation is
  complete, do "Understand the request" in `SKILL.md` again, for what the
  person asked.
- **No:** tell the person what that means, for example:

  > No problem. Without them, I can't see or change your notebook while it's
  > open, or run it for you. I can still explain the code, write code for you
  > to paste into the notebook, and help with an error if you paste it here.
  > If you change your mind, just ask.

  **Only the extra is missing:** you can still start a Jupyter server and open
  a web page, but do not attach. Tell the person how to run the cells: "Click into
  the first box of code and press Shift+Enter to run it. Each press runs one
  box and moves to the next."

## The Jupyter server does not become ready

**What you see:** "Start JupyterLab" gives no port after 60 seconds.

**Why:** the output of JupyterLab tells. Read the `Jupyter` tab with
`read_terminal`, or `{scratchpad}/jupyter.log` if you started it with
`detach`.

**What to do:**

- **`No module named`:** follow "Jupylet, JupyterLab or the extra is
  missing".
- **Other errors:** tell the person, and stop.

## The notebook has no kernel

**What you see:** "Attach" says that there is no kernel.

**Why:** a notebook gets a kernel only when a web page opens it.

**What to do:** check that a web page shows the notebook, that the pane is
visible, and that the person signed in ("Check the sign-in" says `200`). Then
do "Attach to the notebook" again.

## The web page shows the login page

**What you see:** the web page asks for a token.

**Why:** the web page is not signed in. This happens after a reset of the Jupyter
server, and with a browser that did not sign in to this Jupyter server
before. After a restart of a Jupyter server, a web page usually stays signed in.

**What to do:** give the token to the person, as in "Open a web page".

## The Jupyter server stopped by itself

**What you see:** "List the Jupyter servers" does not list it any
more.

**Why:** for example, the computer restarted, or a program stopped it. "Start
JupyterLab" starts it apart from your task, so the end of your task does not
stop it.

**What to do:** tell the person, and ask for a yes to start it again. Do
"Start a Jupyter server" with the same `{folder}` and token, so that the web page
stays signed in. Then do "Open a web page" (the old web page shows an error) and
"Attach to the notebook".

## The Jupyter server does not stop correctly

**What you see:** "Stop JupyterLab" prints `a kernel is still running:
not shutting the server down`, or `still running`.

**What to do:**

- **macOS:** tell the person, and stop. Do not stop processes yourself.
- **Windows:** follow "Stop JupyterLab" in the reference file.

## `!pip` or `!python` in a cell uses the wrong Python

**What you see:** a `!pip` or `!python` line in a cell uses another
environment.

**Why:** the Jupyter server started without the environment activated.

**What to do:** stop it after a yes, and do "Start a Jupyter server", which
activates the environment.

## Something behaves strangely: reset the Jupyter server

**What you see:** something behaves strangely, and no other entry in this file
helps.

**What to do:** reset the Jupyter server. A reset deletes the state files
of Jupyter, and some of them belong to all Jupyter servers of the person. It
also deletes the old log of changes. **A notebook lost cells:** do "Cells that
you added do not show in the web page" in the troubleshooting file of
`working-in-notebooks` first, because it keeps that log.

First do "Find the Jupyter servers that run" in `SKILL.md`. **Another
Jupyter server runs, besides the one that you work with:** a reset would sign
out its web pages. Ask the person if you can stop it first. **No:** do not reset,
and tell the person what is wrong in plain words.

Ask, for example:

> Something isn't working right. May I reset the notebook setup? It closes
> Jupyter and clears its saved state. Your notebooks and code are not
> touched.

After a yes:

1. Do "Stop the Jupyter server".
2. Run "List the state files" from the reference file. The list has only state
   files of Jupyter, never a notebook. Check the list. Then run "Delete the
   state files". This also signs the web pages out, so the person must paste the
   token again.
3. Do "Start a Jupyter server", "Open a web page" and "Attach to the notebook".
