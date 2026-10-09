# Connect to a running Jupyter

Read this when "Find the Jupyters that run" in `SKILL.md` finds one or more
Jupyters with Jupylet. The section names in quotes are in `SKILL.md`.

List the Jupyters with Jupylet to the person by notebook, folder and
environment name. Suggest the best guess, for example the one with a notebook
open in `{code}/examples`, and ask if that is the one, for example:

> I see Jupyter open with `11-spaceship.ipynb`, from this Jupylet folder.
> Is that the one you mean?

If the person wants something new instead, go to "Find the environment".

**The Jupyter serves another Jupylet folder:** tell the person, and offer to
work from there. After a yes, call `change_directory` with that folder, and
start again.

**The Jupyter serves this folder:** its environment (the last column) is
`{env}`, and its port and token are `{port}` and `{token}`. Do these checks
on `{env}`:

1. The checks from "Find the environment" (JupyterLab and the `[claude]`
   extra).
2. `"{python}" -m jupylet.claude prepare`. Expected: `off`.
3. `"{python}" -m jupylet.claude tools {port} {token}`. Expected: the list
   contains `notebook_run-all-cells`.

If all three pass, attach to the notebook (step 8 in
`start-in-app.md`), and help with what the person asked. The notebook already runs, so do
not run all cells again.

If check 2 printed `turned off`, or check 3 failed, this Jupyter must start
again. Explain why, and ask, for example:

> To connect to your notebook, Jupyter has to start again, because a setting
> changes only when it starts. Please save your notebook, and tell me when I
> can continue.

After the person says to continue, stop that Jupyter as in "Stop Jupyter",
with its port and token. Then tell the person only that you closed Jupyter.
You cannot know if the person saved the notebook. Then ask, for example:

> I can start Jupyter here in the app, next to our chat, so we can work in
> the notebook together. Or you can start it yourself, and I'll connect to
> it. Which do you prefer?

- **In the app:** do the steps in `start-in-app.md`, from step 1.
- **The person starts it:** wait until it runs, then run "Find the Jupyters
  that run" again, and connect. Skip steps 5 and 6 in
  `start-in-app.md`: the browser of the person shows the notebook.
