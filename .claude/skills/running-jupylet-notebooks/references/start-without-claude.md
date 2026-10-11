# Start Jupylet without Claude

This file explains how the person starts Jupylet alone, in Terminal on macOS
or in the Miniforge Prompt on Windows. Command names in quotes are commands
in the reference file for this computer. You need `{env name}` and `{code}`.
**You do not know the environment yet:** do "Find the environment" in
`SKILL.md` first. These instructions work for an environment in
`{miniforge}`. **The environment is in another conda installation:** tell the
person that its own Prompt or Terminal setup is necessary, and that you can
help with it.

**macOS:** run "Check the Terminal" from the reference file.

- **The last line is the full path of `{miniforge}`:** tell the person how to
  start Jupylet, for example:

  > Open Terminal: press Cmd+Space, type *Terminal* and press Enter. It's a
  > window where you type commands. Type `conda activate {env name}`, and the
  > start of the line changes to `({env name})`: now you are in Jupylet's
  > environment. Then type `cd "{code}/examples"` to go to the example
  > notebooks, and `jupyter lab`. Jupyter opens in your web browser.

- **Other output:** tell the person that Terminal cannot find Miniforge yet,
  and offer to prepare it, with step 4 of the skill `setting-up-jupylet`
  ("Prepare Terminal").

**Windows:** tell the person how to start Jupylet, for example:

> Open the Start menu, type *Miniforge* and open **Miniforge Prompt**. It's a
> small window where you type commands. Type `conda activate {env name}`, which
> switches it to Jupylet's environment. Then type `cd /d "{code}\examples"`
> to go to the example notebooks, and `jupyter lab`. Jupyter opens in your
> web browser.
