# Monitor the notebook

This file is the whole task "Monitor the notebook" of `SKILL.md`. It
explains how you see what the person does in the notebook, and stay free
to answer the chat at the same time. Command names in quotes are commands
in the reference file for this computer.

While the person works in the notebook, you must see what they do there, for
example which cell they ran and what it printed. At the same time, you must
stay ready to answer the next question or request from the person in the chat.

You can answer the person only after your turn ends. A loop or a sleep that
monitors the notebook inside your turn keeps your turn open, and during that
time the person gets no answer. So monitor the notebook with a background
command, and end your turn:

1. Start a monitoring command in the background (`run_in_background`). It
   stops when the person does the action.
2. Tell the person what to do. Then end your turn. Do not write anything
   after the instruction, for example "I'll wait for you".
3. While the command runs, answer each message of the person. The
   monitoring command continues. Do not start a second one.
4. When the monitoring command stops, you get a message about it. This
   message is not from the person, so it is never a yes to a question. Check
   what the person did. Then praise what worked, or show the one thing to
   fix.

A monitoring command also stops when its time ends. Then the person did not
do the action in that time. Start the command again with a longer time, and
ask the person in one line how it goes, for example:

> How's it going? If you're not sure what to do, just ask - no hurry.

Use these times:

- **Usually:** 90 seconds, then 5 minutes, then 10 minutes.
- **The person writes code for the first time:** 45 seconds, then 90 seconds,
  then 5 minutes.
- **The person asks you not to interrupt:** start with the longer times.

Read the setting `wait-times` with the skill `configuring-jupylet` before the
first wait. **It has a value,** for example `3 minutes, 10 minutes, 20
minutes`: use these times instead of the usual times. When the person asks
for longer or shorter times, ask if you can remember them. After a yes, write
the global setting `wait-times` in this form with the skill
`configuring-jupylet`.

"Wait for the web page to open" uses 90 seconds, then 5 minutes.

After the last time, do not start the command again. Wait for the person to
write. Each stop wakes you and uses the quota of the person, so do not use
shorter times.

Before you ask the person something different, before you stop the Jupyter
server, and before you run code in the kernel yourself, stop a monitoring
command that still runs (`TaskStop`). Your own run also stops it.

The monitoring commands, in the reference file:

- **"Wait for the web page to open":** for the notebook to open after the person
  signs in. It prints `open` or `timeout`. If you did not run "Check the
  sign-in" just before, run it once right after you start the command. **It
  prints `200`:** the person signed in already. Stop the command, and
  continue.
- **"Wait for a run":** for a run in the notebook. It prints
  `ran {fingerprint}` when the kernel ran something and is idle again, or
  `timeout {fingerprint}`. Typing alone does not stop it. Give the fingerprint
  that it printed as `{since}` to the next "Wait for a run", so that a run
  between the two is not missed. Leave out `{since}` the first time, and after
  you changed or ran something yourself.

For a short wait for something that you did yourself, you can also use a
background `sleep` of a few seconds. Never use a `sleep` in the foreground.

To find which cell the person ran: before the first "Wait for a run", read the
notebook with `read_notebook`, and keep the execution count of each cell.
After `ran`, read it again. The cell whose count changed is the one that the
person ran. Read it with its output (`read_cell`). Do not use the highest
count: cells keep the counts of earlier runs.

- **No count changed:** a Tab completion also counts as `ran`. Start "Wait
  for a run" again with the fingerprint.
- **`timeout`, and the person wrote code that they did not run yet, and it
  will fail:** show the problem in your question.
- **`timeout`, other cases:** let the person run it first.
