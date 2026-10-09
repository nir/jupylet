# Wait for the person

While the person works in the notebook, you must see what they do there, for
example which cell they ran and what it printed. At the same time, you must
stay ready to answer the next question or request from the person in the
chat.

You can answer the person only after your turn ends. A loop or a sleep that
monitors the notebook inside your turn keeps your turn open, and during that
time the person gets no answer. So monitor the notebook with a background
command, and end your turn:

1. Start a monitoring command in the background (`run_in_background`). It
   stops when the person does the action.
2. Tell the person what to do. Then end your turn. Do not write anything
   after the instruction, for example "I'll wait for you".
3. If the person writes, answer the person. The monitoring command
   continues. Do not start a second one.
4. When the monitoring command stops, you get a message about it. This
   message is not from the person, so it is never a yes to a question. Check
   what the person did. Then praise what worked, or show the one thing to
   fix.

A monitoring command also stops when its time ends. Then the person did not
do the action in that time. Start the command again with a longer time, and
ask the person in one line how it goes, for example: "How's it going? If
you're not sure what to do, just ask - no hurry." Use these times:

- Usually: 90 seconds, then 5 minutes, then 10 minutes.
- When the person writes code for the first time: 45 seconds, then 90
  seconds, then 5 minutes.
- When the person asks you not to interrupt: start with the longer times.

After the last time, do not start the command again. Wait for the person to
write. Each stop wakes you and uses the quota of the person, so do not use
shorter times.

Before you ask the person something different, or before you stop Jupyter,
stop a monitoring command that still runs (`TaskStop`). Also stop it before
you run code in the kernel yourself, because your own run also stops it.

## The monitoring commands

- **For the notebook to open after sign-in:**
  `"{python}" -m jupylet.claude wait-open {port} {token} {notebook} {seconds}`
  prints `open` or `timeout`. A command that starts too late can wait until
  its time ends. So right after you start it, check once if the notebook is
  already open. If it is, stop the command and continue.
- **For a run in the notebook:**
  `"{python}" -m jupylet.claude wait-change {port} {token} {notebook} {seconds} {since}`
  prints `ran {fingerprint}` when the kernel ran something and is idle again,
  or `timeout {fingerprint}`. Typing alone does not stop it. Give the
  fingerprint that it printed as `{since}` to the next `wait-change`, so that
  a run between the two is not missed. Leave out `{since}` the first time,
  and after you changed or ran something yourself.

For a short wait for something that you did yourself, you can also use a
background `sleep` of a few seconds. Never use a `sleep` in the foreground.

## Find which cell the person ran

Before the first `wait-change`, read the notebook with `read_notebook`, and
keep the execution count of each cell. After `ran`, read it again. The cell
whose count changed is the one that the person ran. Read it with its output
(`read_cell`). Do not use the highest count: cells keep the counts of earlier
runs.

A `ran` is not always a run: a Tab completion also counts. If no count
changed, start `wait-change` again with the fingerprint.

After `timeout`, read the notebook. If the person wrote code that they did
not run yet, and it will fail, show the problem in your question. Otherwise,
let the person run it first.
