# Monitor the notebook

Use this procedure when the person is about to work in the notebook, either
because you asked them to do something there or because they said they
would. You start a background agent, the `notebook-monitor`, which reads the
notebook while the person works and tells you when something calls for your
attention. In the meantime, you remain free to talk with the person.

## Why a background agent

You can answer the person only once your turn ends. Any wait inside the turn
holds it open, and leaves the person without a reply. The monitor runs
outside your turn: it reads the notebook every few seconds, judges how the
work is progressing, and messages you only when you need to act.

Its agent file, `.claude/agents/notebook-monitor.md`, defines the commands
the monitor runs and how it judges the work. From you, it needs only a
brief: the goal, and the values that its commands take.

## Start a monitor

Do these steps in one turn: the turn in which you give the task to the
person.

1. Give the task to the person.
2. Write the brief. "Write the brief" below tells how.
3. Start the monitor with the tool `Agent`:
   - `name`: `notebook-monitor`
   - `subagent_type`: `notebook-monitor`
   - `run_in_background`: `true`
   - `description`: the goal, in a few words
   - `prompt`: the brief
4. End your turn. Make the task the last text of your turn. Do not add a
   promise to wait, such as "I'll wait for you". The monitor waits, not you.

## Write the brief

The monitor judges the work against the brief alone, so the notes are only
as precise as the brief.

```
The person works in the notebook {notebook}. I asked them to: {task}.
Done when: {what the notebook shows when the task is done}.
Also report: {other events worth a note, or "nothing else"}.
Time limit: {minutes} minutes.

Values: code "{code}", python "{python}", port {port}, token {token},
notebook {notebook}.
```

- **The task:** the words that you used with the person, so that your
  instructions and the notes of the monitor refer to the same thing.
- **Done when:** a state of the notebook that the monitor can check, such
  as the source of a cell, an output, or a run with no error. Not an
  intention, such as "understands how labels work".
- **Time limit:** about twice the time that you expect the task to take.

For example:

```
The person works in the notebook 11-spaceship.ipynb. I asked them to: change
the text of the label in cell 18 to their name, and run the cell.
Done when: the Label(...) in cell 18 has a new text instead of 'hello, world',
and the cell ran with no error.
Also report: nothing else.
Time limit: 10 minutes.

Values: code "/Users/sam/jupylet", python "/Users/sam/miniforge3/envs/jupylet/bin/python",
port 8888, token 3f9a1c07, notebook 11-spaceship.ipynb.
```

## Act on notes and reports

Each message from the monitor arrives as a new turn. It comes from the
monitor, not from the person, so it never answers a question that you asked
the person.

- **A note** reports one event: an error, a stall, a pattern of
  uncertainty, a possible bug, or an event that the brief asks for. Decide
  whether the person needs to hear it now. Usually they do: tell them in
  one or two plain lines, and help with the one thing to fix. The monitor
  continues.
- **A final report that the goal is done:** tell the person what worked,
  and continue with the next step.
- **A final report that the time is over:** ask the person in one line how
  the work goes, with no pressure. For example:

  > How's it going? If you're not sure what to do next, just ask - no
  > hurry.

## Change or stop the monitor

Run only one monitor at a time.

- **For a new goal:** send the new goal to the monitor with the tool
  `SendMessage`, to `notebook-monitor`, in the form of the first four lines
  of the brief. The new goal replaces the old one. The monitor keeps its
  clock, and reports only what changed since its last read.
- **To stop the monitor:** call the tool `TaskStop` with the `task_id`
  `notebook-monitor`.

Stop the monitor:

- **Before you change or run cells.** Otherwise it reports your changes as
  the work of the person. When the person starts to work again, start a new
  monitor. A new monitor starts from the notebook as it is then, with its
  clock at zero.
- **When the work in the notebook ends,** or when it changes to a task that
  needs no monitor.
