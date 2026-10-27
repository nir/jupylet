# Monitor the notebook

Use this procedure in two cases:

- **Follow along:** the person agreed that a helper follows their work in
  the notebook. "Offer to follow along" below says when to ask.
- **A task:** you asked the person to do something in the notebook, or they
  said they would.

In both cases you start a background agent, the `notebook-monitor`, which
reads the notebook while the person works and tells you when something
calls for your attention. In the meantime, you remain free to talk with the
person.

## Offer to follow along

Once in a conversation, when a notebook is open and you are attached to it:

1. Offer the helper. Ask the person:

   > If you'd like, I can start a small helper agent that follows along in
   > the notebook as you work. It would read the changes you make and the
   > cells you run, and tell me when you might need help, so I can help or
   > give advice right away. It is most useful for beginners. Shall I?

   - **After a no,** do not ask again in this conversation. The person can
     still ask for the helper later.
   - **After a yes,** continue to step 2.
2. Check the length of this chat. Each note of the helper makes you read
   the whole chat again, so a long chat uses up the usage limit of the
   person faster. Run "Measure the context" from the reference file.
   - **Up to 100k tokens,** continue to step 3.
   - **Above 100k tokens,** run "Wait for a compaction" from the reference
     file, in the background. Then ask the person, and end your turn:

     > This chat is already long, so each time the helper checks in, it
     > uses up more of your usage limit. If you type /compact, the chat
     > gets shorter, and I'll start the helper as soon as it's done. Or
     > just say "start it", and I'll start it right away.

     When "Wait for a compaction" prints `compacted`, continue to step 3.
     When the person answers instead, for example "start it", stop "Wait
     for a compaction" with the tool `TaskStop`, and continue to step 3.
     When it prints `no compaction`, the person did not answer: do nothing.
3. Start a monitor with the follow-along brief, as "Start a monitor" says.
   Tell the person in one line that the helper runs, and that they can ask
   you to stop it at any time.

## Why a background agent

You can answer the person only once your turn ends. Any wait inside the turn
holds it open, and leaves the person without a reply. The monitor runs
outside your turn: it reads the notebook every few seconds, judges how the
work is progressing, and messages you only when you need to act.

Its agent file, `.claude/agents/notebook-monitor.md`, defines the commands
the monitor runs and how it judges the work. From you, it needs only a
brief: the goal, and the values that its commands take.

## Start a monitor

Do these steps in one turn: the turn in which the person agreed, or in
which you give the task to the person.

1. Give the task to the person, if there is one.
2. Write the brief. "Write the brief" below tells how.
3. Start the monitor with the tool `Agent`:
   - `name`: `notebook-monitor`
   - `subagent_type`: `notebook-monitor`
   - `run_in_background`: `true`
   - `description`: the goal, in a few words, or "Follow along"
   - `prompt`: the brief
4. End your turn. With a task, make the task the last text of your turn. Do
   not add a promise to wait, such as "I'll wait for you". The monitor
   waits, not you.

## Write the brief

### The follow-along brief

```
Follow along: the person works in the notebook {notebook}, and is a beginner.
About the person: {what you know about them, or "nothing yet"}.
Time limit: none, until I stop you.

Values: code "{code}", python "{python}", port {port}, token {token},
notebook {notebook}.
```

"About the person" says, in general terms, what you know: for example, a
child or an adult, and how much they know of programming and of Jupyter. No
personal details.

### The task brief

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
the person. A note gives what the monitor saw, and its opinion or
recommendation. The monitor is a small model: check its opinion before you
act on it.

When you follow along:

- **An error:** read the cell with `read_cell`, and find the cause
  yourself. Then tell the person, in plain words, what the error message
  says, where the problem is, and what to change.
- **A teaching moment:** decide whether it fits now. If it does, offer it in
  one or two lines, for example "Want to see how to do all five with one
  line?"
- **A recommendation for praise:** if you agree, praise what the person did,
  specifically.

You do not have to answer every note. When a note needs nothing from you,
end your turn without a message to the person.

With a task:

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
- **Before you change or run cells yourself:** tell the monitor with
  `SendMessage`, for example "I am changing cell 12" or "I am running all
  cells". Otherwise it reports your changes as the work of the person.
- **When you give the person a task or a suggestion,** tell the monitor
  too, with `SendMessage`, for example: "I suggested a loop that prints 1
  to 5, and explained the colon and the indentation." The monitor then
  judges the work against it, and knows what came from you.

Stop the monitor:

- **When the person asks for it.**
- **With a task, when the work in the notebook ends,** or when it changes
  to a task that needs no monitor.
- **Before you stop the Jupyter server.**
