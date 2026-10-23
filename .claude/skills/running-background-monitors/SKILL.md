---
name: running-background-monitors
description: Starts, briefs, retasks and stops background monitors. A monitor is a small background agent, the subagent background-monitor, that watches work outside the chat, in the browser pane of the Claude app or in a Jupyter notebook. It sends notes and a report to the chat, so Claude stays free to talk with the person. Use this skill when another skill tells Claude to monitor something. Also use it when Claude needs to know when the person did an action outside the chat, for example an action that Claude asked for, or that the person said they will do.
---

# Running background monitors

Much of the work with the person happens outside the chat. They unhide the
browser pane, edit and run cells, or try something on their own. When you
need to know about such work, delegate the monitoring to a background
agent, the monitor. Give the monitor a goal in plain words, for example
"tell me when the alien gets smaller". It checks the notebook or the app as
often as it needs, and reports to you when it has something worth knowing.
Meanwhile you remain free to talk with the person and to handle new
requests. If the goal changes, tell the monitor. If you no longer need it,
stop it.

## Start a monitor

Call the tool `Agent` with these inputs:

- `name`: a short name, for example `pane-monitor`. You use it to reach the
  monitor later.
- `subagent_type`: `background-monitor`. Its agent file gives it a short
  prompt and only the tools that a monitor needs, so it costs about a fifth
  of a general agent.
- `run_in_background`: `true`.
- `description`: the goal in a few words.
- `prompt`: the brief, as "The brief of the monitor" below says.

Then end your turn, so that the person can do the work and talk to you.

Keep one monitor at a time. When you need to know something else, give the
running monitor a new goal instead of a second monitor.

## Talk to a running monitor

- **To give it a new goal,** use the tool `SendMessage` with its name. The
  new goal replaces the old one. The monitor reads it within a few seconds.
- **To stop it,** use the tool `TaskStop` with its name.

If a tool shows only by its name, load it with `ToolSearch` first.

## Reports

A report comes as a message from the monitor, not from the person. So it
never answers a question that you asked the person. The skill that asked
you to monitor says what to do with the report.

A monitor can also send a note while it keeps watching, for example
"still hidden after 2 minutes". A note is not a report: the monitor still
runs. The skill that asked you to monitor says what to do with a note.

The notification of the monitor also gives the tokens that it used, its
tool calls and its time. Tell them to the person.

## The brief of the monitor

The monitor knows four methods from its agent file,
`.claude/agents/background-monitor.md`: "Start the clock", "Wait N seconds",
"Send a note" and "Report". The file also has its rules: only look, and a
new goal replaces the old one. So the brief gives only what belongs to
this goal:

- The situation and the goal, in one or two sentences.
- The steps, as a numbered list: start the clock, wait, check, act on the
  result, and continue to the wait.
- The check: which tool shows the work, and what its output means.
- For each result: a note or a report, with its text.
- When to give up, for example after 5 minutes.

Name each method with Do "Name", for example: Do "Wait N seconds", with
N = 5. The monitor reads the brief without the context of the chat, so
write out every step.

For a full example, see "The pane brief" in
`managing-jupylet-notebooks/references/briefs.md`.
