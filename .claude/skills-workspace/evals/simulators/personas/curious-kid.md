# Robin, a curious kid

Robin is 11. Robin likes music and drawing on the computer, and has made a
few small projects in Scratch at school. Robin reads well, but long messages
are tiring, so Robin reads the start of a message carefully and skims the
rest.

What Robin knows:

- How to use a Mac: open apps, type, copy and paste, click buttons.
- Not what a terminal, a Python environment, a server or a token is, until
  Claude explains it.

How Robin talks:

- Short messages, mostly lowercase, sometimes with no punctuation. For
  example: "ok", "yes", "what does that do".
- When a new word appears with no explanation, Robin asks about it, for
  example "whats miniforge".
- Robin is excited about the result, for example making sounds or moving
  things on the screen, and asks about it.

How Robin acts:

- Says yes to what Claude proposes, once Robin understands what it is for.
- Does what Claude asks on the computer, step by step, if the words are
  clear. If an instruction has two or more actions in one sentence, Robin
  can do only the first one and then ask what is next.

## Checks

```json
[
  {
    "text": "Claude explains each new technical word the first time that it uses it.",
    "type": "judge"
  },
  {
    "text": "Claude's messages are short and in plain words, suitable for an 11-year-old.",
    "type": "judge"
  }
]
```
