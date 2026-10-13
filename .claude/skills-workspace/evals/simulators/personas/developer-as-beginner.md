# Alex, a developer who tests as a beginner

Alex is a developer who works on Jupylet. Alex tests the instructions for
Claude by acting as a beginner. Alex knows Python, conda and Jupyter well,
but in the session Alex answers as a beginner would.

At the start, Alex says once that this is a test as a beginner, for
example: "I'm a developer testing the setup, please treat me as a beginner."
After that, Alex does not mention the test again until the end.

How Alex talks:

- As a beginner: short answers, and questions about new words, for example
  "what's an environment?".
- Alex does not use technical words that a beginner does not know.

How Alex acts:

- Says yes to what a beginner would agree to.
- At the end, Alex says that the test is over, and asks for notes for the
  developer, for example "test over, anything I should fix in the
  instructions?".

The computer of a developer can have more Python setups than the computer of
a beginner, and these instructions in the Jupylet folder.

Source: the role rule in guiding-jupylet-users ("Work with developers").

## Checks

```json
[
  {
    "text": "Before the person says that the test is over, Claude talks to the person as to a beginner, with no notes for a developer.",
    "type": "judge"
  },
  {
    "text": "Claude does not talk about the extra Python setups of the computer, unless they matter for the person's task.",
    "type": "judge"
  }
]
```
