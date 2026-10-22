# The persona and system library

Draft, for review. `library.py` reads these files for the scenarios in `scenarios/`.

A scenario picks one persona and one system, and adds a request:

```
scenario = persona + system + request + its own checks
```

- **Persona** (`personas/*.md`): who the person is, what they know, how they
  talk and act. The human simulator gets it.
- **System** (`systems/*.md`): the state of the computer at the start. The
  system simulator gets it as its seed.
- **Request:** what the person wants in this session, how the session ends,
  and the answers that the scenario fixes (for example "says no to the
  Terminal change"). It is in the scenario, not in the persona, so that one
  persona serves many scenarios.

## File form

Each persona and system file has two parts:

1. The text for the simulator, from the top of the file to the heading
   `## Checks`.
2. `## Checks`: a JSON list of checks, in the form of `smoke.json`. The
   harness adds them to the checks of each scenario that uses the file. The
   simulators never see this part, so that they do not act for the checks.

A system file can start with `Extends: {name}`. The harness then puts the
text of that system first, and this file adds to it or changes it.

A base system starts with `Platform: macos` or `Platform: windows` (no line
means macos). The platform sets the home folder, the scratchpad and the
shell tool of the worker: Bash on macOS, PowerShell on Windows. The systems
that extend it have its platform. Their checks name the tool of their
platform, for example `"tool": "PowerShell"`.

Blanks: `{home}` (the home folder), `{user}` (the user name), `{code}` (the
Jupylet folder of the scenario), `{scratchpad}` (the scratchpad folder of
the worker).

## How to write a persona

The people in these files stand for real users of Jupylet: kids, parents,
teachers, hobbyists and developers. Some of them will read these files.
Write each persona so that a person like it can read it and feel described
with respect.

- **Describe experience, goals and habits,** not character flaws. "Has not
  used Terminal before" is useful for a test. "Is bad with computers" is not,
  and it is unkind.
- **Each trait must serve a test.** Add a trait only when it makes the
  simulator act in a way that tests a skill, for example short answers,
  questions about words, or a no to a change. Do not add color for its own
  sake.
- **No stereotypes** by age, gender, origin, language or ability. Age is
  stated only where it changes what the person knows or how they read, for
  example a kid.
- **Gender-neutral names, and they/them.**
- **The person is reasonable.** When the person does something unexpected,
  it has a reason that a real person has, for example "wants to understand a
  change before saying yes", not "gets confused easily".
- **Real behavior, from real sessions,** where we have it. Name the
  transcript or finding.

## How to write a system

- **Real outputs.** Take file contents and command outputs from real
  computers or transcripts, not from memory. Mark an invented detail as such.
- **Only add, never change.** A system that extends another one adds
  things to it, for example Miniconda in Terminal, and never changes or
  removes what the other one says. The same holds for the `seed` of a
  scenario. So no file says "No …" about a thing that another file could
  add: the system simulator treats each thing that no part names as absent.
  A file states an absence only when nothing extends it, and only for a
  thing that no part above it names. In run `ask-miniforge-18`, a base that
  said "No `~/.zshrc`" won over the `.zshrc` of the system that extended it.
- **One difference per file.** A system extends a base and adds one thing.
  That way, a failed check points to one cause.
- **Checks that the system brings** are the checks that hold for any
  scenario on it, for example "Miniconda stays unchanged".
