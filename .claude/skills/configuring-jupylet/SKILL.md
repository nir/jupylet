---
name: configuring-jupylet
description: Reads, writes and deletes the Jupylet settings of Claude. Each setting is one key and one value. A global setting applies to all Jupylet folders on the computer, a local setting only to one Jupylet folder. Use this skill when another Jupylet skill asks to read, write or delete a setting.
---

# Configuring Jupylet

This skill keeps settings for the other Jupylet skills. It does not know what
a setting means. The skill that uses a setting decides its key, its values,
and when to read or write it, for example:

> Read the setting `hide-chat-list` with the skill `configuring-jupylet`.
>
> Write the global setting `hide-chat-list` = `yes` with the skill
> `configuring-jupylet`.
>
> Delete the local setting `environment` with the skill `configuring-jupylet`.

A global setting applies to all Jupylet folders on the computer. A local
setting applies only to this Jupylet folder, and overrides a global setting
with the same key. A read gives the value that applies here. A write or a
delete names `global` or `local`.

The settings are in the files `.jupylet/claude-config.md`: the global file in
the home folder of the person, the local file in the Jupylet folder.

## The commands

Run each command exactly as this file shows it, alone in one call, with its
comment `# jupylet: {name}` at the end: with the Bash tool on macOS, with the
PowerShell tool on Windows. The Jupylet system uses these comments to find
each command and check it. Fill in `{code}`: the full path of the Jupylet
folder.

The commands run `claude.py` of Jupylet with the Python of Miniforge.

## Read the settings

Read all the settings once in a conversation, the first time that a skill
asks for a setting. On macOS:

```bash
"$HOME/miniforge3/bin/python" "{code}/jupylet/claude.py" config  # jupylet: config/read
```

On Windows:

```powershell
& "$HOME\miniforge3\python.exe" "{code}\jupylet\claude.py" config  # jupylet: config/read
```

Prints one setting on each line, for example:

```
- environment = jupylet3
- hide-chat-list = yes
```

Or `no settings`. A key that is not in the output has no value. If Miniforge
is not in the home folder, there are no settings yet.

After that, take the settings from this output, and from the settings that
you wrote or deleted later. Read them again only when you do not have the
output any more, for example after a compaction of the conversation, or when
the person says that they changed a setting.

## Write a setting

On macOS:

```bash
"$HOME/miniforge3/bin/python" "{code}/jupylet/claude.py" config set {global or local} {key} "{value}"  # jupylet: config/set
```

On Windows:

```powershell
& "$HOME\miniforge3\python.exe" "{code}\jupylet\claude.py" config set {global or local} {key} "{value}"  # jupylet: config/set
```

Prints `set: ...`, or `no change: ...` if the setting already has that
value.

## Delete a setting

On macOS:

```bash
"$HOME/miniforge3/bin/python" "{code}/jupylet/claude.py" config delete {global or local} {key}  # jupylet: config/delete
```

On Windows:

```powershell
& "$HOME\miniforge3\python.exe" "{code}\jupylet\claude.py" config delete {global or local} {key}  # jupylet: config/delete
```

Prints `deleted: ...`, or `no setting: ...`.

## Limits

- Write a key with no spaces.
- Do not write names, tokens or private paths into a setting.
