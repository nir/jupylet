---
name: configuring-jupylet
description: Reads, writes and deletes settings in the Jupylet configuration files. Each file is .jupylet/claude-config.md. The global file is in the home folder of the person (for all Jupylet folders). The local file is in a Jupylet folder (for that folder only). Each setting is one key and one value. Use this skill when another Jupylet skill tells Claude to read, write or delete a setting.
---

# Configuring Jupylet

This skill keeps settings for the other Jupylet skills. It does not know what
a setting means. The skill that uses a setting decides its key, its values,
and when to read or write it.

## The files

There are two files, with the same form:

- **The global file:** `.jupylet/claude-config.md` in the home folder of the
  person. Its settings apply to all Jupylet folders on the computer.
- **The local file:** `.jupylet/claude-config.md` in the Jupylet folder, the
  folder that contains `.claude/skills/configuring-jupylet`. Its settings
  apply only to that Jupylet folder.

A local setting overrides the global setting with the same key.

The skill that uses a setting asks for it in one of these forms:

> Read the setting `open-notebooks-in` with the skill `configuring-jupylet`.
>
> Write the global setting `open-notebooks-in` = `claude-app` with the skill
> `configuring-jupylet`.
>
> Write the local setting `environment` = `jp146` with the skill
> `configuring-jupylet`.
>
> Delete the local setting `environment` with the skill `configuring-jupylet`.

A read names no file. A write or a delete names the file: global or local.

A new file has only this header:

```markdown
# Claude configuration for Jupylet

Settings for Claude when it works with Jupylet. One setting on each line, as
`- key = value`. You can change or delete any line.
```

Each setting is one list line after the header, for example:

```markdown
- some-key = some value
```

## Read a setting

Read the local file first, then the global file. The first value of the key
that you find is the value of the setting. Use it in the action that needs it.
If neither file has the key, the setting has no value. A missing file is
normal until you write the first setting. It is not a problem.

## Write a setting

Write to the file that the request names: global or local.

1. If the file does not exist, make the folder `.jupylet` in the home folder
   (global) or in the Jupylet folder (local), and write the file with the
   header above.
2. If the key is in the file, change its value. If not, add a line at the
   end.

## Delete a setting

Delete the line of the key from the file that the request names: global or
local. Keep the file, also when it has no settings.

## Limits

- Keep only `- key = value` lines after the header.
- Do not write names, tokens or private paths into the file.
