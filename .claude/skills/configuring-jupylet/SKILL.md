---
name: configuring-jupylet
description: Reads, writes and deletes settings in the Jupylet configuration files. Each file is .jupylet/claude-config.md. The global file is in the home folder of the person (for all Jupylet folders). The local file is in a Jupylet folder (for that folder only). Each setting is one key and one value. Use this skill when another Jupylet skill asks to read, write or delete a setting.
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

> Read the setting `hide-chat-list` with the skill `configuring-jupylet`.
>
> Write the global setting `hide-chat-list` = `yes` with the skill
> `configuring-jupylet`.
>
> Write boxless the local setting `environment` = `jupylet2` with the skill
> `configuring-jupylet`.
>
> Delete the local setting `environment` with the skill `configuring-jupylet`.

A read names no file. A write or a delete names the file: global or local.

A file starts with this header:

```markdown
# Claude configuration for Jupylet

Settings for Claude when it works with Jupylet. One setting on each line,
as `- key = value`. When several lines set the same key, the last of these
lines counts.
```

Each line after the header sets one key, for example:

```markdown
- some-key = some value
```

A boxless write adds a line at the end of the file, and does not change the
lines above it. So several lines can set the same key. Of these lines, the
last one, nearest to the end of the file, gives the value of the key in that
file. The lines above it that set the same key do not count.

## Read a setting

Read the files with the Read tool, in the order of the steps below. When a
file does not exist, the Read tool gives an error. That error is normal: a file does not exist until you write
the first setting to it.

1. If the local file exists and a line in it sets the key, the last line
   that sets the key gives the value of the setting. Stop.
2. If the global file exists and a line in it sets the key, the last line
   that sets the key gives the value of the setting. Stop.
3. The setting has no value.

## Write a setting

Write to the file that the request names: global or local. Use the Write and
Edit tools.

1. If the file does not exist, make the folder `.jupylet` in the home folder
   (global) or in the Jupylet folder (local), and write the file with the
   header above.
2. Find the lines in the file that set the key:
   - **No line:** add the line `- {key} = {value}` at the end of the file.
   - **One line:** change its value.
   - **Several lines:** change the value in the last line that sets the key.
     Then delete all other lines that set the key.

## Write a setting boxless

The Claude app shows each file that the Write or Edit tool changes as a box
in the chat. During the installation, that box confuses a beginner. A
boxless write does not use these tools, so the app shows no box. Run the
commands below: with the Bash tool on macOS, with the PowerShell tool on
Windows. Do a boxless write only when the request says "boxless".

Run each command exactly as this section shows it, alone in one call, with
its comment `# jupylet: {name}` at the end. The Jupylet system uses these
comments to find each command and check it. If you leave out the comment or
change the command, this may cause failures.

Fill in these blanks, with full paths:

- `{folder}`: the folder `.jupylet` of the file that the request names: in
  the home folder of the person (global), or in the Jupylet folder (local).
- `{header}`: the file `header.md` of this skill,
  `.claude/skills/configuring-jupylet/header.md` in the Jupylet folder. It
  has the header above.

1. Read the file with the Read tool. If the last line that sets the key
   already has the new value, stop. There is nothing to write.
2. If the file does not exist, make the folder and the file with the header.
   On macOS:

   ```bash
   mkdir -p "{folder}"  # jupylet: config/make-folder
   ```

   ```bash
   cp -n "{header}" "{folder}/claude-config.md"  # jupylet: config/copy-header
   ```

   On Windows:

   ```powershell
   New-Item -ItemType Directory -Force "{folder}"  # jupylet: config/make-folder
   ```

   ```powershell
   New-Item -ItemType File -Path "{folder}\claude-config.md" -Value (Get-Content -Raw "{header}")  # jupylet: config/copy-header
   ```

   These two copy commands never change a file that exists: `cp -n` changes
   nothing and ends with exit code 1, and `New-Item` without `-Force` gives
   an error. So a copy at the wrong moment cannot delete settings.

3. Add the setting at the end of the file. On macOS:

   ```bash
   echo "- {key} = {value}" >> "{folder}/claude-config.md"  # jupylet: config/add-line
   ```

   On Windows:

   ```powershell
   Add-Content -Path "{folder}\claude-config.md" -Value "- {key} = {value}"  # jupylet: config/add-line
   ```

## Delete a setting

Delete all lines that set the key from the file that the request names:
global or local. Use the Edit tool. Keep the file, also when it has no settings.

## Limits

- Keep only `- key = value` lines after the header.
- Do not write names, tokens or private paths into the file.
