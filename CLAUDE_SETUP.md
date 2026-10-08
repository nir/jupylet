# CLAUDE_SETUP.md - download Jupylet for a person

These instructions are for Claude Code. A person gave you a link to this
file and asked you to install Jupylet. They are most likely a beginner, a
kid, or a parent. This file only downloads the Jupylet code into a new
folder and moves the session there. The skill `setting-up-jupylet`, which
comes with the code, does the rest of the installation.

Values in `{braces}` are blanks that you fill in.

## Talking to the person

- Be calm and friendly. Use short sentences and plain words. A relaxed tone
  is fine. Do not use emoji.
- Your first words are the hello below. Do not summarize these
  instructions. Do not name tools or commands.
- Ask one question per message, and wait for a clear yes before you change
  anything on their computer. Looking needs no question.
- Never show a raw error. Say in one plain sentence what happened.
- Talk about what you do on their computer, not about these instructions.

## Which code

Use the branch named in the link to this file. For example, the branch is
`claude` in `https://github.com/nir/jupylet/blob/claude/CLAUDE_SETUP.md`.
If the person names a branch in the chat, use that one. Otherwise use
`master`. Call it `{branch}`.

`{source}` is `https://github.com/nir/jupylet/archive/refs/heads/{branch}.tar.gz`.
For a test, the person can ask in the chat to use a local archive instead.
Only then is `{source}` that archive's `file://` URL.

## Commands

- macOS: use the Bash tool.
- Windows 11: use the PowerShell tool, never the Bash tool.

Put every path in double quotes.

## Step 1. Say hello

One short message with no question, for example:

> Hi. I'll set up Jupylet for you, one small step at a time, and I'll ask
> before I change anything on your computer. First, the Jupylet code.

## Step 2. Choose the folder

Propose the first of `jupylet`, `jupylet2`, `jupylet3`, ... in the home
folder that does not exist yet. Check with:

- macOS: `ls -d "$HOME/jupylet" "$HOME/jupylet2" "$HOME/jupylet3" 2>/dev/null`
- Windows 11: `'jupylet', 'jupylet2', 'jupylet3' | ForEach-Object { Join-Path $env:USERPROFILE $_ } | Where-Object { Test-Path $_ }`

Call the full path of the free one `{code}`. Then ask, for example:

> First I'll download the Jupylet code from GitHub, the website where it is
> kept. I'll put it in a new folder, `{code}`. Is that OK, or would you like
> to pick another place?

If they want another place, call `mcp__ccd_directory__request_directory`
without a path. They pick an existing folder. `{code}` is then the first free
name from the list above, inside that folder.

## Step 3. Download the code

After a clear yes, say one line, for example "Downloading the code now...".
Then run one command. It downloads the archive to a temporary place, creates
`{code}`, unpacks the code into it, and deletes the archive.

macOS:

`d="$(mktemp -d)" && curl -fL -o "$d/jupylet.tar.gz" "{source}" && mkdir "{code}" && tar -xzf "$d/jupylet.tar.gz" --strip-components=1 -C "{code}"; rm -rf "$d"`

Windows 11:

`$f = Join-Path $env:TEMP ('jupylet-' + [guid]::NewGuid() + '.tar.gz'); curl.exe -fL -o $f "{source}"; if ($LASTEXITCODE -eq 0) { New-Item -ItemType Directory "{code}" | Out-Null; tar.exe -xzf $f --strip-components=1 -C "{code}" }; Remove-Item $f -ErrorAction SilentlyContinue`

The archive holds one top folder. `--strip-components=1` skips it, so the
code lands in `{code}` itself. The command creates `{code}` only after the
download succeeds. On Windows, use `curl.exe`, not `curl`: in PowerShell,
`curl` is a different command.

If the command fails, tell the person plainly that the download did not
work, and try once more. If it fails again, stop. A common cause is a branch
name that does not exist on GitHub.

Then check that `{code}/.claude/skills/setting-up-jupylet/SKILL.md` exists.
If it does not, this branch does not have the setup skill. Tell the person
plainly that this version of Jupylet cannot be installed this way, and stop.

Tell the person in one line, for example "The code is in the folder
`{code}`." You put it there, so do not say that you found it.

## Step 4. Move the session to the code folder

Tell the person, for example:

> Next, I'll work from inside the folder `{code}`. The app will ask you to
> allow that, and it may call the folder a workspace. Please allow it.

Then call `mcp__ccd_directory__change_directory` with `{code}`. If that tool
is not available, read the skill file in step 5 by its full path anyway.

## Step 5. Hand over

Read `{code}/.claude/skills/setting-up-jupylet/SKILL.md` and follow it from
its start, with `{code}` as the Jupylet folder. Use full paths until your
next turn: the session's folder changes only when the current turn ends.
