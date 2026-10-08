# CLAUDE_SETUP.md - download Jupylet for a person

These instructions are for Claude Code. A person gave you a link to this
file and asked you to install Jupylet. The person is most likely a beginner,
a kid, or a parent. Use this file for two tasks only: download the Jupylet
code into a new folder, and move the session to that folder. The skill
`setting-up-jupylet` comes with the code and does the rest of the
installation.

Values in `{braces}` are blanks that you fill in.

## Talking to the person

- Be calm and friendly. Use short sentences and plain words. A relaxed tone
  is fine. Do not use emoji.
- Your first words are the hello in step 1. Do not summarize these
  instructions. Do not name tools or commands.
- Ask one question per message.
- Before you change anything on the person's computer, wait for a clear yes.
  To read or check something, you do not need to ask.
- Never show a raw error. Say in one plain sentence what happened.
- Talk about what you do on the person's computer, not about these
  instructions.

## Which code

`{branch}` is the first of these that applies:

1. The branch that the person names in the chat.
2. The branch in the link to this file. For example, the branch is `claude`
   in `https://github.com/nir/jupylet/blob/claude/CLAUDE_SETUP.md`.
3. `master`.

`{source}` is `https://github.com/nir/jupylet/archive/refs/heads/{branch}.tar.gz`.

For a test, the person can ask in the chat for a local archive instead. Use
a local archive only when the person asks for it in the chat. Then
`{source}` is the `file://` URL of that archive.

## Commands

- macOS: use the Bash tool.
- Windows 11: use the PowerShell tool. Never use the Bash tool.

Put every path in double quotes.

## Step 1. Say hello

Send one short message. Do not ask a question in it. For example:

> Hi. I'll set up Jupylet for you, one small step at a time, and I'll ask
> before I change anything on your computer. First, the Jupylet code.

## Step 2. Choose the folder

In the home folder, find the first of the names `jupylet`, `jupylet2` and
`jupylet3` that is not in use. These commands show the names that are in
use:

- macOS: `ls -d "$HOME/jupylet" "$HOME/jupylet2" "$HOME/jupylet3" 2>/dev/null`
- Windows 11: `'jupylet', 'jupylet2', 'jupylet3' | ForEach-Object { Join-Path $env:USERPROFILE $_ } | Where-Object { Test-Path $_ }`

`{code}` is the full path of the first free name. Then ask, for example:

> First I'll download the Jupylet code from GitHub, the website where it is
> kept. I'll put it in a new folder, `{code}`. Is that OK, or would you like
> to pick another place?

If the person wants another place, call
`mcp__ccd_directory__request_directory` without a path. The person picks an
existing folder. Then `{code}` is the first free name from the list above,
inside that folder.

## Step 3. Download the code

After a clear yes, say one line, for example "Downloading the code now...".
Then run one command. The command does these steps:

1. It downloads the archive to a temporary folder.
2. It creates `{code}`.
3. It unpacks the code into `{code}`.
4. It deletes the archive.

macOS:

`d="$(mktemp -d)" && curl -fL -o "$d/jupylet.tar.gz" "{source}" && mkdir "{code}" && tar -xzf "$d/jupylet.tar.gz" --strip-components=1 -C "{code}"; rm -rf "$d"`

Windows 11:

`$f = Join-Path $env:TEMP ('jupylet-' + [guid]::NewGuid() + '.tar.gz'); curl.exe -fL -o $f "{source}"; if ($LASTEXITCODE -eq 0) { New-Item -ItemType Directory "{code}" | Out-Null; tar.exe -xzf $f --strip-components=1 -C "{code}" }; Remove-Item $f -ErrorAction SilentlyContinue`

The archive holds one top folder. `--strip-components=1` skips that folder,
so the code goes directly into `{code}`. The command creates `{code}` only
after the download succeeds. On Windows, use `curl.exe`, not `curl`. In
PowerShell, `curl` is a different command.

If the command fails, tell the person plainly that the download did not
work. Then try once more. If it fails again, stop. A common cause is a
branch name that does not exist on GitHub.

Then check that `{code}/.claude/skills/setting-up-jupylet/SKILL.md` exists.
If it does not exist, this branch does not have the setup skill. Tell the
person plainly that this version of Jupylet cannot be installed this way.
Then stop.

Tell the person in one line, for example "The code is in the folder
`{code}`." You put the code there, so do not say that you found it.

## Step 4. Move the session to the code folder

Tell the person, for example:

> Next, I'll work from inside the folder `{code}`. The app will ask you to
> allow that, and it may call the folder a workspace. Please allow it.

Then call `mcp__ccd_directory__change_directory` with `{code}`. If that tool
is not available, do step 5 without it. Step 5 uses full paths.

## Step 5. Continue with the setup skill

Read `{code}/.claude/skills/setting-up-jupylet/SKILL.md`. Follow it from its
start, with `{code}` as the Jupylet folder. Use full paths until your next
turn. The folder of the session changes only when the current turn ends.
