"""Run one simulated session of the Jupylet skills.

Three agents talk: the worker (Claude with the skills, no real tools), the
system simulator (the computer and the Claude app), and the human simulator
(the person). EVALS.md describes the design.

Usage:
    run_eval.py <evals.json> <eval id> <run dir> [--skills DIR | --ref REF]
                [--system NAME] [--fake] [--live] [--model MODEL]

--skills DIR: a Jupylet folder whose skills the worker uses (default: the
  repository that contains this script).
--ref REF: use the skills of a git commit instead, for a baseline.
--system NAME: the system of the library to run the scenario on, instead of
  the system that the scenario names, for example win-miniconda.
--fake: use scripted stand-ins for the three agents. No model is called, so
  this tests the harness itself, for free.
--live: show the whole transcript in the terminal while the run goes on:
  each message in full, and each command output (up to 40 lines). Without
  it, the terminal shows one short line for each event.
"""

import argparse
import datetime
import glob
import json
import ntpath
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import library
import recordings


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
MODEL = 'claude-haiku-5-5'
#
# The effort of all three agents. At low, the system simulator lost a part of
# its seed (run ask-miniforge-20). Nir: all agents at medium.
#
EFFORT = 'medium'


def claude_bin():

    if os.environ.get('CLAUDE_BIN'):
        return os.environ['CLAUDE_BIN']

    pattern = os.path.expanduser(
        '~/Library/Application Support/Claude/claude-code/*/*/claude.app/Contents/MacOS/claude')
    found = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not found:
        sys.exit('claude CLI not found: set CLAUDE_BIN')

    return found[-1]


def read(path):

    with open(path, encoding='utf-8') as f:
        return f.read()


class Agent:
    """One Haiku session, resumed for each message."""

    def __init__(self, name, system, run_dir, model, fake=None, cwd=None, tools='', append=False, extra=(),
                 effort=None):

        self.name = name
        self.system = system
        self.model = model
        self.fake = fake
        self.tools = tools
        self.append = append
        self.seen_tools = 0
        self.seen_lines = 0
        self.extra = list(extra) + (['--effort', effort] if effort else [])
        self.session = str(uuid.uuid4())
        self.started = False
        self.calls = 0
        self.totals = {'duration_ms': 0, 'cost_usd': 0.0}
        self.log = os.path.join(run_dir, 'agents', name + '.jsonl')
        #
        # An empty folder outside the repository: the CLI loads CLAUDE.md
        # and .claude/ of the folders above its working folder, and inside
        # the repository that would give each agent the Jupylet skills.
        #
        self.cwd = cwd or tempfile.mkdtemp(prefix='jupylet-eval-%s-' % name)
        self.own_cwd = not cwd
        os.makedirs(os.path.join(run_dir, 'agents'), exist_ok=True)

        with open(os.path.join(run_dir, 'agents', name + '.system.md'), 'w', encoding='utf-8') as f:
            f.write(system)

    def send(self, text):

        self.calls += 1
        t0 = time.time()

        if self.fake:
            reply, meta = self.fake(self, text), {}
        else:
            reply, meta = self._call(text)

        self.totals['duration_ms'] += meta.get('duration_ms', int(1000 * (time.time() - t0)))
        #
        # modelUsage and total_cost_usd are the totals of the whole session
        # so far, over all its model requests (a worker call makes one
        # request for each tool call). So the last call's figures are the
        # totals; summing them over calls counts early calls many times.
        #
        for u in meta.get('modelUsage', {}).values():
            for key, field in (('input_tokens', 'inputTokens'), ('cache_write_tokens', 'cacheCreationInputTokens'),
                               ('cache_read_tokens', 'cacheReadInputTokens'), ('output_tokens', 'outputTokens'),
                               ('thinking_tokens', 'thinkingTokens')):
                self.totals[key] = u.get(field, 0)
        if meta:
            self.totals['cost_usd'] = meta.get('total_cost_usd', 0.0)

        with open(self.log, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'in': text, 'out': reply, 'meta': meta}) + '\n')

        return reply

    def _call(self, text):

        args = [claude_bin(), '-p', '--model', self.model, '--output-format', 'json',
                '--tools', self.tools, '--strict-mcp-config',
                #
                # Only the settings of the project: without this, the agents
                # loaded Nir's own skills and plugins (run
                # full-install-miniconda-1).
                #
                '--setting-sources', 'project'] + self.extra

        if self.started:
            args += ['--resume', self.session]
        else:
            flag = '--append-system-prompt' if self.append else '--system-prompt'
            args += ['--session-id', self.session, flag, self.system]

        env = dict(os.environ, CLAUDE_CODE_DISABLE_AUTO_MEMORY='1')
        p = subprocess.run(args, input=text, capture_output=True, text=True,
                           cwd=self.cwd, timeout=600, env=env)
        self.started = True

        if p.returncode != 0:
            try:
                why = json.loads(p.stdout).get('result', '')
            except ValueError:
                why = p.stdout[-2000:]
            raise RuntimeError('%s: claude exited %d: %s %s' % (
                self.name, p.returncode, why, p.stderr[-2000:]))

        meta = json.loads(p.stdout)
        if meta.get('is_error'):
            raise RuntimeError('%s: %s' % (self.name, meta.get('result')))

        return meta.get('result', ''), meta

    def transcript(self):
        """The session file that the CLI writes, or None."""
        found = glob.glob(os.path.expanduser('~/.claude/projects/*/%s.jsonl' % self.session))

        return found[0] if found else None

    def new_items(self):
        """What the session added since the last time, in order.

        Returns a list of ('text', str), ('tool_use', id, name, input) and
        ('tool_result', id, text).
        """
        items, self.seen_lines = self.items_since(self.seen_lines)

        return items

    def items_since(self, start):
        """The items of the session file from line `start`, and the next start."""
        path = self.transcript()
        if not path:
            return [], start

        lines = open(path, encoding='utf-8').read().splitlines()
        #
        # The CLI can be in the middle of a line: keep it for the next time.
        #
        if lines and not lines[-1].rstrip().endswith('}'):
            lines = lines[:-1]
        new, end = lines[start:], max(start, len(lines))

        items = []
        for line in new:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            content = e.get('message', {}).get('content') if e.get('type') in ('assistant', 'user') else None
            if not isinstance(content, list):
                continue
            for b in content:
                k = b.get('type')
                if k == 'text' and e['type'] == 'assistant' and b['text'].strip():
                    items.append(('text', b['text'].strip()))
                elif k == 'tool_use':
                    items.append(('tool_use', b['id'], b['name'], b.get('input', {})))
                elif k == 'tool_result':
                    r = b.get('content')
                    if isinstance(r, list):
                        r = '\n'.join(x.get('text', '') for x in r)
                    items.append(('tool_result', b.get('tool_use_id'), str(r)))

        return items, end

    def new_real_tool_calls(self):
        """The real tool calls (Skill) of the session since the last time."""
        path = self.transcript()
        if not path:
            return []

        calls = []
        for line in open(path, encoding='utf-8'):
            e = json.loads(line)
            if e.get('type') != 'assistant':
                continue
            for b in e['message'].get('content', []):
                if isinstance(b, dict) and b.get('type') == 'tool_use':
                    calls.append((b['name'], b.get('input', {})))

        new, self.seen_tools = calls[self.seen_tools:], len(calls)

        return new


#
# Parsing the replies of the agents.
#

TOOL_RE = re.compile(r'<tool name="([^"]+)">\s*(.*?)\s*</tool>', re.S)


def parse_worker(reply):
    """Return (visible text, [(tool, input dict or error string)])."""
    tools = []
    for name, body in TOOL_RE.findall(reply):
        try:
            tools.append((name, json.loads(body)))
        except json.JSONDecodeError as e:
            tools.append((name, 'invalid JSON input: %s' % e))

    text = TOOL_RE.sub('', reply).strip()

    return text, tools


def parse_human(reply):

    return {
        'say': [s.strip() for s in re.findall(r'<say>(.*?)</say>', reply, re.S)],
        'do': [s.strip() for s in re.findall(r'<do>(.*?)</do>', reply, re.S)],
        'wait': '<wait/>' in reply,
        'end': '<end/>' in reply,
    }


def parse_system(reply):

    m = re.search(r'<output>\n?(.*?)\n?</output>', reply, re.S)
    output = m.group(1) if m else reply.strip()
    s = re.search(r'<state>\n?(.*?)\n?</state>', reply, re.S)
    done = re.findall(r'<done id="([^"]+)">\n?(.*?)\n?</done>', reply, re.S)

    return output, (s.group(1) if s else None), done


#
# The skills that the worker sees.
#

#
# Media and 3D files are not copied (Nir): Claude does not read them. The
# list of files in the seed still names them, since they are in the folder.
#
SKIP = ('.png', '.jpg', '.gif', '.ogg', '.flac', '.wav', '.mp4', '.blend', '.bin', '.gltf',
        '.ttf', '.otf', '.sfz')


def tracked_files(args):
    """The paths of the files of the code under test, relative to its folder."""
    src = os.path.abspath(args.skills) if args.skills else REPO
    if args.ref:
        out = subprocess.run(['git', '-C', REPO, 'ls-tree', '-r', '--name-only', '-z', args.ref],
                             capture_output=True, check=True).stdout
    else:
        out = subprocess.run(['git', '-C', src, 'ls-files', '-z'], capture_output=True, check=True).stdout

    return [rel for rel in out.decode().split('\0') if rel]


def skills_folder(args):
    """A copy of the Jupylet code under test, in a new folder named jupylet.

    The copy has every file that git tracks, as a person who downloads
    Jupylet gets it, and the whole folder .claude/skills, also its files that
    git does not track yet. The worker runs in this folder, so the CLI loads
    its CLAUDE.md and skills as in a real session. It is outside the
    repository, so nothing else of the repository loads, for example its
    settings, its memory, or skills-workspace with the evals.
    """
    #
    # realpath: on macOS the temp folder is under /var, a link to
    # /private/var, and the CLI gives the worker the /private/var path.
    #
    d = os.path.join(os.path.realpath(tempfile.mkdtemp(prefix='jupylet-eval-code-')), 'jupylet')
    os.makedirs(d)

    keep = [rel for rel in tracked_files(args) if not rel.lower().endswith(SKIP)]

    if args.ref:
        archive = subprocess.run(['git', '-C', REPO, 'archive', args.ref, '--'] + keep,
                                 capture_output=True, check=True)
        subprocess.run(['tar', '-x', '-C', d], input=archive.stdout, check=True)
        return d

    src = os.path.abspath(args.skills) if args.skills else REPO
    for rel in keep:
        s = os.path.join(src, rel)
        if os.path.isfile(s):
            os.makedirs(os.path.dirname(os.path.join(d, rel)), exist_ok=True)
            shutil.copy2(s, os.path.join(d, rel))

    shutil.copytree(os.path.join(src, '.claude', 'skills'), os.path.join(d, '.claude', 'skills'), dirs_exist_ok=True)

    return d


TAG = re.compile(r'#\s*jupylet:\s*([\w-]+/[\w-]+)')
SHELLS = ('Bash', 'PowerShell')


def command_tags(folder):
    """Each `# jupylet: {group}/{name}` in the files under test, with the
    command lines that carry it (the same tag can be on a macOS and a
    Windows command)."""
    tags = {}
    paths = [os.path.join(d, n) for d, _, names in os.walk(folder) for n in names if n.endswith('.md')]
    for path in sorted(paths):
        rel = os.path.relpath(path, folder)
        for line in read(path).splitlines():
            m = TAG.search(line)
            if m:
                tags.setdefault(m.group(1), []).append({'file': rel, 'line': line.strip().strip('`').strip()})

    return tags


def git_stamp(args):
    """The commit of the files under test, and the files that differ from it.

    Runs are comparable only when these match: the skills, the CLAUDE files
    and claude.py, whose outputs the system simulator gives.
    """
    if args.skills:
        return {'source': os.path.abspath(args.skills)}

    def git(*a):
        return subprocess.run(['git', '-C', REPO] + list(a), capture_output=True, text=True).stdout.rstrip()

    if args.ref:
        return {'commit': git('rev-parse', args.ref)}

    #
    # The tracked files that differ from the commit (the copy has them as on
    # disk), and the files of the skills that git does not track yet.
    #
    changed = git('status', '--porcelain', '-uno', '--', '.', ':(exclude).claude/skills-workspace')
    new = git('ls-files', '--others', '--exclude-standard', '--', '.claude/skills')

    return {'commit': git('rev-parse', 'HEAD'),
            'changed': [line[3:] for line in changed.splitlines()] + new.splitlines()}


def frontmatter(text):

    m = re.match(r'---\n(.*?)\n---\n', text, re.S)
    if not m:
        return {}, text

    meta = dict(line.split(': ', 1) for line in m.group(1).splitlines() if ': ' in line)

    return meta, text[m.end():]


def skill_list(src):

    lines = []
    for path in sorted(glob.glob(os.path.join(src, '.claude', 'skills', '*', 'SKILL.md'))):
        meta, _ = frontmatter(read(path))
        lines.append('- %s: %s' % (meta.get('name', ''), meta.get('description', '')))

    return '\n'.join(lines)


def numbered(text):

    return '\n'.join('%6d\t%s' % (i + 1, line) for i, line in enumerate(text.splitlines()))


class Files:
    """Serve the real Jupylet files of the skills folder to the worker."""

    def __init__(self, src, code):

        self.src = src
        self.code = code.rstrip('/')
        self.changes = []

    def changed(self, tool, inp):
        """Note an action that can change files. A file that an action named,
        or a folder of it, is state from then on: the system simulator
        answers its reads, not the copy."""
        if tool in SHELLS:
            self.changes.append(inp.get('command', '').replace('\\', '/'))
        elif tool in ('Write', 'Edit'):
            self.changes.append(inp.get('file_path', ''))

    def may_have_changed(self, path):
        """True when an earlier action named this file, or named one of its
        folders as a whole (a path that ends there, as in `cd "…/examples"`).
        A deeper path does not count: `cp -n "…/.claude/skills/…/header.md"`
        names only header.md (run open-spaceship-4). The skills never
        change in a run."""
        rel = path[len(self.code) + 1:]
        if rel.startswith('.claude/skills/'):
            return False

        names = [path, rel]
        parent = os.path.dirname(path)
        while len(parent) > len(self.code):
            names += [parent, parent[len(self.code) + 1:]]
            parent = os.path.dirname(parent)
        #
        # Only absolute paths, and relative paths with a folder: a bare
        # name such as `jupylet` is also in `-m jupylet trust_notebooks`.
        #
        names = [n for n in names if '/' in n]

        return any(re.search(re.escape(n) + r'(?=["\'\s;&|)]|$)', c) for c in self.changes for n in names)

    def local(self, path):
        #
        # On Windows, Claude writes the paths of the copy with `\`.
        #
        path = os.path.realpath(path.strip().strip('"\'').replace('\\', '/'))
        if not path.startswith(self.code + '/'):
            return None

        p = os.path.join(self.src, path[len(self.code) + 1:])
        if not os.path.isfile(p) or self.may_have_changed(path):
            return None

        return p

    @staticmethod
    def notebook_names(text):

        return sorted(set(re.findall(r'([\w-]+)\.ipynb', text) + re.findall(r'"notebook_name":\s*"([\w-]+)', text)))

    def notebook_text(self, name, path=None):
        """The cells of an example notebook, or of the notebook at `path`, in
        the percent format of jupytext: each cell under a `# %%` line,
        Markdown as comments. Easier to read than the JSON of the .ipynb
        file. None if no such notebook."""
        p = path or os.path.join(self.src, 'examples', name + '.ipynb')
        try:
            cells = json.load(open(p, encoding='utf-8')).get('cells', [])
        except (OSError, ValueError):
            return None

        out = []
        for c in cells:
            src = ''.join(c.get('source', [])).rstrip('\n')
            if c.get('cell_type') == 'markdown':
                out.append('# %% [markdown]\n' + '\n'.join(('# ' + line).rstrip() for line in src.splitlines()))
            else:
                out.append('# %%\n' + src)

        return '\n\n'.join(out)

    def notebook_cells(self, text):
        """The real cells of each example notebook that a command names, for
        the system simulator: it gives the run state, the copy gives the
        code."""
        out = []
        for name in self.notebook_names(text):
            nb = self.notebook_text(name)
            if nb:
                out.append('<notebook name="%s.ipynb">\nThe real cells of this notebook, in the percent '
                           'format of jupytext. Give their sources exactly. You give only the state of the '
                           'run: which cells ran, and their outputs.\n\n%s\n</notebook>' % (name, nb))

        return '\n\n'.join(out)

    def skill(self, name):

        p = os.path.join(self.src, '.claude', 'skills', name, 'SKILL.md')
        if not os.path.isfile(p):
            return 'Unknown skill: %s' % name

        _, body = frontmatter(read(p))

        return 'Base directory for this skill: %s/.claude/skills/%s\n\n%s' % (self.code, name, body)

    def serve(self, tool, inp):
        """Return the output for a tool call that reads our own files, or None."""
        if tool == 'Skill':
            return self.skill(inp.get('skill', ''))

        if tool == 'Read':
            p = self.local(inp.get('file_path', ''))
            if p and p.endswith('.ipynb'):
                #
                # The real Read tool gives a notebook as cells, not as JSON.
                # Here: the percent format of jupytext.
                #
                return self.notebook_text(os.path.basename(p)[:-6], p)
            return numbered(read(p)) if p else None

        if tool in SHELLS:
            m = re.fullmatch(r'\s*cat((?:\s+"[^"]+"|\s+\S+)+)\s*', inp.get('command', ''))
            if m:
                paths = [self.local(x) for x in re.findall(r'"[^"]+"|\S+', m.group(1))]
                if all(paths):
                    return '\n'.join(read(p) for p in paths)

        return None


#
# The session.
#

class Session:

    def __init__(self, ev, args):

        self.ev = ev
        self.args = args
        self.run_dir = os.path.abspath(args.run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        self.src = self.code = skills_folder(args)
        self.source = args.ref or (os.path.abspath(args.skills) if args.skills else 'working tree of ' + REPO)
        self.files = Files(self.src, self.code)
        self.git = git_stamp(args)
        #
        # The command tags of the files under test, for the grader: a tag
        # that is not in this list is one that Claude made up.
        #
        with open(os.path.join(self.run_dir, 'tags.json'), 'w', encoding='utf-8') as f:
            json.dump(command_tags(self.code), f, indent=2, ensure_ascii=False)
        #
        # The eval as it ran, with the library files resolved, so that
        # grade.py grades the same checks.
        #
        with open(os.path.join(self.run_dir, 'eval.json'), 'w', encoding='utf-8') as f:
            json.dump(ev, f, indent=2, ensure_ascii=False)
        self.state = ev['seed'].strip().replace('{code}', self.code) + '\n\n' + self.file_list()
        self.pending = {}
        self.pages_new, self.pages_shown = [], []
        self.recordings = recordings.load()
        self.task_files, self.task_info = {}, {}
        self.shell = ev.get('shell', 'Bash')
        paths = ntpath if self.shell == 'PowerShell' else os.path
        self.tasks_dir = paths.join(paths.dirname(self.ev.get('scratchpad', '/private/tmp/claude-501/scratchpad')),
                                    'tasks')
        self.events = []
        self.turn = 0
        self.fake = args.fake
        self.live = args.live

        sims = os.path.join(HERE, 'simulators')
        system_prompt = '\n\n'.join([
            read(os.path.join(sims, 'system.md')),
            read(os.path.join(sims, 'commands.md')),
            '# The computer at the start\n\n' + self.state,
        ])
        human_prompt = (read(os.path.join(sims, 'human.md')) + '\n\n## Your persona\n\n'
                        + ev['persona'].strip().replace('{code}', self.code))

        self.versions = {
            'system': version_of(os.path.join(sims, 'system.md')),
            'human': version_of(os.path.join(sims, 'human.md')),
        }

        model = args.model
        fakes = make_fakes() if args.fake else {}
        #
        # The worker runs in the copy of the Jupylet files, with the real
        # Skill tool: the CLI loads CLAUDE.md and the skills as in a real
        # session. Its other tools are text blocks. It gets our system
        # prompt, not the Claude Code one: with that one, it called the real
        # Bash tool, which is off here (run ask-miniforge-2).
        #
        self.records = []
        self.turn_actions = 0
        if not args.fake:
            self.start_socket()

        self.worker = Agent('worker', self.worker_prompt(), self.run_dir, model, fakes.get('worker'),
                            cwd=self.code, tools='Skill', append=True, extra=self.mcp_args(), effort=EFFORT)
        self.system = Agent('system', system_prompt, self.run_dir, model, fakes.get('system'), effort=EFFORT)
        self.human = Agent('human', human_prompt, self.run_dir, model, fakes.get('human'), effort=EFFORT)

    def file_list(self):
        """The files of the worker's folder, for the seed: without it, the
        system simulator did not know that header.md exists (run
        full-install-miniconda-1)."""
        paths = set(tracked_files(self.args))
        for d, _, names in os.walk(os.path.join(self.code, '.claude', 'skills')):
            paths.update(os.path.relpath(os.path.join(d, n), self.code) for n in names)
        rows = ['- `%s`' % rel for rel in paths]

        text = ('These files exist in `%s`, the Jupylet code as a person downloads it, with '
                'their real content. Their paths here are relative to that folder. The harness '
                'gives the content when Claude reads one of them, until a command changes it.'
                '\n\n%s' % (self.code, '\n'.join(sorted(rows))))
        #
        # The header of the settings files: a command copies it into a new
        # settings file (`cp -n`), and a seed can name a settings file "with
        # the header". Without its text, the system simulator made up another
        # header (runs open-spaceship-7, open-spaceship-running-2).
        #
        header = os.path.join(self.code, '.claude', 'skills', 'configuring-jupylet', 'header.md')
        if os.path.isfile(header):
            text += ('\n\nThe file `.claude/skills/configuring-jupylet/header.md` has this content. Each '
                     'settings file `claude-config.md` starts with it, also a settings file that the '
                     'computer at the start names "with the header":\n\n```\n%s```' % read(header))

        return text

    def worker_prompt(self):

        loaded = ''
        for name in self.ev.get('loaded_skills', []):
            loaded += ('\n\n## Skill already loaded: %s\n\nThis skill was loaded earlier in this '
                       'session. Follow it.\n\n%s' % (name, self.files.skill(name)))

        text = read(os.path.join(HERE, 'worker.md')) + loaded
        for key, value in (
                ('{home}', self.ev.get('home', '/Users/sam')),
                ('{scratchpad}', self.ev.get('scratchpad', '/private/tmp/claude-501/scratchpad')),
                ('{platform}', self.ev.get('platform', 'macOS on an Apple chip, shell zsh')),
                ('{shell_tool}', self.SHELL_TEXT[self.shell]),
                ('{shell_name}', self.shell),
                ('{platform_note}', self.PLATFORM_NOTE.get(self.shell, '').replace('{code}', self.code))):
            text = text.replace(key, value)

        return text

    SHELL_TEXT = {
        'Bash': '`mcp__computer__Bash`: the Bash tool. It runs shell commands (zsh on\n  macOS).',
        'PowerShell': '`mcp__computer__PowerShell`: the PowerShell tool. It runs PowerShell\n'
                      '  commands on Windows 11.',
    }

    #
    # The worker runs on this Mac, so the CLI gives it a macOS working
    # folder: the copy of the Jupylet code. On a Windows system, that folder
    # stands for the Jupylet folder of the person.
    #
    PLATFORM_NOTE = {
        'PowerShell': '\n- The Jupylet folder of the person is `{code}`. Its path has the form of a\n'
                      '  macOS path, but here it is a folder of this Windows computer. Use the path as\n'
                      '  it is.',
    }

    SERVERS = ('computer', 'browser', 'ccd_window', 'ccd_directory')
    #
    # The names of the tools in the records, as the skills name them. The CLI
    # drops a server named Claude_Browser (it keeps the name for the real
    # pane), so the browser server has another name (run open-spaceship-1).
    #
    RECORD_AS = {'browser': 'Claude_Browser'}

    def mcp_args(self):
        """The worker's tools: our MCP servers, answered by this harness."""
        if self.fake:
            return []

        servers = {name: {'command': sys.executable,
                          'args': [os.path.join(HERE, 'sim_mcp.py')],
                          'env': {'SIM_SOCKET': self.sock_path, 'SIM_SERVER': name, 'SIM_SHELL': self.shell}}
                   for name in self.SERVERS}

        return ['--mcp-config', json.dumps({'mcpServers': servers}),
                '--allowedTools', ' '.join(['Skill'] + ['mcp__' + n for n in self.SERVERS])]

    def start_socket(self):

        self.sock_dir = tempfile.mkdtemp(prefix='jes-')
        self.sock_path = os.path.join(self.sock_dir, 's')
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.sock_path)
        srv.listen(8)
        self.lock = threading.Lock()

        def serve():
            while True:
                conn, _ = srv.accept()
                threading.Thread(target=self.handle, args=(conn,), daemon=True).start()

        threading.Thread(target=serve, daemon=True).start()

    def handle(self, conn):

        data = b''
        while not data.endswith(b'\n'):
            chunk = conn.recv(65536)
            if not chunk:
                break
            data += chunk

        req = json.loads(data.decode())
        server = self.RECORD_AS.get(req['server'], req['server'])
        tool = req['tool'] if server == 'computer' else 'mcp__%s__%s' % (server, req['tool'])

        with self.lock:
            self.turn_actions += 1
            if self.turn_actions > self.ev.get('max_actions', 40) or self.out_of_time():
                out, served = 'Error: the eval reached its limit of actions or time.', False
                self.log('limit', what='actions or seconds')
            else:
                self.progress('%s: %s' % (tool, short_input(req['input'])))
                out = self.run_tool(tool, req['input'])
                served = getattr(self, 'served', False)
                self.served = False
                if served:
                    self.progress('-> (real file, %d lines)' % (out.count('\n') + 1), attach=True)
                else:
                    first = (out.strip().splitlines() or ['(no output)'])[0]
                    if self.live:
                        self.progress('->' if out.strip() else '-> (no output)', out, attach=True)
                    else:
                        self.progress('-> ' + first, attach=True)
            self.records.append((tool, req['input'], out, served))

        conn.sendall((json.dumps({'output': out}) + '\n').encode())
        conn.close()

    def out_of_time(self):

        return time.time() - self.t0 > self.ev.get('max_seconds', 120)

    def progress(self, text, body=None, attach=False):
        """Show the run in the terminal while it goes on.

        `text` is one short line. `body` is the whole text that belongs to
        it, for example a message or a command output: with --live it shows
        too, indented, so that the terminal shows the whole transcript.

        An empty line comes before each new part: a message, a tool call with
        its output, a notification, the costs. `attach` keeps a line with the
        part above it, for example the output of a tool call.
        """
        line = ' '.join(str(text).split())
        if not attach:
            print(flush=True)
        if not self.live:
            print('[%4.0fs] t%d %s' % (time.time() - self.t0, self.turn, line[:150]), flush=True)
            return

        rows = str(body).strip('\n').splitlines() if body else []
        #
        # The first line of the body goes on the line of the label, as in
        # "PERSON: hi", and the other lines go below it, indented.
        #
        if rows:
            line = '%s %s' % (line, rows.pop(0).strip())
        prefix = '[%4.0fs] t%d ' % (time.time() - self.t0, self.turn)
        print(prefix + line, flush=True)
        if rows:
            indent = ' ' * len(prefix)
            more = len(rows) - self.LIVE_ROWS
            for row in rows[:self.LIVE_ROWS]:
                print(indent + row, flush=True)
            if more > 0:
                print(indent + '... (%d more lines)' % more, flush=True)

    LIVE_ROWS = 40

    def watch_worker(self, stop):
        """With --live, show Claude's messages and skill loads while its turn runs."""
        start = self.worker.seen_lines
        while True:
            ended = stop.wait(0.5)
            items, start = self.worker.items_since(start)
            for item in items:
                if item[0] == 'text':
                    self.progress('CLAUDE:', item[1])
                elif item[0] == 'tool_use' and item[2] == 'Skill':
                    self.progress('Skill: %s' % item[3].get('skill', item[3]))
            if ended:
                return

    def costs(self):

        parts = ['%s $%.4f' % (a.name, a.totals['cost_usd']) for a in (self.worker, self.system, self.human)]
        total = sum(a.totals['cost_usd'] for a in (self.worker, self.system, self.human))
        self.progress('-- %s, total $%.4f' % (', '.join(parts), total))

    def log(self, kind, **data):

        data.update(kind=kind, turn=self.turn)
        self.events.append(data)

    def ask_system(self, body):

        reply = self.system.send(body)
        output, state, done = parse_system(reply)
        if state:
            self.state = state

        return output, done

    def run_tool(self, tool, inp):

        if isinstance(inp, str):
            return 'Error: ' + inp

        served = self.files.serve(tool, inp)
        self.served = served is not None
        if self.served:
            return served

        task = self.task_output(tool, inp)
        if task is not None:
            return task

        self.files.changed(tool, inp)
        self.note_page(tool, inp)

        if tool in SHELLS and inp.get('run_in_background'):
            #
            # As the real Bash tool: the output goes into a file, and the
            # notification at the end has no output (run T8). Claude reads
            # the file (task_output).
            #
            tid = 'b' + uuid.uuid4().hex[:8]
            path = (ntpath if tool == 'PowerShell' else os.path).join(self.tasks_dir, tid + '.output')
            self.pending[tid] = inp.get('command', '')
            self.task_files[path] = ''
            self.task_info[tid] = {'path': path, 'description': inp.get('description') or inp.get('command', '')[:60]}
            return ('Command running in background with ID: %s. Output is being written to: %s. You will be '
                    'notified when it completes. To check interim output, use Read on that file path.' % (tid, path))

        cells = self.files.notebook_cells(json.dumps(inp)) if tool in SHELLS else ''
        extra = [x for x in (cells, self.examples_for(tool, inp)) if x]
        output, _ = self.ask_system('<action tool="%s">\n%s\n</action>%s' % (
            tool, json.dumps(inp), ''.join('\n\n' + x for x in extra)))

        return output

    def task_output(self, tool, inp):
        """The output file of a background command, for a Read or a `cat`
        of it, or None."""
        if tool == 'Read':
            path = inp.get('file_path', '')
            if path in self.task_files:
                return numbered(self.task_files[path]) if self.task_files[path] else ''
        elif tool in SHELLS:
            m = re.fullmatch(r'\s*(?:cat|Get-Content)\s+"?([^"]+?)"?\s*', inp.get('command', ''))
            if m and m.group(1) in self.task_files:
                return self.task_files[m.group(1)]

        return None

    def examples_for(self, tool, inp):
        """Real outputs of the same command or tool, from recordings.json."""
        if tool in SHELLS:
            m = TAG.search(inp.get('command', '').rstrip())
            key = m.group(1) if m else None
        else:
            key = 'tool:' + tool if tool.startswith('mcp__') else None
        platform = self.ev.get('os', 'macos')

        return recordings.examples(key, self.recordings, platform) if key else ''

    PAGE_TOOLS = ('mcp__Claude_Browser__preview_start', 'mcp__Claude_Browser__navigate')

    def note_page(self, tool, inp):
        """Note a page that Claude opens with a notebook: the person sees it,
        so the human simulator gets its cells (page_for_human)."""
        text = inp.get('url', '') if tool in self.PAGE_TOOLS else (
            inp.get('command', '') if tool in SHELLS and '/lab/tree/' in inp.get('command', '') else '')
        for name in self.files.notebook_names(text):
            if name not in self.pages_shown and name not in self.pages_new:
                self.pages_new.append(name)

    def page_for_human(self):
        """The cells of each notebook page that opened since the last message
        to the human simulator."""
        out = []
        for name in self.pages_new:
            nb = self.files.notebook_text(name)
            if nb:
                out.append('<page notebook="%s.ipynb">\nA page with this notebook opened on your screen. If '
                           'it asks for a token, it shows the notebook only after you sign in. These are its '
                           'cells, in the percent format of jupytext: each cell starts with a `# %%%%` line, '
                           'and the text cells are comments. The outputs of a cell show below it, after the '
                           'cell runs.\n\n%s\n</page>' % (name, nb))
            self.pages_shown.append(name)
        self.pages_new = []

        return '\n\n'.join(out)

    def check_pending(self):

        if not self.pending:
            return []

        listing = '\n'.join('<task id="%s">%s</task>' % kv for kv in self.pending.items())
        cells = self.files.notebook_cells(listing)
        extra = [x for x in [cells] + [self.examples_for(self.shell, {'command': c}) for c in self.pending.values()]
                 if x]
        _, done = self.ask_system('<pending>\n%s\n</pending>%s' % (listing, ''.join('\n\n' + x for x in extra)))

        notes = []
        for tid, out in done:
            if tid in self.pending:
                del self.pending[tid]
                self.log('notification', id=tid, output=out)
                if self.live:
                    self.progress('NOTIFICATION %s:' % tid, out)
                else:
                    self.progress('NOTIFICATION %s: %s' % (tid, out))
                m = re.match(r'\s*Exit code (\d+)\s*\n?', out)
                code = int(m.group(1)) if m else 0
                info = self.task_info[tid]
                body = re.sub(r'\s*\[exited with code \d+\]\s*$', '', out[m.end():] if m else out).rstrip()
                self.task_files[info['path']] = '%s\n\n[exited with code %d]' % (body, code)
                notes.append('<task-notification>\n<task-id>%s</task-id>\n<output-file>%s</output-file>\n'
                             '<status>completed</status>\n<summary>Background command "%s" completed (exit code %d)'
                             '</summary>\n</task-notification>' % (tid, info['path'], info['description'], code))

        return notes

    def worker_turn(self, message):

        if self.fake:
            return self.worker_turn_blocks(message)

        self.records = []
        self.turn_actions = 0
        if self.live:
            stop = threading.Event()
            watcher = threading.Thread(target=self.watch_worker, args=(stop,), daemon=True)
            watcher.start()
        try:
            self.worker.send(message)
        finally:
            if self.live:
                stop.set()
                watcher.join()

        visible = []
        records = list(self.records)
        for item in self.worker.new_items():
            if item[0] == 'text':
                visible.append(item[1])
                self.log('claude', text=item[1])
                if not self.live:
                    self.progress('CLAUDE: ' + item[1])
            elif item[0] == 'tool_use' and item[2] == 'Skill':
                self.log('action', tool='Skill', input=item[3], real=True)
                if not self.live:
                    self.progress('Skill: %s' % item[3].get('skill', item[3]))
            elif item[0] == 'tool_use' and item[2].startswith('mcp__') and records:
                tool, inp, out, served = records.pop(0)
                self.log('action', tool=tool, input=inp)
                self.log('machine', tool=tool, output=out, served=served)
            elif item[0] == 'tool_use':
                self.log('action', tool=item[2], input=item[3], real=True)

        return visible

    def worker_turn_blocks(self, message):
        """The worker turn of --fake, with tool blocks in the text."""
        visible = []
        actions = 0

        while True:
            reply = self.worker.send(message)
            for tool, inp in self.worker.new_real_tool_calls():
                self.log('action', tool=tool, input=inp, real=True)
            text, tools = parse_worker(reply)

            if text:
                visible.append(text)
                self.log('claude', text=text)
                self.progress('CLAUDE:', text) if self.live else self.progress('CLAUDE: ' + text)

            if not tools:
                return visible

            results = []
            for tool, inp in tools:
                actions += 1
                self.progress('%s: %s' % (tool, short_input(inp)))
                out = self.run_tool(tool, inp)
                if self.live:
                    self.progress('->', out, attach=True)
                else:
                    self.progress('-> ' + (out.strip().splitlines() or ['(no output)'])[0], attach=True)
                self.log('action', tool=tool, input=inp)
                self.log('machine', tool=tool, output=out, served=getattr(self, 'served', False))
                self.served = False
                results.append('<result name="%s">\n%s\n</result>' % (tool, out))

            if actions >= self.ev.get('max_actions', 40):
                self.log('limit', what='actions')
                return visible

            if self.out_of_time():
                self.log('limit', what='seconds')
                return visible

            message = '\n\n'.join(results)

    def run(self):

        t0 = self.t0 = time.time()
        print('\n%s -> %s' % (self.ev['id'], self.run_dir), flush=True)
        print('  persona %s, system %s, skills %s%s' % (
            self.ev.get('persona_name', 'inline'), self.ev.get('system', 'inline seed'),
            self.git.get('commit', self.git.get('source', '?'))[:10],
            ' + changes in %s' % ', '.join(self.git['changed']) if self.git.get('changed') else ''), flush=True)
        reply = parse_human(self.human.send('The session starts. Write your first message to Claude.'))
        stop = 'end'

        while True:
            self.turn += 1
            for d in reply['do']:
                self.log('person_does', text=d)
                self.progress('PERSON DOES: ' + d, d if len(d) > 100 else None)
                self.ask_system('<event>\nThe person %s\n</event>' % d)

            notes = self.check_pending() if (reply['do'] or reply['wait']) else []
            said = '\n\n'.join(reply['say'])
            if said:
                self.log('person', text=said)
                self.progress('PERSON:', said) if self.live else self.progress('PERSON: ' + said)
            elif reply['wait']:
                self.progress('PERSON waits')

            if reply['end']:
                break

            if self.turn > self.ev.get('max_turns', 10):
                stop = 'max_turns'
                break

            if self.out_of_time():
                stop = 'max_seconds'
                break

            message = '\n\n'.join(notes + ([said] if said else []))
            if not message:
                reply = parse_human(self.human.send('(Nothing new appears in the chat.)'))
                continue

            visible = self.worker_turn(message)
            self.costs()
            page = self.page_for_human()
            if page:
                self.log('page', text=page)
                self.progress('PAGE for the person: the cells of %s' % ', '.join(self.pages_shown[-1:]))
            reply = parse_human(self.human.send('\n\n'.join(visible + ([page] if page else []))
                                                or '(Claude wrote nothing.)'))
            if not (reply['say'] or reply['do'] or reply['wait'] or reply['end']):
                reply['say'] = ['(no answer)']

        self.finish(stop, time.time() - t0)

    def finish(self, stop, seconds):

        if self.system.calls:
            self.ask_system('<final/>')

        #
        # Keep the worker's session file in the run folder, then remove what
        # the run left outside it: the copy of the files under test, the
        # agents' folders, and the CLI's session files of the three agents.
        #
        for agent in (self.worker, self.system, self.human):
            path = agent.transcript()
            if path:
                shutil.copy(path, os.path.join(self.run_dir, 'agents', agent.name + '.session.jsonl'))
                os.remove(path)
                if not os.listdir(os.path.dirname(path)):
                    os.rmdir(os.path.dirname(path))
            if agent.own_cwd:
                shutil.rmtree(agent.cwd, ignore_errors=True)

        shutil.rmtree(os.path.dirname(self.code), ignore_errors=True)

        with open(os.path.join(self.run_dir, 'events.jsonl'), 'w', encoding='utf-8') as f:
            for e in self.events:
                f.write(json.dumps(e) + '\n')

        with open(os.path.join(self.run_dir, 'state.md'), 'w', encoding='utf-8') as f:
            f.write(self.state + '\n')

        with open(os.path.join(self.run_dir, 'conversation.md'), 'w', encoding='utf-8') as f:
            f.write(render(self.events))

        timing = {a.name: dict(a.totals, calls=a.calls) for a in (self.worker, self.system, self.human)}
        timing['total_duration_seconds'] = round(seconds, 1)
        timing['total_tokens'] = sum(t.get(k, 0) for t in timing.values() if isinstance(t, dict)
                                     for k in ('input_tokens', 'cache_write_tokens', 'cache_read_tokens',
                                               'output_tokens'))
        timing['total_cost_usd'] = round(sum(t['cost_usd'] for t in timing.values() if isinstance(t, dict)), 4)
        timing['duration_ms'] = int(seconds * 1000)

        meta = {'eval_id': self.ev['id'], 'stop': stop, 'code': self.code,
                'seconds': round(seconds, 1), 'expected_seconds': self.ev.get('expected_seconds'), 'skills': self.source, 'fake': self.fake,
                'simulators': self.versions, 'library': self.ev.get('library', {}), 'git': self.git,
                'turns': self.turn}

        for name, data in (('timing.json', timing), ('run.json', meta)):
            with open(os.path.join(self.run_dir, name), 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

        print('\n%s: %s after %d turns, %.0f s, %d tokens -> %s' % (
            self.ev['id'], stop, self.turn, seconds, timing['total_tokens'], self.run_dir))


def short_input(inp):

    if not isinstance(inp, dict):
        return str(inp)
    s = inp.get('command') or inp.get('file_path') or inp.get('url') or json.dumps(inp)
    if inp.get('run_in_background'):
        s = '(background) ' + s

    return s


def render(events):

    out = []
    for e in events:
        k = e['kind']
        if k == 'person':
            out.append('PERSON: %s\n' % e['text'])
        elif k == 'person_does':
            out.append('PERSON DOES: %s\n' % e['text'])
        elif k == 'claude':
            out.append('CLAUDE: %s\n' % e['text'])
        elif k == 'action':
            out.append('ACTION %s: %s' % (e['tool'], json.dumps(e['input'])))
        elif k == 'machine' and e.get('served'):
            out.append('MACHINE (real file, %d lines): %s\n' % (
                e['output'].count('\n') + 1, e['output'].splitlines()[0][:100]))
        elif k == 'machine':
            out.append('MACHINE: %s\n' % e['output'])
        elif k == 'page':
            out.append('PAGE (what the person sees):\n%s\n' % e['text'])
        elif k == 'notification':
            out.append('NOTIFICATION %s: %s\n' % (e['id'], e['output']))
        elif k == 'limit':
            out.append('(limit of %s reached)\n' % e['what'])

    return '\n'.join(out)


def version_of(path):

    m = re.search(r'^Version: (\S+)', read(path), re.M)

    return m.group(1) if m else '?'


def make_fakes():
    """Scripted stand-ins for --fake: they test the plumbing, not the skills."""

    def worker(agent, text):
        if agent.calls == 1:
            return ('Let me check.\n<tool name="Skill">\n{"skill": "configuring-jupylet"}\n</tool>\n'
                    '<tool name="Bash">\n{"command": "ls ~/.jupylet", "description": "List settings"}\n</tool>\n'
                    '<tool name="Bash">\n{"command": "sleep 1", "run_in_background": true}\n</tool>')
        if '<task-notification>' in text:
            return 'The wait ended.'
        if agent.calls == 2:
            return 'Done. Anything else?'
        return 'OK.'

    def system(agent, text):
        if '<pending>' in text:
            return ''.join('<done id="%s">\nfinished\n</done>' % t for t in re.findall(r'<task id="(\w+)"', text))
        return '<output>\nclaude-config.md\n</output>\n<state>\nfake state %d\n</state>' % agent.calls

    def human(agent, text):
        if agent.calls == 1:
            return '<say>please remember to hide the chat list</say>'
        if agent.calls == 2:
            return '<do>clicks the canvas</do>'
        return '<end/>'

    return {'worker': worker, 'system': system, 'human': human}


def main():

    p = argparse.ArgumentParser()
    p.add_argument('evals')
    p.add_argument('eval_id')
    p.add_argument('run_dir')
    p.add_argument('--skills')
    p.add_argument('--ref')
    p.add_argument('--system')
    p.add_argument('--fake', action='store_true')
    p.add_argument('--live', action='store_true')
    p.add_argument('--model', default=MODEL)
    args = p.parse_args()

    evals = library.load_evals(args.evals)
    ev = next((e for e in evals if str(e['id']) == args.eval_id), None)
    if not ev:
        sys.exit('no eval %r in %s' % (args.eval_id, args.evals))
    if args.system:
        ev = dict(ev, system=args.system)
    try:
        ev = library.expand(ev)
    except ValueError as e:
        sys.exit(str(e))

    Session(ev, args).run()


if __name__ == '__main__':
    main()
