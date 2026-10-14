"""Real outputs of commands and tools, from exported Claude sessions.

The system simulator gets them as examples (few-shot): when Claude runs a
command, the harness adds the real outputs of the same command from real
sessions to that one request. The simulator then gives an output of the same
form, with the values of the current state.

Usage:
    recordings.py add <export.zip or transcript.jsonl> <source name>
    recordings.py list

`add` reads a session and adds what it finds to simulators/recordings.json:

- each Bash command with a tag `# jupylet: {group}/{name}`, under its tag;
- each call of an app tool (`mcp__Claude_Browser__…`, `mcp__ccd_…`), under
  `tool:{name}`;
- the output of a background command, which the real app writes into a file:
  when Claude reads that file, the output goes under the tag of the command.

The home folder of the session becomes `{home}`, and the session's own
folders become `{scratchpad}`. At most MAX outputs are kept for each key,
and long outputs are shortened.
"""

import json
import os
import re
import sys
import tempfile
import zipfile


HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, 'simulators', 'recordings.json')
TAG = re.compile(r'#\s*jupylet:\s*([\w-]+/[\w-]+)\s*$')
MAX = 3
HEAD, TAIL = 40, 10


def load():

    if not os.path.isfile(STORE):
        return {}
    with open(STORE, encoding='utf-8') as f:
        return json.load(f)


def save(data):

    with open(STORE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write('\n')


def transcript_lines(path):
    """The lines of the session's transcript, from a zip or a jsonl file."""
    if not path.endswith('.zip'):
        with open(path, encoding='utf-8') as f:
            return f.read().splitlines()

    with zipfile.ZipFile(path) as z:
        name = 'transcript.jsonl' if 'transcript.jsonl' in z.namelist() else next(
            n for n in z.namelist() if n.endswith('.jsonl') and '/' not in n)
        return z.read(name).decode('utf-8').splitlines()


def text_of(content):

    if isinstance(content, list):
        return '\n'.join(x.get('text', '') for x in content if isinstance(x, dict) and x.get('type') == 'text')

    return str(content or '')


def shorten(text):

    rows = text.splitlines()
    if len(rows) <= HEAD + TAIL:
        return text

    return '\n'.join(rows[:HEAD] + ['... (%d lines)' % (len(rows) - HEAD - TAIL)] + rows[-TAIL:])


def general(text, home, scratch):
    """Replace the folders of this session with blanks."""
    for folder in scratch:
        text = text.replace(folder, '{scratchpad}')
    if home:
        text = text.replace(home, '{home}')

    return text


def find_home(lines):

    for line in lines:
        m = re.search(r'"(/Users/[^/"]+)/', line)
        if m:
            return m.group(1)

    return None


def collect(lines):
    """(key, command or input, output, is_error) for each recorded call."""
    uses, background, found = {}, {}, []

    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        content = (e.get('message') or {}).get('content')
        if not isinstance(content, list):
            continue

        for b in content:
            if b.get('type') == 'tool_use':
                uses[b['id']] = (b['name'], b.get('input') or {})
                continue
            if b.get('type') != 'tool_result' or b.get('tool_use_id') not in uses:
                continue

            name, inp = uses[b['tool_use_id']]
            out = text_of(b.get('content'))
            read = re.search(r'/tasks/(\w+)\.output', inp.get('file_path', '')) if name == 'Read' else None
            if read and read.group(1) in background:
                tag_name, bg_cmd = background[read.group(1)]
                out = re.sub(r'^\s*\d+\t', '', out, flags=re.M)
                out = re.sub(r'\s*\[exited with code \d+\]\s*$', '', out)
                found.append((tag_name, bg_cmd + '   (run in the background; its output file)', out,
                              bool(b.get('is_error'))))
                continue
            if name == 'Bash':
                cmd = inp.get('command', '')
                tag = TAG.search(cmd.rstrip())
                m = re.match(r'Command running in background with ID: (\w+)', out)
                if tag and m:
                    background[m.group(1)] = (tag.group(1), cmd)
                    continue
                read = re.search(r'/tasks/(\w+)\.output', cmd)
                if read and read.group(1) in background:
                    tag_name, bg_cmd = background[read.group(1)]
                    #
                    # The harness adds `[exited with code N]` to the file itself.
                    #
                    out = re.sub(r'\s*\[exited with code \d+\]\s*$', '', out)
                    found.append((tag_name, bg_cmd + '   (run in the background; its output file)', out,
                                  bool(b.get('is_error'))))
                    continue
                if tag:
                    found.append((tag.group(1), cmd, out, bool(b.get('is_error'))))
            elif name.startswith(('mcp__Claude_Browser__', 'mcp__ccd_')):
                found.append(('tool:' + name, json.dumps(inp, ensure_ascii=False), out, bool(b.get('is_error'))))

    return found


def add(path, source):

    lines = transcript_lines(path)
    home = find_home(lines)
    #
    # The scratchpad of a session is /private/tmp/claude-<uid>/<project>/<session id>/scratchpad.
    # The project part can be long, for example a scratch workspace of the app.
    #
    joined = '\n'.join(lines)
    scratch = sorted(set(re.findall(r'(/private/tmp/claude-\d+/[^"\s]+?/[0-9a-f-]{36}/scratchpad)', joined)),
                     key=len, reverse=True)
    scratch += sorted(set(re.findall(r'(/private/tmp/claude-\d+/[^"\s]+?/[0-9a-f-]{36})(?=/)', joined)),
                      key=len, reverse=True)

    data = load()
    added = 0
    for key, cmd, out, err in collect(lines):
        #
        # An error can be in the output of a command that ran, for example
        # `error Error executing tool: Timeout waiting for result`.
        #
        err = err or bool(re.match(r'\s*(error\b|Exit code [1-9]|ERROR)', out))
        item = {'source': source, 'input': general(cmd, home, scratch),
                'output': shorten(general(out, home, scratch)), 'error': err}
        items = data.setdefault(key, [])
        if any(x['output'] == item['output'] for x in items) or len(items) >= MAX:
            continue
        items.append(item)
        added += 1

    save(data)
    print('%s: %d outputs added, %d keys in %s' % (source, added, len(data), STORE))


def examples(key, data=None):
    """The recorded outputs of a key, as a block for the system simulator, or ''."""
    items = (data if data is not None else load()).get(key, [])
    #
    # An error alone would be copied as the normal output: show errors only
    # next to a normal output (the first run of run-all timed out in T8).
    #
    if not any(not x.get('error') for x in items):
        return ''

    rows = ['<example source="%s"%s>\n%s\n---\n%s\n</example>' % (
        x['source'], ' error="yes"' if x.get('error') else '', x['input'], x['output']) for x in items]

    return ('<examples>\nReal outputs of this %s, from real sessions. They show the form of the output. '
            'Take every value from the current state of this computer, not from the examples: paths, '
            'versions, numbers, and which files exist. In the examples, {home} is the home folder and '
            '{scratchpad} the scratchpad of that session. An example marked error="yes" shows the form of an '
            'error. Give an error only when the current state causes one.\n\n%s\n</examples>' % (
                'tool' if key.startswith('tool:') else 'command', '\n\n'.join(rows)))


def main():

    if sys.argv[1:2] == ['add'] and len(sys.argv) == 4:
        add(sys.argv[2], sys.argv[3])
    elif sys.argv[1:2] == ['list']:
        for key, items in sorted(load().items()):
            print('%-40s %s' % (key, ', '.join(x['source'] for x in items)))
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
