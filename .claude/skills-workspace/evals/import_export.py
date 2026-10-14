"""Turn an exported real Claude session into a run folder, for grade.py.

Usage:
    import_export.py <export.zip or transcript.jsonl> <scenario.json> <run name>

Then grade it as a simulated run:
    grade.py <scenario.json> runs/<run name>

The run folder has the same files as a run of run_eval.py, and run.json
says "real": true. For a real run, grade.py does not grade what only a
simulated run has: the final state, the model of the three agents, and the
files that the harness served.

The checks come from the scenario, with the home folder of the session in
place of the home folder of the persona. Some checks of a scenario can be
about its system only, for example a Miniconda that the real computer does
not have: read such a FAIL as "does not apply".

The actions before Claude has the Jupylet instructions are a preamble, with
the kind "preamble": the checks skip them.
"""

import json
import os
import re
import shutil
import sys

import library
import recordings
import run_eval


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
REMINDER = re.compile(r'<system-reminder>.*?</system-reminder>', re.S)


def person_text(content):
    """What the person typed, without the reminders of the app, or ''."""
    if isinstance(content, str):
        return REMINDER.sub('', content).strip()

    text = '\n'.join(b.get('text', '') for b in content if isinstance(b, dict) and b.get('type') == 'text')

    return REMINDER.sub('', text).strip()


def events_of(lines):
    """The events of the session, in the form of events.jsonl of run_eval.py."""
    events, turn = [], 0

    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        msg = e.get('message') or {}
        content = msg.get('content')

        if e.get('type') == 'user' and not e.get('isMeta'):
            said = person_text(content) if not (isinstance(content, list) and any(
                isinstance(b, dict) and b.get('type') == 'tool_result' for b in content)) else ''
            if said:
                turn += 1
                events.append({'text': said, 'kind': 'person', 'turn': turn})
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get('type') == 'tool_result':
                        events.append({'tool_use_id': b.get('tool_use_id'), 'output': recordings.text_of(b.get('content')),
                                       'kind': 'machine', 'turn': turn})

        if e.get('type') == 'assistant' and isinstance(content, list):
            for b in content:
                if b.get('type') == 'text' and b.get('text', '').strip():
                    events.append({'text': b['text'], 'kind': 'claude', 'turn': turn})
                elif b.get('type') == 'tool_use':
                    events.append({'tool': b['name'], 'input': b.get('input') or {}, 'kind': 'action', 'turn': turn})

    #
    # Before Claude has the Jupylet instructions, nothing it does can come
    # from them, for example the download of CLAUDE_SETUP.md from the address
    # that the person gave (T8). The checks skip this preamble: it ends at
    # the first command with a tag, or the first skill that Claude loads.
    #
    for e in events:
        if e['kind'] == 'action' and (e['tool'] == 'Skill' or run_eval.TAG.search(e['input'].get('command', ''))):
            break
        if e['kind'] == 'action':
            e['kind'] = 'preamble'

    return events


def models_of(lines):

    models = set()
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get('type') == 'assistant' and (e.get('message') or {}).get('model'):
            models.add(e['message']['model'])

    return sorted(models - {'<synthetic>'})


def files_under_test():
    """The command tags of the skills and the CLAUDE files of the repo."""
    tags = run_eval.command_tags(os.path.join(REPO, '.claude', 'skills'))
    for name in ('CLAUDE.md', 'CLAUDE_SETUP.md'):
        for line in run_eval.read(os.path.join(REPO, name)).splitlines():
            m = run_eval.TAG.search(line)
            if m:
                tags.setdefault(m.group(1), []).append({'file': name, 'line': line.strip().strip('`').strip()})

    return tags


def main():

    if len(sys.argv) != 4:
        sys.exit(__doc__)
    source, scenario, name = sys.argv[1:]

    lines = recordings.transcript_lines(source)
    home = recordings.find_home(lines)
    run_dir = os.path.join(HERE, 'runs', name)
    if os.path.exists(run_dir):
        sys.exit('%s exists already' % run_dir)
    os.makedirs(os.path.join(run_dir, 'agents'))

    ev = library.load_evals(scenario)[0]
    ev = dict(ev, home=home, user=os.path.basename(home))
    ev = library.expand(ev)

    events = events_of(lines)
    with open(os.path.join(run_dir, 'events.jsonl'), 'w', encoding='utf-8') as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    with open(os.path.join(run_dir, 'eval.json'), 'w', encoding='utf-8') as f:
        json.dump(ev, f, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, 'tags.json'), 'w', encoding='utf-8') as f:
        json.dump(files_under_test(), f, indent=2, ensure_ascii=False)
    #
    # The transcript has the list of skills that Claude saw, in the form
    # that grade.py reads from the session file of the worker.
    #
    with open(os.path.join(run_dir, 'agents', 'worker.session.jsonl'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    run = {'eval_id': ev['id'], 'real': True, 'source': os.path.abspath(source), 'models': models_of(lines),
           'home': home, 'turns': max([e['turn'] for e in events] or [0])}
    with open(os.path.join(run_dir, 'run.json'), 'w', encoding='utf-8') as f:
        json.dump(run, f, indent=2)

    print('%s: %d events, %d turns, models %s' % (run_dir, len(events), run['turns'], ', '.join(run['models'])))


if __name__ == '__main__':
    main()
