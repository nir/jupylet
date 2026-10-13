"""Grade one run of run_eval.py against the checks of its eval.

Usage:
    grade.py <evals.json> <run dir> [--current]

--current: grade with the checks of the evals file now, not with the
  eval.json that the run saved. Use it after a change of the checks.

Writes grading.json into the run dir, in the skill-creator form: one
expectation for each check, with text, passed and evidence. Checks of type
"judge" need judgment; this script marks them as not graded (passed: null).

Check types, in each check of the eval:
    action      some action of `tool` has an input that matches `match`
    no_action   no action of `tool` has an input that matches `match`
    before      the first action that matches `first` comes before the first
                action that matches `then`, and both exist
    not_in_turn no action that matches `tool`/`match` happens in turn `turn`
    state       the final state matches `match`
    text        some message of Claude to the person matches `match`
    no_text     no message of Claude to the person matches `match`
    questions   each message of Claude to the person has at most `max`
                question marks
    judge       graded by a person or a Haiku grader

Automatic checks, for each eval: every agent used Haiku, every Jupylet file
that Claude read was served by the harness, and every shell command of
Claude has the comment `# jupylet: {group}/{name}` of the skills, with a
tag that is in the tags.json of the run: the tags of the files under test
(the eval can allow other commands with "allowed_commands").

A check can have "depends": what a simulator must do for the check to apply,
for example "the person says yes to the Terminal change". Only the worker
(Claude) is graded. So when such a check fails, it is not a failure: it is
marked for review (passed: null, review: true). The review finds out
whether the simulator did what the check needs. If it did not, the
simulator deviated: note it, and improve the simulator. If it did, the
check is a real failure of Claude.
"""

import json
import os
import re
import sys

import library


def actions(events, tool=None, match=None):

    found = []
    for e in events:
        if e['kind'] != 'action':
            continue
        if tool and e['tool'] != tool:
            continue
        text = json.dumps(e['input']) if not isinstance(e['input'], str) else e['input']
        if match and not re.search(match, text):
            continue
        found.append(e)

    return found


def describe(e):

    inp = e['input']
    s = inp.get('command') or inp.get('file_path') or inp.get('skill') or json.dumps(inp) \
        if isinstance(inp, dict) else inp

    return 'turn %d, %s: %s' % (e['turn'], e['tool'], str(s)[:160])


def index(events, e):

    return events.index(e)


def grade_check(c, events, state):

    kind = c['type']

    if kind == 'action':
        found = actions(events, c.get('tool'), c.get('match'))
        return bool(found), describe(found[0]) if found else 'no such action'

    if kind == 'no_action':
        found = actions(events, c.get('tool'), c.get('match'))
        return not found, describe(found[0]) if found else 'no such action'

    if kind == 'before':
        a = actions(events, c['first'].get('tool'), c['first'].get('match'))
        b = actions(events, c['then'].get('tool'), c['then'].get('match'))
        if not a or not b:
            return False, 'missing: %s' % ('first' if not a else 'then')
        ok = index(events, a[0]) < index(events, b[0])
        return ok, '%s / %s' % (describe(a[0]), describe(b[0]))

    if kind == 'not_in_turn':
        found = [e for e in actions(events, c.get('tool'), c.get('match')) if e['turn'] == c['turn']]
        return not found, describe(found[0]) if found else 'none in turn %d' % c['turn']

    if kind == 'state':
        m = re.search(c['match'], state, re.M)
        return bool(m), m.group(0) if m else 'no match in the final state'

    if kind == 'no_text':
        for e in events:
            if e['kind'] == 'claude' and re.search(c['match'], e['text']):
                return False, 'turn %d: %s' % (e['turn'], e['text'][:200])
        return True, 'no such message'

    if kind == 'text':
        for e in events:
            if e['kind'] == 'claude' and re.search(c['match'], e['text']):
                return True, 'turn %d: %s' % (e['turn'], e['text'][:200])
        return False, 'no such message'

    if kind == 'questions':
        by_turn = {}
        for e in events:
            if e['kind'] == 'claude':
                by_turn[e['turn']] = by_turn.get(e['turn'], '') + '\n' + e['text']
        for turn, text in sorted(by_turn.items()):
            n = text.count('?')
            if n > c.get('max', 1):
                return False, 'turn %d has %d question marks: %s' % (turn, n, text.strip()[:200])
        return True, '%d turns checked' % len(by_turn)

    if kind == 'judge':
        return None, 'needs judgment'

    return False, 'unknown check type %r' % kind


def models_used(run_dir):
    """A check that every call of the three agents used Haiku 5.5."""
    models = set()
    for name in ('worker', 'system', 'human'):
        p = os.path.join(run_dir, 'agents', name + '.jsonl')
        if os.path.exists(p):
            for line in open(p):
                models.update(json.loads(line)['meta'].get('modelUsage', {}).keys())

    ok = models == {'claude-haiku-5-5'}

    return {'text': 'Every agent call used claude-haiku-5-5.', 'passed': ok,
            'evidence': ', '.join(sorted(models)) or 'no model calls (fake run?)'}


TAG = re.compile(r'#\s*jupylet:\s*([\w-]+/[\w-]+)\s*$')


def commands_tagged(events, ev, known=None):
    """A check that each shell command of Claude comes from the skills.

    Each command in the skills ends with a comment `# jupylet: {group}/{name}`.
    A command without it is a command that Claude wrote itself, for example
    several commands joined with `;`, or a command with `| head` added. An
    eval can allow other commands with `allowed_commands`, a list of
    patterns.
    """
    allowed = [re.compile(a) for a in ev.get('allowed_commands', [])]
    bad = []
    for e in actions(events, 'Bash'):
        cmd = e['input'].get('command', '') if isinstance(e['input'], dict) else str(e['input'])
        if any(a.search(cmd) for a in allowed):
            continue
        m = TAG.search(cmd.rstrip())
        if not m:
            bad.append('no tag: ' + describe(e))
        elif known is not None and m.group(1) not in known:
            bad.append('tag not in the skills (%s): %s' % (m.group(1), describe(e)))

    return {'text': 'Each shell command of Claude comes from the skills, with its comment # jupylet: (finding 5).',
            'passed': not bad, 'evidence': '; '.join(bad) or 'all %d tagged' % len(actions(events, 'Bash'))}


def skills_seen(run_dir):
    """A check that the worker saw only the skills of the files under test.

    In full-install-miniconda-1, it also saw Nir's own skills and plugins.
    """
    path = os.path.join(run_dir, 'agents', 'worker.session.jsonl')
    if not os.path.isfile(path):
        return {'text': 'The worker saw no personal skills and no plugin skills.', 'passed': None, 'evidence': 'no session file'}

    listing = ''
    for line in open(path):
        e = json.loads(line)
        a = e.get('attachment') or {}
        if a.get('type') == 'skill_listing':
            listing = a.get('content', '')
            break

    names = re.findall(r'^- ([\w:.-]+):', listing, re.M)
    #
    # Built-in skills of Claude Code, for example simplify, are in every
    # real session too. Not allowed: plugin skills (with a ':') and the
    # personal skills in ~/.claude/skills.
    #
    personal = set(os.listdir(os.path.expanduser('~/.claude/skills'))) if os.path.isdir(
        os.path.expanduser('~/.claude/skills')) else set()
    other = [n for n in names if ':' in n or n in personal]

    return {'text': 'The worker saw no personal skills and no plugin skills.', 'passed': not other,
            'evidence': ', '.join(other) or 'none of the personal or plugin skills'}


def files_served(events):
    """A check that the harness served every read of a Jupylet file itself.

    If it does not, the system simulator answers, and makes up the file.
    """
    bad = []
    for i, e in enumerate(events):
        if e['kind'] != 'action' or e['tool'] != 'Read' or not isinstance(e['input'], dict):
            continue
        path = e['input'].get('file_path', '')
        if not re.search(r'/jupylet/(\.claude/skills/|CLAUDE\.md$|CLAUDE_SETUP\.md$|.*\.py$)', path):
            continue
        m = next((x for x in events[i + 1:] if x['kind'] == 'machine'), None)
        if not (m and m.get('served')):
            bad.append(path)

    return {'text': 'The harness served every Jupylet file that Claude read (none made up).',
            'passed': not bad, 'evidence': ', '.join(bad) or 'all served'}


def main():

    args = [a for a in sys.argv[1:] if a != '--current']
    current = '--current' in sys.argv
    evals_path, run_dir = args[:2]
    run = json.load(open(os.path.join(run_dir, 'run.json')))
    #
    # eval.json is the eval as it ran, with the checks of the persona and the
    # system. Older runs have none: build it from the evals file.
    #
    saved = os.path.join(run_dir, 'eval.json')
    if os.path.isfile(saved) and not current:
        ev = json.load(open(saved))
    else:
        evals = library.load_evals(evals_path)
        ev = library.expand(next(e for e in evals if str(e['id']) == str(run['eval_id'])))
    events = [json.loads(line) for line in open(os.path.join(run_dir, 'events.jsonl'))]
    state = open(os.path.join(run_dir, 'state.md')).read()

    tags_path = os.path.join(run_dir, 'tags.json')
    known = json.load(open(tags_path)) if os.path.isfile(tags_path) else None
    expectations = [models_used(run_dir), skills_seen(run_dir), files_served(events),
                    commands_tagged(events, ev, known)]
    for c in ev['checks']:
        passed, evidence = grade_check(c, events, state)
        x = {'text': c['text'], 'passed': passed, 'evidence': evidence}
        if passed is False and c.get('depends'):
            x.update(passed=None, review=True,
                     evidence='applies only if %s. Review the simulator. Evidence: %s' % (c['depends'], evidence))
        expectations.append(x)

    graded = [x for x in expectations if x['passed'] is not None]
    n_pass = sum(1 for x in graded if x['passed'])
    summary = {'passed': n_pass, 'failed': len(graded) - n_pass, 'total': len(graded),
               'pass_rate': round(n_pass / len(graded), 3) if graded else 0.0,
               'not_graded': len(expectations) - len(graded),
               'to_review': sum(1 for x in expectations if x.get('review'))}

    with open(os.path.join(run_dir, 'grading.json'), 'w') as f:
        json.dump({'expectations': expectations, 'summary': summary}, f, indent=2)

    for x in expectations:
        mark = 'REVW' if x.get('review') else {True: 'PASS', False: 'FAIL', None: '----'}[x['passed']]
        print('\n%s  %s\n      %s' % (mark, x['text'], x['evidence']))
    print('\n%d/%d passed, %d to review, %d need judgment' % (
        n_pass, len(graded), summary['to_review'], summary['not_graded'] - summary['to_review']))


if __name__ == '__main__':
    main()
