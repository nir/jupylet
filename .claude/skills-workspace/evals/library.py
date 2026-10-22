"""Build an eval from a scenario and the persona and system library.

simulators/LIBRARY.md describes the library. A scenario names a persona and
a system by file name, and gives its request:

    {"id": ..., "persona": "careful-adult", "system": "mac-miniconda-terminal",
     "user": "sam", "home": "/Users/sam", "request": "...", "checks": [...]}

expand() turns it into the form of smoke.json: "persona" and "seed" become
the texts for the simulators, and the checks of the persona and the system
are added to the checks of the scenario. An eval with inline "persona" and
"seed" texts, as in smoke.json, stays as it is.
"""

import hashlib
import json
import os
import re


HERE = os.path.dirname(os.path.abspath(__file__))
LIBRARY = os.path.join(HERE, 'simulators')
SCRATCHPAD = '/private/tmp/claude-501/scratchpad'
#
# A system says its platform in its base file, with a line `Platform: …`.
# The platform gives the defaults of the home folder, the scratchpad and the
# shell tool, as on a real computer of that kind (run W1 for Windows).
#
PLATFORMS = {
    'macos': {
        'home': '/Users/{user}',
        'scratchpad': SCRATCHPAD,
        'shell': 'Bash',
        'platform': 'macOS on an Apple chip, shell zsh',
    },
    'windows': {
        'home': 'C:\\Users\\{user}',
        'scratchpad': 'C:\\Users\\{user}\\AppData\\Local\\Temp\\claude\\C--Users-{user}-jupylet\\'
                      '5f0c2a9e-3b71-4d8a-9c6e-1a2b3c4d5e6f\\scratchpad',
        'shell': 'PowerShell',
        'platform': 'Windows 11 on an Intel or AMD processor, shell PowerShell',
    },
}


def load_evals(path):
    """The evals of a file: a list under "evals", or one scenario."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    return data['evals'] if 'evals' in data else [data]


def header(text):
    """The lines `Extends: …` and `Platform: …` at the top of a library
    file, as a dict, and the text after them."""
    found = {}
    while True:
        m = re.match(r'(Extends|Platform): (\S+)\s*\n', text)
        if not m:
            return found, text
        found[m.group(1).lower()] = m.group(2)
        text = text[m.end():]


def split(path):
    """The simulator text of a library file, its checks, and its Extends."""
    with open(path, encoding='utf-8') as f:
        text = f.read()

    head, text = header(text)
    extends = head.get('extends')

    checks = []
    head, sep, tail = text.partition('\n## Checks\n')
    if sep:
        m = re.search(r'```json\n(.*?)```', tail, re.S)
        if not m:
            raise ValueError('%s: "## Checks" has no json block' % path)
        checks = json.loads(m.group(1))

    return head.strip(), checks, extends


def library_file(kind, name):

    path = os.path.join(LIBRARY, kind, name + '.md')
    if not os.path.isfile(path):
        raise ValueError('no %s %r in %s' % (kind[:-1], name, os.path.dirname(path)))

    return path


def digest(path):

    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()[:10]


def system_platform(name):
    """The platform of a system: the `Platform:` of its base file, or macos."""
    for path in system_parts(name):
        with open(path, encoding='utf-8') as f:
            platform = header(f.read())[0].get('platform')
        if platform:
            if platform not in PLATFORMS:
                raise ValueError('%s: unknown platform %r' % (path, platform))
            return platform

    return 'macos'


def system_parts(name, seen=()):
    """The files of a system, base first, following Extends."""
    if name in seen:
        raise ValueError('Extends loop at system %r' % name)

    path = library_file('systems', name)
    _, _, extends = split(path)

    return (system_parts(extends, seen + (name,)) if extends else []) + [path]


def fill(text, blanks):

    for key, value in blanks.items():
        text = text.replace(key, value)

    return text


def fill_checks(checks, blanks, source):

    out = []
    for c in checks:
        c = json.loads(fill(json.dumps(c), blanks))
        c.setdefault('source', source)
        out.append(c)

    return out


def expand(ev):
    """The eval with the library files resolved. Adds "library" with the
    name and content digest of each file used, for run.json."""
    ev = dict(ev)
    if 'system' not in ev and 'persona' in ev and not os.path.isfile(
            os.path.join(LIBRARY, 'personas', ev['persona'] + '.md')):
        return ev

    #
    # The platform of the system gives the home folder, the scratchpad and
    # the shell tool, unless the eval names them: so the same scenario runs
    # on a Mac and on a Windows system.
    #
    os_name = system_platform(ev['system']) if 'system' in ev else 'macos'
    defaults = {k: v.replace('{user}', ev.get('user', 'sam')) for k, v in PLATFORMS[os_name].items()}
    ev['os'] = os_name
    home = ev.get('home') or defaults['home']
    ev['home'] = home
    ev.setdefault('scratchpad', defaults['scratchpad'])
    ev.setdefault('shell', defaults['shell'])
    ev.setdefault('platform', defaults['platform'])
    blanks = {'{user}': ev.get('user', re.split(r'[\\/]', home)[-1]), '{home}': home,
              '{scratchpad}': ev['scratchpad']}
    used = {}
    checks = fill_checks(ev.get('checks', []), blanks, 'scenario')

    if 'persona' in ev:
        ev['persona_name'] = ev['persona']
        path = library_file('personas', ev['persona'])
        text, extra, _ = split(path)
        used['persona/' + ev['persona']] = digest(path)
        if ev.get('request'):
            text += '\n\n## This session\n\n' + ev['request'].strip()
        ev['persona'] = fill(text, blanks)
        checks += fill_checks(extra, blanks, 'persona ' + os.path.basename(path)[:-3])

    if 'system' in ev:
        texts = []
        for path in system_parts(ev['system']):
            text, extra, _ = split(path)
            name = os.path.basename(path)[:-3]
            used['system/' + name] = digest(path)
            if texts:
                texts.append('The part below adds to the computer described above.')
            texts.append(text)
            checks += fill_checks(extra, blanks, 'system ' + name)
        if ev.get('seed'):
            texts.append('The part below adds to the computer described above.')
            texts.append(ev['seed'].strip())
        ev['seed'] = fill('\n\n'.join(texts), blanks)

    ev['checks'] = checks
    ev['library'] = used

    return ev
