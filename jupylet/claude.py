"""
    jupylet/claude.py

    Copyright (c) 2022, Nir Aides - nir.8bit@gmail.com

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

USAGE = """
Helpers for Claude Code sessions that work with a person in a live Jupyter
notebook (see CLAUDE.md). Standard library only, so it also runs by file path.

    python -m jupylet.claude find-env [--all] <version>
    python -m jupylet.claude wait <port> <token>
    python -m jupylet.claude attach <port> <token> <notebook path>
    python -m jupylet.claude kernel <port> <token> <notebook path>
    python -m jupylet.claude tools <port> <token>
    python -m jupylet.claude call <port> <token> <tool> ['<json arguments>']
    python -m jupylet.claude wait-open <port> <token> <notebook path> <seconds>
    python -m jupylet.claude watch <port> <token> <notebook path> <seconds> [<since>]
    python -m jupylet.claude wait-change <port> <token> <notebook path> <seconds>
    python -m jupylet.claude replace-kernel <port> <token> <notebook path>
    python -m jupylet.claude run-cell <port> <token> <cell index> <start of its source>
    python -m jupylet.claude shutdown <port> <token>
    python -m jupylet.claude cleanup [--yes]
    python -m jupylet.claude running [<folder>]
    python -m jupylet.claude nbmodel-off

`find-env` is the one command meant to run from any Python, with or
without jupylet, such as Miniforge's `base` or a plain python3 (run this
file by path: it never imports jupylet itself, on purpose). `running` too. It prints one environment per line:
its path, its kind (conda or venv), and where its jupylet comes from ("this
folder", or a version and its source). With --all it also lists
environments with any other version of jupylet. Every other command needs an
environment that actually has jupylet installed.
"""


import glob
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from concurrent.futures import ThreadPoolExecutor


def _python_of(env):
    """The python executable inside an environment folder, or None."""
    for rel in ('python.exe', os.path.join('Scripts', 'python.exe'), os.path.join('bin', 'python')):
        p = os.path.join(env, rel)
        if os.path.exists(p):
            return p

    return None


# Run by each candidate environment's python: jupylet's version, and where it
# was installed from, if it was installed from a folder (pip records that in
# direct_url.json).
_PROBE = """
import importlib.metadata as m, json
d = m.distribution('jupylet')
u = json.loads(d.read_text('direct_url.json') or '{}')
print(json.dumps([d.version, u.get('url', ''), u.get('dir_info', {}).get('editable', False)]))
"""


def _candidates(root, folder):
    """Folders that may be Python environments, each listed once.

    Miniforge's own folder and its envs, every environment any conda install
    on this computer recorded in ~/.conda/environments.txt, and a venv in the
    jupylet folder itself (.venv or venv).
    """
    paths = [root] + sorted(glob.glob(os.path.join(root, 'envs', '*')))

    try:
        with open(os.path.expanduser(os.path.join('~', '.conda', 'environments.txt'))) as f:
            paths += [line.strip() for line in f if line.strip()]
    except OSError:
        pass

    paths += [os.path.join(folder, '.venv'), os.path.join(folder, 'venv')]

    seen, unique = set(), []

    for path in paths:
        real = os.path.realpath(path)

        if real not in seen and os.path.isdir(real):
            seen.add(real)
            unique.append(path)

    return unique


def _source_folder(url):
    """The folder a file:// url points to, or None."""
    if not url.startswith('file:'):
        return None

    import urllib.parse
    return os.path.realpath(urllib.request.url2pathname(urllib.parse.urlparse(url).path))


def find_env(version, root=None, folder=None, any_version=False):
    """Every environment that can run the jupylet in folder, best first.

    An environment qualifies when its jupylet is installed (editable) from
    folder itself, whatever version it reports, or when it has exactly this
    version from anywhere else, or, with any_version, any jupylet at all.
    They come in that order; within each kind, the most recently changed
    environment comes first.

    root is Miniforge's own folder, ~/miniforge3 by default. folder is the
    jupylet folder this file is in, by default. Only the standard library
    and a subprocess call per candidate, so it runs by file path under any
    Python (Miniforge's base Python, or a plain python3), the same on macOS
    and Windows.

    Returns a list of (env path, python path, kind, source) tuples: kind is
    'conda' or 'venv', and source is 'this folder', or the version and where
    it came from.
    """
    if root is None:
        root = os.path.expanduser(os.path.join('~', 'miniforge3'))

    if folder is None:
        folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    folder = os.path.realpath(folder)
    matches = []

    for env in _candidates(root, folder):
        python = _python_of(env)

        if not python:
            continue

        try:
            out = subprocess.run(
                [python, '-c', _PROBE],
                capture_output=True, text=True, timeout=20,
                # Pinned so a folder named "jupylet" in *our own* cwd (this
                # file's own repo, for instance) can never shadow the real
                # package for the subprocess: import and importlib.metadata
                # both consult the current directory first.
                cwd=env,
            )
            found, url, editable = json.loads(out.stdout.strip().splitlines()[-1])
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            continue

        source = _source_folder(url)
        kind = 'venv' if os.path.exists(os.path.join(env, 'pyvenv.cfg')) else 'conda'

        if editable and source == folder:
            matches.append((0, -os.path.getmtime(env), env, python, kind, 'this folder'))

        elif found == version or any_version:
            where = source or 'a package index'
            rank = 1 if found == version else 2
            matches.append((rank, -os.path.getmtime(env), env, python, kind, '%s from %s' % (found, where)))

    matches.sort()

    return [m[2:] for m in matches]


def _rpc(port, token, body):
    req = urllib.request.Request(
        'http://localhost:%s/mcp' % port,
        data=json.dumps(body).encode(),
        headers={
            'Authorization': 'Bearer ' + token,
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',
        },
    )
    raw = urllib.request.urlopen(req, timeout=300).read().decode()
    return json.loads(re.search(r'\{.*\}', raw, re.S).group(0))


def wait(port, token, timeout=60):
    """Wait until Jupyter answers its health check; True when it does."""
    t0 = time.time()

    while time.time() - t0 < timeout:
        try:
            req = urllib.request.Request(
                'http://localhost:%s/mcp/healthz' % port,
                headers={'Authorization': 'Bearer ' + token},
            )

            if b'healthy' in urllib.request.urlopen(req, timeout=5).read():
                return True

        except Exception:
            pass

        time.sleep(2)

    return False


def tools(port, token):
    """Names of the tools Jupyter offers."""
    out = _rpc(port, token, {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'})
    return [t['name'] for t in out['result']['tools']]


def call(port, token, tool, args=None):
    """Call one Jupyter tool and return its text answer."""
    # The server only knows the notebook_* tools (e.g. run-all) once it has
    # been asked for its tool list, otherwise: "Unknown tool".
    tools(port, token)

    out = _rpc(port, token, {
        'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call',
        'params': {'name': tool, 'arguments': args or {}},
    })

    if 'error' in out:
        return 'ERROR: ' + out['error']['message']

    return '\n'.join(c['text'] for c in out['result']['content'])


def _api(port, token, path, method='GET', body=None):
    req = urllib.request.Request(
        'http://localhost:%s%s' % (port, path),
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={
            'Authorization': 'token ' + token,
            'Content-Type': 'application/json',
        },
    )
    raw = urllib.request.urlopen(req, timeout=60).read()
    return json.loads(raw) if raw else None


def kernel(port, token, path, timeout=60):
    """Id of the idle kernel of an open notebook, waiting for it; else None.

    The notebook only has a kernel once it is open in the browser.
    """
    t0 = time.time()

    while time.time() - t0 < timeout:
        for s in _api(port, token, '/api/sessions'):
            if s['path'] == path and s['kernel']['execution_state'] == 'idle':
                return s['kernel']['id']

        time.sleep(1)


def attach(port, token, path):
    """Attach the Jupyter tools to an open notebook and its kernel."""
    k = kernel(port, token, path)

    if not k:
        raise SystemExit('No kernel for %r: is it open in the browser?' % path)

    name = os.path.splitext(os.path.basename(path))[0]

    return call(port, token, 'use_notebook', {
        'notebook_name': name,
        'notebook_path': path,
        'kernel_id': k,
    })


def _notebook_kernel(port, token, path):
    for s in _api(port, token, '/api/sessions'):
        if s['path'] == path:
            return s['kernel']

    return None


def wait_open(port, token, path, timeout):
    """True once the notebook is open in a signed-in browser page.

    The server cannot see the browser's sign-in, but the page only opens the
    notebook, and so only starts its kernel session, after signing in.
    """
    return _wait_until(lambda: _notebook_kernel(port, token, path) is not None, timeout)


def watch(port, token, path, timeout, since=None):
    """Wait until the notebook's kernel did something after `since`.

    Returns ('ran', time) once the kernel was active after `since` and is idle
    again, or ('timeout', since). Pass the returned time to the next watch so
    nothing in between is missed. Any kernel request counts (a Tab completion
    too), not only running a cell.
    """
    if since is None:
        k = _notebook_kernel(port, token, path)
        since = k['last_activity'] if k else ''

    t0 = time.time()

    while time.time() - t0 < timeout:
        k = _notebook_kernel(port, token, path)

        # Same ISO format on both sides, so comparing the strings is enough.
        if k and k['last_activity'] > since and k['execution_state'] == 'idle':
            return 'ran', k['last_activity']

        time.sleep(2)

    return 'timeout', since


def wait_change(port, token, path, timeout):
    """Wait until the notebook's content changed and things have settled.

    Returns 'changed' once the cells (their code and execution counts) differ
    from when this started, the kernel is idle and nothing moved for one poll
    (about 2 seconds): the person stopped typing, or a run-all finished. A run
    shows up at its first cell, so the idle check is what waits for the last.
    Returns 'timeout' otherwise. The notebook must be attached (`attach`).
    """
    name = os.path.splitext(os.path.basename(path))[0]
    args = {'notebook_name': name, 'response_format': 'detailed', 'limit': 0}

    def snapshot():
        try:
            return call(port, token, 'read_notebook', args)
        except (OSError, ValueError):
            return None

    first = last = snapshot()
    t0 = time.time()

    while time.time() - t0 < timeout:
        time.sleep(2)
        now = snapshot()

        if now is None:
            continue

        settled = now == last
        last = now

        if first is None:
            first = now
        elif settled and now != first:
            k = _notebook_kernel(port, token, path)

            if k and k['execution_state'] == 'idle':
                return 'changed'

    return 'timeout'


def replace_kernel(port, token, path, timeout=60):
    """Shut the notebook's kernel down and start a new one; return its id.

    run-all times out after the kernel was restarted in place (same id), and
    works with a new kernel. Returns once the new kernel is idle and the
    browser page is attached to it (about 10 seconds): run-all fails with
    "Not Found" before that.
    """
    sessions = [s for s in _api(port, token, '/api/sessions') if s['path'] == path]

    if not sessions:
        raise SystemExit('No open notebook %r: is it open in the browser?' % path)

    session = sessions[0]
    new = _api(
        port,
        token,
        '/api/sessions/' + session['id'],
        'PATCH',
        {'kernel': {'name': session['kernel']['name']}},
    )
    new_kernel = new['kernel']['id']

    t0 = time.time()

    while time.time() - t0 < timeout:
        k = _api(port, token, '/api/kernels/' + new_kernel)

        if k['execution_state'] == 'idle' and k['connections'] >= 1:
            break

        time.sleep(1)

    return new_kernel


def _answers(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', int(port))) == 0


def _wait_until(check, timeout):
    t0 = time.time()

    while not check():
        if time.time() - t0 > timeout:
            return False

        time.sleep(1)

    return True


def _pids(token):
    """Ids of the processes started with this Jupyter token.

    On macOS and Linux, the processes whose command has the token option. On
    Windows, every process whose command line has the token: the launcher
    `cmd` passes it on as a plain argument, and all of them belong to that
    Jupyter (CLAUDE.md, Part 6).
    """
    if sys.platform == 'win32':
        script = (
            "Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID "
            "-and $_.CommandLine -match '%s' } | ForEach-Object { $_.ProcessId }" % token
        )
        command = ['powershell', '-NoProfile', '-Command', script]
        match = lambda line: True
    else:
        command = ['ps', '-axo', 'pid=,command=']
        match = lambda line: 'IdentityProvider.token=' + token in line

    try:
        out = subprocess.run(command, capture_output=True, text=True).stdout
    except OSError:
        return []

    pids = [int(line.split(None, 1)[0]) for line in out.splitlines()
            if line.strip() and match(line)]

    return [p for p in pids if p != os.getpid()]


def shutdown(port, token, timeout=30):
    """Shut down the Jupyter server that was started with this token.

    Only for the server you started yourself: it ends every kernel on it.
    In this order: every notebook session, every remaining kernel (there can
    be kernels without a notebook), then the server, then, only if needed, the
    process itself. Shutting down takes a while (about 10 seconds is normal),
    and the process can outlive it. Returns 'stopped', 'not running',
    'stopped after ending its process', or 'still running'.
    """
    if not _answers(port):
        return 'not running'

    # A 404 means it is already gone (two sessions can share one kernel).
    for s in _api(port, token, '/api/sessions'):
        try:
            _api(port, token, '/api/sessions/' + s['id'], 'DELETE')
        except urllib.error.HTTPError:
            pass

    for k in _api(port, token, '/api/kernels'):
        try:
            _api(port, token, '/api/kernels/' + k['id'], 'DELETE')
        except urllib.error.HTTPError:
            pass

    if not _wait_until(lambda: not _api(port, token, '/api/kernels'), timeout):
        return 'a kernel is still running: not shutting the server down'

    req = urllib.request.Request(
        'http://localhost:%s/api/shutdown' % port,
        method='POST',
        headers={'Authorization': 'token ' + token},
    )
    urllib.request.urlopen(req, timeout=30).read()

    _wait_until(lambda: not _answers(port), timeout)

    if _wait_until(lambda: not _pids(token), timeout):
        return 'stopped'

    # Windows has no SIGKILL; there SIGTERM already ends a process at once.
    for sig in (signal.SIGTERM, getattr(signal, 'SIGKILL', signal.SIGTERM)):
        for pid in _pids(token):
            try:
                os.kill(pid, sig)
            except OSError:
                pass

        if _wait_until(lambda: not _pids(token), 5):
            return 'stopped after ending its process'

    return 'still running'


def run_cell(port, token, index, start):
    """Run one cell the way a person would: select it in the page, then run it.

    Moves the page's selection (the person's cursor) to the cell at index, one
    cell at a time, and runs it only if its source starts with start, since
    the wrong cell could restart what the person is running. Needs the
    run-cell tools allowed in the start command (CLAUDE.md, Part 6). Returns
    the answer of notebook_run-cell, or why it did not run.
    """
    import ast

    index = int(index)
    selected = lambda: ast.literal_eval(call(port, token, 'notebook_get-selected-cell'))
    sel = selected()

    while sel['cellIndex'] != index:
        step = 1 if sel['cellIndex'] < index else -1
        call(port, token, 'notebook_move-cursor-down' if step == 1 else 'notebook_move-cursor-up')
        new = selected()

        if new['cellIndex'] != sel['cellIndex'] + step:
            return 'not run: the selection moved from %r to %r' % (sel['cellIndex'], new['cellIndex'])

        sel = new

    if not sel.get('source', '').strip().startswith(start.strip()):
        return 'not run: cell %d starts with %r' % (index, sel.get('source', '')[:60])

    return call(port, token, 'notebook_run-cell')


def _is_jupylet_folder(root):
    """Whether a Jupyter serving root serves Jupylet's code folder: root, or
    a folder up to two levels above it, has the jupylet package in it."""
    for folder in (root, os.path.dirname(root), os.path.dirname(os.path.dirname(root))):
        if os.path.exists(os.path.join(folder, 'jupylet', '__init__.py')):
            return True

    return False


def _env_of(pid):
    """The environment folder a process runs from, from its program's path,
    or '' if it cannot be told."""
    if sys.platform == 'win32':
        script = '(Get-CimInstance Win32_Process -Filter "ProcessId=%d").ExecutablePath' % int(pid)
        command = ['powershell', '-NoProfile', '-Command', script]
    else:
        command = ['ps', '-o', 'command=', '-p', str(int(pid))]

    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return ''

    program = out.split()[0] if out and sys.platform != 'win32' else out

    if not program:
        return ''

    folder = os.path.dirname(program)

    # bin/python on macOS, Scripts\jupyter-lab.exe on Windows; python.exe sits
    # in the environment folder itself on Windows.
    if os.path.basename(folder).lower() in ('bin', 'scripts'):
        folder = os.path.dirname(folder)

    return folder


def running(folder=None):
    """The Jupyters running now, as (port, token, folder, is it Jupylet's,
    the notebooks open in it, the environment it runs from).

    Standard library only, so it runs before any environment is known: it
    reads Jupyter's own records of its running servers (jpserver-*.json in
    Jupyter's runtime folder). A server counts only if it answers on its port
    and accepts the token in its record: Windows keeps records from servers
    long gone, and another Jupyter may have taken their port since. With
    folder, only those serving folder/examples or a folder inside it.
    """
    runtime = os.environ.get('JUPYTER_RUNTIME_DIR')

    if not runtime and os.environ.get('JUPYTER_DATA_DIR'):
        runtime = os.path.join(os.environ['JUPYTER_DATA_DIR'], 'runtime')

    if not runtime:
        if sys.platform == 'win32':
            runtime = os.path.join(os.environ.get('APPDATA', ''), 'jupyter', 'runtime')
        elif sys.platform == 'darwin':
            runtime = os.path.expanduser('~/Library/Jupyter/runtime')
        else:
            runtime = os.path.expanduser('~/.local/share/jupyter/runtime')

    examples = folder and os.path.normcase(os.path.realpath(os.path.join(folder, 'examples')))
    found, seen = [], set()
    records = []

    for path in glob.glob(os.path.join(runtime, 'jpserver-*.json')):
        try:
            with open(path) as f:
                records.append(json.load(f))
        except (OSError, ValueError):
            continue

    # Windows keeps dozens of records from dead servers, and a refused
    # connection there takes seconds: probe each port once, all at once.
    ports = sorted({r['port'] for r in records if r.get('port')})

    with ThreadPoolExecutor(max_workers=16) as pool:
        alive = {p for p, ok in zip(ports, pool.map(_answers, ports)) if ok}

    for info in records:
        root = os.path.realpath(info.get('root_dir', ''))
        port, token = info.get('port'), info.get('token', '')
        key = os.path.normcase(root)

        if not port or (port, token) in seen:
            continue

        if examples and not (key == examples or key.startswith(examples + os.sep)):
            continue

        if port not in alive:
            continue

        try:
            sessions = _api(port, token, '/api/sessions')
        except (OSError, ValueError):
            continue

        notebooks = sorted({s['path'] for s in sessions if s.get('type') == 'notebook'})
        seen.add((port, token))
        found.append((port, token, root, _is_jupylet_folder(root), notebooks,
                      _env_of(info.get('pid', 0))))

    return found


def _runtime_dir():
    try:
        from jupyter_core.paths import jupyter_runtime_dir
        return jupyter_runtime_dir()
    except Exception:
        return None


def state_files(root='.'):
    """Files Jupyter created to keep track of things, never notebooks."""
    root = os.path.abspath(root)
    paths = [
        os.path.join(root, '.jupyter'),
        os.path.join(root, 'examples', '.jupyter'),
    ]

    for folder in (root, os.path.join(root, 'examples')):
        paths += glob.glob(os.path.join(folder, '.jupyter_ystore.db*'))

    runtime = _runtime_dir()

    if runtime:
        for pattern in ('jupyter_cookie_secret', 'jpserver-*', 'kernel-*'):
            paths += glob.glob(os.path.join(runtime, pattern))

    return [p for p in paths if os.path.lexists(p)]


def cleanup(root='.', delete=False):
    """List (or, with delete=True, remove) Jupyter's state files.

    Only for use after Jupyter was shut down: it deletes the cookie secret,
    which signs the browser out, and the runtime files of running servers.
    """
    paths = state_files(root)

    for p in paths:
        print(('deleted ' if delete else 'would delete ') + p)

        if delete:
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)

    if not paths:
        print('nothing to clean up')


NBMODEL = 'jupyter_server_nbmodel'
NBMODEL_LAB = '@datalayer/jupyter-server-nbmodel'


def _nbmodel_on():
    """Whether nbmodel's server part and browser part are on, as a tuple.

    Reads the configuration Jupyter itself would read when started from this
    Python's environment, so a setting at any level (environment, user) counts.
    """
    from jupyter_core.paths import jupyter_config_path, jupyter_path
    from jupyter_server.extension.config import ExtensionConfigManager
    from jupyterlab.commands import get_app_dir
    from jupyterlab_server.config import get_page_config

    manager = ExtensionConfigManager(read_config_path=jupyter_config_path())
    server = bool(manager.enabled(NBMODEL))

    page = get_page_config(jupyter_path('labextensions'), os.path.join(get_app_dir(), 'settings'))
    browser = NBMODEL_LAB not in page.get('disabledExtensions', [])

    return server, browser


def _news_off():
    """Turn off JupyterLab's "get notified about official Jupyter news?"
    pop-up in this Python's environment, unless its settings file exists
    already. Setup does it too, but Jupylet may have been installed by hand,
    and an editable install never copies the file setup.py lists."""
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'assets', 'jupyterlab', 'overrides.json')
    folder = os.path.join(sys.prefix, 'share', 'jupyter', 'lab', 'settings')

    if os.path.exists(src) and not os.path.exists(os.path.join(folder, 'overrides.json')):
        os.makedirs(folder, exist_ok=True)
        shutil.copy(src, folder)


def nbmodel_off():
    """Make sure jupyter_server_nbmodel is off in this Python's environment.

    jupyter-mcp-server requires it, but with it Jupyter runs cells on the
    server, and the server reads the kernel's messages only while a cell
    runs. A panel or a thread that keeps sending messages then fills that
    unread queue, and a few minutes later a cell hangs at [*] for good.
    Without it, the page runs cells itself, as in plain JupyterLab.

    Both its parts are turned off, in this environment's own configuration,
    using Jupyter's own commands. It also turns off JupyterLab's news pop-up
    there, which says nothing about nbmodel but belongs to the same moment. Returns 'off', 'turned off',
    'not installed', or 'still on: ...'.
    """
    import importlib.util

    _news_off()

    if importlib.util.find_spec(NBMODEL) is None:
        return 'not installed'

    if _nbmodel_on() == (False, False):
        return 'off'

    for command in (
        ['server', 'extension', 'disable', '--sys-prefix', NBMODEL],
        ['labextension', 'disable', '--level=sys_prefix', NBMODEL_LAB],
    ):
        subprocess.run([sys.executable, '-m', 'jupyter'] + command, capture_output=True, timeout=120)

    server, browser = _nbmodel_on()

    if server or browser:
        return 'still on: ' + ', '.join(n for n, on in (('server', server), ('browser', browser)) if on)

    return 'turned off'


def main(argv):
    cmd, args = (argv[0], argv[1:]) if argv else ('', [])

    if cmd == 'find-env' and len(args) in (1, 2) and args[:-1] in ([], ['--all']):
        for env, python, kind, source in find_env(args[-1], any_version=args[0] == '--all'):
            print('%s\t%s\t%s' % (env, kind, source))

    elif cmd == 'wait' and len(args) == 2:
        ok = wait(*args)
        print('ready' if ok else 'not ready after 60 seconds')
        return 0 if ok else 1

    elif cmd == 'attach' and len(args) == 3:
        print(attach(*args))

    elif cmd == 'kernel' and len(args) == 3:
        print(kernel(*args) or 'no kernel yet: is the notebook open in the browser?')

    elif cmd == 'tools' and len(args) == 2:
        print('\n'.join(tools(*args)))

    elif cmd == 'call' and len(args) in (3, 4):
        print(call(*args[:3], _json_arg(args[3]) if len(args) == 4 else None))

    elif cmd == 'wait-open' and len(args) == 4:
        print('open' if wait_open(*args[:3], float(args[3])) else 'timeout')

    elif cmd == 'watch' and len(args) in (4, 5):
        print(*watch(*args[:3], float(args[3]), *args[4:]))

    elif cmd == 'wait-change' and len(args) == 4:
        print(wait_change(*args[:3], float(args[3])))

    elif cmd == 'replace-kernel' and len(args) == 3:
        print(replace_kernel(*args))

    elif cmd == 'run-cell' and len(args) == 4:
        print(run_cell(*args))

    elif cmd == 'shutdown' and len(args) == 2:
        print(shutdown(*args))

    elif cmd == 'running' and len(args) <= 1:
        for port, token, root, jupylet, notebooks, env in running(*args):
            print('%s\t%s\t%s\t%s\t%s\t%s' % (
                port, token, root, 'jupylet' if jupylet else '-',
                ', '.join(notebooks) or '-', env or '-'))

    elif cmd == 'cleanup' and args in ([], ['--yes']):
        cleanup(delete=args == ['--yes'])

    elif cmd == 'nbmodel-off' and not args:
        result = nbmodel_off()
        print(result)
        return 1 if result.startswith('still on') else 0

    else:
        print(USAGE)
        return 1


def _json_arg(arg):
    """JSON arguments of a tool call: the JSON itself, @file to read it from
    a file, or - to read it from stdin. Windows PowerShell 5.1 strips the
    double quotes from a JSON argument, so there the file or stdin is safer."""
    if arg == '-':
        return json.loads(sys.stdin.read().lstrip('﻿'))

    if arg.startswith('@'):
        with open(arg[1:], encoding='utf-8-sig') as f:
            return json.load(f)

    return json.loads(arg)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
