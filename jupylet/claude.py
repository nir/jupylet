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
    python -m jupylet.claude detach <log file> <command> [<argument> ...]
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
    python -m jupylet.claude prepare

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
    """Find every environment that can run the jupylet in folder, best first.

    Looks in Miniforge's own folder and its envs, in every environment listed in
    ~/.conda/environments.txt, and in a venv in folder (.venv or venv). In each,
    it runs a short probe with that environment's python. Uses only the
    standard library, so it runs by file path under any Python, on macOS and
    Windows.

    Args:
        version (str): The version of the jupylet in folder.
        root (str): Miniforge's folder; ~/miniforge3 by default.
        folder (str): The jupylet folder; the one this file is in by default.
        any_version (bool): Also list environments with any other version.

    Returns:
        list: (env path, python path, kind, source) tuples. kind is 'conda' or
            'venv'. source is 'this folder' for an editable install from
            folder, or '<version> from <path>', or '<version> from a package
            index'. 'this folder' comes first, then the same version, then,
            with any_version, other versions; within each, the most recently
            changed environment first.

    Notes:
        The source is decided by where pip installed it from (direct_url.json),
        not by the version: two copies can report the same version and differ.
        'this folder' does not mean the version still matches: a folder can be
        reused, or pulled since.
        The probe runs from inside the environment's folder, so a folder named
        jupylet in the current folder cannot shadow the real package.

    History:
        2026-09-26, macOS: an environment had jupylet installed editable from
        another checkout; notebooks imported it, and failed on anything new.
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
    """Wait until Jupyter answers its health check.

    Polls http://localhost:<port>/mcp/healthz every 2 seconds.

    Args:
        port (int): Jupyter's port.
        token (str): Jupyter's token.
        timeout (float): Seconds to wait at most.

    Returns:
        bool: True once it answers healthy, False on timeout.
    """
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


def detach(log, command):
    """Start a command detached, with its output going to a log file.

    On macOS and Linux it runs in its own session. On Windows it runs with no
    window, in its own process group, and out of the job that Claude Code's
    task may be in, if that job allows it.

    Args:
        log (str): The file the command's output is appended to.
        command (list): The command and its arguments.

    Returns:
        int: The process id.

    Notes:
        A background task of Claude Code ends when it reaches its time limit
        (10 minutes at most), when the session ends, or when the person rewinds
        the conversation, and a Jupyter started as one ends with it, with
        whatever its notebooks were running. A detached process belongs to no
        task, and runs until it is stopped.
        Not tested yet: rewinding or quitting the app with a detached Jupyter,
        and detaching on Windows.

    History:
        2026-09-25, macOS: rewinding the conversation ended the background task
        running Jupyter, and its kernel.
        2026-10-02, macOS: the background task reached its time limit, and both
        kernels ended with it. Jupyter has been started detached since.
    """
    if sys.platform != 'win32':
        kwargs = dict(start_new_session=True)

    else:
        #
        # No window, its own process group, and out of the job that the
        # task's processes may be in, so that ending the job does not end
        # it. Not every job allows that, so try once more without it.
        #
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        kwargs = dict(creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB)

    with open(log, 'a') as out:

        def start():
            return subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                **kwargs,
            )

        try:
            return start().pid

        except PermissionError:
            if sys.platform != 'win32':
                raise

            kwargs = dict(creationflags=flags)
            return start().pid


def tools(port, token):
    """List the tools Jupyter offers.

    Returns:
        list: The tools' names.
    """
    out = _rpc(port, token, {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'})
    return [t['name'] for t in out['result']['tools']]


def call(port, token, tool, args=None):
    """Call one of Jupyter's tools.

    Args:
        port (int): Jupyter's port.
        token (str): Jupyter's token.
        tool (str): The tool's name, such as 'read_cell'.
        args (dict): The tool's arguments.

    Returns:
        str: The tool's text answer, or 'ERROR: <message>'.

    Notes:
        It asks for the tool list first, every time: the server only knows the
        page's tools, such as notebook_run-all-cells, once it has been asked for
        its list, and answers "Unknown tool" otherwise (CLAUDE.md, Problem 3).
    """
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
    """Find the kernel of a notebook open in the browser, once it is idle.

    Args:
        port (int): Jupyter's port.
        token (str): Jupyter's token.
        path (str): The notebook's path, relative to Jupyter's folder.
        timeout (float): Seconds to wait at most.

    Returns:
        str: The kernel's id, or None on timeout.

    Notes:
        A notebook only gets a kernel once it is open in the browser.
    """
    t0 = time.time()

    while time.time() - t0 < timeout:
        for s in _api(port, token, '/api/sessions'):
            if s['path'] == path and s['kernel']['execution_state'] == 'idle':
                return s['kernel']['id']

        time.sleep(1)


def attach(port, token, path):
    """Attach the Jupyter tools to a notebook open in the browser.

    Args:
        port (int): Jupyter's port.
        token (str): Jupyter's token.
        path (str): The notebook's path, relative to Jupyter's folder.

    Returns:
        str: The answer of use_notebook; it contains "Successfully activate
            notebook" when it worked.

    Raises:
        SystemExit: If the notebook has no kernel within a minute.
    """
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
    """Wait until the notebook is open in a signed-in browser page.

    Args:
        port (int): Jupyter's port.
        token (str): Jupyter's token.
        path (str): The notebook's path, relative to Jupyter's folder.
        timeout (float): Seconds to wait at most.

    Returns:
        bool: True once it is open, False on timeout.

    Notes:
        The server cannot see the browser's sign-in, but the page only opens the
        notebook, and so only starts its session, after signing in. It looks
        the session up on every check, so it keeps working after
        replace_kernel().

    History:
        2026-09-22, Windows 11: with the page signed out, on the login page, no
        session appeared.
    """
    return _wait_until(lambda: _notebook_kernel(port, token, path) is not None, timeout)


def watch(port, token, path, timeout, since=None):
    """Wait until the notebook's kernel did something after `since`.

    Polls the kernel's last_activity in /api/sessions every 2 seconds.

    Args:
        port (int): Jupyter's port.
        token (str): Jupyter's token.
        path (str): The notebook's path, relative to Jupyter's folder.
        timeout (float): Seconds to wait at most.
        since (str): The time returned by the previous watch; now if None.

    Returns:
        tuple: ('ran', <time>) once the kernel was active after since and is
            idle again, or ('timeout', since). Pass the time on to the next
            watch, so nothing in between is missed.

    Notes:
        Any kernel request moves last_activity, a Tab completion too, so 'ran'
        means something happened, not that a cell ran. To find which cell ran,
        compare execution counts before and after; the highest count misleads,
        since cells keep the counts of earlier kernels.

    History:
        2026-09-22, Windows 11: over idle minutes with the page open, nothing
        moved last_activity by itself; two cells showed 16 next to a fresh 1.
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
    """Wait until the notebook's content changed, and things settled.

    Reads the notebook every 2 seconds. Its cells (their code and execution
    counts) must differ from when it started, or the kernel must have been
    seen busy since; the kernel must be idle, and nothing may have moved for one
    read: the person stopped typing, or a run-all finished. A run shows up at
    its first cell, so the idle check is what waits for the last.

    Args:
        port (int): Jupyter's port.
        token (str): Jupyter's token.
        path (str): The notebook's path, relative to Jupyter's folder; it must
            be attached (attach()).
        timeout (float): Seconds to wait at most.

    Returns:
        str: 'changed', or 'timeout'.

    Notes:
        Not known whether it sees what is being typed before it is saved.
        A run that leaves the same counts and is over between two reads goes
        unseen, and ends in 'timeout': look at the notebook then anyway.

    History:
        2026-10-05, macOS: after setup's first run-all it waited until its
        timeout: the shipped notebook has counts 1 to 14 saved, and a fresh
        kernel's run-all gives the same counts, so the cells never differed.
        The kernel's last_activity could not tell either: while the game ran,
        it read "just now" every time.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    args = {'notebook_name': name, 'response_format': 'detailed', 'limit': 0}

    def snapshot():
        try:
            return call(port, token, 'read_notebook', args)
        except (OSError, ValueError):
            return None

    busy = False
    first = last = snapshot()
    t0 = time.time()

    while time.time() - t0 < timeout:
        time.sleep(2)
        now = snapshot()

        k = _notebook_kernel(port, token, path)
        busy = busy or bool(k and k['execution_state'] != 'idle')

        if now is None:
            continue

        settled = now == last
        last = now

        if first is None:
            first = now
            continue

        if not settled or not k or k['execution_state'] != 'idle':
            continue

        if now != first or busy:
            return 'changed'

    return 'timeout'


def replace_kernel(port, token, path, timeout=60):
    """Shut the notebook's kernel down, and start a new one in its place.

    Returns once the new kernel is idle, and the browser page is attached to it
    (about 10 seconds).

    Args:
        port (int): Jupyter's port.
        token (str): Jupyter's token.
        path (str): The notebook's path, relative to Jupyter's folder.
        timeout (float): Seconds to wait at most for the new kernel.

    Returns:
        str: The new kernel's id.

    Raises:
        SystemExit: If the notebook is not open in the browser.

    Notes:
        It ends whatever the old kernel was running, a game or a sound
        included: look first whether run-all really failed.
        run-all timed out after the kernel was restarted in place (the Restart
        Kernel button, which keeps the id), and worked with a new kernel. The
        cause is unknown. Before the page is attached, run-all fails with
        "Not Found".

    History:
        2026-09, macOS: run-all timed out after Restart Kernel; not seen in one
        test with nbmodel off.
        2026-09-22, Windows 11: a run-all that timed out had run every cell
        anyway, as their new execution counts showed.
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
    """Find the processes of the Jupyter started with this token.

    On macOS and Linux, the processes whose command has the token option. On
    Windows, every process whose command line has the token: the launcher cmd
    passes it on as a plain argument, and all of them belong to that Jupyter
    (CLAUDE.md, Part 6). This process, and those that started it, never count.

    Args:
        token (str): Jupyter's token.

    Returns:
        list: Process ids.

    Notes:
        Processes are found by the token, never by a process id from Jupyter's
        records, which can be stale.

    History:
        2026-09-22, Windows 11: Jupyter's records named two dozen servers long
        gone, and a recorded process id had come to belong to the Claude app.
        2026-10-03, macOS: shutdown ended the shell running it, whose longer
        command had the token in it; since then, this process and those that
        started it are left out (_ancestors()).
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

    #
    # Leave out this process and those that started it: a shell running a
    # longer command can have the token in its command line too, and must
    # not be ended with Jupyter.
    #
    ancestors = _ancestors()

    return [p for p in pids if p not in ancestors]


def _ancestors():
    """Find this process, and every process that started it.

    Returns:
        set: Process ids, up to the first process of the system.
    """

    if sys.platform == 'win32':
        script = (
            "Get-CimInstance Win32_Process | ForEach-Object "
            "{ '{0} {1}' -f $_.ProcessId, $_.ParentProcessId }"
        )
        command = ['powershell', '-NoProfile', '-Command', script]
    else:
        command = ['ps', '-axo', 'pid=,ppid=']

    try:
        out = subprocess.run(command, capture_output=True, text=True).stdout
    except OSError:
        return {os.getpid()}

    parents = dict(map(int, line.split()) for line in out.splitlines() if len(line.split()) == 2)

    pid, ancestors = os.getpid(), set()

    while pid and pid not in ancestors:
        ancestors.add(pid)
        pid = parents.get(pid)

    return ancestors


def shutdown(port, token, timeout=30):
    """Shut down the Jupyter server started with this token, and its kernels.

    Ends every notebook session and every kernel (there can be kernels without
    a notebook), asks the server to shut down, and waits for its port to close
    and its processes to exit. If any process lingers, it ends it: first
    politely, then by force. Only for the server you started yourself. About 10
    seconds is normal.

    Args:
        port (int): Jupyter's port.
        token (str): Jupyter's token.
        timeout (float): Seconds to wait at each stage.

    Returns:
        str: 'stopped', 'not running', 'stopped after ending its process',
            'a kernel is still running: not shutting the server down', or
            'still running'.

    Notes:
        The processes can outlive the shutdown request; ending them here
        spares the person a problem they cannot solve.

    History:
        2026-09-22 and 2026-09-23, Windows 11: in two of three runs, the port
        closed but all five of the launcher's processes stayed alive; ending
        each by its id worked at once.
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

    Moves the page's selection (the person's cursor) to the cell, one cell at a
    time, and runs it only if its source starts with start. Needs the run-cell
    tools allowed in the start command (CLAUDE.md, Part 6).

    Args:
        port (int): Jupyter's port.
        token (str): Jupyter's token.
        index (int): The cell's index.
        start (str): How the cell's source starts.

    Returns:
        str: The answer of notebook_run-cell ('True' when it ran), or
            'not run: ...', with the reason.

    Notes:
        The wrong cell could restart what the person is running, and indices
        shift whenever the person adds or deletes a cell, so the source is
        checked before running.

    History:
        2026-09-22, Windows 11: 33 moves took about 1.6 seconds.
        2026-09-25, macOS: edits made by indices read a few minutes earlier
        replaced three cells the person had just added.
        2026-10-01: replaced a longer script; not tested on Windows since.
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
    """Tell whether a Jupyter serving root serves Jupylet's code folder.

    Returns:
        bool: True if root, or a folder up to two levels above it, has the
            jupylet package in it.
    """
    for folder in (root, os.path.dirname(root), os.path.dirname(os.path.dirname(root))):
        if os.path.exists(os.path.join(folder, 'jupylet', '__init__.py')):
            return True

    return False


def _env_of(pid):
    """Find the environment folder a process runs from, by its program's path.

    Returns:
        str: The folder, or '' if it cannot be told.
    """
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
    """List the Jupyters running now.

    Reads Jupyter's own records of its running servers (jpserver-*.json in
    Jupyter's runtime folder). A server counts only if it answers on its port
    and accepts the token in its record. Uses only the standard library, so it
    runs before any environment is known.

    Args:
        folder (str): If given, only the Jupyters serving folder/examples, or a
            folder inside it.

    Returns:
        list: (port, token, folder served, whether it is Jupylet's, the
            notebooks open in it, the environment it runs from) tuples.

    Notes:
        Records can outlive their servers, and another Jupyter may have taken
        their port since, hence both checks. Ports are probed all at once:
        a refused connection takes seconds on Windows.
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
    """List the files Jupyter created to keep track of things, never notebooks.

    Args:
        root (str): The Jupylet folder.

    Returns:
        list: Paths: the .jupyter folders, any .jupyter_ystore.db, and the
            cookie secret and jpserver-* and kernel-* files in Jupyter's
            runtime folder.

    Notes:
        .jupyter_ystore.db is only left from before prepare() set up the
        environment (see _collaboration_config()).
    """
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
    """List Jupyter's state files, or delete them.

    Only after Jupyter was shut down: it deletes the cookie secret, which signs
    the browser out, and the runtime files of running servers.

    Args:
        root (str): The Jupylet folder.
        delete (bool): Delete them, rather than only list them.

    Prints:
        'would delete <path>' or 'deleted <path>' per file, or 'nothing to
        clean up'.
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


def _collaboration_config():
    """Keep a notebook's live copy for as long as Jupyter runs.

    Writes two settings into this Python's environment, in
    etc/jupyter/jupyter_server_config.json, keeping whatever else is there,
    unless both are set there already:

    - YDocExtension.document_cleanup_delay = null: Jupyter keeps a notebook's
      live copy, the one the page and the tools share, for as long as it runs.
    - YDocExtension.ystore_class = TempFileYStore: the log of edits it could be
      rebuilt from lives in a temporary folder, only for that run.

    Notes:
        By default, Jupyter drops the live copy a minute after the last
        connection closes, and rebuilds it from .jupyter_ystore.db, which
        survives restarts. A page still open from before then stops taking the
        tools' changes: a cell inserted or edited through them is in
        read_notebook and in the saved file, but the page shows fewer cells, or
        that cell as one empty line, even after a reload.
        Settings in the jupyter_server_config.d folder have no effect: Jupyter
        reads that folder only to turn extensions on and off.
        To check: `jupyter lab --show-config` lists both under YDocExtension,
        Jupyter's log says the notebook was "loaded from file", and no
        .jupyter_ystore.db appears.

    History:
        2026-09-22, Windows 11; 2026-09-28, macOS: cells added over MCP did not
        show in the page; fixed by moving the state files aside and restarting.
        2026-10-02 and 2026-10-04, macOS: twice more; the log showed the cause
        ("Deleting Y document from memory", then "loaded from the ystore
        SQLiteYStore" when the page came back, after the computer slept).
    """
    #
    # Jupyter reads general settings from this file, in each environment's
    # own folder. The jupyter_server_config.d folder beside it is read only
    # for turning extensions on and off.
    #
    path = os.path.join(sys.prefix, 'etc', 'jupyter', 'jupyter_server_config.json')

    config = {}

    if os.path.exists(path):
        with open(path) as f:
            config = json.load(f)

    ydoc = config.setdefault('YDocExtension', {})

    if 'ystore_class' in ydoc and 'document_cleanup_delay' in ydoc:
        return

    ydoc.setdefault('ystore_class', 'jupyter_server_ydoc.stores.TempFileYStore')
    ydoc.setdefault('document_cleanup_delay', None)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def prepare():
    """Prepare this Python's environment to run Jupylet with Claude.

    Run before each start of Jupyter. Each part is set once, saved in the
    environment, and only checked after that:

    - Turn off jupyter_server_nbmodel (_nbmodel_off()).
    - Keep a notebook's live copy for as long as Jupyter runs
      (_collaboration_config()).
    - Turn off JupyterLab's news pop-up (_news_off()).

    Returns:
        str: nbmodel's state: 'off', 'turned off', 'not installed', or
            'still on: ...'.
    """
    _news_off()
    _collaboration_config()

    return _nbmodel_off()


def _nbmodel_off():
    """Make sure jupyter_server_nbmodel is off in this Python's environment.

    Turns off both its parts, the server's and the browser's, in this
    environment's own configuration, with Jupyter's own commands.

    Returns:
        str: 'off', 'turned off', 'not installed', or 'still on: ...', naming
            the part still on.

    Notes:
        jupyter-mcp-server requires nbmodel, but with it Jupyter runs cells on
        the server, through one kernel client that reads the kernel's messages
        only while a cell runs. A panel or a thread that keeps sending messages
        fills that unread queue past its limit of 1000 messages, newer ones are
        dropped, and among them a cell's 'idle' message, which the server waits
        for with no timeout: the cell hangs at [*] for good. Output a thread
        prints after its cell finished never shows either.
        Without nbmodel, the page runs cells itself, as in plain JupyterLab, and
        every tool still works except execute_cell and insert_execute_code_cell,
        which hung anyway.

    History:
        2026-09-25, macOS: a cell hung at [*], seen 4 times, after a live loop
        or a self-refreshing widget had run for a while.
        2026-10-01, macOS: seen again with a Panel refreshing while a live loop
        turned a knob (about 8 messages a second); every tool used was verified
        to work with nbmodel off.
        2026-10-05, macOS: setup's prepare printed 'still on: server, browser'
        for a new environment, because `python -m jupyter` ran base's Jupyter,
        found first on PATH, and wrote to base's configuration instead.
    """
    import importlib.util

    if importlib.util.find_spec(NBMODEL) is None:
        return 'not installed'

    if _nbmodel_on() == (False, False):
        return 'off'

    #
    # Run the modules behind `jupyter server extension` and `jupyter
    # labextension` directly: `python -m jupyter` looks its subcommands up on
    # PATH first, and may run another environment's Jupyter.
    #
    for command in (
        ['jupyter_server.extension.serverextension', 'disable', '--sys-prefix', NBMODEL],
        ['jupyterlab.labextensions', 'disable', '--level=sys_prefix', NBMODEL_LAB],
    ):
        subprocess.run([sys.executable, '-m'] + command, capture_output=True, timeout=120)

    server, browser = _nbmodel_on()

    if server or browser:
        return 'still on: ' + ', '.join(n for n, on in (('server', server), ('browser', browser)) if on)

    return 'turned off'


def main(argv):
    cmd, args = (argv[0], argv[1:]) if argv else ('', [])

    if cmd == 'find-env' and len(args) in (1, 2) and args[:-1] in ([], ['--all']):
        for env, python, kind, source in find_env(args[-1], any_version=args[0] == '--all'):
            print('%s\t%s\t%s' % (env, kind, source))

    elif cmd == 'detach' and len(args) >= 2:
        print(detach(args[0], args[1:]))

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

    elif cmd == 'prepare' and not args:
        result = prepare()
        print(result)
        return 1 if result.startswith('still on') else 0

    else:
        print(USAGE)
        return 1


def _json_arg(arg):
    """Read a tool call's JSON arguments, given on the command line.

    Args:
        arg (str): The JSON itself, '@<file>' to read it from a file, or '-' to
            read it from stdin.

    Returns:
        dict: The arguments.

    Notes:
        Windows PowerShell 5.1 strips the double quotes from a JSON argument,
        so there the file or stdin is the safe way.
    """
    if arg == '-':
        return json.loads(sys.stdin.read().lstrip('﻿'))

    if arg.startswith('@'):
        with open(arg[1:], encoding='utf-8-sig') as f:
            return json.load(f)

    return json.loads(arg)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
