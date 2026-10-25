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
    python -m jupylet.claude wait <log file> <token>
    python -m jupylet.claude attach <port> <token> <notebook path>
    python -m jupylet.claude kernel <port> <token> <notebook path>
    python -m jupylet.claude tools <port> <token>
    python -m jupylet.claude call <port> <token> <tool> ['<json arguments>']
    python -m jupylet.claude wait-open <port> <token> <notebook path> <seconds>
    python -m jupylet.claude wait-change <port> <token> <notebook path> <seconds> [<since>]
    python -m jupylet.claude run-all <port> <token> <notebook path> <seconds>
    python -m jupylet.claude watch <port> <token> <notebook path> <seconds> <state file or new>
    python -m jupylet.claude cell <port> <token> <notebook path> <cell id or index>
    python -m jupylet.claude replace-kernel <port> <token> <notebook path>
    python -m jupylet.claude run-cell <port> <token> <cell index> <start of its source>
    python -m jupylet.claude shutdown <port> <token>
    python -m jupylet.claude cleanup [--yes]
    python -m jupylet.claude running [<folder>]
    python -m jupylet.claude prepare [--check]

`find-env` is the one command meant to run from any Python, with or
without jupylet, such as Miniforge's `base` or a plain python3 (run this
file by path: it never imports jupylet itself, on purpose). `running` too. It prints one environment per line:
its path, its kind (conda or venv), and where its jupylet comes from ("this
folder", or a version and its source). With --all it also lists
environments with any other version of jupylet. Every other command needs an
environment that actually has jupylet installed.

`running` prints a JSON list, one object per Jupyter server that runs:

    port, token   how to reach it.
    folder        the folder it serves.
    environment   the environment it runs from, or null if unknown.
    jupylet_version
                  the version of jupylet installed in that environment, or
                  null if it has none or the environment is unknown.
    mcp_tools     true if the server answers with its MCP tools: the [claude]
                  extra is installed and loaded.
    nbmodel_off   true if jupyter_server_nbmodel is off in the server, both
                  its server part and its browser part.
    notebooks     the notebooks open in it, each as {"path", "connections"}.
                  connections counts the pages and other clients connected
                  to the notebook's kernel: 0 means no page shows it.

With <folder>, only the servers that serve <folder>/examples or a folder
inside it. For example:

    [
      {
        "port": 8888,
        "token": "3f9a1c07",
        "folder": "/Users/sam/jupylet/examples",
        "environment": "/Users/sam/miniforge3/envs/jupylet",
        "jupylet_version": "0.10.0",
        "mcp_tools": true,
        "nbmodel_off": true,
        "notebooks": [
          {"path": "11-spaceship.ipynb", "connections": 1},
          {"path": "02-hello-jupylet.ipynb", "connections": 0}
        ]
      }
    ]

What the other commands print:

detach <log file> <command> [<argument> ...]
    The process id of the started command, for example `23413`, at once.
    The command runs on, apart from the caller; its output goes to <log file>.

wait <log file> <token>
    `ready <port>` once the server started with <token> answers, for example
    `ready 8889`, or `not ready after 60 seconds` (exit code 1). <log file>
    is the log file given to detach: the server prints its address there.

attach <port> <token> <notebook path>
    The answer of the Jupyter MCP tool use_notebook, about ten lines. On
    success it contains `Successfully activate notebook`, for example:

        [INFO] Connected to kernel '3f9a1c52-...'.
        [INFO] Existing Jupyter session already binds '11-spaceship.ipynb' to kernel '3f9a1c52-...'.
        [INFO] Successfully activate notebook '11-spaceship'.

        Notebook has 34 cells.
        ...

kernel <port> <token> <notebook path>
    The id of the notebook's kernel, or `no kernel yet: is the notebook open
    in the browser?`. A notebook gets a kernel only when a web page opens it.

tools <port> <token>
    The names of the Jupyter MCP tools, one on each line.

call <port> <token> <tool> ['<json arguments>']
    The tool's own answer, as it gives it. The arguments are JSON, `@<file>`,
    or `-` to read them from stdin.

wait-open <port> <token> <notebook path> <seconds>
    `open` when a new web page opens the notebook, signed in, or `timeout`.
    A web page that had the notebook open before the wait started does not
    count.

wait-change <port> <token> <notebook path> <seconds> [<since>]
    `finished: ...` once the kernel ran something and is idle again, or
    `not finished: ...` after <seconds>. Each ends with the fingerprint,
    for example `Fingerprint: 3c9d07e5.5f2a81b4`. Give the fingerprint as
    <since> to the next wait-change, so that a run between the two is not
    missed. A Tab completion counts as a run too. Edits without a run do
    not end it.

run-all <port> <token> <notebook path> <seconds>
    Runs all cells through the web page, and waits until the run is over:
    `finished: ...` or `still running: ...`, each with the fingerprint, or
    `not started: ...` with the message of the page if the run did not
    start.

watch <port> <token> <notebook path> <seconds> <state file or new>
    Waits for any change in the notebook, an edit or a run, until the
    notebook is still for a few seconds and the kernel is idle. Prints
    `changed: <n> cells.` and, for each changed cell, its index, its id
    when it has one, its type, whether it ran or moved, the changed lines
    of its source (`-` old, `+` new) and its new output, short. Or `no change: ...` after <seconds>. The state
    file keeps the notebook of the last call, so each call reports only
    what changed since the one before. A new state file starts from the
    notebook as it is. When a cell runs for more than two reads, about 4
    seconds, it prints `running: a cell is still running.` at once, with
    the changes so far; the next call reports the output when the cell
    ends. A cell that ends in an error is always shown with its error.
    Each answer starts with `watching for <time>.`, the time since the
    first call with this state file. With `new` in place of the state file,
    it makes a new state file in the temporary folder, and the answer starts
    with `state file: <path>.`: give that path to the next calls.

cell <port> <token> <notebook path> <cell id or index>
    Prints the source and the whole output of one cell, for example of a
    cell whose block watch trimmed. An error comes with its whole
    traceback. Give the id or the index that
    watch prints. An id stays the same when cells are added or moved; an
    index does not. A notebook older than nbformat 4.5 has no ids.

replace-kernel <port> <token> <notebook path>
    The id of the new kernel, once it is idle and the web page is attached
    to it.

run-cell <port> <token> <cell index> <start of its source>
    `True` when the cell ran. Before it runs, it checks that the cell at
    <cell index> starts with <start of its source>, so a cell that moved is
    never run by mistake: `not run: cell 4 starts with 'import math'`, or
    `not run: the selection moved from 4 to 5`.

shutdown <port> <token>
    `stopped`, `not running`, or `stopped after ending its process` (a
    process outlived the shutdown, and was ended). Also `a kernel is still
    running: not shutting the server down`, or `still running`.

cleanup [--yes]
    One line for each state file: `would delete <path>`, or `deleted <path>`
    with --yes; or `nothing to clean up`.

prepare
    The state of jupyter_server_nbmodel: `turned off` the first time, `off`
    after that, `not installed`, or `still on: <parts>` (exit code 1). It
    also keeps the notebook's live copy, and turns off JupyterLab's news.

prepare --check
    `ready`, or `needs: ` and what prepare would change, for example
    `needs: nbmodel, live copy, news`. It changes nothing.
"""


import difflib
import glob
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
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


def wait(log, token, timeout=60):
    """Wait until a Jupyter just started with `token` answers, and find its port.

    Jupyter picks the first free port from 8888 on, and prints its address
    to its output a few seconds after it starts. That output goes to `log`.
    Every 2 seconds, the ports of the addresses in `log` are tried, newest
    first. The server is ours if it accepts `token`, and ready once it also
    answers its health check at /mcp/healthz.

    Args:
        log (str): The log file that detach gave the Jupyter command.
        token (str): The token Jupyter was started with.
        timeout (float): Seconds to wait at most.

    Returns:
        int: The port, or None on timeout.

    Notes:
        detach appends to the log, so it can hold the addresses of servers
        started before. Their tokens differ, so they never pass. Jupyter
        prints a token given on the command line as `...`, so the address
        alone cannot tell.
    """
    t0 = time.time()

    while time.time() - t0 < timeout:
        for port in reversed(_ports_in(log)):
            try:
                _api(port, token, '/api/status')

                req = urllib.request.Request(
                    'http://localhost:%s/mcp/healthz' % port,
                    headers={'Authorization': 'Bearer ' + token},
                )

                if b'healthy' in urllib.request.urlopen(req, timeout=5).read():
                    return port

            except Exception:
                pass

        time.sleep(2)

    return None


def _ports_in(log):
    try:
        with open(log, errors='replace') as f:
            text = f.read()
    except OSError:
        return []

    ports = re.findall(r'https?://(?:localhost|127\.0\.0\.1|\[::1\]):(\d+)/', text)
    return list(dict.fromkeys(int(p) for p in ports))


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
    """Wait until a new web page opens the notebook, signed in.

    Args:
        port (int): Jupyter's port.
        token (str): Jupyter's token.
        path (str): The notebook's path, relative to Jupyter's folder.
        timeout (float): Seconds to wait at most.

    Returns:
        bool: True once a new web page has it open, False on timeout.

    Notes:
        The server cannot see the browser's sign-in, but a web page opens the
        notebook, and so connects to its kernel, only after signing in. If
        the notebook has no session yet, its first session means a new web
        page opened it. If it has one, another web page opened it earlier, so
        only one more connection to its kernel than at the start means a new
        web page. It looks the session up on every check, so it keeps working
        after replace_kernel().

    History:
        2026-09-22, Windows 11: with the page signed out, on the login page, no
        session appeared.
        2026-10-14, Windows 11 (W2): the notebook was already open in the
        default web browser. The wait printed 'open' at once, before the new
        page in the Claude app signed in, and Claude missed the real sign-in.
    """
    first = _notebook_kernel(port, token, path)
    before = first['connections'] if first else None

    def opened():
        k = _notebook_kernel(port, token, path)
        return k is not None and (before is None or k['connections'] > before)

    return _wait_until(opened, timeout)


def run_all(port, token, path, timeout):
    """Run all cells of a notebook, and wait until the run is over.

    Starts watching the notebook first, then runs all its cells through the
    page, as Run > Run All Cells does, so the watch sees the run happen.

    Args:
        port (int): Jupyter's port.
        token (str): Jupyter's token.
        path (str): The notebook's path, relative to Jupyter's folder; it must
            be attached (attach()) and open in a page.
        timeout (float): Seconds to wait at most.

    Returns:
        tuple: ('ran', <fingerprint>) or ('timeout', <fingerprint>), as
            wait_change() returns them, or ('error', <message>) if the run
            did not start.

    Notes:
        wait_change() sees a run by the kernel turning busy, or by changed
        execution counts. A run-all on a fresh kernel gives the same counts
        as the saved notebook, so a watch that starts after the run is over
        sees nothing, and times out. Starting the watch first is the reason
        this command exists.

    History:
        2026-10-12, macOS: Claude ran the run-all first and started the watch
        after it; the watch timed out, and Claude reported success from a
        screenshot only.
        2026-10-22, macOS: the first run in a fresh environment took longer
        than 30 seconds; the tool said "Timeout waiting for result", and
        run_all reported an error although the run went on.
    """
    result = {}

    def watch_it():
        result['r'] = wait_change(port, token, path, timeout)

    watcher = threading.Thread(target=watch_it, daemon=True)
    watcher.start()

    #
    # Let the watch take its first reading before the run starts.
    #
    time.sleep(1)

    out = str(call(port, token, 'notebook_run-all-cells')).strip()

    #
    # The tool waits for the run to end, and gives up after about 30
    # seconds with this message, while the run goes on.
    #
    if out != 'True' and 'Timeout waiting for result' not in out:
        return 'error', out

    watcher.join()

    return result.get('r', ('timeout', ''))


def wait_change(port, token, path, timeout, since=None):
    """Wait until the person, or a run-all, ran something in the notebook.

    Reads the notebook every 2 seconds, and compares it with where it
    started: `since`, or what it first reads. A run is when the execution
    counts differ, or the kernel was seen busy. It returns once the kernel is
    idle and nothing moved for one read, so a run-all is over before it
    returns. Edits without a run do not end it.

    Args:
        port (int): Jupyter's port.
        token (str): Jupyter's token.
        path (str): The notebook's path, relative to Jupyter's folder; it must
            be attached (attach()).
        timeout (float): Seconds to wait at most.
        since (str): The fingerprint the previous wait_change printed; what it
            first reads if None.
    Returns:
        tuple: ('ran', <fingerprint>), or ('timeout', <the starting
            fingerprint>). Pass the fingerprint on to
            the next wait_change, so nothing in between is missed.

    Notes:
        Each return wakes Claude for a turn that rereads the whole
        conversation, which costs quota: it returns only on a run or at the
        timeout. Claude reads what the person typed at the timeout, when it
        checks in.
        A Tab completion makes the kernel busy too, so 'ran' means the kernel
        did something; compare execution counts to see whether a cell ran.
        A run that leaves the same counts and is over between two reads goes
        unseen, and ends in 'timeout': look at the notebook then anyway.
        Not known whether it sees what is being typed before it is saved.

    History:
        2026-10-05, macOS: after setup's first run-all it waited until its
        timeout: the shipped notebook has counts 1 to 14 saved, and a fresh
        kernel's run-all gives the same counts, so the cells never differed.
        The kernel's last_activity could not tell either: while the game ran,
        it read "just now" every time.
        2026-10-05, macOS: it returned 2 seconds after the person stopped
        typing, and Claude pointed out a typo before they had run the cell; it
        no longer returns on edits. A run in the gap between two waits went
        unseen; hence `since`.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    args = {'notebook_name': name, 'response_format': 'detailed', 'limit': 0}

    def snapshot():
        try:
            return _fingerprint(call(port, token, 'read_notebook', args))
        except (OSError, ValueError):
            return None

    start = since or None
    busy = False
    last = None
    t0 = time.time()

    while True:
        now = snapshot()
        k = _notebook_kernel(port, token, path)
        idle = bool(k and k['execution_state'] == 'idle')
        busy = busy or bool(k and not idle)

        if now is not None:
            start = start or now
            settled = now == last
            last = now
            counts = now.split('.')[0]

            if idle and settled and (busy or counts != start.split('.')[0]):
                return 'ran', now

        if time.time() - t0 >= timeout:
            return 'timeout', start or ''

        time.sleep(2)


def _fingerprint(text):
    """A short fingerprint of a read_notebook text: its execution counts, a
    dot, and the whole text, each hashed."""
    counts = ','.join(re.findall(r'\| execution count: ([^=]*)=', text))
    digest = lambda x: hashlib.md5(x.encode()).hexdigest()[:8]
    return digest(counts) + '.' + digest(text)


def watch(port, token, path, timeout, state, every=2):
    """Wait for any change in a notebook, an edit or a run, and describe it.

    Checks the time of the last save of the notebook every `every` seconds,
    reads the notebook when that time changes, and compares it with the
    notebook of the last call, which it keeps in the file `state`. It
    returns once something changed, two reads in a row were the same, and
    the kernel is idle, so it does not return in the middle of typing or of
    a run. Then it writes the new notebook to `state`, so the
    next call describes only what changed after this one.

    Args:
        port (int): Jupyter's port.
        token (str): Jupyter's token.
        path (str): The notebook's path, relative to Jupyter's folder.
        timeout (float): Seconds to wait at most.
        state (str): The file that keeps the notebook of the last call. If
            it does not exist, the notebook as it is now is the start. If it
            is 'new', a new file in the temporary folder, whose path the
            answer gives in its first line.
        every (float): Seconds between two reads.

    Returns:
        str: A short report: `changed: ...` with one block for each changed
            cell, `running: ...` when a cell runs for more than two reads,
            or `no change: ...` at the timeout.

    Notes:
        It reads the notebook through the contents API, not through a web
        page or the MCP tools, so it changes nothing. Most checks ask only
        for the time of the last save, which is cheap; it reads the whole
        notebook only when that time changes. The
        Jupyter server saves each change of the web page within a few
        seconds, so a change shows here a few seconds late.
        Cells are matched by their id, so a moved cell is not a new one.
    """
    first = ''

    if state == 'new':
        fd, state = tempfile.mkstemp(prefix='notebook-monitor-', suffix='.json')
        os.close(fd)
        first = 'state file: %s. Give it to the next calls.\n' % state

    start, reported, started = _watch_load(state)
    started = started or time.time()
    say = lambda text: first + _watch_elapsed(started) + '\n' + text
    last = None
    stamp = None
    busy = 0
    t0 = time.time()

    while True:
        saved = _watch_stamp(port, token, path)
        now = last if saved is not None and saved == stamp else _watch_read(port, token, path)
        stamp = saved

        if now is not None:
            start = start if start is not None else now
            still = now == last
            last = now

            k = _notebook_kernel(port, token, path)
            idle = not k or k['execution_state'] == 'idle'
            busy = 0 if idle else busy + 1

            if now != start and still and idle:
                _watch_save(state, now, started)
                return say(_watch_report(start, now))

            #
            # A cell that runs longer than two reads is reported as running,
            # so that the chat hears of it at once; the next call reports
            # its output when it ends. Once reported, the same run is not
            # reported again, unless something else changed.
            #
            if busy >= 2 and (now != start or not reported):
                _watch_save(state, now, started, running=True)
                return say('running: a cell is still running.' + (
                    '\n' + _watch_report(start, now) if now != start else ''))

        if time.time() - t0 >= timeout:
            break

        time.sleep(every)

    if last is None:
        return say('no change: the notebook could not be read in %g seconds.' % timeout)

    #
    # A change that did not settle in time is still reported, so that a
    # person who keeps typing is not missed; the next call goes on from it.
    #
    if last != start:
        _watch_save(state, last, started)
        return say(_watch_report(start, last, settled=False))

    _watch_save(state, last, started)
    return say('no change: nothing changed in %g seconds.' % timeout)


def cell_text(port, token, path, which):
    """The source and the whole output of one cell, as text.

    For a monitor or Claude that wants more than the short report of
    watch(), for example the whole output of a cell whose block `watch`
    trimmed.

    Args:
        port (int): Jupyter's port.
        token (str): Jupyter's token.
        path (str): The notebook's path, relative to Jupyter's folder.
        which (str): The cell's id, or its index from 0, as watch() prints
            them. An id stays the same when cells are added or moved; an
            index does not.

    Returns:
        str: The cell's type and count, its source, and its output, with an
            image or a widget as one word, and an error with its traceback.
    """
    cells = _watch_read(port, token, path, lines=None, traceback=True)

    if cells is None:
        return 'not read: the notebook could not be read.'

    index = next((i for i, c in enumerate(cells) if c['id'] and c['id'] == which), None)

    if index is None and which.isdigit():
        index = int(which)

    if index is None or index >= len(cells):
        return 'no cell %s: the notebook has %d cells, from 0.' % (which, len(cells))

    c = cells[index]
    count = '' if c['count'] is None else ', count %s' % c['count']

    return '%s (%s%s):\n%s\n%s' % (
        _watch_name(index, c), c['type'], count, c['source'], '\n'.join(_watch_yaml(c['output'])))


def _watch_stamp(port, token, path):
    """The time of the last save of the notebook, without its content."""
    try:
        return _api(port, token, '/api/contents/' + urllib.parse.quote(path) + '?content=0')['last_modified']
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _watch_read(port, token, path, lines=3, traceback=False):
    """The notebook as a list of cells, each a dict with the id, the type,
    the source, the execution count and a text of the outputs, with at most
    `lines` lines of each text output, or all of them for None, and with
    the traceback of each error if `traceback` is true."""
    try:
        nb = _api(port, token, '/api/contents/' + urllib.parse.quote(path) + '?content=1')
    except (OSError, ValueError):
        return None

    return [{
        'id': c.get('id', ''),
        'type': c['cell_type'],
        'source': ''.join(c['source']) if isinstance(c['source'], list) else c['source'],
        'count': c.get('execution_count'),
        'output': _watch_outputs(c.get('outputs', []), lines, traceback),
    } for c in nb['content']['cells']]


def _watch_outputs(outputs, lines=3, traceback=False):
    """The outputs of a cell as a YAML list, one item for each output, with
    its kind as the key: `stdout`, `stderr`, `result`, `display` or
    `error`. Text is clipped to `lines` lines, or not for None, and an image
    or a widget is one word. An error has its traceback if `traceback` is
    true."""
    out = []

    for o in outputs:
        kind = o.get('output_type')

        if kind == 'error':
            text = '%s: %s' % (o.get('ename', ''), o.get('evalue', ''))
            if traceback:
                #
                # The traceback has the color codes of the terminal: take
                # them out, so that it reads as plain text.
                #
                text += '\n' + re.sub(r'\x1b\[[0-9;]*m', '', '\n'.join(o.get('traceback', [])))
            out += _yaml_item('error', text)
            continue

        if kind == 'stream':
            label, text = o.get('name', 'stdout'), o.get('text', '')
        else:
            label, data = 'result' if kind == 'execute_result' else 'display', o.get('data', {})
            if any(k.startswith('image/') for k in data):
                out += _yaml_item(label, '[image]')
                continue
            if 'application/vnd.jupyter.widget-view+json' in data:
                out += _yaml_item(label, '[widget]')
                continue
            text = data.get('text/plain', '')

        text = ''.join(text) if isinstance(text, list) else text
        rows = text.rstrip('\n').split('\n')
        if lines is not None and len(rows) > lines:
            rows = rows[:lines] + ['... %d more lines' % (len(rows) - lines)]
        out += _yaml_item(label, '\n'.join(rows))

    return '\n'.join(out)


def _yaml_item(key, text):
    """The lines of one item of a YAML list, `- key: text`, with a block
    scalar for text of more than one line, and quotes for text that YAML
    could read as something else."""
    if '\n' in text:
        return ['- %s: |' % key] + ['    ' + l if l else '' for l in text.split('\n')]

    plain = re.fullmatch(r"[A-Za-z_(][^:#]*", text) and not re.fullmatch(r'(?i)true|false|null|yes|no|on|off|~', text)
    return ['- %s: %s' % (key, text if plain and not text.endswith(' ') else json.dumps(text))]


#
# The most cells that ran between two reports and still show whole. More
# is a run of many cells, such as Run All, and its report folds.
#
WHOLE = 3


def _watch_report(old, new, settled=True):
    """The changes from the cells `old` to the cells `new`, as text.

    The two lists are aligned as sequences, as diff aligns lines, so an
    inserted or deleted cell shows as that one cell, and not as a change of
    every cell after it. With cell ids, a cell that moved shows as moved.
    The report follows the order of the notebook. When up to `WHOLE`
    cells ran, each shows whole, its source and its outputs: the person ran
    them one by one, and works on them. When more ran, as with Run All,
    cells in a row that ran with no output, and no other change, go into
    one line, so that the report stays short.
    """
    pairs = []
    added = []
    deleted = []
    opcodes, by_id = _watch_align(old, new)

    for tag, i1, i2, j1, j2 in opcodes:
        #
        # Without ids, a 'replace' of cells by cells is an edit of those
        # cells. With ids, it is other cells: added, deleted or moved.
        #
        p = list(zip(range(i1, i2), range(j1, j2))) if tag == 'equal' or not by_id else []
        pairs += [(i, j) for i, j in p if old[i] != new[j]]
        added += range(j1 + len(p), j2)
        deleted += [(i, j1) for i in range(i1 + len(p), i2)]

    gone = {old[i]['id']: (i, at) for i, at in deleted if by_id}
    moved = {j: gone.pop(new[j]['id'])[0] for j in added if by_id and new[j]['id'] in gone}
    deleted = list(gone.values()) if by_id else deleted

    ran = [j for i, j in pairs if new[j]['count'] is not None and new[j]['count'] != old[i]['count']]
    ran += [j for j in added if new[j]['count'] is not None]
    whole = set(ran) if len(ran) <= WHOLE else set()

    entries = []

    for i, j in pairs:
        o, c = old[i], new[j]
        if j in whole:
            entries.append(((j, 1), _watch_whole(j, c)))
        elif j in ran and o['source'] == c['source'] and o['type'] == c['type'] and not c['output']:
            entries.append(((j, 1), j))
        else:
            entries.append(((j, 1), _watch_changed(j, o, c)))

    for j in added:
        c = new[j]
        extra = 'moved from %d' % moved[j] if j in moved else 'new'
        if j in whole:
            entries.append(((j, 1), _watch_whole(j, c, extra)))
        elif j in moved:
            entries.append(((j, 1), _watch_changed(j, old[moved[j]], c, moved=moved[j])))
        else:
            entries.append(((j, 1), _watch_new(j, c)))

    for i, at in deleted:
        entries.append(((at, 0), _watch_deleted(i, old[i])))

    #
    # A blank line before each block, so that the lines of one cell do not
    # read as part of the next.
    #
    rows = []
    quiet = []

    for key, e in sorted(entries, key=lambda x: x[0]):
        if isinstance(e, int):
            quiet.append(e)
            continue

        rows += _watch_quiet(quiet) + [''] + e
        quiet = []

    rows += _watch_quiet(quiet)

    n = len(entries)
    head = 'changed: %d cell%s%s.' % (n, '' if n == 1 else 's', '' if settled else ', still changing')
    return '\n'.join([head] + rows)


def _watch_whole(j, c, extra=''):
    """The block of a cell with its whole source, and its outputs."""
    state = ', '.join(x for x in (c['type'], extra, 'ran, count %s' % c['count']) if x)
    lines = ['%s (%s):' % (_watch_name(j, c), state)]

    if '\n' in c['source']:
        lines += ['source: |'] + _watch_clip(['  ' + l if l else '' for l in c['source'].split('\n')], 30)
    else:
        lines += [_yaml_item('source', c['source'])[0][2:]]

    lines += _watch_yaml(c['output'])
    return _watch_trimmed(lines, j, c)


def _watch_quiet(ran):
    """The line for cells in a row that ran with no output, or nothing."""
    if not ran:
        return []

    return ['', '%s %s ran, with no output.' % ('cell' if len(ran) == 1 else 'cells', _watch_ranges(ran))]


def _watch_ranges(numbers):
    """Sorted numbers as short ranges, for example '3, 5-9, 12'."""
    out = []

    for n in sorted(numbers):
        if out and out[-1][1] == n - 1:
            out[-1][1] = n
        else:
            out.append([n, n])

    return ', '.join('%d' % a if a == b else '%d-%d' % (a, b) for a, b in out)


def _watch_align(old, new):
    """The opcodes of difflib that turn the cells `old` into `new`, and
    whether the cells were matched by their ids.

    Cells are matched by their ids when every cell has one, and the ids are
    unique. A notebook of nbformat 4.4, such as most of the examples, has no
    ids: then cells are matched by their type and source, and an edited cell
    is a 'replace' of one cell by one cell.
    """
    ids = [c['id'] for c in old + new]

    by_id = all(ids) and len(set(c['id'] for c in old)) == len(old) and len(set(c['id'] for c in new)) == len(new)
    key = (lambda c: c['id']) if by_id else (lambda c: (c['type'], c['source']))

    m = difflib.SequenceMatcher(None, [key(c) for c in old], [key(c) for c in new], autojunk=False)
    return m.get_opcodes(), by_id


def _watch_changed(j, o, c, moved=None):
    if c['count'] is None and o['count'] is not None and c['type'] == 'code':
        state = ', running'
    elif c['count'] != o['count'] and c['count'] is not None:
        state = ', ran, count %s' % c['count']
    else:
        state = ''

    if moved is not None:
        state += ', moved from %d' % moved

    lines = ['%s (%s%s):' % (_watch_name(j, c), c['type'], state)]

    if c['source'] != o['source']:
        #
        # An empty source has no lines, not one empty line, so that a cell
        # that was empty shows only the new lines.
        #
        a = o['source'].split('\n') if o['source'] else []
        b = c['source'].split('\n') if c['source'] else []
        diff = difflib.unified_diff(a, b, lineterm='', n=0)
        lines += _watch_clip([l[0] + ' ' + l[1:] for l in diff if l[:1] in '+-' and l[:3] not in ('+++', '---')])

    #
    # The outputs show when they changed, or when the cell ran; an edit
    # alone leaves the old outputs, so they do not show.
    #
    if c['output'] != o['output'] or (c['count'] != o['count'] and c['count'] is not None):
        lines += _watch_yaml(c['output'])

    return _watch_trimmed(lines, j, c)


def _watch_error(c):
    """True if the outputs of the cell `c` have an error."""
    return any(l.startswith('- error: ') for l in c['output'].split('\n'))


def _watch_yaml(output):
    """The outputs of a cell, from _watch_outputs(), under the key
    `outputs`, or `outputs: []` for none."""
    return ['outputs:'] + ['  ' + l if l else '' for l in output.split('\n')] if output else ['outputs: []']


def _watch_new(j, c):
    ran = ', ran, count %s' % c['count'] if c['count'] is not None else ''
    lines = ['%s (%s, new%s):' % (_watch_name(j, c), c['type'], ran)]
    lines += _watch_clip(['+ ' + l for l in c['source'].split('\n')])
    lines += _watch_yaml(c['output']) if c['output'] else []
    return _watch_trimmed(lines, j, c)


def _watch_deleted(i, o):
    return ['%s (%s, deleted):' % (_watch_name(i, o), o['type'])] + ['- ' + l for l in o['source'].split('\n')[:3]]


def _watch_clip(lines, limit=10):
    """At most `limit` lines of a source, and a line that says how many
    more there are."""
    return lines if len(lines) <= limit else lines[:limit] + ['... %d more lines' % (len(lines) - limit)]


def _watch_trimmed(lines, index, c):
    """The lines of a cell's block, and a last line that tells how to read
    the whole cell, if a part of the block was clipped."""
    if not any(l.lstrip().startswith('... ') and l.endswith(' more lines') for l in lines):
        return lines

    return lines + ['(trimmed: `cell %s` prints the whole cell)' % (c['id'] or index)]


def _watch_name(index, c):
    """'cell 3', with the id of the cell when it has one: 'cell 3, id 1a2b'."""
    return 'cell %d, id %s' % (index, c['id']) if c['id'] else 'cell %d' % index


def _watch_load(state):
    """The cells of the last call, whether it reported a running cell, and
    the time of the first call, or None."""
    try:
        with open(state) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None, False, None

    if isinstance(data, list):
        return data, False, None

    return data['cells'], data['running'], data.get('started')


def _watch_save(state, cells, started, running=False):
    with open(state, 'w') as f:
        json.dump({'cells': cells, 'running': running, 'started': started}, f)


def _watch_elapsed(started):
    """'watching for 1 min 35 s.': the time since the first call."""
    m, sec = divmod(int(time.time() - started), 60)
    return 'watching for %s.' % ('%d min %d s' % (m, sec) if m else '%d s' % sec)


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


def _jupylet_version(env):
    """Find the version of jupylet installed in an environment.

    Runs _PROBE with the environment's own python, from inside its folder,
    as find_env() does.

    Returns:
        str: The version, or None if the environment has no jupylet, or
            cannot be told.
    """
    python = env and _python_of(env)

    if not python:
        return None

    try:
        out = subprocess.run(
            [python, '-c', _PROBE],
            capture_output=True, text=True, timeout=20, cwd=env,
        )
        return json.loads(out.stdout.strip().splitlines()[-1])[0]
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


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
        list: (port, token, folder served, the notebooks open in it, the
            environment it runs from, the version of jupylet in that
            environment, whether it answers with its MCP tools, whether
            nbmodel is off in it, the number of connections to each notebook's
            kernel) tuples. main() prints them
            as JSON: see USAGE.

    Notes:
        Records can outlive their servers, and another Jupyter may have taken
        their port since, hence both checks. Ports are probed all at once:
        a refused connection takes seconds on Windows.
        A kernel's connections are the pages, and other clients, connected to
        it: 0 means no page shows that notebook now, for example after the
        person closed the browser tab. Not tested yet: whether Claude's own
        attach counts as one.
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

        connections = {}

        for s in sessions:
            if s.get('type') == 'notebook':
                n = (s.get('kernel') or {}).get('connections', 0)
                connections[s['path']] = max(n, connections.get(s['path'], 0))

        notebooks = sorted(connections)
        seen.add((port, token))
        env = _env_of(info.get('pid', 0))
        found.append((port, token, root, notebooks, env, _jupylet_version(env),
                      _mcp_tools(port, token), _nbmodel_off_in(port, token),
                      [connections[n] for n in notebooks]))

    return found


def _mcp_tools(port, token):
    """Tell whether a running Jupyter answers with its MCP tools.

    Returns:
        bool: True if it lists its tools: the [claude] extra is installed and
            loaded in it.
    """
    try:
        return bool(tools(port, token))
    except (OSError, ValueError, KeyError, AttributeError):
        return False


def _nbmodel_off_in(port, token):
    """Tell whether jupyter_server_nbmodel is off in a running Jupyter.

    Asks the running server itself, not the configuration on disk, which can
    have changed since the server started.

    Returns:
        bool: True if both its parts are off.

    Notes:
        Its server part adds /api/nbmodel/settings/output-recovery, so that
        address answers 404 when the part is off. Its browser part is listed
        in the page configuration of /lab: it is off when it is not among
        federated_extensions, or when it is among disabledExtensions.
    """
    try:
        _api(port, token, '/api/nbmodel/settings/output-recovery')
        return False
    except urllib.error.HTTPError as e:
        if e.code != 404:
            return False
    except (OSError, ValueError):
        return False

    req = urllib.request.Request(
        'http://localhost:%s/lab' % port,
        headers={'Authorization': 'token ' + token},
    )

    try:
        html = urllib.request.urlopen(req, timeout=60).read().decode()
        page = json.loads(re.search(r'id="jupyter-config-data"[^>]*>(.*?)</script>', html, re.S).group(1))
    except (OSError, ValueError, AttributeError):
        return False

    names = [e.get('name') for e in page.get('federated_extensions', [])]

    return NBMODEL_LAB not in names or NBMODEL_LAB in page.get('disabledExtensions', [])


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
    src, folder = _news_paths()

    if os.path.exists(src) and not os.path.exists(os.path.join(folder, 'overrides.json')):
        os.makedirs(folder, exist_ok=True)
        shutil.copy(src, folder)


def _news_paths():
    """The news settings file that Jupylet ships, and the folder it goes to."""
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'assets', 'jupyterlab', 'overrides.json')
    folder = os.path.join(sys.prefix, 'share', 'jupyter', 'lab', 'settings')

    return src, folder


def _server_config_path():
    """This Python's environment's own jupyter_server_config.json."""
    return os.path.join(sys.prefix, 'etc', 'jupyter', 'jupyter_server_config.json')


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
    path = _server_config_path()

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


def prepare_needs():
    """List the parts of prepare() that would change something.

    Changes nothing, so it can run before the person agrees to the change.

    Returns:
        list: Names of the parts that are not set yet: 'nbmodel', 'live copy'
            and 'news'. Empty if the environment is ready.
    """
    import importlib.util

    needs = []

    if importlib.util.find_spec(NBMODEL) is not None and _nbmodel_on() != (False, False):
        needs.append('nbmodel')

    path = _server_config_path()
    ydoc = {}

    if os.path.exists(path):
        with open(path) as f:
            ydoc = json.load(f).get('YDocExtension', {})

    if 'ystore_class' not in ydoc or 'document_cleanup_delay' not in ydoc:
        needs.append('live copy')

    src, folder = _news_paths()

    if os.path.exists(src) and not os.path.exists(os.path.join(folder, 'overrides.json')):
        needs.append('news')

    return needs


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
        port = wait(*args)
        print('ready %s' % port if port else 'not ready after 60 seconds')
        return 0 if port else 1

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

    elif cmd == 'run-all' and len(args) == 4:
        print(_plain('run-all', *run_all(*args[:3], float(args[3])), args[3]))

    elif cmd == 'wait-change' and len(args) in (4, 5):
        print(_plain('wait-change', *wait_change(*args[:3], float(args[3]), *args[4:]), args[3]))

    elif cmd == 'watch' and len(args) == 5:
        print(watch(*args[:3], float(args[3]), args[4]))

    elif cmd == 'cell' and len(args) == 4:
        print(cell_text(*args))

    elif cmd == 'replace-kernel' and len(args) == 3:
        print(replace_kernel(*args))

    elif cmd == 'run-cell' and len(args) == 4:
        print(run_cell(*args))

    elif cmd == 'shutdown' and len(args) == 2:
        print(shutdown(*args))

    elif cmd == 'running' and len(args) <= 1:
        print(json.dumps([{
            'port': port,
            'token': token,
            'folder': root,
            'environment': env or None,
            'jupylet_version': version,
            'mcp_tools': mcp,
            'nbmodel_off': nbmodel_off,
            'notebooks': [{'path': n, 'connections': c} for n, c in zip(notebooks, connections)],
        } for port, token, root, notebooks, env, version, mcp, nbmodel_off, connections
            in running(*args)], indent=2))

    elif cmd == 'cleanup' and args in ([], ['--yes']):
        cleanup(delete=args == ['--yes'])

    elif cmd == 'prepare' and args == ['--check']:
        needs = prepare_needs()
        print('needs: ' + ', '.join(needs) if needs else 'ready')

    elif cmd == 'prepare' and not args:
        result = prepare()
        print(result)
        return 1 if result.startswith('still on') else 0

    else:
        print(USAGE)
        return 1


_PLAIN = {
    'run-all': {
        'ran': 'finished: the run is over. Fingerprint: %(v)s',
        'timeout': 'still running: the run is not over after %(t)s seconds.'
                   ' Fingerprint: %(v)s',
        'error': 'not started: the page did not start the run. It said: %(v)s',
    },
    'wait-change': {
        'ran': 'finished: the kernel ran something, and is idle again.'
               ' Fingerprint: %(v)s',
        'timeout': 'not finished: no run ended in %(t)s seconds. Fingerprint: %(v)s',
    },
}


def _plain(cmd, kind, value, seconds):
    """The answer of run_all() or wait_change() as a keyword and a sentence."""
    return _PLAIN[cmd][kind] % {'v': value, 't': seconds}


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
