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

    python -m jupylet.claude wait <port> <token>
    python -m jupylet.claude attach <port> <token> <notebook path>
    python -m jupylet.claude kernel <port> <token> <notebook path>
    python -m jupylet.claude tools <port> <token>
    python -m jupylet.claude call <port> <token> <tool> ['<json arguments>']
    python -m jupylet.claude replace-kernel <port> <token> <notebook path>
    python -m jupylet.claude shutdown <port> <token>
    python -m jupylet.claude cleanup [--yes]
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
    """Ids of the processes started with this Jupyter token (macOS, Linux)."""
    try:
        out = subprocess.run(
            ['ps', '-axo', 'pid=,command='], capture_output=True, text=True
        ).stdout
    except OSError:
        return []

    pids = [int(line.split(None, 1)[0]) for line in out.splitlines()
            if 'IdentityProvider.token=' + token in line]

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

    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in _pids(token):
            try:
                os.kill(pid, sig)
            except OSError:
                pass

        if _wait_until(lambda: not _pids(token), 5):
            return 'stopped after ending its process'

    return 'still running'


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


def main(argv):
    cmd, args = (argv[0], argv[1:]) if argv else ('', [])

    if cmd == 'wait' and len(args) == 2:
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
        print(call(*args[:3], json.loads(args[3]) if len(args) == 4 else None))

    elif cmd == 'replace-kernel' and len(args) == 3:
        print(replace_kernel(*args))

    elif cmd == 'shutdown' and len(args) == 2:
        print(shutdown(*args))

    elif cmd == 'cleanup' and args in ([], ['--yes']):
        cleanup(delete=args == ['--yes'])

    else:
        print(USAGE)
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
