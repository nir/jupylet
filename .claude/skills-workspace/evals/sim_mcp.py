"""An MCP server whose tools are answered by the eval harness.

The worker's CLI starts this server (one process for each server name). For
each tool call, the server sends the tool and its input to the harness over
the Unix socket in SIM_SOCKET, and returns the harness's answer as the
tool's result. Nothing runs on the real computer.

Environment:
    SIM_SOCKET  the harness's socket.
    SIM_SERVER  the server name: computer, browser, ccd_window or
                ccd_directory. It selects the tools. The browser server has
                the tools of the browser pane of the Claude app. It cannot
                have the name Claude_Browser: the CLI keeps that name for the
                real pane, and drops another server with it.
"""

import json
import os
import socket
import sys


def obj(props, required=()):

    return {'type': 'object', 'properties': props, 'required': list(required)}


S = {'type': 'string'}
B = {'type': 'boolean'}
N = {'type': 'number'}

TOOLS = {
    'computer': {
        'Bash': ('Runs a shell command (zsh on macOS) on the computer, and returns its output. '
                 'With run_in_background, it returns an ID at once, and a task-notification '
                 'arrives when the command ends.',
                 obj({'command': S, 'description': S, 'timeout': N, 'run_in_background': B}, ['command'])),
        'Read': ('Reads a file and returns its lines with line numbers.',
                 obj({'file_path': S, 'offset': N, 'limit': N}, ['file_path'])),
        'Write': ('Writes a file, and replaces it if it exists.',
                  obj({'file_path': S, 'content': S}, ['file_path', 'content'])),
        'Edit': ('Replaces exact text in a file.',
                 obj({'file_path': S, 'old_string': S, 'new_string': S, 'replace_all': B},
                     ['file_path', 'old_string', 'new_string'])),
    },
    'browser': {
        'tabs_context': ('Lists the tabs of the browser pane of the Claude app.', obj({})),
        'preview_start': ('Opens an address in a new tab of the browser pane.', obj({'url': S}, ['url'])),
        'navigate': ('Opens an address in a tab of the browser pane.', obj({'url': S, 'tabId': S}, ['url'])),
        'javascript_tool': ('Runs JavaScript in the page of a tab, and returns its value.',
                            obj({'action': S, 'text': S, 'tabId': S}, ['text'])),
        'computer': ('Mouse, keyboard and screenshots in the browser pane.',
                     obj({'action': S, 'coordinate': {'type': 'array'}, 'text': S}, ['action'])),
        'tabs_close': ('Closes a tab of the browser pane.', obj({'tabId': S}, ['tabId'])),
    },
    'ccd_window': {
        'set_sidebar_collapsed': ('Hides or shows the list of chats of the Claude app.',
                                  obj({'collapsed': B}, ['collapsed'])),
    },
    'ccd_directory': {
        'change_directory': ('Moves the session to another folder when the turn ends.',
                             obj({'path': S}, ['path'])),
    },
}


#
# On Windows, the shell tool is PowerShell, as in the Claude app (run W1).
#
if os.environ.get('SIM_SHELL') == 'PowerShell':
    TOOLS['computer'] = dict(
        [('PowerShell', ('Runs a PowerShell command on the computer (Windows 11), and returns its output. '
                         'With run_in_background, it returns an ID at once, and a task-notification '
                         'arrives when the command ends.',
                         TOOLS['computer']['Bash'][1]))]
        + [(k, v) for k, v in TOOLS['computer'].items() if k != 'Bash'])


def ask_harness(tool, args):

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(os.environ['SIM_SOCKET'])
    s.sendall((json.dumps({'server': os.environ['SIM_SERVER'], 'tool': tool, 'input': args}) + '\n').encode())

    data = b''
    while not data.endswith(b'\n'):
        chunk = s.recv(65536)
        if not chunk:
            break
        data += chunk
    s.close()

    return json.loads(data.decode())


def answer(msg):

    method = msg.get('method')
    server = os.environ['SIM_SERVER']

    if method == 'initialize':
        return {'protocolVersion': msg['params'].get('protocolVersion', '2025-06-18'),
                'capabilities': {'tools': {}},
                'serverInfo': {'name': server, 'version': '1'}}

    if method == 'tools/list':
        return {'tools': [{'name': n, 'description': d, 'inputSchema': schema}
                          for n, (d, schema) in TOOLS[server].items()]}

    if method == 'tools/call':
        p = msg['params']
        r = ask_harness(p['name'], p.get('arguments', {}))
        return {'content': [{'type': 'text', 'text': r.get('output', '')}],
                'isError': bool(r.get('error'))}

    if method == 'ping':
        return {}

    return None


def main():

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if 'id' not in msg:
            continue

        try:
            result = answer(msg)
            if result is None:
                reply = {'jsonrpc': '2.0', 'id': msg['id'],
                         'error': {'code': -32601, 'message': 'unknown method %s' % msg.get('method')}}
            else:
                reply = {'jsonrpc': '2.0', 'id': msg['id'], 'result': result}
        except Exception as e:
            reply = {'jsonrpc': '2.0', 'id': msg['id'], 'error': {'code': -32000, 'message': str(e)}}

        sys.stdout.write(json.dumps(reply) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
