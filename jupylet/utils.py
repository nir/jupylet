"""
    jupylet/utils.py
    
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


import collections
import ipywidgets
import ipyevents
import functools
import traceback
import hashlib
import html
import inspect
import logging
import pickle
import types
import glm
import sys
import re
import os

import numpy as np


LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class StreamHandler(logging.StreamHandler):
    pass


class LoggingWidget(logging.Handler):
    """Logging handler that shows the last log messages in a widget.

    The messages are shown in a scrolling box that stays scrolled to the
    newest message. While the mouse is over the box, the box is not updated,
    so it can be scrolled back through without jumping; the messages that
    arrive meanwhile are shown once the mouse leaves.

    Args:
        height (str): Height of the box, as a CSS length.
        max_lines (int): Number of most recent messages to keep.
    """

    def __init__(self, height='256px', max_lines=320, *args, **kwargs):
        super(LoggingWidget, self).__init__(*args, **kwargs)

        self.lines = collections.deque(maxlen=max_lines)
        self.height = height
        self.hover = False

        #
        # With _view_count set to a number, the frontend keeps it up to date
        # with the number of places the widget is displayed in. The widget is
        # only redrawn while it is displayed somewhere, and once when it is
        # displayed again.
        #
        self.out = ipywidgets.HTML(_view_count=0)
        self.out.observe(lambda change: change['new'] and self.show(), names='_view_count')

        self._event = ipyevents.Event(
            source=self.out,
            watched_events=['mouseenter', 'mouseleave']
        )
        self._event.on_dom_event(self._on_hover)

        self.show()

    def set_layout(self, height='256px'):
        self.height = height
        self.show()

    def show(self):
        """Show the kept messages in the widget."""

        #
        # A column-reverse flex box starts out scrolled to its end, so the
        # box is scrolled to the newest message each time it is redrawn.
        #
        self.out.value = (
            '<div style="height: %s; overflow: auto; display: flex; flex-direction: column-reverse;">'
            '<pre style="margin: 0; flex-shrink: 0; font-size: var(--jp-code-font-size); line-height: normal;">%s</pre>'
            '</div>'
        ) % (self.height, html.escape(''.join(self.lines)))

    def _on_hover(self, event):

        self.hover = event['type'] == 'mouseenter'

        if not self.hover:
            self.show()

    def emit(self, record):

        self.lines.append(self.format(record) + '\n')

        if self.out._view_count and not self.hover:
            self.show()


def get_logging_widget(height='256px', quiet_default_logger=True, max_lines=320):

    if type(height) is int:
        height = str(height) + 'px'

    logger = logging.getLogger()

    wl = [h for h in logger.handlers if isinstance(h, LoggingWidget)]
    if wl:
        w = wl[-1]
        w.set_layout(height)
        return w.out

    handler = LoggingWidget(height, max_lines)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(LOGGING_FORMAT))
    
    logger.addHandler(handler)

    if quiet_default_logger:
        wl = [h for h in logger.handlers if isinstance(h, StreamHandler)]
        if wl:
            wl[-1].setLevel(logging.ERROR)

    return handler.out


_logging_level = logging.WARNING


def get_logging_level():
    return _logging_level

    
def setup_basic_logging(level):
    """Set up basic logging

    Args:
        level (int): The log level
    """
    
    global _logging_level

    if type(level) is str:
        level = logging._nameToLevel.get(level, None)
        
    if level is None:
        return

    _logging_level = level
    
    logger = logging.getLogger()
    logger.setLevel(level)

    if not logger.handlers:

        handler = StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(LOGGING_FORMAT))

        logger.addHandler(handler)


def abspath(path):

    dirname = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(dirname, path))


def callerpath(levelsup=1):

    ff = inspect.currentframe().f_back
    for i in range(levelsup):
        ff = ff.f_back

    pp = ff.f_globals.get('__file__', '')
    return os.path.dirname(pp)


def callerframe(levelsup=1):

    ff = inspect.currentframe().f_back
    for i in range(levelsup):
        ff = ff.f_back

    return ff


def auto_read(s):
    return s if '\n' in s else open(s).read()


def o2h(o, n=12):
    return hashlib.sha256(pickle.dumps(o)).hexdigest()[:n]


class Dict(dict):
    
    def __dir__(self):
        return list(self.keys()) + super().__dir__()

    def __getattr__(self, k):
        
        if k not in self:
            raise AttributeError(k)
            
        return self[k]
    
    def __setattr__(self, k, v):
        self[k] = v


def patch_method(obj, key, method):
    """Wrap obj's existing obj.key method with `method`, so obj.key(...)
    calls `method` instead - with the original method passed to it as the
    `foo` keyword, for it to call through to.

    `method` must therefore take a `foo` keyword, and be written to be
    bound as a method: its first parameter stands in for `self`, bound to
    `obj` (not to whatever class defines `method`) when called as
    obj.key(...). See app.py's `_clear` for an example.

    Idempotent: patching an already-patched obj.key again is a no-op (an
    already-wrapped method's __func__ is the functools.partial below, not
    a plain function - that's how it's detected), so it's safe to call
    this more than once on the same obj.key.
    """

    foo = getattr(obj, key)
    
    if isinstance(foo.__func__, functools.partial):
        return foo
    
    par = functools.partial(method, foo=foo)
    bar = types.MethodType(par, obj)
    bar.__func__.__name__ = foo.__func__.__name__
    
    setattr(obj, key, bar)
    
    return bar


def glm_dumps(o):
    
    if "'glm." not in repr(o.__class__):
        return o
    
    return ('__glm__', o.__class__.__name__, tuple(o))


def glm_loads(o):
    
    if type(o) is not tuple or not o or o[0] != '__glm__':
        return o
    
    return getattr(glm, o[1])(o[2])

    
def trimmed_traceback():
    
    e = ''.join(traceback.format_exception(*sys.exc_info()))
    e = re.sub(r'(?s)^.*?The above exception was the direct cause of the following exception:\s*', '', e)
    return e


def auto(o):
    
    t = type(o)
    
    if t in (tuple, list):
        return t(auto(v) for v in o)
    
    if t is dict:
        return {k: auto(v) for k, v in o.items()}
    
    if t is not str:
        return o
        
    if o.isdecimal():
        return int(o)
    
    try:
        return float(o)
    except:
        pass
    
    return o

    
def settable(o, name):
    
    if name[0] == '_':
        return False
    
    if name in o.__dict__:
        return True
    
    v = getattr(o, name, '__NONE__')
    
    return v != '__NONE__' and not callable(v)


def np_is_zero(a):
    return np.abs(a).sum().item() == 0


class Enum(object):
    
    def __init__(self, **kwargs):
        
        for k, v in kwargs.items():
            setattr(self, k, v)

