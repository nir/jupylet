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


import ipywidgets
import functools
import traceback
import hashlib
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
    """ Custom logging handler sending logs to an output widget """

    def __init__(self, height='256px', *args, **kwargs):    
        super(LoggingWidget, self).__init__(*args, **kwargs)
        
        self.out = ipywidgets.Output()
        self.set_layout(height)

    def set_layout(self, height='256px', overflow='scroll', **kwargs):
        self.out.layout=ipywidgets.Layout(
            height=height, 
            overflow=overflow, 
            **kwargs
        )

    def emit(self, record):
        with self.out:
            print(self.format(record))


def get_logging_widget(height='256px', quiet_default_logger=True):

    if type(height) is int:
        height = str(height) + 'px'

    logger = logging.getLogger()

    wl = [h for h in logger.handlers if isinstance(h, LoggingWidget)]
    if wl:
        w = wl[-1]
        w.set_layout(height)
        return w.out

    handler = LoggingWidget(height)
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

