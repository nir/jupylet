"""
    jupylet/env.py

    Copyright (c) 2020, Nir Aides - nir@winpdb.org

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


import functools
import platform
import os
import sys

import multiprocessing as mp


@functools.lru_cache()
def is_remote():

    if is_binder_env():
        return True

    if is_aws_linux():
        return True


def is_aws_linux():

    if platform.system() == 'Linux':
        cmd = 'find /sys/devices/virtual/dmi/id/ -type f | xargs grep "Amazon EC2" 2> /dev/null'
        return 'Amazon' in os.popen(cmd).read()


def is_binder_env():
    return 'BINDER_REQUEST' in os.environ


_has_display = None


def has_display():

    global _has_display

    if _has_display is not None:
        return _has_display

    v = mp.Value('i', 0)

    if 'pyglet' in sys.modules:
        _has_display0(v)

    else:
        p = mp.Process(target=_has_display0, args=(v,))
        p.start()
        p.join()

    _has_display = v.value

    return _has_display


def _has_display0(v):

    try:
        import pyglet
        pyglet.canvas.get_display()
        v.value = 1
    except:
        pass


_xvfb = None


def start_xvfb():

    global _xvfb

    if platform.system() == 'Linux' and _xvfb is None:

        import xvfbwrapper
        _xvfb = xvfbwrapper.Xvfb()
        _xvfb.start()


def is_xvfb():
    return _xvfb is not None
