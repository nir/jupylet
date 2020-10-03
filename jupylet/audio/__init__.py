"""
    jupylet/audio/__init__.py
    
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


import asyncio
import time

from ..utils import callerframe


FPS = 44100


dtd = {}
syd = {}


def use(synth, **kwargs):

    if kwargs:
        synth = synth.copy().set(**kwargs)

    cf = callerframe()
    cn = cf.f_code.co_name
    hh = cn if cn == '<module>' else hash(cf) 

    syd[hh] = synth


def play(note, **kwargs):

    cf = callerframe()
    cn = cf.f_code.co_name
    hh = cn if cn == '<module>' else hash(cf) 

    sy = syd[hh]

    return sy.play_new(note, **kwargs)


def sleep(dt=0):
    
    tt = time.time()

    cf = callerframe()
    cn = cf.f_code.co_name
    hh = cn if cn == '<module>' else hash(cf) 

    t0 = dtd.get(hh) or tt
    t1 = dtd[hh] = max(t0 + dt, tt)

    return asyncio.sleep(t1 - tt)

