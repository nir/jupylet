"""
    jupylet/rl.py
    
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


import importlib
import traceback
import datetime
import tempfile
import hashlib
import random
import pickle
import sys
import os

import multiprocessing as mp
import numpy as np


def o2h(o, n=12):
    return hashlib.sha256(pickle.dumps(o)).hexdigest()[:n]


def get_tmp_dir():
    
    if os.path.exists('/dev/shm/'):
        return '/dev/shm'
    else:
        return tempfile.gettempdir()

def get_random_mm_path():
    return os.path.join(get_tmp_dir(), 'jupylet-mm-array-%s.tmp' % o2h(random.random()))


def rgetattr(o, name):
    
    nl = name.split('.')
    for n in nl:
        o = getattr(o, n)
        
    return o


def rsetattr(o, name, value):
    
    nn = name.rsplit('.', 1)
    
    if len(nn) > 1:
        o = rgetattr(o, nn[0])
        name = nn[1]
        
    setattr(o, name, value)


class ModuleProcess(object):
    
    def __init__(self, name, debug=False):
        
        self.name = name
        self.path = get_random_mm_path()
        self.debug = debug

        self.c0, c1 = mp.Pipe()
        self.c1 = None
        self.p0 = mp.Process(target=self._worker, args=(c1,))

        self.mm = None
        
    def __del__(self):
        self.stop()
        
    def _worker(self, c1):
        
        self.c0 = None
        self.c1 = c1

        module = importlib.import_module(self.name)
        
        for name, args, kwargs in iter(self.c1.recv, 'STOP'):            
            
            try:
                if kwargs is not None:
                    foo = rgetattr(module, name)
                    r = foo(*args, **kwargs)
                    self._c1_send(r)
                    
                elif args is not None:
                    rsetattr(module, name, args)

                else:
                    self._c1_send(rgetattr(module, name))
             
            except:
                self.c1.send(('ee', traceback.format_exception(*sys.exc_info())))
                
        if self.mm is not None:
            del self.mm
            self.mm = None
            os.remove(self.path)
    
    """def log(self, msg, *args):
        if self.debug:
            msg = msg % args
            date = datetime.datetime.now().isoformat()[:23].replace('T', ' ')
            self.q3.put('[%s] [%s]  %s' % (date, self.p0.ident, msg))"""

    def start(self):
        if self.p0.ident is not None:
            return

        assert 'pyglet' not in sys.modules, 'Avoid importing pyglet or jupylet modules except rl.py before starting worker processes.'

        return self.p0.start()
    
    def get(self, name):
        self.send(name, None, None)
        return self.recv()

    def set(self, name, value):
        self.send(name, value, None)
        
    def call(self, name, *args, **kwargs):
        self.send(name, args, kwargs)
        return self.recv()
    
    def send(self, name, args, kwargs):
        self.c0.send((name, args, kwargs))
    
    def recv(self):
        return self._c0_recv()
    
    def stop(self):
        if self.p0.ident is None or not self.p0.is_alive():
            return

        try:
            self.c0.send('STOP')
            self.p0.join()
        except AssertionError:
            pass

    def _c0_recv(self, timeout=5.):
        
        t, v = self.c0.recv()#timeout=timeout)
        
        if t == 'vv':
            return v
        
        if t == 'mm':
            path, dtype, shape = v
            return np.memmap(path, dtype=dtype, mode='r', shape=shape)
        
        if t == 'ee':
            sys.stderr.write('\n'.join(v))
        
    def _c1_send(self, v):
        
        if isinstance(v, np.ndarray):
            self.c1.send(('mm', self._mma(v)))
        else:
            self.c1.send(('vv', v))
        
    def _mma(self, v):
        
        if self.mm is None or self.mm.dtype != v.dtype or self.mm.shape != v.shape:
            self.mm = np.memmap(self.path, dtype=v.dtype, mode='w+', shape=v.shape)
            
        self.mm[:] = v[:]
        self.mm.flush()
        
        return self.path, v.dtype, v.shape

            
class GameProcess(ModuleProcess):
    
    def start(self, size=224):

        super(GameProcess, self).start()
        
        self.call('app.start')
        self.call('app.scale_window_to', size)
        
    def step(self):
        return self.call('app.step')


class Games(object):
    
    def __init__(self, games):
        
        if type(games[0]) is str:
            self.games = [ModuleProcess(game) for game in games]
        else:
            self.games = games
        
    def start(self, size=224):
        
        for g in self.games:
            if type(g) is GameProcess:
                super(GameProcess, g).start()
            else:
                g.start()
                
        self.call('app.start')
        self.call('app.scale_window_to', size)
        
    def step(self):
        return self.call('app.step') 
    
    def get(self, name):
        self.send(name, None, None)
        return self.recv()

    def set(self, name, value):
        self.send(name, value, None)
        
    def call(self, name, *args, **kwargs):
        self.send(name, args, kwargs)
        return self.recv()

    def send(self, name, args, kwargs):
        for g in self.games:
            g.send(name, args, kwargs)
    
    def recv(self):
        return [g.recv() for g in self.games]
    
