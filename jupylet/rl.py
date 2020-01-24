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
    
    def __init__(self, name):
        
        self.name = name
        self.path = get_random_mm_path()

        self.q1 = mp.Queue()
        self.q2 = mp.Queue()
        self.p0 = mp.Process(target=self._worker, args=(self.q1, self.q2))
        self.mm = None
        
    def __dell__(self):
        self.stop()
        
    def _worker(self, q1, q2):
        
        module = importlib.import_module(self.name)
        
        for name, args, kwargs in iter(self.q1.get, 'STOP'):
            
            try:
                if kwargs is not None:
                    foo = rgetattr(module, name)
                    self._q2_put(foo(*args, **kwargs))

                elif args is not None:
                    rsetattr(module, name, args)

                else:
                    self._q2_put(rgetattr(module, name))
             
            except Exception as e:
                self.q2.put(('ee', traceback.format_exception(*sys.exc_info())))
                
        if self.mm is not None:
            del self.mm
            self.mm = None
        
    def start(self):
        if self.p0.ident is None:
            return self.p0.start()
    
    def get(self, name):
        self.q1.put((name, None, None))
        return self._q2_get()

    def set(self, name, value):
        self.q1.put((name, value, None))
        
    def call(self, name, *args, **kwargs):
        self.q1.put((name, args, kwargs))
        return self._q2_get()
    
    def stop(self):
        if self.p0.ident is not None:
            self.q1.put('STOP')
            self.p0.join()
        
    def _q2_get(self, timeout=5.):
        
        t, v = self.q2.get(timeout=timeout)
        
        if t == 'vv':
            return v
        
        if t == 'mm':
            path, dtype, shape = v
            return np.memmap(path, dtype=dtype, mode='r', shape=shape)
        
        if t == 'ee':
            sys.stderr.write('\n'.join(v))
        
    def _q2_put(self, v):
        
        if isinstance(v, np.ndarray):
            self.q2.put(('mm', self._mma(v)))
        else:
            self.q2.put(('vv', v))
        
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

