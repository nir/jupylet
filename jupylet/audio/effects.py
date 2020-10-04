"""
    jupylet/audio/effects.py
    
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
import logging

import scipy.signal
import scipy.fft

import soundfile as sf
import numpy as np

from ..resource import find_path
from ..utils import np_is_zero
from ..audio import FPS, t2frames

from .sound import Sound, ButterFilter


logger = logging.getLogger(__name__)
    

class ConvolutionReverb(Sound):
    
    def __init__(
        self, 
        path,
        compress=True,
        fidelity=256,
        buffer_size=1024,
    ):
        
        super(ConvolutionReverb, self).__init__()
        
        self.path = str(find_path(path))

        self.buffer_size = buffer_size
        self.compress = compress
        self.fidelity = fidelity
        
        self._ir, self._fs = load_impulse_response(self.path, self.compress, self.fidelity)
        self._ir = self._ir.astype('float32')
        
        self._gain0 = 1 / compute_impulse_gain(self._ir)
        self._buffi = None
        self._state = None
        self._buffo = None
        self._zeros = 0
        
    def forward(self, x):
        
        if self._buffo is None:
            
            nchannels = max(self._ir.shape[-1], x.shape[-1])
            
            self._buffi = np.zeros((0, x.shape[-1]))
            self._state = np.zeros((0, nchannels))
            self._buffo = np.zeros((self.buffer_size, nchannels))
            
        self._buffi = np.concatenate((self._buffi, x))
        
        bsize = self.buffer_size if self.buffer_size else len(x)

        if len(self._buffi) >= bsize:
            
            a0, self._buffi = self._buffi[:bsize], self._buffi[bsize:]
            
            if np_is_zero(a0):
                self._zeros += len(a0)
            else:
                self._zeros = 0
                
            if self._zeros < len(self._ir) + bsize:
                a0 = scipy.signal.fftconvolve(
                    a0.astype('float32'), 
                    self._ir, 
                    axes=0
                ).astype('float64')
                a0[:len(self._state)] += self._state
            
            a1, self._state = a0[:bsize], a0[bsize:]
            self._buffo = np.concatenate((self._buffo, a1))
        
        a2, self._buffo = self._buffo[:len(x)], self._buffo[len(x):]
        
        return a2 * self._gain0
        
        
@functools.lru_cache(maxsize=64)
def load_impulse_response(path, compress=True, fidelity=256):
    
    ir, fs = sf.read(path)

    if len(ir.shape) == 1:
        ir = ir[:,None]

    if compress:
        ir = ir * np.linspace(1, 0, len(ir))[:,None]

    mi0 = np.abs(ir).max()
    mi1 = (np.abs(ir) > (mi0 / fidelity)).astype('int32')
    
    trim = np.argwhere(mi1).max()
    
    ir = ir[:trim].astype('float32')
    
    return ir, fs


def compute_impulse_gain(impulse):
    
    ir = scipy.signal.fftconvolve(impulse, impulse, axes=0)

    e0 = (scipy.fft.rfft(impulse).real ** 2).sum()    
    e1 = (scipy.fft.rfft(ir).real ** 2).sum()
    
    return (e1 / e0) ** 0.5


class Delay(Sound):
    
    def __init__(
        self,
        dt=0.5,
        gain=0.5,
        freq=2048,
        btype='highpass'
    ):
    
        super(Delay, self).__init__()
        
        self.dt = dt
        self.gain = gain
        self.freq = freq
        
        self.filter = ButterFilter(freq, btype, db=12)
        
        self._buffer = None
        
    def forward(self, x):
        
        if self._buffer is None:
            self._buffer = np.zeros((0, x.shape[-1]))
        
        df = t2frames(self.dt)
        
        if len(self._buffer) <= df - len(x):
            self._buffer = np.concatenate((self._buffer, x))
            return x
                    
        dl = self._buffer[-df:-df+len(x)]
        
        a0 = x.copy()
        a0[-len(dl):] += dl * self.gain
        
        a1 = a0
        
        if self.freq and self.freq > 10:
            self.filter.freq = self.freq
            a1 = self.filter(a1)
        
        self._buffer = np.concatenate((self._buffer, a1))[-df:]

        return a0

