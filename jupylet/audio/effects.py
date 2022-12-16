"""
    jupylet/audio/effects.py
    
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


import functools
import logging
import math

import scipy.signal
import scipy.fft

import soundfile as sf
import numpy as np

from ..resource import find_path
from ..utils import np_is_zero, Enum
from ..audio import FPS, t2frames

from .filters import ButterFilter
from .sound import Sound


logger = logging.getLogger(__name__)


impulse = Enum(
    StAndrewsChurch='sounds/impulses/StAndrewsChurch.flac',
    InsidePiano='sounds/impulses/InsidePiano.flac',
    MaesHowe='sounds/impulses/MaesHowe.flac',
)
 

class ConvolutionReverb(Sound):
    
    def __init__(
        self, 
        path,
        compress=True,
        fidelity=256,
        buffer_size=0, 
        shared=False,
    ):
        
        super().__init__(shared=shared)
        
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
        
    def forward(self, x):
        
        if self._buffo is None:
            
            nchannels = max(self._ir.shape[-1], x.shape[-1])
            
            self._buffi = np.zeros((0, x.shape[-1]))
            self._state = np.zeros((0, nchannels))
            self._buffo = np.zeros((self.buffer_size, nchannels))
            
        self._buffi = np.concatenate((self._buffi, x))
        bsize = max(self.buffer_size, len(x))

        if len(self._buffi) >= bsize:
            
            a0, self._buffi = self._buffi, self._buffi[:0]
            
            if np_is_zero(a0):
                a1 = a0

            else:
                a1 = scipy.signal.fftconvolve(
                    a0.astype('float32'), 
                    self._ir, 
                    axes=0
                ).astype('float64')

            if len(a1) >= len(self._state):
                a1[:len(self._state)] += self._state
            else:
                self._state[:len(a1)] += a1
                a1 = self._state

            a1, self._state = a1[:len(a0)], a1[len(a0):]
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


class CombFilter(Sound):
    
    def __init__(
        self,
        delay=0.040,
        gain=0.5,
        rt=None, 
        shared=False,
    ):
    
        super().__init__(shared=shared)
        
        self.delay = delay
        self.gain = gain
        
        if rt is not None:
            self.rt = rt
 
    def forward(self, x):
        
        mm = t2frames(self.delay)
        gg = self.gain
        
        lx = len(x)
        
        if lx > mm:
            al = [self.forward(x[i:i+mm]) for i in range(0, lx, mm)]
            return np.concatenate(al)

        # note buffer expects constant input number of channels.
        if self._buffer is None:
            self._buffer = np.zeros((mm, x.shape[-1]))
        
        d0 = self._buffer[:lx]
        a0 = x + d0 * gg
            
        self._buffer = np.concatenate((self._buffer, a0))[-mm:]

        return a0
    
    @property
    def rt(self):
        return 3 * self.delay / -math.log(abs(self.gain))
    
    @rt.setter
    def rt(self, t):
        self.gain = math.exp(-3 * self.delay / abs(t)) * np.sign(t)


class AllpassFilter(Sound):
    
    def __init__(
        self,
        delay=0.040,
        gain=0.5, 
        shared=False,
    ):
    
        super().__init__(shared=shared)
        
        self.delay = delay
        self.gain = gain
 
    def forward(self, x):
        
        mm = t2frames(self.delay)
        gg = self.gain
        
        lx = len(x)
        
        if lx > mm:
            al = [self.forward(x[i:i+mm]) for i in range(0, lx, mm)]
            return np.concatenate(al)

        # note buffer expects constant input number of channels.
        if self._buffer is None:
            self._buffer = np.zeros((mm, x.shape[-1]))
        
        d0 = self._buffer[:lx]
        d1 = self.nested(d0)
        
        a0 = x + d1 * gg
            
        self._buffer = np.concatenate((self._buffer, a0))[-mm:]

        return x * -gg + d1 * (1 - gg**2)
    
    def nested(self, x):
        return x


class SchroederReverb(Sound):
    """A Schroeder reverb.

    Implemented according to Natural Sounding Artificial Reverberation (1962)
    http://www2.ece.rochester.edu/~zduan/teaching/ece472/reading/Schroeder_1962.pdf
    """
    def __init__(self, mix=0.25, rt=0.750, shared=False):
    
        super().__init__(shared=shared)
        
        self.comb0 = CombFilter(0.030)
        self.comb1 = CombFilter(0.033)
        self.comb2 = CombFilter(0.041)
        self.comb3 = CombFilter(0.045)

        self.allp0 = AllpassFilter(0.0050, 0.7)
        self.allp1 = AllpassFilter(0.0017, 0.7)
        
        self.mix = mix
        self.rt = rt
        
    @property
    def rt(self):
        return self.comb0.rt
    
    @rt.setter
    def rt(self, t):
        
        self.comb0.rt = t
        self.comb1.rt = t
        self.comb2.rt = t
        self.comb3.rt = t
        
    def forward(self, x):
        
        c0 = self.comb0(x)
        c1 = self.comb1(x)
        c2 = self.comb2(x)
        c3 = self.comb3(x)

        cs = np.stack((c0, c1, c2, c3)).sum(0)
        
        a0 = self.allp0(cs)
        a1 = self.allp1(a0)
        
        return x * (1 - self.mix) + a1 * self.mix


class SchroederReverb2(AllpassFilter):
    """A Schroeder reverb.

    Implemented according to Natural Sounding Artificial Reverberation (1962)
    http://www2.ece.rochester.edu/~zduan/teaching/ece472/reading/Schroeder_1962.pdf
    """    
    def __init__(
        self,
        delay=0.030,
        gain=0.5, 
        shared=False,
    ):
    
        super().__init__(delay, gain, shared=shared)

        self.allp0 = AllpassFilter(0.1 / 1.0, gain=0.7)
        self.allp1 = AllpassFilter(0.1 / 3.1, gain=0.7)
        self.allp2 = AllpassFilter(0.1 / 8.9, gain=0.7)
        self.allp3 = AllpassFilter(0.1 / 28., gain=0.7)
        self.allp4 = AllpassFilter(0.1 / 79., gain=0.7)
        
    def nested(self, x):
        
        x = self.allp0(x)
        x = self.allp1(x)
        x = self.allp2(x)
        x = self.allp3(x)
        x = self.allp4(x)
        
        return x


class Overdrive(Sound):
     
    def __init__(self, gain=1., amp=0.5, shared=False):
    
        super().__init__(amp=amp, shared=shared)
        
        self.gain = gain

    def forward(self, x):
        
        return np.tanh(x * self.gain) * self.amp


class Overdrive2(Sound):
     
    def __init__(self, gain=1., amp=0.5, shared=False):
    
        super().__init__(amp=amp, shared=shared)
        
        self.gain = gain

    def forward(self, x):
        
        a = 0.45 * self.gain * x
        return a * (1 + np.abs(a)) * self.amp
        
