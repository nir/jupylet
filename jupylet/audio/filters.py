"""
    jupylet/audio/filters.py
    
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
import time

import scipy.signal

import numpy as np

from ..audio import FPS, t2frames
from .sound import Sound, key2freq, freq2key


logger = logging.getLogger(__name__)


class BaseFilter(Sound):
    
    def __init__(self, freq=8192):
        
        super().__init__(freq=freq)
        
        self._f = None
        self._x = None
        self._z = None

    def reset(self, shared=False):
        
        super().reset(shared)

        self._f = None
        self._x = None
        self._z = None
        
    def forward(self, x, key_modulation=None):
        
        if self._x is None:
            self._x = x * 0

        if key_modulation is None:
            freq = self.freq
        elif isinstance(key_modulation, np.ndarray):
            freq = key2freq(self.key + np.mean(key_modulation[-1]).item())
        else:
            freq = key2freq(self.key + key_modulation)

        freq = int(freq)

        if self._f == freq:

            a0, self._z = self.filter(x, self._f, self._z)

            self._f = freq
            self._x = x
            return a0

        if self._f is None:

            xx = np.concatenate((self._x, x))
            a1, self._z = self.filter(xx, freq)
            a1 = a1[-len(x):]

            self._f = freq
            self._x = x
            return a1

        a0, self._z = self.filter(x, self._f, self._z)
        
        xx = np.concatenate((self._x, x))
        a1, self._z = self.filter(xx, freq)
        a1 = a1[-len(x):]

        self._f = freq
        self._x = x

        ww = np.arange(0., 1., 1/len(x))[:,None]
        a1 = a1 * ww + a0 * (1. - ww)
        return a1
            
    def filter(self, x, freq, z=None):
        return x, None
    

def fround(freq):
    return key2freq(round(freq2key(freq), 1))


class ButterFilter(BaseFilter):
    
    def __init__(self, freq=8192, btype='lowpass', db=24, bandwidth=500, output='ba'):
        
        super().__init__(freq)
        
        self.bandwidth = bandwidth
        self.output = output
        self.btype = {'l': 'lowpass', 'h': 'highpass', 'b': 'bandpass'}[btype[0]]
        self.db = db

        self.warmup()

    def warmup(self):

        for freq in sorted(set(fround(f) for f in range(1, FPS//2))):
            signal_butter(self.get_wp(freq), 3, self.db, self.btype, self.output)
            time.sleep(0)

    def get_wp(self, freq):

        freq = key2freq(round(freq2key(freq), 1))

        nyq = FPS // 2

        if self.btype[:3] in ('low', 'hig'):
            return max(1, min(nyq-1, freq))

        lc = max(1, min(nyq-1, freq - self.bandwidth / 2))
        hc = max(1, min(nyq-1, freq + self.bandwidth / 2))
        
        return (lc, hc)

    def filter(self, x, freq, z=None, _retry=True):
        
        try:
            wp = self.get_wp(freq)

            if self.output == 'ba':
                b, a, z0 = signal_butter(wp, 3, self.db, self.btype, self.output)
                return scipy.signal.lfilter(b, a, x, 0, z0 if z is None else z)                
            else:
                sos, z0 = signal_butter(wp, 3, self.db, self.btype, self.output)
                return scipy.signal.sosfilt(sos, x, 0, z0 if z is None else z)
        
        except ValueError:
            if z is None or not _retry:
                raise

            return self.filter(x, freq, None, _retry=False)


@functools.lru_cache(maxsize=4096)
def signal_butter(wp, gpass=3, gstop=24, btype='lowpass', output='ba', fs=FPS):

    nyq = fs // 2

    if btype[:3] == 'low':
        wp = min(wp, nyq - 1)
        ws = min(wp * 2, nyq)
    elif btype[:4] == 'high':
        wp = max(wp, 1)
        ws = wp / 2
    else:
        wp = [max(wp[0], 1), min(wp[1], nyq - 1)]
        ws = [wp[0] / 2, min(wp[1] * 2, nyq)]

    N, Wn = scipy.signal.buttord(wp, ws, gpass, gstop, fs=fs)

    if output == 'ba':
        b, a = scipy.signal.butter(N, Wn, btype, output='ba', fs=fs)
        z = scipy.signal.lfilter_zi(b, a)[:,None]
        return b, a, z

    else:
        sos = scipy.signal.butter(N, Wn, btype, output='sos', fs=fs)
        z = scipy.signal.sosfilt_zi(sos)[:,:,None]
        return sos, z


class PeakFilter(BaseFilter):
    
    def __init__(self, freq=8192, q=10.):
        
        super().__init__(freq)
        
        self.q = q

        self.warmup()

    def warmup(self):

        for freq in sorted(set(fround(f) for f in range(1, FPS//2))):
            signal_iirpeak(freq, self.q)
            time.sleep(0)

    def filter(self, x, freq, z=None):
        
        b, a, z0 = signal_iirpeak(freq, self.q)
        return scipy.signal.lfilter(b, a, x, 0, z0 if z is None else z) 
        

@functools.lru_cache(maxsize=4096)
def signal_iirpeak(w0, q, fs=FPS):

    nyq = fs // 2
    w0 = max(1, min(w0, nyq - 1))

    b, a = scipy.signal.iirpeak(w0, q, fs=fs)
    z = scipy.signal.lfilter_zi(b, a)[:,None]

    return b, a, z


class ResonantFilter(ButterFilter):
    
    def __init__(
        self, 
        freq=8192, 
        btype='lowpass', 
        db=24, 
        bandwidth=500, 
        output='ba',
        resonance=0,
        q=10,
        ):
        
        super().__init__(freq, btype, db, bandwidth, output)
        
        self.resonance = resonance
        self.q = q

        self.pf = PeakFilter()

    def forward(self, x, key_modulation=None):
        
        a0 = super().forward(x, key_modulation)

        if self.btype[0] == 'b' or self.resonance <= 0:
            return a0

        resonance = max(0, self.resonance)

        self.pf.freq = self.freq
        self.pf.q = max(5, self.q if self.q else resonance ** 2)

        a1 = self.pf(a0, key_modulation)
        
        return a0 + a1 * self.resonance
        #return a0 / (resonance / 2 + 1) + a1 

