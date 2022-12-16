"""
    jupylet/audio/sample.py
    
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
import random
import os
import re

import soundfile as sf
import numpy as np

from ..resource import find_path
from ..utils import auto

from ..audio import FPS, MIDDLE_C, DEFAULT_AMP

from .sound import GatedSound, Envelope, Oscillator, Noise
from .sound import key2freq


logger = logging.getLogger(__name__)
    

_SFCACHE_THRESHOLD = 10 * FPS
_SFCACHE_SIZE = 64

_sfcache = {}


def soundfile_read(path, zero_pad=False):
    """Read sound file as a numpy array."""
    logger.info('Enter soundfile_read(path=%r).', path)

    data, fps = _sfcache.get(path, (None, None))
    
    if data is None:
    
        data, fps = sf.read(path, dtype='float64') 
        data = np.pad(data, ((0, 1), (0, 0))[:len(data.shape)])
        
        if len(data) <= _SFCACHE_THRESHOLD:

            if len(_sfcache) >= _SFCACHE_SIZE:
                _sfcache.pop(random.choice(list(_sfcache.keys())))

            _sfcache[path] = (data, fps)

    if not zero_pad:
        data = data[:-1]
        
    return data, fps


#
# Do not change this "constant"!
#
_NP_ZERO = np.zeros((1,), dtype='float64')


def get_indices(intervals=1, start=0, frames=8192):
        
    if isinstance(intervals, np.ndarray):
        pt = intervals.reshape(-1)
    else:
        pt = intervals * np.ones((frames,), dtype='float64')
            
    p0 = start + _NP_ZERO
    p1 = np.concatenate((p0, pt))
    p2 = np.cumsum(p1)
    
    indices = p2[:-1]
    next_start = p2[-1] 
    
    return indices, next_start


def compute_loop(indices, buff_end, loop=False, loop_start=0, loop_end=0):
    
    indices = indices.astype('int64')
    
    if not loop or loop_end <= 0:
        return indices.clip(0, buff_end-1)
    
    il = indices < loop_start
    i0 = indices * il
    i1 = ((indices - loop_start) % (loop_end - loop_start) + loop_start) * (1 - il)
    
    return i0 + i1


@functools.lru_cache(maxsize=1024)
def get_sfz_region(key, path):
    
    rl = read_sfz(path)
    
    md = 1e6
    mr = None

    for r in rl:

        if 'pitch_keycenter' not in r:
            r['pitch_keycenter'] = r['key']
            
        d = abs(key - r['pitch_keycenter'])
        if  md > d:
            md = d
            mr = r

    return mr    


@functools.lru_cache(maxsize=32)
def read_sfz(path):
    
    sfz = open(path).read()
    sfz = '\n' + re.sub(r'//.*', '', sfz)
    
    rl0 = re.findall(r'(?s)<region>.*?(?=<|$)', sfz)
    rl1 = [auto(dict(re.findall('(\w+)=(.*?(?= \w+=|\n|$))', l))) for l in rl0]
    
    return rl1


class Sample(GatedSound):
    """A class to play audio files and samples.

    The Sample class can play audio files of type WAV, OGG, or FLAC, and it 
    can also play basic SFZ format multi sampled virtual instruments. If a 
    sample of a virtual instrument includes markers for a loop segment, the 
    `loop` argument will cause the sample to loop around its loop segment.

    Args:
        path (str): Path to audio file.
        freq (float): Fundamental frequency.
        key (float, optional): Fundamental frequency of generator in semitone
            units where middle C is 60.
        loop (str): Loop sample back to its beginning.
        amp (float): Output amplitude - a value between 0 and 1.
        pan (float): Balance between left (-1) and right (1) output channels.
        duration (float, optional): Duration to play note, in whole notes.    
    """
    def __init__(
        self, 
        path, 
        freq=MIDDLE_C, 
        key=None, 
        loop=False,
        amp=DEFAULT_AMP,
        pan=0.,
        duration=None,
    ):
        
        super().__init__(amp=amp, pan=pan, duration=duration)
        
        self.env0 = Envelope(0., 0., 1., 1., linear=False)

        self.path = str(find_path(path))
        self.buff = None
        
        self.phase = 0        
        self.freq = freq
        
        if key is not None:
            self.key = key

        self.loop = loop
        self.loop_power = None
        
        self.region = dict(
            loop_start = 0,
            loop_end = 0,
            pitch_keycenter = None,
        )
        
    def reset(self, shared=False):
        
        super().reset(shared)
        
        self.phase = 0
        
    def forward(self, key_modulation=None):
        
        self.load()
           
        lb = len(self.buff)
        ls = self.region.get('loop_start', 0)
        le = self.region.get('loop_end', lb - 1)
        
        pitch_keycenter = self.region['pitch_keycenter']
        
        if pitch_keycenter is None and key_modulation is None:
            
            indices, self.phase = get_indices(1., self.phase, self.frames)
            indices = compute_loop(indices, lb, self.loop, ls, le)
            
            return self.buff[indices]
          
        if pitch_keycenter is None:
            interval = key2freq(self.key + key_modulation) / self.freq
            
        elif key_modulation is None:
            interval = self.freq / key2freq(pitch_keycenter)
            
        else:
            interval = key2freq(self.key + key_modulation) / key2freq(pitch_keycenter)         
            
        indices, self.phase = get_indices(interval, self.phase, self.frames)
        
        t3 = (indices % 1.)[:, None]
        
        indices0 = compute_loop(indices, lb, self.loop, ls, le)
        indices1 = compute_loop(indices + 1, lb, self.loop, ls, le)
        
        a0 = self.buff        
        a1 = a0[indices0]
        a2 = a0[indices1]
        
        a3 = a2 * t3 + a1 * (1 - t3)
        
        lp = self.get_loop_power(indices)
        if lp is not None:
            a3 = a3 * lp[:, None]
        
        g0 = self.gate()
        e0 = self.env0(g0)

        return a3 * e0

    def get_loop_power(self, indices):
        
        if self.loop_power is not None:
            
            ls = self.region['loop_start']
            le = self.region['loop_end']

            p0 = ((indices - ls) // (le - ls)).clip(0)

            return self.loop_power ** p0
        
    def load(self):
        """Load specified sample file into memory.
        
        Returns:
            Sample: self
        """
        if self.path.endswith('.sfz'):
            self.load_sfz()
            return self
            
        if self.buff is not None:
            return self
        
        self.buff = soundfile_read(self.path, zero_pad=True)[0]

        if len(self.buff.shape) == 1:
            self.buff = self.buff[:,None]

        if self.region['loop_end'] == 0:
            self.region['loop_end'] = len(self.buff) - 1
            
        return self
    
    def load_sfz(self):
        
        region = get_sfz_region(round(self.key), self.path)
        
        if region['pitch_keycenter'] == self.region['pitch_keycenter']:
            return
        
        path = os.path.join(os.path.dirname(self.path), region['sample'])
        
        self.buff = soundfile_read(path, zero_pad=True)[0] 
        
        if len(self.buff.shape) == 1:
            self.buff = self.buff[:,None]

        self.region = region

        if 'loop_end' in region:
            
            ls = region['loop_start']
            le = region['loop_end']

            rms0 = (self.buff[ls:(ls+le)//2] ** 2).mean() ** 0.5
            rms1 = (self.buff[(ls+le)//2:le] ** 2).mean() ** 0.5

            self.loop_power = min((rms1 / rms0) ** 2, 1.0)

        else:
            self.loop_power = None

