"""
    jupylet/audio/synth.py
    
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


import logging

from .sound import Sound, GatedSound, Envelope, Oscillator, Noise, noise_color
from .sound import PhaseModulator
from .effects import SchroederReverb, Overdrive
from .filters import ResonantFilter

from ..audio import note, DEFAULT_AMP

import numpy as np


logger = logging.getLogger(__name__)


class Synth(GatedSound):
    
    def __init__(self, amp=DEFAULT_AMP, pan=0., duration=None):
        
        super().__init__(amp=amp, pan=pan, duration=duration)

        self.env0 = Envelope(0.03, 0.3, 0.7, 1., linear=False)
        self.osc0 = Oscillator('sine', 4)
        self.osc1 = Oscillator('tri')
                
    def forward(self):

        self.osc1.freq = self.freq

        g0 = self.gate()        
        e0 = self.env0(g0)
                
        o0 = self.osc0()        
        o1 = self.osc1(key_modulation=o0/2)
        
        return o1 * e0


class Drums(GatedSound):
    
    def __init__(self, amp=DEFAULT_AMP, pan=0.):
        
        super().__init__(amp=amp, pan=pan)

        self.env0 = Envelope(0.002, 0.15, 0., 0., linear=False)
        self.noise = Noise()
                
    def forward(self):
        
        color = (self.key - note.C1) / (note.B7 - note.C1) * 12 - 6
        
        g0 = self.gate()        
        e0 = self.env0(g0)
        a0 = self.noise(color)        
        
        return a0 * e0


drawbars = [16, 5+1/3, 8, 4, 2+2/3, 2, 1+3/5, 1+1/3, 1]

def d2f(d):
    return 440 * 16 / (drawbars[d])


class Chorus(Sound):

    def __init__(self, mix=0.5, depth=1/3, shared=False):

        super().__init__(shared=shared)

        self.mix = mix
        self.depth = depth

        self.osc = Oscillator('tri', freq=7)
        self.phm = PhaseModulator(beta=0.85*44.1)

    def forward(self, x):

        if self.mix <= 0:
            return x

        vo = self.osc()
        vb = self.phm(x, vo * self.depth)     

        return x * (1 - self.mix) + vb * self.mix

    @property
    def vibrato_and_chorus(self):
        
        mode = max(0, min(2, round(self.mix * 2)))
        if mode == 0:
            return None
        
        mode = 'c' if mode == 1 else 'v'
        valu = max(1, min(3, round(self.depth * 3)))

        return mode + str(int(valu))

    @vibrato_and_chorus.setter
    def vibrato_and_chorus(self, v):

        assert v in ['c1', 'c2', 'c3', 'v1', 'v2', 'v3', None]

        if v is None:
            self.mix = 0
            return

        self.mix = 0.5 if v[0] == 'c' else 1.
        self.depth = float(v[1]) / 3


class Hammond(GatedSound):
    
    def __init__(self, configuration='888000000', amp=DEFAULT_AMP, pan=0., duration=None):
        
        super().__init__(amp=amp, pan=pan, duration=duration)
        
        self.configuration = configuration
        
        self.reve = SchroederReverb(mix=0.25, rt=0.750, shared=True) 
        self.over = Overdrive(gain=1/amp, amp=amp, shared=True)
        self.chor = Chorus(shared=True)

        self.leak = Noise(noise_color.violet)
        self.env0 = Envelope(0.005, 0., 1., 0.01, linear=False)
        self.prec = Envelope(0., 0.2, 0., 0.01, linear=False)
        
        self.bass = Oscillator(freq=440)
        self.quin = Oscillator(freq=1320)
        self.neut = Oscillator(freq=880)
        self.octa = Oscillator(freq=1760)
        self.naza = Oscillator(freq=2640)
        self.bloc = Oscillator(freq=3520)
        self.tier = Oscillator(freq=4400)
        self.lari = Oscillator(freq=5280)
        self.siff = Oscillator(freq=7040)
        
        self.chorus = True
        self.reverb = True
        self.overdrive = True

        self.leak_gain = 0.05

        self.precussion = True
        self.precussion_gain = 1.5
        self.precussion_decay = 0.2
        self.precussion_drawbar = 3
        
    def get_effects(self):

        el = []

        if self.chorus:
            el.append(self.chor)

        if self.reverb:
            el.append(self.reve)

        if self.overdrive:
            el.append(self.over)

        return tuple(el) + self._effects

    def get_vibrato_and_chorus(self):
        return self.chor.vibrato_and_chorus

    def set_vibrato_and_chorus(self, v):
        self.chor.vibrato_and_chorus = v

    def parse_configuration(self, c):
        return [(i, float(c) / 8) for i, c in enumerate(c)]

    def get_drawbar(self, i):
        dbl = ['bass', 'quin', 'neut', 'octa', 'naza', 'bloc', 'tier', 'lari', 'siff']
        return getattr(self, dbl[i])
    
    def forward(self):
        
        self.prec.decay = self.precussion_decay
        
        g0 = self.gate()        
        e0 = self.env0(g0)
        ep = self.prec(g0)
        
        km = self.key - 69
        
        al = []
        
        for i, a in self.parse_configuration(self.configuration):
            
            if i == self.precussion_drawbar and self.precussion:

                db = self.get_drawbar(i)
                a0 = db(key_modulation=km)
                al.append(a0 * self.precussion_gain * ep)
                
            elif a > 0:
                
                db = self.get_drawbar(i)
                a0 = db(key_modulation=km)
                al.append(a0 * a)
        
        a0 = np.stack(al).sum(0)
        a1 = a0 + self.leak() * self.leak_gain * ep
        a2 = a1 * e0
        
        return a2
               

class TB303(GatedSound):
    
    def __init__(
        self, 
        shape='sawtooth',
        resonance=1,
        cutoff=0,
        decay=2,
        amp=DEFAULT_AMP, 
        pan=0., 
        duration=None
    ):
        
        super().__init__(amp=amp, pan=pan, duration=duration)
                        
        self.shape = shape
        self.resonance = resonance
        self.cutoff = cutoff
        self.decay = decay

        self.env0 = Envelope(0.01, 0., 1., 0.01)
        self.env1 = Envelope(0., decay, 0., 0., linear=False)
        
        self.osc0 = Oscillator(shape)
        
        self.filter = ResonantFilter(btype='lowpass', resonance=resonance)
        
    def forward(self):
        
        g0 = self.gate()
        
        e0 = self.env0(g0)
        e1 = self.env1(g0, decay=self.decay) * 12 * 8
                
        a0 = self.osc0(shape=self.shape, freq=self.freq) 
        
        a1 = self.filter(
            a0, 
            key_modulation=e1+self.cutoff, 
            resonance=self.resonance,
            freq=self.freq,
        )
        
        return a1 * e0


tb303 = TB303()

