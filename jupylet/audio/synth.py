"""
    jupylet/audio/synth.py
    
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


import logging

from .sound import GatedSound, Envelope, Oscillator, Noise, noise_color
from .sound import ResonantFilter, PhaseModulator
from .effects import Delay

import numpy as np


logger = logging.getLogger(__name__)


class Synth(GatedSound):
    
    def __init__(self, amp=1., pan=0., duration=None):
        
        super(Synth, self).__init__(amp, pan, duration)

        self.env0 = Envelope(0.03, 0.3, 0.7, 1., linear=False)
        self.osc0 = Oscillator('sine', 4)
        self.osc1 = Oscillator('tri')
                
    def forward(self):
        
        g0 = self.gate()        
        e0 = self.env0(g0)
                
        o0 = self.osc0()        
        o1 = self.osc1(key_modulation=o0/2)
        
        return o1 * e0

    def play(self, note=None, **kwargs):
        #logger.info('Enter Synth.play(note=%r, **kwargs=%r).', note, kwargs)

        super(Synth, self).play(note, **kwargs)        
        self.osc1.freq = self.freq      
        

class Drums(GatedSound):
    
    def __init__(self, amp=2., pan=0., duration=0.02):
        
        super(Drums, self).__init__(amp, pan, duration)

        self.env0 = Envelope(0.01, 0., 1., 0.3, linear=False)
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


class Hammond(GatedSound):
    
    def __init__(self, configuration='888000000', amp=0.5, pan=0., duration=None):
        
        super().__init__(amp, pan, duration)
        
        self.configuration = configuration
        
        self.revr = Delay(0.2, 0.15, 600, 'bandpass')
        self.revr.bandwidth = 800
        
        self.leak = Noise(noise_color.violet)

        self.vibo = Oscillator('tri', freq=5)
        self.vibr = PhaseModulator(beta=44)

        self.env0 = Envelope(0., 0., 1., 0.01, linear=False)
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
        
        self.reverb = True
        
        self.chorus = True
        self.chorus_depth = 0.1

        self.precussion = True
        self.precussion_gain = 1.5
        self.precussion_decay = 0.2
        self.precussion_drawbar = 3
        
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
        
        if self.chorus:
            
            vo = self.vibo()
            vb = self.vibr(a0, vo)            
            a0 = a0 * (1 - self.chorus_depth) + vb * self.chorus_depth
                    
        a1 = a0 + self.leak() * 0.04 * ep
        a2 = a1 * e0
        
        if self.reverb:
            a2 = self.revr(a2)
                
        return a2
               

class TB303(GatedSound):
    
    def __init__(self, resonance=1., amp=1., pan=0., duration=None):
        
        super(TB303, self).__init__(amp, pan, duration)
                
        self.env0 = Envelope(0.01, 0., 1., 0.01, linear=False)
        self.env1 = Envelope(0., 2., 0., 2., linear=False)
        
        self.osc0 = Oscillator('saw')
        
        self.filter = ResonantFilter(btype='lowpass', resonance=resonance)
        
    def forward(self):
        
        self.osc0.freq = self.freq
        self.filter.freq = self.freq

        g0 = self.gate()
        
        e0 = self.env0(g0)
        e1 = self.env1(g0) * 12 * 8
                
        a0 = self.osc0()      
        a1 = self.filter(a0, key_modulation=e1)
        
        return a1 * e0


tb303 = TB303()

