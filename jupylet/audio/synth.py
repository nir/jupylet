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

from .sound import GatedSound, Envelope, Oscillator, Noise, ResonantFilter


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

