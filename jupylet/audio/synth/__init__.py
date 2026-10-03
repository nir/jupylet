"""
    jupylet/audio/synth/__init__.py
    
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

from ..sound import GatedSound, Envelope, Oscillator, Noise
from .. import note, DEFAULT_AMP

from .hammond_organ import Hammond, Chorus, drawbars
from .tb303_bassline import TB303, get_tb303_panel


logger = logging.getLogger(__name__)


#
# Synth and Drums are small examples of how a synthesizer is built in
# jupylet: a few building blocks from sound.py (an oscillator or noise,
# an envelope and the gate) wired together in a forward() method. They
# are kept simple on purpose, to read rather than to play. For full
# instruments, see TB303 in tb303_bassline.py, a replica of the Roland
# TB-303, and Hammond in hammond_organ.py.
#
# The Programming Synthesizers chapter of the programmer's reference
# guide shows step by step how to build synthesizers like these:
# https://jupylet.readthedocs.io/en/latest/programmers_reference_guide/synthesis.html
#
# For example, in a notebook:
#
#     from jupylet.audio.bundle import *
#
#     synth = Synth()
#     synth.play(C4, 1/4)
#
#     # The note sets the noise's color: lower notes sound darker.
#     drums = Drums()
#     drums.play(C2, 1/8)
#


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
