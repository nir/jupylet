"""
    jupylet/audio/sound.py
    
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
import weakref
import random
import copy
import math
import time
import os

import scipy.signal

import numpy as np

from ..utils import settable, Dict

from ..audio import FPS, MIDDLE_C, DEFAULT_AMP, t2frames, frames2t   
from ..audio import get_time, get_bpm, get_note_value

from .note import note2key, key2note
from .device import add_sound, get_schedule
from .device import set_device_latency, get_device_latency_ms


logger = logging.getLogger(__name__)


DEBUG = False

EPSILON = 1e-6


#
# Played sounds are schedulled a little into the future so as to start at a 
# particular planned moment in time rather than at the arbitrary time of the 
# start of the next sound buffer.
#
_latency = get_device_latency_ms() / 1000


def set_latency(latency='high'):

    assert latency in ['high', 'low', 'lowest']

    global _latency

    set_device_latency(latency)
    _latency = get_device_latency_ms(latency) / 1000


def get_latency_ms():
    return _latency * 1000

    
def _expand_channels(a0, channels):

    if len(a0.shape) == 1:
        a0 = np.expand_dims(a0, -1)

    if a0.shape[1] < channels:
        a0 = a0.repeat(channels, 1)

    if a0.shape[1] > channels:
        a0 = a0[:,:channels]

    return a0


#
# A helper function to amplify and pan (balance) audio between the left and
# right channels.
#

#@functools.lru_cache(maxsize=1024)
def _ampan(amp, pan):
    return np.array([1 - pan, 1 + pan]) * (amp / 2)


_LOG_C4 = math.log(MIDDLE_C)
_LOG_CC = math.log(2) / 12
_LOG_CX = _LOG_C4 - 60 * _LOG_CC


def key2freq(key):
    
    if isinstance(key, np.ndarray):
        return np.exp(key * _LOG_CC + _LOG_CX)
    else:
        return math.exp(key * _LOG_CC + _LOG_CX)
    
 
def freq2key(freq):
    
    if isinstance(freq, np.ndarray):
        return (np.log(freq) - _LOG_CX) / _LOG_CC
    else:
        return (math.log(freq) - _LOG_CX) / _LOG_CC
        

class Sound(object):
    """The base class for all other sound classes, including audio samples, 
    oscillators and effects.
    """

    def __init__(self, freq=MIDDLE_C, amp=DEFAULT_AMP, pan=0., shared=False):
        
        self.freq = freq
        
        # MIDI attribute corresponding to velocity of pressed key,
        # between 0 and 128.
        self.velocity = 64

        # Amplitude (or volume) beween 0 and 1.
        self.amp = amp
        
        # Left-right audio balance - a value between -1 and 1.
        self.pan = pan
        
        # The number of frames the forward() method is expected to return.
        self.frames = 1024

        # The frame counter.
        self.index = 0
        
        self._buffer = None

        # Indicate if sound is shared by multiple sounds. For example
        # an effect may be shared by multiple sounds. This affects how it 
        # should react to reset() calls.
        self._shared = shared
        
        # A somewhat brittle mechanism to force a note to keep "playing"
        # for a few seconds after it's done, so a shared effect may still
        # be applied to it (for example in the case of a long reverb).
        self._done = 0
        self._done_decay = 5 * FPS

        # The lastest output arrays of the forward() function.
        self._a0 = None
        self._ac = None
        self._al = []
                
        self._polys = []
        self._effects = ()
        
    def _rset(self, key, value, force=False):
        """Recursively, but lazily, set property to given value on all child sounds.
        
        This function is used for example to set number of required frames on the entire
        tree of sound objects before calling the forward() method.
        """
        if force or self.__dict__.get(key, '__NONE__') != value:
            for s in self.__dict__.values():
                if isinstance(s, Sound):
                    s._rset(key, value, force=True)
            
        self.__dict__[key] = value
    
    def _ccall(self, name, *args, **kwargs):
        """Recursively call given function of each sound object in the tree 
        of sounds.
        """
        for s in self.__dict__.values():
            if isinstance(s, Sound):
                getattr(s, name)(*args, **kwargs)
                
    def play_release(self, **kwargs):
        
        polys = []

        while self._polys:
            wr = self._polys.pop(-1)
            ps = wr()
            if ps is not None:
                ps.play_release(**kwargs)
                polys.append(wr)

        for wr in polys[:512]:
            self._polys.append(wr)
        
    def play_poly(self, note=None, **kwargs):
        """Play new copy of sound.

        If sound is already playing it will play the new copy in parallel. 
        
        Returns:
            Sound object: The newly copied and playing sound object.
        """
        o = self.copy(track=True)
        o.play(note, **kwargs)

        return o

    def play(self, note=None, **kwargs):
        #logger.info('Enter Sample.play(note=%r, **kwargs=%r).', note, kwargs)
        
        self.reset(self._shared)
        
        if note is not None:
            self.note = note

        # This mechanism allows the play() function to modify any of the 
        # sound properties before playing.
        self.set(**kwargs)

        # Send sound to audio device for playing. 
        add_sound(self)

    def set(self, **kwargs):

        for k, v in kwargs.items():
            if settable(self, k):
                setattr(self, k, v)  
    
        return self

    def copy(self, track=False):
        """Create a copy of sound object.

        This function is a mixture of shallow and deep copy. It deep-copies 
        the entire tree of child sound objects, but shallow-copies the other
        properties of each sound object in the tree. The motivation is to 
        avoid creating unnecessary copies of numpy buffers.

        However, this means it should be followed by a reset() call on 
        the newly copied sound to prevent unintentionally sharing buffers.

        Returns:
            Sound object: new copy of sound object. 
        """
        o = copy.copy(self)

        for k, v in o.__dict__.items():
            if isinstance(v, Sound) and not v._shared:
                setattr(o, k, v.copy())

        if track:
            self._polys.append(weakref.ref(o))
          
        o._polys = []

        return o
       
    def reset(self, shared=False):
        
        # TODO: think how to handle reset of shared index.
        self.index = 0

        # When a sound (effect) is shared by multiple other sounds, its state
        # should not be reset in the usual way. However this is probably not 
        # correctly implemented. For example, the self.index should probably 
        # not reset either - need to think about this more.
        if not shared:
            self._buffer = None 

        self._done = 0
        self._a0 = None
        self._ac = None
        self._al = []
        
        self._ccall('reset', shared=shared or self._shared)
        
    @property
    def done(self):
        
        # The done() function is used by the sound device to determine when
        # a playing sound may be considered done and discarded.
        # There are various criteria and the logic is probably brittle and 
        # needs to be considered again and simplified.
        #
        # The general idea is to consider a sound done if after it has played
        # for a while, it becomes nearly zero for an entire output buffer
        # length. 
        #
        # However, in the case effects are applied to the sound, it may be
        # needed around for a while longer even if its output has become 
        # zero. For example in the case of a reverb effect.

        if self.index < FPS / 8:
            return False
        
        if self._a0 is None or self._ac is None:
            return False
        
        if not self._done:
            if np.abs(self._a0).max() < 1e-4:
                self._done = self.index or 1
                self._a0 = self._a0 * 0
                self._ac = self._ac * 0

            return False

        if not self.get_effects():
            return True
            
        if self.index - self._done < self._done_decay:
            return False

        return True
        
    #
    # This is the function called by the sound device to compute the next
    # *frames* to be sent to the sound device for playing.
    #

    def consume(self, frames, channels=2, *args, **kwargs):
        
        self._rset('frames', frames)
        
        a0 = self(*args, **kwargs)

        # The following mechanism is a brittle way to minimize the
        # computation time in case the sound is done but is kept around for 
        # an effect applied to it.

        if not self._done or self._ac is None or len(self._ac) != self.frames:

            a0 = _expand_channels(a0, channels)
            
            if channels == 2:
                self._ac = a0 * _ampan(self.velocity / 128 * self.amp, self.pan)
            else:
                self._ac = self.velocity / 128 * self.amp * a0

        return self._ac

    def __call__(self, *args, **kwargs):
        
        assert getattr(self, 'frames', None) is not None, 'You must call super() from your sound class constructor'
        
        if not self._done or self._a0 is None or len(self._a0) != self.frames:
            self._a0 = self.forward(*args, **kwargs)

        self.index += len(self._a0)
        
        if DEBUG:
            self._al = self._al[-255:] + [self._a0]

        return self._a0

    # This is for debugging.
    @property
    def _a1(self):
        return np.concatenate(self._al)

    #
    # The pytorch style forward function to compute the next sound buffer.
    #
    def forward(self, *args, **kwargs):
        return np.zeros((self.frames,))
    
    @property
    def key(self):
        return freq2key(self.freq)
    
    @key.setter
    def key(self, value):
        self.freq = key2freq(value)
        
    @property
    def note(self):
        return key2note(self.key)
    
    @note.setter
    def note(self, value):
        self.key = note2key(value) if type(value) is str else value

    def get_effects(self):
        return self._effects

    def set_effects(self, *effects):
        self._effects = effects


class Gate(Sound):
    """A synthesizer gate is traditionally an on/off signal that is used to 
    indicate key presses and other events.

    This gate class functions by producing schedulled on/off events in its
    output. These events can be fed to other sound objects designed to 
    consume such events; for example envelopes.

    For more info: https://www.synthesizers.com/gates.html
    """
    def __init__(self):
        
        super().__init__()
        
        self.states = []
        self.opened = False

    def reset(self, shared=False):
        
        super().reset(shared)
        
        self.states = []
        self.opened = False

    def forward(self):
        
        #
        # open/close events are scheduled in terms of absolute time. Here these 
        # timestamps are converted into a frame index.
        #

        states = []
        end = self.index + self.frames
        
        while self.states:
            
            t, event = self.states[0]
            
            dt = max(0, t + _latency - get_schedule())
            index = self.index + t2frames(dt)

            if index >= end:
                break

            if event == 'open':
                self.opened = True

            self.states.pop(0)
            states.append((index, event))
        
        return states
        
    def open(self, t=None, dt=None, **kwargs):
        self.schedule('open', t, dt)
        
    def close(self, t=None, dt=None, **kwargs):
        self.schedule('close', t, dt)
        
    def schedule(self, event, t=None, dt=None):
        logger.debug('Enter Gate.schedule(event=%r, t=%r, dt=%r).', event, t, dt)

        tt = get_time()

        if not self.states:
            last_t = tt
        else:
            last_t = self.states[-1][0]

        if dt is not None:
            t = dt + last_t
        else:
            t = t or tt

        t = max(t, tt)

        # Discard events scheduled to run after this new event.
        while self.states and self.states[-1][0] > t:
            self.states.pop(-1)

        self.states.append((t, event))


class GatedSound(Sound):
    
    def __init__(self, amp=DEFAULT_AMP, pan=0., duration=None):
        
        super().__init__(amp=amp, pan=pan)

        self.gate = Gate()

        self.duration = duration
        
    @property
    def done(self):
        return Sound.done.fget(self) if self.gate.opened else False

    def play_poly(self, note=None, duration=None, **kwargs):
        """Play new copy of sound.

        If sound is already playing it will play the new copy in parallel. 
        
        Returns:
            Sound object: The newly copied and playing sound object.
        """
        o = self.copy(track=True)
        o.play(note, duration, **kwargs)

        return o

    def play(self, note=None, duration=None, **kwargs):

        if duration is None:
            duration = self.duration

        t = kwargs.pop('t', None)
        dt = kwargs.pop('dt', None)

        super().play(note, **kwargs)
        self.gate.open(t, dt)

        if duration is not None:
            self.gate.close(dt=duration * get_note_value() * 60 / get_bpm())
        
    def play_release(self, **kwargs):

        super().play_release(**kwargs)

        kwargs = dict(kwargs)

        t = kwargs.pop('t', None)
        dt = kwargs.pop('dt', None)

        self.set(**kwargs)
        self.gate.close(t, dt)


#
# An envelope curve may span multiple buffers and it is therefore generated
# piece by piece. The code to do that is very delicate. Be extra careful to 
# modify it. Computations appeared to require float64 precision (!) since in 
# float32 they occasionally emit buffers of the wrong length.
#

def get_exponential_adsr_curve(dt, start=0, end=None, th=0.01):
    """Compute a section of an exponential envelope curve.
    
    Args:
        dt (float): The time it should take the curve to go from 0. to 1.
            minus the given threshold (th).
        start (int): The start frame for the curve.
        end (int): The end frame for the curve.

    Returns:
        ndarray: Array with curve values.    
    """
    df = max(math.ceil(dt * FPS), 1)
    end = min(df, end if end is not None else 60 * FPS)
    start = start + 1
        
    a0 = np.arange(start/df, end/df + EPSILON, 1/df, dtype='float64')
    a1 = np.exp(a0 * math.log(th))
    a2 = (1. - a1) / (1. - th)
    
    return a2


def get_linear_adsr_curve(dt, start=0, end=None):
    """Compute a section of a linear envelope curve.
    
    Args:
        dt (float): The time it should take the curve to go from 0. to 1.
        start (int): The start frame for the curve.
        end (int): The end frame for the curve.

    Returns:
        ndarray: Array with curve values.    
    """
    df = max(math.ceil(dt * FPS), 1)
    end = min(df, end if end is not None else 60 * FPS)
    start = start + 1
    
    a0 = np.arange(start/df, end/df + EPSILON, 1/df, dtype='float64')
    
    return a0


#
# Envelopes are currently the only consumers of gate open/close signals.
#

class Envelope(Sound):
    
    def __init__(
        self, 
        attack=0.,
        decay=0., 
        sustain=1., 
        release=0.,
        linear=True,
    ):
        
        super().__init__()
        
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release

        # Linear or exponential envelope curve.
        self.linear = linear
        
        # The current state of the envelope, one of attack, decay, ...
        self._state = None

        # The first frame index of the current envelope state.
        self._start = 0

        #
        # Pure envelope curves go from 0 to 1, but in practice a curve may go
        # from arbitrary level A to level B. e.g. release may start at sustain
        # level and go down to 0. The following two properties are use to 
        # implement this.
        #
        self._valu0 = 0
        self._valu1 = 0
        
    def reset(self, shared=False):
        
        super().reset(shared)
        
        self._state = None
        self._start = 0
        self._valu0 = 0
        self._valu1 = 0
        
    def forward(self, states):
        
        end = self.index + self.frames
        index = self.index
        
        # TODO: This code assumes the envelope frame index and the gate frame
        # index are synchronized (the same). In practice this is correct, but
        # it should not be assumed. Instead the gate itself should include 
        # its buffer start and end index. also, in this way there will be no 
        # need to add the end (continue) index artificially here.

        states = states + [(end, 'continue')]
        curves = []
        
        for event_index, event in states:
            #print(frame, event)
            
            while index < event_index:
                curves.append(self.get_curve(index, event_index))
                index += len(curves[-1])
                    
            if event == 'open' and self._state != 'attack':
                self._state = 'attack'
                self._start = index
                self._valu0 = self._valu1
            
            if event == 'close' and self._state not in ('release', None):
                self._state = 'release'
                self._start = index
                self._valu0 = self._valu1
            
        return np.concatenate(curves)[:,None]
    
    def get_curve(self, start, end):

        end = max(start, end)

        if self._state in (None, 'sustain'):
            return np.ones((end - start,), dtype='float64') * self._valu0
        
        start = start - self._start
        end = end - self._start
        dt = getattr(self, self._state)
                    
        if self.linear:
            curve = get_linear_adsr_curve(dt, start, end)
        else:
            curve = get_exponential_adsr_curve(dt, start, end)
    
        if len(curve) == 0:
            return curve

        done = curve[-1] >= 1 - EPSILON
        
        if self._state == 'attack':
            target = 1.
            next_state = 'decay'
            
        elif self._state == 'decay':
            target = self.sustain * self._valu0
            next_state = 'sustain'
            
        elif self._state == 'release':
            target = 0.
            next_state = None
        
        else:
            target = 0.
            next_state = None

        curve = (target - self._valu0) * curve  + self._valu0
        
        if done:
            self._state = next_state
            self._start += start + len(curve)
            self._valu0 = curve[-1]
            
        self._valu1 = curve[-1]
        
        return curve


#
# Do not change this "constant"!
#
_NP_ZERO = np.zeros((1,), dtype='float64')


def get_radians(freq, start=0, frames=8192):
    
    pt = 2 * math.pi / FPS * freq
    
    if isinstance(pt, np.ndarray):
        pt = pt.reshape(-1)
    else:
        pt = pt * np.ones((frames,), dtype='float64')
            
    p0 = start + _NP_ZERO
    p1 = np.concatenate((p0, pt))
    p2 = np.cumsum(p1)
    
    radians = p2[:-1]
    next_start = p2[-1] 
    
    return radians, next_start


def get_sine_wave(freq, phase=0, frames=8192, **kwargs):
    
    radians, phase_o = get_radians(freq, phase, frames)
    
    a0 = np.sin(radians)
    
    return a0, phase_o


def get_triangle_wave(freq, phase=0, frames=8192, **kwargs):
    
    radians, phase_o = get_radians(freq, phase, frames)

    a0 = radians % (2 * math.pi)
    a1 = a0 / math.pi - 1
    a2 = a1 * np.sign(-a1)
    a3 = a2 * 2 + 1

    return a3, phase_o


@functools.lru_cache(maxsize=256)
def get_sawtooth_cycle(nharmonics, size=1024):
    
    k = nharmonics
    radians = 2 * math.pi * np.arange(0, 1, 1 / size)
    harmonic = - 2 / math.pi * ((-1) ** k) / k * np.sin(k * radians)

    if k == 1:
        return harmonic

    return harmonic + get_sawtooth_cycle(nharmonics - 1, size)
    
# Warmup
len(get_sawtooth_cycle(128))


def get_sawtooth_wave(freq, phase=0, frames=8192, sign=1., **kwargs):
    
    radians, phase_o = get_radians(freq, phase, frames)

    size = 1024

    # Use mean frequency for the purpose of determining number of 
    # harmonics to use - this may introduce some aliasing.
    if type(freq) not in (int, float):
        freq = float(np.mean(freq))

    nharmonics = max(1, min(128, FPS / 2 // freq))
    nharmonics = kwargs.get('nharmonics', nharmonics)

    sawtooth = get_sawtooth_cycle(nharmonics, size)

    indices = (size / 2 / math.pi * radians).astype('int32') % size
    samples = sawtooth[indices]
    
    if sign != 1.:
        samples = samples * sign

    return samples, phase_o


_nduties = 64
_nharmonics = 128

_km = np.arange(0, _nharmonics+1)[:, None, None] 
_dm = np.linspace(0, 1, _nduties+1)[None, :, None]
_kdm = _km * _dm


@functools.lru_cache(maxsize=256)
def get_square_cycle(nharmonics, size=1024):
    
    k = nharmonics
    radians = 2 * math.pi * np.arange(0, 1, 1 / size)
    harmonic = 4 / math.pi / k * np.sin(math.pi * _kdm[k]) * np.cos(k * radians)[None, :]
    
    if k == 1:
        return harmonic + 2 * _kdm[1] - 1

    return harmonic + get_square_cycle(nharmonics - 1, size)

# Warmup
len(get_square_cycle(128))


def get_square_wave(freq, phase=0, frames=8192, duty=0.5, **kwargs):
    
    if isinstance(duty, np.ndarray):
        duty = duty.reshape(-1)
        
    radians, phase_o = get_radians(freq, phase, frames)

    # Use mean frequency for the purpose of determining number of 
    # harmonics to use - this may introduce some aliasing.
    if type(freq) not in (int, float):
        freq = float(np.mean(freq))

    nharmonics = max(1, min(128, FPS / 2 // freq))
    nharmonics = kwargs.get('nharmonics', nharmonics)

    size = 1024

    square0 = get_square_cycle(nharmonics, size)

    indices = (size / 2 / math.pi * radians).astype('int32') % size

    #
    # When duty is a modulating array, the following simple scheme
    # may result in aliasing. It would be preferable to find
    # a scheme that can efficiently sync changes in duty with the
    # begining of wave cycles.
    #
    if type(duty) in (int, float):
        duty = int(duty * _nduties)
    else:
        duty = (duty * _nduties).astype('int32')

    samples = square0[duty, indices]

    return samples, phase_o


class Oscillator(Sound):
    
    def __init__(self, shape='sine', freq=MIDDLE_C, key=None, phase=0., sign=1, duty=0.5, **kwargs):
        
        super().__init__(freq=freq)
        
        self.shape = shape
        self.phase = phase
        
        if key is not None:
            self.key = key
        
        self.sign = sign
        self.duty = duty
        self.kwargs = kwargs
        
    def forward(self, key_modulation=None, sign=None, duty=None, **kwargs):
        
        if key_modulation is not None:
            freq = key2freq(self.key + key_modulation)
        else:
            freq = self.freq
            
        if sign is None:
            sign = self.sign
            
        if duty is None:
            duty = self.duty
            
        if self.kwargs:
            kwargs = dict(kwargs)
            kwargs.update(self.kwargs)
        
        get_wave = dict(
            sine = get_sine_wave,
            tri = get_triangle_wave,
            saw = get_sawtooth_wave,
            pulse = get_square_wave,
        ).get(self.shape, self.shape)
        
        a0, self.phase = get_wave(
            freq, 
            self.phase, 
            self.frames, 
            sign=self.sign,
            duty=duty, 
            **kwargs
        )
        
        return a0[:,None]


noise_color = Dict(
    brownian = -6,
    brown = -6,
    red = -6,
    pink = -3,
    white = 0,
    blue = 3,
    violet = 6,
)


class Noise(Sound):
    
    def __init__(self, color=noise_color.white):
        
        super().__init__()
        
        self.color = color
        self.state = None
        self.noise = None

        self._color = color

    def forward(self, color_modulation=0):
        
        if isinstance(color_modulation, np.ndarray):
            color = self.color + np.mean(color_modulation[-1]).item()
        else:
            color = self.color + color_modulation
            
        if self._color != color:
            self._color = color
            self.noise = None

        if self.noise is None or len(self.noise) < self.frames:

            a0, self.state = get_noise(
                self._color, 
                max(2048, self.frames), 
                self.state, 
            )

            if self.noise is None:
                self.noise = a0
            else:
                self.noise = np.concatenate((self.noise, a0))

        a0, self.noise = self.noise[:self.frames], self.noise[self.frames:]

        return a0[:,None]


def get_noise(color, frames=4096, state=None, kernel_size=2048, fs=FPS):
    
    assert kernel_size % 2 == 0
    
    if state is None or len(state) != kernel_size:
        state = np.random.randn(kernel_size) / math.pi
        
    wn = np.random.randn(frames) / math.pi
    wn = np.concatenate((state, wn))

    if color == noise_color.red:
        
        pad = kernel_size // 2
        
        c0 = np.cumsum(wn)
        c1 = np.cumsum(c0)

        c2 = (c1[pad:] - c1[:-pad]) / pad
        c3 = (c0[2*pad:] - c2[:-pad]) / 30
    
        return c3, wn[-kernel_size:]
    
    if color == noise_color.white:
        return wn[-frames:], wn[-kernel_size:]
    
    if color == noise_color.violet:
        return np.diff(wn[-frames-1:]), wn[-kernel_size:]
    
    kernel = get_noise_kernel(color, kernel_size, fs)
    
    cn = scipy.signal.convolve(
        wn[1:].astype('float32'), 
        kernel.astype('float32'), 
        'valid'
    ).astype('float64') / 17
    
    return cn[:frames], wn[-kernel_size:]


@functools.lru_cache(maxsize=128)
def get_noise_kernel(color, kernel_size=8192, fs=FPS):
    
    cc = 6.020599915832349
    
    f0 = get_fftfreq(kernel_size, 1/fs, 1)
    f1 = f0 ** (color / cc)
    f2 = fftnoise(f1)
    f3 = (kernel_size / 8 / (f2 ** 2).sum()) ** 0.5 * f2 
    
    return f3


@functools.lru_cache(maxsize=16)
def get_fftfreq(n, d=1., clip=0):
    return np.abs(np.fft.fftfreq(n, d)).clip(clip, 1e6)


def fftnoise(freqs):
    
    f = np.array(freqs, dtype='complex')
    n = (len(f) - 1) // 2
    
    phases = 2 * math.pi * np.random.rand(n) 
    phases = np.cos(phases) + 1j * np.sin(phases)
    
    f[1:n+1] *= phases
    f[-1:-1-n:-1] = np.conj(f[1:n+1])
    
    return np.fft.ifft(f).real


class PhaseModulator(Sound):
    
    def __init__(self, beta=1., shared=False):
        """A sort of phase modulator.
        
        It can be used for aproximate frequency modulation by using the 
        normalized cumsum of the modulated signal, but the signal should be 
        balanced so its cumsum does not drift.
        """
        
        super().__init__(shared=shared)
        
        self.beta = beta
                
    def forward(self, carrier, signal):
        
        signal = signal.mean(-1).clip(-1, 1)
        beta = int(self.beta) + 1
        
        if self._buffer is None:
            self._buffer = np.zeros((2 * beta, carrier.shape[1]), dtype=carrier.dtype)
            
        t1 = np.arange(beta, beta + len(carrier), dtype='float64') + self.beta * signal
        t2 = t1.astype('int64')
        t3 = (t1 - t2.astype('float64'))[:, None]
        
        a0 = np.concatenate((self._buffer, carrier))
        a1 = a0[t2]
        a2 = a0[t2 + 1]
        a3 = a2 * t3 + a1 * (1 - t3)
        
        self._buffer = a0[-2 * beta:]
        
        return a3

