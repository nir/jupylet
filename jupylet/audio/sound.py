"""
    jupylet/audio/sound.py
    
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
import inspect
import logging
import weakref
import random
import copy
import math
import time
import sys
import os

import scipy.signal

import numpy as np

from ..utils import settable, Dict, trimmed_traceback

from ..audio import FPS, MIDDLE_C, DEFAULT_AMP, t2frames, frames2t   
from ..audio import get_time, get_bpm, get_note_value

from .note import note2key, key2note
from .device import add_sound, get_schedule
from .device import set_device_latency, get_device_latency_ms


logger = logging.getLogger(__name__)


DEBUG = False

EPSILON = 1e-6


def get_plot(*args, grid=True, figsize=(10, 5), xlim=None, ylim=None, **kwargs):
    
    import matplotlib.pyplot as plt
    import PIL.Image
    import io
    
    b = io.BytesIO()

    plt.figure(figsize=figsize)
    plt.grid(grid)
    
    if xlim:
        plt.xlim(*xlim)

    if ylim:
        plt.ylim(*ylim)

    plt.plot(*args, **kwargs)    
    plt.savefig(b, format='PNG', bbox_inches='tight')
    plt.close()
    
    return PIL.Image.open(b)


def compute_running_mean(x, n=1024):
    
    nb = n // 2
    na = n - nb
    
    px = np.pad(x, (na, nb))
    cs = np.cumsum(px) 
    
    po = np.pad(np.ones(len(x)), (na, nb))
    ns = np.cumsum(po) 

    return (cs[n:] - cs[:-n]) / (ns[n:] - ns[:-n])


def get_power_spectrum_plot(a0, sampling_frequency=FPS, window=None, **kwargs):
    
    ft = np.fft.fft(a0.squeeze())
    sa = np.square(np.abs(ft))
    ps = 10 * np.log10(sa)
    
    ff = np.fft.fftfreq(len(a0), 1/sampling_frequency)

    #print(a0.shape, ps.shape, ff.shape)
    
    if window == 1:
        return get_plot(ff, ps, **kwargs)
    
    if window is None:
        window = len(a0) // 4096
    
    rm = compute_running_mean(ps, window)
    
    return get_plot(ff, rm, **kwargs)


#
# Played sounds are schedulled a little into the future so as to start at a 
# particular planned moment in time rather than at the arbitrary time of the 
# start of the next sound buffer.
#
_latency = get_device_latency_ms() / 1000


def set_latency(latency='high'):

    assert latency in ['high', 'low', 'lowest', 'minimal']

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

    The Jupylet Sound class is the basic element for defining a sound 
    processing computational graph, as a hierarchy of sound classes; e.g, a 
    synthesizer containing a reverb effect containing an allpass filter.

    Audio building blocks and components such as oscillators, effects, etc...,
    typically inherit from the Sound class, while instruments such as 
    synthesizers should typically inherit the :class:`GatedSound` class.

    Args:
        freq (float): Default frequency.
        amp (float): Output amplitude - a value between 0 and 1.
        pan (float): Balance between left (-1) and right (1) output channels.
        shared (bool): Designate sound object as shared by multiple other
            sound instances. 
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

        self._fargs = None  
        self._error = None

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
                
    def play_release(self, stop=True, **kwargs):
        """Stop playing sound and all of its polyphonic copies."""

        polys = []

        while self._polys:
            wr = self._polys.pop(-1)
            ps = wr()
            if ps is not None:
                ps.play_release(stop=stop, **kwargs)
                polys.append(wr)

        for wr in polys[:512]:
            self._polys.append(wr)

        if stop:
            self._done = self.index or 1
        
    def play_poly(self, note=None, **kwargs):
        """Play given note polyphonically.

        This function will play note on a new copy of self.
        If sound is already playing, the new note will join it.
        
        Args:
            note (float): Note to play in units of semitones 
                where 60 is middle C.
            **kwargs: Properties of intrument to modify.
        
        Returns:
            Sound: The sound object representing the newly playing note.
        """
        o = self.copy(track=True)
        o.play(note, **kwargs)

        return o

    def play(self, note=None, **kwargs):
        """Play given note monophonically.

        If sound is already playing, it will be reset.
        
        Args:
            note (float): Note to play in units of semitones 
                where 60 is middle C.
            **kwargs: Properties of intrument to modify.
        """
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

        self._error = None

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

        if self._error:
            return True

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

    def consume(self, frames, channels=2, raw=False, *args, **kwargs):
        
        self._rset('frames', frames)
        
        a0 = self(*args, **kwargs)

        if raw:
            return a0

        # The following mechanism is a brittle way to minimize the
        # computation time in case the sound is done but is kept around for 
        # an effect applied to it.

        if not self._done or self._ac is None or len(self._ac) != self.frames:

            a0 = _expand_channels(a0, channels)
            
            if channels == 2:
                self._ac = a0 * _ampan(self.velocity / 128 * self.amp, self.pan)
            else:
                self._ac = a0 * (self.velocity / 128 * self.amp)

        return self._ac

    def __call__(self, *args, **kwargs):
        
        assert getattr(self, 'frames', None) is not None, 'You must call super() from the sound class constructor'
        
        for k in list(kwargs.keys()):
            if hasattr(self, k) and k not in self._get_forward_args():
                if k == 'frames':
                    self._rset('frames', kwargs.pop('frames'))
                else:        
                    setattr(self, k, kwargs.pop(k))

        if not self._done or self._a0 is None or len(self._a0) != self.frames:
            try:
                self._a0 = self.forward(*args, **kwargs)
            except:
                self._error = trimmed_traceback()
                logger.error(self._error)
                self._a0 = np.zeros((self.frames, 1))

        if isinstance(self._a0, np.ndarray):
            self.index += len(self._a0)
        
        if DEBUG:
            self._al = self._al[-255:] + [self._a0]

        return self._a0

    def _get_forward_args(self):
        if self._fargs is None:
            self._fargs = set(inspect.getfullargspec(self.forward).args)
        return self._fargs

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
        """float: Get current sound frequency in semitone units where 60 is middle C."""
        return freq2key(self.freq)
    
    @key.setter
    def key(self, value):
        self.freq = key2freq(value)
        
    @property
    def note(self):
        """str: Get note closest to current sound frequency, as a string."""
        return key2note(self.key)
    
    @note.setter
    def note(self, value):
        self.key = note2key(value) if type(value) is str else value

    def get_effects(self):
        """Get list of effects for this sound object.
        
        Returns:
            list: A (possibly empty) list of sound effects.
        """
        return self._effects

    def set_effects(self, *effects):
        """Set effects to be applied to the output of this sound instance.

        Args:
            *effects: Sound effects instances.
        """
        self._effects = effects


class LatencyGate(Sound):
    """A synthesizer on/off gate.
    
    A synthesizer gate outputs an on/off signal that is used to 
    trigger signal processing such as envelope generators etc...

    This particular latency gate is designed to schedule `on` and `off` 
    transitions using system time to enable triggering notes with precise 
    timing despite fluctuations in the latency of the operating system.
    """
    def __init__(self):
        
        super().__init__()
        
        self.states = []
        self.opened = False
        self.value = 0

    def reset(self, shared=False):
        
        super().reset(shared)
        
        self.states = []
        self.opened = False
        self.value = 0

    def forward(self):
        
        #
        # open/close events are scheduled in terms of absolute time. Here these 
        # timestamps are converted into a frame index.
        #

        #states = []

        a0 = np.zeros((self.frames, 1))
        v0 = self.value
        i0 = 0

        t0 = time.time()
        schedule = get_schedule()
        
        while self.states:
            
            t, event = self.states[0]
            
            if schedule:
                dt = max(0, t + _latency - schedule)
            else:
                dt = max(0, t - t0)
                
            df = t2frames(dt)
            i1 = min(df, self.frames)

            if df > i1:
                break

            if self.value == 0 and event == 'open':
                self.value = 1
                self.opened = True
                i0 = i1
                #if df <= i1:
                #    states.append((self.index + i1, 'open'))          

            elif self.value == 1 and event == 'close':
                self.value = 0
                a0[i0:i1] += 1
                #if df <= i1:
                #    states.append((self.index + i1, 'close'))          

            self.states.pop(0)

        if self.value == 1 and i0 < self.frames:
            a0[i0:self.frames] += 1

        #states.append((self.index + self.frames, 'continue'))          

        return a0

        
    def open(self, t=None, dt=None, **kwargs):
        """Schedule gate open at specified time.

        The schedule can be an absolute time given by the argument `t`, or 
        a delta `dt` after the schedule of the latest event already scheduled.

        Args:
            t (float, optional): Time in seconds since epoch, as returned by 
                Python's standard library ``time.time()``.
            dt (float, optional): Time in seconds after the current last 
                scheduled event.
        """
        self.schedule('open', t, dt)
        
    def close(self, t=None, dt=None, **kwargs):
        """Schedule gate close at specified time.

        The schedule can be an absolute time given by the argument `t`, or 
        a delta `dt` after the schedule of the latest event already scheduled.

        Args:
            t (float, optional): Time in seconds since epoch, as returned by 
                Python's standard library ``time.time()``.
            dt (float, optional): Time in seconds after the current last 
                scheduled event.
        """
        self.schedule('close', t, dt)
        
    def schedule(self, event, t=None, dt=None):
        logger.debug('Enter LatencyGate.schedule(event=%r, t=%r, dt=%r).', event, t, dt)

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


def gate2events(gate, v0=0, index=0):
    
    states = []

    end = index + len(gate)
    gate = gate > 0
    
    while len(gate):

        if v0 == 0:

            am = int(gate.argmax())
            gv = int(bool(gate[am]))

            if gv == v0:
                break
            
            v0 = gv
            index += am            
            states.append((index, 'open'))
            gate = gate[am:]
            
        else:
            
            am = int(gate.argmin())
            gv = int(bool(gate[am]))

            if gv == v0:
                break
            
            v0 = gv
            index += am            
            states.append((index, 'close'))
            gate = gate[am:]
            
    states.append((end, 'continue'))
    
    return states, v0, end

    
class GatedSound(Sound):
    """A sound class capable of precise timing and duration of notes.

     Args:
        freq (float): Fundamental frequency.
        amp (float): Output amplitude - a value between 0 and 1.
        pan (float): Balance between left (-1) and right (1) output channels.
        duration (float, optional): Duration to play note, in whole notes.    
    """
    def __init__(self, freq=MIDDLE_C, amp=DEFAULT_AMP, pan=0., duration=None):
        
        super().__init__(freq=freq, amp=amp, pan=pan)

        self.gate = LatencyGate()

        self.duration = duration
        
    @property
    def done(self):

        if self._error:
            return True

        return Sound.done.fget(self) if self.gate.opened else False

    def play_poly(self, note=None, duration=None, **kwargs):
        """Play given note polyphonically.

        This function will play note on a new copy of self.
        If sound is already playing, the new note will join it.
        
        Args:
            note (float): Note to play in units of semitones 
                where 60 is middle C.
            duration (float, optional): Duration to play note, in whole notes.    
            **kwargs: Properties of intrument to modify.
        
        Returns:
            GatedSound: The sound object representing the newly playing note.
        """
        o = self.copy(track=True)
        o.play(note, duration, **kwargs)

        return o

    def play(self, note=None, duration=None, **kwargs):
        """Play given note monophonically.

        If sound is already playing, it will be reset.
        
        Args:
            note (float): Note to play in units of semitones 
                where 60 is middle C.
            duration (float, optional): Duration to play note, in whole notes.    
            **kwargs: Properties of intrument to modify.
        """
        if duration is None:
            duration = self.duration

        t = kwargs.pop('t', None)
        dt = kwargs.pop('dt', None)

        super().play(note, **kwargs)
        self.gate.open(t, dt)

        if duration is not None:
            self.gate.close(dt=duration * get_note_value() * 60 / get_bpm())
        
    def play_release(self, stop=False, **kwargs):

        super().play_release(stop=stop, **kwargs)

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
        
        # Last gate value.
        self._lgate = 0

    def reset(self, shared=False):
        
        super().reset(shared)
        
        self._state = None
        self._start = 0
        self._valu0 = 0
        self._valu1 = 0
        self._lgate = 0        
        
    def forward(self, gate):
        
        if isinstance(gate, np.ndarray):
            states, self._lgate, end = gate2events(gate, self._lgate, self.index)
        else:
            states = gate

        #print(states)

        index = self.index
        
        # TODO: This code assumes the envelope frame index and the gate frame
        # index are synchronized (the same). In practice this is correct, but
        # it should not be assumed. Instead the gate itself should include 
        # its buffer start and end index. 

        curves = []
        
        for event_index, event in states:
            #print(event_index, event)
            
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
            next_state = 'sustain' if self.sustain else None
            
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
        duty = duty.reshape(-1).clip(0.01, 0.99)
        
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

    """Waveform generator for `sine`, `triangle`, anti-aliased `sawtooth`, and 
    variable duty anti-aliased `square` waveforms.

    Args:
        shape (str): Waveform to generate - one of `sine`, `triangle`, 
            `sawtooth`, or `square`.
        freq (float): Fundamental frequency of generator.
        key (float, optional): Fundamental frequency of generator in semitone
            units where middle C is 60.
        sign (float): Set to -1 to flip sawtooth waveform upside down.
        duty (float): The fraction of the square waveform cycle its value is 1.

    Note:
        An Oscillator inherits all the methods and properties of a Sound class.
    """
    
    def __init__(self, shape='sine', freq=MIDDLE_C, key=None, phase=0., sign=1, duty=0.5, **kwargs):
        """"""

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
            triangle = get_triangle_wave,
            sawtooth = get_sawtooth_wave,
            square = get_square_wave,
            pulse = get_square_wave,
            saw = get_sawtooth_wave,
            tri = get_triangle_wave,
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
    purple = 6,
)


class Noise(Sound):
    
    def __init__(self, color=noise_color.white):
        
        super().__init__()
        
        if type(color) is str:
            assert color in noise_color, 'Noise color name should be one of %s.' % ', '.join(noise_color.keys())

        self.color = color
        self.state = None
        self.noise = None

        self._color = color

    def forward(self, color_modulation=0):
        
        if type(self.color) is str:
            color = noise_color[self.color]
        else:
            color = self.color

        if isinstance(color_modulation, np.ndarray):
            color = color + np.mean(color_modulation[-1]).item()
        else:
            color = color + color_modulation
            
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

