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
import numba

from ..utils import settable, Dict, trimmed_traceback

from ..audio import FS, MIDDLE_C, DEFAULT_AMP, t2frames, frames2t   
from ..audio import get_time, get_bpm, get_note_value

from .note import note2key, key2note
from .device import add_sound, get_schedule
from .device import set_device_latency, get_device_latency_ms


logger = logging.getLogger(__name__)


DEBUG = False

EPSILON = 1e-6


def plot(
    *args,
    grid=True,
    figsize=(10, 5),
    xlim=None,
    ylim=None,
    xscale=None,
    labels=None,
    **kwargs
):
    """Plot with matplotlib and return the plot as an image.

    Args:
        *args: Arguments to ``matplotlib.pyplot.plot()``, for example ``y``,
            or ``x, y``, or several of those for several lines.
        labels (list): Optional names of the lines, shown in a legend.

    Returns:
        IPython.display.Image: The plot as a small JPEG image, which Jupyter
            displays, also when the notebook is loaded from disk.
    """

    import matplotlib.pyplot as plt
    import IPython.display
    import io

    fig = plt.figure(figsize=figsize)
    plt.grid(grid)
    
    if xlim:
        plt.xlim(*xlim)

    if ylim:
        plt.ylim(*ylim)

    if xscale:
        plt.xscale(xscale)
    
    plt.plot(*args, **kwargs)

    if labels:
        plt.legend(labels)

    #
    # Return the plot as a small JPEG image, rather than let Jupyter show
    # the figure itself as a PNG. A notebook saves every image it shows
    # inside its file, so plots add up quickly: in a lesson notebook with
    # a few dozen plots, the images are most of the file. A JPEG at this
    # size takes about half the space of Jupyter's PNG, and looks nearly
    # the same on screen. Closing the figure keeps Jupyter from showing
    # it a second time.
    #
    b = io.BytesIO()
    fig.savefig(b, format='jpeg', dpi=72, bbox_inches='tight', pil_kwargs={'quality': 75})
    plt.close(fig)

    return IPython.display.Image(b.getvalue(), format='jpeg')


get_plot = plot


def compute_running_mean(x, n=1024):
    
    nb = n // 2
    na = n - nb
    
    px = np.pad(x, (na, nb))
    cs = np.cumsum(px) 
    
    po = np.pad(np.ones(len(x)), (na, nb))
    ns = np.cumsum(po) 

    return (cs[n:] - cs[:-n]) / (ns[n:] - ns[:-n])


def get_power_spectrum_plot(*sounds, sampling_frequency=FS, window=None, **kwargs):
    """Plot the power of each frequency in one or more sounds, in decibels.

    Args:
        *sounds: One or more sounds, as arrays of samples.
        sampling_frequency (int): The sample rate of the sounds.
        window (int): Number of neighboring frequencies to average, to smooth
            the plot; by default one for every 4096 samples of the sound.
        **kwargs: Arguments to :func:`plot`, for example ``xlim``,
            ``ylim``, ``xscale='log'`` or ``labels``.
    """
    args = []

    for a0 in sounds:

        a0 = a0.squeeze()

        #
        # Fading the sound in and out with a Hann window keeps the loud
        # frequencies from smearing over the whole spectrum, as they would
        # with the sound cut off abruptly at its ends. The window is scaled
        # to keep the average power of the sound as it is.
        #
        hw = np.hanning(len(a0))
        hw /= np.sqrt(np.mean(np.square(hw)))

        ft = np.fft.rfft(a0 * hw)
        sa = np.square(np.abs(ft))
        ps = 10 * np.log10(sa)

        ff = np.fft.rfftfreq(len(a0), 1/sampling_frequency)

        if window != 1:
            ps = compute_running_mean(ps, window or len(a0) // 4096)

        args += [ff, ps]

    return plot(*args, **kwargs)


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
        self._done_decay = 5 * FS

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

        if self.index < FS / 8:
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
        
    @property
    def is_open(self):
        """bool: True if the gate is open now, or is scheduled to open.

        A gate that is open but has a close scheduled still counts as open 
        until the close actually happens.
        """
        return self.value == 1 or any(e == 'open' for _, e in self.states)

    def extend(self):
        """Cancel any scheduled close, so the gate stays open.

        This is what makes legato possible: a note that should run straight 
        into the next one cancels its scheduled close, and the next note 
        schedules a close of its own.
        """
        self.states = [s for s in self.states if s[1] != 'close']

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

        # True while the current note continues the previous one legato,
        # see play().
        self.legato = False
        
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
        # Each polyphonic note is a new note on a new copy, so it can never 
        # continue a previous note legato.
        kwargs.pop('legato', None)

        o = self.copy(track=True)
        o.play(note, duration, **kwargs)

        return o

    def play(self, note=None, duration=None, legato=False, **kwargs):
        """Play given note monophonically.

        If sound is already playing, it will be reset, unless legato is True.
        
        With legato=True and the sound still playing, the new note continues 
        the current one instead of starting over: the gate stays open, nothing 
        is reset, so envelopes, oscillator phases and filters carry on, and 
        only the note and the given properties change. If the sound is not 
        playing, the note starts normally.

        After the call, the ``legato`` attribute tells whether the note was 
        actually played legato. A subclass can use it to implement a slide,
        for example by gliding from the previous pitch to ``self.key`` in its
        ``forward()`` method, and to keep per-note settings that should only 
        change when a new note really starts:

        ::

            def play(self, note=None, duration=None, slide=False, **kwargs):

                super().play(note, duration, legato=slide, **kwargs)

                if not self.legato:
                    self._glide = self.key

        Args:
            note (float): Note to play in units of semitones 
                where 60 is middle C.
            duration (float, optional): Duration to play note, in whole notes.    
            legato (bool): Continue the currently playing note, if any, 
                instead of restarting the sound.
            **kwargs: Properties of intrument to modify.
        """
        if duration is None:
            duration = self.duration

        t = kwargs.pop('t', None)
        dt = kwargs.pop('dt', None)

        self.legato = legato and self.gate.is_open

        if self.legato:

            self.gate.extend()

            if note is not None:
                self.note = note

            self.set(**kwargs)

        else:
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
    df = max(math.ceil(dt * FS), 1)
    end = min(df, end if end is not None else 60 * FS)
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
    df = max(math.ceil(dt * FS), 1)
    end = min(df, end if end is not None else 60 * FS)
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
# Waveform generators.
#
# Imagine an array holding one cycle of a sine wave. Read it going around and
# around, taking every entry, and you get a low tone. Read it in big steps,
# skipping entries, and you go around faster, for a higher tone. The step size
# is the frequency, divided by the sampling frequency. To change the
# frequency, even from one sample to the next, just change the step size.
#
# The position in the cycle is called the phase. Steps rarely land exactly
# on an entry, so the two nearest entries are blended. For the sine wave, the
# value at the phase is simply computed with math.sin() instead.
#
# The sine wave is computed exactly. The triangle, sawtooth and square waves
# are sums of sine waves, their harmonics. They are read from tables that
# hold one cycle of each wave, made of only as many harmonics as fit below
# the Nyquist frequency, half the sampling frequency. Higher harmonics cannot
# be held by digital sound and would fold back down as out of tune tones, an
# effect called aliasing.
#
# The loops below are compiled by numba to fast machine code. Inside them, the
# phase is measured in cycles, from 0 to 1. Outside of them, it is measured in
# radians, from 0 to 2π, as it always was in the Oscillator API.
#


def _per_sample(value, frames):
    """Return a number or an array as an array with one float per sample."""

    # A number, e.g. freq=440: the same value at every sample.
    if np.isscalar(value):
        return np.full(frames, float(value))

    # An array, e.g. a gliding frequency computed by another sound, of
    # shape (frames, 1): one value per sample, flattened to (frames,).
    return np.asarray(value, dtype='float64').reshape(-1)


def _get_steps(freq, frames):
    """Return how far the phase advances at each sample, in cycles."""
    return _per_sample(freq / FS, frames)


def _get_nharmonics(freq):
    """Return the number of harmonics below the Nyquist frequency, up to 128.

    For a changing frequency, its highest value is used, so that no harmonic
    aliases.
    """
    if isinstance(freq, np.ndarray):
        freq = freq.max()

    freq = max(1., float(freq))
    return int(max(1, min(128, FS / 2 // freq)))


@numba.njit(cache=True)
def _sine(steps, phase):

    samples = np.empty(len(steps))

    for i in range(len(steps)):

        samples[i] = math.sin(2 * math.pi * phase)

        # Advance, and keep only the fraction of the cycle.
        phase += steps[i]
        phase %= 1.

    return samples, phase


@numba.njit(cache=True)
def _lookup(table, phase):
    """Read a waveform's cycle table at a phase, blending the two nearest entries.

    Args:
        table (ndarray): One cycle of a waveform, as values at equal steps
            of the cycle, e.g. 1024 values of one cycle of a sawtooth.
        phase (float): Where in the cycle to read, in cycles: 0 is the
            start, 0.5 halfway, and 1 is the start of the next cycle.
            Values outside 0 to 1 wrap around.

    Returns:
        float: The waveform's value at that phase, blended in a straight
            line between the two table entries around it.
    """

    size = len(table)

    # The position in the table falls between entries j and j + 1, at
    # fraction f of the way from one to the next.
    x = phase % 1. * size
    j = int(x)
    f = x - j

    a = table[j % size]
    b = table[(j + 1) % size]

    return a + f * (b - a)


@numba.njit(cache=True)
def _read_table(table, steps, phase):

    samples = np.empty(len(steps))

    for i in range(len(steps)):

        samples[i] = _lookup(table, phase)

        phase += steps[i]
        phase %= 1.

    return samples, phase


@numba.njit(cache=True)
def _read_pulse(sawtooth, steps, duties, phase):
    """Read a pulse wave from a sawtooth table, as the difference of two readers.

    A pulse wave is the difference of two sawtooth waves. Picture two
    readers going around the sawtooth table with the same steps, a duty
    of a cycle apart. The sawtooth drops once per cycle, so their
    difference jumps up as the reader in front passes the drop, and down
    as the one behind does: it is high for the duty fraction of the
    cycle. Adding 2 * duty - 1 makes it swing between -1 and 1.

    Only one phase is kept, with the readers half a duty on either side of
    it. When the duty changes, the readers move apart or together, and the
    gap between them always equals the duty exactly, with no rounding
    errors building up, as there would be with a phase for each.

    Args:
        sawtooth (ndarray): One cycle of a band-limited sawtooth, from
            get_sawtooth_cycle(), which drops at the middle of the table.
        steps (ndarray): How far the phase advances at each sample, in
            cycles.
        duties (ndarray): The duty at each sample: the fraction of the
            cycle the pulse is high, from 0 to 1.
        phase (float): The phase to start from, in cycles.

    Returns:
        tuple: The samples, and the phase to continue from.
    """
    samples = np.empty(len(steps))

    for i in range(len(steps)):

        #
        # The two readers, half a duty on either side of the phase. The 0.5
        # brings the sawtooth's drop, at the middle of its table, to the
        # start of the cycle, so the pulse is high around the start.
        #
        d = duties[i]
        a = _lookup(sawtooth, phase + 0.5 - d / 2)
        b = _lookup(sawtooth, phase + 0.5 + d / 2)
        samples[i] = a - b + 2 * d - 1

        phase += steps[i]
        phase %= 1.

    return samples, phase


def _get_table_size(nharmonics):
    """Return how many entries a cycle table with nharmonics harmonics gets.

    Reading between the entries of a table, in a straight line, adds a
    little noise. 24 entries for each cycle of the highest harmonic, and at
    least 768 in all, keep it 76dB or more below the sound, for every note.
    """
    return max(768, 24 * nharmonics)


@functools.lru_cache(maxsize=256)
def get_sawtooth_cycle(nharmonics, size=None):
    """Return one cycle of a band-limited sawtooth wave, as a table.

    It is the sum of its first nharmonics harmonics, where harmonic k is a
    sine wave with k times the frequency and 1/k of the amplitude. It starts
    at 0, rises to 1, drops to -1 at the middle of the cycle, and rises back.

    The table has size entries, by default as many as _get_table_size()
    says. They are stored as float32, which halves the memory, and is still
    far more precise than needed.
    """
    size = size or _get_table_size(nharmonics)
    radians = np.linspace(0, 2 * math.pi, size, endpoint=False)
    k = np.arange(1, nharmonics + 1)[:, None]

    table = 2 / math.pi * ((-1) ** (k + 1) / k * np.sin(k * radians)).sum(0)

    return table.astype('float32')


@functools.lru_cache(maxsize=256)
def get_triangle_cycle(nharmonics, size=None):
    """Return one cycle of a band-limited triangle wave, as a table.

    It is the sum of its odd harmonics up to harmonic nharmonics, where
    harmonic k is a cosine wave with k times the frequency and 1/k² of the
    amplitude. It starts at -1, rises to 1 at the middle of the cycle, and
    falls back.

    The table has size entries, by default as many as _get_table_size()
    says. They are stored as float32, which halves the memory, and is still
    far more precise than needed.
    """
    size = size or _get_table_size(nharmonics)
    radians = np.linspace(0, 2 * math.pi, size, endpoint=False)
    k = np.arange(1, nharmonics + 1, 2)[:, None]

    table = -8 / math.pi ** 2 * (np.cos(k * radians) / k ** 2).sum(0)

    return table.astype('float32')


def get_sine_wave(freq, phase=0, frames=8192, **kwargs):

    steps = _get_steps(freq, frames)
    samples, phase = _sine(steps, phase / 2 / math.pi)

    return samples, phase * 2 * math.pi


def get_triangle_wave(freq, phase=0, frames=8192, **kwargs):

    nharmonics = kwargs.get('nharmonics') or _get_nharmonics(freq)
    triangle = get_triangle_cycle(nharmonics, kwargs.get('size'))

    steps = _get_steps(freq, frames)
    samples, phase = _read_table(triangle, steps, phase / 2 / math.pi)

    return samples, phase * 2 * math.pi


def get_sawtooth_wave(freq, phase=0, frames=8192, sign=1., **kwargs):

    nharmonics = kwargs.get('nharmonics') or _get_nharmonics(freq)
    sawtooth = get_sawtooth_cycle(nharmonics, kwargs.get('size'))

    steps = _get_steps(freq, frames)
    samples, phase = _read_table(sawtooth, steps, phase / 2 / math.pi)

    if sign != 1.:
        samples *= sign

    return samples, phase * 2 * math.pi


def get_square_wave(freq, phase=0, frames=8192, duty=0.5, **kwargs):

    nharmonics = kwargs.get('nharmonics') or _get_nharmonics(freq)
    sawtooth = get_sawtooth_cycle(nharmonics, kwargs.get('size'))

    steps = _get_steps(freq, frames)
    duties = _per_sample(duty, len(steps)).clip(0, 1)

    if len(duties) != len(steps):
        raise ValueError('duty must have one value per sample, like freq.')

    samples, phase = _read_pulse(sawtooth, steps, duties, phase / 2 / math.pi)

    return samples, phase * 2 * math.pi


# Warmup: compile the loops now rather than while playing.
get_sine_wave(MIDDLE_C, frames=64)
get_triangle_wave(MIDDLE_C, frames=64)
get_sawtooth_wave(MIDDLE_C, frames=64)
get_square_wave(MIDDLE_C, frames=64)


class Oscillator(Sound):

    """Waveform generator for `sine`, and anti-aliased `triangle`, `sawtooth`,
    and variable duty `square` waveforms.

    Args:
        shape (str): Waveform to generate - one of `sine`, `triangle`, 
            `sawtooth`, or `square`.
        freq (float): Fundamental frequency of generator.
        key (float, optional): Fundamental frequency of generator in semitone
            units where middle C is 60.
        sign (float): Set to -1 to flip sawtooth waveform upside down.
        duty (float): The fraction of the square waveform cycle its value is 1,
            from 0 to 1.

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
            sign=sign,
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


def get_noise(color, frames=4096, state=None, kernel_size=2048, fs=FS):
    
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
def get_noise_kernel(color, kernel_size=8192, fs=FS):
    
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

