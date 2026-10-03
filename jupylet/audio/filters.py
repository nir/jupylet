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


import cmath
import functools
import logging
import math

import numba

import numpy as np

from ..audio import FS
from .sound import Sound, key2freq


logger = logging.getLogger(__name__)


#
# The filters below are loops over the samples of a block, compiled by numba
# to fast machine code. Each one takes a block and the state z it returned
# for the previous block, and returns the filtered block and its new state.
# Its controls, like the cutoff frequency, ramp linearly across the block,
# from their values at the end of the previous block, kept in z, to the new
# ones, so that they can be swept smoothly.
#
# Each compiled filter has a plain Python wrapper, which calls it with every
# argument by position: numba can add up to 13us to a call with keyword or
# left out arguments.
#
# A compiled filter is compiled on its first call, or loaded from numba's
# cache on disk, which can take a few seconds. So each filter class warms up
# its compiled filters when it is first created, rather than let that happen
# in the middle of playing sound.
#


class BaseFilter(Sound):
    """Base class for filters that carry their state from block to block.

    Subclasses implement :meth:`filter`, which filters a block of samples,
    given the filter state from the end of the previous block, and returns the
    state at the end of this one. A change in the cutoff frequency or in
    another control is ramped smoothly across the block by the filter itself.

    Filters take one channel, since instruments are mono up to their output,
    where they are panned.

    Args:
        freq (float): Cutoff frequency in Hz.
    """

    def __init__(self, freq=8192):

        super().__init__(freq=freq)

        self._z = None

    def reset(self, shared=False):

        super().reset(shared)

        self._z = None

    def forward(self, x, key_modulation=None):
        """Filter one block of samples.

        Args:
            x (ndarray): Block of samples of shape (frames, 1).
            key_modulation (float or ndarray, optional): Offset of the cutoff
                frequency from the filter's key, in semitones. If an array,
                the cutoff ramps to its value at the end of the block.

        Returns:
            ndarray: The filtered block, of the same shape as x.
        """
        if key_modulation is None:
            freq = self.freq
        elif isinstance(key_modulation, np.ndarray):
            freq = key2freq(self.key + np.mean(key_modulation[-1]).item())
        else:
            freq = key2freq(self.key + key_modulation)

        assert x.shape[1] == 1, 'A filter takes one channel, got %d.' % x.shape[1]

        out, self._z = self.filter(x[:, 0], float(freq), self._z)

        return out[:, None]

    def filter(self, x, freq, z=None):
        """Filter a block of samples.

        Subclasses override this method. The base implementation returns the
        samples unchanged.

        Args:
            x (ndarray): Samples to filter, of shape (frames,).
            freq (float): Cutoff frequency in Hz, at the end of the block.
            z (ndarray, optional): Filter state at the end of the previous
                block. If None, the filter starts from its initial state.

        Returns:
            tuple: The filtered samples, and the filter state at their end.
        """
        return x, None


#
# The chaser
#
# The simplest filter, from the examples/18-filters.ipynb notebook: a 1-pole
# lowpass filter, also known as a one-pole, a lag, or a leaky integrator. Its
# output chases its input, moving a fraction g of the way toward it at each
# sample.
#


def chaser(x, g, z=0.):
    """Filter a block of samples with a chaser, a 1-pole lowpass filter.

    At each sample, the output moves a fraction g of the way toward the
    input: z += g * (x[i] - z). So it follows the input slowly, rising and
    falling with a time constant of about 1 / g samples. For a time constant
    in seconds, use g = 1 / FS / seconds.

    It is also handy for smoothing control signals, like a capacitor, for
    example the accent sweep of a TB-303.

    Args:
        x (ndarray): A block of samples, of shape (frames,).
        g (float): The fraction to move at each sample, from 0 (never moves)
            to 1 (follows the input exactly).
        z (float): The output at the end of the previous block, to continue
            from.

    Returns:
        tuple: The filtered block, and its last value, to pass as z for the
            next block.
    """
    return _chaser(x, float(g), float(z))


@numba.njit(fastmath=True, cache=True)
def _chaser(x, g, z):

    out = np.empty(len(x))

    for i in range(len(x)):
        z += g * (x[i] - z)
        out[i] = z

    return out, z


# Warmup: compile the loop now rather than while playing.
chaser(np.zeros(16), 0.1)


#
# The Butterworth filter
#
# A Butterworth lowpass or highpass of a given order is a chain of 2-pole
# sections, with a 1-pole section at the end for an odd order. The poles of a
# Butterworth filter are spread evenly on a half circle, and each pair of them
# makes one section, with a damping k of 2cos() of the pair's angle.
#
# Each section is written in the zero-delay feedback (TPT) form, which keeps
# behaving well when the cutoff changes from sample to sample. With a fixed
# cutoff, the filter gives the same result as scipy.signal.butter().
#
# Resonance lowers the damping of the last 2-pole section, the one with the
# highest Q, so that it bumps at the cutoff; at 1 it rings by itself. At 0
# the filter is exactly Butterworth.
#


#
# One sample v through the chain of sections of the given order, with their
# states in s, a coefficient g set by the cutoff, and resonance r.
#
@numba.njit(fastmath=True, cache=True)
def _butter_chain(v, g, r, order, s, highpass):

    for j in range(order // 2):

        k = 2 * math.cos((2 * j + 1 + order % 2) * math.pi / (2 * order))

        if j == order // 2 - 1:
            k *= 1 - r

        a = 1 / (1 + g * (g + k))
        v1 = a * s[2*j] + g * a * (v - s[2*j+1])
        v2 = s[2*j+1] + g * v1
        s[2*j] = 2 * v1 - s[2*j]
        s[2*j+1] = 2 * v2 - s[2*j+1]

        v = v - k * v1 - v2 if highpass else v2

    if order % 2:
        v1 = g / (1 + g) * (v - s[-1])
        lp = v1 + s[-1]
        s[-1] = lp + v1
        v = v - lp if highpass else lp

    return v


#
# The state z: the g and resonance at the end of the previous block, and
# the states of the sections. It is changed in place, and returned.
#
@numba.njit(fastmath=True, cache=True)
def _butter_filter(x, cutoff, r1, order, highpass, z):

    g1 = math.tan(math.pi * min(max(cutoff, 1.), 0.49 * FS) / FS)
    r1 = min(max(r1, 0.), 1.)

    if z is None:
        z = np.zeros(2 + order)
        z[0], z[1] = g1, r1

    g, r = z[0], z[1]
    s = z[2:]

    dg = (g1 - g) / len(x)
    dr = (r1 - r) / len(x)

    out = np.empty(len(x))

    for i in range(len(x)):

        g += dg
        r += dr

        out[i] = _butter_chain(x[i], g, r, order, s, highpass)

    z[0], z[1] = g, r

    return out, z


#
# The frequency g and damping k of each 2-pole section of a bandpass with n
# poles on each side, and the overall gain that makes the center 0dB. Each
# pole p of the lowpass turns into the two roots of s² - pbs + w0² = 0, where
# w0 is the center and b the width of the band, both prewarped like g, and
# each root makes a section together with its mirror image (complex
# conjugate). The real pole of an odd n turns into a single section.
#
@numba.njit(cache=True)
def _butter_bandpass_sections(n, lo, hi):

    wl = math.tan(math.pi * lo / FS)
    wh = math.tan(math.pi * hi / FS)
    w0 = math.sqrt(wl * wh)
    b = wh - wl

    g = np.empty(n)
    k = np.empty(n)
    j = 0

    for m in range(n // 2):
        p = cmath.exp(1j * math.pi * (2 * m + 1 + n) / (2 * n))
        d = cmath.sqrt(p * p * b * b - 4 * w0 * w0)
        for q in ((p * b + d) / 2, (p * b - d) / 2):
            g[j] = abs(q)
            k[j] = -2 * q.real / abs(q)
            j += 1

    if n % 2:
        g[j] = w0
        k[j] = b / w0

    return g, k, b ** n / np.prod(g)


#
# A bandpass with n poles on each side, and its edges at cutoff -
# bandwidth/2 and cutoff + bandwidth/2. The state z: the gain c, and the g
# and k of each section at the end of the previous block, and the states
# of the sections. It is changed in place, and returned.
#
@numba.njit(fastmath=True, cache=True)
def _butter_bandpass(x, cutoff, n, bandwidth, z):

    lo = min(max(cutoff - bandwidth / 2, 1.), 0.49 * FS)
    hi = min(max(cutoff + bandwidth / 2, lo + 1.), 0.49 * FS)

    g1, k1, c1 = _butter_bandpass_sections(n, lo, hi)

    if z is None:
        z = np.zeros(1 + 4 * n)
        z[0] = c1
        z[1:1+n] = g1
        z[1+n:1+2*n] = k1

    c = z[0]
    g, k, s = z[1:1+n], z[1+n:1+2*n], z[1+2*n:]

    dc = (c1 - c) / len(x)
    dg = (g1 - g) / len(x)
    dk = (k1 - k) / len(x)

    out = np.empty(len(x))

    for i in range(len(x)):

        c += dc
        v = x[i]

        for j in range(n):

            g[j] += dg[j]
            k[j] += dk[j]

            a = 1 / (1 + g[j] * (g[j] + k[j]))
            v1 = a * s[2*j] + g[j] * a * (v - s[2*j+1])
            v2 = s[2*j+1] + g[j] * v1
            s[2*j] = 2 * v1 - s[2*j]
            s[2*j+1] = 2 * v2 - s[2*j+1]

            v = v1

        out[i] = c * v

    z[0] = c

    return out, z


def butter_filter(
    x,
    cutoff,
    mode='lowpass',
    order=4,
    bandwidth=500.,
    resonance=0.,
    z=None,
):
    """Filter a block of samples with a Butterworth filter.

    Call it block after block, passing back the returned state z. The cutoff,
    bandwidth and resonance ramp linearly across each block, from the previous
    block's values to the new ones, so they can be swept smoothly.

    Args:
        x (ndarray): A block of samples, of shape (frames,).
        cutoff (float): Cutoff frequency in Hz, or the center of the band.
        mode (str): 'lowpass', 'highpass' or 'bandpass'.
        order (int): The number of poles. For a bandpass it must be even,
            half of them on each side of the band.
        bandwidth (float): The width of the band in Hz, for a bandpass.
        resonance (float): From 0, exactly Butterworth, to 1, where the
            filter rings by itself. It has no effect on a bandpass, or at
            order 1.
        z (ndarray, optional): The state returned by the previous call, or
            None to start afresh.

    Returns:
        tuple: The filtered block, and the state z for the next call.
    """
    if mode == 'bandpass':

        if order < 2 or order % 2:
            raise ValueError('A bandpass order must be even, got %r.' % order)

        return _butter_bandpass(x, cutoff, order // 2, bandwidth, z)

    if mode not in ('lowpass', 'highpass'):
        raise ValueError('Unknown mode %r.' % mode)

    if order < 1:
        raise ValueError('The order must be at least 1, got %r.' % order)

    return _butter_filter(x, cutoff, resonance, order, mode == 'highpass', z)


@functools.lru_cache(maxsize=None)
def _warmup_butter():

    x = np.zeros(16)

    for mode, order in (('lowpass', 4), ('bandpass', 4)):
        _, z = butter_filter(x, 1000., mode, order, 500., 0., None)
        butter_filter(x, 1000., mode, order, 500., 0., z)


class ButterFilter(BaseFilter):
    """A Butterworth filter, with optional resonance.

    At resonance 0 it is exactly a Butterworth filter, flat up to its cutoff.
    Resonance raises the Q of its last 2-pole section, as in synth filters, so
    that it bumps at the cutoff, and at 1 it rings by itself.

    Args:
        freq (float): Cutoff frequency in Hz, or the center of the band.
        mode (str): 'lowpass', 'highpass' or 'bandpass'.
        order (int): The number of poles, each adding about 6dB per octave to
            the slope. For a bandpass it must be even, half of them on each
            side of the band.
        bandwidth (float): The width of the band in Hz, for a bandpass.
        resonance (float): From 0 to 1. It has no effect on a bandpass, or at
            order 1.
    """

    def __init__(
        self,
        freq=8192,
        mode='lowpass',
        order=4,
        bandwidth=500,
        resonance=0.,
    ):

        super().__init__(freq)

        self.mode = mode
        self.order = order
        self.bandwidth = bandwidth
        self.resonance = resonance

        self._shape = None

        _warmup_butter()

    def filter(self, x, freq, z=None):

        #
        # The size and layout of the state depend on the order, and on
        # whether it is a bandpass, so start afresh when either changes.
        #
        shape = (self.mode == 'bandpass', int(self.order))

        if self._shape != shape:
            self._shape = shape
            z = None

        return butter_filter(
            x,
            freq,
            self.mode,
            int(self.order),
            float(self.bandwidth),
            float(self.resonance),
            z,
        )


#
# The ladder filter
#
# The filter built step by step in the examples/18-filters.ipynb notebook: a
# chain of four chasers (1-pole lowpass filters) with the output fed back to
# the input, and every stage saturating like the transistors of the Moog
# ladder. Highpass and bandpass, and lower orders, are mixes of the input and
# the outputs of the four stages, as in the Oberheim Xpander.
#
# At quality, it runs at twice the sample rate, going up and down with a
# half-band filter, which reduces the aliasing of the saturation, and brings
# its resonance closer to the analog one.
#


# A fast approximation of math.tanh, a Padé approximant, within about
# 1e-4 of it. Past 4.97 it is simply 1, and -1 below -4.97.
@numba.njit(fastmath=True, inline='always', cache=True)
def _tanh77(x):
    if x > 4.97: return 1.0
    if x < -4.97: return -1.0
    x2 = x*x
    return (x*(135135 + x2*(17325 + x2*(378 + x2))) /
            (135135 + x2*(62370 + x2*(3150 + x2*28))))


# The g that gives a chaser the given cutoff frequency, in Hz, at the
# sample rate fs.
@numba.njit(cache=True)
def _cutoff2g(cutoff, fs):
    return 1 - math.exp(-2 * math.pi * cutoff / fs)


#
# A half-band lowpass filter with 31 weights, called taps: a sinc, the
# ideal lowpass, cut at 22050Hz for 88200 samples per second, and smoothed
# at its ends by a Kaiser window. It is used to go up to 88200 samples per
# second, and back down to 44100.
#
# Every other tap is exactly zero, except the center one, which is 0.5. So
# only the 16 other taps, kept in _TAPS, are ever multiplied, and the center
# tap is applied by itself, as 0.5.
#
_n = np.arange(31) - 15
_halfband = 0.5 * np.sinc(_n / 2) * np.kaiser(31, 8.)
_halfband /= _halfband.sum()
_TAPS = _halfband[0::2].copy()


#
# The half-band filter's weighted sum of the last 16 samples in the ring
# buffer b, the newest at position p. It applies the 16 taps of _TAPS; the
# center tap is applied by the caller.
#
@numba.njit(fastmath=True, inline='always', cache=True)
def _halfband_sum(b, p):
    a = 0.
    for j in range(16):
        a += _TAPS[j] * b[(p - j) & 15]
    return a


#
# One step of the ladder, for one input sample xk. It takes the stage
# outputs y, and returns the output, mixed as m says, and the new y.
#
@numba.njit(fastmath=True, inline='always', cache=True)
def _ladder_step(xk, y, g, r, drive, m):

    y1, y2, y3, y4 = y

    # The input and every stage go through tanh, like the transistors of
    # the ladder.
    u = _tanh77(drive * xk - r * y4)
    y1 += g * (u - _tanh77(y1))
    y2 += g * (_tanh77(y1) - _tanh77(y2))
    y3 += g * (_tanh77(y2) - _tanh77(y3))
    y4 += g * (_tanh77(y3) - _tanh77(y4))

    #
    # Make up for the level lost at low drive, where the filter is nearly
    # linear, so its output is the drive times the full level. Above 1,
    # driving harder is meant to be louder.
    #
    makeup = 1 / min(drive, 1.)
    out = (m[0] * u + m[1] * y1 + m[2] * y2 + m[3] * y3 + m[4] * y4) * makeup

    return out, (y1, y2, y3, y4)


#
# How much of the input u and of each stage y1 to y4 to mix, for each mode
# and order.
#
_LADDER_MIXES = {
    ('lowpass', 1): (0., 1., 0., 0., 0.),
    ('lowpass', 2): (0., 0., 1., 0., 0.),
    ('lowpass', 3): (0., 0., 0., 1., 0.),
    ('lowpass', 4): (0., 0., 0., 0., 1.),
    ('highpass', 1): (1., -1., 0., 0., 0.),
    ('highpass', 2): (1., -2., 1., 0., 0.),
    ('highpass', 3): (1., -3., 3., -1., 0.),
    ('highpass', 4): (1., -4., 6., -4., 1.),
    ('bandpass', 2): (0., 2., -2., 0., 0.),
    ('bandpass', 4): (0., 0., 4., -8., 4.),
}


#
# The compiled ladder, called by ladder_filter(). It takes the target
# cutoff, r1 and drive1 for the end of the block, and the mix m, the
# weights of the input u and of the stages y1 to y4.
#
@numba.njit(fastmath=True, cache=True)
def _ladder_filter(x, cutoff, r1, drive1, m, quality, z):

    # At quality, the ladder runs at twice the sample rate.
    steps = 2 if quality else 1
    g1 = _cutoff2g(cutoff, FS * steps)

    #
    # The state z: y1 to y4, the g, r and drive of the previous block, the
    # position of the ring buffers, and the three ring buffers of the
    # quality path: the input, and the two outputs of each input sample.
    # It is changed in place, and returned.
    #
    if z is None:
        z = np.zeros(56)
        z[4], z[5], z[6] = g1, r1, drive1

    y = (z[0], z[1], z[2], z[3])
    g, r, drive = z[4], z[5], z[6]
    xb, vb0, vb1 = z[8:24], z[24:40], z[40:56]
    p = int(z[7])

    # Steps to ramp g, r and drive from the previous block's to the new
    # ones, across the block.
    dg = (g1 - g) / len(x)
    dr = (r1 - r) / len(x)
    dd = (drive1 - drive) / len(x)

    out = np.zeros(len(x))

    for i in range(len(x)):

        g += dg
        r += dr
        drive += dd

        if quality:

            # Keep the last 16 input samples in the ring buffer xb.
            p = (p + 1) & 15
            xb[p] = x[i]

            #
            # Up to 88200: two new samples for every input sample. The
            # first is the half-band filter's weighted sum. The second needs
            # only the center tap, so it is simply the input from 7 samples
            # ago.
            #
            x0 = 2 * _halfband_sum(xb, p)
            x1 = xb[(p - 7) & 15]

            vb0[p], y = _ladder_step(x0, y, g, r, drive, m)
            vb1[p], y = _ladder_step(x1, y, g, r, drive, m)

            #
            # Down to 44100: the same filter, with the same shortcut, and
            # only one output for every two.
            #
            out[i] = _halfband_sum(vb0, p) + 0.5 * vb1[(p - 8) & 15]

        else:
            out[i], y = _ladder_step(x[i], y, g, r, drive, m)

        #
        # WARNING: do not remove this check. It stops the filter as soon as
        # its output passes 3, protecting you from a runaway sound blasting
        # out at full volume.
        #
        if abs(out[i]) > 3:
            raise ValueError('The filter ran away: its output passed 3.')

    z[:4] = y
    z[4], z[5], z[6] = g, r, drive
    z[7] = p

    return out, z


def ladder_filter(
    x,
    cutoff,
    resonance=0.,
    drive=1.,
    mode='lowpass',
    order=4,
    quality='performance',
    z=None,
):
    """Filter a block of samples with a saturating ladder filter.

    Call it block after block, passing back the returned state z. The cutoff,
    resonance and drive ramp linearly across each block, from the previous
    block's values to the new ones, so they can be swept smoothly.

    Args:
        x (ndarray): A block of samples, of shape (frames,).
        cutoff (float): Cutoff frequency in Hz.
        resonance (float): How much of the output is fed back. Around 4 the
            filter rings by itself.
        drive (float): Input gain into the saturation. Below 1 the filter is
            nearly linear, and its level is made up; above 1 it saturates and
            gets louder.
        mode (str): 'lowpass', 'highpass' or 'bandpass'.
        order (int): The number of poles, 1 to 4 for lowpass and highpass,
            2 or 4 for bandpass.
        quality (str): 'performance', or 'quality' to run the ladder at twice
            the sample rate, with less aliasing and a resonance closer to the
            analog one, at about twice the cost and a delay of 15 samples.
            Choose it before playing: switching it mid-sound can click.
        z (ndarray, optional): The state returned by the previous call, or
            None to start afresh.

    Returns:
        tuple: The filtered block, and the state z for the next call.

    Raises:
        ValueError: If the output passes 3, a runaway.
    """
    if (mode, order) not in _LADDER_MIXES:
        raise ValueError('No ladder mix for mode %r and order %r.' % (mode, order))

    if quality not in ('performance', 'quality'):
        raise ValueError('Unknown quality %r.' % quality)

    return _ladder_filter(
        x,
        cutoff,
        resonance,
        drive,
        _LADDER_MIXES[mode, order],
        quality == 'quality',
        z,
    )


@functools.lru_cache(maxsize=None)
def _warmup_ladder():

    x = np.zeros(16)

    _, z = ladder_filter(x, 1000., 0., 1., 'lowpass', 4, 'performance', None)
    ladder_filter(x, 1000., 0., 1., 'lowpass', 4, 'performance', z)


class LadderFilter(BaseFilter):
    """A saturating ladder filter, like the Moog ladder.

    See the examples/18-filters.ipynb notebook for how it works.

    Args:
        freq (float): Cutoff frequency in Hz.
        mode (str): 'lowpass', 'highpass' or 'bandpass'.
        order (int): The number of poles, 1 to 4 for lowpass and highpass,
            2 or 4 for bandpass.
        resonance (float): How much of the output is fed back, from 0. Around
            4 the filter rings by itself.
        drive (float): Input gain into the saturation.
        quality (str): 'performance', or 'quality' for less aliasing at about
            twice the cost. Choose it before playing: switching it mid-sound
            can click.
    """

    def __init__(
        self,
        freq=8192,
        mode='lowpass',
        order=4,
        resonance=0.,
        drive=1.,
        quality='performance',
    ):

        super().__init__(freq)

        self.mode = mode
        self.order = order
        self.resonance = resonance
        self.drive = drive
        self.quality = quality

        _warmup_ladder()

    def filter(self, x, freq, z=None):
        return ladder_filter(
            x,
            freq,
            float(self.resonance),
            float(self.drive),
            self.mode,
            int(self.order),
            self.quality,
            z,
        )
