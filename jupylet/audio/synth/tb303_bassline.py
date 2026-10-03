"""
    jupylet/audio/synth/tb303_bassline.py

    TB303, a replica of the Roland TB-303 Bass Line.

    Copyright (c) 2026, Nir Aides - nir.8bit@gmail.com

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

"""                   
    THE TB-303

    Roland released the TB-303 Bass Line in 1981, a small silver box meant to
    play bass lines for guitarists practicing on their own. It
    sounded little like a bass guitar, and was hard to program, so it sold
    poorly and was discontinued after a few years. Second hand, it became
    cheap, and in the mid 1980s producers in Chicago found that turning its
    knobs while a pattern plays makes the bass squelch, chirp and scream.
    That sound became acid house, starting with Phuture's Acid Tracks in
    1987, and it has run through electronic music ever since. (History:
    https://en.wikipedia.org/wiki/Roland_TB-303)

    It plays one note at a time (it is monophonic), from patterns of up to
    16 steps stored in its built-in step sequencer. Each step holds a note,
    and may also carry an accent, which makes it louder and brighter, and a
    slide, which glides to the next note's pitch instead of jumping.


    HOW IT IS BUILT

    The TB-303 is an analog synthesizer: every sound and every control in it
    is a voltage that changes over time. Its signal path is short:

        sequencer --- pitch ---> slide ---> VCO ---> VCF ---> VCA ---> out
            |                                         ^        ^
            | gate                                    |        |
            +----> MEG --+--- Env Mod --------------->+        |
            |            |                            |        |
            |            +--> accent --+--> sweep ----+        |
            |                 switch   |                       |
            |                 & knob   +--> volume ----------->+
            |                                                  |
            +----> VEG --------------------------------------->+

    - The VCO, the voltage controlled oscillator, makes the raw sound: a
      sawtooth or a square wave, at the pitch set by its control voltage.

    - The VCF, the voltage controlled filter, takes away the harmonics above
      its cutoff frequency, the brightness of the sound. Most of the TB-303's
      character comes from moving that cutoff.

    - The VCA, the voltage controlled amplifier, sets the volume.

    - The two envelope generators make the voltages that move the VCF and
      the VCA on every note: the MEG, the main envelope generator, for the
      cutoff, and the VEG, the volume envelope generator, for the volume.
    
    - On accented notes, a part of the MEG's voltage is let through a
      switch and the Accent knob, and drives two small circuits: the accent
      sweep, which raises the cutoff further, and a path into the VCA,
      which makes the note louder.

    Besides its volume, six knobs shape the sound: Tuning, Cut Off Freq,
    Resonance, Env Mod, Decay and Accent. A switch picks the waveform.


    CONCEPTS

    Control voltage (CV). A voltage that sets a parameter of the sound
    rather than being heard. The TB-303's pitch is a CV of one volt per
    octave: one more volt doubles the frequency. Here, CVs are numbers:
    the pitch is a key, in semitones, and the cutoff's modulation is in
    octaves.

    Gate. A voltage that is high while a note is held, and low between
    notes. The envelopes start when it rises.

    Envelope. A voltage that shapes a note over time. Most synthesizers
    use an ADSR envelope: it rises at the start of a note (attack), falls
    to a level (decay), stays there while the key is held (sustain), and
    fades out when the key is let go (release). The TB-303's envelopes are
    simpler: they jump up at the start of each note, and then only decay.
    Letting go of the note does not change them. The Decay knob sets how
    fast the MEG decays; the VEG's decay is fixed and long.

    Cutoff and resonance. A lowpass filter lets through the frequencies
    below its cutoff, and takes away those above it. Resonance feeds part
    of the filter's output back to its input, which boosts the frequencies
    near the cutoff into a whistling peak. Sweeping the cutoff with high
    resonance is the TB-303's signature sound.

    Env Mod. The envelope modulation: how far the MEG sweeps the cutoff.
    At the start of each note, the MEG lifts the cutoff above where the
    Cut Off Freq knob sets it, and as the MEG decays, the cutoff sinks
    back, down to somewhat below the knob's setting. The Env Mod knob sets
    how far, in octaves, each one a doubling of the frequency: at its
    minimum, the cutoff starts about half an octave above the knob's
    setting, and at its maximum, over 3 octaves above, about 9 times the
    frequency [O]. Even at its minimum the sweep does not go away [W].

    Accent. On an accented note, the MEG decays at its fastest, and the
    accent signal makes the note louder and sweeps the cutoff higher. On
    several accented notes in a row, the sweep builds up from note to note,
    more so at high resonance (see accent_sweep_circuit()).

    Slide. A note with a slide keeps its gate high into the next note, so
    the envelopes do not restart, and its pitch glides to the next note's.

    RC circuit and time constant (TAU, the Greek letter τ). The TB-303's
    slide, envelopes and accent circuits are each built from resistors (R,
    in ohms) and capacitors (C, in farads). A capacitor is a small tank of
    charge, and its voltage chases a target: fast while far from it, slower
    as it gets close, like a hot drink cooling to room temperature. The
    resistor sets how fast: more resistance, slower chase.

    R times C is the circuit's time constant, TAU, in seconds: the one
    number that tells how slow the circuit is. Since the voltage changes
    more slowly as it nears its target, it never quite gets there, so we
    measure how far it gets: in one TAU, 63% of the way; in about 2.3 TAU,
    90%; and in 3 TAU, 95%, close enough to call it done. For example, the
    slide charges a 0.22 uF (microfarad, a millionth of a farad) capacitor
    through 100k ohms (100,000 ohms), so its TAU is 100,000 x 0.00000022 =
    0.022 seconds, or 22 ms (SLIDE_TAU), and a slide is done in about 3 x
    22 = 66 ms.

    This module models these circuits from their parts, using the values on
    the TB-303's schematic, and solves them sample by sample.

    Potentiometer, or pot. The part behind a knob: a resistor with a
    contact, the wiper, that slides along it as the knob turns. The wiper
    splits the resistor in two, one part on each side of it, and turning
    the knob makes one part longer and the other shorter. On the schematic,
    pots are named VR, for variable resistor, with a number: VR4 is the
    Resonance knob's pot. It is a dual pot: two pots stacked on the knob's
    one axle, so that turning the knob turns both at once. One sets the
    filter's resonance; the other is part of the accent sweep circuit.

    Summing point. Where several control voltages are added up into one,
    such as the cutoff's: each comes in through its own resistor, and an
    amplifier adds up their currents (see diode_rc()).


    HOW TB303 MODELS IT

    Every value in the code is tagged with its source, and whatever no
    source gives is marked UNSOURCED. The knobs go from 0 to 1, like the
    knobs on the instrument. The main departure from the original is the
    filter: TB303 uses jupylet's LadderFilter, a model of the Moog-style
    transistor ladder, not of the TB-303's own diode ladder.

    Everything found about the original instrument, its sources, and what is
    still unknown, is gathered in other-docs/TB-303.md.

    Roland and TB-303 are trademarks of Roland Corporation. This module is
    an independent replica, not affiliated with or endorsed by Roland.


    SOURCES, for the original TB-303, not its modifications:

    [W] Robin Whittle, TB-303's unique characteristics, and TB-303 Slide:
        https://www.firstpr.com.au/rwi/dfish/303-unique.html
        https://www.firstpr.com.au/rwi/dfish/303-slide.html

    [M] Robin Whittle, the Devil Fish manual, its statements about the
        stock TB-303:
        https://archive.org/stream/manualzz-id-1172463/1172463_djvu.txt

    [O] Robin Schmidt, Open303, an emulation whose cutoff and env mod
        mapping is marked as measured:
        https://github.com/RobinSchmidt/Open303

    [S] Roland, TB-303 service notes, main board schematic (page 5):
        http://machines.hyperreal.org/manufacturers/Roland/TB-303/
        schematics/roland.TB-303.schem-5.gif

    Where they disagree, the schematic wins: its Resonance pot, VR4, is a
    dual 50k [S], not the 100k in [W]'s description.
"""


import math
import numba

import numpy as np

from .. import FS, DEFAULT_AMP
from ..sound import GatedSound, Envelope, Oscillator, DecayEnvelope, key2freq
from ..filters import LadderFilter
from ..sequencer import StepSequencer


@numba.njit
def diode_rc(x, r1, ra, rb, rm, c, v):
    
    r"""A diode RC network of the TB-303, solved exactly sample by sample.

              D        r1       ra    node     rm
    x o------|>|------/\/------/\/-----+------/\/------o summing point
                                       |                 (held at 0 V)
                                       |
                                  rb  /\/
                                       |
                                       |
                                   + ----- v
                                     ----- c
                                       |
                                      GND

    The diode D is a one-way valve: current flows through it only from x
    toward the node. The resistors r1, ra, rb and rm slow the current down.
    The capacitor c is a small tank of charge: its voltage v rises as it
    fills, and falls as it empties.

    While the voltage x is high, current flows in through the diode, and
    the capacitor fills. When x drops, the diode shuts, and the capacitor
    slowly empties through rb and rm. So v holds on to past input, and
    fades away.

    The summing point is where the signals of several circuits are added
    up. An amplifier holds it at 0 V, and what it adds up is the current
    flowing into it from each circuit. Here, that is the current through
    rm, which is the node's voltage divided by rm. So diode_rc() returns
    the node's voltage, which rises and falls with that current.

    Each sample is solved in three steps:

    1. Is the diode open? With it shut, the node sits between the
       capacitor and the summing point, at v * rm / (rb + rm). The diode
       opens when x is above that.
       
    2. Fill or empty the capacitor. It heads toward a target voltage,
       vth: where it would settle if x stayed as it is. It fills or
       empties through a resistance rth, and gets 63% of the way there in
       rth * c seconds, its time constant. The code moves it over one
       sample exactly as a real capacitor would.
       
    3. Find the node's voltage. The current into the capacitor flows
       through rb, so the node sits rb times that current above v.

    Args:
        x (ndarray): The signal, of shape (frames,).
        r1, ra, rb, rm (float): The resistors, in ohms.
        c (float): The capacitor, in farads.
        v (float): The capacitor's voltage at the end of the previous block.

    Returns:
        tuple: The node's voltage, which drives the summing point, and the
            capacitor's voltage at the end of the block.
    """
    rs = r1 + ra
    out = np.empty(len(x))

    for i in range(len(x)):

        #
        # With the diode off, the node is on a divider between the
        # capacitor and the summing point. The diode conducts while x
        # is above that.
        #
        if x[i] > v * rm / (rb + rm):
            vth = x[i] * rm / (rs + rm)
            rth = rb + rs * rm / (rs + rm)
            
        else:
            vth = 0.
            rth = rb + rm

        #
        # The capacitor moves toward vth through rth, exactly as an RC
        # circuit does over one sample, and the node is rb away from
        # it.
        #
        v = vth + (v - vth) * math.exp(-1 / FS / (rth * c))
        
        out[i] = v + (vth - v) / rth * rb

    return out, v


def accent_sweep_circuit(x, resonance, v):
    
    """The TB-303's accent sweep: on accented notes, it adds to the MEG's
    sweep of the filter's cutoff, raising it further, so the note sounds
    brighter [W, S].

    Its input, the accent signal, is the MEG's own voltage: a switch lets
    it through on accented notes only, and the Accent knob scales it [W].
    So it jumps up at the start of the note, and decays with the MEG. It
    charges a capacitor, which then slowly drains, and the cutoff rises
    with the result.

    The Resonance knob changes how the sweep behaves too. Its pot, VR4
    (see CONCEPTS, at the top of this module), is a dual pot: the knob
    turns two pots at once, one that sets the filter's resonance, and one
    that is part of this circuit.

    - At low resonance, the sweep follows the accent itself: each accented
      note rises and falls on its own.
      
    - At high resonance, the sweep follows the capacitor. It drains slowly,
      so on accented notes in a row, each starts with charge left over from
      the one before, and the sweep climbs higher and higher [W].

    It is diode_rc() with the parts on the schematic [S]: the diode D24; r1,
    the 47k resistor R46; ra and rb, the two halves of VR4's second section,
    50k, on either side of its wiper, which is the node; c, the 1uF
    capacitor C13; and rm, the 100k resistor into the cutoff's summing
    point. The pot is taken as linear, as its B marking says [S].

    Args:
        x (ndarray): The accent signal, of shape (frames,).
        resonance (float): The Resonance knob, from 0 to 1.
        v (float): The capacitor's voltage at the end of the previous block.

    Returns:
        tuple: The sweep, and the capacitor's voltage at the end of the
            block.

    UNSOURCED: the diode is taken as ideal, with no voltage drop. A real
    silicon diode conducts only once the voltage across it reaches about
    0.6V, which gives the sweep a threshold, and cuts its tail short as the
    envelope decays. Modeling that needs the accent signal in real volts,
    not yet known: here it is the filter envelope times the Accent knob,
    from 0 to 1. The same holds for D27 in accent_volume_circuit().
    """
    pot = 50e3
    return diode_rc(
        x, 
        47e3,                    # r1
        pot * resonance,        # ra
        pot * (1 - resonance), # rb
        100e3,                # rm
        1e-6,                # c
        v
    )


def accent_volume_circuit(x, v):
    """The accent's path into the VCA, which makes accented notes louder
    [W, S].

    It is diode_rc() with the parts on the schematic [S]: the diode D27;
    r1, the 22k resistor R120; c, the 0.033uF capacitor C36; and rm, the
    47k resistor R119 into the VCA's control input. There is no pot, so ra
    and rb are 0. The capacitor charges in about 0.5ms, which softens the
    accent's attack a little [W].

    UNSOURCED: the diode is taken as ideal, as in accent_sweep_circuit().

    Args:
        x (ndarray): The accent signal, of shape (frames,).
        v (float): The capacitor's voltage at the end of the previous block.

    Returns:
        tuple: The accent's volume, and the capacitor's voltage at the end
            of the block.
    """
    return diode_rc(
        x, 
        22e3,         # r1
        0.,          # ra
        0.,         # rb
        47e3,      # rm
        0.033e-6, # c
        v
    )


class TB303(GatedSound, StepSequencer):
    """A replica of the Roland TB-303 Bass Line, built around the ladder filter.

    The knobs go from 0 to 1, like the knobs on the real instrument. Each
    part is modeled after the sources listed in this module's docstring,
    which also introduces the TB-303 and the concepts used here, and
    whatever they do not give is marked UNSOURCED in the code.

    Play notes with play(), with accented=True for an accent, and
    slide=True to glide from the previous note. As a StepSequencer, it can
    also play patterns, with /a for an accent and /s for a slide, a dot (.)
    for a rest, and a dash (-) to hold the note before through the step.
    For example:

    ::

        tb = TB303(cutoff=0.2, resonance=0.8, env_mod=0.6)

        tb.sonic_live_loop('''

            C2/a  C2    C3/s  C2
            C2    Eb2/a C2    C3/s
            C2/a  .     Bb2/s C2
            G2/a  C2    C3/s  -

        ''')

        tb.stop()

    Args:
        waveform (str): `sawtooth` or `square`.
        cutoff (float): Filter cutoff knob.
        resonance (float): Filter resonance knob.
        env_mod (float): How far the filter envelope sweeps the cutoff.
        decay (float): Filter envelope decay knob.
        accent (float): How strongly accented notes are emphasized.
        tuning (float): Pitch offset from one octave down (0) to one octave
            up (1), in tune at 0.5 (UNSOURCED range).
    """

    # The slide: "the 100K resistance of the DAC feeding a 0.22uF
    # capacitor" [M], an RC time constant of 22ms.
    SLIDE_TAU = 100e3 * 0.22e-6

    #
    # The filter envelope (MEG). Unlike the two time constants (TAU)
    # around it, this one is not computed as R x C: the schematic prints
    # the intended decay time, not the parts that set it [S]. It says the
    # Decay knob sets the time to fall to 10% of the start, from 200ms to
    # 2.5s, and [W] gives 200ms on accented notes, the knob's minimum.
    # A decay falls to 10% in ln(10) = 2.3 TAU, so TAU = T / ln(10).
    #
    DECAY_TAU_MIN = 0.2 / math.log(10)
    DECAY_TAU_MAX = 2.5 / math.log(10)

    # The volume envelope (VEG): R123, 1.5M, and C42, 1uF [S].
    VOLUME_TAU = 1.5e6 * 1e-6

    #
    # How much the accent adds to the volume: it comes in through
    # R119, 47k, and the volume envelope through R131, 220k [S], so
    # per volt the accent counts 220 / 47 times as much. (UNSOURCED:
    # the two signals' voltages, so this ratio is taken as the whole
    # gain.)
    #
    ACCENT_VOLUME = 220e3 / 47e3

    #
    # How far the accent sweep raises the cutoff, in octaves, per unit
    # of accent, taken from Open303 [O]. (UNSOURCED: it depends on the
    # voltages of the envelope and of the accent signal, not yet
    # known.)
    #
    ACCENT_SWEEP_OCTAVES = 1.

    MAX_RESONANCE = 3.9

    # Pattern flags for the built-in step sequencer: /a for accent, /s for slide.
    flags = {'a': 'accented', 's': 'slide'}

    def __init__(
        self,
        waveform='sawtooth',
        cutoff=0.3,
        resonance=0.5,
        env_mod=0.5,
        decay=0.5,
        accent=0.5,
        tuning=0.5,
        amp=DEFAULT_AMP,
        pan=0.,
        duration=None,
        quality='performance',
    ):

        super().__init__(amp=amp, pan=pan, duration=duration)

        self.waveform = waveform
        self.cutoff = cutoff
        self.resonance = resonance
        self.env_mod = env_mod
        self.decay = decay
        self.accent = accent
        self.tuning = tuning

        #
        # The note's start and end: a 3ms attack, and an 8ms linear
        # fade when the gate closes [M]. (The 4ms delay before the
        # note starts, and the 8ms at full volume before the fade, are
        # left out.) The schematic smooths the volume envelope's
        # trigger over 2.2ms, R134 22k and C41 0.1uF [S], which agrees
        # with the 3ms attack.
        #
        self.env0 = Envelope(0.003, 0., 1., 0.008)

        #
        # The filter envelope (MEG) and the volume envelope (VEG).
        # Both start sharply at each note, and decay whether the note
        # is held or not [W]. The MEG's capacitor, C62 1uF, charges
        # through R152, 100 ohms, in 0.1ms [S], instantly.
        #
        self.meg = DecayEnvelope(self.DECAY_TAU_MIN)
        self.veg = DecayEnvelope(self.VOLUME_TAU)

        self.osc0 = Oscillator(waveform)

        self.filter = LadderFilter(quality=quality)

        #
        # The circuits' state, which must carry over from note to note.
        # play() resets only the child sounds (the envelopes, the
        # oscillator and the filter), so it does not touch these:
        #
        # - _glide: the pitch the slide has reached, for a slid note to
        #   glide on from.
        # - _accent_v: the voltage of the accent sweep's capacitor. Its
        #   leftover charge is what builds up over accents in a row.
        # - _accent_volume: the voltage of the capacitor in the accent's
        #   volume path.
        #
        self._glide = None
        self._accent_v = 0.
        self._accent_volume = 0.

        # The current note's settings, set by play(): whether it is
        # accented, and the MEG's decay time constant.
        self._accented = False
        self._decay = self.DECAY_TAU_MIN

    def play(self, note=None, duration=None, accented=False, slide=False, **kwargs):
        """Play a note, like one step of the TB-303's sequencer.

        A new note opens the gate, which restarts both envelopes, and sets
        the note's pitch, accent and filter decay. A slide instead continues
        the note before it, if that note's gate is still open: the gate
        stays open, the envelopes run on, and only the pitch glides to the
        new note [W]. If the gate is already closed, a slide plays as a new
        note.

        A slide keeps the accent and decay of the note it slides from,
        since changing them halfway through the filter envelope would make
        the cutoff jump.

        Args:
            note (float): The note, in semitones, where 60 is middle C.
            duration (float, optional): How long to hold the gate, in whole
                notes.
            accented (bool): Play the note accented.
            slide (bool): Glide to this note from the one before.
            **kwargs: Knobs to set, such as cutoff=0.4.
        """
        #
        # Open the gate for a new note, or keep it open for a slide.
        # Afterwards, self.legato tells whether this really was a slide:
        # only if the gate of the note before was still open.
        #
        super().play(note, duration, legato=slide, **kwargs)

        # A slide continues the note before, so its settings stay.
        if self.legato:
            return

        # A new note: its pitch starts right at the note, with no glide.
        self._glide = self.key
        self._accented = accented

        # An accented note always takes the shortest decay, whatever the
        # Decay knob says [W].
        if accented:
            self._decay = self.DECAY_TAU_MIN
            return

        #
        # Otherwise the Decay knob sets it. Its pot, VR6, is 1M with an A
        # (logarithmic) taper [S]: turning it by the same amount multiplies
        # the decay time by the same factor. So the knob, from 0 to 1, moves
        # the time constant from DECAY_TAU_MIN to DECAY_TAU_MAX on a
        # logarithmic scale.
        #
        self._decay = self.DECAY_TAU_MIN * (
            self.DECAY_TAU_MAX / self.DECAY_TAU_MIN
        ) ** self.decay

    def forward(self):

        """Compute the next block of sound.

        The sound is computed in blocks: each call computes a short chunk of
        audio, `self.frames` samples long, as set by the audio system. The
        gate, the envelopes and the oscillator each compute one block at a
        time, and carry their state over to the next.

        Most of the values here are arrays with one number per sample: the
        gate, the envelopes (meg, veg), the accent signals, the volume, and
        the sound itself. The knobs, the pitch and the cutoff are single
        numbers, the same for the whole block.
        """

        gate = self.gate()

        #
        # The two envelopes, one value per sample of this block, from 1
        # when the gate opens, decaying toward 0. The MEG's TAU follows the
        # Decay knob; the VEG's is fixed. (They return a column, shape
        # (frames, 1); [:, 0] takes it as a plain array of frames.)
        #
        meg = self.meg(gate, tau=self._decay)[:, 0]
        veg = self.veg(gate)[:, 0]

        #
        # The slide: the pitch glides toward the note's pitch, like a
        # capacitor charging, a fraction of the remaining way each moment.
        # We do not compute it sample by sample. We compute only where
        # it will be at the end of this block, and the oscillator ramps 
        # from where the previous block ended to that, across this block.
        #
        self._glide = self.key + (self._glide - self.key) * math.exp(
            -self.frames / FS / self.SLIDE_TAU
        )

        # UNSOURCED: the tuning knob's range.
        key = self._glide + 24 * (self.tuning - 0.5)

        # The VCO: the raw sawtooth or square wave, at the pitch above.
        wave = self.osc0(shape=self.waveform, freq=key2freq(key))

        #
        # On accented notes, the filter envelope goes through a
        # switch, on only for accented notes, and the Accent knob [W].
        #
        if self._accented:
            accent = meg * self.accent
        else:
            accent = np.zeros_like(meg)

        #
        # The accent sweep circuit, which raises the cutoff [W].
        # self._accent_v keeps its capacitor's charge from block to block,
        # so the next block carries on from where this one ended.
        #
        sweep, self._accent_v = accent_sweep_circuit(accent, self.resonance, self._accent_v)

        #
        # The VCF's cutoff: how the knobs and the MEG set it. The numbers are
        # measurements of a real TB-303, as marked in Open303's code [O].
        #
        # The Cut Off knob sets a base frequency, from 314Hz to 2394Hz.
        # The cutoff does not follow the note played [M].
        #
        # On each note, the MEG sweeps the cutoff around that base: it
        # starts above it, and falls to below it as the MEG decays. The
        # Env Mod knob sets how wide the sweep is: about 3/4 of an octave
        # at its minimum, since it never goes to zero [W], and about 4.5
        # to 5 octaves at its maximum. The sweep starts about 70% of its
        # width above the base, and ends about 30% below it.
        #
        # The numbers below are straight lines fitted to the measurements.
        # The sweep's width was measured with the Cut Off knob at each end
        # of its range, so for the settings in between, the code mixes
        # the two in proportion.
        #
        c, e = self.cutoff, self.env_mod

        # The base frequency of the cutoff, evenly spaced in octaves 
        # across the knob.
        freq = 313.815 * (2394.412 / 313.815) ** c

        # The width of the sweep, in octaves. It also grows a little with
        # the Cut Off knob.
        scale = (1 - c) * (3.774 * e + 0.737) + c * (4.195 * e + 0.864)

        #
        # Shift the sweep down, so that it starts above the base cutoff
        # frequency and ends below it: about 70% of it above, and 30%
        # below.
        #
        offset = 0.0483 * c + 0.2944

        #
        # The final shift of the cutoff from its base frequency, in
        # octaves, up or down: the MEG's sweep plus the accent sweep. The
        # filter below applies it, in semitones (key_modulation).
        #
        octaves = scale * (meg - offset) + self.ACCENT_SWEEP_OCTAVES * sweep

        # UNSOURCED: the resonance knob as the ladder's feedback.
        resonance = self.resonance * self.MAX_RESONANCE

        # The VCF: the filter, at the cutoff computed above.
        out = self.filter(
            wave,
            key_modulation=12 * octaves[:, None],
            resonance=resonance,
            freq=freq,
        )

        # UNSOURCED: make up for the volume the ladder loses as
        # resonance goes up.
        out = out * (1 + resonance / 2)

        #
        # The VCA: the volume, from two control signals that add up at
        # its control input [S].
        #
        # First the VEG, shaped by the note's short attack and release
        # (env0), so that the note starts and stops cleanly [M].
        #
        note_volume = self.env0(gate)[:, 0] * veg

        #
        # Then the accent, through its own RC network [W, S], weighted by
        # how much more its input resistor lets through than the VEG's
        # (ACCENT_VOLUME). It does not stop with the gate: after an
        # accented note's gate closes, the MEG still drives the volume [W].
        # self._accent_volume keeps its capacitor's charge from block to
        # block, so the next block carries on from where this one ended.
        #
        accent_volume, self._accent_volume = accent_volume_circuit(
            accent, 
            self._accent_volume
        )
        accent_volume = self.ACCENT_VOLUME * accent_volume

        volume = note_volume + accent_volume

        return out * volume[:, None]


def get_tb303_panel(tb303, **kwargs):
    """A front panel for a TB303: its waveform switch and six knobs, laid
    out like the controls on the real instrument, as widgets in a notebook.

    The controls change the synth as soon as they are moved, so they can
    shape the sound while it plays. For example:

    ::

        tb303 = TB303()

        panel = get_tb303_panel(tb303)
        panel

    Args:
        tb303 (TB303): The synth the panel controls.
        **kwargs: Passed on to Panel, such as refresh_interval.

    Returns:
        Panel: The panel, to display in a notebook cell.
    """
    # Imported here, so the synth itself does not depend on the widgets.
    from ..panel import Panel, Slider, Switch

    return Panel(tb303, [
        Switch('WAVEFORM', 'waveform', [
            ('SAW', 'sawtooth'),
            ('SQUARE', 'square'),
        ]),
        Slider('TUNING', 'tuning'),
        Slider('CUT OFF FREQ', 'cutoff'),
        Slider('RESONANCE', 'resonance'),
        Slider('ENV MOD', 'env_mod'),
        Slider('DECAY', 'decay'),
        Slider('ACCENT', 'accent'),
    ], **kwargs)
