"""
    jupylet/audio/sequencer.py

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


import logging
import math

from ..audio import FS, sleep, get_bpm, get_note_value
from ..app import get_app

from .note import note2key


logger = logging.getLogger(__name__)


#
# The shapes of a sweep: each maps how far along the sweep is in time, from 0
# to 1, to how far along the knob is, from 0 to 1.
#
SWEEP_SHAPES = {

    # The same amount in each moment: a straight line.
    'linear': lambda x: x,

    # Starts and ends gently, fastest in the middle: half a cosine.
    'ease': lambda x: (1 - math.cos(math.pi * x)) / 2,

    #
    # Fast at first, then settling, like a capacitor charging (an RC curve),
    # scaled to arrive exactly at the end.
    #
    'chase': lambda x: (1 - math.exp(-5 * x)) / (1 - math.exp(-5)),

    #
    # One period of a wave, from the knob's value to the sweep's value and
    # back, so it ends where it started; with repeat, a low frequency
    # oscillator, timed by the pattern.
    #

    # Up and down smoothly: a raised cosine.
    'sine': lambda x: (1 - math.cos(2 * math.pi * x)) / 2,

    # Straight up, then straight down.
    'triangle': lambda x: 1 - abs(2 * x - 1),

    #
    # Up over most of the period, then down quickly: a sawtooth, with a
    # drop sharp enough to sound like one, but not a jump.
    #
    'saw': lambda x: x / 0.9 if x < 0.9 else (1 - x) / 0.1,
}


class StepSequencer:
    """Mixin that gives a synth a built-in step sequencer, like the TB-303's.

    A step sequencer plays a pattern: a loop of equal steps, each a note with
    optional switches, or a rest. The pattern is written as text, with steps 
    separated by whitespace, so it can be laid out as a grid, for example one 
    beat of four steps per row:

    ::

        class TB303r(GatedSound, StepSequencer):

            flags = {'a': 'accented', 's': 'slide'}

        tb303.sonic_live_loop(\"\"\"

            C2/a  C2    C3/s  C2
            C2    Eb2/a C2    C3/s
            C2/a  C2    Bb2/s C2
            G2/a  C2    C3/s  Eb2

        \"\"\")

        tb303.stop()

    Each step is a note name, such as C2, Eb2 or C#2, optionally followed by
    a slash and flags, such as /a or /as. A . step is a rest, and a - step 
    is a tie: the note before it keeps sounding through the step.

    The flags table of the synth maps each flag to a keyword argument of its 
    play() method, which is set to True for steps with that flag. The s flag 
    has a fixed meaning: the step connects to the previous note, which is 
    held for its whole step so that it is still sounding when this step 
    starts. By default it maps to play(legato=True).

    The synth needs only a play(note, duration, **kwargs) method, like 
    GatedSound.play().
    """

    flags = {'s': 'legato'}

    _step_duration = 1 / 16

    # The number of the next cycle to start; cycles are numbered from 0.
    _next_cycle = 0

    # The (cycle, step) of the next step to play, or None when stopped.
    _next_onset = None

    # True once finish() was called, until the last cycle ends.
    _finishing = False

    def sonic_live_loop(self, pattern, step_duration=1/16):
        """Play the pattern in a loop, as a live loop of the app.

        Calling it again with a new pattern, while the loop plays, switches to 
        the new pattern once the current cycle through the pattern completes, 
        so the music stays on the beat.

        Args:
            pattern (str): The pattern, as text.
            step_duration (float): Duration of one step, in whole notes.
        """
        self._pattern = self.parse_pattern(pattern)
        self._step_duration = step_duration

        async def loop():
            await self._play_pattern()

        # Live loops are known by name; one per synth.
        loop.__name__ = 'step_sequencer_%x' % id(self)
        get_app().sonic_live_loop2(loop)

    def finish(self):
        """Stop playing the pattern, and its sweeps, once the current cycle
        through it completes."""

        self._finishing = True

        get_app().finish('step_sequencer_%x' % id(self))

    def stop(self):
        """Stop playing the pattern, and its sweeps."""

        get_app().stop('step_sequencer_%x' % id(self))

        self._finishing = False
        self._next_onset = None
        self._init_sweeps(clear=True)

    def _init_sweeps(self, clear=False):
        """Initialize or clear the main sweep data structures.

        Replacing the dictionaries, rather than clearing them, lets the sound
        thread finish a block with the old ones, and see the new ones from
        its next block.
        """
        if getattr(self, '_active_sweeps', None) is None or clear:

            # knob -> the sweep in progress.
            self._active_sweeps = {}

            #
            # (cycle, step) -> {knob: sweep}, the sweeps waiting to start at
            # that step of that cycle.
            #
            self._sweeps_by_onset = {}

            #
            # knob -> the (cycle, step) where it has sweeps waiting, to
            # cancel them fast.
            #
            self._onsets_by_knob = {}

            # knob -> the (cycle, step) where its last sweep scheduled ends.
            self._ends_by_knob = {}

    def sweep(
        self,
        knob,
        value,
        start_cycle=0,
        start_step=0,
        duration_cycles=None,
        duration_steps=None,
        shape='linear',
        repeat=1,
    ):
        """Sweep a knob to the given value, in time with the pattern.

        The sweep is timed by the pattern's own cycles and steps, so it lands
        on the pattern's beats, and a sequence of sweeps can be laid out in
        one cell, like a score. Its start counts from the pattern's next
        cycle, the moment a new pattern would take over. A cycle is one time
        through the pattern: one bar for a pattern of 16 steps of 16th notes.

        The start and the duration are each a number of cycles plus a number
        of steps, so either can be used, or both: with 16 steps to a cycle,
        start_cycle=1, start_step=8 and start_step=24 are the same moment.
        With no duration given, the sweep lasts one cycle; with a duration of
        0, the knob jumps to the value on the start step.

        The knob then moves smoothly, its value computed for each block of
        sound, from wherever it is when the sweep starts. So sweeps of the
        same knob chain naturally, and a new sweep of a knob takes over from
        the one in progress when it starts.

        A hand on the knob wins: if the knob is changed while a sweep moves
        it, by the panel or by code, that sweep stops, and the knob's sweeps
        still waiting are cancelled. Sweeps run only while the pattern plays,
        and stop() cancels them all.

        For example, to open the filter over two cycles, starting one cycle
        after the next, and then raise the resonance over a quarter cycle:

        ::

            tb303.sweep('cutoff', 0.8, start_cycle=1, duration_cycles=2)
            tb303.sweep('resonance', 0.98, start_cycle=3, duration_steps=4)

        And to wobble the cutoff up to 0.7 and back, four times in a cycle:

        ::

            tb303.sweep('cutoff', 0.7, duration_steps=4, shape='sine', repeat=4)

        Args:
            knob (str): The name of the knob, an attribute, such as 'cutoff'.
            value (float): Where to turn the knob to.
            start_cycle (int): When the sweep starts: cycles after the start
                of the next cycle,
            start_step (int): plus steps.
            duration_cycles (int): How long the sweep takes: cycles,
            duration_steps (int): plus steps.
            shape (str or callable): 'linear', 'ease' or 'chase', or a wave:
                'sine', 'triangle' or 'saw' (see SWEEP_SHAPES), or a function
                that maps 0 to 1 onto 0 to 1.
            repeat (int): How many times to sweep, one after another, each
                for the duration. Meant for the waves, which end where they
                started.
        """
        # The onset: when the sweep starts, as the cycle of the pattern it
        # starts in, and how many steps into that cycle.
        onset = (self._next_cycle + start_cycle, start_step)

        self._schedule_sweep(
            onset, 
            knob, 
            value, 
            duration_cycles, 
            duration_steps, 
            shape, 
            repeat
        )

    def sweep_now(
        self, 
        knob, 
        value, 
        duration_cycles=None, 
        duration_steps=None, 
        shape='linear', 
        repeat=1
    ):
        """Sweep a knob to the given value, starting on the pattern's next
        step, or at the start of its next cycle if it is not playing.

        It is sweep() for live playing: the same, except for when it starts.
        For example, to close the filter over half a cycle, from now on:

        ::

            tb303.sweep_now('cutoff', 0.1, duration_steps=8)

        Args:
            knob (str): The name of the knob, an attribute, such as 'cutoff'.
            value (float): Where to turn the knob to.
            duration_cycles (int): How long the sweep takes: cycles,
            duration_steps (int): plus steps.
            shape (str or callable): 'linear', 'ease' or 'chase', or a wave:
                'sine', 'triangle' or 'saw' (see SWEEP_SHAPES), or a function
                that maps 0 to 1 onto 0 to 1.
            repeat (int): How many times to sweep, one after another, each
                for the duration. Meant for the waves, which end where they
                started.
        """
        onset = self._next_onset or (self._next_cycle, 0)

        self._schedule_sweep(
            onset, 
            knob, 
            value, 
            duration_cycles, 
            duration_steps, 
            shape, 
            repeat
        )

    def sweep_next(
        self, 
        knob, 
        value, 
        duration_cycles=None, 
        duration_steps=None, 
        shape='linear', 
        repeat=1
    ):
        """Sweep a knob to the given value, starting when its last sweep
        scheduled ends.

        It chains sweeps of the same knob without working out their starts.
        If that sweep has already ended, it starts on the pattern's next step,
        as sweep_now() does, and with no sweep of the knob scheduled yet, at
        the start of the next cycle, as sweep() does. For example, to open the
        filter over two cycles and then snap it shut over four steps:

        ::

            tb303.sweep('cutoff', 0.8, duration_cycles=2)
            tb303.sweep_next('cutoff', 0.1, duration_steps=4)

        Args:
            knob (str): The name of the knob, an attribute, such as 'cutoff'.
            value (float): Where to turn the knob to.
            duration_cycles (int): How long the sweep takes: cycles,
            duration_steps (int): plus steps.
            shape (str or callable): 'linear', 'ease' or 'chase', or a wave:
                'sine', 'triangle' or 'saw' (see SWEEP_SHAPES), or a function
                that maps 0 to 1 onto 0 to 1.
            repeat (int): How many times to sweep, one after another, each
                for the duration. Meant for the waves, which end where they
                started.
        """
        self._init_sweeps()

        onset = self._ends_by_knob.get(knob)

        if onset is None:
            onset = (self._next_cycle, 0)

        elif self._is_past(onset):
            onset = self._next_onset or (self._next_cycle, 0)

        self._schedule_sweep(
            onset, 
            knob, 
            value, 
            duration_cycles, 
            duration_steps, 
            shape, 
            repeat
        )

    def _is_past(self, onset):
        """Whether an onset, a (cycle, step), comes before the next step.

        Its step may run past the end of its cycle, so it is counted in steps
        from the next step, taking every cycle to be as long as the pattern
        now playing.
        """
        if self._next_onset is None:
            return onset[0] < self._next_cycle

        cycle, step = self._next_onset

        return (onset[0] - cycle) * len(self._pattern) + onset[1] - step < 0

    def _schedule_sweep(self, onset, knob, value, duration_cycles, duration_steps, shape, repeat=1):
        """File a sweep under its onset, a (cycle, step), to start there, and
        its repeats, each where the one before it ends."""

        #
        # Check the knob now, so a mistyped name fails here, in the cell
        # that has it, and not later inside the pattern's loop, stopping it.
        #
        getattr(self, knob)

        if not callable(shape):
            shape = SWEEP_SHAPES[shape]

        if duration_cycles is None and duration_steps is None:
            duration_cycles = 1

        duration_cycles = duration_cycles or 0
        duration_steps = duration_steps or 0

        self._init_sweeps()

        #
        # The repeats share one base: the knob's value when the first one
        # starts. Read again at each repeat, it would creep, whenever a
        # repeat starts a moment before the one before it has ended.
        #
        base = {}

        for i in range(repeat):

            self._sweeps_by_onset.setdefault(onset, {})[knob] = dict(
                value=value,
                duration_cycles=duration_cycles,
                duration_steps=duration_steps,
                shape=shape,
                base=base,
            )

            self._onsets_by_knob.setdefault(knob, set()).add(onset)

            # Where it ends: its onset, plus its duration.
            onset = (onset[0] + duration_cycles, onset[1] + duration_steps)

        # Where the last one ends, for sweep_next().
        self._ends_by_knob[knob] = onset

    def _orchestrate_sweeps(self, cycle, step, cycle_length, step_duration):
        """On each step: clear out the sweeps that ended, cancel the waiting
        sweeps of knobs changed by hand, and start those due at this step.

        Args:
            cycle (int): The number of the cycle playing.
            step (int): The step of that cycle playing.
            cycle_length (int): The number of steps in that cycle.
            step_duration (float): The duration of one step, in whole notes.
        """
        for knob, s in list(self._active_sweeps.items()):

            if s['done']:
                self._active_sweeps.pop(knob)

            if s['touched']:
                self._cancel_sweeps(knob)

        if step == 0:
            self._carry_over(cycle, cycle_length)

        onset = (cycle, step)

        for knob, s in self._sweeps_by_onset.pop(onset, {}).items():

            onsets = self._onsets_by_knob[knob]
            onsets.discard(onset)

            if not onsets:
                del self._onsets_by_knob[knob]

            steps = s['duration_cycles'] * cycle_length + s['duration_steps']
            seconds = steps * step_duration * get_note_value() * 60 / get_bpm()

            self._active_sweeps[knob] = dict(
                s,
                v0=s['base'].setdefault('v0', getattr(self, knob)),
                last=None,
                length=round(seconds * FS),
                frames=0,
                done=False,
                touched=False,
            )

    def _carry_over(self, cycle, cycle_length):
        """Move the sweeps due past the end of this cycle into the next one.

        A sweep's start can have more steps than a cycle has, such as
        start_step=40 with 16 steps to a cycle. It is filed under the cycle
        it counts from, and moved on one cycle at a time, as each cycle
        starts and its length is known: (c, 40) becomes (c + 1, 24), then
        (c + 2, 8), and it starts on step 8 of cycle c + 2.

        Moving it one cycle at a time, rather than all the way at once,
        counts each cycle by its own length: if a new pattern of another
        length takes over in between, the sweep still starts on exactly its
        40th step. This is also why the start is not worked out when the
        sweep is scheduled: the cycle's length may not be known yet, before
        the pattern is set.
        """

        # A list, since the loop changes the dictionary.
        for onset in list(self._sweeps_by_onset):

            onset_cycle, onset_step = onset

            # Only this cycle's onsets past its end.
            if onset_cycle != cycle or onset_step < cycle_length:
                continue

            later = (cycle + 1, onset_step - cycle_length)

            for knob, s in self._sweeps_by_onset.pop(onset).items():

                self._sweeps_by_onset.setdefault(later, {})[knob] = s

                onsets = self._onsets_by_knob[knob]
                onsets.discard(onset)
                onsets.add(later)

    def _cancel_sweeps(self, knob):
        """Cancel the sweeps of a knob still waiting to start."""

        for onset in self._onsets_by_knob.pop(knob, ()):
            self._sweeps_by_onset.get(onset, {}).pop(knob, None)

        self._ends_by_knob.pop(knob, None)

    def pre_forward(self):
        """Move the knobs being swept to their values for this block.

        Sound calls it before computing each block, on the sound device's
        thread. Each sweep counts the samples it has run, so it moves forward
        by the block's length, and sets its knob to its value at the end of
        the block. The filters ramp from the previous block's value to it,
        sample by sample, so the sweep is continuous.

        It only reads the sweeps, and writes into each one's own record, so
        it shares them with the pattern's steps without locks: the steps
        alone add and remove sweeps (see _orchestrate_sweeps()).
        """
        sweeps = getattr(self, '_active_sweeps', None)
        if not sweeps:
            return

        for knob, s in list(sweeps.items()):

            if s['done']:
                continue

            #
            # A knob holding anything but this sweep's own last value was
            # changed by someone else. The sweep stops at once, and the
            # pattern's next step cancels the rest (see _orchestrate_sweeps()).
            # A new sweep has no last value yet, so it skips this.
            #
            if s['last'] is not None and getattr(self, knob) != s['last']:
                s['touched'] = s['done'] = True
                continue

            s['frames'] += self.frames
            x = min(1., s['frames'] / s['length']) if s['length'] else 1.

            s['last'] = s['v0'] + (s['value'] - s['v0']) * s['shape'](x)
            s['done'] = x >= 1

            setattr(self, knob, s['last'])

    def parse_pattern(self, text):
        """Parse a pattern written as text into a list of steps.

        Returns:
            list: One item per step: None for a rest, '-' for a tie, or a 
                (note, flags) tuple, where flags is the string of the step's 
                flags.
        """
        steps = []

        for token in text.split():

            if token in ('.', '-'):
                steps.append(None if token == '.' else '-')
                continue

            note, _, flags = token.partition('/')

            unknown = set(flags) - set(self.flags)
            if unknown:
                raise ValueError('Unknown flags %r in pattern step %r; known flags are %r.' % (
                    ''.join(sorted(unknown)), token, ''.join(self.flags)
                ))

            steps.append((note2key(note), flags))

        return steps

    async def _play_pattern(self):
        """Play one cycle through the pattern."""

        steps = self._pattern
        step_duration = self._step_duration

        self._init_sweeps()

        cycle = self._next_cycle
        self._next_cycle += 1

        for i, s in enumerate(steps):

            # Update the sweeps for this step, in time with its note.
            self._orchestrate_sweeps(cycle, i, len(steps), step_duration)

            # Where sweep_now() starts a sweep: the step after this one.
            self._next_onset = (cycle, i + 1) if i + 1 < len(steps) else (cycle + 1, 0)

            if isinstance(s, tuple):

                note, flags = s

                # The note lasts n steps: its own and the ties after it.
                n = 1
                while n < len(steps) and steps[(i + n) % len(steps)] == '-':
                    n += 1

                next_step = steps[(i + n) % len(steps)]
                held = isinstance(next_step, tuple) and 's' in next_step[1]

                kwargs = {self.flags[f]: True for f in flags}
                self.play(note, step_duration * n if held else step_duration * (n - 1/2), **kwargs)

            await sleep(step_duration)

        # The last cycle after finish(): end its sweeps, as stop() does.
        if self._finishing:
            self._finishing = False
            self._next_onset = None
            self._init_sweeps(clear=True)
