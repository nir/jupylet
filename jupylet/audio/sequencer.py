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

from ..app import get_app
from ..audio import sleep
from .note import note2key


logger = logging.getLogger(__name__)


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

    def sonic_live_loop(self, pattern, step=1/16):
        """Play the pattern in a loop, as a live loop of the app.

        Calling it again with a new pattern, while the loop plays, switches to 
        the new pattern once the current pass through the pattern completes, 
        so the music stays on the beat.

        Args:
            pattern (str): The pattern, as text.
            step (float): Duration of one step, in whole notes.
        """
        self._pattern = self.parse_pattern(pattern)
        self._pattern_step = step

        async def loop():
            await self._play_pattern()

        # Live loops are known by name; one per synth.
        loop.__name__ = 'step_sequencer_%x' % id(self)
        get_app().sonic_live_loop2(loop)

    def stop(self):
        """Stop playing the pattern."""

        get_app().stop('step_sequencer_%x' % id(self))

    def finish(self):
        """Stop playing the pattern once the current pass through it completes."""

        get_app().finish('step_sequencer_%x' % id(self))

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
        """Play one pass through the pattern."""

        steps = self._pattern
        step = self._pattern_step

        for i, s in enumerate(steps):

            if isinstance(s, tuple):

                note, flags = s

                # The note lasts n steps: its own and the ties after it.
                n = 1
                while n < len(steps) and steps[(i + n) % len(steps)] == '-':
                    n += 1

                next_step = steps[(i + n) % len(steps)]
                held = isinstance(next_step, tuple) and 's' in next_step[1]

                kwargs = {self.flags[f]: True for f in flags}
                self.play(note, step * n if held else step * (n - 1/2), **kwargs)

            await sleep(step)
