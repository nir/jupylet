"""
    jupylet/audio/panel.py

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
Front panels for sound objects: rows of sliders and switches, like the knobs
and buttons on the front of a hardware synthesizer, shown as ipywidgets in a
Jupyter notebook.

Each control is bound to one attribute of a sound object. Moving it sets the
attribute at once, so the sound can be shaped while it plays. The other way
round, a displayed panel refreshes its controls in the background, so that
they follow changes made to the sound by code, such as a live loop turning a
knob.

    from jupylet.audio.panel import Panel, Slider, Switch

    # tb303 is any sound object with these attributes.
    Panel(tb303, [
        Switch('WAVEFORM', 'waveform', [('SAW', 'sawtooth'), ('SQUARE', 'square')]),
        Slider('CUTOFF', 'cutoff'),
        Slider('RESONANCE', 'resonance'),
    ])
"""


import asyncio
import logging

import ipywidgets


logger = logging.getLogger(__name__)


class Control:
    """Base class for a panel control bound to one attribute of a sound.

    Args:
        label (str): Text to show above the control.
        attribute (str): Name of the attribute the control sets.
    """

    def __init__(self, label, attribute):

        self.label = label
        self.attribute = attribute

        self.target = None
        self.control = None

    def widget(self, target):
        """Create the widget, bound to the given sound object."""

        self.target = target
        self.control = self.create_control(getattr(target, self.attribute))

        self.control.observe(
            lambda change: setattr(self.target, self.attribute, change['new']),
            names='value'
        )

        return ipywidgets.VBox(
            [ipywidgets.Label(self.label), self.control],
            layout=ipywidgets.Layout(align_items='center', width=self.width),
        )

    def refresh(self):
        """Update the control to the attribute's current value."""

        if self.control is not None:
            value = getattr(self.target, self.attribute)
            if self.control.value != value:
                self.control.value = value


class Slider(Control):
    """A vertical slider, with a label above it, for one attribute of a sound.

    Args:
        label (str): Text to show above the slider.
        attribute (str): Name of the attribute the slider sets.
        min (float): Value at the bottom of the slider.
        max (float): Value at the top of the slider.
        step (float): Smallest change the slider makes.
    """

    width = '120px'

    def __init__(self, label, attribute, min=0., max=1., step=0.01):

        super().__init__(label, attribute)

        self.min = min
        self.max = max
        self.step = step

    def create_control(self, value):

        return ipywidgets.FloatSlider(
            value=value,
            min=self.min,
            max=self.max,
            step=self.step,
            orientation='vertical',
            readout_format='.2f',
            layout=ipywidgets.Layout(height='180px'),
        )


class Switch(Control):
    """A row of buttons, with a label above it, that picks one value of an attribute.

    Args:
        label (str): Text to show above the buttons.
        attribute (str): Name of the attribute the switch sets.
        options (list): The buttons, as (button label, value) pairs.
    """

    width = '180px'

    def __init__(self, label, attribute, options):

        super().__init__(label, attribute)

        self.options = options

    def create_control(self, value):

        return ipywidgets.ToggleButtons(
            options=self.options,
            value=value,
            style=ipywidgets.ToggleButtonsStyle(button_width='80px'),
        )


class Panel(ipywidgets.HBox):
    """A control panel for a sound object: its controls side by side.

    Each control sets an attribute of the sound object as soon as it is
    moved, so a panel can shape a sound while it plays. The other way round,
    the panel refreshes its controls every refresh_interval seconds, so that
    they follow changes made to the sound by code, such as a live loop
    turning a knob.

    Refreshing runs in the background, and only while the panel is displayed
    somewhere. It pauses when the panel is no longer displayed, for example
    after the cell that showed it is run again, and resumes when the panel
    is displayed again.

    A panel can also hold other panels, each with its own sound object, and
    refreshes them together with its own controls.

    With debug=True, the panel logs when its refreshing starts and stops,
    at the INFO level.

    Args:
        target: The sound object the controls act on.
        controls (list): Slider, Switch and Panel objects, from left to right.
        refresh_interval (float): Seconds between refreshes, or 0 or None for
            no automatic refreshing.
        debug (bool): Log when refreshing starts and stops.
    """

    def __init__(self, target, controls, refresh_interval=0.5, debug=False):

        widgets = [c.set_inner() if isinstance(c, Panel) else c.widget(target) for c in controls]

        # With _view_count set to a number, the frontend keeps it up to date
        # with the number of places the panel is displayed in.
        super().__init__(widgets, _view_count=0)

        self.controls = controls
        self.target = target
        self.inner = False
        self.debug = debug

        self.refresh_interval = refresh_interval
        self.refresh_pending = False

        self._refresh()
        self.observe(lambda change: self._refresh(change), names='_view_count')

    def refresh(self):
        """Update all controls, including those of any inner panels."""
        for c in self.controls:
            c.refresh()
            
    def set_inner(self):
        self.inner = True
        return self

    def _refresh(self, change=None):
        self.debug and logger.info('Enter Panel._refresh(change=%.64r).', change)
        
        if change:
            if self.refresh_pending:
                return
            if change['old'] != 0 or change['new'] != 1:
                return
                
        if not self._view_count or not self.refresh_interval:
            self.refresh_pending = False
            return

        self.debug and logger.info('Refresh panel.')
        self.refresh()
        
        if not self.inner:
            self.refresh_pending = True
            asyncio.get_event_loop().call_later(
                self.refresh_interval, 
                self._refresh
            )
