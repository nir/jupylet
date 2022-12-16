"""
    jupylet/event.py
    
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


import ipyevents
import platform
import moderngl
import inspect
import logging
import sys
import re

import pyglet.window.key

import moderngl
import moderngl_window as mglw

from moderngl_window.context.pyglet.window import Window

from typing import Tuple, Type

from .utils import trimmed_traceback


logger = logging.getLogger(__name__)


#
# JupyterWindow inherits the pyglet Window to take advantage of 
# working logic translating from ipywidgets to pyglet events that was 
# working just fine in previous versions of jupylet.
#

_events = []
_ed = {}


def _stripped_event(e):
    e = dict(e)
    e.pop('timeStamp')
    e.pop('repeat')
    return e


class JupyterWindow(Window):
    
    name = 'Jupyter Window'

    def __init__(self, **kwargs):

        # Call directly parent of parent!
        super(Window, self).__init__(**kwargs)

        self._fbo = None
        self._vsync = False  # We don't care about vsync in headless mode
        self._resizable = False  # headless window is not resizable
        self._cursor = False  # Headless don't have a cursor

        self.init_mgl_context()
        self.set_default_viewport()

        self._event = None

        self._dragstart = False

    def _watch_canvas(self, canvas):
        
        self._event = ipyevents.Event(source=canvas)
        self._event.on_dom_event(self._on_dom_event)
        self._event.watched_events = self._event.supported_key_events + self._event.supported_mouse_events        

    @property
    def fbo(self) -> moderngl.Framebuffer:
        """moderngl.Framebuffer: The default framebuffer"""
        return self._fbo

    def init_mgl_context(self) -> None:
        """Create an standalone context and framebuffer"""

        if platform.system() == 'Linux':
            self._ctx = moderngl.create_standalone_context(
                require=self.gl_version_code,
                backend='egl'
            )
        else:
            self._ctx = moderngl.create_standalone_context(
                require=self.gl_version_code
            )

        self.create_framebuffer(*self.size)

    def create_framebuffer(self, w, h):

        w = int(w)
        h = int(h)
        
        if self._fbo is not None:
            self._fbo.release()

        self._fbo = self.ctx.framebuffer(
            color_attachments=self.ctx.texture((w, h), 4, samples=self._samples),
            depth_attachment=self.ctx.depth_texture((w, h), samples=self._samples),
        )

        self.use()        

    def use(self):
        """Bind the window's framebuffer"""
        self._fbo.use()

    def clear(self, red=0.0, green=0.0, blue=0.0, alpha=0.0, depth=1.0, viewport=None):
        """
        Binds and clears the default framebuffer

        Args:
            red (float): color component
            green (float): color component
            blue (float): color component
            alpha (float): alpha component
            depth (float): depth value
            viewport (tuple): The viewport
        """
        self.use()
        self.ctx.clear(red=red, green=green, blue=blue, alpha=alpha, depth=depth, viewport=viewport)

    def swap_buffers(self) -> None:
        """
        Placeholder. We currently don't do double buffering in headless mode.
        This may change in the future.
        """
        # NOTE: No double buffering currently
        self._frames += 1
        self.ctx.finish()

    def destroy(self) -> None:
        """Destroy the context"""
        self.ctx.release()

    @property
    def is_closing(self) -> bool:
        pass
    
    def close(self) -> None:
        """Signal for the window to close"""
        self._close = True

    @property
    def size(self) -> Tuple[int, int]:
        """Tuple[int, int]: current window size.

        This property also support assignment::

            # Resize the window to 1000 x 1000
            window.size = 1000, 1000
        """
        return self._width, self._height

    @size.setter
    def size(self, value: Tuple[int, int]):
        self._width, self._height = int(value[0]), int(value[1])

    @property
    def position(self) -> Tuple[int, int]:
        """Tuple[int, int]: The current window position.

        This property can also be set to move the window::

            # Move window to 100, 100
            window.position = 100, 100
        """
        return self._position

    @position.setter
    def position(self, value: Tuple[int, int]):
        self._position = int(value[0]), int(value[1])

    @property
    def cursor(self) -> bool:
        """bool: Should the mouse cursor be visible inside the window?

        This property can also be assigned to::

            # Disable cursor
            window.cursor = False
        """
        return self._cursor

    @cursor.setter
    def cursor(self, value: bool):
        self._cursor = value

    @property
    def mouse_exclusivity(self) -> bool:
        """bool: If mouse exclusivity is enabled.

        When you enable mouse-exclusive mode, the mouse cursor is no longer
        available. It is not merely hidden – no amount of mouse movement
        will make it leave your application. This is for example useful
        when you don't want the mouse leaving the screen when rotating
        a 3d scene.

        This property can also be set::

            window.mouse_exclusivity = True
        """
        return self._mouse_exclusivity

    @mouse_exclusivity.setter
    def mouse_exclusivity(self, value: bool):
        self._mouse_exclusivity = value

    @property
    def title(self) -> str:
        """str: Window title.

        This property can also be set::

            window.title = "New Title"
        """
        return self._title

    @title.setter
    def title(self, value: str):
        self._title = value

    def on_resize(self, width: int, height: int):
        """Pyglet specific callback for window resize events forwarding to standard methods

        Args:
            width: New window width
            height: New window height
        """
        pass

    def dispatch_event(self, event, *args, **kwargs):
        foo = getattr(self, event)
        if foo:
            return foo(*args, **kwargs)

    def _on_dom_event(self, event):
        #logger.info('Enter JupyterWindow._on_dom_event(event=%r).', event)

        #
        # Fix repeat bug in jupyter dom handler.
        #
        if event['event'][:3] == 'key':
            if event['event'] == 'keydown' and event['repeat'] == False and event['code'] in _ed:
                if _stripped_event(event) == _stripped_event(_ed[event['code']]):
                    event['repeat'] = True
            _ed[event['code']] = event

        kd = {
            'event': 'event',
            'timeStamp': 't',
            'offsetX': 'x',
            'offsetY': 'y',
            'movementX': 'dx',
            'movementY': 'dy',
            'key': 'key',
            'code': 'code',
            'repeat': 'repeat',
            'ctrlKey': 'ctrl_key',
            'altKey': 'alt_key',
            'shiftKey': 'shift_key',
            'metaKey': 'meta_key',
            'button': 'button',
            'buttons': 'buttons',
            'deltaX': 'scroll_x',
            'deltaY': 'scroll_y',
        }

        e = {k1:event[k0] for k0, k1 in kd.items() if k0 in event}
        e['event_obj'] = event

        if 'x' in e:
            e['x'] = e['x'] + 1
            e['y'] = event['boundingRectHeight'] - e['y']
            
        if 'dy' in e:
            e['dy'] = -e['dy']

        if 't' in e:
            e['t'] = round(e['t'] / 1000, 3)
           
        foo = getattr(self, '_dom_on_' + e['event'], None)
        if not foo:
            return

        keys = foo.__code__.co_varnames[:foo.__code__.co_argcount]
        
        kwargs = {k: e.get(k, None) for k in keys if k != 'self'}

        try:
            foo(**kwargs)
        except:
            logger.error(trimmed_traceback())

    def _dom_on_keydown(self, code, key, repeat, ctrl_key, alt_key, shift_key, meta_key):

        if not repeat:        
            self._on_key('on_key_press', code, key, ctrl_key, alt_key, shift_key, meta_key)
        
        if len(key) == 1:
            self.dispatch_event('on_text', key)

        motions = {
            'ArrowUp': pyglet.window.key.MOTION_UP,
            'ArrowRight': pyglet.window.key.MOTION_RIGHT,
            'ArrowDown': pyglet.window.key.MOTION_DOWN,
            'ArrowLeft': pyglet.window.key.MOTION_LEFT,
            'PageUp': pyglet.window.key.MOTION_PREVIOUS_PAGE,
            'PageDown': pyglet.window.key.MOTION_NEXT_PAGE,
            'Backspace': pyglet.window.key.MOTION_BACKSPACE,
            'Delete': pyglet.window.key.MOTION_DELETE,
        }

        if key in motions:
            pass #self.dispatch_event('on_text_motion', motions[key])
            
    def _dom_on_keyup(self, code, key, ctrl_key, alt_key, shift_key, meta_key):
        self._on_key('on_key_release', code, key, ctrl_key, alt_key, shift_key, meta_key)
        
    def _on_key(self, event_type, code, key, ctrl_key, alt_key, shift_key, meta_key):
        logger.info('Enter JupyterWindow._on_key(event_type=%r, code=%r, key=%r, ctrl_key=%r, alt_key=%r, shift_key=%r, meta_key=%r).', event_type, code, key, ctrl_key, alt_key, shift_key, meta_key)

        modifiers = 0
        modifiers |= (alt_key or 0) and pyglet.window.key.MOD_ALT
        modifiers |= (ctrl_key or 0) and pyglet.window.key.MOD_CTRL
        modifiers |= (shift_key or 0) and pyglet.window.key.MOD_SHIFT
        
        symbol = self._code2symbol(code)

        if symbol:
            self.dispatch_event(event_type, symbol, modifiers)
    
    #def on_mouseenter(self, x, y):
    #    self._dispatcher.dispatch_event('on_mouse_enter', x, y)
    
    #def on_mouseleave(self, x, y):
    #    self._dispatcher.dispatch_event('on_mouse_leave', x, y)
    
    def _dom_on_wheel(self, x, y, delta_x, delta_y):
        self.dispatch_event('on_mouse_scroll', x, y, delta_x, delta_y)

    def _dom_on_mousemove(self, x, y, dx, dy, buttons, ctrl_key, alt_key, shift_key):
        if buttons:
            self._dom_on_drag(x, y, dx, dy, buttons, ctrl_key, alt_key, shift_key)
        else:
            self.dispatch_event('on_mouse_motion', x, y, dx, dy)

    def _dom_on_dragstart(self):
        self._dragstart = True

    def _dom_on_drag(self, x, y, dx, dy, buttons, ctrl_key, alt_key, shift_key):
        logger.debug('Enter JupyterWindow._dom_on_drag(%r)', (x, y, dx, dy, buttons, ctrl_key, alt_key, shift_key))

        # Discard first flawed drag event.
        if self._dragstart:
            self._dragstart = False
            return

        # Discard last flawed drag event.
        if x < -100:
            return

        # Map from https://developer.mozilla.org/en-US/docs/Web/API/MouseEvent/buttons
        # to https://pyglet.readthedocs.io/en/latest/programming_guide/mouse.html

        bi = bin(8 + (buttons & 7))[3:]
        buttons = int(bi[1] + bi[0] + bi[2], 2)
        
        modifiers = 0
        modifiers |= (alt_key or 0) and pyglet.window.key.MOD_ALT
        modifiers |= (ctrl_key or 0) and pyglet.window.key.MOD_CTRL
        modifiers |= (shift_key or 0) and pyglet.window.key.MOD_SHIFT
        
        self.dispatch_event('on_mouse_drag', x, y, dx, dy, buttons, modifiers)

    def _dom_on_dragend(self, x, y, button, ctrl_key, alt_key, shift_key):
        logger.debug('Enter JupyterWindow._dom_on_dragend(%r)', (x, y, button, ctrl_key, alt_key, shift_key))
        self._on_mouse('on_mouse_release', x, y, button, ctrl_key, alt_key, shift_key)
        
    def _dom_on_mousedown(self, x, y, button, ctrl_key, alt_key, shift_key):
        logger.debug('Enter JupyterWindow._dom_on_mousedown(%r)', (x, y, button, ctrl_key, alt_key, shift_key))
        self._on_mouse('on_mouse_press', x, y, button, ctrl_key, alt_key, shift_key)

    def _dom_on_mouseup(self, x, y, button, ctrl_key, alt_key, shift_key):
        logger.debug('Enter JupyterWindow._dom_on_mouseup(%r)', (x, y, button, ctrl_key, alt_key, shift_key))
        self._on_mouse('on_mouse_release', x, y, button, ctrl_key, alt_key, shift_key)
        
    def _on_mouse(self, event_type, x, y, button, ctrl_key, alt_key, shift_key):
        logger.debug('Enter JupyterWindow._on_mouse(%r).', (event_type, x, y, button, ctrl_key, alt_key, shift_key))

        # Map from https://developer.mozilla.org/en-US/docs/Web/API/MouseEvent/button
        # to https://pyglet.readthedocs.io/en/latest/programming_guide/mouse.html

        button = {
            0:1, 1:2, 2:4,
        }[button]
        
        modifiers = 0
        modifiers |= (alt_key or 0) and pyglet.window.key.MOD_ALT
        modifiers |= (ctrl_key or 0) and pyglet.window.key.MOD_CTRL
        modifiers |= (shift_key or 0) and pyglet.window.key.MOD_SHIFT
        
        self.dispatch_event(event_type, x, y, button, modifiers)

    @staticmethod
    def _code2symbol(code):
        
        code = re.sub(r'Arrow(Up|Right|Down|Left)', r'\1', code)
        code = re.sub(r'Digit(\d)', r'_\1', code)
        code = re.sub(r'Key(\w)', r'\1', code)
        code = re.sub(r'(\w+)(L|R)(?:eft|ight)', r'\2\1', code)
        code = re.sub(r'Control', 'CTRL', code)
        code = code.upper()

        symbol = getattr(pyglet.window.key, code, None)

        logger.info('In JupyterWindow._code2symbol() code=%r, symbol=%r.', code, symbol)
        
        if type(symbol) is int:
            return symbol


def nop(self, *args, **kwargs):
    pass


class EventLeg(mglw.WindowConfig):

    log_level = logging.WARNING

    window = 'pyglet'

    vsync = False

    aspect_ratio = None

    def __init__(self, ctx, wnd, timer):
        
        super(EventLeg, self).__init__(ctx, wnd, timer)

        self._event_handlers = {}
        self._exit = False

        self.wnd.config = self

        self.wnd.mouse_position_event_func = self.mouse_position_event_ul
        self.wnd.mouse_drag_event_func = self.mouse_drag_event_ul
        self.wnd.mouse_press_event_func = self.mouse_press_event_ul
        self.wnd.mouse_release_event_func = self.mouse_release_event_ul
        #self.wnd.mouse_scroll_event_func = self.mouse_scroll_event
        
    def mouse_position_event_ul(self, x, y, dx, dy):
        return self.mouse_position_event(x, self.wnd.height - y, dx, -dy)

    def mouse_position_event(self, x, y, dx, dy):
        logger.debug('Enter EventLeg.mouse_position_event(%r, %r, %r, %r).', x, y, dx, dy)
        self.dispatch_event('mouse_position_event', x, y, dx, dy)

    def mouse_drag_event_ul(self, x, y, dx, dy):
        return self.mouse_drag_event(x, self.wnd.height - y, dx, -dy)

    def mouse_drag_event(self, x, y, dx, dy):
        logger.debug('Enter EventLeg.mouse_drag_event(%r, %r, %r, %r).', x, y, dx, dy)
        self.dispatch_event('mouse_drag_event', x, y, dx, dy)

    def mouse_press_event_ul(self, x, y, button):
        return self.mouse_press_event(x, self.wnd.height - y, button)

    def mouse_press_event(self, x, y, button):
        logger.debug('Enter EventLeg.mouse_press_event(%r, %r, %r).', x, y, button)
        self.dispatch_event('mouse_press_event', x, y, button)

    def mouse_release_event_ul(self, x, y, button):
        return self.mouse_release_event(x, self.wnd.height - y, button)

    def mouse_release_event(self, x, y, button):
        logger.debug('Enter EventLeg.mouse_release_event(%r, %r, %r).', x, y, button)
        self.dispatch_event('mouse_release_event', x, y, button)

    def key_event(self, key, action, modifiers):
        """Handle the on_key event. 

        Args:
            key (int): An integer specifying the key.
            action (int): An integer specifying whether the key was pressed or
                released.
            modifiers (dict): A dictionary specifying the state the CTRL, ALT, 
                and SHIFT, keys at the time of the event.
        """
        logger.debug('Enter EventLeg.key_event(%r, %r, %r).', key, action, modifiers)
        self.dispatch_event('key_event', key, action, modifiers)
        
    def render(self, current_time: float, delta: float):
        """Handle the on_render event. 

        Args:
            current_time (float): Time in seconds since start of game.
            delta (float): Time in seconds since previous call to handler.
        """
        #logger.debug('Enter EventLeg.render(%r, %r).', current_time, delta)
        self.dispatch_event('render', current_time, delta)

    def close(self):
        logger.info('Enter EventLeg.close().')
        self._exit = True
        self.dispatch_event('close')
       
    def dispatch_event(self, event, *args, **kwargs):
        foo = self._event_handlers.get(event)
        if foo:
            return foo(*args, **kwargs)

    def event(self, *args):
        """Function decorator for an event handler."""
        
        logger.info('Enter EventLeg.event(*args=%r).', args) 

        if len(args) == 0:                      # @window.event()
            def decorator(func):
                name = func.__name__
                self.set_event_handler(name, func)
                return func
            return decorator
        
        elif inspect.isroutine(args[0]):        # @window.event
            func = args[0]
            name = func.__name__
            self.set_event_handler(name, func)
            return args[0]
        
        elif isinstance(args[0], str):          # @window.event('on_resize')
            name = args[0]
            def decorator(func):
                self.set_event_handler(name, func)
                return func
            return decorator        

    def set_event_handler(self, name, func):
        self._event_handlers[name] = func

