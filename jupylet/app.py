#
#    jupylet/app.py
#    
#    Copyright (c) 2020, Nir Aides - nir@winpdb.org
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright notice, this
#       list of conditions and the following disclaimer.
#    2. Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
#    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

"""Jupylet application functionality.
"""


import ipywidgets
import functools
import ipycanvas
import ipyevents
import asyncio
import hashlib
import inspect
import pickle
import pyglet
import random
import time
import sys
import io
import os
import re

import PIL.Image
import concurrent.futures

import numpy as np

from .env import is_remote, start_xvfb
from .color import color2rgb
from .state import State


__all__ = ['App']


REMOTE_WARNING = 'Game video will be compressed and may include noticeable artifacts and latency since it is streamed from a remote server. This is expected. If you install Jupylet on your computer video quality will be high.'


_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def async_thread(foo, *args, **kwargs):
    future = _thread_pool.submit(foo, *args, **kwargs)
    return asyncio.wrap_future(future)


def get_glscalef():
    
    x0 = (pyglet.gl.GLfloat * 16)()
    pyglet.gl.glGetFloatv(pyglet.gl.GL_MODELVIEW_MATRIX, x0)
    sx, sy, sz = list(x0)[::5][:3]
    
    return sx, sy, sz


def o2h(o, n=12):
    return hashlib.sha256(pickle.dumps(o)).hexdigest()[:n]


class EventLoop(pyglet.app.EventLoop):

    def __init__(self, clock):
        
        super(EventLoop, self).__init__()
        
        self.clock = clock
        self.ndraw = 0

    def start(self):

        self.has_exit = False        
        self._legacy_setup()

        pyglet.app.platform_event_loop.start()
        
        self.dispatch_event('on_enter')
        self.is_running = True

    def run(self):
                
        loop = asyncio.get_event_loop()
        task = loop.create_task(self._run())

        if not loop.is_running():
            loop.run_until_complete(task)

        return task
        
    async def _run(self):
        
        while not self.has_exit:
            
            dt = self.idle()
            t1 = self.clock.time() + (dt if dt is not None else 1.)
             
            pyglet.app.platform_event_loop.step(0)
            
            t2 = self.clock.time()
            st = max(0, t1 - t2)
            
            await asyncio.sleep(st - 0.0005)
            
        self.is_running = False
        self.dispatch_event('on_exit')
        pyglet.app.platform_event_loop.stop()

    def step(self, fake_time):
        
        ndraw = self.ndraw

        while self.is_running and not self.has_exit and ndraw == self.ndraw:
            
            dt = self.idle() or 1.
            #pyglet.app.platform_event_loop.step(0)
            fake_time.sleep(dt)
            
        if self.has_exit:
            self.is_running = False
            self.dispatch_event('on_exit')
            pyglet.app.platform_event_loop.stop()

    def idle(self):
        """Called during each iteration of the event loop.

        The method is called immediately after any window events (i.e., after
        any user input).  The method can return a duration after which
        the idle method will be called again.  The method may be called
        earlier if the user creates more input events.  The method
        can return `None` to only wait for user events.

        For example, return ``1.0`` to have the idle method called every
        second, or immediately after any user events.

        The default implementation dispatches the
        :py:meth:`pyglet.window.Window.on_draw` event for all windows and uses
        :py:func:`pyglet.clock.tick` and :py:func:`pyglet.clock.get_sleep_time`
        on the default clock to determine the return value.

        This method should be overridden by advanced users only.  To have
        code execute at regular intervals, use the
        :py:func:`pyglet.clock.schedule` methods.

        :rtype: float
        :return: The number of seconds before the idle method should
            be called again, or `None` to block for user input.
        """
        dt = self.clock.update_time()
        self.clock.call_scheduled_functions(dt)

        # Update timout
        return self.clock.get_sleep_time(True)


class FakeTime(object):
    
    def __init__(self):
        self._time = 0
        
    def time(self):
        return self._time        

    def sleep(self, dt):
        self._time += dt


class _ClockLeg(object):

    def __init__(self, fake_time=False, **kwargs):
        
        super(_ClockLeg, self).__init__()
        
        self.schedules = {}

        self.fake_time = FakeTime()

        if fake_time:
            self.clock = pyglet.clock.Clock(self.fake_time.time)
        else:
            self.clock = pyglet.clock.Clock()
        
    def run_me_now(self, *args, **kwargs):
        return self.schedule_once(0, *args, **kwargs)
    
    def run_me_once(self, delay, *args, **kwargs):
        return self.schedule_once(delay, *args, **kwargs)
    
    def run_me_again_and_again(self, interval, *args, **kwargs):
        return self.schedule_interval(interval, *args, **kwargs)
    
    def run_me_no_more(self, foo=None):
        return self.unschedule(foo, levels_up=2)
    
    def schedule_once(self, delay, *args, **kwargs):
        """Schedule decorated function to be called once after ``delay`` seconds.
        
        This function uses the default clock. ``delay`` can be a float. The
        arguments passed to ``func`` are ``dt`` (time since last function call),
        followed by any ``*args`` and ``**kwargs`` given here.
        
        :Parameters:
            `delay` : float
                The number of seconds to wait before the timer lapses.
        """
        def schedule0(foo):

            self.unschedule(foo)
            
            @functools.wraps(foo)
            def bar(dt, *args, **kwargs):
                
                if inspect.isgeneratorfunction(foo):
                    
                    goo = self.schedules[foo.__name__].get('gen')
                    if goo is None:
                        goo = foo(dt, *args, **kwargs)
                        self.schedules[foo.__name__]['gen'] = goo
                        delay = next(goo)

                    else:
                        delay = goo.send(dt)
                    
                    if delay is not None:
                        self.clock.schedule_once(bar, delay, *args, **kwargs)
                        
                elif inspect.iscoroutinefunction(foo):
                    task = asyncio.create_task(foo(dt, *args, **kwargs))
                    self.schedules[foo.__name__]['task'] = task
                    
                else:
                    foo(dt, *args, **kwargs)
                
            self.schedules.setdefault(foo.__name__, {})['func'] = bar
            self.clock.schedule_once(bar, delay, *args, **kwargs)

            return foo

        return schedule0

    def schedule_interval(self, interval, *args, **kwargs):
        """Schedule decorated function on the default clock every interval seconds.
        
        The arguments passed to ``func`` are ``dt`` (time since last function
        call), followed by any ``*args`` and ``**kwargs`` given here.
        
        :Parameters:
            `interval` : float
                The number of seconds to wait between each call.
        """
        def schedule0(foo):

            if inspect.iscoroutinefunction(foo):
                raise TypeError('Coroutine functions can only be scheduled with schedule_once() and its aliases.')
                
            if inspect.isgeneratorfunction(foo):
                raise TypeError('Generator functions can only be scheduled with schedule_once() and its aliases.')
                
            self.unschedule(foo)
            self.schedules.setdefault(foo.__name__, {})['func'] = foo
            self.clock.schedule_interval(foo, interval, *args, **kwargs)

            return foo

        return schedule0

    def schedule_interval_soft(self, interval, *args, **kwargs):
        """Schedule a function to be called every ``interval`` seconds.
        
        This method is similar to `schedule_interval`, except that the
        clock will move the interval out of phase with other scheduled
        functions so as to distribute CPU more load evenly over time.
        """
        def schedule0(foo):

            if inspect.iscoroutinefunction(foo):
                raise TypeError('Coroutine functions can only be scheduled with schedule_once() and its aliases.')
                
            if inspect.isgeneratorfunction(foo):
                raise TypeError('Generator functions can only be scheduled with schedule_once() and its aliases.')
                
            self.unschedule(foo)
            self.schedules.setdefault(foo.__name__, {})['func'] = foo
            self.clock.schedule_interval_soft(foo, interval, *args, **kwargs)

            return foo

        return schedule0

    def unschedule(self, foo=None, **kwargs):
        """Remove function from the default clock's schedule.
        
        No error is raised if the ``func`` was never scheduled.
        
        :Parameters:
            `foo` : callable
                The function to remove from the schedule. If no function is given
                unschedule the caller.
        """
        if foo is None:
            fname = inspect.stack()[kwargs.get('levels_up', 1)][3] 
        else:
            fname = foo.__name__
            
        d = self.schedules.pop(fname, {})
        
        if 'func' in d:
            self.clock.unschedule(d.get('func'))
        
        if 'task' in d:
            d['task'].cancel()
        

class _EventLeg(object):
 
    def __init__(self, **kwargs):
        
        super(_EventLeg, self).__init__()
        
        self._dispatcher = None
        self._event = None

        self._event_handlers = {}
        self._event = None
        
    def _watch(self, dispatcher, canvas=None):
        
        self._dispatcher = dispatcher
        
        if canvas:
            self._event = ipyevents.Event(source=canvas)
            self._event.on_dom_event(self._event_handle)
            self._event.watched_events = self._event.supported_key_events + self._event.supported_mouse_events        
        
    def _event_handle(self, event):

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
            'buttons': 'buttons',
            'deltaX': 'scroll_x',
            'deltaY': 'scroll_y',
        }

        e = {k1:event[k0] for k0, k1 in kd.items() if event.get(k0)}

        if 'x' in e:
            e['x'] = e['x'] + 1
            e['y'] = event['boundingRectHeight'] - e['y']
            
        if 'dy' in e:
            e['dy'] = -e['dy']

        if 't' in e:
            e['t'] = round(e['t'] / 1000, 3)
           
        foo = getattr(self, 'on_' + e['event'], None)
        if foo:
            keys = foo.__code__.co_varnames[:foo.__code__.co_argcount]
            kwargs = {k: e.get(k, None) for k in keys if k != 'self'}
            foo(**kwargs)
            
    def on_keydown(self, code, key, repeat, ctrl_key, alt_key, shift_key, meta_key):

        if not repeat:        
            self._on_key('on_key_press', code, key, ctrl_key, alt_key, shift_key, meta_key)
        
        if len(key) == 1:
            self._dispatcher.dispatch_event('on_text', key)

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
            self._dispatcher.dispatch_event('on_text_motion', motions[key])
            
    def on_keyup(self, code, key, ctrl_key, alt_key, shift_key, meta_key):
        self._on_key('on_key_release', code, key, ctrl_key, alt_key, shift_key, meta_key)
        
    def _on_key(self, event_type, code, key, ctrl_key, alt_key, shift_key, meta_key):

        modifiers = 0
        modifiers |= alt_key or 0 and pyglet.window.key.MOD_ALT
        modifiers |= ctrl_key or 0 and pyglet.window.key.MOD_CTRL
        modifiers |= shift_key or 0 and pyglet.window.key.MOD_SHIFT
        modifiers |= meta_key or 0 and pyglet.window.key.MOD_WINDOWS
        
        symbol = self._code2symbol(code)

        if symbol:
            self._dispatcher.dispatch_event(event_type, symbol, modifiers)
    
    def on_mouseenter(self, x, y):
        self._dispatcher.dispatch_event('on_mouse_enter', x, y)
    
    def on_mouseleave(self, x, y):
        self._dispatcher.dispatch_event('on_mouse_leave', x, y)
    
    def on_mousemove(self, x, y, dx, dy):
        self._dispatcher.dispatch_event('on_mouse_motion', x, y, dx, dy)

    def on_wheel(self, x, y, delta_x, delta_y):
        self._dispatcher.dispatch_event('on_mouse_scroll', x, y, delta_x, delta_y)

    def on_mousedown(self, x, y, buttons, ctrl_key, alt_key, shift_key):
        self._on_mouse('on_mouse_press', x, y, buttons, ctrl_key, alt_key, shift_key)

    def on_mouseup(self, x, y, buttons, ctrl_key, alt_key, shift_key):
        self._on_mouse('on_mouse_release', x, y, buttons, ctrl_key, alt_key, shift_key)
        
    def _on_mouse(self, event_type, x, y, buttons, ctrl_key, alt_key, shift_key):
        
        buttons = (buttons & 1) + ((buttons & 2) << 1) + ((buttons & 4) >> 1)
        
        modifiers = 0
        modifiers |= alt_key or 0 and pyglet.window.key.MOD_ALT
        modifiers |= ctrl_key or 0 and pyglet.window.key.MOD_CTRL
        modifiers |= shift_key or 0 and pyglet.window.key.MOD_SHIFT
        
        self._dispatcher.dispatch_event(event_type, x, y, buttons, modifiers)

    @staticmethod
    def _code2symbol(code):
        
        code = re.sub(r'Arrow(Up|Right|Down|Left)', r'\1', code)
        code = re.sub(r'Digit(\d)', r'_\1', code)
        code = re.sub(r'(\w+)(L|R)(?:eft|ight)|(?:Key(\w))', r'\2\1\3', code)
        code = code.upper()

        symbol = getattr(pyglet.window.key, code, None)
        if type(symbol) is int:
            return symbol

    def event(self, *args):
        """Function decorator for an event handler.
        
        Examples::

            @app.event
            def on_resize(self, width, height):
                # ...

        or::

            @app.event('on_resize')
            def foo(self, width, height):
                # ...
        """
        if len(args) == 0:                      # @window.event()
            def decorator(func):
                name = func.__name__
                self._dispatcher.set_handler(name, func)
                return func
            return decorator
        
        elif inspect.isroutine(args[0]):        # @window.event
            func = args[0]
            name = func.__name__
            self._dispatcher.set_handler(name, func)
            return args[0]
        
        elif isinstance(args[0], str):          # @window.event('on_resize')
            name = args[0]
            def decorator(func):
                self._dispatcher.set_handler(name, func)
                return func
            return decorator        


class App(_ClockLeg, _EventLeg):
    """Create a game object.
    
    Args:
        width (int): The width of the game canvas in pixels (default is 512).
        height (int): The height of the game canvas in pixels (default is 512).
        mode (str): Can be either "window" or "jupyter". Set to
            "window" to run the game in a regular operating system window.
            Set to "jupyter" to run the game in a Jupyter notebook
            canvas (the default).
    """

    def __init__(self, width=512, height=512, mode='jupyter', buffer=False, resource_path=['.', 'images/'], quality=None):

        assert mode in ['window', 'jupyter', 'hidden']
        assert not (is_remote() and mode == 'window')

        if is_remote() and mode =='jupyter' and quality is None:
            quality = 20
            sys.stderr.write(REMOTE_WARNING + '\n')
            self._show_remote_warning = True
                
        super(App, self).__init__(fake_time=(mode=='hidden'))
        
        self.event_loop = EventLoop(self.clock)
        
        # temporary hack.
        pyglet.app.event_loop = self.event_loop

        if resource_path:
            pyglet.resource.path = resource_path
            pyglet.resource.reindex()

        self._width = width
        self._height = height

        self.buffer = buffer or mode == 'hidden'
        self.mode = mode
        
        visible = mode in ['window', 'both']
        canvas_ = mode in ['jupyter', 'both']
        
        self.window = pyglet.window.Window(visible=visible)
        self.window.register_event_type('on_buffer')
        self.window.set_size(width, height)	
        self.window.set_vsync(False)  
      
        self.array0 = None
        
        self.canvas = ipycanvas.Canvas(size=(width, height)) if canvas_ else None
        self.canvas_quality = quality
        self.canvas_interval = 1 / 15
        self.canvas_last_update = 0

        self._watch(self.window, self.canvas)

        self._run_timestamp = None

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height
    
    def run(self, interval=1/60):
        
        assert self.window._context, 'Window has closed. Create a new app object to run.'

        hasattr(self, '_show_remote_warning') and sys.stderr.write(REMOTE_WARNING + '\n')

        self._run_timestamp = time.time()

        self.set_redraw_interval(interval)
        
        if not self.event_loop.is_running:
            self.event_loop.start()
            self.event_loop.run()
            
        return self.canvas
        
    def start(self, interval=1/60):
        
        assert self.mode == 'hidden', 'start() is only possible in "hidden" reinforcement learning mode. Use run() instead.'
        
        self.set_redraw_interval(interval)
        if not self.event_loop.is_running:
            self.event_loop.start()
            
        return self.canvas
       
    def step(self, n=1):

        assert self.mode == 'hidden', 'step() is only possible in "hidden" reinforcement learning mode. Use run() instead.'
        
        for i in range(n):
            self.event_loop.step(self.fake_time)
        
        return self.array0

    def stop(self):

        if self._run_timestamp and time.time() - self._run_timestamp < 0.5:
            sys.stderr.write('Ignoring call to stop() since it appears to have been done accidentally.')
            return

        if self.event_loop.is_running:
            self.event_loop.exit()
            self.clock.unschedule(self._redraw_windows)

    def set_redraw_interval(self, interval):
        self.clock.unschedule(self._redraw_windows)
        self.clock.schedule_interval_soft(self._redraw_windows, interval)

    def _redraw_windows(self, dt, force_redraw=False):
        
        #
        # This should be real time, not fake time, since it updates to jupyter canvas.
        #
        t0 = time.time()

        if self.window.invalid or force_redraw:

            self.window.switch_to()
            self.window.dispatch_event('on_draw')
            self.window.flip()

            nc = self.canvas_last_update + self.canvas_interval
            
            if self.canvas is not None and t0 >= nc:
                self.canvas_last_update = t0
                self.array0 = self._get_buffer()
                if self.canvas_quality:
                    self.canvas.draw_image(_a2w(self.array0, 'JPEG', quality=self.canvas_quality)[0], 0, 0)
                else:
                    self.canvas.put_image_data(self.array0) 

            elif self.buffer:
                self.array0 = self._get_buffer()
            
        if self.buffer:
            self.window.dispatch_event('on_buffer', self.array0)
            self.event_loop.ndraw += 1
        
    def _get_buffer(self):
        
        bm = pyglet.image.get_buffer_manager()
        
        x0, y0, w0, h0 = bm.get_viewport() 
        sx, sy, sz = get_glscalef()
        
        sx = self.window.width ** 2 / self.width / w0 / sx
        sy = self.window.height ** 2 / self.height / h0 / sy
        
        if sx != 1. or sy != 1.:
            pyglet.gl.glScalef(sx, sy, 1.)
            #return

        cb = bm.get_color_buffer()
        
        buffer = (pyglet.gl.GLubyte * (len(cb.format) * w0 * h0))()

        pyglet.gl.glReadBuffer(cb.gl_buffer)
        pyglet.gl.glPushClientAttrib(pyglet.gl.GL_CLIENT_PIXEL_STORE_BIT)
        pyglet.gl.glPixelStorei(pyglet.gl.GL_PACK_ALIGNMENT, 1)

        pyglet.gl.glReadPixels(0, 0, self.window.width, self.window.height, cb.gl_format, pyglet.gl.GL_UNSIGNED_BYTE, buffer)
        pyglet.gl.glPopClientAttrib()
        
        return np.frombuffer(buffer, dtype='uint8').reshape(self.window.height, self.window.width, 4)[::-1,:,:3]

    def scale_window_to(self, px):
        """Scale window size so that its bigges dimension (either width or height)
        is px pixels.

        This is useful for RL applications since smaller windows render faster.
        """
        assert self.mode not in ['jupyter', 'both'], 'Cannot rescale window in Jupyter mode.'
        assert self.event_loop.is_running, 'Window can only be scaled once app has been started.'

        width0 = self.window.width
        height0 = self.window.height
        
        scale = px / max(width0, height0)

        self.window.width = round(scale * width0)
        self.window.height = round(scale * height0)

        sx = self.window.width / width0
        sy = self.window.height / height0

        pyglet.gl.glScalef(sx, sy, scale)

    def play_once(self, sound):
        
        if self.mode == 'hidden':
            return
            
        player = pyglet.media.Player()
        player.queue(sound)
        player.play()
        
        self.clock.schedule_once(lambda dt: player.pause(), sound.duration)

    def set_window_color(self, color):
        pyglet.gl.glClearColor(*color2rgb(color, zero2one=True))


    def save_state(self, name, path, *args):
        
        if not path:
            path = '%s-%s.state' % (name, o2h(random.random()))

        with open(path, 'wb') as f:
            sl = [o.get_state() for o in args]
            pickle.dump(sl, f)
            
        return path
            
    def load_state(self, path, *args):
        
        with open(path, 'rb') as f:
            sl = pickle.load(f)
            for o, s in zip(args, sl):
                o.set_state(s)

        self._redraw_windows(0, force_redraw=True)
        self._redraw_windows(0, force_redraw=True)

        return self.array0


def _a2w(a, format='JPEG', **kwargs):
    """Convert a numpy array of a JPEG image to an ipywidget image."""

    b0 = io.BytesIO()
    i0 = PIL.Image.fromarray(a)
    i0.save(b0, format, **kwargs)
    w0 = ipywidgets.Image(value=b0.getvalue(), format=format)
    return w0, b0.getvalue()

