"""
    jupylet.py
    
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


VERSION = '0.5.0.dev'


import functools
import ipycanvas
import ipyevents
import traceback
import webcolors
import asyncio
import inspect
import pyglet
import time
import re

import concurrent.futures

import numpy as np


_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def async_thread(foo, *args, **kwargs):
    future = _thread_pool.submit(foo, *args, **kwargs)
    return asyncio.wrap_future(future)


def color2rgb(color, zero2one=False):
    
    try:
        rgb = webcolors.name_to_rgb(color) + (255,)
    except:
        rgb = webcolors.hex_to_rgb(color) + (255,)
        
    if zero2one:
        return tuple(v / 255 for v in rgb)
    
    return rgb


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
        Usage::
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
    
    def __init__(self, width=512, height=512, mode='jupyter', buffer=False, resource_path=['.', 'images/']):
        
        assert mode in ['window', 'jupyter', 'both', 'hidden']
        
        super(App, self).__init__(fake_time=(mode=='hidden'))
        
        if resource_path:
            pyglet.resource.path = resource_path
            pyglet.resource.reindex()
            
        self.width = width
        self.height = height
        self.buffer = buffer or mode == 'hidden'
        self.mode = mode
        
        visible = mode in ['window', 'both']
        canvas_ = mode in ['jupyter', 'both']
        
        self.window = pyglet.window.Window(visible=visible)
        self.window.register_event_type('on_buffer')
        self.window.set_size(width, height)
        self.window.set_vsync(False)        

        self.event_loop = EventLoop(self.clock)
        
        self.array0 = None
        
        self.canvas = ipycanvas.Canvas(size=(width, height)) if canvas_ else None
        self.canvas_interval = 1 / 15
        self.canvas_last_update = 0
        
        self._watch(self.window, self.canvas)

    def run(self, interval=1/60):
        
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
       
    def step(self):

        assert self.mode == 'hidden', 'step() is only possible in "hidden" reinforcement learning mode. Use run() instead.'
        
        self.event_loop.step(self.fake_time)
        return self.array0

    def stop(self):
        if self.event_loop.is_running:
            self.event_loop.exit()
            self.clock.unschedule(self._redraw_windows)

    def set_redraw_interval(self, interval):
        self.clock.unschedule(self._redraw_windows)
        self.clock.schedule_interval_soft(self._redraw_windows, interval)

    def _redraw_windows(self, dt):
        
        if self.window.invalid:

            self.window.switch_to()
            self.window.dispatch_event('on_draw')
            self.window.flip()

            #
            # This should be real time, not fake time, since it updates to jupyter canvas.
            #
            t0 = time.time()

            nc = self.canvas_last_update + self.canvas_interval
            
            if self.canvas is not None and t0 >= nc:
                self.array0 = self._get_buffer()
                self.canvas.put_image_data(self.array0) 
                self.canvas_last_update = t0

            elif self.buffer:
                self.array0 = self._get_buffer()
            
        if self.buffer:
            self.window.dispatch_event('on_buffer', self.array0)
            self.event_loop.ndraw += 1
        
    def _get_buffer(self):
        bm = pyglet.image.get_buffer_manager()
        cb = bm.get_color_buffer()
        di = cb.get_image_data()
        d0 = di.get_data('RGBA', di.width * 4)
        a0 = np.array(d0, dtype='uint8').reshape(self.height, self.width, 4)[::-1,:,:3]
        return a0

    def play_once(self, sound):
        
        if self.mode == 'hidden':
            return
            
        player = pyglet.media.Player()
        player.queue(sound)
        player.play()
        
        self.clock.schedule_once(lambda dt: player.pause(), sound.duration)

    def set_window_color(self, color):
        pyglet.gl.glClearColor(*color2rgb(color, zero2one=True))


def image_from(o):
    
    if type(o) is str:
        return image_from_resource(o)
    
    if isinstance(o, np.ndarray):
        return image_from_array(o)
    
    if isinstance(o, PIL.Image.Image):
        return image_from_pil(o)
    
    return o

    
def image_from_resource(name):
    
    try:
        return pyglet.resource.image(name)
    
    except pyglet.resource.ResourceNotFoundException:
        pyglet.resource.reindex()
        return pyglet.resource.image(name)

    
def image_from_array(array):
    
    array = array.astype('uint8')

    height, width = array.shape[:2]
    
    if len(array.shape) == 2:
        channels = 1
    else:
        channels = array.shape[-1]
        
    mode = ['L', None, 'RGB', 'RGBA'][channels-1]
        
    return pyglet.image.ImageData(width, height, mode, array.reshape(-1).tobytes(), -width*channels)


def image_from_pil(image):
    
    height, width = image.size 
    channels = {'L': 1, 'RGB': 3, 'RGBA': 4}[image.mode]
        
    return pyglet.image.ImageData(width, height, image.mode, image.tobytes(), -width*channels)


class Sprite(pyglet.sprite.Sprite):
    
    def __init__(self,
                 img, x=0, y=0,
                 blend_src=pyglet.gl.GL_SRC_ALPHA,
                 blend_dest=pyglet.gl.GL_ONE_MINUS_SRC_ALPHA,
                 batch=None,
                 group=None,
                 usage='dynamic',
                 subpixel=True,
                 scale=1.0):

        img = image_from(img)
            
        super(Sprite, self).__init__(img, x, y,
                 blend_src,
                 blend_dest,
                 batch,
                 group,
                 usage,
                 subpixel)

        self.image.anchor_x = self.image.width / 2
        self.image.anchor_y = self.image.height / 2
        
        self.scale = scale
        self.rotation = 0

    @property
    def image(self):
        return pyglet.sprite.Sprite.image.fget(self)
    
    @image.setter
    def image(self, img):
        
        img = image_from(img)
            
        pyglet.sprite.Sprite.image.fset(self, img)
        self.image.anchor_x = self.image.width / 2
        self.image.anchor_y = self.image.height / 2
        self.rotation = self.rotation
        
    def distance_to(self, o, pos=None):

        x, y = pos or (o.x, o.y)
        
        dx = x - self.x
        dy = y - self.y

        return (dx ** 2 + dy ** 2) ** 0.5
    
    def angle_to(self, o, pos=None):
        
        qd = {
            (True, True): 0,
            (True, False): 180,
            (False, False): 180,
            (False, True): 360,
        }
        
        x, y = pos or (o.x, o.y)
        
        dx = x - self.x
        dy = y - self.y

        return math.atan(dy / (dx or 1e-7)) / math.pi * 180 + qd[(dy >= 0, dx >= 0)]

    @property
    def top(self):
        return self.y + self.height - self.image.anchor_y
        
    @property
    def right(self):
        return self.x + self.width - self.image.anchor_x
        
    @property
    def bottom(self):
        return self.y - self.height + self.image.anchor_y
        
    @property
    def left(self):
        return self.x - self.width + self.image.anchor_x
        
    def wrap_position(self, width, height, margin=50):
        self.x = (self.x + margin) % (width + 2 * margin) - margin
        self.y = (self.y + margin) % (height + 2 * margin) - margin

    def clip_position(self, width, height, margin=0):
        self.x = max(-margin, min(margin + width, self.x))
        self.y = max(-margin, min(margin + height, self.y))


def canvas2sprite(c):
    
    a = c.get_image_data()
    b = a.tostring()
    x = ctypes.string_at(id(b)+20, len(b))
    d = pyglet.image.ImageData(a.shape[1], a.shape[0], 'RGBA', x)
    s = Sprite(d)
    
    return s


class Label(pyglet.text.Label):
    
    def __init__(self, text='',
                 font_name=None, font_size=16, bold=False, italic=False,
                 color='white',
                 x=10, y=10, width=None, height=None,
                 anchor_x='left', anchor_y='baseline',
                 align='left',
                 multiline=False, dpi=None, batch=None, group=None):
                
        if type(color) == str:
            color = color2rgb(color)
                    
        super(Label, self).__init__(text,
                 font_name, font_size, bold, italic,
                 color,
                 x, y, width, height,
                 anchor_x, anchor_y,
                 align,
                 multiline, dpi, batch, group)
        
    @property
    def alpha(self):
        return self.color[-1]
        
    @alpha.setter
    def alpha(self, alpha):
        self.color = self.color[:3] + (alpha,)
        
    @property
    def color_name(self):
        return webcolors.rgb_to_name(self.color[:3])

    @property
    def color(self):
        """Text color.
        Color is a 4-tuple of RGBA components, each in range [0, 255].
        :type: (int, int, int, int)
        """
        return self.document.get_style('color')

    @color.setter
    def color(self, color):
        
        if type(color) == str:
            color = webcolors.name_to_rgb(color) + (255,)
            
        self.document.set_style(0, len(self.document.text), {'color': color})


