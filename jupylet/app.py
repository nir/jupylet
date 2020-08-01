"""
    jupylet/app.py
    
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


import ipywidgets
import asyncio
import inspect
import logging
import pickle
import pyglet
import random
import time
import glm
import sys
import io

import PIL.Image
import concurrent.futures

import numpy as np

import moderngl
import moderngl_window as mglw

from .resource import register_dir, set_shader_2d
from .env import is_remote
from .color import c2v
from .clock import ClockLeg, Timer
from .event import EventLeg, JupyterWindow
from .utils import Dict, o2h, abspath


__all__ = ['App']


logger = logging.getLogger(__name__)


LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class StreamHandler(logging.StreamHandler):
    pass


class LoggingWidget(logging.Handler):
    """ Custom logging handler sending logs to an output widget """

    def __init__(self, height='256px', *args, **kwargs):    
        super(LoggingWidget, self).__init__(*args, **kwargs)
        
        self.out = ipywidgets.Output()
        self.set_layout(height)

    def set_layout(self, height='256px', overflow_y='scroll', **kwargs):
        self.out.layout=ipywidgets.Layout(
            height=height, 
            overflow_y=overflow_y, 
            **kwargs
        )

    def emit(self, record):
        with self.out:
            print(self.format(record))


def get_logging_widget(height='256px', quiet_default_logger=True):

    if type(height) is int:
        height = str(height) + 'px'

    logger = logging.getLogger()

    wl = [h for h in logger.handlers if isinstance(h, LoggingWidget)]
    if wl:
        w = wl[-1]
        w.set_layout(height)
        return w.out

    handler = LoggingWidget(height)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(LOGGING_FORMAT))
    
    logger.addHandler(handler)

    if quiet_default_logger:
        wl = [h for h in logger.handlers if isinstance(h, StreamHandler)]
        if wl:
            wl[-1].setLevel(logging.WARNING)

    return handler.out


def setup_basic_logging(level: int):
    """Set up basic logging

    Args:
        level (int): The log level
    """
    
    if level is None:
        return

    logger = logging.getLogger()
    logger.setLevel(level)

    if not logger.handlers:

        handler = StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(LOGGING_FORMAT))

        logger.addHandler(handler)


REMOTE_WARNING = 'Game video will be compressed and may include noticeable artifacts and latency since it is streamed from a remote server. This is expected. If you install Jupylet on your computer video quality will be high.'


_thread_pool = None


def async_thread(foo, *args, **kwargs):

    global _thread_pool

    if _thread_pool is None:
        _thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
    future = _thread_pool.submit(foo, *args, **kwargs)
    return asyncio.wrap_future(future)


def in_python_script():

    f0 = inspect.currentframe()
    
    while f0:
        if not f0.f_back and f0.f_globals.get('__name__') == '__main__':
            return True
        
        f0 = f0.f_back
        
    return False


def get_config_dict(config_cls):
    return {
        k: getattr(config_cls, k) for k in dir(config_cls) 
        if k[0] != '_' and not callable(getattr(config_cls, k))
    }


class App(EventLeg, ClockLeg):
    
    def __init__(
        self, 
        width=512, 
        height=512, 
        mode='auto', 
        buffer=False, 
        resource_dir='.', 
        quality=None,
        **kwargs
    ):
        
        if mode == 'auto':
            mode = 'window' if in_python_script() else 'jupyter'

        assert mode in ['window', 'jupyter', 'hidden']
        assert not (is_remote() and mode == 'window')

        if is_remote() and mode =='jupyter' and quality is None:
            quality = 20
            sys.stderr.write(REMOTE_WARNING + '\n')
            self._show_remote_warning = True

        self._width = width
        self._height = height

        self.window_size = (width, height)
        
        conf = Dict(get_config_dict(self))
        conf.update(kwargs)
        for k, v in conf.items():
            self.__dict__[k] = v

        setup_basic_logging(conf.log_level)

        if mode in ['jupyter', 'hidden']:
            window_cls = JupyterWindow

        elif mode == 'window':
            window_cls = mglw.get_local_window_cls(conf.window)

        self.window = window_cls(size=(width, height), **conf)
        self.window.print_context_info()

        mglw.activate_context(window=self.window)

        # TODO: support fake time.
        self.timer = Timer()

        ClockLeg.__init__(self, timer=self.timer)

        EventLeg.__init__(
            self,
            ctx=self.window.ctx, 
            wnd=self.window, 
            timer=self.timer
        )

        self.window.config = self
        
        self.buffer = buffer or mode == 'hidden'
        self.mode = mode
                
        self.array0 = np.zeros((height, width, 3), dtype='uint8')
        self.canvas = None

        if mode == 'jupyter':

            empty = _ime(PIL.Image.fromarray(self.array0))
            self.canvas = ipywidgets.Image(value=empty, format='JPEG')
            self.canvas_quality = quality or 85
            self.canvas_interval = 1 / 24 if self.canvas_quality > 50 else 1 / 12

            self.window._watch_canvas(self.canvas)

        self._run_timestamp = None

        self.is_running = False
        self.ndraws = 0

        register_dir(resource_dir)
        register_dir(abspath('assets')) 

        shader = set_shader_2d(self.load_program('shaders/sprite.glsl'))
        shader['projection'].write(glm.ortho(
            0, width, 0, height, -1, 1
        ))

        self.ctx.enable(moderngl.BLEND)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height
    
    def run(self, interval=1/48):
        
        #assert self.window._context, 'Window has closed. Create a new app object to run.'

        hasattr(self, '_show_remote_warning') and sys.stderr.write(REMOTE_WARNING + '\n')

        if self.canvas:
            interval = max(interval, self.canvas_interval)

        self.set_redraw_interval(interval)

        if not self.is_running:

            self._run_timestamp = time.time()
            self.timer.start()

            self.is_running = True
            self._exit = False        

            loop = asyncio.get_event_loop()
            task = loop.create_task(self._run())

            if not loop.is_running():
                loop.run_until_complete(task)

        return self.canvas
        
    async def _run(self):
        
        while not self._exit:            
            dt = self.scheduler.call()
            await asyncio.sleep(max(0, dt - 0.0005))

        self.is_running = False
        self.timer.pause()

    def start(self, interval=1/48):
        
        assert self.mode == 'hidden', 'start() is only possible in "hidden" reinforcement learning mode. Use run() instead.'
        
        if self.canvas:
            interval = max(interval, self.canvas_interval)

        self.set_redraw_interval(interval)

        if not self.is_running:
            self.is_running = True
            self._exit = False        
            
        return self.canvas
       
    def step(self, n=1):

        assert self.mode == 'hidden', 'step() is only possible in "hidden" reinforcement learning mode. Use run() instead.'
        
        ndraws = self.ndraws

        while self.is_running and not self._exit and self.ndraws < ndraws + n:
            
            dt = self.scheduler.call()
            fake_time.sleep(dt)
            
        if self._exit:
            self.is_running = False

        return self.array0

    def stop(self):

        if self._run_timestamp and time.time() - self._run_timestamp < 0.5:
            sys.stderr.write('Ignoring call to stop() since it appears to have been done accidentally.')
            return

        if self.is_running:
            self.scheduler.unschedule(self._redraw_windows)
            self._exit = True

    def set_redraw_interval(self, interval):

        self.scheduler.unschedule(self._redraw_windows)
        self.scheduler.schedule_interval_soft(self._redraw_windows, interval)

    def _redraw_windows(self, ct, dt):
        
        #self.window.switch_to()
        self.window.render(ct, dt)
        self.window.swap_buffers()
        
        buffer = None
        
        if self.canvas is not None:

            buffer = buffer or self.window.ctx.fbo.read(components=3)
            im0 = _b2i(buffer, self.window.ctx.fbo.size)

            self.canvas.value = _ime(
                im0, 
                'JPEG', 
                quality=self.canvas_quality
            )
            
        if self.buffer:

            w, h = self.window.ctx.fbo.size
            buffer = buffer or self.window.ctx.fbo.read(components=3)
            self.array0 = np.frombuffer(buffer, dtype='uint8').reshape(h, w, -1)
        
        self.ndraws += 1

    def _get_buffer(self):
        """Return buffer as upside-down-flipped numpy array."""

        w, h = self.window.ctx.fbo.size
        d = self.window.ctx.fbo.read(components=3)
        a = np.frombuffer(d, dtype='uint8').reshape(h, w, -1)

        return a

    # TODO: check if this still works
    def scale_window_to(self, px):
        """Scale window size so that its bigges dimension (either width or height)
        is px pixels.

        This is useful for RL applications since smaller windows render faster.
        """
        assert self.mode not in ['jupyter'], 'Cannot rescale window in Jupyter mode.'
        assert self.is_running, 'Window can only be scaled once app has been started.'

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
        
        self.scheduler.schedule_once(lambda dt: player.pause(), sound.duration)

    # TODO: change ctx.clear(*color)
    def set_window_color(self, color):
        pyglet.gl.glClearColor(*c2v(color)) 

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

        self._redraw_windows()
        #self._redraw_windows() # TODO: check if still required

        return self.array0

    def get_logging_widget(self, height='256px', quiet_default_logger=True):
        return get_logging_widget(height, quiet_default_logger)

    """
    def set2d(self):
        self.window.projection = pyglet.window.Projection2D()

    def set3d(self):

        pyglet.gl.glEnable(pyglet.gl.GL_DEPTH_TEST)
        pyglet.gl.glEnable(pyglet.gl.GL_CULL_FACE)
        
        self.window.projection = pyglet.window.Projection3D()
    """


def _b2i(buffer, size, format='RGB'):
    """Convert bytes buffer to PIL image."""
    return PIL.Image.frombytes(format, size, buffer, 'raw', format, 0, -1)


def _ime(im, format='JPEG', **kwargs):
    """Encode a numpy array of an image using given format."""

    b0 = io.BytesIO()
    im.save(b0, format, **kwargs)
    return b0.getvalue()

