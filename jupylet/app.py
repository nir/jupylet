"""
    jupylet/app.py
    
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


import ipywidgets
import asyncio
import logging
import pickle
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

try:
    from multiprocessing import shared_memory
except:
    shared_memory = None

from .resource import register_dir, set_shader_2d, set_shader_3d, set_context
from .resource import get_context
from .model import get_xr_view
from .sprite import Sprite
from .env import is_remote, is_osx, has_display, set_window_size, is_python_script, is_rl_worker
from .env import parse_args
from .color import c2v
from .clock import ClockLeg, Timer, setup_fake_time
from .event import EventLeg, JupyterWindow
from .utils import Dict, o2h, abspath, callerpath, patch_method
from .utils import get_logging_widget, setup_basic_logging

from .audio import midi


#__all__ = ['App']


logger = logging.getLogger(__name__)


REMOTE_WARNING = 'Game video will be compressed and may include noticeable artifacts and latency since it is streamed from a remote server. This is expected. If you install Jupylet on your computer video quality will be high.'


def get_config_dict(config_cls):
    return {
        k: getattr(config_cls, k) for k in dir(config_cls) 
        if k[0] != '_' and not callable(getattr(config_cls, k))
    }


def _clear(
    self,
    red=0.0,
    green=0.0,
    blue=0.0,
    alpha=1.0,
    depth=1.0,
    viewport=None,
    color=None,
    foo=None
):
    """Installed over window.clear() by patch_method() (utils.py, called
    from App.__init__) - self is the window, foo its real clear(). Adds a
    color= shorthand, and redirects to the current headset eye's own
    framebuffer while one is being rendered (jupylet.vr).
    """

    if color:
        red, green, blue, alpha = c2v(color, alpha).rgba

    if get_xr_view() is not None:
        return get_context().fbo.clear(red, green, blue, alpha, depth, viewport)

    return foo(red, green, blue, alpha, depth, viewport)


_app = None


def get_app():
    """Return the most recently created app.

    Raises:
        RuntimeError: If no app has been created yet.
    """
    if _app is None:
        raise RuntimeError('No app has been created yet. Create one first, for example with app = App().')

    return _app


class App(EventLeg, ClockLeg):
    
    """A Jupylet game object.

    Args:
        width (int): The width of the game canvas in pixels.
        height (int): The height of the game canvas in pixels.
        mode (str): Game mode can be 'jupyter' for running the game in a 
            Jupyter notebook, 'window' for running the game in a Python 
            script, 'hidden' for machine learning, or 'auto' for let
            Jupylet choose the appropriate mode automatically.
        resource_dir (path): A path to the directory of the game assets.
        quality (int): Quality of game video compression specified as an
            integer between 10 and 100 for when the Jupyter notebook is 
            run on a remote server.
    """

    def __init__(
        self, 
        width=512, 
        height=512, 
        mode='auto', 
        resource_dir='.', 
        quality=None,
        **kwargs
    ):
        """"""

        if mode == 'auto':
            if is_rl_worker():
                mode = 'hidden'
            elif is_python_script():
                mode = 'window'
            else:
                mode = 'jupyter'

        assert mode in ['window', 'jupyter', 'hidden']
        assert has_display() or mode != 'window'

        if is_remote() and mode =='jupyter' and quality is None:
            quality = 20
            sys.stderr.write(REMOTE_WARNING + '\n')
            self._show_remote_warning = True

        self._use_shm = False
        self._shm = None

        self._width = width
        self._height = height

        self.window_size = (width, height)
        
        set_window_size(self.window_size)

        conf = Dict(get_config_dict(self))
        conf.update(kwargs)

        # pyglet may crash on osx.
        if conf.window == 'pyglet' and is_osx():
            conf.window = 'glfw'

        if mode == 'window' and is_python_script():
            for k, v in vars(parse_args()).items():
                if v is not None:
                    conf[k] = v

        for k, v in conf.items():
            self.__dict__[k] = v

        setup_basic_logging(conf.log_level)
        
        #logger.warning('Create app object with width %r, height %r, and the following configuration: %r', width, height, conf)

        if mode in ['jupyter', 'hidden']:
            window_cls = JupyterWindow

        elif mode == 'window':
            window_cls = mglw.get_local_window_cls(conf.window)

        # Hack to scale glfw window to monitor DPI.
        if conf.window == 'glfw':
            import glfw
            glfw.init()
            glfw.window_hint(0x0002200C, True)

        self.window = window_cls(size=(width, height), **conf)
        self.print_context_info(self.window)

        patch_method(self.window, 'clear', _clear)

        mglw.activate_context(window=self.window)

        if mode == 'hidden':
            self.fake_time = setup_fake_time()

        self.timer = Timer()

        ClockLeg.__init__(self, timer=self.timer)

        EventLeg.__init__(
            self,
            ctx=self.window.ctx, 
            wnd=self.window, 
            timer=self.timer
        )
        
        self.mode = mode

        self.buffer = b''
        self.canvas = None

        # Texture to draw over the canvas/window instead of rendering
        # normally - see set_vr_mirror() for more details.
        self._vr_mirror = None
        
        if mode == 'jupyter':

            empty = _ime(PIL.Image.new('RGB', (width, height)))
            self.canvas = ipywidgets.Image(value=empty, format='JPEG')
            self.canvas_quality = quality or 85
            self.canvas_interval = 1 / 24 if self.canvas_quality > 50 else 1 / 12

            self.window._watch_canvas(self.canvas)

        self._run_timestamp = None

        self.is_running = False
        self.interval = 1/48
        self.ndraws = 0

        register_dir(resource_dir, callerpath())
        register_dir(abspath('assets')) 

        set_context(self.ctx)
        self.ctx.enable_only(moderngl.BLEND)

        #max_textures = self.ctx.info['GL_MAX_TEXTURE_IMAGE_UNITS']

        #
        # Only two shader programs in the whole engine, compiled once, here:
        # one for all 3D drawing, one for all 2D. Every Mesh/Sprite/Label
        # draw, for every object in every scene, runs through whichever of
        # these two it is - not a program per object - with per-object
        # state (matrices, textures, color) fed in as uniforms (constant
        # arguments, fixed for one draw call) before each draw call.
        # set_shader_3d()/set_shader_2d() (resource.py) are how
        # the rest of the codebase reaches the one shared instance.
        #
        set_shader_3d(self.load_program(
            vertex_shader='shaders/default-vertex-shader.glsl',
            fragment_shader='shaders/default-fragment-shader.glsl',
        ))
        shader = set_shader_2d(self.load_program('shaders/sprite.glsl'))
        shader.extra['projection'] = glm.ortho(0, width, 0, height, -1, 1)
        shader.extra['canvas_size'] = (width, height)
        shader['projection'].write(shader.extra['projection'])

        self._time2draw = 0
        self._time2draw_rm = 0

        global _app
        _app = self

    def __del__(self):

        # __init__ may raise before self._shm is set (e.g. failed assertions);
        # don't let a partially-constructed object's __del__ mask that error
        # with a second, unrelated AttributeError.
        if getattr(self, '_shm', None) is not None:
            self._shm.close()
            self._shm.unlink()

    def print_context_info(self, window):
        """Prints moderngl context info."""
        logger.info("Context Version:")
        logger.info("ModernGL: %s", moderngl.__version__)
        logger.info("vendor: %s", window._ctx.info["GL_VENDOR"])
        logger.info("renderer: %s", window._ctx.info["GL_RENDERER"])
        logger.info("version: %s", window._ctx.info["GL_VERSION"])
        logger.info("python: %s", sys.version)
        logger.info("platform: %s", sys.platform)
        logger.info("code: %s", window._ctx.version_code)

        # Consume potential glerror from querying info
        err = window._ctx.error
        if err != "GL_NO_ERROR":
            logger.info("glerror consumed after getting context info: %s", err)

    def set_event_handler(self, name, func):

        if name == 'midi_message':
            midi.set_midi_callback(func)
            self.scheduler.unschedule(midi.midi_port_handler)
            self.scheduler.schedule_interval(midi.midi_port_handler, 1)
            return

        EventLeg.set_event_handler(self, name, func)

    def set_midi_sound(self, s):
        """Start the default MIDI handler with given sound object as its intrument.
        
        Args:
            s (jupylet.audio.sound.Sound): The sound object to use as MIDI 
                instrument.
        """
        if midi.test_rtmidi():
            
            midi.set_midi_sound(s)
            midi.set_midi_callback(midi.simple_midi_callback)
            
            self.scheduler.unschedule(midi.midi_port_handler)
            self.scheduler.schedule_interval(midi.midi_port_handler, 1)

    @property
    def width(self):
        """bool: Width of game canvas in pixels."""
        return self._width

    @property
    def height(self):
        """bool: Height of game canvas in pixels."""
        return self._height
    
    def run(self, interval=1/30):
        """Start the game.
        
        If the game is run in a Jupyter notebook, the call will return
        immedately with the canvas object on which the game frames will be 
        redrawn, otherwise the call will block until the game is done.

        Args:
            interval (float): The interval in seconds between frame redraws.

        Returns:
            ipywidgets.widgets.widget_media.Image: The ipywidgets object on 
            which the game will be redrawn.
        """
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

            # Python 3.14 removed the get_event_loop() fallback that created a
            # loop when none was running. Get the running loop (Jupyter) or make
            # one (plain script).
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            task = loop.create_task(self._run())

            if not loop.is_running():
                loop.run_until_complete(task)

        return self.canvas
        
    async def _run(self):
        
        while not self._exit and not self.window.is_closing:            
            dt = self.scheduler.call()
            await asyncio.sleep(max(0, dt - 0.0005))
        
        self.is_running = False
        self.timer.pause()

    def start(self, interval=1/20):
        
        assert self.mode == 'hidden', 'start() is only possible in "hidden" reinforcement learning mode. Use run() instead.'
        
        if self.canvas:
            interval = max(interval, self.canvas_interval)

        self.set_redraw_interval(interval)

        if not self.is_running:

            self.timer.start()

            self.is_running = True
            self._exit = False        
           
        return self.canvas
       
    def step(self, n=1):

        assert self.mode == 'hidden', 'step() is only possible in "hidden" reinforcement learning mode. Use run() instead.'
        
        ndraws = self.ndraws

        while self.is_running and not self._exit and self.ndraws < ndraws + n:
            
            dt = self.scheduler.call()
            self.fake_time.sleep(dt + 1e-4)
            
        if self._exit:
            self.is_running = False

    def stop(self, foo=None):
        """Stop given handler from running. If no handler is given stop the game.
        
        Args:
            foo (function): The handler to stop running.
        """
        if foo is not None:
            return self.unschedule(foo, levels_up=2)

        if self._run_timestamp and time.time() - self._run_timestamp < 0.5:
            sys.stderr.write('Ignoring call to stop() since it appears to have been done accidentally.')
            return

        if self.is_running:
            self.scheduler.unschedule(self._redraw_windows)
            self._exit = True

    def finish(self, foo):
        """Stop given live loop once it completes its current pass through the loop.

        Args:
            foo (function or str): The live loop, or its name.
        """
        sc = self.schedules.get(foo if type(foo) is str else foo.__name__)

        if sc and 'task' in sc:
            sc['times'] = sc['ncall'] + 1

    def set_redraw_interval(self, interval):

        self.interval = interval
        self.scheduler.unschedule(self._redraw_windows)

        if interval > 0:
            self.scheduler.schedule_interval(self._redraw_windows, interval)

    def set_vr_mirror(self, texture):
        """Show a texture instead of running the render handler.

        While set, each redraw draws the texture over the whole canvas or
        window, as it currently is - even if stale - and the render handler
        is not called. jupylet.vr uses it to mirror a headset eye while it
        drives the render handler itself.
        """

        self._vr_mirror = texture

    def disable_vr_mirror(self):
        self._vr_mirror = None

    def _blit_vr_mirror(self):
        """Draw the VR mirror texture into the canvas/window: scaled to fit
        without stretching, centered, black bars on the sides. Called by
        _redraw_windows() instead of running the render handler, while a
        mirror is set (see set_vr_mirror()).
        """

        # A Sprite showing the mirror texture directly, not one it loaded
        # itself - built fresh every call and discarded right after.
        # texture_owner=False: this texture is borrowed, still owned and
        # reused by vr.py, so Sprite.__del__ must not release it along
        # with the geometry it built for itself. collisions=False skips
        # the per-sprite hit-testing setup, meaningless here.
        # flip=False: Sprite defaults to flip=True to compensate for PIL's
        # top-to-bottom row order when loading an image file. This texture
        # was never loaded from a file - vr.py builds it with
        # ctx.copy_framebuffer(), a GPU-to-GPU copy of the eye's own
        # already-correctly-oriented framebuffer - so that compensation
        # doesn't apply, and would flip an image that's already right way up.
        sprite = Sprite(
            texture=self._vr_mirror, 
            texture_owner=False, 
            collisions=False, 
            flip=False
        )

        # Bind and clear the canvas/window itself: this also blacks out
        # whatever the scaled image below doesn't cover, since it won't
        # fill the whole canvas unless the aspect ratios happen to match.
        self.window.clear()

        # Fit, don't stretch: as large as the canvas allows at the
        # texture's own aspect ratio, centered.
        tw, th = self._vr_mirror.size

        sprite.scale = min(self.width / tw, self.height / th)
        sprite.x = self.width / 2
        sprite.y = self.height / 2
        sprite.render()

    def _redraw_windows(self, ct, dt):

        t0 = time.time()

        if self._vr_mirror is not None:
            self._blit_vr_mirror()
        else:
            self.window.render(ct, dt)

        #
        # Empirically, calling ctx.error (glGetError()) here reliably avoids
        # a GL_INVALID_OPERATION that otherwise fires inside swap_buffers()
        # (pyglet's internal resize handler calling glViewport() during
        # dispatch_events(), for heavier scenes in mode='window' - e.g.
        # shadow maps + skybox + several textured meshes). Ruled out: a
        # plain time.sleep() doesn't fix it (not a timing issue), and
        # ctx.finish() - a stronger, real synchronization primitive - also
        # doesn't fix it (not a render-completion issue either). The exact
        # mechanism by which querying the error state specifically avoids
        # this isn't confirmed; this is a working, cheap fix, not a fully
        # understood one.
        #
        if not isinstance(self.window, JupyterWindow):
            self.window.ctx.error

        self.window.swap_buffers()
        
        self._time2draw = time.time() - t0
        self._time2draw_rm = self._time2draw_rm * 0.95 + self._time2draw * 0.05

        if self.mode != 'window':
            self.buffer = self._read_fbo().read(components=4)
        
        if self.mode == 'jupyter':

            im0 = _b2i(self.buffer, self.window.fbo.size, 'RGBA').convert('RGB')

            self.canvas.value = _ime(
                im0, 
                'JPEG', 
                quality=self.canvas_quality
            )
            
        self.ndraws += 1

    def observe(self):
        """Return game canvas content as upside-down-flipped numpy array.
        
        Returns:
            numpy.ndarray: The content of the game canvas an a Numpy array with
            the shape H x W x 4 corresponding to an RGBA bitmap image.
        """
        w, h = self.window.fbo.size
        b = self.get_buffer() 

        if self._use_shm:
            shm = self.get_shared_memory()
            shm.buf[:len(b)] = b
            return ('__ndarray__', (w, h, 4), 'uint8', shm.name)

        return np.frombuffer(b, dtype='uint8').reshape(h, w, -1)

    def _read_fbo(self):
        # Only JupyterWindow has read_fbo: the single-sample copy of a
        # multisampled framebuffer, which can't be read directly.
        return getattr(self.window, 'read_fbo', self.window.fbo)

    def get_buffer(self):

        w, h = self.window.fbo.size
        if w * h * 4 != len(self.buffer):
            self.buffer = self._read_fbo().read(components=4)

        return self.buffer

    def use_shared_memory(self):
        if shared_memory is not None:
            self._use_shm = True

    def get_shared_memory(self):

        w, h = self.window.fbo.size
        size = w * h * 4

        if self._shm is not None and self._shm.size != size:
        
            self._shm.close()
            self._shm.unlink()
            self._shm = None

        if self._shm is None:
            self._shm = shared_memory.SharedMemory(create=True, size=size)

        return self._shm

    # TODO: check if this still works
    def scale_window_to(self, px):
        """Scale window size so that its biggest dimension (either width or height)
        is px pixels.

        This is useful for RL applications since smaller windows render faster.

        Args:
            px (int): The target size of the rescaled canvas in pixels.
        """
        #assert self.mode == 'hidden', 'Can only scale hidden window.'
        assert self.is_running, 'Window can only be scaled once app has been started.'

        w, h = self.window.fbo.size 
        s = px / max(w, h)

        self.window.create_framebuffer(s * w, s * h)

    def save_state(self, name, path=None, *args):
        """Save the state of given game objects to disk.

        Args:
            name (str): If explicit path is not given, `name` will be used
                to automatically generate a filename.
            path (str, optional): An explicit path to use for file.
            *args: A list of object implementing the save_state() method, to 
                save to disk.

        Returns:
            str: path to saved file.
        """
        if not path:
            path = '%s-%s.state' % (name, o2h(random.random()))

        with open(path, 'wb') as f:
            sl = [o.get_state() for o in args]
            pickle.dump(sl, f)
            
        return path
            
    def load_state(self, path, *args):
        """Load the state of given game objects from disk.

        Will also render the scene once all objects are loaded.

        Args:
            path (str, optional): An explicit path to use for file.
            *args: A list of object implementing the load_state() method, to load
                from disk.
        """
        with open(path, 'rb') as f:
            sl = pickle.load(f)
            for o, s in zip(args, sl):
                o.set_state(s)

        if self.is_running:
            self._redraw_windows(0, 0)

    def get_logging_widget(self, height='256px', quiet_default_logger=True, max_lines=320):
        """Returns an ipywidget that shows the last log messages.

        Returns:
            ipywidgets.HTML: an ipywidget showing the last max_lines log 
            messages.
        """
        return get_logging_widget(height, quiet_default_logger, max_lines)
       

def _b2i(buffer, size, format='RGBA'):
    """Convert bytes buffer to PIL image."""
    return PIL.Image.frombytes(format, size, buffer, 'raw', format, 0, -1)


def _ime(im, format='JPEG', **kwargs):
    """Encode a numpy array of an image using given format."""

    b0 = io.BytesIO()
    im.save(b0, format, **kwargs)
    return b0.getvalue()

