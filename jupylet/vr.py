"""
    jupylet/vr.py

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

"""
Experimental VR support: shows a jupylet scene on a VR headset through
OpenXR (the standard programming interface for VR headsets), using its
Python binding, pyopenxr. It talks to whatever OpenXR "runtime" is active
- the vendor software that actually drives the headset, such as Virtual
Desktop or SteamVR.

Windows 11 only. The module imports anywhere, but start() and probe() warn
and do nothing on other platforms, without pyopenxr installed, without an
OpenXR runtime, or when the runtime sees no headset.

The module is the singleton: state lives in module globals and the API is
plain module functions, like jupylet.audio.

    import jupylet.vr as vr

    app = App()
    scene = load_blender_gltf('scenes/moon/alien-moon.gltf')

    @app.event
    def render(ct, dt):
        app.window.clear()
        scene.draw()
        label.draw()

    app.run()
    ...
    vr.probe()      # which runtime, is a headset seen
    vr.start(app)   # the headset takes over render(); returns immediately
    ...
    vr.stop()       # render() goes back to the canvas/window

OpenXR add-ons. OpenXR itself doesn't assume any graphics API. Anything
beyond the basics, OpenGL included, is an optional add-on (OpenXR calls
them extensions) that an app must ask for when it connects to the
runtime. XR_KHR_opengl_enable is the OpenGL one - "KHR" marks add-ons
standardized by Khronos, the group behind both OpenXR and OpenGL.

How it works. OpenXR builds each frame in three steps: xr.wait_frame()
pauses until the headset is ready for the next frame and says when it
will be shown; xr.begin_frame() tells the runtime we've started drawing
it; xr.end_frame() hands the runtime the finished images to show.

The drawing itself is one image per eye. Each eye has a few images to
use, not one, because the headset is still showing the previous frame's
image while the next one is being drawn. So the app asks OpenXR for an
image that isn't in use, draws the frame into it, and hands it back so
the headset can show it. (OpenXR calls this set of images a swapchain.)

The waiting is a blocking call - it would freeze the App - so it runs on
a thread of its own. All the drawing stays on the App's thread, with its
one GL context, exactly as without VR. For every frame the VR thread
hands the App's asyncio loop a single job - start the frame, draw both
eyes, give the finished frame to the headset - using
asyncio.run_coroutine_threadsafe(), and waits for it. This is the same
trick jupylet uses for DOM events from the notebook (event.py). Being one
callback on the loop, nothing else - no scheduled handler, no input
handler - can run between the two eyes, so both see the same scene state,
with no locking.

Why not run begin and end on the VR thread too? Because of a rule in
OpenXR's OpenGL add-on (XR_KHR_opengl_enable, above): xr.begin_frame(),
xr.end_frame(), and asking for and handing back the eye images, must run
on the thread that owns the GL context. From any other thread, while the
App holds the context, they just hang. xr.wait_frame() is the only frame
call without this rule, so it alone lives on the VR thread. (pyopenxr's
xr.foo_bar() functions are the OpenXR spec's xrFooBar.)

For each eye, the job:

  1. Takes that eye's image from the swapchain and makes it the current
     drawing target. app.window.clear() now clears it, not the canvas.
  2. Switches every Camera to that eye's point of view
     (model.set_xr_view). The scene's camera stays the player's place in
     the world - move it and the player moves - with head tracking added
     on top. Sprites and labels are drawn on a virtual screen floating in
     front of the head.
  3. Runs the App's own render handler, unchanged.

The first eye's image is also copied into a mirror texture (a picture
kept in GPU memory). While VR runs, the App shows that picture on the
canvas/window instead of running the render handler itself
(App.set_vr_mirror), so the canvas keeps updating at the App's own frame
rate.

start() and stop() must be called from the App's thread: a notebook cell,
or a scheduled handler in a script. Don't call stop() from the render
handler itself.

Not implemented yet:
  - shadows are still computed for the region the scene camera sees, not
    each eye's (Scene.render_shadowmaps) - fine while the two roughly
    agree;
  - controller/input tracking;
  - other platforms: pyopenxr already has the Linux equivalents (GLX, EGL);
    only the WGL shortcut below (WGL is Windows' way of connecting OpenGL
    to a window) and the Windows 11 check are Windows-specific.
"""


import asyncio
import concurrent.futures
import logging
import sys
import threading
import time

try:
    import xr
    import xr.utils.gl as xrgl
    from OpenGL import GL
except:
    xr = xrgl = GL = None

from .resource import get_context
from .model import set_xr_view, disable_xr_view, reset_xr_cameras


logger = logging.getLogger(__name__)


#
# How many real-world meters one scene unit represents. Applied to head/eye
# tracking only, not scene geometry - the effect is equivalent to scaling
# the world, since this is the only place OpenXR's real-meter data ever
# meets the scene's own coordinates. 1.0 assumes the scene follows glTF's
# own convention of using meters; set this if it doesn't - e.g. a scene
# built assuming 1 unit = 1 foot wants world_scale = 0.3048 (a foot, in
# meters). A bigger world_scale makes the world feel bigger: each scene
# unit stands for more real distance, so the same physical movement covers
# proportionally less of it.
#
world_scale = 1.0


_thread = None
_stop = None
_loop = None
_app = None
_render = None  # the App's own render handler
_context = None  # pyopenxr's ContextObject: instance, session, swapchains
_mirror = None  # (texture, framebuffer) the first eye is copied into
_msaa = None  # multisampled framebuffer the eyes are drawn into, if any
_t0 = 0.
_stage = ''  # the last step reached, for status()


def status():
    """Where things are: the VR thread's state and the last step reached."""

    return dict(
        running=is_running(),
        stage=_stage,
        session_state=_context.session_state if _context else None,
        session_is_running=_context.session_is_running if _context else None,
        exit_render_loop=_context.exit_render_loop if _context else None,
    )


def _is_windows_11():
    """Windows 11 still identifies as major version 10; what changed is the
    build number, 22000 and up.
    """
    return sys.platform == 'win32' and sys.getwindowsversion().build >= 22000


def _instance_create_info():
    """Connection settings that ask the OpenXR runtime for the OpenGL
    add-on (see the module docstring).

    pyopenxr's default settings ask for no add-ons, and without this one it
    fails with FunctionUnsupportedError when it goes looking for the OpenGL
    functions. probe() uses the same settings, so it reports on the
    connection start() will actually get.
    """
    return xr.InstanceCreateInfo(
        enabled_extension_names=[xr.KHR_OPENGL_ENABLE_EXTENSION_NAME],
    )


def probe():
    """Report the active OpenXR runtime and whether it currently sees a
    headset.

    OpenXR has no device enumeration: there is one active runtime, selected
    outside the API (on Windows, the registry key
    HKLM\\SOFTWARE\\Khronos\\OpenXR\\1\\ActiveRuntime), and it reports at
    most one head mounted display. So this is two checks - which runtime
    answers, and whether it has a headset - not a list.

    Returns:
        dict: runtime_name, runtime_version, and, when a headset is
        detected, system_name and vendor_id (otherwise None). None, with a
        warning, when VR isn't available here at all.
    """

    if not _is_windows_11():
        logger.warning('jupylet.vr is only supported on Windows 11.')
        return None

    if xr is None:
        logger.warning('jupylet.vr requires pyopenxr - pip install pyopenxr.')
        return None

    try:
        instance = xr.create_instance(_instance_create_info())
    except xr.XrException as e:
        logger.warning('No OpenXR runtime is available (%s). Is one installed and set as the active runtime?', e)
        return None

    try:
        ip = xr.get_instance_properties(instance)

        info = dict(
            runtime_name=ip.runtime_name,
            runtime_version=str(ip.runtime_version),
            system_name=None,
            vendor_id=None,
        )

        try:
            system = xr.get_system(
                instance,
                xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY),
            )
            sp = xr.get_system_properties(instance, system)
            info.update(system_name=sp.system_name, vendor_id=sp.vendor_id)

        except xr.FormFactorUnavailableError:
            logger.warning('OpenXR runtime %s sees no headset. Is it connected (and streaming)?', ip.runtime_name)

        return info

    finally:
        xr.destroy_instance(instance)


def is_running():
    return _thread is not None and _thread.is_alive()


def start(app, samples=0):
    """Take over the App's render handler and run it for the headset. Warns
    and does nothing if VR isn't available here (see probe()), if the App
    has no render handler, or if already running.

    Args:
        app (jupylet.app.App): The running App to take over.
        samples (int, optional): Number of samples per pixel for multisample
            anti-aliasing (MSAA), which smooths jagged edges, for example 4.
            It costs frame time. Independent of the App's own `samples`,
            which only affects its window or canvas. Defaults to 0 (off).
    """

    global _thread, _stop, _loop, _app, _render, _context, _mirror, _msaa, _t0

    if is_running():
        return

    if _context is not None:
        stop()  # the thread died on its own - clean up its session first

    info = probe()
    if info is None or info['system_name'] is None:
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning('vr.start() needs the App\'s event loop running: call it from a notebook cell after app.run(), or from a scheduled handler in a script.')
        return

    render = app._event_handlers.get('render')

    if render is None:
        logger.warning('The App has no render handler to take over - define one with @app.event first.')
        return

    #
    # Session setup happens here, on the App's thread: pyopenxr's WGL
    # binding reads whatever GL context is current at this point, and the
    # App's is - the one and only context, that everything will keep
    # drawing through.
    #
    # session_create_info is passed explicitly because ContextObject's
    # default for it is a mutable default argument that __enter__ writes
    # the graphics-binding pointer into: a second ContextObject would
    # inherit the first run's (destroyed) binding and skip creating its
    # own, so stop() followed by start() would crash.
    #
    # LOCAL space puts the origin at the headset's position when the
    # session starts (eye level), so the eyes begin where the scene's
    # camera is. The default, STAGE, is floor level at the play area's
    # center - the eyes would float a head-height above the camera.
    #
    context = xrgl.ContextObject(
        context_provider=_ContextProvider(),
        instance_create_info=_instance_create_info(),
        session_create_info=xr.SessionCreateInfo(),
        reference_space_create_info=xr.ReferenceSpaceCreateInfo(
            reference_space_type=xr.ReferenceSpaceType.LOCAL,
        ),
    )
    context.__enter__()

    # What ContextObject.frame_loop() does before its loop, since _run()
    # replaces that loop.
    xr.attach_session_action_sets(
        session=context.session,
        attach_info=xr.SessionActionSetsAttachInfo(
            count_action_sets=len(context.action_sets),
            action_sets=(xr.ActionSet * len(context.action_sets))(*context.action_sets),
        ),
    )

    ctx = get_context()
    size = context.swapchains[0].width, context.swapchains[0].height
    texture = ctx.texture(size, 4)
    _mirror = texture, ctx.framebuffer(color_attachments=[texture])
    app.set_vr_mirror(texture)

    #
    # With samples > 1 the eyes are drawn into this multisampled framebuffer
    # (one for both eyes: they are the same size) and averaged into each
    # eye's image - see _resolve_msaa().
    #
    if samples > 1:
        _msaa = ctx.framebuffer(
            color_attachments=ctx.texture(size, 4, samples=samples),
            depth_attachment=ctx.depth_texture(size, samples=samples),
        )

    _loop, _app, _render, _context = loop, app, render, context
    _t0 = app.timer.time

    _stop = threading.Event()
    _thread = threading.Thread(target=_run, daemon=True)
    _thread.start()


def stop():
    """Stop rendering to the headset and give the render handler back to
    the App. Does nothing if not running.
    """

    global _thread, _loop, _app, _render, _context, _mirror, _msaa

    if _context is None:
        return

    _stop.set()

    if _thread is not None and _thread is not threading.current_thread():
        _thread.join(timeout=2)

        if _thread.is_alive():
            # Destroying the session under a thread still inside an OpenXR
            # call crashes the process; better to leak the session.
            logger.warning('The VR thread did not exit (stuck in %s); leaving the OpenXR session up.', _stage)
            return

    _thread = None

    _app.disable_vr_mirror()
    disable_xr_view()
    reset_xr_cameras()

    for o in _mirror[::-1]:
        o.release()

    if _msaa is not None:
        _msaa.color_attachments[0].release()
        _msaa.depth_attachment.release()
        _msaa.release()

    try:
        _context.__exit__(None, None, None)  # session, then instance
    except Exception:
        logger.exception('Error while ending the OpenXR session.')

    _loop = _app = _render = _context = _mirror = _msaa = None


class _ContextProvider(xrgl.GraphicsContextProvider if xrgl else object):
    """pyopenxr's hook for making a GL context current before it draws.
    There is nothing to switch: every GL call in this module runs on the
    App's thread, where the App's context is always current. (When
    pyopenxr is absent the base is a plain object, so the module still
    imports and warns.)
    """

    def make_current(self):
        pass

    def done_current(self):
        pass


#
# pyopenxr workarounds, applied once at import (when pyopenxr is present).
#

def _select_color_swapchain_format(runtime_formats):
    """Prefer an sRGB swapchain.

    jupylet's fragment shader gamma-encodes its output itself (the final
    `pow(color, 1/2.2)` in default-fragment-shader.glsl), as any shader
    drawing to an ordinary 8-bit window does. pyopenxr prefers *linear*
    swapchain formats, which the compositor gamma-encodes again for the
    display - washed-out, too-bright colors. An sRGB format tells it the
    values are already encoded. GL_FRAMEBUFFER_SRGB stays off, so GL writes
    the shader's values as they are.
    """

    for f in (GL.GL_SRGB8_ALPHA8, GL.GL_SRGB8):
        if f in runtime_formats:
            return f

    return _select_color_swapchain_format0(runtime_formats)


def _patch_pyopenxr():

    global _select_color_swapchain_format0

    # create_graphics_binding() probes EGL before WGL and only falls through
    # on AttributeError, but on Python 3.14 the released egl_util.py fails at
    # import (`from _ctypes import pointer` - no longer exported there), which
    # is an ImportError and escapes. Git HEAD has the import fixed; until that
    # ships, go straight to WGL.
    xrgl.create_graphics_binding = xrgl.WGLGraphicsBinding

    _select_color_swapchain_format0 = xrgl.OpenGLGraphics.select_color_swapchain_format
    xrgl.OpenGLGraphics.select_color_swapchain_format = staticmethod(_select_color_swapchain_format)


if xrgl is not None:
    _patch_pyopenxr()


def _run():
    """The VR thread: session events and xr.wait_frame() only. Everything
    else about a frame - xr.begin_frame(), the eyes, xr.end_frame() - is
    handed to the App's thread; see the module docstring for why.
    """

    rendering = (
        xr.SessionState.READY,
        xr.SessionState.SYNCHRONIZED,
        xr.SessionState.VISIBLE,
        xr.SessionState.FOCUSED,
    )

    try:
        while not _stop.is_set():

            _context.poll_xr_events()

            if _context.exit_render_loop:
                logger.warning('The OpenXR runtime ended the session.')
                break

            if not _context.session_is_running or _context.session_state not in rendering:
                time.sleep(0.05)
                continue

            _set_stage('vr: wait_frame')
            frame_state = xr.wait_frame(_context.session)

            _set_stage('vr: dispatched')
            future = asyncio.run_coroutine_threadsafe(_render_frame(frame_state), _loop)
            _wait(future)

    except Exception:
        logger.exception('VR thread exiting with an exception - call vr.stop() to clean up.')


def _wait(future):
    """Wait for the App's thread to draw the frame - in short slices, so a
    stop() called from a cell (which is what keeps the loop from getting to
    the job) is noticed, and the job dropped, instead of deadlocking.
    """

    n = 0

    while True:
        try:
            return future.result(timeout=0.05)

        except concurrent.futures.TimeoutError:
            n += 1
            _set_stage('vr: waiting for the App thread (%.1fs)' % (n * 0.05))

            if _stop.is_set():
                future.cancel()
                return None


def _set_stage(stage):

    global _stage
    _stage = stage


async def _render_frame(frame_state):
    """One frame, both eyes, on the App's thread - see the module docstring."""

    global _t0

    _set_stage('app: begin_frame')

    xr.begin_frame(_context.session)
    _context.render_layers = []

    try:
        _draw_eyes(frame_state)

    finally:
        # Always paired with begin_frame, even if a render handler raised,
        # or the next begin_frame is a call-order error.
        _app.window.use()
        _set_stage('app: end_frame')

        xr.end_frame(
            _context.session,
            frame_end_info=xr.FrameEndInfo(
                display_time=frame_state.predicted_display_time,
                environment_blend_mode=_context.environment_blend_mode,
                layers=_context.render_layers,
            ),
        )


def _draw_eyes(frame_state):

    global _t0

    ct = _app.timer.time
    dt, _t0 = ct - _t0, ct

    head = _locate_head(frame_state)
    ctx = get_context()

    _set_stage('app: view_loop (should_render=%s)' % frame_state.should_render)

    for i, view in enumerate(_context.view_loop(frame_state)):

        _set_stage('app: eye %d bound' % i)

        #
        # view_loop() has just bound this eye's swapchain image (with raw
        # GL, so moderngl doesn't know). Wrap and re-bind it as a moderngl
        # framebuffer so ctx.fbo is right - for app.window.clear(), and
        # for anything that restores it later (Scene.render_shadowmaps) -
        # then clear color and depth. When multisampling, the eye is drawn
        # into _msaa instead and _resolve_msaa() fills the eye afterwards.
        #
        fbo = ctx.detect_framebuffer()
        target = fbo if _msaa is None else _msaa
        target.use()
        target.clear(0., 0., 0., 1.)

        p, o, f = view.pose.position, view.pose.orientation, view.fov

        set_xr_view(
            (p.x, p.y, p.z),
            (o.x, o.y, o.z, o.w),
            (f.angle_left, f.angle_right, f.angle_up, f.angle_down),
            head,
            world_scale,
        )

        _set_stage('app: eye %d render' % i)

        try:
            _render(ct, dt)
        finally:
            disable_xr_view()

        if _msaa is not None:
            _set_stage('app: eye %d resolve' % i)
            _resolve_msaa(fbo)

        if i == 0:
            _set_stage('app: mirror copy')
            ctx.copy_framebuffer(_mirror[1], fbo)

        _set_stage('app: eye %d done' % i)

    _app.window.use()
    _set_stage('app: frame done')


def _resolve_msaa(eye):
    """Average the multisampled framebuffer into the eye's image.

    A multisampled framebuffer stores several colour samples per pixel, and
    the headset needs one colour per pixel: this averages the samples, which
    is what smooths the edges.
    """

    #
    # Straight OpenGL instead of ctx.copy_framebuffer(): that also copies
    # depth, and the depth formats differ (24 bit in _msaa, 32 bit in
    # OpenXR's eye images), which OpenGL rejects, and then copies nothing.
    # The headset needs only the colour.
    #
    w, h = _msaa.size

    GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, _msaa.glo)
    GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, eye.glo)
    GL.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)

    eye.use()


def _locate_head(frame_state):
    """This frame's head pose - the midpoint of the eyes, with the first
    eye's orientation - as ((x, y, z), (x, y, z, w)), or None when there is
    nothing to render. The HUD is drawn relative to the head, not per eye.
    """

    if not frame_state.should_render:
        return None

    _, views = xr.locate_views(
        session=_context.session,
        view_locate_info=xr.ViewLocateInfo(
            view_configuration_type=_context.view_configuration_type,
            display_time=frame_state.predicted_display_time,
            space=_context.space,
        ),
    )

    ps = [v.pose.position for v in views]
    o = views[0].pose.orientation
    n = len(ps)

    return (
        (sum(p.x for p in ps) / n, sum(p.y for p in ps) / n, sum(p.z for p in ps) / n),
        (o.x, o.y, o.z, o.w),
    )
