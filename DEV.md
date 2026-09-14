# Developer notes

Notes for developers/maintainers of jupylet - not user-facing, see `README.md` and
`docs/` for that.

## Testing jupylet on a remote Ubuntu EC2 instance as if it were a desktop

This is for developers/maintainers who want to exercise jupylet's `mode='window'`
and audio code paths on a remote Ubuntu EC2 instance, interacting with it the
same way a user would on a real desktop (a visible window, mouse/keyboard,
and audio played back locally) - as opposed to the headless `mode='hidden'`
RL setup, which never needs any of this.

It works by installing a desktop environment (XFCE) and an RDP server (xrdp)
on the instance, and using xrdp's built-in audio redirection to send jupylet's
audio to the local machine's speakers over the RDP session.

VNC was tried first and abandoned: the VNC/RFB protocol has no audio channel
at all, so audio would never reach the local machine regardless of any package
installed remotely. RDP has native audio redirection support, which is why
xrdp is used instead.

This section covers only the xrdp/audio infrastructure itself. It does not
cover what jupylet needs installed on top - that's the same, platform-generic
list documented in the README's Ubuntu section (currently: `build-essential`,
`libx11-dev`, `libportaudio2`, `libgl1-mesa-dev`, `libegl1-mesa-dev`), which
applies equally to a real desktop and to this simulated one.

### 1. System packages

```bash
sudo apt update
sudo apt install xfce4 xfce4-goodies xrdp pipewire pipewire-module-xrdp
```

### 2. xrdp login setup

```bash
echo "startxfce4" > ~/.xsession
sudo usermod -aG ssl-cert $(whoami)
sudo passwd ubuntu
sudo systemctl restart xrdp-sesman xrdp
```

`sudo passwd ubuntu` sets a password for xrdp's own PAM-based login screen,
which is separate from SSH key auth and has no password set by default.
Always restart `xrdp-sesman` and `xrdp` together - restarting only one of
them can leave an orphaned Xorg process and break subsequent logins with
"Xorg server closed connection" (if that happens, `pkill -u ubuntu Xorg`
before restarting both).

### 3. Start PipeWire's user services and load the xrdp audio bridge

```bash
systemctl --user start pipewire pipewire-pulse wireplumber
bash /usr/libexec/pipewire-module-xrdp/load_pw_modules.sh
```

Run this over the same SSH session - no need to log in via RDP first, since
an SSH login already creates a systemd user session for PipeWire to run
under. This step currently has to be repeated every session: PipeWire's
`--user` services are enabled and socket-activated, but don't reliably
auto-start via the bare `.xsession` → `startxfce4` login path (there's no
display-manager PAM hook to trigger them). Not yet solved - if you find a
clean fix (an autostart entry, or a `.xsession` addition), this section
should be updated.

### 4. Tunnel and connect

From your local machine, open an SSH tunnel - don't open the RDP port in the
EC2 security group:

```bash
ssh -L 3389:localhost:3389 ubuntu@<EC2_PUBLIC_IP>
```

Connect with Microsoft Remote Desktop (or any RDP client) to `localhost:3389`,
user `ubuntu`, with the password set in step 2. In the client's connection
settings, enable audio redirection - in Microsoft Remote Desktop for Mac:
**Devices & Audio → Play sound → this computer**.

### 5. Verify

These are diagnostic tools only, not part of the recipe itself:

```bash
sudo apt install pulseaudio-utils alsa-utils

pactl list sinks short          # should show a real "xrdp-sink", not just "auto_null"
pactl get-default-sink          # should print xrdp-sink; if not: pactl set-default-sink xrdp-sink
paplay /usr/share/sounds/alsa/Front_Center.wav
```

You should hear the sound through the RDP client's speakers. A sink listed as
`SUSPENDED` in `pactl list sinks short` is a normal idle state - it wakes to
`RUNNING` the moment audio actually plays.

Once this passes, install jupylet's own requirements (see the README) inside
the RDP session and run e.g. `python examples/spaceship.py` or
`python examples/shadertoy_demo.py` to confirm both windowed rendering and
audio work end to end.

## Precompiled wheels for macOS and Windows

Some of jupylet's dependencies need a C++ compiler to install from source on at
least some platforms/Python versions: `moderngl` and `glcontext` currently have
no PyPI wheel for Python 3.14 on macOS or Windows, and `python-rtmidi` (needed
for the optional `[midi]` extra) has no PyPI wheel at all, for any Python
version, on either platform. Building from source would mean every such user
first needs Xcode Command Line Tools (macOS) or Visual Studio Build Tools
(Windows) installed - a reasonable ask for an experienced developer, but a
rough first impression for jupylet's other audiences (kids, parents,
musicians).

For `moderngl` and `glcontext`, this turned out to already be solved upstream:
conda-forge publishes prebuilt conda packages for both, including Python 3.14
on macOS and Windows (and Linux). Since Miniforge already defaults to the
conda-forge channel, `conda install moderngl glcontext` (as documented in the
README, run once before `pip install jupylet`) gets a precompiled version with
no compiler needed - no custom hosting required.

`python-rtmidi` has no such option - it isn't on conda-forge at all, and has no
PyPI wheel for any platform or Python version. For that one, precompiled
wheels are hosted in a separate companion repository,
[jupylet-wheels](https://github.com/nir/jupylet-wheels), served as a static
[PEP 503](https://peps.python.org/pep-0503/) package index via GitHub Pages at
`https://nir.github.io/jupylet-wheels/`, and offered as one of two install
paths in `sound.rst`'s MIDI Keyboards section. See that repo's own README for
why it's structured the way it is, and for the exact commands to regenerate or
add to the wheels it hosts (needed whenever jupylet bumps its pinned version of
`python-rtmidi`, or adds support for a new Python version).

This is a living workaround, not a permanent fix - once `python-rtmidi` starts
publishing wheels of its own, the jupylet-wheels repo and the corresponding
install instructions in `sound.rst` should be retired.

## Building and viewing the docs locally

```bash
pip install -e .                        # jupylet itself must be importable (autodoc pulls docstrings)
pip install -r docs/requirements.txt    # Sphinx, sphinx-rtd-theme, Jinja2
cd docs && make html
open _build/html/index.html             # macOS; xdg-open on Linux, or just double-click on Windows
```

No compiler toolchain is needed for the docs build itself - Sphinx/sphinx-rtd-theme/Jinja2 are
all pure-Python with wheels everywhere. The only toolchain dependency is transitive, through
`pip install -e .` needing whatever jupylet itself needs (see the README's `conda install
moderngl glcontext` step, on Python versions without a wheel for those two).

`jupylet/env.py`'s `is_sphinx_build()` guards against Sphinx's autodoc import triggering live
audio device initialization, so the build itself won't crash on a headless machine.

`docs/requirements.txt` uses compatible-release version ranges (`~=`) rather than exact pins,
so patch/security releases (e.g. a future Jinja2 3.1.x fix) are picked up automatically without
needing a manual bump commit, while still blocking an unplanned major-version jump (e.g. Sphinx
5.x to 9.x) from landing unnoticed on a ReadTheDocs rebuild.
