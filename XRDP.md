# Testing jupylet on a remote Ubuntu EC2 instance as if it were a desktop

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

This document covers only the xrdp/audio infrastructure itself. It does not
cover what jupylet needs installed on top - that's the same, platform-generic
list documented in the README's Ubuntu section (currently: `build-essential`,
`libx11-dev`, `libportaudio2`, `libgl1-mesa-dev`, `libegl1-mesa-dev`), which
applies equally to a real desktop and to this simulated one.

## 1. System packages

```bash
sudo apt update
sudo apt install xfce4 xfce4-goodies xrdp pipewire pipewire-module-xrdp
```

## 2. xrdp login setup

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

## 3. Start PipeWire's user services and load the xrdp audio bridge

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

## 4. Tunnel and connect

From your local machine, open an SSH tunnel - don't open the RDP port in the
EC2 security group:

```bash
ssh -L 3389:localhost:3389 ubuntu@<EC2_PUBLIC_IP>
```

Connect with Microsoft Remote Desktop (or any RDP client) to `localhost:3389`,
user `ubuntu`, with the password set in step 2. In the client's connection
settings, enable audio redirection - in Microsoft Remote Desktop for Mac:
**Devices & Audio → Play sound → this computer**.

## 5. Verify

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
