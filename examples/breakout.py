"""
    examples/breakout.py

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
    Breakout!

    A version of the classic 1978 Atari game, built with Jupylet. A ball
    bounces around a walled court breaking colored brick rows for points;
    miss it with the paddle and you lose one of your balls, and the game
    ends when they run out.

    This file is organized top to bottom as:

        1. Layout     - sizes and positions. Almost everything is worked
                         out from one number, `unit`, so the whole game
                         scales together if you change it.
        2. Colors      - every color the game uses, in one place.
        3. Game feel   - paddle speed, ball speed, starting lives... the
                         numbers that decide how the game plays.
        4. Sound       - a tiny square-wave "Blip" instrument, and the
                         musical notes it plays for each kind of hit.
        5. Sprites     - the walls, scoreboard digits, bricks, paddle and
                         ball - the actual things drawn on screen.
        6. Game loop   - functions that run every frame: read the
                         keyboard/mouse, move the paddle, move the ball
                         and check every kind of collision it can have.

    Curious what a number does? Change it and run the game again - almost
    everything here is a named constant you can experiment with.

    Not implemented yet: the ball doesn't speed up as you play, the paddle
    doesn't shrink after first breaking through to the ceiling, and there's
    no start screen.
"""

import math
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jupylet.app import App
from jupylet.state import State
from jupylet.sprite import Sprite

from jupylet.audio.sound import GatedSound, Envelope, Oscillator
from jupylet.audio.note import note2key


app = App(width=512, height=382)  # ~3:2, matching a reference screenshot


#
# 1. Layout
#
# `unit` is the height of one brick row. Everything below is measured as a
# multiple of it (or of `wall_unit`, a second, independent unit the walls
# are measured in), so bumping `unit` up or down changes the whole look of
# the game at once instead of needing dozens of separate edits.
#

wall_unit = 12.8  # base size the wall width is measured in
SIDE = 2 * wall_unit  # thickness of the left/right walls

unit = wall_unit * 0.9  # thickness of the ceiling and paddle, and one brick row's base size
CEIL_H = round(2*unit)  # rounded once here so every position derived from it lines up exactly, with no gap
top_gap = 3*unit  # empty black space between the ceiling and the top brick row

HUD_H = 3*unit  # height of the black strip above the ceiling, where the scoreboard digits live

PLAY_TOP = round(app.height - HUD_H - CEIL_H)  # y of the ceiling's underside - the ball bounces here
PLAY_LEFT = SIDE
PLAY_RIGHT = app.width - SIDE

COLS, ROWS = 16, 6  # the brick grid
BRICKW, BRICKH = (app.width - 2*SIDE) / COLS, unit * 1.10  # bricks are a bit thicker than the shared unit

ty = PLAY_TOP - top_gap  # y of the top brick row's top edge
bricks_bottom = ty - ROWS * BRICKH  # y of the bottom brick row's bottom edge - the open play area starts here

PADW, PADH = 56, unit / 2  # paddle width and thickness
BALLS = round(PADH)  # the ball is a square the same thickness as the paddle


#
# 2. Colors
#

FRAME_COLOR = '#808280'  # the walls, ceiling, and scoreboard digits
TAN = '#cc9955'  # the paddle and ball

# brick rows, top to bottom - sampled by eye from a reference screenshot, so
# feel free to tweak the hex values.
row_color = ['#3e8c58', '#70b28a', '#c4566c', '#58a8b2', '#d4ac5c', '#5a62d0']


#
# 3. Game feel - tweak these to change how the game plays.
#

STARTING_LIVES = 3

PADDLE_SPEED = 480  # pixels per second, when moved by keyboard
PADDLE_DEFLECT_ANGLE = 60  # degrees - how sharply the ball bounces off the paddle's edges

BALL_START_VX, BALL_START_VY = 162, -198  # pixels per second, at the start of each ball

# where each ball starts: just below the bottom brick row, near the right
# wall - as if it had just been thrown in - but inset by half the ball's own
# size so it's fully inside the court rather than drawn over the wall.
BALL_START_X = PLAY_RIGHT - BALLS/2
BALL_START_Y = bricks_bottom - BALLS


#
# 4. Sound
#
# Real Atari 2600 games have no hardware envelope generator - a game just
# holds a flat volume then snaps it off, instead of fading out like a
# synth does. `Blip` copies that: near-instant on, held flat, then a very
# short (not instant, to avoid a click) release, explicitly timed rather
# than left to decay on its own.
#

class Blip(GatedSound):
    """A short square-wave pulse, like the original's beeps and bloops."""

    def __init__(self, amp=0.3, pan=0., hold=0.16):
        super().__init__(amp=amp, pan=pan)
        self.env0 = Envelope(0.001, 0.002, 1., 0.02, linear=True)
        self.osc0 = Oscillator('square')
        self.hold = hold  # how long the note stays on, in seconds

    def play(self, note=None, **kwargs):
        super().play(note, **kwargs)
        self.gate.close(dt=self.hold)

    def forward(self):
        self.osc0.freq = self.freq
        g0 = self.gate()
        e0 = self.env0(g0)
        o0 = self.osc0()
        return o0 * e0


blip = Blip()

WALL_NOTE = note2key('C6')   # walls, ceiling, and a new ball's onset
PADDLE_NOTE = note2key('D5')

# brick rows, top to bottom (the original numbering was bottom to top, so
# this list is reversed from how it was first given). The top row's note
# wasn't given, so C6 (matching the wall/onset tone) is a guess.
row_notes = [note2key(n) for n in ['C6', 'G4', 'D#4', 'C#4', 'A#3', 'F3']]


#
# 5. Sprites
#

# a Sprite normally loads an image file; here we hand it a plain numpy
# array instead - a solid block of white pixels - and let `color=` tint it.
# That's how every flat-colored shape in this game (walls, bricks, paddle,
# ball, digits) is made, with no image files at all.

paddle = Sprite(np.ones((round(PADH), round(PADW))) * 255, x=app.width/2, y=BRICKH + PADH/2, color=TAN)
ball = Sprite(np.ones((BALLS, BALLS)) * 255, x=BALL_START_X, y=BALL_START_Y, color=TAN)

# the walls - open at the bottom, like the original.
frame_top = Sprite(np.ones((CEIL_H, app.width)) * 255, x=app.width/2, y=PLAY_TOP + CEIL_H/2, color=FRAME_COLOR)
frame_left = Sprite(np.ones((PLAY_TOP, round(SIDE))) * 255, x=SIDE/2, y=PLAY_TOP/2, color=FRAME_COLOR)
frame_right = Sprite(np.ones((PLAY_TOP, round(SIDE))) * 255, x=app.width - SIDE/2, y=PLAY_TOP/2, color=FRAME_COLOR)


#
# A blocky pixel digit font, in the spirit of the original console's
# scoreboard - hand drawn, not lifted from an authentic ROM font. Each
# digit is a 3x5 grid of cells, stretched to a 2:1 (w:h) overall shape.
#
DIGITS = {
    '0': ['111', '101', '101', '101', '111'],
    '1': ['010', '010', '010', '010', '010'],
    '2': ['111', '001', '111', '100', '111'],
    '3': ['111', '001', '111', '001', '111'],
    '4': ['101', '101', '111', '001', '001'],
    '5': ['111', '100', '111', '001', '111'],
    '6': ['111', '100', '111', '101', '111'],
    '7': ['111', '001', '001', '001', '001'],
    '8': ['111', '101', '111', '101', '111'],
    '9': ['111', '101', '111', '001', '111'],
}

DIGIT_H = CEIL_H * 0.85  # a bit less than the ceiling's thickness
DIGIT_W = DIGIT_H * 2
DIGIT_GAP = DIGIT_W / 3  # space between two neighboring digits

BLOCK_W, BLOCK_H = DIGIT_W / 3, DIGIT_H / 5  # size of one "pixel" of the font


def digits_image(s):
    """Draw a string of digits (e.g. '007') as a single black-and-white image."""

    w = len(s) * (DIGIT_W + DIGIT_GAP) - DIGIT_GAP
    img = np.zeros((round(DIGIT_H), round(w)))

    for i, c in enumerate(s):
        x0 = i * (DIGIT_W + DIGIT_GAP)

        for row, bits in enumerate(DIGITS[c]):
            for col, bit in enumerate(bits):
                if bit == '1':
                    y0, y1 = round(row*BLOCK_H), round((row+1)*BLOCK_H)
                    x1, x2 = round(x0 + col*BLOCK_W), round(x0 + (col+1)*BLOCK_W)
                    img[y0:y1, x1:x2] = 255

    return img


# point value of each row, top to bottom - just a made-up descending scale,
# not a documented value from the original.
row_points = [6, 5, 4, 3, 2, 1]

brick_img = np.ones((round(BRICKH), round(BRICKW))) * 255
bricks = []

for row in range(ROWS):
    for col in range(COLS):
        brick = Sprite(
            brick_img,
            x=SIDE + col * BRICKW + BRICKW / 2,
            y=ty - row * BRICKH - BRICKH / 2,
            color=row_color[row],
        )
        brick.points = row_points[row]
        brick.note = row_notes[row]
        bricks.append(brick)

# score and lives sit side by side, centered under the ceiling, separated by
# the width of an empty digit (including its two flanking gaps): 5/3 of a
# digit's width.
group_gap = DIGIT_W * 5 / 3

score = Sprite(digits_image('000'), x=app.width/2, y=app.height - HUD_H/2, anchor_x='right', color=FRAME_COLOR, collisions=False)
lives = Sprite(digits_image(str(STARTING_LIVES)), x=app.width/2 + group_gap, y=app.height - HUD_H/2, anchor_x='left', color=FRAME_COLOR, collisions=False)


@app.event
def render(ct, dt):

    app.window.clear(color='black')

    frame_top.draw()
    frame_left.draw()
    frame_right.draw()

    for brick in bricks:
        brick.draw()

    score.draw()
    lives.draw()

    paddle.draw()
    ball.draw()


#
# 6. Game loop
#

state = State(
    left=False,
    right=False,

    playing=True,

    score=0,
    lives=STARTING_LIVES,

    bvx=BALL_START_VX,
    bvy=BALL_START_VY,
)


@app.event
def key_event(key, action, modifiers):
    """Track which arrow/WASD keys are currently held down."""

    keys = app.window.keys
    pressed = action != keys.ACTION_RELEASE

    if key in (keys.LEFT, keys.A):
        state.left = pressed

    if key in (keys.RIGHT, keys.D):
        state.right = pressed


@app.event
def mouse_position_event(x, y, dx, dy):
    # mimics the original's paddle controller - a potentiometer wheel whose
    # absolute position set the paddle's position directly.
    paddle.x = max(PLAY_LEFT + PADW/2, min(PLAY_RIGHT - PADW/2, x))


@app.run_me_every(1/120)
def update_paddle(ct, dt):
    """Move the paddle from held keys, then keep it inside the walls."""

    if state.left:
        paddle.x -= dt * PADDLE_SPEED

    if state.right:
        paddle.x += dt * PADDLE_SPEED

    paddle.x = max(PLAY_LEFT + PADW/2, min(PLAY_RIGHT - PADW/2, paddle.x))


@app.run_me_every(1/60)
def update_ball(ct, dt):
    """Move the ball, then check every kind of collision it can have:
    the walls, the paddle, the bricks, and missing the paddle entirely."""

    if not state.playing:
        return

    ball.x += state.bvx * dt
    ball.y += state.bvy * dt

    if ball.left <= PLAY_LEFT:
        ball.x -= ball.left - PLAY_LEFT
        state.bvx = -state.bvx
        blip.play(note=WALL_NOTE)

    if ball.right >= PLAY_RIGHT:
        ball.x -= ball.right - PLAY_RIGHT
        state.bvx = -state.bvx
        blip.play(note=WALL_NOTE)

    if ball.top >= PLAY_TOP:
        ball.y -= ball.top - PLAY_TOP
        state.bvy = -state.bvy
        blip.play(note=WALL_NOTE)

    # bounce off the paddle - hitting off-center sends the ball off at a
    # sharper angle, same as the original; hitting dead center keeps it
    # close to vertical.
    if state.bvy < 0 and ball.bottom <= paddle.top and paddle.left <= ball.x <= paddle.right:
        ball.y -= ball.bottom - paddle.top

        speed = (state.bvx ** 2 + state.bvy ** 2) ** 0.5
        offset = max(-1, min(1, (ball.x - paddle.x) / (PADW / 2)))
        angle = math.radians(offset * PADDLE_DEFLECT_ANGLE)

        state.bvx = speed * math.sin(angle)
        state.bvy = speed * math.cos(angle)

        blip.play(note=PADDLE_NOTE)

    for brick in bricks:
        if brick.opacity <= 0:
            continue

        if ball.right >= brick.left and ball.left <= brick.right and ball.top >= brick.bottom and ball.bottom <= brick.top:
            brick.opacity = 0
            state.bvy = -state.bvy

            state.score += brick.points
            score.image = digits_image(str(state.score).zfill(3))

            blip.play(note=brick.note)
            break

    if all(brick.opacity <= 0 for brick in bricks):
        state.playing = False  # the original has no "you win" message, it just stops.

    # ball got past the paddle.
    if ball.top < 0:
        state.lives -= 1
        lives.image = digits_image(str(state.lives))

        if state.lives <= 0:
            state.playing = False  # ditto for "game over" - no message, and the paddle stays controllable.
        else:
            ball.x, ball.y = BALL_START_X, BALL_START_Y
            state.bvx, state.bvy = BALL_START_VX, BALL_START_VY
            blip.play(note=WALL_NOTE)


if __name__ == '__main__':
    app.run()
