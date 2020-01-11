# jupylet

_jupylet_ is the marriage of Jupyter and the pyglet game programming library. It is intended for two types of audiences:
* Kids and their parents interested in learning to program, 
* People doing reinforcement learning research and development.

A Jupyter notebook is in essence a laboratory for programming. It is the ideal environment for playing around with code, experimenting, and exploring ideas. It is used by professional machine learning scientists who come every day to play at work, so why not by kids?

pyglet is a powerful game programming library. It is wonderfully easy to use for creating simple 2D games, and since it is built on top of OpenGL it is powerfull enough for creating 3D engines if you really want to. The sky is the limit really.

By marrying Jupyter and pyglet, you get the best of both worlds. Learn to program by creating games interactively. Watch the game change as you type new code. Change a variable or a function and the game will be affected immediately. 

_jupylet_ is also intended as a tool for reinforcement learning research and development. It will be easy to use to create novel environments in which to experiment with reinforcement learning algorithms; and thanks to pyglet and OpenGL it should be possible to render thousands of frames per second. Most of the parts are already there. Stay tuned...

# Requirements

_jupylet_ should run on Python 3.4+ on Windows, Mac, and Linux.

# Installation

To run _jupylet_ first install its dependencies with:

    pip install pyglet webcolors numpy

    pip install jupyter ipyevents ipycanvas

Download the _jupylet_ archive or clone this repository with:

    git clone git@github.com:nir/jupylet.git

Then enter the _./jupylet/examples/_ folder and start a jupyter notebook with:

    jupyter notebook spaceship.ipynb

Run the notebook and a game canvas should appear with the spaceship example:

<img src="https://raw.githubusercontent.com/nir/jupylet/master/docs/images/spaceship.jpg" width="256" height="256" />

You can run essentially the same code from the console with:

    python spaceship.py

The only difference in the code is that the application object is instantiated with:

    app = App(mode='window')

