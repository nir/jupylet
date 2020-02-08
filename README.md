# Jupylet

*Jupylet* is a Python library that lets you create and run games interactively
in a Jupyter notebook. It is intended for two types of audiences:

* Kids and their parents interested in learning to program,
* Researchers and students of deep reinforcement learning.


## Kids Learning to Program

A Jupyter notebook is in essence a laboratory for programming. It is the ideal
environment for playing around with code, experimenting, and exploring ideas.
It is used by professional machine learning scientists who come every day to
play at work, so why not by kids?

*Jupylet* is built on top of pyglet, a powerful game programming library. It is
wonderfully easy to use for creating simple 2D games and since pyglet uses
OpenGL you can use it to create 3D games if you really want to. The sky is the
limit really.

By marrying Jupyter and pyglet, *Jupylet* lets you have the best of both
worlds. Create games interactively and experimentally, change a variable or a
function and see how the game is affected immediately while it is running.

## Deep Reinforcement Learning with Jupylet

*Jupylet* makes it is super easy to create and modify environments in which to
experiment with deep reinforcement learning algorithms and it includes the code
required to programmatically control multiple simultaneous games and render
thousands of frames per second.

Consider for example the pong game included in this repository. With a few
lines of code you can modify the colors of the game or turn the background into
a swirling giant flower to experiment with transfer learning, or turn the game
into 4-way pong with agents on all four sides of the game court to experiment
with cooperation between multiple agents. And since you can modify the game
interactively in Jupyter this process is not only easy but fun.  

Head to [examples/pong-RL.ipynb](https://github.com/nir/jupylet/blob/master/examples/pong-RL.ipynb) to see how to programmatically control and render a 2-player version of pong.

## Requirements

_jupylet_ should run on Python 3.4+ on Windows, Mac, and Linux.

## How to Install and Run Jupylet

If you are new to Python, I strongly recommend that you install and use the
[Miniconda Python](https://docs.conda.io/en/latest/miniconda.html)
distribution. Download and run the 64-bit installer and stick to the default
install options.

Once Miniconda is installed start a Miniconda Prompt. To do this on Windows
click the `⊞ Winkey` then type *Miniconda* and press the
`Enter` key. This should open a small dark window that programmers
call *console* or *shell* in which you can enter commands and run programs.

To run *jupylet* first install it by typing the following command in the
console:

    pip install jupylet

Next, you need to download the *jupylet* repository since it contains the
example notebooks. If you have [Git](https://git-scm.com/) installed you
can use it to clone the *jupylet* repository with:

    git clone https://github.com/nir/jupylet.git

Alternatively, if you don't have Git installed, you can download and unzip
the *jupylet* archive by typing:

    python -m wget https://github.com/nir/jupylet/archive/master.zip
    python -m zipfile -e jupylet-master.zip .
    move jupylet-master jupylet

| `⚠️ NOTE:` On Mac OS X or Linux type *mv* instead of *move* in the command above. |
| --- |

Next, enter the *jupylet/examples/* directory with the change directory
command:

    cd ./jupylet/examples/

And start a jupyter notebook with:

    jupyter notebook spaceship.ipynb

Run the notebook by following the instructions in the notebook and a game
canvas should appear with the spaceship example:

<img src="https://raw.githubusercontent.com/nir/jupylet/master/docs/images/spaceship.gif" width="256" height="256" />

You can run essentially the same code from the console with:

    python spaceship.py

The only difference in the code is that the application object is instantiated with:

    app = App(mode='window')

## Documentation

In terms of its interface _jupylet_ introduces only minor additions and modifications to the underlying pyglet library. Therefore the bulk of learning to use it is covered by the [pyglet documentation](https://pyglet.readthedocs.io/en/stable/). I have started writing a [guide](https://jupylet.readthedocs.io/en/latest/) and will add documentation in the comming days; in the mean time head to the [spaceship.ipynb](https://github.com/nir/jupylet/blob/master/examples/spaceship.ipynb) example to get started. 

## Contact

For questions and feedback send an email to [Nir Aides](mailto:nir@winpdb.org).
