# Jupylet

*Jupylet* is a Python library that lets you create and run games interactively
in a Jupyter notebook. It is intended for two types of audiences:

* Kids and their parents interested in learning to program,
* Computer scientists, researchers, and students of deep reinforcement learning.

## TL;DR

Here are two examples of 2D and 3D games included in this repository:

<p float="left">
    <img src="docs/images/spaceship.gif" width="256" />
    <img src="docs/images/spaceship_3d.gif" width="384" />
</p>

## Jupylet for Kids

A Jupyter notebook is in essence a laboratory for programming. It is the ideal
environment for playing around with code, experimenting, and exploring ideas.
It is used by professional machine learning scientists who come every day to
play at work, so why not by kids?

*Jupylet* is wonderfully easy to use for creating simple 2D and 3D games interactively and experimentally. Change a variable or a function and see how the game is affected immediately while it is running.

## Jupylet for Deep Reinforcement Learning

*Jupylet* makes it is super easy to create and modify environments in which to
experiment with deep reinforcement learning algorithms and it includes the API
to programmatically control multiple simultaneous games and render thousands 
of frames per second.

Consider for example the pong game included in this repository. With a few
lines of code you can modify the colors of the game or turn the background into
a swirling giant flower to experiment with transfer learning, or turn the game
into 4-way pong with agents on all four sides of the game court to experiment
with cooperation between multiple agents. And since you can modify the game
interactively in Jupyter this process is not only easy but also fun.  

Head to [examples/22-pong-RL.ipynb](examples/22-pong-RL.ipynb) to see how to programmatically control a 2-player version of pong.

## Requirements

_jupylet_ should run on Python 3.6+ on Windows, Mac, and Linux.

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

If you are using Python 3.8 on Windows you then need to run following command as well:

    python -m jupylet postinstall

Next, download the *jupylet* repository since it contains the
example notebooks. If you have [Git](https://git-scm.com/) installed you
can use it to clone the *jupylet* repository with:

    git clone https://github.com/nir/jupylet.git

Alternatively, if you don't have Git installed, you can download and unzip
the *jupylet* archive by typing:

    python -m wget https://github.com/nir/jupylet/archive/master.zip
    python -m zipfile -e jupylet-master.zip .
    move jupylet-master jupylet

| `⚠️ NOTE` On Mac OS X or Linux type *mv* instead of *move* in the command above. |
| --- |

Next, enter the *jupylet/examples/* directory with the change directory
command:

    cd ./jupylet/examples/

And start a jupyter notebook with:

    jupyter notebook 11-spaceship.ipynb

Run the notebook by following the instructions in the notebook and a game
canvas should appear with the spaceship example:

<img src="docs/images/spaceship.gif" width="256" height="256" />

Alternatively, you can run the same game as a Python script from the console with:

    python spaceship.py

## Documentation

At the moment the bulk of *Jupylet's* documentation is to be found in the [example notebooks](examples/). Head to [examples/01-hello-world.ipynb](examples/01-hello-world.ipynb) to get started. 

I have started writing a [guide](https://jupylet.readthedocs.io/en/latest/). If you like *Jupylet*, one of the best ways to contribute to this project is to help to document it. 

## Contact

For questions and feedback send an email to [Nir Aides](mailto:nir@winpdb.org).
