# Jupylet

*Jupylet* is a Python library for programming 2D and 3D games, graphics, music 
and sound synthesizers, interactively in a Jupyter notebook. It is intended 
for three types of audiences:

* Computer scientists, researchers, and students of deep reinforcement learning.
* Musicians interested in sound synthesis and live music coding.
* Kids and their parents interested in learning to program.

&nbsp;

<p float="left">
    <img src="https://github.com/nir/jupylet/raw/master/docs/images/spaceship.gif" width="256" />
    <img src="https://github.com/nir/jupylet/raw/master/docs/images/spaceship_3d.gif" width="384" />
</p>

## Jupylet for Kids

A Jupyter notebook is in essence a laboratory for programming. It is the ideal
environment for playing around with code, experimenting, and exploring ideas.
It is used by professional machine learning scientists who come every day to
play at work, so why not by kids?

*Jupylet* is wonderfully easy to use for creating simple 2D and 3D games and 
music interactively and experimentally. Change a variable or a function and 
see how the game is affected immediately while running.

## Jupylet for Deep Reinforcement Learning

*Jupylet* makes it is super easy to create and modify environments in which to
experiment with deep reinforcement learning algorithms and it includes the API
to programmatically control multiple simultaneous games and render thousands 
of frames per second.

Consider for example the pong game included in this code base. With a few
lines of code you can modify the colors of the game to experiment with transfer 
learning, or turn the game into 4-way pong with agents on all four sides of the 
game court to experiment with cooperation between multiple agents. And since you 
can modify the game interactively in Jupyter this process is not only easy but 
also fun.  

Check out the [*Programming Graphics*](https://jupylet.readthedocs.io/en/latest/programmers_reference_guide/graphics.html) 
and the [*Reinforcement Learning*](https://jupylet.readthedocs.io/en/latest/programmers_reference_guide/rl.html) 
chapters in the Jupylet Programmer's Reference Guide.

## Jupylet for Musicians

*Jupylet* imports ideas and methods from machine learning into the domain
of sound synthesis to easily let you create sound synthesizers as wild as you
can dream up - it includes impulse response reverb effects, colored noise 
generators, resonant filters with cutoff frequency sweeping, oscillators with 
LFO modulation, multi sampled instruments, and much more... And all of it in 
pure Python for you to modify and experiment with.

In addition *Jupylet* draws inspiration from the wonderful [*Sonic Pi*](https://sonic-pi.net/)
and brings live loops and live music coding to Jupyter and Python. Hook up 
your MIDI keyboard and take off.

Check out the [*Programming Sound and Music*](https://jupylet.readthedocs.io/en/latest/programmers_reference_guide/sound.html) 
and the [*Programming Synthesizers*](https://jupylet.readthedocs.io/en/latest/programmers_reference_guide/synthesis.html)
chapters in the Jupylet Programmer's Reference Guide.

## Requirements

*Jupylet* should run on Python 3.7 and up on Windows, Mac, and Linux.

## How to Install and Run Jupylet

If you are new to Python, I recommend that you install and use the
[Miniconda Python](https://docs.conda.io/en/latest/miniconda.html)
distribution. 

**On Windows** &ndash; download and run the 64-bit installer for Python 3.11. Once 
Miniconda is installed press the `⊞ Winkey` and then type *Miniconda* and 
press the `Enter` key. This should open a small window that programmers call 
*console* or *shell* in which you can enter commands and run programs.

**On macOS with Intel processor** &ndash; download and run "Miniconda3 macOS Intel x86 64-bit pkg" 
for Python 3.11. Once installed click the Spotlight icon `🔍` and in the search 
field type *terminal* and press the `Enter` key to open the console.

**On macOS with M1 processor** &ndash; download and run "Miniconda3 macOS Apple M1 64-bit pkg" 
for Python 3.11. Once installed click the Spotlight icon `🔍` and in the search 
field type *terminal* and press the `Enter` key to open the console. Then
you need to run the following two commands:

    conda install -c conda-forge numpy "libblas=*=*accelerate"
    pip install --pre -i https://pypi.anaconda.org/scipy-wheels-nightly/simple scipy

**On Linux** &ndash; download "Miniconda3 Linux 64-bit". This should download the file
Miniconda3-latest-Linux-x86_64.sh. Install it by running the following command 
in a bash shell (once installed start a new bash shell):

    bash Miniconda3-latest-Linux-x86_64.sh

---

Once Miniconda is installed it is time to install *jupylet* by typing the 
following command in the console:

    pip install jupylet

Next, to run the example notebooks, download the *jupylet* source code. 
If you have [Git](https://git-scm.com/) installed type the following command:

    git clone https://github.com/nir/jupylet.git

Alternatively, you can download the source code with the following command:

    python -m jupylet download

Next, enter the *jupylet/examples/* directory with the change directory
command:

    cd jupylet/examples/

And start a jupyter notebook with:

    jupyter notebook 11-spaceship.ipynb

Run the notebook by following the instructions in the notebook and a game
canvas should appear with the spaceship example:

<img src="https://github.com/nir/jupylet/raw/master/docs/images/spaceship.gif" width="256" height="256" />

Alternatively, you can run the same game as a Python script from the console 
with:

    python spaceship.py

## Documentation

To get started with Jupylet head over to the *Jupylet Programmer's Reference 
Guide* which you can find at 
[jupylet.readthedocs.io](https://jupylet.readthedocs.io/).  

To complement the online guide check out the growing collection of 
[*example notebooks*](examples/) that you can download and run on your 
computer as explained above.

## Contact

For questions and feedback send an email to [Nir Aides](mailto:nir.8bit@gmail.com) or [join the discussion](https://github.com/nir/jupylet/discussions).

## Spread the Word

Jupylet is a new library and you can help it grow with a few clicks - 
if you like it let your friends know about it!

## Acknowledgements

* [Einar Forselv](https://github.com/einarf) - The programmer behind ModernGL 
for his endless help in the trenches of OpenGL programming.
* [Alban Fichet](https://afichet.github.io/) - For kindly licensing his 
sound visualizer Shadertoy as CC BY 4.0 license.

## What's New in Version 0.9.1

* Support for Python 3.10 and Python 3.11 with MIDI functionality.
* Seamlessly track changes to audio devices on macOS.
* Workaround PIL api change - thanks to [@misolietavec](https://github.com/misolietavec).
* Bug fixes. 

## What's New in Version 0.8.9

* Support for Python 3.10 and Python 3.11 - except for MIDI functionality.
* Support for macOS M1.
* Spectrum analyzer.
* Bug fixes. 

<img src="https://user-images.githubusercontent.com/124126/208634912-6cf956ec-b1e3-43c2-87a1-3c437953b739.png" width=50% height=50%>

## What's New in Version 0.8.8

* Support for Python 3.9. 

## What's New in Version 0.8.7

* Workaround auto-completion bug in Jupyter notebooks. 

## What's New in Version 0.8.6

* Support for rendering Shadertoy OpenGL shaders. 
[Shadertoy](https://www.shadertoy.com/) is an awesome online platform for  
programming and sharing OpenGL shaders online, and now you can 
[use and render shadertoy shaders in Jupylet!](https://jupylet.readthedocs.io/en/latest/programmers_reference_guide/graphics-3d.html#shadertoys)

