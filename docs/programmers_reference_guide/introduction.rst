INTRODUCTION
============

*Jupylet* is a Python library that lets you create and run games interactively
in a Jupyter notebook. It is intended for two types of audiences:

* Kids and their parents interested in learning to program,
* Researchers and students of deep reinforcement learning.


Kids Learning to Program
------------------------

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
:any:`Let's get started!<getting_started>`


Deep Reinforcement Learning with Jupylet
----------------------------------------

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

