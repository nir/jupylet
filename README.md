# jupylet

_jupylet_ is the marriage of Jupyter and the pyglet game programming library.

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

