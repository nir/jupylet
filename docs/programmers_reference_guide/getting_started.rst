GETTING STARTED
===============

How to Install and Run Jupylet
------------------------------

Install and use the
`Miniforge Python <https://github.com/conda-forge/miniforge>`_ distribution,
following the instructions below for your operating system.

**On Windows 11** -- download and execute `the Miniforge Windows installer
<https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe>`_.
Once Miniforge is installed press the :guilabel:`⊞ Winkey` and then type
*Miniforge* and press the :guilabel:`Enter` key. This should open a small
window that programmers call *console* or *shell* in which you can enter
commands and run programs.

**On macOS with Apple Silicon** -- download and execute `the Miniforge PKG installer for Apple Silicon
<https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.pkg>`_.
Once installed click the Spotlight icon :guilabel:`🔍` and in the search field
type *terminal* and press the :guilabel:`Enter` key to open the console.

**On Ubuntu 26.04** -- download `"Miniforge3-Linux-x86_64.sh"
<https://github.com/conda-forge/miniforge/releases/download/26.7.2-0/Miniforge3-Linux-x86_64.sh>`_.
Install it by running the following command in a bash shell (once installed
start a new bash shell):

.. code-block:: bash

    bash Miniforge3-Linux-x86_64.sh

On Ubuntu Linux Jupylet also needs a few system packages: ``libportaudio2`` for sound, and
``libgl1-mesa-dev``/``libegl1-mesa-dev`` to run scripts directly with
``python foo.py`` (as opposed to from a notebook):

.. code-block:: bash

    sudo apt update
    sudo apt install libportaudio2 libgl1-mesa-dev libegl1-mesa-dev

------------

Once Miniforge is installed, first install precompiled versions of two of
jupylet's dependencies (this avoids possibly needing a C++ compiler,
depending on your Python version):

.. code-block:: bash

    conda install moderngl glcontext

Now it's time to install *jupylet* itself by typing the following command in
the console:

.. code-block:: bash

    pip install jupylet

Next, to run the example notebooks download the *jupylet* source code. If 
you have `Git <https://git-scm.com/>`_ installed type the following command:

.. code-block:: bash

    git clone https://github.com/nir/jupylet.git

Alternatively, you can download the source code with the following command:

.. code-block:: bash

    python -m jupylet download

Next, enter the *jupylet/examples/* directory with the change directory
command:

.. code-block:: bash

    cd jupylet/examples/

The example notebooks need to be trusted once on your computer, or the game
canvas might not show up. Make sure to run the following command from
inside the *examples/* folder:

.. code-block:: bash

    python -m jupylet trust_notebooks

And start a jupyter notebook with:

.. code-block:: bash

    jupyter lab 11-spaceship.ipynb

Run the notebook by following the instructions in the notebook and a game
canvas should appear with the spaceship example:

.. image:: ../images/spaceship.gif

Alternatively, you can run the same game as a Python script from the console 
with:

.. code-block:: bash

    python spaceship.py

The Python Programming Language
-------------------------------

Python is an awesome programming language. It is both simple for kids to
learn and powerful enough to be `one of the most popular programming languages
<https://www.tiobe.com/tiobe-index/>`_ among computer scientists and
programmers.

However, this reference guide is not designed to teach the Python programming
language. If you don't already have a working knowlege of Python and how to
use it to program, I would like to suggest a few resources that may help you
get started:

- `futurecoder <https://futurecoder.io/>`_ \- a free and open source course
  that teaches programming and Python from scratch, fully interactively,
  right in your browser. No installation or account needed - just start
  typing code and follow along.

- `Python's own tutorial <https://docs.python.org/3/tutorial/index.html>`_
  \- Perhaps not as interactive, but it is a good idea to get familiar with
  Python's official documentation.

- `Mike Dane's Learn Python Yotube tutorial <https://www.youtube.com/watch?v=rfscVS0vtbw>`_
  \- Appears to be a good didactic introduction to Python.

These guides will instruct you how to start a python interpreter where you
can type and run Python code. You may do that, but once you gain a little bit
of confidence or if you feel adventurous try starting a Jupyter notebook
instead of a simple python interpreter.

To do that start the Miniforge Prompt
`as explained above <#how-to-install-and-run-jupylet>`_, then change
directory into the *jupylet/examples/* directory and start a new notebook by
typing:

.. code-block:: bash

    jupyter lab 01-hello-world.ipynb

Jupyter Notebooks
-----------------

Jupyter notebooks are awesome but they can be a little confusing at
first. Here are a couple of resources that explain how to use them:

- The *01-hello-world.ipynb* notebook you already have in the
  *jupylet/examples/* directory doubles as a hands-on introduction to
  Jupyter notebooks - open it and give it a try. It walks you through the
  difference between markdown and code cells, editing and running cells, and
  points you to Jupyter's own built-in :guilabel:`Help` menu's
  :guilabel:`User Interface Tour` for a guided tour of the rest of the
  interface.

- `JupyterLab Tutorial: Python as a Calculator <https://www.youtube.com/watch?v=AoqM3TqTB6c&list=PLG7vrjhTP1d7DZJQh9ee8q4B63enMv8DV&index=3>`_
  \- A friendly, unintimidating walkthrough of the JupyterLab interface
  itself - cells, the toolbar, the menus, and more - picking up a bit of
  basic Python syntax along the way. It's part of a short video series, so
  if you like this style feel free to continue watching from there.

