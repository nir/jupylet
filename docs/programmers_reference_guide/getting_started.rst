GETTING STARTED
===============

How to Install and Run Jupylet
------------------------------

If you are new to Python, I recommend that you install and use the
`Miniconda Python <https://docs.conda.io/en/latest/miniconda.html>`_
distribution. 

On Windows download and run the 64-bit installer for Python 3.9. Once 
Miniconda is installed press the :guilabel:`⊞ Winkey` and then type 
*Miniconda* and press the :guilabel:`Enter` key. This should open a small 
window that programmers call *console* or *shell* in which you can enter 
commands and run programs.

On Mac OS X download and run "Miniconda3 MacOSX 64-bit pkg" for Python 3.9.
Once installed click the Spotlight icon :guilabel:`🔍` and in the search field 
type *terminal* and press the :guilabel:`Enter` key to open the console.

To run *jupylet* first install it by typing the following command in the
console:

.. code-block:: bash

    pip install jupylet

If you are using Python 3.8 or 3.9 on Windows you also need to run following command:

.. code-block:: bash

    python -m jupylet postinstall

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

And start a jupyter notebook with:

.. code-block:: bash

    jupyter notebook 11-spaceship.ipynb

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

- `Microsoft's introduction to Python <https://docs.microsoft.com/en-us/learn/modules/intro-to-python/1-introduction>`_
  \- Microsoft has a long tradition of publishing good guides to programming
  languages and this tutorial appears to be in line with this tradition. 
  However, their Azure Cloud Shell is unfortunately a distraction. You would 
  be better off trying out their exercises in Python's own `online shell <https://www.python.org/shell/>`_.

- `Python's own tutorial <https://docs.python.org/3/tutorial/index.html>`_
  \- Perhaps not as didactic as Microsoft's guide, but it is a good idea to
  get familiar with Python's official documentation.

- `Mike Dane's Learn Python Yotube tutorial <https://www.youtube.com/watch?v=rfscVS0vtbw>`_
  \- Appears to be a good didactic introduction to Python.

These guides will instruct you how to start a python interpreter where you
can type and run Python code. You may do that, but once you gain a little bit
of confidence or if you feel adventurous try starting a Jupyter notebook
instead of a simple python interpreter.

To do that start the Miniconda Prompt
`as explained above <#how-to-install-and-run-jupylet>`_, then change
directory into the *jupylet/examples/* directory and start a new notebook by
typing:

.. code-block:: bash

    jupyter notebook 01-hello-world.ipynb

Jupyter Notebooks
-----------------

Jupyter notebooks are awesome but they can be a little confusing at
first. Here are a few resources that explain how to use them:

- `examples/01-hello-world.ipynb <https://github.com/nir/jupylet/blob/master/examples/01-hello-world.ipynb>`_ 
  notebook contains a basic introduction to Jupyter notebooks. Check it out.

- `Running Code <https://mybinder.org/v2/gh/jupyter/notebook/master?filepath=docs%2Fsource%2Fexamples%2FNotebook%2FRunning%20Code.ipynb>`_
  \- This is a Jupyter notebook explaining how to use Jupyter notebooks 🙂.
  It is in fact a live notebook running in a web service called mybinder. The
  first time you click it may take a moment to start, so give it a moment.
  Since it is "live" you can play around with it. It works!

- `Jupyter's documentation <https://jupyter-notebook.readthedocs.io/en/latest/notebook.html>`_
  \- There's a whole lot of text in there.

