GETTING STARTED
===============

How to Install Jupylet
----------------------

If you are new to Python, I strongly recommend that you install and use the
`Miniconda Python <https://docs.conda.io/en/latest/miniconda.html>`_
distribution. Download and run the 64-bit installer and stick to the default
install options.

Once Miniconda is installed start a Miniconda Prompt. To do this on Windows
click the :guilabel:`⊞ Winkey`  then search for "Miniconda Prompt" and
click it. This should open a small dark window that programmers call *console*
or *shell* in which you can enter commands and run programs.

To run *jupylet* first install it by typing the following command in the console:

.. code-block:: bash

    pip install jupylet

Next, download the *jupylet* archive or use `Git <https://git-scm.com/>`_ to clone this repository with:

.. code-block:: bash

    git clone https://github.com/nir/jupylet.git

Then enter the *jupylet/examples/* folder and start a jupyter notebook with:

.. code-block:: bash

    jupyter notebook spaceship.ipynb

Run the notebook and a game canvas should appear with the spaceship example:

.. image:: ../images/spaceship.gif

