DEEP REINFORCEMENT LEARNING
===========================

Pong
----

`Pong <https://en.wikipedia.org/wiki/Pong>`_ is the first computer game I 
ever played back in the 70s, and therefore I like the idea of teaching it to  
a computer. It feels like returning the favor.

Besides, the great Andrej Karpathy has written a beautifully clear post on 
training machines to play `Pong` using the conceptually simple 
`Policy Gradients` technique. You can find the post here - 
`http://karpathy.github.io/2016/05/31/rl/ <http://karpathy.github.io/2016/05/31/rl/>`_

So `Pong` it is. 


Prepare a Game for RL
---------------------

When I program a new game in Jupylet I enjoy doing it interactively in a 
Jupyter notebook while it is running. This is how all the Jupylet example 
notebooks were created. However, to programmatically control a game in RL we 
need it in the form of a Python module, and we need to define a ``step()``
function and a ``reset()`` function.

To convert a Jupylet game into a Python module, simply select 
`Download as Python` from the Jupyter notebook :guilabel:`File` menu. 

Alternatively you can select and copy all the notebook cells with 
:guilabel:`Ctrl+A` and :guilabel:`Ctrl+C` and then paste everything into a 
text editor. However, in this case you will need to manually comment out 
any markup text.

Let's try that with the `Pong` example notebook `examples/21-pong.ipynb <https://github.com/nir/jupylet/blob/master/examples/21-pong.ipynb>`_.
That notebook implements a two player version of `Pong`. Run it to see what 
it does and then convert it to a Python module by the name `21-pong.py` and 
save it in the `examples/` folder.

If you did it correctly you should be able to run the game from a Miniconda 
prompt with:

.. code-block:: bash

    python examples/21-pong.py

Technically speaking `21-pong.py` is a Python script, not a module. A Python
module is a collection of functions and data structures that is meant to be 
imported, not run; a library that other modules or scripts may import for its 
functionality. It should not start a game of its own accord.

However, in Python the distinction between scripts and modules is a bit blurry 
and scripts often double as modules and vice versa.

It's actually nice to be able to run the `Pong` game from the command line, 
so let's keep that functionality and modify `21-pong.py` to double as a python 
module.

To do that open the file in a text editor and replace the ``app.run()`` call 
at the end of the file with:

.. code-block:: python

    if __name__ == '__main__':
        app.run()

This is a very `common programming pattern in Python <https://realpython.com/python-main-function/>`_.
The magical `__name__` variable will only equal `'__main__'` if  
`21-pong.py` is run as a Python script (e.g. as we did above).

Next we need to fix the module's filename. While you can name 
a Python script anything you want, Python module names follow the same 
restrictions as Python variable names and cannot contain dashes; therefore, a 
filename like `pong.py` would be much better.

However, don't rename the file just yet since we have already done that for 
you, and we have also added the required definitions for ``step()`` and 
``reset()`` in there. 

You can find the final version of the `Pong` game in its module form at 
`examples/pong.py <https://github.com/nir/jupylet/blob/master/examples/pong.py>`_.

Let's look at the ``step()`` and ``reset()`` functions now. The ``step()`` 
function is r

.. code-block:: python

    def step(player0=[0, 0, 0, 0, 0], player1=[0, 0, 0, 0, 0], n=1):
        
        state.key_a, state.key_d = player0[:2]
        
        state.left, state.right = player1[:2]
        
        sl0 = state.sl
        sr0 = state.sr
        
        if app.mode == 'hidden': 
            app.step(n)
            
        reward = (state.sl - sl0) - (state.sr - sr0)

        return {
            'screen0': app.observe(),
            'player0': {'score': state.sl, 'reward': reward},
            'player1': {'score': state.sr, 'reward': -reward},
        }


    def reset():
        return load('pong-start.state')
        
        
    def load(path):
        return app.load_state(path, state, ball, padl, padr, scorel, scorer)
        

    def save(path=None):
        app.save_state('pong', path, state, ball, padl, padr, scorel, scorer)


Reinforcement learning is often episodic. For example in `Pong` the agent 
does not need to play indefinitely or to a particular score, but instead the 
game can be periodically reset to its beginning. 

In the case of the this particular implementation of `Pong`, the `state`, 
`ball`, `padl`, `padr`, `scorel`, `scorer` arguments are the objects that 
uniquely determine the game state. In general you can pass any object that 
implements the ``get_state()`` and ``set_state()`` methods.


Control a Game Instance
-----------------------

To help you get started with Jupylet for Deep RL, I have created the 
`examples/22-pong-RL.ipynb <https://github.com/nir/jupylet/blob/master/examples/22-pong-RL.ipynb>`_ 
notebook. This section walks through that notebook and explains it.

.. note::
    The two functions ``show_image()`` and ``show_images()`` used here to show
    numpy arrays as bitmap images are defined in `examples/22-pong-RL.ipynb <https://github.com/nir/jupylet/blob/master/examples/22-pong-RL.ipynb>`_.

Starting a game instance is as easy as this:

.. code-block:: python

    import jupylet.rl

    pong = jupylet.rl.GameProcess('pong')


Render Thousands of Frames Per Second
-------------------------------------


Jupylet in the Cloud
--------------------

To train any non trivial deep learning agent you need a machine that can 
compute trillions of multiplications and additions per second. Traditionally 
this simply means a machine with an Nvidia GPU.

If you have such a machine at home you can skip this section. If not, this 
section explains how to setup and run Jupylet on a remote Amazon EC2 instance
with a GPU.

Jupylet was tested on Amazon EC2 GPU servers running Ubuntu 18.04. If you 
haven't already setup an EC2 instance I recommend that you instantiate it 
from the `AWS Deep Learning AMI (Ubuntu 18.04) <https://aws.amazon.com/marketplace/pp/Amazon-Web-Services-AWS-Deep-Learning-AMI-Ubuntu-1/B07Y43P7X5>`_
since it includes the required Nvidia drivers, CUDA, cuDNN, and conda.

To connect to your EC2 server you will need an SSH client. On Windows machines
you won't find anything better than the awesome `PuTTY <https://www.putty.org/>`_.

On a remote EC2 instance Jupylet runs in so called headless mode. This means 
it uses the Nvidia GPU to render game frames without creating a game window. 
To make this possible you will need to install a few packages by running 
the following commands in an SSH terminal on the remote instance:

.. code-block:: bash

    sudo apt-get update -y  
    sudo apt-get install -y mesa-utils libegl1-mesa xvfb freeglut3-dev

Next, create a new conda environment, activate it, and install Jupylet:

.. code-block:: bash
    
    conda create -y -n jpl python=3.8 pip
    conda activate jpl

    pip install jupylet

Next, download the jupylet repository so you may run its example notebooks:

.. code-block:: bash

    sudo apt-get install -y git

    git clone https://github.com/nir/jupylet.git

Now each time you would like to start a Jupyter notebook server on the remote 
instance, open an SSH terminal and type the following:

.. code-block:: bash
    
    screen
    conda activate jpl
    cd jupylet/examples
    jupyter notebook --no-browser --ip=*

.. note::
    The `screen` program will prevent the Jupyter server from exiting if the 
    SSH terminal accidentally disconnects. If it does disconnect you may 
    reconnect to the running screen session with the ``screen -dr`` command.

The ``jupyter notebook`` command above should produce some output including a 
security token in the form of a long string of hex digits. Copy that token 
since you will soon need it.

Finally, open a new tab in your browser and navigate to port 8888 of the 
public DNS address of your EC2 instance. It should look something like 
`http://ec2-BLAH.BLAH.BLAH.compute.amazonaws.com:8888/`

If you did everything correctly you will be prompted to enter the security 
token that you copied above. Paste it in and you are done.

.. note::
    Jupyter notebook sessions use regular unsecure HTTP connections. If you 
    wish you can setup the Jupyter server to use HTTPS or limit the EC2 
    firewall to only allow connections from your IP address.

