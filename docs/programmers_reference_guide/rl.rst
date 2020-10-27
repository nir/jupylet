DEEP REINFORCEMENT LEARNING
===========================

Jupylet came to life when I was trying to learn Deep Reinforcement Learning 
(Deep RL) and found myself spending too much time trying to hack 
`OpenAI Gym <https://gym.openai.com/>`_ to do what I want. 

I wanted an easy way to run and control multiple game instances directly 
from inside a Jupyter notebook and I wanted it to be easy to modify and 
control any aspect of the game environment.


Run Jupylet on AWS servers
--------------------------

To train any non trivial deep learning agent you need a machine that can 
compute trillions of multiplications per second. Traditionally this simply 
means a machine with an Nvidia GPU.

If you have such a machine at home you can skip this section. If not, this 
section explains how to setup and run Jupylet on a remote Amazon EC2 instance
with a GPU.

Jupylet was tested on Amazon EC2 GPU servers running Ubuntu 18.04. If you 
haven't already setup an EC2 instance I recommend that you instantiate it 
from the `AWS Deep Learning AMI (Ubuntu 18.04) <https://aws.amazon.com/marketplace/pp/Amazon-Web-Services-AWS-Deep-Learning-AMI-Ubuntu-1/B07Y43P7X5>`_
since includes the required Nvidia drivers, CUDA, cuDNN, and conda.

To connect to your EC2 server you will need an SSH client. On Windows machines
you won't find anything better than the awesome `PuTTY <https://www.putty.org/>`_.

On a remote EC2 instance Jupylet runs in so called headless mode. This means 
it uses the Nvidia GPU to render game frames without creating a game window. 
To make this possible you will need to install a few packages. To do that run 
the following commands in an SSH terminal on the remote instance:

.. code-block:: bash

    sudo apt-get update -y  
    sudo apt-get install -y mesa-utils libegl1-mesa xvfb freeglut3-dev

Next, create a new conda environment, activate it, and install Jupylet:

.. code-block:: bash
    
    conda create -y -n jpl python=3.8 pip
    conda activate jpl

    pip install jupylet

Next let's download the jupylet repository so we may run its example 
notebooks:

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
    The `screen` program will prevent Jupyter server from exiting if the SSH 
    terminal accidentally disconnects. If it does disconnect you can 
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


Prepare game for RL
-------------------



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

