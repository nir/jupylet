PROGRAMMING SYNTHESIZERS
========================

The Synthesizers Playground
---------------------------

Jupylet includes a flexible and novel sound synthesis framework with which
you can create substractive, additive, frequency modulation, and sample based 
`sound synthesizers <https://en.wikipedia.org/wiki/Software_synthesizer>`_, 
as wild as you can dream up.

It includes the following building blocks, all of which are implemented in 
pure Python and `Numpy <https://numpy.org/>`_, the Python scientific 
computing library, for you to play with:

* Continuous colored noise generators.
* Antialiased wave oscillators with frequency modulation.
* Resonant digital filters with sweepable cutoff frequency.
* Multisampled instruments with frequency modulation.
* ADSR envelopes with linear and non-linear curves.
* Schroeder type algorithmic reverbs.
* Convolution reberb.
* Phase modulator.
* Overdrive.

A sound synthesizer is in essense a audio signal processing graph. Audio 
signals are manipulated and transformed as they travel through the signal 
processing graph from its inputs to its output.

More generally, since each transformation applied to the audio signal is a 
computation, a software sound synthesizer is a computational graph. 

In recent years there has been an explosion of tools and frameworks to 
represent and work with a different kind of computational graph that is 
rapidly growing in popularity and importance - the `artificial neural network <https://en.wikipedia.org/wiki/Artificial_neural_network>`_.

One of the most successful and surely the most Pythonic of them all is the 
wonderful `Pytorch <https://pytorch.org/>`_. Jupylet borrows from Pytorch 
`the natural way in which it represents computational graphs <https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#define-the-network>`_. 

Let's start with a simple sawtooth oscillator:

.. code-block:: python

    osc = Oscillator('saw')

In Jupylet, all audio elements from basic building blocks to compound
synthesizers are ``Sound`` instances - the parallel of a Pytorch nn.Module; 
and as done in Pytorch you typically apply them to an input to produce output.

In some cases you do not need to provide an input; an oscillator can generate
output without receiving any input:

.. code-block:: python

    In []: a0 = osc()
    In []: a0
    Out[]: array([[-0.07100377],
                  [-0.05801778],
                  [-0.04733572],
                  ...,
                  [-0.86131287],
                  [-0.81941793],
                  [-0.83591645]])

The oscillator output a Numpy array of 1024 numbers. You can verify this with
the following instruction:

.. code-block:: python

    In []: a0.shape
    Out[]: (1024, 1)

More interestingly we can visualize the sawtooth signal. Let's plot the first 
169 numbers:

.. code-block:: python

    get_plot(a0[:169])
    
.. image:: ../images/sawtooth.png 

The little waves that can be seen in the plot are actually a good thing. This 
is how an anti-aliased sawtooth wave should look like.

You can also play this array hear how it sounds like:

.. code-block:: python

    sd.play(a0)

However, 1024 samples last for about 25ms, so the sound will be shortish. We 
can generate and second long signal by calling the ``consume()`` method:

.. code-block:: python

    sd.play(osc.consume(FPS))

