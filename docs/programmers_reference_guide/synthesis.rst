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

A sound synthesizer is in essense an audio signal processing graph. Audio 
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


Oscillators
-----------

Let's start with a simple sawtooth oscillator:

.. code-block:: python

    osc = Oscillator('saw')

In Jupylet, all audio elements from basic building blocks to compound
synthesizers are ``Sound`` instances, the parallel of a Pytorch ``nn.Module``, 
and similarly you typically apply them to an input to produce output.

Let's ask the oscilator to generate `44100 frames <https://en.wikipedia.org/wiki/44,100_Hz>`_ 
which is the default number of samples per second used by Jupylet and the 
sampling rate most commonly used in recorded audio:

.. code-block:: python

    In []: a0 = osc(frames=44100)
    In []: a0
    Out[]: array([[-0.07100377],
                  [-0.05801778],
                  [-0.04733572],
                  ...,
                  [-0.86131287],
                  [-0.81941793],
                  [-0.83591645]])

These numbers are time series values corresponding to a sawtooth signal. Let's 
visualize them by plotting the first 169 numbers:

.. code-block:: python

    get_plot(a0[:169])
    
.. image:: ../images/sawtooth.png 

The little waves on the sawtooth are actually a good thing. This is how an 
anti-aliased sawtooth wave should look like.

You can play this array to hear how it sounds with:

.. code-block:: python

    sd.play(a0)

.. raw:: html

   <audio controls="controls">
         <source src="../_static/audio/sawtooth.ogg" type="audio/ogg">
         Your browser does not support the <code>audio</code> element.
   </audio>
   <br>
   <br>

In Jupylet you can use an audio signal to modulate the frequency of an 
oscillator; it is called `frequency modulation (FM) <https://en.wikipedia.org/wiki/Frequency_modulation>`_. 
Let's use a 100Hz sine wave to modulate the frequency of a 1000Hz sine wave:

.. code-block:: python

    osc0 = Oscillator('sine', 100)
    osc1 = Oscillator('sine', 1000)

    a0 = osc0() * 12
    a1 = osc1(a0)

Frequency modulation is done in logarithmic scale with semitones as units;
in this case we multiply the modulating signal by 12 so the carrier signal is 
modulated by one octave (12 semitones) up and down. Let's see how the signal 
looks like:

.. code-block:: python

    get_plot(a1)

.. image:: ../images/fm-sawtooth.png 


A Simple Synthesizer
--------------------

We can now take these two oscillators and write our first simple FM 
synthesizer:

.. code-block:: python

    class SimpleFMSynth(Sound):
        
        def __init__(self):
            
            super().__init__()
                    
            self.osc0 = Oscillator('sine', 10)
            self.osc1 = Oscillator('sine')
        
        def forward(self):
            
            a0 = self.osc0() * 12
            a1 = self.osc1(a0, freq=self.freq)
            
            return a1

Let's instantiate it and play a few notes, and while we're at it, let's also 
learn how to grab a recording of the audio output:

.. code-block:: python

    synth = SimpleFMSynth()

    start_recording()

    synth.play(C6)
    await sleep(1)
    synth.play_release()

    synth.play(D6)
    await sleep(1)
    synth.play_release()

    synth.play(E6)
    await sleep(1)
    synth.play_release()

    a0 = stop_recording()
    sf.write('simple-fm-synth.ogg', a0, 44100)

.. raw:: html

   <audio controls="controls">
         <source src="../_static/audio/simple-fm-synth.ogg" type="audio/ogg">
         Your browser does not support the <code>audio</code> element.
   </audio>
   <br>
   <br>

