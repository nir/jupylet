PROGRAMMING SOUND AND MUSIC
===========================

Da-Da-Da-DUM!
-------------

*Jupylet* is full of sound. Here are some of its features:

* Play audio tracks and samples in WAV, FLAC, and OGG formats.
* Basic support of `SFZ format <https://sfzformat.com/>`_ for playing 
  multi-sampled instruments.
* Simple to use MIDI support - just hook your piano keyboard and play.
* A novel framework for programming sound synthesizers and effects.
* Live loops and music coding in the spirit of `Sonic Pi <https://sonic-pi.net/>`_.


Play it, Sam
------------

If all you want is to add some music or sound effects to your game then there 
is nothing easier. First, let's power up *Jupylet's* audio with the following 
import statement:

.. code-block:: python

    from jupylet.audio.bundle import *

Then create a ``Sample`` instance and load it into memory. Here is how its 
done in the `examples/21-pong.ipynb <https://github.com/nir/jupylet/blob/master/examples/21-pong.ipynb>`_
example notebook:

.. code-block:: python

    pong_sound = Sample('sounds/pong-blip.wav').load()

.. note::
    You can play any `WAV`, `FLAC`, or `OGG` audio file. If you would like to
    play an unsupported audio file such as `MP3` you can convert it to one of 
    the supported formats with a conversion tool or an online service such as 
    `Convertio <https://convertio.co/audio-converter/>`_.

Once you have loaded the sample you can play it any time you want with:

.. code-block:: python
    
    pong_sound.play()

The ``Sample.play()`` method accepts an `amp` parameter between 0 and 1 that 
controls output volume, and a `pan` parameter between -1 and 1 that controls
the left-right balance. So to play the sound effect half as loud and just on
the left side try:

.. code-block:: python
    
    pong_sound.play(pan=-1, amp=0.5)

By default, Jupylet sounds are `monophonic <https://en.wikipedia.org/wiki/Polyphony_and_monophony_in_instruments#Monophonic>`_, 
which means that if you call the ``play()`` method twice in quick succession 
you will not hear two instances of the sample being played simultaneously. 
Instead Jupylet will stop the first sound before it starts playing the second.

You can play multiple instances of the same sound polyphonically (e.g. think 
of how fireworks sound) like this:

.. code-block:: python
    
    pong_sound.play_poly()
    pong_sound.play_poly()
    pong_sound.play_poly()

If the sample is a long playing sound track you may stop it any time with:

.. code-block:: python
    
    pong_sound.play_release()


A Few Notes
-----------

Let's move on to something more interesting; let's play a simple middle C 
note for the duration of a full note on a predefined synthesizer:

.. code-block:: python

    tb303.play(C4, 1)

.. note::
    The tb303 is a predefined synthesizer that produces a sound reminiscent 
    of the `Roland TB-303 <https://en.wikipedia.org/wiki/Roland_TB-303>`_ 
    synthesizer from the early 80s that initially failed commercially but 
    years later became a staple of electronic music. We will see later how 
    it can be implemented in Jupylet using just a few lines of code.

To play a sequence of notes insert a special sleep instructions between them:

.. code-block:: python

    tb303.play(G, 1/8)
    await sleep(1/8)

    tb303.play(G, 1/8)
    await sleep(1/8)

    tb303.play(G, 1/8)
    await sleep(1/8)

    tb303.play(Eb, 3/4)

.. note::
    The `await` instruction is part of `asynchronous Python programming <https://realpython.com/async-io-python/>`_
    which may be considered advanced Python. In general, you can only use 
    `await` in an IPython interpreter session, in a Jupyter notebook cell, 
    or inside asynchronous functions such as a Jupylet live loop. You cannot 
    directly use `await` in a regular python script. i.e. if you copy the 
    code above into a Jupyter notebook cell, it should work just fine, but 
    if you copy it into a text file and try to run it as a Python script, 
    it will exit with an error.

In the code above `Eb` means `E flat`, and similarly `Es` would mean `E sharp`. 
The unit of duration is a full note, and you can set the `beats per minute` and 
the `note value` with ``set_bpm()`` and ``set_note_value()`` respectively.

Normally if you play a new note the previous note will cease as if the 
synthesizer is monophonic. If you would like to play multiple notes together 
call the ``play_poly()`` method instead. It will start playing a new note and 
return a reference to the new sound instance so you may control it as it plays:

.. code-block:: python

    c = tb303.play_poly(C5)
    f = tb303.play_poly(F5)

    await sleep(1)
    c.play_release()

    await sleep(1)
    f.play_release()

Playing the tb303 without specifying a duration will generate a note that goes
on indefinitely, like pressing a keyboard key without releasing it. The code 
above calls ``play_release()`` to release the notes individually. You can also 
release all the currently playing sounds of a synthesizer like this:

.. code-block:: python

    tb303.play_poly(C5)
    tb303.play_poly(F5)

    await sleep(1)

    tb303.play_release()

There are a few more predefined synthesizers to choose from, notably one that 
is reminiscent of the famous `Hammond organ <https://en.wikipedia.org/wiki/Hammond_organ>`_ 
that you can instantiate and use like this:

.. code-block:: python

    hammond = Hammond()

    hammond.play_poly(C4)
    await sleep(1/2)

    hammond.play_poly(D4)
    await sleep(1/2)

    hammond.play_poly(G4)
    await sleep(1)

    hammond.play_release()


Playing with Virtual Instruments
--------------------------------

Besides synthesizers, you can also play music with virtual sampled instruments. 
Jupylet includes basic support for the `SFZ format <https://sfzformat.com/>`_ 
that will enable you to play a variety of instruments that you can find online.

The piano notebook `examples/21-pong.ipynb <https://github.com/nir/jupylet/blob/master/examples/21-pong.ipynb>`_ 
uses a multisampled xylophone from the `VCSL library <https://vis.versilstudios.com/vcsl.html>`_ 
by Versilian Studios. Let's see how:

.. code-block:: python

    xylo = Sample('sounds/VCSL/Xylophone/Xylophone - Medium Mallets.sfz', amp=12)

    xylo.play_poly(C)
    await sleep(1/8)

    xylo.play_poly(D)
    await sleep(1/8)

    xylo.play_poly(E)
    await sleep(1/8)

    xylo.play_poly(F)
    await sleep(1/8)

    xylo.play_poly(G)
    await sleep(1/8)

    xylo.play_poly(F)
    await sleep(1/8)

    xylo.play_poly(E)
    await sleep(1/8)

    xylo.play_poly(D)
    await sleep(1/8)

    xylo.play_poly(C)
    await sleep(1/2)

    xylo.play_poly(C5)
    await sleep(1/2)


Make Some Space
---------------

Jupylet let's you apply a varaiety of effects to a sound instance or to 
the entire audio stream.

For example let's add a distortion effect to the tb303 synthesizer:

.. code-block:: python

    tb303.set_effects(Overdrive(gain=4))

    tb303.play_poly(C3)
    tb303.play_poly(E3)
    tb303.play_poly(G3)
    await sleep(4)

    tb303.play_release()
    tb303.set_effects()

Another kind of effect is the `convolution reverb <https://en.wikipedia.org/wiki/Convolution_reverb>`_ 
which applies a recorded impulse response to a sound instance or to the 
entire audio stream. Impulse responses are recorded by specialists and capture 
the sonic signature of a room or any other kind of physical space.

Jupylet includes `three impulses responses <https://github.com/nir/jupylet/tree/master/jupylet/assets/sounds/impulses>`_ 
recorded by `Damian Murphy <https://www.openairlib.net/>`_ and you can find 
many more impulse responses in his website and elsewhere.

I like Damian's `Maes Howe <https://www.openair.hosted.york.ac.uk/?page_id=602>`_ 
impulse response in particular. It adds a wonderful sense of space and realism 
to the produced sound.

Let's apply it to the entire audio stream, this time; make sure to try it with 
a good pair of headphones:

.. code-block:: python

    set_effects(ConvolutionReverb('sounds/impulses/MaesHowe.flac'))

    for i in range(3):
        
        tb303.play_poly(C, duration=1)
        await sleep(1)

        tb303.play_poly(E, duration=1)
        await sleep(1)

        tb303.play_poly(G, duration=1)
        await sleep(1)


Sonic Py(thon)
--------------


MIDI
----


The Synthesis Playground
------------------------

