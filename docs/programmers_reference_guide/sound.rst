PROGRAMMING SOUND AND MUSIC
===========================

Da-Da-Da-DUM!
-------------

.. warning::
    Loud noise may damage your hearing and your speakers! Make sure to turn 
    your computer volume down to a safe level so you don't end up like 
    Beethoven.

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

Then create a :class:`~jupylet.audio.sample.Sample` instance and load it into 
memory. Here is how its done in the `examples/21-pong.ipynb <https://github.com/nir/jupylet/blob/master/examples/21-pong.ipynb>`_
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

The :meth:`~jupylet.audio.sound.GatedSound.play()` method accepts an `amp` 
parameter between 0 and 1 that controls output volume, and a `pan` parameter 
between -1 and 1 that controls the left-right balance. So to play the sound 
effect half as loud and just on the left side try:

.. code-block:: python
    
    pong_sound.play(amp=0.5, pan=-1)

By default, Jupylet sounds are `monophonic <https://en.wikipedia.org/wiki/Polyphony_and_monophony_in_instruments#Monophonic>`_, 
which means that if you call the :meth:`~jupylet.audio.sound.GatedSound.play()` 
method twice in quick succession you will not hear two instances of the sample 
being played simultaneously. Instead Jupylet will stop the first sound before 
it starts playing the second.

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
the `note value` with :func:`~jupylet.audio.set_bpm` and 
:func:`~jupylet.audio.set_note_value` respectively.

Normally if you play a new note the previous note will cease as if the 
synthesizer is monophonic. If you would like to play multiple notes together 
call the :meth:`~jupylet.audio.sound.GatedSound.play_poly` method instead. It 
will start playing a new note and return a reference to the new sound instance 
so you may control it as it plays:

.. code-block:: python

    c = tb303.play_poly(C5)
    f = tb303.play_poly(F5)

    await sleep(1)
    c.play_release()

    await sleep(1)
    f.play_release()

Playing the tb303 without specifying a duration will generate a note that goes
on indefinitely, like pressing a keyboard key without releasing it. The code 
above calls :meth:`~jupylet.audio.sound.GatedSound.play_release` to release the 
notes individually. You can also release all the currently playing sounds of a 
synthesizer like this:

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

The piano notebook `examples/14-piano.ipynb <https://github.com/nir/jupylet/blob/master/examples/14-piano.ipynb>`_ 
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
impulse response in particular. It adds a nice sense of space and a touch of 
realism to the generated sound.

Let's apply it to the entire audio stream intermittently so you may notice 
the effect; and make sure to try it with a good pair of headphones:

.. code-block:: python

    for i in range(5):
        
        if i % 2:
            print('Reverb on')
            set_effects(ConvolutionReverb('sounds/impulses/MaesHowe.flac'))
        else:
            print('Reverb off')
            set_effects()

        tb303.play_poly(C, 1)
        await sleep(1)

        tb303.play_poly(E, 1)
        await sleep(1)

        tb303.play_poly(G, 1)
        await sleep(1)


Sonic Py(thon)
--------------

You may have noticed how the examples above became progressively more 
elaborate, starting with playing a single note, then multiple notes at the 
same time, then a sequence of notes, and finally a sequence of notes in a 
loop.

As the code becomes more elaborate we can do more interesting stuff but we
also have a new problem.

When we play a single note the Jupyter notebook appears to remain responsive.
This allows us for example to type in an instruction to start a second note or 
to release the first note.

However if you run the loop above you may notice that while you can type in 
a new instruction in the next notebook cell, it will not be run until the 
loop is done. In other words, in some sense the notebook becomes unresponsive.

We have already seen a similar problem when we programmed the alien drifting
animation in the :any:`previous chapter<graphics-3d>` and we solved it there
by setting up a schedulled handler.

A similar construct can help us here as well. It is called the live loop and
it is a central concept in Sam Aaron's totally awesome code-based music 
creation and performance tool `Sonic Pi <https://sonic-pi.net/>`_.

It turns out a Jupyter notebook is the perfect environment for Python based 
music live coding and live loops.

To program live loops we first need to create an `app` instance like this:

.. code-block:: python

    app = sonic_py()

Now let's rewrite the code above as a live loop:

.. code-block:: python

    @app.sonic_live_loop(times=5)
    async def loop0(ncall):

        if ncall % 2:
            print('Reverb on')
            set_effects(ConvolutionReverb('sounds/impulses/MaesHowe.flac'))
        else:
            print('Reverb off')
            set_effects()
        
        tb303.play_poly(C, 1)
        await sleep(1)

        tb303.play_poly(E, 1)
        await sleep(1)

        tb303.play_poly(G, 1)
        await sleep(1)

The function name `loop0` is arbitrary. You can name the function anything you 
want. The `times` parameter is optional. Without it the loop will continue 
indefinitely. To stop the loop at any time call:

.. code-block:: python

    app.stop(loop0)

The `ncall` parameter is also optional. A simpler live loop would look like 
this:

.. code-block:: python

    @app.sonic_live_loop
    async def loop0():

        tb303.play_poly(C, 1)
        await sleep(1)

        tb303.play_poly(E, 1)
        await sleep(1)

        tb303.play_poly(G, 1)
        await sleep(1)

There is another problem that we need to take care of. When you call 
:meth:`~jupylet.audio.sound.GatedSound.play_poly` the new note is scheduled to 
play as soon as possible. The problem with that is that minor mistimings in 
"wakeups" from :func:`~jupylet.audio.sleep` calls are normal in desktop 
operating systems and may result in noticeable playing out of tempo. 

The correct way to play notes with accurate tempo in a live loop is the 
following:

.. code-block:: python

    @app.sonic_live_loop
    async def loop0():

        use(tb303)

        play(C3, 1)
        await sleep(1)

        play(E3, 1)
        await sleep(1)

        play(G3, 1)
        await sleep(1)

You can play multiple loops simultaneously. Let's add another voice:

.. code-block:: python

    @app.sonic_live_loop
    async def loop1():

        use(hammond)

        play(E, 1)
        await sleep(1)

        play(C, 2)
        await sleep(2)

        play(G, 1)
        await sleep(1)

        play(C, 2)
        await sleep(2)
        
        play(B, 2-1/3)
        await sleep(2-1/3)

        play(G, 1/3)
        await sleep(1/3)

        play(F, 2/3)
        await sleep(2/3)

        play(G, 1/3)
        await sleep(1/3)

        play(F, 2/3)
        await sleep(2/3)

        await sleep(1/3)

        play(E, 2)
        await sleep(2)    

Select both Jupyter cells and run them together to start the two loops in sync.

You can modify the code of a live loop while it is playing, and when you run 
the Jupyter cell with the new code, the live loop will immediately restart 
and play the new code.

However, sometimes it is more desirable to wait for the currently running 
loop to complete its cycle. If you decorate a live loop with 
:func:`@app.sonic_live_loop2 <jupylet.app.App.sonic_live_loop2>` and run it, 
the new code will kick in only after the currently playing loop completes a 
cycle.


MIDI Keyboards
--------------

The `MIDI <https://en.wikipedia.org/wiki/MIDI>`_ (`Musical Instrument Digital Interface`) 
standard is a specification that makes it possible to connect digital musical 
instruments to your computer. 

If you have an electronic (piano) keyboard, chances are it has a MIDI port 
that you can connect to your computer with a MIDI to USB cable.

If you installed Jupylet with MIDI support you are good to go. If not, open a
miniconda console and type in:

.. code-block:: bash

    pip install jupylet[midi]
    
To enable midi in Jupylet you just need to choose a sound instance to use. 
Let's hook it up with the hammond synthesizer:

.. code-block:: python

    app.set_midi_sound(hammond)

Alternatively, if you want full control you can program your own MIDI handler 
like this:

.. code-block:: python

    @app.event 
    def midi_message(msg):

        ... do whatever you want here ...

That's all there is to it.

Well, almost. By default most computer audio systems incur a short delay 
(also called latency) between the time you insturct the computer to play a 
note to the time it is actually played.

Normally, for games and live loops this short delay is not noticeable, but 
you may find that it makes it difficult to play a MIDI keyboard.

To minimize audio latency you can try this command:

.. code-block:: python

    set_latency('minimal')

.. warning::
    Reducing audio latency may cause the computer audio system to emit 
    unpleasant stuttering sound. If this happens Jupylet will automatically 
    attenuate output volume. Nevertheless, make sure to turn your computer's 
    volume down to prevent damage to your speakers and ears!
    
Lowering audio latency may cause the computer audio system to emit unpleasant 
stuttering sounds if your computer is unable to keep up with the required 
computations. If this happens you may set latency back to its default 
value with:

.. code-block:: python

    set_latency('high')

Then, you may try to address the problem by switching your computer's power 
mode to `Best performance` or by eliminating CPU intensive sound computations. 
Once you do that you may try to set latency back to `minimal`.

To switch your computer's power mode to `Best performance` on Windows 10 
select the `Battery` icon on the taskbar and then drag the slider all the way 
to the right to `Best performance` mode as shown in the following figure:

.. image:: ../images/power-mode.png 

To reduce CPU load try removing sound effects or changing instruments.

