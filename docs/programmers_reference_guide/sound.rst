PROGRAMMING SOUND AND MUSIC
===========================

Da-Da-Da-DUM!
-------------

*Jupylet* is full of sound. Here is a list of features:

* Play audio tracks and samples in WAV, FLAC, and OGG formats.
* Basic support of `SFZ format <https://sfzformat.com/>`_ for playing multi-sampled instruments.
* Simple to use MIDI support - just hook your piano keyboard and play.
* A Novel Framework for creating sound synthesizers and effects.
* Live loops and music coding in the spirit of `Sonic Pi <https://sonic-pi.net/>`_.

Note to Self
------------

Power up *Jupylet's* audio with the following import statement:

.. code-block:: python

    from jupylet.audio.synth import tb303
    from jupylet.audio.sound import note

And let's start by playing a simple note:

.. code-block:: python

    tb303.play(note.C4, duration=1)



