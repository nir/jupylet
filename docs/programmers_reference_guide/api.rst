API REFERENCE
=============


Module jupylet.app
------------------

Class App
^^^^^^^^^

.. py:currentmodule:: jupylet.app
.. autoclass:: App


Methods
"""""""

.. automethod:: App.run
.. automethod:: App.stop
.. automethod:: App.set_midi_sound
.. automethod:: App.observe
.. automethod:: App.scale_window_to
.. automethod:: App.save_state
.. automethod:: App.load_state
.. automethod:: App.get_logging_widget
.. automethod:: App.sonic_live_loop2
.. automethod:: App.sonic_live_loop
.. automethod:: App.run_me_every
.. automethod:: App.run_me
.. automethod:: App.mouse_position_event
.. automethod:: App.mouse_press_event
.. automethod:: App.mouse_release_event
.. automethod:: App.key_event
.. automethod:: App.render
.. automethod:: App.event
.. automethod:: App.close
.. automethod:: App.load_program


Properties
""""""""""

.. autoattribute:: App.width
.. autoattribute:: App.height


Module jupylet.sprite
---------------------


Class Sprite
^^^^^^^^^^^^

.. py:currentmodule:: jupylet.sprite
.. autoclass:: Sprite

Methods
"""""""

.. automethod:: Sprite.render
.. automethod:: Sprite.draw
.. automethod:: Sprite.set_anchor
.. automethod:: Sprite.collisions_with
.. automethod:: Sprite.distance_to
.. automethod:: Sprite.angle_to
.. automethod:: Sprite.wrap_position
.. automethod:: Sprite.clip_position
.. automethod:: Sprite.get_state
.. automethod:: Sprite.set_state


Properties
""""""""""    

.. autoattribute:: Sprite.scale
.. autoattribute:: Sprite.x
.. autoattribute:: Sprite.y
.. autoattribute:: Sprite.angle
.. autoattribute:: Sprite.width
.. autoattribute:: Sprite.height
.. autoattribute:: Sprite.image
.. autoattribute:: Sprite.top
.. autoattribute:: Sprite.right
.. autoattribute:: Sprite.bottom
.. autoattribute:: Sprite.left
.. autoattribute:: Sprite.radius
.. autoattribute:: Sprite.opacity
.. autoattribute:: Sprite.color

.. py:attribute:: Sprite.flip

    Flip the image upside down while rendering.

.. py:attribute:: Sprite.mipmap

    Compute mipmap textures when first loading the sprite image.

.. py:attribute:: Sprite.autocrop

    Auto crop the image to its bounding box when first loading the sprite image.

.. py:attribute:: Sprite.anisotropy

    Use anisotropic filtering when rendering the sprite image.


Module jupylet.label
--------------------


Class Label
^^^^^^^^^^^

.. py:currentmodule:: jupylet.label
.. autoclass:: Label


Properties
""""""""""    

.. py:attribute:: Label.text

    Text to render as label.

.. py:attribute:: Label.font_path

    Path to a true type or open type font.

.. py:attribute:: Label.font_size=16

    Font size to use. 

.. py:attribute:: Label.line_height=1.2

    Determines the distance between lines.

.. py:attribute:: Label.align='left'

    The desired alignment for the text label. May be one of 'left', 'center', 
    and 'right'.


Module jupylet.model
--------------------


Class Scene
^^^^^^^^^^^

.. py:currentmodule:: jupylet.model
.. autoclass:: Scene


Methods
"""""""

.. automethod:: Scene.draw


Properties
""""""""""    

.. py:attribute:: Scene.meshes
.. py:attribute:: Scene.lights
.. py:attribute:: Scene.cameras
.. py:attribute:: Scene.materials


Class Mesh
^^^^^^^^^^

.. py:currentmodule:: jupylet.model
.. autoclass:: Mesh


Methods
"""""""

.. automethod:: Mesh.move_local
.. automethod:: Mesh.move_global
.. automethod:: Mesh.rotate_local
.. automethod:: Mesh.rotate_global


Properties
""""""""""    

.. autoattribute:: Mesh.draw
.. autoattribute:: Mesh.matrix
.. autoattribute:: Mesh.up
.. autoattribute:: Mesh.front


Module jupylet.audio
--------------------

.. py:currentmodule:: jupylet.audio
.. py:module:: jupylet.audio


.. autofunction:: sonic_py
.. autofunction:: set_bpm
.. autofunction:: set_note_value
.. autofunction:: use
.. autofunction:: play
.. autofunction:: sleep


Module jupylet.audio.sound
--------------------------


Class Sound
^^^^^^^^^^^

.. py:currentmodule:: jupylet.audio.sound
.. autoclass:: Sound


Methods
"""""""

.. automethod:: Sound.play
.. automethod:: Sound.play_poly
.. automethod:: Sound.play_release
.. automethod:: Sound.set_effects
.. automethod:: Sound.get_effects


Properties
""""""""""    

.. autoattribute:: Sound.note
.. autoattribute:: Sound.key

.. py:attribute:: Sound.velocity
.. py:attribute:: Sound.freq
.. py:attribute:: Sound.amp
.. py:attribute:: Sound.pan


Class Oscillator
^^^^^^^^^^^^^^^^

.. py:currentmodule:: jupylet.audio.sound
.. autoclass:: Oscillator


Methods
"""""""

.. automethod:: Oscillator.forward


Properties
""""""""""    

.. py:attribute:: Oscillator.freq
.. py:attribute:: Oscillator.shape
.. py:attribute:: Oscillator.sign
.. py:attribute:: Oscillator.duty


Class LatencyGate
^^^^^^^^^^^^^^^^^

.. py:currentmodule:: jupylet.audio.sound
.. autoclass:: LatencyGate


Methods
"""""""

.. automethod:: LatencyGate.open
.. automethod:: LatencyGate.close


Class GatedSound
^^^^^^^^^^^^^^^^

.. py:currentmodule:: jupylet.audio.sound
.. autoclass:: GatedSound


Methods
"""""""

.. automethod:: GatedSound.play
.. automethod:: GatedSound.play_poly
.. automethod:: GatedSound.play_release


Module jupylet.audio.sample
---------------------------


Class Sample
^^^^^^^^^^^^

.. py:currentmodule:: jupylet.audio.sample
.. autoclass:: Sample


Methods
"""""""

.. automethod:: Sample.load
.. automethod:: Sample.play
.. automethod:: Sample.play_poly
.. automethod:: Sample.play_release


Properties
""""""""""    

.. py:attribute:: Sample.path

