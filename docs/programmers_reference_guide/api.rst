API REFERENCE
=============


Class App
---------

.. py:currentmodule:: jupylet.app
.. autoclass:: App


Methods
~~~~~~~

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
~~~~~~~~~~

.. autoattribute:: App.width
.. autoattribute:: App.height


Class Sprite
------------

.. py:currentmodule:: jupylet.sprite
.. autoclass:: Sprite

Methods
~~~~~~~

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
~~~~~~~~~~    

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
.. py:attribute:: Sprite.mipmap
.. py:attribute:: Sprite.autocrop
.. py:attribute:: Sprite.anisotropy


Class Label
-----------

.. py:currentmodule:: jupylet.label
.. autoclass:: Label


Properties
~~~~~~~~~~    

.. py:attribute:: Label.text
.. py:attribute:: Label.font_path
.. py:attribute:: Label.font_size=16
.. py:attribute:: Label.line_height=1.2
.. py:attribute:: Label.align='left'

