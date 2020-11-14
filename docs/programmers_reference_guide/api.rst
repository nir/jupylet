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

.. py:attribute:: Sprite.flip=True

    Flip the image upside down while rendering.

    :type: bool

.. py:attribute:: Sprite.mipmap=True

    Compute mipmap textures when first loading the sprite image.

    :type: bool

.. py:attribute:: Sprite.autocrop=False

    Auto crop the image to its bounding box when first loading the sprite image.

    :type: bool
    
.. py:attribute:: Sprite.anisotropy=8.0

    Use anisotropic filtering when rendering the sprite image.

    :type: float
    

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

    :type: str

.. py:attribute:: Label.font_path

    Path to a true type or open type font.

    :type: str
    
.. py:attribute:: Label.font_size=16

    Font size to use. 

    :type: float
    
.. py:attribute:: Label.line_height=1.2

    Determines the distance between lines.

    :type: float
    
.. py:attribute:: Label.align='left'

    The desired alignment for the text label. May be one of 'left', 'center', 
    and 'right'.

    :type: str
    

Module jupylet.loader
---------------------

.. py:currentmodule:: jupylet.loader
.. py:module:: jupylet.loader

.. autofunction:: load_blender_gltf


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

    A list of meshes.

    :type: list

.. py:attribute:: Scene.lights

    A list of lights.

    :type: list
    
.. py:attribute:: Scene.cameras

    A list of cameras.

    :type: list
    
.. py:attribute:: Scene.materials

    A list of materials.

    :type: list
    
.. py:attribute:: Scene.skybox

    A Skybox object.

    :type: Skybox
    
.. py:attribute:: Scene.shadows

    Set to True to enable shadows.

    :type: bool
    
.. py:attribute:: Scene.name

    Name of scene.

    :type: str
 

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

.. autoattribute:: Mesh.front
.. autoattribute:: Mesh.up

.. py:attribute:: Mesh.primitives

    List of primitives the mesh consists of.

    :type: list 

.. py:attribute:: Mesh.children

    List of child meshes.

    :type: list 
    
.. py:attribute:: Mesh.hide

    Set to False to hide mesh from view.

    :type: bool 
    
.. py:attribute:: Mesh.name

    Name of mesh.

    :type: str 
    

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

.. py:attribute:: Sound.freq

    Fundamental requency of sound object.

    :type: float

.. py:attribute:: Sound.amp

    Output amplitude - a value between 0 and 1.

    :type: float

.. py:attribute:: Sound.pan

    Balance between left (-1) and right (1) output channels.

    :type: float


Class Oscillator
^^^^^^^^^^^^^^^^

.. py:currentmodule:: jupylet.audio.sound
.. autoclass:: Oscillator


Methods
"""""""

.. automethod:: Oscillator.forward


Properties
""""""""""    

.. py:attribute:: Oscillator.shape

    Waveform to generate - one of `sine`, `triangle`, `sawtooth`, or `square`.

    :type: str
    
.. py:attribute:: Oscillator.sign

    Set to -1 to flip sawtooth waveform upside down.

    :type: float

.. py:attribute:: Oscillator.duty

    The fraction of the square waveform cycle its value is 1.

    :type: float


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


Properties
""""""""""    

.. py:attribute:: Sample.path

    Path to audio file.

    :type: str
    
