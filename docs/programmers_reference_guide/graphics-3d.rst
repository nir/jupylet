PROGRAMMING 3D GRAPHICS
=======================

Python in Space
---------------

Programming 2D games is awesome, but if you are anything like me what really
blows your mind is 3D graphics; and yes, you can do it in Python!

Nevertheless, programming 3D games can be more demanding than programming 2D 
games. It involves using more complex tools, new concepts like lights and 
cameras, and some understanding of algebra in 3D space.

To help you get started I have created a 3D version of the spaceship game. 
You can find the notebook at `examples/12-spaceship-3d.ipynb <https://github.com/nir/jupylet/blob/master/examples/12-spaceship-3d.ipynb>`_.
Run it and open the magical portal to 3D space: 

.. image:: ../images/spaceship_3d.gif


Blending in 
-----------

In 2D game programming you may use a simple image editor to create game assets. 
The 3D parallel of the 2D image editor is a 3D modelling program. 
3D modelling programs are complex tools and it takes time to master them. The 
great news is that one of the best tools around is completely free.

`Blender <https://www.blender.org/>`_ is a free open source and totally awesome 
3D creation suite, and Jupylet is designed to load assets created with it.

To get started with Blender I recommend the wonderful youtube videos by 
`Grant Abbitt <https://www.youtube.com/c/GrantAbbitt/playlists>`_:

* `Complete Beginners Guide to Blender 2.8 <https://www.youtube.com/watch?v=7MRonzqYJgw&list=PLn3ukorJv4vs_eSJUQPxBRaDS8PrVmIri>`_
* `Beginner Exercises <https://www.youtube.com/watch?v=98FkRIbihyQ&list=PLn3ukorJv4vvv3ZpWJYvV5Tmvo7ISO-NN>`_
* `Sculpting <https://www.youtube.com/watch?v=lKY2FIy60nc&list=PLn3ukorJv4vvJM7tvjet4PP-LVjJx13oB>`_
* `Unwrapping & Placing 2d Textures <https://www.youtube.com/watch?v=bHLT5Xh_tzQ&list=PLn3ukorJv4vve0s-cq8VWS4jRQCdWSU3N>`_

You may recognize the lego bricks in the lego notebook at `examples/13-lego-3d.ipynb <https://github.com/nir/jupylet/blob/master/examples/13-lego-3d.ipynb>`_. 
They were created by following Abbitt's beginner exercises.

**How to export a Blender scene:**

Jupylet can load Blender scenes exported using the glTF 2.0 format. To properly 
export a scene in this format follow these steps:

* Apply scale to all scene objects - to apply scale to an object select it and 
  press ``CTRL+A``.
* Choose export scene as glTF 2.0 format from the `File` menu.
* In the `Format` option select `glTF Separate mode`.
* Select the option to include the `Cameras and Punctual Lights`.
* Select the transform option named `+Y Up`. 
* Select the option to `Apply Modifiers`
* Select the options to include `UVs`, `Normals`, and `Materials`.
* Set the file name and export.

The export should create a `.gltf` file, a `.bin` file, and image files for 
all the textures in the scene.

.. note::
    * At the moment Jupylet does not load animations. 
    * Jupylet can load directional lights, spot lights, and point lights. 
      However, at the moment point lights will only cast shadows in a 90 
      degrees cone. To properly set the cone's direction, temporarily change 
      the point light to spot light, set its direction and then switch it 
      back to point light.


Lights, Camera, Action!
-----------------------

Modern API for 3D graphics are very flexible and may possibly accomodate any  
visual idea you may have regardless of how wild it may be.

However, when many people think of 3D graphics we often imagine being able to 
move around in an environment that is reminiscent of real life in some 
ways; most notably the role played by light and the way it interacts with 
objects in the environment, affecting their color and casting their shadow.

To make this possible, 3D game engines and 3D creation software such as Blender 
employ concepts we are familiar with from real life such as `scenes`, `lights`, 
`cameras`, and `materials`.

When you load an exported Blender scene into Jupylet, it is represented as a
collection of lights, cameras, meshes (i.e. scene objects), and materials.

You can access, introspect, and manipulate these objects, to bring the scene
to life. Let's see how it is done in `examples/12-spaceship-3d.ipynb <https://github.com/nir/jupylet/blob/master/examples/12-spaceship-3d.ipynb>`_.

We start by loading the exported Blender scene:

.. code-block:: python

    from jupylet.loader import load_blender_gltf

    scene = load_blender_gltf('./scenes/moon/alien-moon.gltf')

Shadows are turned off by default. You can turn them on with:

.. code-block:: python

    scene.shadows = True

If you just want to draw the scene, simply call the ``scene.draw()`` 
method in the ``render()`` function. That's it:

.. code-block:: python

    @app.event
    def render(ct, dt):
            
        app.window.clear()
        scene.draw()

The best way to get a grasp on these concepts is to play around with the 
various objects in the scene. Let's modify the camera's `field of view`:

.. code-block:: python

    camera = scene.cameras['Camera']

    camera.yfov = 0.4

If the game was already running, you should see the camera zoomed in. If you
increase the field of view the camera would appear to zoom out.

.. note::
    In Jupyter you can manipulate the properties of objects while the game is
    running and see the effect immediately and interactively.

Let's turn the color of the sun into bright red:

.. code-block:: python

    sun = scene.lights['Light.Sun']

    sun.intensity = 16
    sun.color = 'red'

Let's make the moon twice as big:

.. code-block:: python

    moon = scene.meshes['Moon']

    moon.scale *= 2

Take a few minutes to play around with the objects of the scene and you will 
soon get the idea. After all it's not rocket science.

.. note::
    In Jupyter you can find out the various method and properties of an object
    with the auto `complete function`. e.g. type ``moon.`` (don't forget the 
    dot) and then tap the :guilabel:`Tab` key.


A Little Bit of Math
--------------------

To move objects around and rotate them in 3D space we need to understand 
vectors in space. 



.. image:: ../images/coordinate_systems_right_handed.png 



The Sky in a Box
----------------


Diving into OpenGL
------------------

Over the years many sophisticated algorithms have been developed to enable
computer graphics to reproduce the visual effects of the interaction of light 
with matter, and by default Jupylet employs some of these algorithms to 


