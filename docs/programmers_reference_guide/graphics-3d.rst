PROGRAMMING 3D GRAPHICS
=======================

Python in Space
---------------

Programming 2D games is awesome, but if you are anything like me what really
blows your mind is 3D graphics; and it may come as a surprise but you can do 
awesome 3D with Python.

Nevertheless, programming 3D games can be more demanding than programming 2D 
games. It involves using more complex tools, new concepts like lights and 
cameras, and some understanding of algebra in 3D space.

To help you get started I have created a 3D version of the spaceship game. 
You can find the notebook at `examples/12-spaceship-3d.ipynb <https://github.com/nir/jupylet/blob/master/examples/12-spaceship-3d.ipynb>`_.
Run it to open the magical portal to 3D space: 

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
* Set filename and export.

The export should create a `.gltf` file, a `.bin` file, and image files for 
all the textures in the scene.

**Current limitations:**

* At the moment Jupylet does not load animations. 
* Jupylet can load directional, spot, and point lights. However, at the moment 
  point lights will only cast shadows in a 90 degrees cone. To properly set 
  the cone's direction, temporarily change the point light to spot light, set
  its direction and then switch it back to point light.

.. note::
    A blender scene is a very complex data structure, and you may occasionally 
    encounter all kinds of gotchas that you will need to take care of in order
    to properly export it. e.g. recalculating normals, applying modifiers, 
    correctly setting object anchor points, etc... Don't be discouraged for 
    your patience will be well rewarded. 


Lights, Camera, Action!
-----------------------


A Little Bit of Math
--------------------


Diving into OpenGL
------------------

