PROGRAMMING GRAPHICS
====================

Hello, Jupylet!
---------------

We begin with the simplest *Jupylet* app. It displays a scolling banner with 
the text *"hello, world"*. You can find the notebook at
`examples/02-hello-jupylet.ipynb <https://github.com/nir/jupylet/blob/master/examples/02-hello-jupylet.ipynb>`_.

.. note::
    To understand this code you need to know about Python imports, functions, 
    and classes.

The code begins with two import statements that import the `App` class which
represents a game and the `Label` class which will be used to display the 
text:

.. code-block:: python

    from jupylet.label import Label
    from jupylet.app import App

Next, we create the game object and specify the width and height of the
game canvas, and then we create the label:

.. code-block:: python

    app = App(width=320, height=64)
    hello = Label('hello, world', color='cyan', font_size=32, x=app.width, y=16)

The *x* and *y* coordinates of a label correspond to its lower
left corner. By setting the initial *x* position to *app.width* we
effectively position the label just outside the right hand side of the
game canvas.

The label color can be any name defined by the `W3C SVG standard <https://www.w3.org/TR/SVG11/types.html#ColorKeywords>`_
or it can be any RGB color of the form :code:`'#abcdef'` - see here `<https://www.color-hex.com/>`_.

Next, we define a function to scroll the label from right to left. The 
line :code:`@app.run_me_every(1/30)` above the function definition is called a 
decorator. Python decorators are kind of magical, and this particular one
makes sure the function will be called every 1/30 of a second, or 30 times
per second, once the game is run:

.. code-block:: python

    @app.run_me_every(1/30)
    def scroll(ct, dt):
        hello.x -= 1
        if hello.right < 0:
            hello.x = app.width

The two function arguments *ct* and *dt* will contain the current game time
and the time since the function was last called (delta time). We can use 
these arguments to do interesting things, but we can ignore them for now.

Note that the function above does not actually draw the label in its new
position. For that we need the *render()* function. The *render()* function is a 
special function responsible for drawing each frame. In this particular case 
it will clear the game canvas (paint it black) and draw the label in 
its new position:

.. code-block:: python

    @app.event
    def render(ct, dt):
        app.window.clear()
        hello.draw()

Finally we start the game by calling:

.. code-block:: python

    app.run()

If you run the notebook the game canvas should appear with the following 
animation:

.. image:: ../images/hello-world.gif


