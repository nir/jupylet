PROGRAMMING GRAPHICS
====================

Hello, Jupylet!
---------------

We will get back to the spaceship example, but let's start with a more
traditional first program called `Hello, World! <https://en.wikipedia.org/wiki/%22Hello,_World!%22_program>`_.
You can find the notebook at `examples/hello-jupylet.ipynb <https://github.com/nir/jupylet/blob/unstable/examples/hello-jupylet.ipynb>`_.
Here is its code:

.. code-block:: python

    from jupylet.label import Label
    from jupylet.app import App

    app = App(width=320, height=64)
    hello = Label('hello, world', color='cyan', font_size=32, x=app.width, y=16)

    @app.event
    def on_draw():
        app.window.clear()
        hello.draw()

    @app.run_me_again_and_again(1/30)
    def scroll(dt):
        hello.x -= 1
        if hello.x < -220:
            hello.x = app.width

    app.run()

If you run the notebook a canvas will appear with the following animation:

.. image:: ../images/hello-world.gif


