

import setuptools


with open('README.md', 'rb') as f:
    long_description = f.read().decode()


setuptools.setup(
    name = 'jupylet',
    packages = ['jupylet'],
    version = '0.8.0',
    license='bsd-2-clause',
    description = 'A marriage of Jupyter and the pyglet game programming library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author = 'Nir Aides',
    author_email = 'nir@winpdb.org',
    url = 'https://github.com/nir/jupylet',
    download_url = 'https://github.com/nir/jupylet/archive/v0.8.0.tar.gz',
    keywords = [
        'python', 
        'pyglet', 
        'jupyter', 
        'kids', 
        'games', 
        'children', 
        'deep learning', 
        'reinforcement learning', 
        'RL',
    ],
    install_requires=[
        'loky',
        'wget',
        'numpy', 
        'PyGLM',
        'scipy', 
        'pillow', 
        'gltflib',
        'jupyter',
        'moderngl',
        'requests',
        'soundfile',
        'webcolors', 
        'ipyevents', 
        'ipywidgets', 
        'matplotlib', 
        'sounddevice', 
        'xvfbwrapper',
        'scikit-image',
        'moderngl-window',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Developers', 
        'Intended Audience :: Science/Research',
        'Topic :: Education',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)

