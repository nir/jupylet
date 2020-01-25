
from distutils.core import setup

setup(
  name = 'jupylet',
  packages = ['jupylet'],
  version = '0.6.0',
  license='bsd-2-clause',
  description = 'A marriage of Jupyter and the pyglet game programming library',
  author = 'Nir Aides',
  author_email = 'nir@winpdb.org',
  url = 'https://github.com/nir/jupylet',
  download_url = 'https://github.com/nir/jupylet/archive/v0.6.0.tar.gz',
  keywords = ['python', 'pyglet', 'jupyter', 'kids', 'games', 'children', 'deep learning', 'reinforcement learning', 'RL'],
  install_requires=[
          'pyglet', 'webcolors', 'numpy', 'scipy', 'pillow', 'jupyter', 'ipyevents', 'ipycanvas'
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Education',
    'Intended Audience :: Developers', 
    'Intended Audience :: Science/Research',
    'Topic :: Education',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)