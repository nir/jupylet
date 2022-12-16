"""
    jupylet/__init__.py
    
    Copyright (c) 2022, Nir Aides - nir.8bit@gmail.com

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import platform
import sys
import os
import re

from .env import is_remote, has_display, is_numpy_openblas


VERSION = '0.8.9'


if platform.system() == 'Linux' and not has_display():
    setattr(sys, 'is_pyglet_doc_run', True)
    

#
# Workaround segmentation fault when calling np.linalg.inv() in 
# mutlithreaded app.
#
if platform.system() == 'Darwin':
   if 'numpy' in sys.modules and is_numpy_openblas():
      sys.stderr.write(
         'WARNING: numpy was imported before jupylet. ' + 
         'On macOS you should import jupylet first to let it work around ' +
         'a bug in the algebra libraries used by numpy that may cause the ' +
         'program to exit.' + '\n'
      )

   os.environ['OPENBLAS_NUM_THREADS'] = '1'


#
# Work around problem in pip install jupyter in python 3.8 as described in:
# https://github.com/jupyter/notebook/issues/4980#issuecomment-600992296
#
if platform.system() == 'Windows' and sys.version_info >= (3, 8):
   if sys.argv[-2:] == ['-m', 'postinstall']:
      os.system('python %s\Scripts\pywin32_postinstall.py -install' % os.__file__.rsplit('\\', 2)[0])
      sys.exit(0)


def download_url(url, progress=False):

   if not progress:
      r = urllib.request.urlopen(GITHUB_MASTER_URL)
      return zipfile.ZipFile(io.BytesIO(r.read()))

   import tqdm

   pbar = tqdm.tqdm(unit='B', unit_scale=True, desc=url.split('/')[-1])

   def update(b=1, bsize=1, tsize=None):
      if tsize is not None:
         pbar.total = tsize
      pbar.update(b * bsize - pbar.n)

   path, headers = urllib.request.urlretrieve(url, reporthook=update)
   pbar.close()
   
   return zipfile.ZipFile(open(path, 'rb'))


def extract_master(zf, to='jupylet', noisy=False):
    
    for p0 in zf.namelist():
        p1 = re.sub(r'jupylet-master', to, p0)
        if p1[-1] == '/':
            os.makedirs(p1, exist_ok=True)
        else:
            noisy and print('%s -> %s' % (p0, p1))
            open(p1, 'wb').write(zf.read(p0))


if sys.argv[-2:] == ['-m', 'download']:

   import urllib.request
   import zipfile
   import io

   while os.path.exists('jupylet'):
      r = input('Target directory ./jupylet/ already exists. Would you like to overwrite it (y/n)? ')
      if r == 'n':
         sys.exit(0)
      if r == 'y':
         break

   GITHUB_MASTER_URL = 'https://github.com/nir/jupylet/archive/master.zip'

   sys.stderr.write('Downloading jupylet source code ZIP from %s...\n' % GITHUB_MASTER_URL)
   zf = download_url(GITHUB_MASTER_URL, progress=True)

   sys.stderr.write('Extracting source code from ZIP file to ./jupylet/.\n')
   extract_master(zf)

   sys.stderr.write('Type "cd ./jupyter/examples" to enter the examples directory.\n')

   sys.exit(0)

