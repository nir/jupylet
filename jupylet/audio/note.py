"""
    jupylet/audio/note.py
    
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
       

_notes = dict(
    C = 1,
    Cs = 2, Db = 2,
    D = 3,
    Ds = 4, Eb = 4,
    E = 5,
    F = 6,
    Fs = 7, Gb = 7,
    G = 8, 
    Gs = 9, Ab = 9,
    A = 10,
    As = 11, Bb = 11,
    B = 12,
)


#
# Fill name space with note definitions from C1 to B7.
#

for o in range(8):
    for n, k in _notes.items():
        no = (n + str(o)).rstrip('0')
        globals()[no] = k + 11 + 12 * (o if o else 4)


def note2key(n):
    
    if n[-1].isdigit():
        octave = int(n[-1])
        n = n[:-1]
    else:
        octave = 4
        
    n = n.replace('#', 's')
    return _notes[n] + octave * 12 + 11


def key2note(key):
    """Convert keyboard key to note. e.g. 60 to 'C4'.

    Args:
        key (float): keyboard key to convert.

    Returns:
        str: A string representing the note. In the conversion process
            the floating point key will be rounded in a special way that 
            preserves the nearest note to the key. e.g. 60.9 and 61.1 
            will converted to Cs4, Db4 respectively.
    """ 
    i = (key - 11) % 12 

    octave = (round(key) - 12) // 12

    n0, i0 = 'B', 0

    for n1, i1 in _notes.items():
        if i <= i1:
            break

        n0, i0 = n1, i1
    
    _n = n0 if i1 - i > 0.5 else n1
    
    return _n + str(int(octave))

