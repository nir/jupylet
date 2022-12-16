"""
    jupylet/lru.py
    
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


import logging


logger = logging.getLogger(__name__)


SPRITE_TEXTURE_UNIT = 0
SKYBOX_TEXTURE_UNIT = 1
SHADOW_TEXTURE_UNIT = 2
TARRAY_TEXTURE_UNIT = 3


# TODO: an lru is actually not an ideal policy for allocating texture units.
# fix it.

class LRU(object):
    """Mechanism to allocate least recently used slot in array."""

    def __init__(self, min_items, max_items):
        
        self.mini = min_items
        self.step = max_items
        self.items = {i: [i, i, i, 0] for i in range(min_items, max_items)}
        
    def reset(self, min_items, max_items):

        self.mini = min_items
        self.step = max_items
        self.items = {i: [i, i, i, 0] for i in range(min_items, max_items)}

    def allocate(self, lid=None):
        """Allocate slot.
        
        Args:
            lid (int): An id that identifies "object" in array. A new id will 
                be generated if None is given.

        Returns:
            tuple: A 4-tuple consisting of (step, lid, slot, new) where 
                *step* indicates the lru "timestamp" for this object, 
                *lid* is the allocated object id, 
                *slot* is the array index allocated for the "object", and 
                *new* is 1 if "object" was allocated a new slot or 0 if it 
                remains in the same slot it was before.
        """
        self.step += 1
        
        if lid is None:
            lid = self.step
            
        r = self.items.get(lid)
        
        if r is None:
            
            lid0, slot = min(self.items.values())[1:3]
            self.items.pop(lid0)
            self.items[lid] = [self.step, lid, slot, 0]
            
            logger.debug('Allocated a new LRU slot with step=%r, lid=%r, slot=%r, 1.', self.step, lid, slot)

            return self.step, lid, slot, 1
            
        r[0] = self.step

        return r


_MAX_MATERIALS = 12
_lru_materials = LRU(0, _MAX_MATERIALS)

