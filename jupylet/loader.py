"""
    jupylet/loader.py
    
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


import gltflib
import logging
import struct
import math
import glm
import os

import numpy as np

from .utils import abspath
from .resource import load_texture, load_image, find_path, unresolve_path
from .model import Scene, Material, Light, Camera, Mesh, Primitive


__all__ = ['load_blender_gltf']


def load_blender_gltf(path):
    """Load a Blender scene or model exported with the Blender glTF 2.0 exporter.

    The Blender scene should be exported using the glTF Separate (.gltf + .bin + textures)
    format. In the exporter make sure to check the Apply Modifiers, UVs, and 
    Normals, options checkboxes.

    Note that in principle this function should be able to load any gltf 2.0 scene,
    however it was only tested with and tuned to load scenes exported with 
    Blender 2.83 and Blender 3.4.

    Args:
        path (str): path to glTF 2.0 file.

    Returns:
        Scene: A scene object.
    """

    pp = find_path(path)
    g0 = gltflib.GLTF.load(str(pp))
    s0 = g0.model.scenes[g0.model.scene]
    
    scene = Scene(s0.name)
    scene._source = g0
    
    for m0 in g0.model.materials:
        material = _load_blender_gltf_material(g0, m0)
        scene.add_material(material)
    
    for n0 in s0.nodes:
        n0 = g0.model.nodes[n0]

        if _is_blender_gltf_light(g0, n0):
            light = _load_blender_gltf_light(g0, n0)
            scene.add_light(light)
            
        elif _is_blender_gltf_camera(g0, n0):
            camera = _load_blender_gltf_camera(g0, n0)
            scene.add_camera(camera)
        
        elif _is_blender_gltf_mesh(g0, n0):
            mesh = _load_blender_gltf_mesh(g0, n0, list(scene.materials.values()))
            scene.add_mesh(mesh)
        
    return scene


def _load_blender_gltf_material(g0, m0):
    
    lbi = _load_blender_gltf_image
    pbr = m0.pbrMetallicRoughness
    
    c = lbi(g0, pbr.baseColorTexture) if pbr.baseColorTexture else pbr.baseColorFactor
    m = pbr.metallicFactor
    r = lbi(g0, pbr.metallicRoughnessTexture) if pbr.metallicRoughnessTexture else pbr.roughnessFactor
    s = 0.1
    e = lbi(g0, m0.emissiveTexture) if m0.emissiveTexture else m0.emissiveFactor
    o = lbi(g0, m0.occlusionTexture)
    n = lbi(g0, m0.normalTexture)
    
    ns = getattr(m0.normalTexture, 'scale', 1.)

    material = Material(m0.name, c, m, r, s, e, o, n, ns)
    material._source = m0
    
    return material


def _load_blender_gltf_texture(g0, ti):
    
    if ti is not None:
        
        t0 = g0.model.textures[getattr(ti, 'index', ti)]
        i0 = g0.model.images[t0.source]
        r0 = [r for r in g0.resources if r._uri == i0.uri][0]
            
        dirname = unresolve_path(r0._basepath)

        return load_texture(os.path.join(dirname, r0.filename), flip=True)


def _load_blender_gltf_image(g0, ti):
    
    if ti is not None:
        
        t0 = g0.model.textures[getattr(ti, 'index', ti)]
        i0 = g0.model.images[t0.source]
        r0 = [r for r in g0.resources if r._uri == i0.uri][0]
            
        dirname = unresolve_path(r0._basepath)

        return load_image(os.path.join(dirname, r0.filename), flip=True)


def _is_blender_gltf_light(g0, n0):

    if getattr(n0, 'extensions') and n0.extensions.get('KHR_lights_punctual') is not None:
        return True
    
    if not n0.children:
        return False
        
    for c0 in n0.children:
        
        nc = g0.model.nodes[c0]
        if _is_blender_gltf_light(g0, nc):
            return True
        
    return False

    
def _load_blender_gltf_light(g0, n0):
    
    if n0.children:
        
        nc = g0.model.nodes[n0.children[0]]
        
        l0 = nc.extensions.get('KHR_lights_punctual')['light']
        l1 = g0.model.extensions['KHR_lights_punctual']['lights'][l0]
        
        m0 = glm.mat4_cast(glm.quat(xyzw2wxyz(nc.rotation)))
        m1 = glm.mat4_cast(glm.quat(xyzw2wxyz(n0.rotation))) if n0.rotation else glm.mat4(1.0) 
        qq = glm.quat_cast(m1 * m0)
    
    else:
        
        l0 = n0.extensions.get('KHR_lights_punctual')['light']
        l1 = g0.model.extensions['KHR_lights_punctual']['lights'][l0]
        
        m1 = glm.mat4_cast(glm.quat(xyzw2wxyz(n0.rotation))) if n0.rotation else glm.mat4(1.0) 
        qq = glm.quat_cast(m1)
 
    attenuation = {
        'spot': 1 / 10,
        'point': 1 / 10,
        'directional': 5 / 4,
    }

    ambient = 0.001
    
    intensity = l1['intensity'] * attenuation.get(l1['type'], 1.)

    if g0.model.asset.generator >= 'Khronos glTF Blender I/O v3':

        if l1['type'] == 'directional':
            intensity /= 683
        else:
            intensity /= 54.35
        
    light = Light(
        n0.name, 
        qq, 
        n0.scale, 
        n0.translation,
        l1['type'],
        l1['color'],
        intensity,
        ambient,
        l1.get('spot', {}).get('outerConeAngle', math.pi / 4),
        l1.get('spot', {}).get('innerConeAngle', math.pi / 4 * 0.9),
    )
    light._source = n0
    
    return light


def _is_blender_gltf_camera(g0, n0):
    
    if getattr(n0, 'camera', None) is not None:
        return True
    
    if not n0.children:
        return False

    for c0 in n0.children:
        
        nc = g0.model.nodes[c0]
        if _is_blender_gltf_camera(g0, nc):
            return True
        
    return False


def xyzw2wxyz(q):
    
    if q:
        x, y, z, w = q
        return w, x, y, z


def _load_blender_gltf_camera(g0, n0):
    
    if n0.children:
        
        nc = g0.model.nodes[n0.children[0]]
        c0 = g0.model.cameras[nc.camera]

        m0 = glm.mat4_cast(glm.quat(xyzw2wxyz(nc.rotation)))
        m1 = glm.mat4_cast(glm.quat(xyzw2wxyz(n0.rotation)))  
        qq = glm.quat_cast(m1 * m0)
    
    else:
        
        c0 = g0.model.cameras[n0.camera]

        m1 = glm.mat4_cast(glm.quat(xyzw2wxyz(n0.rotation)))  
        qq = glm.quat_cast(m1)
        
    camera = Camera(
        n0.name, 
        qq, 
        n0.scale, 
        n0.translation,
        c0.type,
        **vars(c0.perspective or c0.orthographic)
    )
    camera._source = n0
    
    return camera


def _is_blender_gltf_mesh(g0, n0):
    
    if n0.mesh is not None:
        return True
    
    for nc in n0.children or []:
        if _is_blender_gltf_mesh(g0, g0.model.nodes[nc]):
            return True
        
    return False

    
def _load_blender_gltf_mesh(g0, n0, materials, parent=None):
    
    mesh = Mesh(
        n0.name, 
        xyzw2wxyz(n0.rotation), 
        n0.scale, 
        n0.translation,
        parent
    )
    mesh._source = n0
    
    m0 = g0.model.meshes[n0.mesh]
    
    for p0 in m0.primitives:
        mesh.add_primitive(_load_blender_gltf_primitive(g0, p0, materials))
    
    for cn in n0.children or []:
        n1 = g0.model.nodes[cn]
        mesh.children[n1.name] = _load_blender_gltf_mesh(g0, n1, materials, mesh)
        
    return mesh


def _load_blender_gltf_primitive(g0, p0, materials):
    
    primitive = Primitive(
        material=materials[p0.material], 
        indices=get_buffer0(g0, p0.indices)[0],
        vertices=get_buffer0(g0, p0.attributes.POSITION),
        normals=get_buffer0(g0, p0.attributes.NORMAL),
        coords=get_buffer0(g0, p0.attributes.TEXCOORD_0)
    )
    primitive._source = p0
    
    return primitive


def get_buffer0(g0, ai):    
    if ai is not None:
        return get_buffer(g0, g0.model.accessors[ai])  
    
    
def get_buffer(g0, a0):
    
    t2s = dict(
        BYTE = 'b',
        UNSIGNED_BYTE = 'B',
        SHORT = 'h',
        UNSIGNED_SHORT = 'H',
        UNSIGNED_INT = 'I',
        FLOAT = 'f',
    )
    
    t2f = dict(
        BYTE = 'i1',
        UNSIGNED_BYTE = 'u1',
        SHORT = 'i2',
        UNSIGNED_SHORT = 'u2',
        UNSIGNED_INT = 'u4',
        FLOAT = 'f4',
    )
    
    t2n = {
        'SCALAR': '1',
        'VEC2': '2',
        'VEC3': '3',
        'VEC4': '4',
    }
    
    ctn = gltflib.ComponentType._value2member_map_[a0.componentType].name
    fmt = t2n[a0.type] + t2f[ctn]
    
    v0 = g0.model.bufferViews[a0.bufferView]
    b0 = g0.model.buffers[v0.buffer]
    r0 = [r for r in g0.resources if r._uri == b0.uri][0]
    
    if not r0.loaded:
        r0.load()
        
    ao = a0.byteOffset or 0
    d0 = r0.data[ao + v0.byteOffset: ao + v0.byteOffset + v0.byteLength]
    data = np.frombuffer(d0, fmt[1:])   
    
    return data, fmt

