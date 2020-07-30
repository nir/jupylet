
#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in vec2 in_texcoord_0;

uniform mat4 model;
uniform mat4 projection;

out vec2 frag_uv;

void main() {
    gl_Position = projection * model * vec4(in_position, 1.0);
    frag_uv = in_texcoord_0;
}

#elif defined FRAGMENT_SHADER

out vec4 fragColor;

in vec2 frag_uv;

struct Texture {
    sampler2D t;
};

#define MAX_TEXTURES 32

uniform Texture textures[MAX_TEXTURES];
uniform int texture_id;

void main() {
    fragColor = texture(textures[texture_id].t, frag_uv);
}

#endif
