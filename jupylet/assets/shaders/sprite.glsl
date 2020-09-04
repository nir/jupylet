
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

uniform sampler2D texture0;

uniform vec4 color;

uniform int components;
uniform int flip;

void main() {

    vec2 uv = frag_uv;

    if (flip == 1) {
        uv.y = 1.0 - frag_uv.y;
    }
    
    fragColor = color;

    if (components == 4) {
        fragColor *= texture(texture0, uv);
    }
    else if (components == 1) {
        fragColor.a *= texture(texture0, uv).x;        
    }
    else {
        fragColor.rgb *= texture(texture0, uv).rgb;
    }
}

#endif
