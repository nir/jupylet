#version 330 core

out vec4 FragColor;

in vec2 tex_coord0;

uniform vec4 diffuse;
uniform int has_diffuse_texture;
uniform sampler2D diffuse_texture;

void main()
{
    if (has_diffuse_texture != 0) {
        FragColor = texture(diffuse_texture, tex_coord0);
    } else {
        FragColor = diffuse;
    }
} 
