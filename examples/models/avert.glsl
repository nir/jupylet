#version 330 core

layout (location = 0) in vec2 tex_coord;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec2 tex_coord0;

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
    //gl_Position = vec4(position, 1.0) * model * view * projection;
    tex_coord0 = tex_coord;
}
