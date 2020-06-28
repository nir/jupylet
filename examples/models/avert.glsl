#version 330 core

layout (location = 0) in vec2 uv;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 frag_position;
out vec3 frag_normal;
out vec2 frag_uv;

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
    
    frag_position = vec3(model * vec4(position, 1.0));
    frag_normal = mat3(transpose(inverse(model))) * normal;
    frag_uv = uv;
}
