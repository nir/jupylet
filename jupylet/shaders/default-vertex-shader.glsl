#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;
layout (location = 3) in vec4 tangent;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 vert_position;
out vec3 frag_position;
out vec3 frag_normal;
out vec2 frag_uv;

uniform int has_tangents;
out float frag_tangent_handedness;
out vec3 frag_tangent;

struct Cubemap {
    
    int render_cubemap;
    int texture_exists;
    float intensity;
    samplerCube texture;
};

uniform Cubemap cubemap;

struct Camera { 

    vec3 position;
};  

uniform Camera camera;


void main()
{
    if (cubemap.texture_exists == 1 && cubemap.render_cubemap == 1) {
        
        mat4 model = mat4(1.0);
        model[3].xyz = camera.position;

        gl_Position = projection * view * model * vec4(position, 1.0);
        gl_Position = gl_Position.xyww;
    }
    else {
        gl_Position = projection * view * model * vec4(position, 1.0);
        //gl_Position = view * model * vec4(position, 1.0);
    }
    
    vert_position = position;
    frag_position = vec3(model * vec4(position, 1.0));
    frag_uv = vec2(uv.x, 1.0 - uv.y);

    mat3 timodel = mat3(transpose(inverse(model)));
    
    frag_normal = timodel * normal;

    if (has_tangents == -1) {
        frag_tangent = timodel * tangent.xyz;
        frag_tangent_handedness = tangent.w;
    }
}

