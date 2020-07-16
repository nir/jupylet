#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 vert_position;
out vec3 frag_position;
out vec3 frag_normal;
out vec2 frag_uv;

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

#define DIRECTIONAL_LIGHT 0
#define POINT_LIGHT 1
#define SPOT_LIGHT 2

struct Light { 

    int type;

    vec3 position;
    vec3 direction;   

    vec3 color;
    float intensity;

    float inner_cone;
    float outer_cone;
 
    int shadows;
    int shadowmap_texture;
    int shadowmap_texture_size;
    mat4 shadowmap_projection;
};  

#define MAX_LIGHTS 16

uniform Light lights[MAX_LIGHTS];
uniform int nlights;

uniform int shadowmap_pass;
uniform int shadowmap_light;
out vec4 shadowmap_frag_position[MAX_LIGHTS];


void main()
{
    if (shadowmap_pass == 1) {
        mat4 projection = lights[shadowmap_light].shadowmap_projection;
        gl_Position =  projection * model * vec4(position, 1.0);
        return;
    }

    if (cubemap.texture_exists == 1 && cubemap.render_cubemap == 1) {
        
        mat4 model = mat4(1.0);
        model[3].xyz = camera.position;

        gl_Position = projection * view * model * vec4(position, 1.0);
        gl_Position = gl_Position.xyww;
    }
    else {
        gl_Position = projection * view * model * vec4(position, 1.0);
    }
    
    vert_position = position;
    frag_position = vec3(model * vec4(position, 1.0));
    frag_normal = mat3(transpose(inverse(model))) * normal;
    frag_uv = vec2(uv.x, 1.0 - uv.y);

    if (shadowmap_pass == 2) {
        for (int i = 0; i < nlights; i++) {
            if (lights[i].shadows == 1) {
                shadowmap_frag_position[i] = lights[i].shadowmap_projection * vec4(frag_position, 1.0);
            }
        }
    }
}

