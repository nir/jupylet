#version 330 core

in vec3 in_position;
in vec3 in_normal;
in vec2 in_texcoord_0;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec4 frag_view;
out vec3 vert_position;
out vec3 frag_position;
out vec3 frag_normal;
out vec2 frag_uv;

struct Skybox {
    
    int render_skybox;
    int texture_exists;
    float intensity;
    samplerCube texture;
};

uniform Skybox skybox;

struct Camera { 

    vec3 position;
    float zfar;
};  

uniform Camera camera;

#define DIRECTIONAL_LIGHT 0
#define POINT_LIGHT 1
#define SPOT_LIGHT 2

#define MAX_CASCADES 4

struct ShadowmapTexture {
    int layer;
    float depth;
    mat4 projection;
};

struct Light { 

    int type;

    vec3 position;
    vec3 direction;   

    vec3 color;
    float intensity;
    float ambient;

    float inner_cone;
    float outer_cone;

    float scale;
    float snear;
 
    int shadows;

    ShadowmapTexture shadowmap_textures[MAX_CASCADES];
    
    int shadowmap_textures_count;

    int shadowmap_pcf;
    float shadowmap_bias;
    mat4 shadowmap_projection;
};  

#define MAX_LIGHTS 12

uniform Light lights[MAX_LIGHTS];
uniform int nlights;

uniform int shadowmap_pass;
uniform int shadowmap_light;


void main()
{
    vec4 mp4 = model * vec4(in_position, 1.0);

    if (shadowmap_pass == 1) {
        
        int li = shadowmap_light;
        
        gl_Position = lights[li].shadowmap_projection * mp4;

        vec3 light_direction;

        if (lights[li].type == DIRECTIONAL_LIGHT) {
            light_direction = normalize(lights[li].direction);
        } 
        else {
            frag_position = vec3(mp4);
            light_direction = normalize(lights[li].position - frag_position);
        }

        frag_normal = normalize(mat3(transpose(inverse(model))) * in_normal);

        float nl = dot(frag_normal, light_direction);
        if (nl < 0.001) {
            return;
        }

        float bias = lights[li].shadowmap_bias / nl;

        if (lights[li].type == DIRECTIONAL_LIGHT) {
            gl_Position.z += bias * lights[li].scale / 100;
            return;
        }

        float d0 = length(frag_position - lights[li].position);

        bias *= 2 * lights[li].snear / d0;

        gl_Position.z += bias * gl_Position.w;
        
        return;
    }

    frag_view = view * mp4;

    if (skybox.texture_exists == 1 && skybox.render_skybox == 1) {
        
        mat4 model = mat4(1.0);
        model[3].xyz = camera.position;

        gl_Position = projection * view * model * vec4(in_position, 1.0);
        gl_Position = gl_Position.xyww;

        vert_position = in_position;
    }
    else {
        gl_Position = projection * frag_view;
    }
    
    frag_position = vec3(mp4);
    frag_normal = mat3(transpose(inverse(model))) * in_normal;
    frag_uv = vec2(in_texcoord_0.x, 1.0 - in_texcoord_0.y);
}

