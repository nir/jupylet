#version 330 core

out vec4 FragColor;

uniform mat4 model;

in vec3 vert_position;
in vec3 frag_position;
in vec3 frag_normal;
in vec2 frag_uv;

struct Cubemap {

    int render_cubemap;
    int texture_exists;
    float intensity;
    samplerCube texture;
};

uniform Cubemap cubemap;

// Python code to dynamically retreive max units.
// mt = ctypes.c_int()
// glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, mt)

struct Texture {
    sampler2D t;
};

#define MAX_TEXTURES 28

uniform Texture textures[MAX_TEXTURES];

struct Material { 

    vec4 color; // Expected as linear - physical.
    int color_texture;

    int normals_texture;
    float normals_gamma;
    float normals_scale;

    float specular;
    float metallic;

    float roughness;
    int roughness_texture;

    vec3 emissive;  
    int emissive_texture;
};  

#define MAX_MATERIALS 32

uniform Material materials[MAX_MATERIALS];
uniform int material;

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

    vec3 color; // Expected as linear - physical.
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
in vec4 shadowmap_frag_position[MAX_LIGHTS];

//
// https://gamedev.stackexchange.com/questions/68612/how-to-compute-tangent-and-bitangent-vectors
// https://www.gamasutra.com/blogs/RobertBasler/20131122/205462/Three_Normal_Mapping_Techniques_Explained_For_the_Mathematically_Uninclined.php
// http://www.thetenthplanet.de/archives/1180
//

mat3 cotangent_frame( vec3 N, vec3 p, vec2 uv ) { 
    
    // get edge vectors of the pixel triangle 
    vec3 dp1 = dFdx( p ); 
    vec3 dp2 = dFdy( p ); 
    vec2 duv1 = dFdx( uv ); 
    vec2 duv2 = dFdy( uv );   
    // solve the linear system 
    vec3 dp2perp = cross( dp2, N ); 
    vec3 dp1perp = cross( N, dp1 ); 
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x; 
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;   
    // construct a scale-invariant frame 
    float invmax = inversesqrt( max( dot(T,T), dot(B,B) ) ); 
    return mat3( T * invmax, B * invmax, N ); 
}


float pi = 3.141592653589793;
float eps = 0.000001;


float dggx(float nh, float a) {

    float v = nh * nh * (a * a - 1.0) + 1.0;
    return a * a / (pi * v * v + eps);
}


float gschlick(float nv, float k) {

    return nv / (nv * (1.0 - k) + k + eps);
}


float gsmith(float nv, float nl, float k) {

    return gschlick(nv, k) * gschlick(nl, k);
}


vec3 fschlick(float vh, vec3 f0) {

    return f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
}


float compute_shadow(int light_index) {

    float texture_size = lights[light_index].shadowmap_texture_size;
    
    vec4 frag_pos4 = shadowmap_frag_position[light_index];
    vec3 frag_pos3 = (frag_pos4.xyz / frag_pos4.w) * 0.5 + 0.5;

    float shadow = 0;

    for (int i = -1; i <= 1; i++)   {
        for (int j = -1; j <= 1; j++) {

            float shadow_depth = texture(
                textures[lights[light_index].shadowmap_texture].t, 
                frag_pos3.xy + vec2(i, j) / texture_size
            ).r;

            shadow += (frag_pos3.z > shadow_depth && frag_pos3.z <= 1.0) ? 0.95 : 0.0;
        }
    }

    return shadow / 9.0;
} 


vec3 compute_light(int light_index) {

    int li = light_index;
    int mi = material;

    float light_distance = 1.0;
    vec3 light_direction;

    if (lights[li].type == DIRECTIONAL_LIGHT) {
        light_direction = normalize(lights[li].direction);
    } 
    else {
        light_direction = normalize(lights[li].position - frag_position);
        light_distance = length(lights[li].position - frag_position);
    }

    //
    // https://nicedoc.io/KhronosGroup/glTF/extensions/2.0/Khronos/KHR_lights_punctual
    //

    float cone_attenuation = 1.0;

    if (lights[li].type == SPOT_LIGHT) {
        
        float scale = 1.0 / max(lights[li].inner_cone - lights[li].outer_cone, 0.0001);
        float offset = -lights[li].outer_cone * scale;
        float cd = dot(light_direction, normalize(lights[li].direction));
        
        cone_attenuation = clamp(cd * scale + offset, 0.0, 1.0);
        cone_attenuation *- cone_attenuation;
    }

    vec3 view_direction = normalize(camera.position - frag_position);

    vec3 normal = normalize(frag_normal);
    
    if (materials[mi].normals_texture >= 0) {

        mat3 TBN = cotangent_frame(frag_normal, -view_direction, frag_uv); 
       
        normal = texture(textures[materials[mi].normals_texture].t, frag_uv).rgb;
        normal = pow(normal, vec3(materials[mi].normals_gamma)) * 2 - 1;
        normal.xy *= materials[mi].normals_scale;
        normal = normalize(TBN * normalize(normal)); 
    }

    vec3 halfway_direction = normalize(light_direction + view_direction);

    float nv = max(dot(normal, view_direction), 0);
    float nl = max(dot(normal, light_direction), 0);
    float nh = max(dot(normal, halfway_direction), 0);
    float vh = max(dot(view_direction, halfway_direction), 0);

    vec3 color = materials[mi].color.xyz;

    if (materials[mi].color_texture >= 0) {
        color = texture(textures[materials[mi].color_texture].t, frag_uv).xyz;
        color = pow(color, vec3(2.2));
    }

    float metallic = materials[mi].metallic;
    float roughness = materials[mi].roughness;

    if (materials[mi].roughness_texture >= 0) {
        vec4 r4 = texture(textures[materials[mi].roughness_texture].t, frag_uv);
        roughness = r4.y;
        metallic = 1.0 - r4.w;
    }

    vec3 f0 = mix(color * materials[mi].specular, color, metallic);
    vec3 f = fschlick(vh, f0);

    float r = materials[mi].roughness;
    float k = (r + 1.0) * (r + 1.0) / 8.0;

    float d = dggx(nh, r * r);
    float g = gsmith(nv, nl, k);

    vec3 ks = f;
    vec3 kd = (vec3(1.0) - ks) * (1.0 - metallic);

    vec3 radiance = cone_attenuation * lights[li].intensity * lights[li].color;
    radiance = radiance / (light_distance * light_distance);

    vec3 fcooktor = d * g * f / (4 * nv * nl + eps);
    vec3 flambert = color / pi;

    vec3 fr = kd * flambert + fcooktor;

    return fr * radiance * nl;
} 


void main() {

    if (shadowmap_pass  == 1) {
        return;
    }
    
    //FragColor = vec4(1.0, 0.0, 0.0, 1.0);
    //return;

    if (cubemap.texture_exists == 1 && cubemap.render_cubemap == 1) {
        FragColor = cubemap.intensity * texture(cubemap.texture, vert_position);
        return;
    }

    int mi = material;

    vec3 color = materials[mi].emissive;

    if (materials[mi].emissive_texture >= 0) {
        color = texture(textures[materials[mi].emissive_texture].t, frag_uv).xyz;
        color = pow(color, vec3(2.2));
    }

    for (int i = 0; i < nlights; i++) {
        color += compute_light(i) * (1.0 - compute_shadow(i));
    }
   
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    FragColor = vec4(color, 1.0);
} 

