#version 330 core

out vec4 FragColor;

uniform float mesh_shadow_bias;

uniform mat4 model;

in vec4 frag_view;
in vec3 vert_position;
in vec3 frag_position;
in vec3 frag_normal;
in vec2 frag_uv;

struct Skybox {

    int render_skybox;
    int texture_exists;
    float intensity;
    samplerCube texture;
};

uniform Skybox skybox;


uniform sampler2DArray tarr;


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

#define MAX_MATERIALS 12

uniform Material materials[MAX_MATERIALS];
uniform int material;

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

    vec3 color; // Expected as linear - physical.
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

uniform sampler2D shadowmap_texture;

uniform int shadowmap_width;
uniform int shadowmap_size;
uniform int shadowmap_pad;

uniform int shadowmap_pass;
uniform int shadowmap_light;


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
    float invmax = inversesqrt( max( dot(T, T), dot(B, B) ) ); 
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

    if (shadowmap_pass != 2 || lights[light_index].shadows != 1) {
        return 0.0;
    }

    float depth = -frag_view.z / camera.zfar;

    for (int n = lights[light_index].shadowmap_textures_count - 1; n >= 0; n--) {

        int layer = lights[light_index].shadowmap_textures[n].layer;
        vec2 xxyy = floor(vec2(layer % shadowmap_width, layer / shadowmap_width));

        if (depth <= lights[light_index].shadowmap_textures[n].depth) {

            vec4 frag_pos4 = lights[light_index].shadowmap_textures[n].projection * vec4(frag_position, 1.0);
            vec3 frag_pos3 = (frag_pos4.xyz / frag_pos4.w) * 0.5 + 0.5;

            frag_pos3.xy *= shadowmap_size - 2 * shadowmap_pad;
            frag_pos3.xy += shadowmap_pad;
            frag_pos3.xy = clamp(frag_pos3.xy, 0, shadowmap_size);
            frag_pos3.xy /= shadowmap_size;
            frag_pos3.xy += xxyy;
            frag_pos3.xy /= shadowmap_width;

            float shadow = 0;

            int n = 0;
            int pcf = lights[light_index].shadowmap_pcf / 2;

            for (int i = -pcf; i <= pcf; i++)   {
                for (int j = -pcf; j <= pcf; j++) {

                    float shadow_depth = texture(
                        shadowmap_texture, 
                        frag_pos3.xy + vec2(i, j) / shadowmap_size / shadowmap_width
                    ).r;

                    shadow += (frag_pos3.z - mesh_shadow_bias >= shadow_depth && frag_pos3.z <= 1.0) ? 1.0 : 0.0;
                    n++;
                }
            }

            return shadow / n;
        }
    }

    return 0.0;
} 


struct Light0 { 

    vec3 view_direction;
    vec3 normal;   
    vec3 color; 

    float roughness;
    float metallic;
};  

Light0 l0;


void compute_light0() {

    int mi = material;

    l0.view_direction = normalize(camera.position - frag_position);

    l0.normal = normalize(frag_normal);
    
    if (materials[mi].normals_texture >= 0) {

        mat3 TBN = cotangent_frame(l0.normal, -l0.view_direction, frag_uv); 
        int layer = materials[mi].normals_texture;

        l0.normal = texture(tarr, vec3(frag_uv, layer)).rgb;
        l0.normal = pow(l0.normal, vec3(materials[mi].normals_gamma)) * 2 - 1;
        l0.normal.xy *= materials[mi].normals_scale;
        l0.normal = normalize(TBN * normalize(l0.normal)); 
    }

    l0.color = materials[mi].color.xyz;

    if (materials[mi].color_texture >= 0) {
        int layer = materials[mi].color_texture;
        l0.color = texture(tarr, vec3(frag_uv, layer)).xyz;
        l0.color = pow(l0.color, vec3(2.2));
    }

    l0.metallic = materials[mi].metallic;
    l0.roughness = materials[mi].roughness;

    if (materials[mi].roughness_texture >= 0) {
        int layer = materials[mi].roughness_texture;
        vec4 r4 = texture(tarr, vec3(frag_uv, layer));
        l0.roughness = r4.y;
        l0.metallic = 1.0 - r4.w;
    }
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

    vec3 halfway_direction = normalize(light_direction + l0.view_direction);

    float nv = max(dot(l0.normal, l0.view_direction), 0);
    float nl = max(dot(l0.normal, light_direction), 0);
    float nh = max(dot(l0.normal, halfway_direction), 0);
    float vh = max(dot(l0.view_direction, halfway_direction), 0);

    vec3 f0 = mix(l0.color * materials[mi].specular, l0.color, l0.metallic);
    vec3 f = fschlick(vh, f0);

    float r = materials[mi].roughness;
    float k = (r + 1.0) * (r + 1.0) / 8.0;

    float d = dggx(nh, r * r);
    float g = gsmith(nv, nl, k);

    vec3 ks = f;
    vec3 kd = (vec3(1.0) - ks) * (1.0 - l0.metallic);

    vec3 radiance = cone_attenuation * lights[li].intensity * lights[li].color;
    radiance = radiance / (light_distance * light_distance);

    vec3 fcooktor = d * g * f / (4 * nv * nl + eps);
    vec3 flambert = l0.color / pi;

    vec3 fr = kd * flambert + fcooktor;

    float shadow = compute_shadow(light_index);
    vec3 ambient = lights[li].ambient * l0.color;

    return ambient * radiance + (1.0 - shadow) * fr * radiance * nl;
} 


void main() {

    if (shadowmap_pass  == 1) {
        return;
    }
    
    if (skybox.texture_exists == 1 && skybox.render_skybox == 1) {
        FragColor = skybox.intensity * texture(skybox.texture, normalize(vert_position));
        return;
    }

    int mi = material;

    vec3 color = materials[mi].emissive;

    if (materials[mi].emissive_texture >= 0) {
        int layer = materials[mi].emissive_texture;
        color = texture(tarr, vec3(frag_uv, layer)).xyz;
        color = pow(color, vec3(2.2));
    }

    compute_light0();

    for (int i = 0; i < nlights; i++) {
        color += compute_light(i);
    }
   
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    FragColor = vec4(color, 1.0);
} 

