#version 330 core

out vec4 FragColor;

uniform mat4 model;

in vec3 vert_position;
in vec3 frag_position;
in vec3 frag_normal;
in vec2 frag_uv;

uniform int has_tangents;
in float frag_tangent_handedness;
in vec3 frag_tangent;

struct Cubemap {

    int render_cubemap;
    int texture_exists;
    float intensity;
    samplerCube texture;
};

uniform Cubemap cubemap;

struct Material { 

    vec4 color;
    sampler2D color_texture;
    int color_texture_exists;

    sampler2D normals_texture;
    int normals_texture_exists;
    float normals_gamma;

    float specular;
    float metallic;

    float roughness;
    sampler2D roughness_texture;
    int roughness_texture_exists;

    vec3 emissive;  
    sampler2D emissive_texture;
    int emissive_texture_exists;
};  

uniform Material material;

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
};  

uniform Light lights[16];
uniform int nlights;


//
// https://gamedev.stackexchange.com/questions/68612/how-to-compute-tangent-and-bitangent-vectors
// https://www.gamasutra.com/blogs/RobertBasler/20131122/205462/Three_Normal_Mapping_Techniques_Explained_For_the_Mathematically_Uninclined.php
// http://www.thetenthplanet.de/archives/1180
//

mat3 cotangent_frame1( vec3 N, vec3 p, vec2 uv ) { 
    
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


//
// https://www.reddit.com/r/GraphicsProgramming/comments/gahx0l/help_calculating_correct_tangent_matrix_for/
//

mat3 cotangent_frame0() {

    vec3 N = normalize(frag_normal);

    vec3 T = normalize(frag_tangent);

    vec3 X = normalize(T - N * dot(N, T));

    vec3 B = cross(X, N) * frag_tangent_handedness;

    return mat3(T, B, N);    
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


vec3 compute_light(Light light) {

    float light_distance = 1.0;
    vec3 light_direction;

    if (light.type == DIRECTIONAL_LIGHT) {
        light_direction = -normalize(light.direction);
    } 
    else {
        light_direction = normalize(light.position - frag_position);
        light_distance = length(light.position - frag_position);
    }

    vec3 view_direction = normalize(camera.position - frag_position);

    vec3 normal = normalize(frag_normal);
    //vec3 normal = normalize(frag_tangent);
    
    if (material.normals_texture_exists == 1) {

        mat3 TBN;
        
        if (has_tangents == -1) {
            TBN = cotangent_frame0(); 
        }
        else {
            TBN = cotangent_frame1(frag_normal, -view_direction, frag_uv); 
        } 

        normal = texture(material.normals_texture, frag_uv).rgb;
        normal = pow(normal, vec3(material.normals_gamma)) * 2 - 1;
        normal = normalize(TBN * normal); 
    }

    vec3 halfway_direction = normalize(light_direction + view_direction);

    float nv = max(dot(normal, view_direction), 0);
    float nl = max(dot(normal, light_direction), 0);
    float nh = max(dot(normal, halfway_direction), 0);
    float vh = max(dot(view_direction, halfway_direction), 0);

    vec3 color = material.color.xyz;

    if (material.color_texture_exists == 1) {
        color = texture(material.color_texture, frag_uv).xyz;
    }

    color = pow(color, vec3(2.2));

    float metallic = material.metallic;
    float roughness = material.roughness;

    if (material.roughness_texture_exists == 1) {
        vec4 r4 = texture(material.roughness_texture, frag_uv);
        roughness = r4.y;
        metallic = 1.0 - r4.w;
    }

    vec3 f0 = mix(color * material.specular, color, metallic);
    vec3 f = fschlick(vh, f0);

    //float r = material.roughness + 1.0;
    //float k = r * r / 8.0;

    float d = dggx(nh, material.roughness * material.roughness);
    float g = gsmith(nv, nl, (material.roughness + 1.0) * (material.roughness + 1.0) / 8.0);

    vec3 ks = f;
    vec3 kd = (vec3(1.0) - ks) * (1.0 - metallic);

    vec3 radiance = light.intensity * pow(light.color, vec3(2.2));
    radiance = radiance / (light_distance * light_distance);

    vec3 fcooktor = d * g * f / (4 * nv * nl + eps);
    vec3 flambert = color / pi;

    vec3 fr = kd * flambert + fcooktor;

    return fr * radiance * nl;
} 


void main() {

    //FragColor = vec4(1.0, 0.0, 0.0, 1.0);
    //return;

    if (cubemap.texture_exists == 1 && cubemap.render_cubemap == 1) {
        FragColor = cubemap.intensity * pow(texture(cubemap.texture, vert_position), vec4(2.2));
        return;
    }

    vec3 color = material.emissive;

    if (material.emissive_texture_exists == 1) {
        color = texture(material.emissive_texture, frag_uv).xyz;
    }

    color = pow(color, vec3(2.2));

    for (int i = 0; i < nlights; i++) {
        color += compute_light(lights[i]);
    }

    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    FragColor = vec4(color, 1.0);
} 

