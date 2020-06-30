#version 330 core

uniform mat4 model;

out vec4 FragColor;

in vec3 frag_position;
in vec3 frag_normal;
in vec2 frag_uv;

struct Material { 

    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    vec4 emissive;
    
    float shininess;

    int texture_exists;
    sampler2D texture;

    int texture_bump_exists;
    sampler2D texture_bump;

    int texture_specular_highlight_exists;
    sampler2D texture_specular_highlight;
};  

uniform Material material;

struct Camera { 

    vec3 position;
    vec3 direction;
};  

uniform Camera camera;

#define DIRECTIONAL_LIGHT 0
#define POINT_LIGHT 1
#define SPOT_LIGHT 2

struct Light { 

    int type;

    vec3 position;
    vec3 direction;   
    
    float constant;
    float linear;
    float quadratic;  

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};  

uniform Light lights[16];
uniform int nlights;


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


vec3 compute_light(Light light) {

    float attenuation = 1.0;
    vec3 light_direction;

    if (light.type == DIRECTIONAL_LIGHT) {
        light_direction = normalize(light.direction);
    } 
    else {
        light_direction = normalize(light.position - frag_position);
        float r = length(light.position - frag_position);
        attenuation = 1.0 / (light.constant + light.linear * r + light.quadratic * r * r);
    }

    vec3 view_direction = normalize(frag_position - camera.position);

    vec3 normal = frag_normal;
    
    if (material.texture_bump_exists == 1) {

        mat3 TBN = cotangent_frame(frag_normal, view_direction, frag_uv); 

        normal = texture(material.texture_bump, frag_uv).rgb * 2 - 1;
        normal = normalize(TBN * normal); 

    }

    vec3 light_reflection = normalize(reflect(-light_direction, normal));

    float specular_intensity = pow(max(0.0, dot(-view_direction, light_reflection)), material.shininess) / 10;
    float diffuse_intensity = max(0.0, dot(normal, light_direction));

    vec3 color = material.diffuse.xyz;

    if (material.texture_exists == 1) {
        color = texture(material.texture, frag_uv).xyz;
    }

    vec3 ambient = light.ambient * color * attenuation;
    vec3 diffuse = light.diffuse * diffuse_intensity * color * attenuation;
    vec3 specular = light.specular * specular_intensity * material.specular.xyz * attenuation;

    if (material.texture_specular_highlight_exists == 1) {
        specular *= texture(material.texture_specular_highlight, frag_uv);
    }

    return ambient + diffuse + specular; 
} 


void main() {

    vec3 color = vec3(0.0);

    for (int i = 0; i < nlights; i++) {
        color += compute_light(lights[i]);
    }

    FragColor = vec4(color, 1.0);
} 

