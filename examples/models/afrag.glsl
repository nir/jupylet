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

    vec3 normal = frag_normal;

    if (material.texture_bump_exists == 1) {
        normal += texture(material.texture_bump, frag_uv).xyz * 2 - 1;
    }

    normal = normalize(normal);

    vec3 light_reflection = normalize(reflect(-light_direction, normal));
    vec3 camera_direction = normalize(camera.position - frag_position);

    float specular_intensity = pow(max(0.0, dot(camera_direction, light_reflection)), material.shininess * 10);
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
