
#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in vec2 in_texcoord_0;

uniform mat4 model;
uniform mat4 projection;

out vec2 frag_uv;

void main() {
    gl_Position = projection * model * vec4(in_position, 1.0);
    frag_uv = in_texcoord_0;
}

#elif defined FRAGMENT_SHADER

uniform vec3      iResolution;           // viewport resolution (in pixels)
uniform float     iTime;                 // shader playback time (in seconds)
uniform float     iTimeDelta;            // render time (in seconds)
uniform int       iFrame;                // shader playback frame
uniform float     iChannelTime[4];       // channel playback time (in seconds)
uniform vec3      iChannelResolution[4]; // channel resolution (in pixels)
uniform vec4      iMouse;                // mouse pixel coords. xy: current (if MLB down), zw: click
uniform sampler2D iChannel0;             // input channel. XX = 2D/Cube
uniform sampler2D iChannel1;             // input channel. XX = 2D/Cube
uniform sampler2D iChannel2;             // input channel. XX = 2D/Cube
uniform sampler2D iChannel3;             // input channel. XX = 2D/Cube
uniform vec4      iDate;                 // (year, month, day, time in seconds)
uniform float     iSampleRate;           // sound sample rate (i.e., 44100)


void mainImage( out vec4 fragColor, in vec2 fragCoord ) {}


uniform int components;
uniform vec4 color;

in vec2 frag_uv;

out vec4 fragColor;


void main() {

    vec2 uv = frag_uv;

    uv *= iResolution.xy;

    fragColor = color;

    vec4 color0;

    mainImage(color0, uv);
    
    if (components == 4) {
        fragColor *= color0;
    }
    else if (components == 1) {
        fragColor.a *= color0.x;        
    }
    else {
        fragColor.rgb *= color0.rgb;
    }
}

#endif
