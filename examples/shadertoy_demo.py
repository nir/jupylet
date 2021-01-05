"""
    examples/shadertoy_demo.py

    This script demonstrates how to display shadertoys. Check out the 
    documentation for more information: 
    https://jupylet.readthedocs.io/en/latest/programmers_reference_guide/graphics-3d.html#shadertoys    
    
    The code displays the beautiful Star Nest Shadertoy by Pablo Roman Andrioli
    https://www.shadertoy.com/view/XlfGRj. 
"""


import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jupylet.app import App
from jupylet.label import Label
from jupylet.shadertoy import Shadertoy

from jupylet.audio.bundle import *


app = App(width=533, height=300)


st = Shadertoy("""

    // Star Nest by Pablo Roman Andrioli

    // This content is under the MIT License.

    #define iterations 17
    #define formuparam 0.53

    #define volsteps 20
    #define stepsize 0.1

    #define zoom   0.800
    #define tile   0.850
    #define speed  0.010 

    #define brightness 0.0015
    #define darkmatter 0.300
    #define distfading 0.730
    #define saturation 0.850


    void mainImage( out vec4 fragColor, in vec2 fragCoord )
    {
        //get coords and direction
        vec2 uv=fragCoord.xy/iResolution.xy-.5;
        uv.y*=iResolution.y/iResolution.x;
        vec3 dir=vec3(uv*zoom,1.);
        float time=iTime*speed+.25;

        //mouse rotation
        float a1=.5+iMouse.x/iResolution.x*2.;
        float a2=.8+iMouse.y/iResolution.y*2.;
        mat2 rot1=mat2(cos(a1),sin(a1),-sin(a1),cos(a1));
        mat2 rot2=mat2(cos(a2),sin(a2),-sin(a2),cos(a2));
        dir.xz*=rot1;
        dir.xy*=rot2;
        vec3 from=vec3(1.,.5,0.5);
        from+=vec3(time*2.,time,-2.);
        from.xz*=rot1;
        from.xy*=rot2;

        //volumetric rendering
        float s=0.1,fade=1.;
        vec3 v=vec3(0.);
        for (int r=0; r<volsteps; r++) {
            vec3 p=from+s*dir*.5;
            p = abs(vec3(tile)-mod(p,vec3(tile*2.))); // tiling fold
            float pa,a=pa=0.;
            for (int i=0; i<iterations; i++) { 
                p=abs(p)/dot(p,p)-formuparam; // the magic formula
                a+=abs(length(p)-pa); // absolute sum of average change
                pa=length(p);
            }
            float dm=max(0.,darkmatter-a*a*.001); //dark matter
            a*=a*a; // add contrast
            if (r>6) fade*=1.-dm; // dark matter, don't render near
            //v+=vec3(dm,dm*.5,0.);
            v+=fade;
            v+=vec3(s,s*s,s*s*s*s)*a*brightness*fade; // coloring based on distance
            fade*=distfading; // distance fading
            s+=stepsize;
        }
        v=mix(vec3(length(v)),v,saturation); //color adjust
        fragColor = vec4(v*.01,1.);	

    }
    
""", 533, 300)#, -1066, -600)#, 0, 'center', 'center')


l0 = Label('Star Nest by Pablo Roman Andrioli', x=10, y=10, color='cyan')


@app.event
def render(ct, dt):

    st.render(ct, dt)    
    l0.render()


sample = Sample('../docs/_static/audio/tb303.5.ogg', loop=True, amp=8.)
sample.play()


if __name__ == '__main__':
    app.run()

