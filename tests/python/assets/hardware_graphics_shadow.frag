#version 450
layout(set = 0, binding = 0) uniform sampler2DShadow shadow_map;
layout(location = 0) out vec4 result;
void main() {
    result = vec4(textureLod(shadow_map, vec3(0.25, 0.5, 0.5), 0.0),
                  textureLod(shadow_map, vec3(0.75, 0.5, 0.5), 0.0),
                  textureLod(shadow_map, vec3(0.5, 0.5, 0.5), 0.0), 1.0);
}
