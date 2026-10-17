#version 460
layout(location = 0) in vec2 position;
layout(location = 1) in vec3 input_color;
layout(location = 0) out vec3 color;
layout(push_constant) uniform Parameters { float scale; } params;
void main() {
    gl_Position = vec4(position * params.scale, 0.0, 1.0);
    color = input_color;
}
