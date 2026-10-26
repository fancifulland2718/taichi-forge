#version 450
layout(constant_id = 0) const uint selected = 0;
layout(set = 0, binding = 0) readonly buffer Input { vec4 value; } inputs[selected + 1];
layout(location = 0) out vec4 color;
void main() { color = inputs[selected].value; }
