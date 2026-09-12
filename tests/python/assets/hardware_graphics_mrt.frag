#version 450
// Compile with glslangValidator -V hardware_graphics_mrt.frag -o hardware_graphics_mrt.frag.spv
layout(location = 0) in vec3 vertex_color;
layout(location = 0) out vec4 color;
layout(location = 1) out uint object_id;
layout(location = 2) out int primitive_id;
void main() {
    color = vec4(vertex_color * 4.0, 0.5);
    object_id = 0xf1234567u;
    primitive_id = -123456789;
}
