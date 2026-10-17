#version 450
layout(location = 0) flat in float value;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 1, std430) writeonly buffer State { float data[]; } state;
void main() {
  uint i = uint(gl_FragCoord.x) + 8 * uint(gl_FragCoord.y);
  state.data[i] = value + float(i);
  color = vec4(1.0);
}
