#version 450
layout(set = 0, binding = 1, std430) buffer State {
  readonly uint input_value;
  uint output_value;
} state;
layout(location = 0) out vec4 color;
void main() {
  state.output_value = state.input_value;
  color = vec4(float(state.input_value));
}
