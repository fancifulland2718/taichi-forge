#version 450
layout(constant_id = 0) const uint count = 1;
layout(constant_id = 1) const bool enabled = false;
layout(constant_id = 2) const float gain = 0.25;
layout(constant_id = 3) const int bias = 0;
layout(location = 0) out vec4 color;
void main() {
  color = vec4(float(count + bias) * gain, enabled ? 0.75 : 0.0, 0.5, 1.0);
}
