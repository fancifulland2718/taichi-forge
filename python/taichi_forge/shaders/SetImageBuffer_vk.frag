#version 450

layout(binding = 0) readonly buffer PackedFrame {
  uint pixels[];
} frame;

layout(location = 0) in vec2 frag_texcoord;

layout(location = 0) out vec4 out_color;

layout(binding = 1) uniform UBO {
  vec2 lower_bound;
  vec2 upper_bound;
  float x_factor;
  float y_factor;
  int is_transposed;
  int image_width;
  int image_height;
} ubo;

void main() {
  vec2 coord = frag_texcoord * vec2(ubo.x_factor, ubo.y_factor);
  coord = clamp(coord, ubo.lower_bound, ubo.upper_bound);

  int i;
  int j;
  if (ubo.is_transposed != 0) {
    i = int(coord.x * float(ubo.image_width));
    j = int(coord.y * float(ubo.image_height));
  } else {
    i = int(coord.y * float(ubo.image_width));
    j = int(coord.x * float(ubo.image_height));
  }
  i = clamp(i, 0, ubo.image_width - 1);
  j = clamp(j, 0, ubo.image_height - 1);

  uint packed = frame.pixels[i * ubo.image_height + j];
  out_color = vec4(
      float(packed & 0xffu),
      float((packed >> 8) & 0xffu),
      float((packed >> 16) & 0xffu),
      float((packed >> 24) & 0xffu)) / 255.0;
}
