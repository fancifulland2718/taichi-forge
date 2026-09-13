#version 450
layout(location = 0) out vec4 sum;
layout(location = 1) out vec4 difference;
void main() {
    sum = vec4(4.0, 0.25, 0.5, 0.5);
    difference = vec4(2.0, 0.5, 1.0, 0.25);
}
