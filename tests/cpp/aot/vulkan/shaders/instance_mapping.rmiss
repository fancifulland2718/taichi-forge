#version 460
#extension GL_EXT_ray_tracing : require
struct Hit { uvec4 ids; vec4 geometry; };
layout(location = 0) rayPayloadInEXT Hit payload;
layout(shaderRecordEXT, std430) buffer Record { uint marker; } record_data;
void main() {
    payload.ids = uvec4(0xffffffff, 0xffffffff, 0xffffffff, record_data.marker);
    payload.geometry = vec4(-1.0);
}
