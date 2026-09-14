#version 460
#extension GL_EXT_ray_tracing : require
struct Hit { uvec4 ids; vec4 geometry; };
layout(location = 0) rayPayloadInEXT Hit payload;
hitAttributeEXT vec2 barycentrics;
layout(shaderRecordEXT, std430) buffer Record { uint marker; } record_data;
void main() {
    payload.ids = uvec4(gl_PrimitiveID, gl_InstanceID, gl_InstanceCustomIndexEXT, record_data.marker);
    payload.geometry = vec4(gl_HitTEXT, barycentrics, 1.0);
}
