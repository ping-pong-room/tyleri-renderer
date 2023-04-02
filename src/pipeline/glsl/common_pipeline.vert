#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec3 pos;
layout (location = 1) in vec2 uv;

layout( push_constant ) uniform constants
{
	mat4 view_x_model;
	mat4 projection;
} MVP;


layout (location = 0) out vec2 o_uv;
void main() {
    o_uv = uv;
    gl_Position = MVP.projection * MVP.view_x_model * vec4(pos, 1.0);
}
