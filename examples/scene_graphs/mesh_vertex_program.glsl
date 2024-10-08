#version 330
in vec3 position;
uniform vec4 color;
uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;

out vec4 fragColor;

void main()
{
    fragColor = color;
    gl_Position = projection * view * transform * vec4(position, 1.0f);
}