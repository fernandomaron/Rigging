#version 330
uniform mat4 transform;
uniform mat4 bone0; uniform mat4 bone1; uniform mat4 bone2; uniform mat4 bone3; uniform mat4 bone4; uniform mat4 bone5; uniform mat4 bone6; uniform mat4 bone7; uniform mat4 bone8; uniform mat4 bone9;
uniform mat4 bone10; uniform mat4 bone11; uniform mat4 bone12; uniform mat4 bone13; uniform mat4 bone14; uniform mat4 bone15; uniform mat4 bone16; uniform mat4 bone17; uniform mat4 bone18; uniform mat4 bone19;
uniform mat4 bone20; uniform mat4 bone21; uniform mat4 bone22; uniform mat4 bone23; uniform mat4 bone24; uniform mat4 bone25; uniform mat4 bone26; uniform mat4 bone27; uniform mat4 bone28; uniform mat4 bone29; 
uniform mat4 bone30; uniform mat4 bone31; uniform mat4 bone32; uniform mat4 bone33; uniform mat4 bone34; uniform mat4 bone35; uniform mat4 bone36; uniform mat4 bone37; uniform mat4 bone38; uniform mat4 bone39;
uniform mat4 bone40; uniform mat4 bone41; uniform mat4 bone42; uniform mat4 bone43; uniform mat4 bone44; uniform mat4 bone45; uniform mat4 bone46; uniform mat4 bone47; uniform mat4 bone48; uniform mat4 bone49;
uniform mat4 bone50; uniform mat4 bone51; uniform mat4 bone52; uniform mat4 bone53; uniform mat4 bone54; uniform mat4 bone55; uniform mat4 bone56; uniform mat4 bone57; uniform mat4 bone58; uniform mat4 bone59;
uniform mat4 bone60; uniform mat4 bone61; uniform mat4 bone62; uniform mat4 bone63; uniform mat4 bone64; uniform mat4 bone65; uniform mat4 bone66; uniform mat4 bone67; uniform mat4 bone68; uniform mat4 bone69;
in vec3 Position;
in vec3 Normal;	
in vec4 Color;
in vec4 BoneWeights;
in vec4 BoneIndices;
out vec4 frag_color;
out vec3 frag_normal;
out vec3 frag_position;

void main() {
	mat4 bones[70] = mat4[70](bone0, bone1, bone2, bone3, bone4, bone5, bone6, bone7, bone8, bone9, bone10, bone11, bone12, bone13, bone14, bone15, bone16, bone17, bone18, bone19, bone20, bone21, bone22, bone23, bone24, bone25, bone26, bone27, bone28, bone29, bone30, bone31, bone32, bone33, bone34, bone35, bone36, bone37, bone38, bone39, bone40, bone41, bone42, bone43, bone44, bone45, bone46, bone47, bone48, bone49, bone50, bone51, bone52, bone53, bone54, bone55, bone56, bone57, bone58, bone59, bone60, bone61, bone62, bone63, bone64, bone65, bone66, bone67, bone68, bone69);

	mat4 BoneTransform = bones[ int(BoneIndices[0]) ] * BoneWeights[0];
	BoneTransform += bones[ int(BoneIndices[1]) ] * BoneWeights[1];
	BoneTransform += bones[ int(BoneIndices[2]) ] * BoneWeights[2];
	BoneTransform += bones[ int(BoneIndices[3]) ] * BoneWeights[3];

	vec4 skinned = BoneTransform * vec4(Position, 1.0);

	vec3 skinned_normal = inverse(transpose(
		  mat3(BoneWeights.x * bones[ int(BoneIndices[0]) ])
		+ mat3(BoneWeights.y * bones[ int(BoneIndices[1]) ])
		+ mat3(BoneWeights.z * bones[ int(BoneIndices[2]) ])
		+ mat3(BoneWeights.w * bones[ int(BoneIndices[3]) ]) )) * Normal;

	frag_normal = mat3(transpose(inverse(transform))) * skinned_normal;
	frag_color = vec4(Color);
	frag_position = vec3(transform * skinned);
	gl_Position = vec4(frag_position, 1.0f);

}