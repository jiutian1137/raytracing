#pragma once

/// High Level Library of OpenGL. 
///@license Free 
///@review 2022-6-9 
///@contact Jiang1998Nan@outlook.com 
#define _OPENGL_LIBRARY_HIGHLEVEL_

#include "library.hpp"

#include <string>
#include <vector>
#include <memory>
#include <cassert>

struct GLbuffer;
struct GLimage;
struct GLsampler;

struct GLbuffer_info {
	GLbuffer_usages usage = GL_STATIC_DRAW;
	GLsizei byteLength = 0;
}; 

struct GLimage_info {
	GLtexture_targets target = GL_TEXTURE_2D;
	GLtexture_formats format = GL_RGBA8;
	GLsizei width  = 0;
	GLsizei height = 0;
	GLsizei depth  = 1;
	GLsizei mipLevals = 1;
	GLsizei samples = 1;
	GLboolean fixedsamplelocations = GL_FALSE;
}; 

struct GLsampler_info {
	GLtexture_filters minFilter = GL_NEAREST_MIPMAP_LINEAR;
	GLtexture_filters magFilter = GL_LINEAR;
	GLint minLod = -1000;
	GLint maxLod = 1000;
	GLint mipLodBias = 0;
	GLtexture_address_modes addressModeU = GL_REPEAT;
	GLtexture_address_modes addressModeV = GL_REPEAT;
	GLtexture_address_modes addressModeW = GL_REPEAT;
	GLenum compareEnable = GL_NONE;
	GL_compare_ops compareOp = GL_LEQUAL;
	GLfloat borderColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
}; 

struct GLbuffer_view {
/* gl.BindBufferRange( */
	GLbuffer_targets target = GL_ARRAY_BUFFER;
	/* slot, */
	GLbuffer* buffer = nullptr;
	GLsizei byteOffset = 0;
	GLsizei byteLength = 0;
/* ) */
};

struct GLvertices_view : public GLbuffer_view {
	GLsizei byteStride = 0;
};

struct GLindices_view : public GLbuffer_view {
	GL_basic_types type = GL_INT;
};

struct GLtexture_view {
/* gl.ActiveTexture(GL_TEXTURE0+slot); */
/* gl.BindTexture( */
	GLtexture_targets target = GL_TEXTURE_2D;
	GLimage* image = nullptr;
/* ) */
/* gl.BindSampler( 
	slot, */
	GLsampler* sampler = nullptr;
/* ) */
};

struct GLimage_view {
/* gl.BindImageTexture( 
	slot, */
	GLimage* image = nullptr;
	GLint level = 0;
	GLboolean layered = GL_FALSE;
	GLint layer = 0;
	GL_access access = GL_READ_WRITE;
	GLtexture_formats format = GL_RGBA32F;
/* ) */
};

struct GLimage_level_data {
/* gl.TexSubImage[*]D( 
	target, 
	level, 
	[*]offset, 
	[*]size, */
	GLtexture_channels format = GL_RGBA;
	GL_basic_types type = GL_UNSIGNED_BYTE;
	void* pixels = nullptr;
/* ) */
};

using GLimage_data = std::vector<GLimage_level_data>;

struct GLshader;
struct GLprogram;
struct GLframebuffer;

struct GLuniform_info {
	GLuint location      = -1;
	GL_basic_types type  = GL_BOOL;
	GLsizei size         = 0;
	GLchar name[128]     = {'\0'};
	GLsizei namelength   = 0;
	GLuint blockIndex    = -1;
	GLsizei offset       = 0;
	GLboolean isRowMajor = GL_FALSE;
};

struct GLuniformblock_info {
	GLuint binding = -1;
	GLchar name[128] = {'\0'};
	GLsizei namelength = 0;
	GLsizei datasize = 0;
	std::vector<GLuniform_info> members;
};

struct GLpipeline_shaderstage_info {
	GLshader_types type = GL_VERTEX_SHADER;
	std::string source;
	//bool sprivblob = false;/* unsupport spriv */
};

struct GLpipeline_vertexinput_info {
	GLuint    slot = 0;
	GLsizei   size = 0;
	GLenum    type = GL_FLOAT;
	GLboolean normalized = GL_FALSE;
	GLsizei   byteOffset = 0;
};

struct GLpipeline_inputassembly_info {
	GLprimitive_types topology = GL_POINTS;
	GLboolean primitiveRestartEnable = GL_FALSE;
};

struct GLpipeline_tessellation_info {
	GLsizei patchControlPoints = 0;
	//GLfloat patchDefaultOuterLevel = 0;
	//GLfloat patchDefaultInnerLevel = 0;
};

struct GLpipeline_rasterization_info {
	GLprimitive_polygon_modes polygonmode = GL_FILL;
	GLprimitive_windings winding = GL_CCW;
	GLprimitive_cull_modes cullmode = GL_CULL_MODE_BACK;
};

struct GLpipeline_depthstencial_info {
	GLboolean depthEnable = GL_FALSE;
	GLboolean depthWritemask = GL_TRUE;
	GLfloat   depthRange[2] = { -1.0f, +1.0f };
	GL_compare_ops depthFunc = GL_LESS;
	GLboolean stencialEnable = GL_FALSE;
	struct GLstencialOp {
		GLenum sfail = GL_KEEP;
		GLenum dpfail = GL_KEEP;
		GLenum dppass = GL_KEEP;
	} frontface, backface;
};

struct GLpipeline_colorblend_info {
	GLboolean enable = GL_FALSE;
	GLenum sfactorRGB = GL_SRC_ALPHA;
	GLenum sfactorAlpha = GL_ONE;
	GLenum dfactorRGB = GL_ONE_MINUS_SRC_ALPHA;
	GLenum dfactorAlpha = GL_ONE_MINUS_SRC_ALPHA;
	GLenum func = GL_FUNC_ADD;
};

struct GLpipeline_info {
	// Function
	std::vector<GLpipeline_shaderstage_info> shaderStages;
	// Output
	std::vector<GLenum> drawbuffers;
	// Input
	std::vector<GLpipeline_vertexinput_info> vertexInputElements;
	// Pipeline
	GLpipeline_inputassembly_info inputAssembly;
	GLpipeline_tessellation_info tessellation;
	GLpipeline_rasterization_info rasterization;
	GLpipeline_depthstencial_info depthStencial;
	std::vector<GLpipeline_colorblend_info> colorBlends;
};

inline GLpipeline_info GLpipeline_info_postprocess(const std::string& fragment_source) {
	GLpipeline_info info;
	info.shaderStages = {
		{GL_VERTEX_SHADER, 
		"#version 450 core\n"

		"const vec4 VERTICES[3] = vec4[3](\n"
			"vec4(-1,-1,-1,1),\n"
			"vec4(+3,-1,-1,1),\n"
			"vec4(-1,+3,-1,1)\n"
		");\n"

		"out vec2 texcoord;\n"
		"void main(){\n"
			"gl_Position = VERTICES[gl_VertexID];\n"
			"texcoord = VERTICES[gl_VertexID].xy*0.5+0.5;\n"
		"}"
		},
		{GL_FRAGMENT_SHADER,
		fragment_source
		}
	};

	info.inputAssembly.topology = GL_TRIANGLES;

	info.rasterization.cullmode = GL_CULL_MODE_NONE;

	info.depthStencial.depthEnable = GL_FALSE;

	info.drawbuffers = {
		GL_COLOR_ATTACHMENT0
	};

	info.colorBlends = {
		{GL_FALSE}
	};

	return info;
}

struct GLpipeline;

class GLlibrary2 {
public:
#pragma warning(disable: 4311; disable: 4302)
	GLlibrary gl;
	const GLpipeline* currentPipeline;
	const GLpipeline_vertexinput_info* currentVertexInputElements;
	GLsizei currentVertexInputCount;
	GLprimitive_types currentPrimitiveTopology;

	nullptr_t default_;

	GLlibrary2()
		: gl(), 
		currentPipeline(NULL), 
		currentVertexInputElements(NULL), currentVertexInputCount(0), 
		currentPrimitiveTopology(GL_POINTS), default_(nullptr) {}

	GLlibrary2(GLplatformArg arg)
		: gl(arg), 
		currentPipeline(NULL), 
		currentVertexInputElements(NULL), currentVertexInputCount(0), 
		currentPrimitiveTopology(GL_POINTS), default_(nullptr) {}

	void Reset() noexcept {
		gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		gl.UseProgram(0);
		currentPipeline = nullptr;
		currentVertexInputElements = nullptr;
		currentVertexInputCount    = 0;
		currentPrimitiveTopology = GL_POINTS;
	}

	GLenum GetError() { return gl.GetError(); }

	void GetIntegerv(GLenum pname, GLint* params) { gl.GetIntegerv(pname, params); }


	GLenum CreateBuffer(const GLbuffer_info& info, const void* data, GLbuffer* &buffer) {
		if (gl.GetError() != GL_NO_ERROR) {
			throw std::exception("gl.GetError() != GL_NO_ERROR");
		}

		GLuint& handler = reinterpret_cast<GLuint&>(buffer = nullptr);
		gl.GenBuffers(1, &handler);
		gl.BindBuffer(GL_ARRAY_BUFFER, handler);
		if (info.byteLength != 0) {
			gl.BufferData(GL_ARRAY_BUFFER, info.byteLength, data, info.usage);
		}

		if ( !gl.IsBuffer(handler) ) {
			return gl.GetError();
		} else {
			return GL_NO_ERROR;
		}
	}

	void DeleteBuffer(GLbuffer* &buffer) noexcept {
		GLuint& handler = reinterpret_cast<GLuint&>(buffer);
		gl.DeleteBuffers(1, &handler);
		handler = 0;
	}
	
	void GetBufferInfo(const GLbuffer& buffer, GLbuffer_info &info) {
		GLuint handler = reinterpret_cast<GLuint>(&buffer);
		if (!gl.IsBuffer(handler)) { throw std::exception("!gl.IsBuffer(buffer)"); }
		gl.BindBuffer(GL_ARRAY_BUFFER, handler);
		gl.GetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_USAGE, reinterpret_cast<GLint*>(&info.usage));
		gl.GetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, reinterpret_cast<GLint*>(&info.byteLength));
	}
	
	GLboolean IsBuffer(const GLbuffer& buffer) noexcept {
		return gl.IsBuffer(reinterpret_cast<GLuint>(&buffer));
	}
	
	/// Bind 'buffer' into GPUregister
	void BindBuffer(GLenum target, GLbuffer& buffer) {
		gl.BindBuffer(target, reinterpret_cast<GLuint>(&buffer));
	}

	/// Bind nullptr into GPUregister
	void UnbindBuffer(GLenum target) {
		gl.BindBuffer(target, 0);
	}

	void ReallocBuffer(GLenum target, const GLbuffer_info& info, const void* data) {
		if (info.byteLength != 0) {
			gl.BufferData(target, info.byteLength, data, info.usage);
		}
	}

	/// Upload CPUmemory{['source', ...)} into GPUmemory{'target',[offset, byteOffset+byteLength)}
	void UploadBufferRange(GLenum target, GLintptr taroffset, GLsizei tarbytesize, const void* source) {
		gl.BufferSubData(target, taroffset, tarbytesize, source);
	}

	/// Upload CPUmemory{['source', ...)} into GPUmemory{'target', ...}
	void UploadBuffer(GLenum target, const void* source) {
		GLsizei tarbytesize; gl.GetBufferParameteriv(target, GL_BUFFER_SIZE, reinterpret_cast<GLint*>(&tarbytesize));
		gl.BufferSubData(target, 0, tarbytesize, source);
	}

	/// Readback GPUmemory{'target',[offset, byteOffset+byteLength)} into CPUmemory{['destination', ...)}
	void ReadbackBufferRange(GLenum target, GLintptr taroffset, GLsizeiptr tarbytesize, void* destination) {
		gl.GetBufferSubData(target, taroffset, tarbytesize, destination);
	}

	/// Readback GPUmemory{'target', ...} into CPUmemory{['destination', ...)}
	void ReadbackBuffer(GLenum target, void* destination) {
		GLint tarbytesize; gl.GetBufferParameteriv(target, GL_BUFFER_SIZE, &tarbytesize);
		gl.GetBufferSubData(target, 0, tarbytesize, destination);
	}

	/// copy GPUmemory{source[soffset, soffset+copybytes)} into GPUmemory{destination[doffset, doffset+copybytes)}
	void CopyBufferRange(const GLbuffer& source, GLintptr soffset, GLbuffer& destination, GLintptr doffset, GLsizei copybytes) {
		gl.BindBuffer(GL_COPY_READ_BUFFER, reinterpret_cast<GLuint>(&source));
		gl.BindBuffer(GL_COPY_WRITE_BUFFER, reinterpret_cast<GLuint>(&destination));
		gl.CopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, soffset, doffset, copybytes);
	}

	/// copy GPUmemory{source} into GPUmemory{destination}
	void CopyBuffer(const GLbuffer& source, GLbuffer& destination, GLintptr offset = 0) {
		gl.BindBuffer(GL_COPY_READ_BUFFER, reinterpret_cast<GLuint>(&source));
		gl.BindBuffer(GL_COPY_WRITE_BUFFER, reinterpret_cast<GLuint>(&destination));
		GLint sbytesize;
		gl.GetBufferParameteriv(GL_COPY_READ_BUFFER, GL_BUFFER_SIZE, &sbytesize);
		GLint dbytesize;
		gl.GetBufferParameteriv(GL_COPY_WRITE_BUFFER, GL_BUFFER_SIZE, &dbytesize);
		if (sbytesize != dbytesize) {
			throw std::exception("sbytesize != dbytesize");
		}
		gl.CopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, offset, offset, sbytesize);
	}
	

	GLenum CreateImage(const GLimage_info& info, const GLimage_data& data, GLimage* &image) {
		if (gl.GetError() != GL_NO_ERROR)
			throw std::exception("gl.GetError() != GL_NO_ERROR");
		if (info.width == 0)
			return GL_INVALID_VALUE;
		if (data.size() > info.mipLevals)
			return GL_INVALID_VALUE;
		if (data.size() > 2'147'483'647)
			return GL_OUT_OF_MEMORY;

		GLuint& handler = reinterpret_cast<GLuint&>(image = nullptr);
		gl.GenTextures(1, &handler);
		gl.BindTexture(info.target, handler);
		switch (info.target) {
		case GL_TEXTURE_1D:
			/*gl.TexImage1D(info.target, 0, info.format, info.width, 0, data.format, data.type, data.pixels);
			gl.GenerateMipmap(info.target);*/
			gl.TexStorage1D(info.target, info.mipLevals, info.format, info.width);
			for(GLint level = 0; level != (GLint)data.size(); ++level) {
				gl.TexSubImage1D(info.target, level, 0, info.width>>level, data[level].format, data[level].type, data[level].pixels);
			}
			break;
		case GL_TEXTURE_2D:
		case GL_TEXTURE_1D_ARRAY:
		case GL_TEXTURE_CUBE_MAP:
		case GL_TEXTURE_RECTANGLE:
			/*gl.TexImage2D(info.target, 0, info.format, info.width, info.height, 0, data.format, data.type, data.pixels);
			gl.GenerateMipmap(info.target);*/
			gl.TexStorage2D(info.target, info.mipLevals, info.format, info.width, info.height);
			for (GLint level = 0; level != (GLint)data.size(); ++level) {
				gl.TexSubImage2D(info.target, level, 0, 0, info.width>>level, info.height>>level, data[level].format, data[level].type, data[level].pixels);
			}
			break;
		case GL_TEXTURE_3D:
		case GL_TEXTURE_2D_ARRAY:
			/*gl.TexImage3D(info.target, 0, info.format, info.width, info.height, info.depth, 0, data.format, data.type, data.pixels);
			gl.GenerateMipmap(info.target);*/
			gl.TexStorage3D(info.target, info.mipLevals, info.format, info.width, info.height, info.depth);
			for (GLint level = 0; level != (GLint)data.size(); ++level) {
				gl.TexSubImage3D(info.target, 0, 0, 0, 0, info.width>>level, info.height>>level, info.depth>>level, data[level].format, data[level].type, data[level].pixels);
			}
			break;
		case GL_TEXTURE_2D_MULTISAMPLE:
		case GL_PROXY_TEXTURE_2D_MULTISAMPLE:
			gl.TexStorage2DMultisample(info.target, info.samples, info.format, info.width, info.height, info.fixedsamplelocations);
			break;
		case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
		case GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY:
			gl.TexStorage3DMultisample(info.target, info.samples, info.format, info.width, info.height, info.depth, info.fixedsamplelocations);
			break;
		default:
			throw std::exception();
			break;
		}

		if (!gl.IsTexture(handler)) {
			return gl.GetError();
		} else {
			return GL_NO_ERROR;
		}
	}

	void DeleteImage(GLimage* &image) noexcept {
		GLuint& handler = reinterpret_cast<GLuint&>(image);
		gl.DeleteTextures(1, &handler);
		handler = 0;
	}
	
	void GetImageLevelInfo(GLenum target, const GLimage& image, GLint level, GLimage_info &info) {
		GLuint handler = reinterpret_cast<GLuint>(&image);
		if (!gl.IsTexture(handler)) { throw std::exception("!gl.IsTexture(image)"); }
		gl.BindTexture(target, handler);
		gl.GetTexLevelParameteriv(target == GL_TEXTURE_CUBE_MAP ? GL_TEXTURE_CUBE_MAP_RIGHT : target, level, GL_TEXTURE_WIDTH, (GLint*)(&info.width));
		gl.GetTexLevelParameteriv(target == GL_TEXTURE_CUBE_MAP ? GL_TEXTURE_CUBE_MAP_RIGHT : target, level, GL_TEXTURE_HEIGHT, (GLint*)(&info.height));
		gl.GetTexLevelParameteriv(target == GL_TEXTURE_CUBE_MAP ? GL_TEXTURE_CUBE_MAP_RIGHT : target, level, GL_TEXTURE_DEPTH, (GLint*)(&info.depth));
		gl.GetTexLevelParameteriv(target == GL_TEXTURE_CUBE_MAP ? GL_TEXTURE_CUBE_MAP_RIGHT : target, level, GL_TEXTURE_INTERNAL_FORMAT, (GLint*)(&info.format));
		gl.GetTexLevelParameteriv(target == GL_TEXTURE_CUBE_MAP ? GL_TEXTURE_CUBE_MAP_RIGHT : target, level, GL_TEXTURE_SAMPLES, (GLint*)(&info.samples));
		info.target = static_cast<GLtexture_targets>(target);
	}

	GLimage_info GetImageLevelInfo(GLenum target, const GLimage& image, GLint level) {
		GLimage_info info;
		GLuint handler = reinterpret_cast<GLuint>(&image);
		if (!gl.IsTexture(handler)) { throw std::exception("!gl.IsTexture(image)"); }
		gl.BindTexture(target, handler);
		gl.GetTexLevelParameteriv(target == GL_TEXTURE_CUBE_MAP ? GL_TEXTURE_CUBE_MAP_RIGHT : target, level, GL_TEXTURE_WIDTH, (GLint*)(&info.width));
		gl.GetTexLevelParameteriv(target == GL_TEXTURE_CUBE_MAP ? GL_TEXTURE_CUBE_MAP_RIGHT : target, level, GL_TEXTURE_HEIGHT, (GLint*)(&info.height));
		gl.GetTexLevelParameteriv(target == GL_TEXTURE_CUBE_MAP ? GL_TEXTURE_CUBE_MAP_RIGHT : target, level, GL_TEXTURE_DEPTH, (GLint*)(&info.depth));
		gl.GetTexLevelParameteriv(target == GL_TEXTURE_CUBE_MAP ? GL_TEXTURE_CUBE_MAP_RIGHT : target, level, GL_TEXTURE_INTERNAL_FORMAT, (GLint*)(&info.format));
		gl.GetTexLevelParameteriv(target == GL_TEXTURE_CUBE_MAP ? GL_TEXTURE_CUBE_MAP_RIGHT : target, level, GL_TEXTURE_SAMPLES, (GLint*)(&info.samples));
		info.target = static_cast<GLtexture_targets>(target);
		return info;
	}

	GLboolean IsImage(const GLimage& image) noexcept {
		return gl.IsTexture(reinterpret_cast<GLuint>(&image));
	}

	/// bind 'image' into target[GL_TEXTURE0]
	void BindImage(GLenum target, GLimage& image) {
		gl.ActiveTexture(GL_TEXTURE0);
		gl.BindTexture(target, reinterpret_cast<GLuint>(&image));
	}

	/// bind 'nullptr' into target[GL_TEXTURE0]
	void UnbindImage(GLenum target) {
		gl.ActiveTexture(GL_TEXTURE0);
		gl.BindTexture(target, 0);
	}

	/// bind 'image' into target[GL_TEXTURE0+slot]
	void BindImage(GLenum target, GLimage& image, GLuint slot) {
		gl.ActiveTexture(GL_TEXTURE0 + slot);
		gl.BindTexture(target, reinterpret_cast<GLuint>(&image));
	}
	
	/// bind 'nullptr' into target[GL_TEXTURE0+slot]
	void UnbindImage(GLenum target, GLuint slot) {
		gl.ActiveTexture(GL_TEXTURE0 + slot);
		gl.BindTexture(target, 0);
	}

	/// @note target cannot be GL_TEXTURE_RECTANGLE
	void GenerateMipmap(GLenum target) {
		gl.GenerateMipmap(target);
	}

	/// gl.SetPixelStorei(GL_UNPACK_ALIGNMENT, 1,2,4,8,16...); gl.SetPixelStorei(GL_PACK_ALIGNMENT, 1,2,4,8,16...).
	void SetPixelStorei(GLenum pname, GLint value) {
		gl.PixelStorei(pname, value);
	}

	//[[deprecated]]
	void ReallocImage(GLenum target, const GLimage_info& info, const GLimage_data& data) {
		switch (info.target) {
		case GL_TEXTURE_1D:
			/*gl.TexImage1D(info.target, 0, info.format, info.width, 0, data.format, data.type, data.pixels);
			gl.GenerateMipmap(info.target);*/
			gl.TexStorage1D(info.target, info.mipLevals, info.format, info.width);
			for(GLint level = 0; level != (GLint)data.size(); ++level) {
				gl.TexSubImage1D(info.target, level, 0, info.width>>level, data[level].format, data[level].type, data[level].pixels);
			}
			break;
		case GL_TEXTURE_2D:
		case GL_TEXTURE_1D_ARRAY:
		case GL_TEXTURE_CUBE_MAP:
		case GL_TEXTURE_RECTANGLE:
			/*gl.TexImage2D(info.target, 0, info.format, info.width, info.height, 0, data.format, data.type, data.pixels);
			gl.GenerateMipmap(info.target);*/
			gl.TexStorage2D(info.target, info.mipLevals, info.format, info.width, info.height);
			for (GLint level = 0; level != (GLint)data.size(); ++level) {
				gl.TexSubImage2D(info.target, level, 0, 0, info.width>>level, info.height>>level, data[level].format, data[level].type, data[level].pixels);
			}
			break;
		case GL_TEXTURE_3D:
		case GL_TEXTURE_2D_ARRAY:
			/*gl.TexImage3D(info.target, 0, info.format, info.width, info.height, info.depth, 0, data.format, data.type, data.pixels);
			gl.GenerateMipmap(info.target);*/
			gl.TexStorage3D(info.target, info.mipLevals, info.format, info.width, info.height, info.depth);
			for (GLint level = 0; level != (GLint)data.size(); ++level) {
				gl.TexSubImage3D(info.target, 0, 0, 0, 0, info.width>>level, info.height>>level, info.depth>>level, data[level].format, data[level].type, data[level].pixels);
			}
			break;
		case GL_TEXTURE_2D_MULTISAMPLE:
		case GL_PROXY_TEXTURE_2D_MULTISAMPLE:
			gl.TexStorage2DMultisample(info.target, info.samples, info.format, info.width, info.height, info.fixedsamplelocations);
			break;
		case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
		case GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY:
			gl.TexStorage3DMultisample(info.target, info.samples, info.format, info.width, info.height, info.depth, info.fixedsamplelocations);
			break;
		default:
			throw std::exception();
			break;
		}
	}

	/// upload CPUmemory{[source, ...)} into GPUmemory{target2D,level,[tarxoffset + taryoffset*tarwidth, ... + tarheight*tarwidth)}
	void UploadImageLevelRange(GLenum target, GLint level, GLint tarxoffset, GLint taryoffset, GLint tarwidth, GLint tarheight, const GLimage_level_data& source) {
		gl.TexSubImage2D(target, level, tarxoffset, taryoffset, tarwidth, tarheight, source.format, source.type, source.pixels);
	}

	/// upload CPUmemory{[source, ...)} into GPUmemory{target3D,level,[tarxoffset + taryoffset*tarwidth + tarzoffset*tarwidth*tardepth, ... + tarheight*tarwidth*tardepth)}
	void UploadImageLevelRange(GLenum target, GLint level, GLint tarxoffset, GLint taryoffset, GLint tarzoffset, GLint tarwidth, GLint tarheight, GLint tardepth, const GLimage_level_data& source) {
		gl.TexSubImage3D(target, level, tarxoffset, taryoffset, tarzoffset, tarwidth, tarheight, tardepth, source.format, source.type, source.pixels);
	}
	
	/// upload CPUmemory{[source, ...)} into GPUmemory{targetXD,level, ...}
	void UploadImageLevel(GLenum target, GLint level, const GLimage_level_data& source) {
		switch ( target ) {
		case GL_TEXTURE_1D: {
			GLint width;
			gl.GetTexLevelParameteriv(target, level, GL_TEXTURE_WIDTH, &width);
			gl.TexSubImage1D(target, level, 0, width, source.format, source.type, source.pixels);
			}
			break;
		case GL_TEXTURE_2D:
		case GL_TEXTURE_1D_ARRAY:
		case GL_TEXTURE_CUBE_MAP_RIGHT:
		case GL_TEXTURE_CUBE_MAP_LEFT:
		case GL_TEXTURE_CUBE_MAP_TOP:
		case GL_TEXTURE_CUBE_MAP_BOTTOM:
		case GL_TEXTURE_CUBE_MAP_FRONT:
		case GL_TEXTURE_CUBE_MAP_BACK:
		case GL_TEXTURE_RECTANGLE: {
			GLint width, height;
			gl.GetTexLevelParameteriv(target, level, GL_TEXTURE_WIDTH, &width);
			gl.GetTexLevelParameteriv(target, level, GL_TEXTURE_HEIGHT, &height);
			gl.TexSubImage2D(target, level, 0, 0, width, height, source.format, source.type, source.pixels);
			}
			break;
		case GL_TEXTURE_3D:
		case GL_TEXTURE_2D_ARRAY: {
			GLint width, height, depth;
			gl.GetTexLevelParameteriv(target, level, GL_TEXTURE_WIDTH, &width);
			gl.GetTexLevelParameteriv(target, level, GL_TEXTURE_HEIGHT, &height);
			gl.GetTexLevelParameteriv(target, level, GL_TEXTURE_DEPTH, &depth);
			gl.TexSubImage3D(target, level, 0, 0, 0, width, height, depth, source.format, source.type, source.pixels);
			}
			break;
		default:
			break;
		}
	}
	
	void UploadImage(GLenum target, const GLimage_data& source) {
		for (GLint level = 0; level != (GLint)source.size(); ++level) {
			this->UploadImageLevel(target, level, source[level]);
		}
	}

	/// readback GPUmemory{targetXD,level, ...} into CPUmemory{[destination, ...)}
	void ReadbackImageLevel(GLenum target, GLint level, const GLimage_level_data& destination) {
		gl.GetTexImage(target, level, destination.format, destination.type, destination.pixels);
	}
	
	void ReadbackImage(GLenum target, const GLimage_data& destination) {
		for (GLint level = 0; level != (GLint)destination.size(); ++level) {
			this->ReadbackImageLevel(target, level, destination[level]);
		}
	}


	GLenum CreateSampler(const GLsampler_info& info, GLsampler* &sampler) {
		if (gl.GetError() != GL_NO_ERROR) {
			throw std::exception("gl.GetError() != GL_NO_ERROR");
		}

		GLuint& handler = reinterpret_cast<GLuint&>(sampler = nullptr);
		gl.GenSamplers(1, &handler);
		gl.BindSampler(0, handler);
		gl.SamplerParameteri(handler, GL_TEXTURE_MIN_FILTER, info.minFilter);
		gl.SamplerParameteri(handler, GL_TEXTURE_MAG_FILTER, info.magFilter);
		gl.SamplerParameteri(handler, GL_TEXTURE_MIN_LOD, info.minLod);
		gl.SamplerParameteri(handler, GL_TEXTURE_MAX_LOD, info.maxLod);
		gl.SamplerParameteri(handler, GL_TEXTURE_LOD_BIAS, info.mipLodBias);
		gl.SamplerParameteri(handler, GL_TEXTURE_WRAP_S, info.addressModeU);
		gl.SamplerParameteri(handler, GL_TEXTURE_WRAP_T, info.addressModeV);
		gl.SamplerParameteri(handler, GL_TEXTURE_WRAP_R, info.addressModeW);
		gl.SamplerParameteri(handler, GL_TEXTURE_COMPARE_MODE, info.compareEnable);
		gl.SamplerParameteri(handler, GL_TEXTURE_COMPARE_FUNC, info.compareOp);
		gl.SamplerParameterfv(handler, GL_TEXTURE_BORDER_COLOR, info.borderColor);

		if (!gl.IsSampler(handler)) {
			return gl.GetError();
		} else {
			return GL_NO_ERROR;
		}
	}

	void DeleteSampler(GLsampler* &sampler) noexcept {
		GLuint& handler = reinterpret_cast<GLuint&>(sampler);
		gl.DeleteSamplers(1, &handler);
		handler = 0;
	}

	void GetSamplerInfo(const GLsampler& sampler, GLsampler_info &info) {
		GLuint handler = reinterpret_cast<GLuint>(&sampler);
		gl.GetSamplerParameteriv(handler, GL_TEXTURE_MAG_FILTER, (GLint*)(&info.magFilter));
		gl.GetSamplerParameteriv(handler, GL_TEXTURE_MIN_FILTER, (GLint*)(&info.minFilter));
		gl.GetSamplerParameteriv(handler, GL_TEXTURE_MIN_LOD, &info.minLod);
		gl.GetSamplerParameteriv(handler, GL_TEXTURE_MAX_LOD, &info.maxLod);
		gl.GetSamplerParameteriv(handler, GL_TEXTURE_LOD_BIAS, &info.mipLodBias);
		gl.GetSamplerParameteriv(handler, GL_TEXTURE_WRAP_S, (GLint*)(&info.addressModeU));
		gl.GetSamplerParameteriv(handler, GL_TEXTURE_WRAP_T, (GLint*)(&info.addressModeV));
		gl.GetSamplerParameteriv(handler, GL_TEXTURE_WRAP_R, (GLint*)(&info.addressModeW));
		gl.GetSamplerParameteriv(handler, GL_TEXTURE_COMPARE_MODE, (GLint*)(&info.compareEnable));
		gl.GetSamplerParameteriv(handler, GL_TEXTURE_COMPARE_FUNC, (GLint*)(&info.compareOp));
		gl.GetSamplerParameterfv(handler, GL_TEXTURE_BORDER_COLOR, info.borderColor);
	}

	GLboolean IsSampler(const GLsampler& sampler) noexcept {
		return gl.IsSampler(reinterpret_cast<GLuint>(&sampler));
	}
	

	GLenum CreateShader(GLenum type, const std::string& source, GLshader* &shader) {
		GLuint& handler = reinterpret_cast<GLuint&>(shader = nullptr);
		handler = gl.CreateShader(type);
		const GLchar* sources[1] = { source.c_str() };
		gl.ShaderSource(handler, 1, sources, nullptr);
		gl.CompileShader(handler);

		GLint status; gl.GetShaderiv(handler, GL_COMPILE_STATUS, &status);
		if (status == GL_FALSE) {
			char message[1024]; GLsizei message_size = 0;
			gl.GetShaderInfoLog(handler, 1023, &message_size, message);
			message[message_size] = '\0';
			throw std::exception(message);
		}
		return GL_NO_ERROR;
	}

	void DeleteShader(GLshader* &shader) noexcept {
		gl.DeleteShader(reinterpret_cast<GLuint>(shader));
		shader = nullptr;
	}

	GLboolean IsShader(const GLshader& shader) noexcept {
		return gl.IsShader(reinterpret_cast<GLuint>(&shader));
	}
	
	std::string GetShaderSource(GLuint shader, GLenum* type = nullptr) {
		GLsizei length;
		gl.GetShaderiv(shader, GL_SHADER_SOURCE_LENGTH, reinterpret_cast<GLint*>(&length));
		std::string source = std::string(static_cast<size_t>(length), '\0');
		gl.GetShaderSource(shader, length, &length, source.data());
		if (type != nullptr) {
			gl.GetShaderiv(shader, GL_SHADER_TYPE, reinterpret_cast<GLint*>(type));
		}
		return source;
	}

	/// std::cout << "layout(location = "<<uniform.location<<") in "<<uniform.type<<" "<<uniform.name<<std::endl;
	GLenum GetUniformInfo(const GLprogram& program_, GLuint index, GLuniform_info& uniform) {
		GLuint program = reinterpret_cast<GLuint>(&program_);

		gl.GetActiveUniformsiv(program, 1, &index, GL_UNIFORM_NAME_LENGTH, /*OUT*/(GLint*)(&uniform.namelength));
		if (uniform.namelength == 0) {
			return -1;
		}

		gl.GetActiveUniformsiv(program, 1, &index, GL_UNIFORM_BLOCK_INDEX, /*OUT*/(GLint*)(&uniform.blockIndex));
		if (uniform.blockIndex != GLuint(-1)) {
			return -2;
		}

		gl.GetActiveUniformName(program, index, uniform.namelength, /*OUT*/&uniform.namelength, uniform.name);
		uniform.name[uniform.namelength] = '\0';

		uniform.location = gl.GetUniformLocation(program, uniform.name);

		gl.GetActiveUniformsiv(program, 1, &index, GL_UNIFORM_TYPE, /*OUT*/(GLint*)(&uniform.type));

		gl.GetActiveUniformsiv(program, 1, &index, GL_UNIFORM_SIZE, /*OUT*/(GLint*)(&uniform.size));

		gl.GetActiveUniformsiv(program, 1, &index, GL_UNIFORM_OFFSET, /*OUT*/(GLint*)(&uniform.offset));

		gl.GetActiveUniformsiv(program, 1, &index, GL_UNIFORM_IS_ROW_MAJOR, /*OUT*/(GLint*)(&uniform.isRowMajor));

		return GL_NO_ERROR;
	}

	/// std::cout << "layout(binding = "<<ublock.binding<<") uniform "<<ublock.name<<"{\n";
	/// for(GLuniformInfo& member : ublock.members)
	///		std::cout << "layout(offset = "<<member.offset<<") "<<member.type<<" "<<member.name<<";\n";
	/// std::cout << "}" << std::endl;
	///
	///@note
	/// Set layout(offset=?) of Variable in Block, The following exceptions may occur,
	///		error C7601: 'offset' needs to be a multiple of the natural alignment of 'Variable', which is '16'
	///		error C3021 : offset '31' specified for 'Variable' overlaps with the previous member of the block
	GLenum GetUniformblockInfo(const GLprogram& program_, GLuint index, GLuniformblock_info& uniformblock) {
		GLuint program = reinterpret_cast<GLuint>(&program_);

		gl.GetActiveUniformBlockiv(program, index, GL_UNIFORM_BLOCK_NAME_LENGTH, /*OUT*/(GLint*)(&uniformblock.namelength));
		gl.GetActiveUniformBlockName(program, index, uniformblock.namelength, /*OUT*/&uniformblock.namelength, uniformblock.name);
		uniformblock.name[uniformblock.namelength] = '\0';
		if (gl.GetUniformBlockIndex(program, uniformblock.name) != index) {
			return -1;
		}

		gl.GetActiveUniformBlockiv(program, index, GL_UNIFORM_BLOCK_BINDING, /*OUT*/(GLint*)(&uniformblock.binding));

		gl.GetActiveUniformBlockiv(program, index, GL_UNIFORM_BLOCK_DATA_SIZE, /*OUT*/(GLint*)(&uniformblock.datasize));

		GLsizei uniforms;
		gl.GetActiveUniformBlockiv(program, index, GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS, /*OUT*/(GLint*)(&uniforms));

		if (uniforms != 0) {
			std::vector<GLuint> indices(uniforms, -1);
			uniformblock.members.resize(uniforms);
			gl.GetActiveUniformBlockiv(program, index, GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES, /*OUT*/(GLint*)(indices.data()));

			for (GLsizei i = 0; i != uniforms; ++i) {
				GLuint uniindex = indices[i];
				GLuniform_info& uniform = uniformblock.members[i];
				gl.GetActiveUniformsiv(program, 1, &uniindex, GL_UNIFORM_NAME_LENGTH, /*OUT*/(GLint*)(&uniform.namelength));
				//assert(uniform.namelength != 0);

				gl.GetActiveUniformName(program, uniindex, uniform.namelength, /*OUT*/&uniform.namelength, uniform.name);
				uniform.name[uniform.namelength] = '\0';

				gl.GetActiveUniformsiv(program, 1, &uniindex, GL_UNIFORM_TYPE, /*OUT*/(GLint*)(&uniform.type));

				gl.GetActiveUniformsiv(program, 1, &uniindex, GL_UNIFORM_SIZE, /*OUT*/(GLint*)(&uniform.size));

				gl.GetActiveUniformsiv(program, 1, &uniindex, GL_UNIFORM_BLOCK_INDEX, /*OUT*/(GLint*)(&uniform.blockIndex));

				gl.GetActiveUniformsiv(program, 1, &uniindex, GL_UNIFORM_OFFSET, /*OUT*/(GLint*)(&uniform.offset));

				gl.GetActiveUniformsiv(program, 1, &uniindex, GL_UNIFORM_IS_ROW_MAJOR, /*OUT*/(GLint*)(&uniform.isRowMajor));
			}
		}

		return GL_NO_ERROR;
	}


	GLenum CreateFramebuffer(GLframebuffer* &rendertarget) {
		GLuint& handler = reinterpret_cast<GLuint&>(rendertarget);
		gl.GenFramebuffers(1, &handler);
		return GL_NO_ERROR;
	}

	void DeleteFramebuffer(GLframebuffer* &rendertarget) noexcept {
		gl.DeleteFramebuffers(1, &reinterpret_cast<GLuint&>(rendertarget));
		rendertarget = nullptr;
	}

	void BindFramebuffer(GLenum target, GLframebuffer& rendertarget) {
		gl.BindFramebuffer(target, reinterpret_cast<GLuint>(&rendertarget));
		if (target == GL_FRAMEBUFFER) {
			this->ResetFramebuffer(target);
		}
	}

	void BindFramebuffer(GLenum target, nullptr_t) {
		assert( target != GL_READ_FRAMEBUFFER );
		gl.BindFramebuffer(target, 0);
	}
	
	GLboolean IsFramebuffer(const GLframebuffer& rendertarget) {
		return gl.IsFramebuffer(reinterpret_cast<GLuint>(&rendertarget));
	}

	GLenum CheckFramebufferStatus(GLenum target) {
		return gl.CheckFramebufferStatus(target);
	}

	void ResetFramebuffer(GLenum target) {
		if (target != GL_FRAMEBUFFER) {
			throw std::exception();
		}
		GLsizei MAX_COLOR_ATTACHMENT;
		gl.GetIntegerv(GL_MAX_COLOR_ATTACHMENTS, (GLint*)(&MAX_COLOR_ATTACHMENT));
		for (GLsizei i = 0; i != MAX_COLOR_ATTACHMENT; ++i)
			gl.FramebufferTexture2D(target, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, GL_NONE, 0);
		gl.FramebufferTexture2D(target, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, GL_NONE, 0);
		gl.FramebufferTexture2D(target, GL_STENCIL_ATTACHMENT, GL_TEXTURE_2D, GL_NONE, 0);
		/// gl.GetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, ...);
		/// equivalent to:
		///   depth_attach = gl.GetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, ...);
		///   stencel_attach = gl.GetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, ...);
		/// so, We just need to manage the two statuses of GL_DEPTH_ATTACHMENT GL_STENCIL_ATTACHMENT,
		///     do not require GL_DEPTH_STENCIL_ATTACHMENT.
	}

	void SetFramebuffer(GLenum target, GLuint attachment, GLenum imagetarget, GLimage& image, GLint level = 0, GLint layer = 0) {
		if (target != GL_FRAMEBUFFER) {
			throw std::exception();
		}
		GLsizei MAX_COLOR_ATTACHMENT;
		gl.GetIntegerv(GL_MAX_COLOR_ATTACHMENTS, (GLint*)(&MAX_COLOR_ATTACHMENT));
		if(attachment != GL_DEPTH_ATTACHMENT && attachment != GL_STENCIL_ATTACHMENT && attachment != GL_DEPTH_STENCIL_ATTACHMENT
			&& (attachment < GL_COLOR_ATTACHMENT0 || GL_COLOR_ATTACHMENT0+MAX_COLOR_ATTACHMENT < attachment) ) { 
			throw std::exception("attachment is invalid");
		}
		
		switch (imagetarget) {
		case GL_TEXTURE_1D:
		case GL_PROXY_TEXTURE_1D:
		case GL_TEXTURE_BUFFER:
			gl.FramebufferTexture1D(target, attachment, imagetarget, reinterpret_cast<GLuint>(&image), level);
			break;
		case GL_TEXTURE_2D:
		case GL_TEXTURE_2D_MULTISAMPLE:
		case GL_PROXY_TEXTURE_2D:
		case GL_PROXY_TEXTURE_2D_MULTISAMPLE:
		case GL_TEXTURE_1D_ARRAY:
		case GL_PROXY_TEXTURE_1D_ARRAY:
		case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
		case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
		case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
		case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
		case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
		case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
		case GL_TEXTURE_RECTANGLE:
		case GL_PROXY_TEXTURE_RECTANGLE:
			gl.FramebufferTexture2D(target, attachment, imagetarget, reinterpret_cast<GLuint>(&image), level);
			break;
		case GL_TEXTURE_3D:
		case GL_PROXY_TEXTURE_3D:
		/*case GL_TEXTURE_CUBE_MAP:
		case GL_PROXY_TEXTURE_CUBE_MAP:*/
		/*case GL_TEXTURE_CUBE_MAP_ARRAY:
		case GL_PROXY_TEXTURE_CUBE_MAP_ARRAY:*/
		case GL_TEXTURE_2D_ARRAY:
		case GL_PROXY_TEXTURE_2D_ARRAY:
		case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
		case GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY:
			gl.FramebufferTexture3D(target, attachment, imagetarget, reinterpret_cast<GLuint>(&image), level, layer);
			break;
		default:
			break;
		}
	}
	

	GLboolean IsEnabled(GLenum cap) { return gl.IsEnabled(cap); }
	GLboolean IsEnabledi(GLenum cap, GLuint index) { return gl.IsEnabledi(cap, index); }
	void Enable(GLenum cap) { gl.Enable(cap); }
	void Disable(GLenum cap) { gl.Disable(cap); }
	void Enablei(GLenum cap, GLuint index) { gl.Enablei(cap, index); }
	void Disablei(GLenum cap, GLuint index) { gl.Disablei(cap, index); }

	void SetViewport(GLint x, GLint y, GLsizei width, GLsizei height) { gl.Viewport(x, y, width, height); }
	///@note requires gl.Enable(GL_SCISSOR_TEST).
	void SetScissor(GLint x, GLint y, GLsizei width, GLsizei height) { gl.Scissor(x, y, width, height); }
	
	void SetClearColor(GLfloat r, GLfloat g, GLfloat b, GLfloat a) { gl.ClearColor(r, g, b, a); }
	void SetClearDepth(GLclampd depth) { gl.ClearDepth(depth); }
	void SetClearStencil(GLint s) { gl.ClearStencil(s); }
	void Clear(GLbitfield mask) { gl.Clear(mask); }
	
	///@note requires gl.Draw[*](GL_PATCHES, ...).
	void SetPatchi(GLenum pname, GLint param) { gl.PatchParameteri(pname, param); }
	
	GLenum CreatePipeline(const GLpipeline_info& info, GLpipeline*& pipeline) {
		std::vector<GLuint> shaders;
		for (const GLpipeline_shaderstage_info& shaderStage : info.shaderStages) {
			if (!shaderStage.source.empty()) {
				const GLchar* sources[1] = { shaderStage.source.c_str() };
				GLuint        shader     = gl.CreateShader(shaderStage.type);
				gl.ShaderSource(shader, 1, sources, nullptr);
				gl.CompileShader(shader);
				GLint status;
				gl.GetShaderiv(shader, GL_COMPILE_STATUS, &status);
				if (status == GL_FALSE) {
					GLchar message[1024];
					GLsizei message_size = 0;
					gl.GetShaderInfoLog(shader, 1023, &message_size, message);
					message[message_size] = '\0';
					throw std::exception(message);
				}

				shaders.push_back(shader);
			}
		}

		GLuint program = gl.CreateProgram();
		for (GLuint shader : shaders)
			gl.AttachShader(program, shader);
		gl.LinkProgram(program);
		for (GLuint shader : shaders)
			gl.DeleteShader(shader);
		GLint status; 
		gl.GetProgramiv(program, GL_LINK_STATUS, &status);
		if (status == GL_FALSE) {
			GLchar message[1024]; 
			GLsizei message_size = 0;
			gl.GetProgramInfoLog(program, 1023, &message_size, message); 
			message[message_size] = '\0';
			throw std::exception(message);
		}

		GLsizei implBytesize = 
				sizeof(GLuint)
			+ sizeof(GLsizei)
			+ sizeof(GLenum) * (GLsizei)info.drawbuffers.size()
			+ sizeof(GLsizei)
			+ sizeof(GLpipeline_vertexinput_info) * (GLsizei)info.vertexInputElements.size()
			+ sizeof(GLpipeline_inputassembly_info)
			+ sizeof(GLpipeline_tessellation_info)
			+ sizeof(GLpipeline_rasterization_info)
			+ sizeof(GLpipeline_depthstencial_info)
			+ sizeof(GLpipeline_colorblend_info) * (GLsizei)info.drawbuffers.size();
		pipeline = (GLpipeline*)new char[implBytesize];
		char* impl = (char*)pipeline;

		*reinterpret_cast<GLuint*>(impl) = program;
		impl += sizeof(GLuint);

		*reinterpret_cast<GLsizei*>(impl) = (GLsizei)info.drawbuffers.size();
		impl += sizeof(GLsizei);

		for (GLsizei i = 0; i != info.drawbuffers.size(); ++i) {
			*reinterpret_cast<GLenum*>(impl) = info.drawbuffers[i];
			impl += sizeof(GLenum);
		}

		*reinterpret_cast<GLsizei*>(impl) = (GLsizei)info.vertexInputElements.size();
		impl += sizeof(GLsizei);

		for (GLsizei i = 0; i != info.vertexInputElements.size(); ++i) {
			*reinterpret_cast<GLpipeline_vertexinput_info*>(impl) = info.vertexInputElements[i];
			impl += sizeof(GLpipeline_vertexinput_info);
		}

		*reinterpret_cast<GLpipeline_inputassembly_info*>(impl) = info.inputAssembly;
		impl += sizeof(GLpipeline_inputassembly_info);

		*reinterpret_cast<GLpipeline_tessellation_info*>(impl) = info.tessellation;
		impl += sizeof(GLpipeline_tessellation_info);

		*reinterpret_cast<GLpipeline_rasterization_info*>(impl) = info.rasterization;
		impl += sizeof(GLpipeline_rasterization_info);

		*reinterpret_cast<GLpipeline_depthstencial_info*>(impl) = info.depthStencial;
		impl += sizeof(GLpipeline_depthstencial_info);
		
		for (GLsizei i = 0; i != info.drawbuffers.size(); ++i) {
			*reinterpret_cast<GLpipeline_colorblend_info*>(impl) = info.colorBlends[i];
			impl += sizeof(GLpipeline_colorblend_info);
		}

		return gl.GetError();
	}

	GLenum CreateComputePipeline(const std::string& shadersource, GLpipeline*& pipeline) {
		GLuint shader;
		if (!shadersource.empty()) {
			const GLchar* sources[1] = { shadersource.c_str() };
			              shader     = gl.CreateShader(GL_COMPUTE_SHADER);
			gl.ShaderSource(shader, 1, sources, nullptr);
			gl.CompileShader(shader);
			GLint status;
			gl.GetShaderiv(shader, GL_COMPILE_STATUS, &status);
			if (status == GL_FALSE) {
				GLchar message[1024];
				GLsizei message_size = 0;
				gl.GetShaderInfoLog(shader, 1023, &message_size, message);
				message[message_size] = '\0';
				throw std::exception(message);
			}
		} else {
			throw std::exception("empty shadersource");
		}

		GLuint program = gl.CreateProgram();
		gl.AttachShader(program, shader);
		gl.LinkProgram(program);
		gl.DeleteShader(shader);
		GLint status;
		gl.GetProgramiv(program, GL_LINK_STATUS, &status);
		if (status == GL_FALSE) {
			GLchar message[1024]; 
			GLsizei message_size = 0;
			gl.GetProgramInfoLog(program, 1023, &message_size, message); 
			message[message_size] = '\0';
			throw std::exception(message);
		}

		reinterpret_cast<GLuint&>(pipeline) = program;
		return GL_NO_ERROR;
	}

	GLboolean GetProgramFromPipeline(GLpipeline& pipeline, GLprogram* &program) {
		reinterpret_cast<GLuint&>( program = nullptr ) = reinterpret_cast<GLuint&>(pipeline);
		return GL_TRUE;
	}

	void DeletePipeline(GLpipeline*& pipeline) noexcept {
		if (pipeline != nullptr) {
			if (currentPipeline == pipeline) {
				currentPipeline = nullptr;
				currentVertexInputElements = nullptr;
				currentVertexInputCount    = 0;
				currentPrimitiveTopology = GL_POINTS;
			}
			GLuint program = *reinterpret_cast<GLuint*>(pipeline);
			gl.DeleteProgram(program);
			delete[] reinterpret_cast<char*>(pipeline); 
			pipeline = nullptr;
		}
	}

	void DeleteComputePipeline(GLpipeline* &pipeline) {
		if (pipeline != nullptr) {
			if (currentPipeline == pipeline) {
				currentPipeline = nullptr;
				currentVertexInputElements = nullptr;
				currentVertexInputCount    = 0;
				currentPrimitiveTopology = GL_POINTS;
			}
			GLuint& handler = reinterpret_cast<GLuint&>(pipeline);
			gl.DeleteProgram(handler);
			handler = 0;
		}
	}

	/// bind 'program' and 'program.rasterize' and 'program.depthtest' and 'program.blend'.
	///@note pipeline has only one slot.
	void BindPipeline(const GLpipeline& pipeline) {
		if (currentPipeline != &pipeline) {
			currentPipeline = &pipeline;

			const char* impl = reinterpret_cast<const char*>(&pipeline);
			if (impl == nullptr) {
				throw std::exception("pipeline.impl == nullptr");
			}

			GLuint program = *reinterpret_cast<const GLuint *>(impl);
			impl += sizeof(GLuint);
			gl.UseProgram(program);

			GLsizei drawbuffersCount = *reinterpret_cast<const GLsizei *>(impl);
			impl += sizeof(GLsizei);

			const GLenum* drawbuffers = reinterpret_cast<const GLenum *>(impl);
			impl += sizeof(GLenum)*drawbuffersCount;
			if (drawbuffersCount > 1) {
				gl.DrawBuffers(drawbuffersCount, drawbuffers);
			}
			
			currentVertexInputCount = *reinterpret_cast<const GLsizei *>(impl);
			impl += sizeof(GLsizei);
			
			currentVertexInputElements = reinterpret_cast<const GLpipeline_vertexinput_info *>(impl);
			impl += sizeof(GLpipeline_vertexinput_info) * currentVertexInputCount;
			this->ClearVertexAttribs();

			auto inputAssambly = reinterpret_cast<const GLpipeline_inputassembly_info *>(impl);
			impl += sizeof(GLpipeline_inputassembly_info);
			currentPrimitiveTopology = inputAssambly->topology;
			if (inputAssambly->primitiveRestartEnable) {
				gl.Enable(GL_PRIMITIVE_RESTART);
			} else {
				gl.Disable(GL_PRIMITIVE_RESTART);
			}

			auto tessellation  = reinterpret_cast<const GLpipeline_tessellation_info *>(impl);
			impl += sizeof(GLpipeline_tessellation_info);
			if (currentPrimitiveTopology == GL_PATCHES) {
				gl.PatchParameteri(GL_PATCH_VERTICES, tessellation->patchControlPoints);
			}

			auto rasterization = reinterpret_cast<const GLpipeline_rasterization_info *>(impl);
			impl += sizeof(GLpipeline_rasterization_info);
			gl.PolygonMode(GL_FRONT_AND_BACK, rasterization->polygonmode);
			gl.FrontFace(rasterization->winding);
			if (rasterization->cullmode != GL_CULL_MODE_NONE) {
				gl.Enable(GL_CULL_FACE);
				gl.CullFace(rasterization->cullmode);
			} else {
				gl.Disable(GL_CULL_FACE);
			}

			auto depthStencil = reinterpret_cast<const GLpipeline_depthstencial_info *>(impl);
			impl += sizeof(GLpipeline_depthstencial_info);
			if (depthStencil->depthEnable) {
				gl.Enable(GL_DEPTH_TEST);
				gl.DepthMask(depthStencil->depthWritemask);
				gl.DepthRangef(depthStencil->depthRange[0], depthStencil->depthRange[1]);
				gl.DepthFunc(depthStencil->depthFunc);
			} else {
				gl.Disable(GL_DEPTH_TEST);
			}
			if (depthStencil->stencialEnable) {
				gl.Enable(GL_STENCIL_TEST);
				gl.StencilOpSeparate(GL_FRONT, depthStencil->frontface.sfail, depthStencil->frontface.dpfail, depthStencil->frontface.dppass);
				gl.StencilOpSeparate(GL_BACK, depthStencil->backface.sfail, depthStencil->backface.dpfail, depthStencil->backface.dppass);
			} else {
				gl.Disable(GL_STENCIL_TEST);
			}

			for (GLsizei i = 0; i != drawbuffersCount; ++i) {
				auto blendN = reinterpret_cast<const GLpipeline_colorblend_info *>(impl);
				impl += sizeof(GLpipeline_colorblend_info);
				if (blendN->enable) {
					gl.Enablei(GL_BLEND, i);
					gl.BlendEquationi(i, blendN->func);
					gl.BlendFuncSeparatei(i, blendN->sfactorRGB, blendN->dfactorRGB, blendN->sfactorAlpha, blendN->dfactorAlpha);
				} else {
					gl.Disablei(GL_BLEND, i);
				}
			}
		}
	}
	
	/// bind 'program'.
	///@note pipeline has only one slot.
	void BindComputePipeline(const GLpipeline& pipeline) {
		if (currentPipeline != &pipeline) {
			currentPipeline = &pipeline;
			gl.UseProgram(reinterpret_cast<GLuint>(&pipeline));
		}
	}
	
	/// clear all VertexAttributes.
	///@note called in BindPipeline(pipeline).
	void ClearVertexAttribs(GLsizei slot_start = 0) {
		GLsizei max_vertex_attribs;
		gl.GetIntegerv(GL_MAX_VERTEX_ATTRIBS, reinterpret_cast<GLint*>(&max_vertex_attribs));
		for ( ; slot_start != max_vertex_attribs; ++slot_start) {
			gl.DisableVertexAttribArray(slot_start++);
		}
	}
	
	/// bind 'currentVertexInputElements' and 'attributes' into 'layout(location = slot) in [*]' .
	///@pre BindPipeline(pipeline).
	void BindVertexBuffers(GLsizei firstindex, GLsizei num_attributes, const GLvertices_view* attributes) {
		if (currentVertexInputCount < firstindex + num_attributes) {
			throw std::exception("currentVertexInputCount != attributes.count");
		}

		for (GLsizei i = 0; i != num_attributes; ++i) {
			const GLsizei index = firstindex + i;
			const GLpipeline_vertexinput_info& vin = currentVertexInputElements[index];
			const GLvertices_view& attribute = attributes[i];

			gl.BindBuffer(attribute.target, reinterpret_cast<GLuint>(attribute.buffer));
			switch ( vin.type ) {
				case GL_FLOAT:
				case GL_UNSIGNED_BYTE:/* RGB8, RGBA8 */
					gl.VertexAttribPointer(vin.slot, vin.size, vin.type, vin.normalized, attribute.byteStride,
						reinterpret_cast<const void*>(static_cast<intptr_t>(vin.byteOffset + attribute.byteOffset)));
					break;
				// in glsl "#version 300 core\n"
				case GL_SHORT:
				case GL_UNSIGNED_SHORT:
				case GL_INT:
				case GL_UNSIGNED_INT:
					gl.VertexAttribIPointer(vin.slot, vin.size, vin.type, attribute.byteStride,
						reinterpret_cast<const void*>(static_cast<intptr_t>(vin.byteOffset + attribute.byteOffset)));
					break;
				// in glsl "#version 410 core\n"
				case GL_DOUBLE:
					gl.VertexAttribLPointer(vin.slot, vin.size, vin.type, attribute.byteStride,
						reinterpret_cast<const void*>(static_cast<intptr_t>(vin.byteOffset + attribute.byteOffset)));
					break;
				default:
					break;
			}
			gl.EnableVertexAttribArray(vin.slot);
		}
	}

	/// bind 'view' into 'layout(binding = slot) uniform BufferName{...}'.
	///@pre UseProgram(program)|BindPipeline(pipeline).
	void BindBufferView(GLuint slot, const GLbuffer_view& view) {
		gl.BindBufferRange(view.target, slot, reinterpret_cast<GLuint>(view.buffer), 
			view.byteOffset, view.byteLength);
	}

	void BindBufferViews(GLuint first_slot, GLsizei num_views, const GLbuffer_view* views) {
		for (GLsizei i = 0; i != num_views; ++i) {
			gl.BindBufferRange(views[i].target, first_slot+i, reinterpret_cast<GLuint>(views[i].buffer),
				views[i].byteOffset, views[i].byteLength);
		}
	}

	/// bind 'view' into 'layout(binding = slot, view.format) uniform image*'.
	///@pre UseProgram(program)|BindPipeline(pipeline).
	///@note unsupport GL_DEPTH_COMPONENT[*].
	void BindImageView(GLuint slot, const GLimage_view& view) {
		gl.BindImageTexture(slot, reinterpret_cast<GLuint>(view.image), view.level,
			view.layered, view.layer, view.access, view.format);
	}

	void BindImageViews(GLuint first_slot, GLsizei num_views, const GLimage_view* views) {
		for (GLsizei i = 0; i != num_views; ++i) {
			gl.BindImageTexture(first_slot+i, reinterpret_cast<GLuint>(views[i].image), views[i].level,
				views[i].layered, views[i].layer, views[i].access, views[i].format);
		}
	}

	/// bind 'view' into 'layout(binding = slot) uniform sampler*'.
	///@pre UseProgram(program)|BindPipeline(pipeline).
	void BindTextureView(GLuint slot, const GLtexture_view& view) {
		gl.ActiveTexture(GL_TEXTURE0 + slot);
		gl.BindTexture(view.target, reinterpret_cast<GLuint>(view.image));
		if (view.sampler) {
			gl.BindSampler(slot, reinterpret_cast<GLuint>(view.sampler));
		}
	}

	void BindTextureViews(GLuint first_slot, GLsizei num_views, const GLtexture_view* views) {
		for (GLsizei i = 0; i != num_views; ++i) {
			gl.ActiveTexture(GL_TEXTURE0 + first_slot+i);
			gl.BindTexture(views[i].target, reinterpret_cast<GLuint>(views[i].image));
			if (views[i].sampler) {
				gl.BindSampler(first_slot+i, reinterpret_cast<GLuint>(views[i].sampler));
			}
		}
	}

	void DrawArrays(GLint first, GLsizei count) { gl.DrawArrays(currentPrimitiveTopology, first, count); }

	void DrawElements(GLsizei count, const GLindices_view& indices) {
		gl.BindBuffer(GL_ELEMENT_ARRAY_BUFFER, reinterpret_cast<GLuint>(indices.buffer));
		gl.DrawElements(currentPrimitiveTopology, count, indices.type,
			reinterpret_cast<const void*>(static_cast<intptr_t>(indices.byteOffset)));
	}

	void DrawArrays(GLenum mode, GLint first, GLsizei count) { gl.DrawArrays(mode, first, count); }

	void DrawElements(GLenum mode, GLsizei count, const GLindices_view& indices) {
		gl.BindBuffer(GL_ELEMENT_ARRAY_BUFFER, reinterpret_cast<GLuint>(indices.buffer));
		gl.DrawElements(mode, count, indices.type,
			reinterpret_cast<const void*>(static_cast<intptr_t>(indices.byteOffset)));
	}

	void DispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z) { gl.DispatchCompute(num_groups_x, num_groups_y, num_groups_z); }

	//GLenum CreateProgram(const GLuint* shaders, GLsizei shader_count, GLuint& program) {
	//  if (gl.GetError() != GL_NO_ERROR)
	//    throw std::exception("gl.GetError() != GL_NO_ERROR");

	//  program = gl.CreateProgram();
	//  for (GLsizei i = 0; i != shader_count; ++i)
	//    if (gl.IsShader(shaders[i]))
	//      gl.AttachShader(program, shaders[i]);
	//  gl.LinkProgram(program);

	//  GLint status; gl.GetProgramiv(program, GL_LINK_STATUS, &status);
	//  if (status == GL_FALSE) {
	//    char message[1024]; GLsizei message_size = 0;
	//    gl.GetProgramInfoLog(program, 1023, &message_size, message); 
	//    message[message_size] = '\0';
	//    throw std::exception(message);
	//  }
	//  return GL_NO_ERROR;
	//}

	//GLenum CreateProgram(const GLshader* shaders, GLsizei shader_count, GLprogram& program) {
	//  if (gl.GetError() != GL_NO_ERROR)
	//    throw std::exception("gl.GetError() != GL_NO_ERROR");

	//  program.identifier = gl.CreateProgram();
	//  for (GLsizei i = 0; i != shader_count; ++i)
	//    if (gl.IsShader(shaders[i]))
	//      gl.AttachShader(program, shaders[i]);
	//  gl.LinkProgram(program);

	//  GLint status; gl.GetProgramiv(program, GL_LINK_STATUS, &status);
	//  if (status == GL_FALSE) {
	//    char message[1024]; GLsizei message_size = 0;
	//    gl.GetProgramInfoLog(program, 1023, &message_size, message); 
	//    message[message_size] = '\0';
	//    throw std::exception(message);
	//  }

	//  // Get information
	//  GLSLinformation input_info;
	//  GLSLinformation all_info;
	//  for (size_t i = 0; i != shader_count; ++i) {
	//    if ( gl.IsShader(shaders[i]) ) {
	//      GLSLinformation info = GLSLinformation(shaders[i].source);
	//      if (shaders[i].type == GL_VERTEX_SHADER)
	//        input_info.merge_from(info);
	//      all_info.merge_from(std::move(info));
	//    }
	//  }

	//  // Cast input_info to inputlayout
	//  std::vector<GLbufferAccessor> inputlayout;
	//  for (const auto& varying : input_info.varyings) {
	//    assert( varying.second.qualifers.contains("location") );
	//    // layout(location = ..)
	//    GLint location = atoi(varying.second.qualifers.find("location")->second.c_str());
	//    GLenum type = GL_NONE;
	//    GLint size = 0;
	//    GLsizei sizeof_v = 0;
	//    if (varying.second.type == "float") {
	//      type = GL_FLOAT;
	//      size = 1;
	//      sizeof_v = sizeof(float) * size;
	//    } else if (varying.second.type == "double") {
	//      type = GL_DOUBLE;
	//      size = 1;
	//      sizeof_v = sizeof(double) * size;
	//    } else if (varying.second.type == "int") {
	//      type = GL_INT;
	//      size = 1;
	//      sizeof_v = sizeof(int) * size;
	//    } else if (varying.second.type == "vec2") {
	//      type = GL_FLOAT;
	//      size = 2;
	//      sizeof_v = sizeof(float) * size;
	//    } else if (varying.second.type == "vec3") {
	//      type = GL_FLOAT;
	//      size = 3;
	//      sizeof_v = sizeof(float) * size;
	//    } else if (varying.second.type == "vec4") {
	//      type = GL_FLOAT;
	//      size = 4;
	//      sizeof_v = sizeof(float) * size;
	//    } else if (varying.second.type == "dvec2") {
	//      type = GL_DOUBLE;
	//      size = 2;
	//      sizeof_v = sizeof(double) * size;
	//    } else if (varying.second.type == "dvec3") {
	//      type = GL_DOUBLE;
	//      size = 3;
	//      sizeof_v = sizeof(double) * size;
	//    } else if (varying.second.type == "dvec4") {
	//      type = GL_DOUBLE;
	//      size = 4;
	//      sizeof_v = sizeof(double) * size;
	//    } else if (varying.second.type == "mat4") {
	//      type = GL_FLOAT;
	//      size = 16;
	//      sizeof_v = sizeof(float) * size;
	//    } else if (varying.second.type == "dmat4") {
	//      type = GL_DOUBLE;
	//      size = 16;
	//      sizeof_v = sizeof(double) * size;
	//    } else if (varying.second.type == "imat4") {
	//      type = GL_INT;
	//      size = 16;
	//      sizeof_v = sizeof(int) * size;
	//    }

	//    if (inputlayout.size() <= size_t(location))
	//      inputlayout.resize(location + 1);
	//    inputlayout[location].type = type;
	//    inputlayout[location].size = size;
	//  }

	//  program.information = all_info;
	//  program.inputlayout = inputlayout;
	//  return GL_NO_ERROR;
	//}

	//GLenum CreateProgram(
	//  const std::string& vert_str, 
	//  const std::string& tesc_str, const std::string& tese_str,
	//  const std::string& geom_str,
	//  const std::string& frag_str, 
	//  GLprogram& program) {
	//  GLshader shaders[5];
	//  GLsizei shaders_size = 0;
	//  
	//  if (!vert_str.empty())
	//    this->CreateShader(GL_VERTEX_SHADER, vert_str, (shaders[shaders_size++]));
	//  if (!tesc_str.empty())
	//    this->CreateShader(GL_TESS_CONTROL_SHADER, tesc_str, (shaders[shaders_size++]));
	//  if (!tese_str.empty())
	//    this->CreateShader(GL_TESS_EVALUATION_SHADER, tese_str, (shaders[shaders_size++]));
	//  if (!geom_str.empty())
	//    this->CreateShader(GL_GEOMETRY_SHADER, geom_str, (shaders[shaders_size++]));
	//  if (!frag_str.empty())
	//    this->CreateShader(GL_FRAGMENT_SHADER, frag_str, (shaders[shaders_size++]));
	//  if (shaders_size == 0)
	//    return GL_INVALID_OPERATION;
	//  return this->CreateProgram(shaders, shaders_size, program);
	//}

	//GLenum CreateProgram(
	//  const std::filesystem::path& vert_file,
	//  const std::filesystem::path& tesc_file, const std::filesystem::path& tese_file,
	//  const std::filesystem::path& geom_file,
	//  const std::filesystem::path& frag_file, 
	//  GLprogram& program) {
	//  return this->CreateProgram(
	//    vert_file.empty() ? std::string() : GLSLpreprocessor::read_file(vert_file),
	//    tesc_file.empty() ? std::string() : GLSLpreprocessor::read_file(tesc_file),
	//    tese_file.empty() ? std::string() : GLSLpreprocessor::read_file(tese_file),
	//    geom_file.empty() ? std::string() : GLSLpreprocessor::read_file(geom_file),
	//    frag_file.empty() ? std::string() : GLSLpreprocessor::read_file(frag_file), 
	//    program
	//  );
	//}

	//GLenum CreateProgram(const std::string& comp_str, GLprogram& program) {
	//  assert(!comp_str.empty());
	//  GLshader comp;
	//  this->CreateShader(GL_COMPUTE_SHADER, comp_str, comp);
	//  return this->CreateProgram(&comp, 1, program);
	//}

	//GLenum CreateProgram(const std::filesystem::path& comp_file, GLprogram& program) {
	//  return this->CreateProgram(comp_file.empty() ? std::string() : GLSLpreprocessor::read_file(comp_file), program);
	//}

	//void DeleteProgram(GLuint& program) noexcept {
	//  gl.DeleteProgram(program);
	//  program = GL_INVALID_INDEX;
	//}

	//void DeleteProgram(GLprogram& program) noexcept {
	//  gl.DeleteProgram(program.identifier);
	//  program.identifier = GL_INVALID_INDEX;
	//  program.information.clear();
	//  program.inputlayout.clear();
	//}

	//GLboolean IsProgram(GLuint program) noexcept {
	//  return gl.IsProgram(program);
	//}

 
	struct BufferDeleter {
		GLlibrary2* gl;
		void operator()(GLbuffer* buffer) const {
			gl->DeleteBuffer(buffer);
		}
	};
	
	struct ImageDeleter {
		GLlibrary2* gl;
		void operator()(GLimage* buffer) const {
			gl->DeleteImage(buffer);
		}
	};

	struct SamplerDeleter {
		GLlibrary2* gl;
		void operator()(GLsampler* sampler) const {
			gl->DeleteSampler(sampler);
		}
	};

	struct PipelineDeleter {
		GLlibrary2* gl;
		void operator()(GLpipeline* pipeline) const {
			gl->DeletePipeline(pipeline);
		}
	};

	struct FramebufferDeleter {
		GLlibrary2* gl;
		void operator()(GLframebuffer* framebuffer) const {
			gl->DeleteFramebuffer(framebuffer);
		}
	};

	struct ComputePipelineDeleter {
		GLlibrary2* gl;
		void operator()(GLpipeline* pipeline) const {
			gl->DeleteComputePipeline(pipeline);
		}
	};

	inline std::shared_ptr<GLbuffer> CreateBuffer(const GLbuffer_info& info, const void* data,
		GLenum* outerror = nullptr) {
		GLbuffer* buffer = nullptr;
		GLenum error = CreateBuffer(info, data, std::ref(buffer)); if (outerror) { (*outerror) = error; }
		return std::shared_ptr<GLbuffer>(buffer, BufferDeleter{this});
	}

	inline std::shared_ptr<GLimage> CreateImage(const GLimage_info& info, const GLimage_data& data,
		GLenum* outerror = nullptr) {
		GLimage* image = nullptr;
		GLenum error = CreateImage(info, data, std::ref(image)); if (outerror) { (*outerror) = error; }
		return std::shared_ptr<GLimage>(image, ImageDeleter{this});
	}

	inline std::shared_ptr<GLsampler> CreateSampler(const GLsampler_info& info,
		GLenum* outerror = nullptr) {
		GLsampler* sampler = nullptr;
		GLenum error = CreateSampler(info, std::ref(sampler)); if (outerror) { (*outerror) = error; }
		return std::shared_ptr<GLsampler>(sampler, SamplerDeleter{this});
	}

	inline std::shared_ptr<GLpipeline> CreatePipeline(const GLpipeline_info& info, 
		GLenum* outerror = nullptr) {
		GLpipeline* pipeline = nullptr;
		GLenum error = CreatePipeline(info, std::ref(pipeline)); if (outerror) { (*outerror) = error; }
		return std::shared_ptr<GLpipeline>(pipeline, PipelineDeleter{this});
	}
	
	inline std::shared_ptr<GLframebuffer> CreateFramebuffer(GLenum* outerror = nullptr) {
		GLframebuffer* framebuffer = nullptr;
		GLenum error = CreateFramebuffer(std::ref(framebuffer)); if (outerror) { (*outerror) = error; }
		return std::shared_ptr<GLframebuffer>(framebuffer, FramebufferDeleter{this});
	}

	inline std::shared_ptr<GLpipeline> CreateComputePipeline(const std::string& shadersource, 
		GLenum* outerror = nullptr) {
		GLpipeline* pipeline = nullptr;
		GLenum error = CreateComputePipeline(shadersource, std::ref(pipeline)); if (outerror) { (*outerror) = error; }
		return std::shared_ptr<GLpipeline>(pipeline, ComputePipelineDeleter{this});
	}
#pragma warning(default: 4311; default: 4302)
};

#include <algorithm>

struct GLbuffer_maps {
	using buffer = GLbuffer;
	using buffer_view = GLbuffer_view;
	using dataset_type = std::vector<uint8_t>;

	//using _Pair_type = std::pair<buffer_view, dataset_type>;
	struct _Pair_type { buffer_view origin; dataset_type cache; };
	std::vector<_Pair_type> _My_datatable;

	static auto _Keycomp() {
		return [](const _Pair_type& left, const _Pair_type& right) {
			if (left.origin.buffer == right.origin.buffer) {
				return left.origin.byteOffset <= right.origin.byteOffset;
			} else {
				return left.origin.buffer < right.origin.buffer;
			}
		};
	}

	static auto _Keycomp_with_view() {
		return [](const _Pair_type& left, const buffer_view& right) {
			if (left.origin.buffer == right.buffer) {
				return left.origin.byteOffset <= right.byteOffset;
			} else {
				return left.origin.buffer < right.buffer;
			}
		};
	}

	static bool _In_range(buffer_view view, buffer_view range) {
		return view.buffer == range.buffer
			&& range.byteOffset <= view.byteOffset
			&& view.byteOffset+view.byteLength <= range.byteOffset+range.byteLength;
	}

	static bool _Cross_range(buffer_view view, buffer_view range) {
		return view.buffer == range.buffer
			&& view.byteOffset <= range.byteOffset+range.byteLength
			&& range.byteOffset <= view.byteOffset+view.byteLength;
	}

	static buffer_view _Union(buffer_view view0, buffer_view view1) {
		if (view0.target != view1.target || view0.buffer != view1.buffer) 
			{ throw std::exception("_Expand(...)"); }
		GLsizei byteStart = std::min(view0.byteOffset, view1.byteOffset);
		GLsizei byteEnd = std::max(view0.byteOffset + view0.byteLength, view1.byteOffset + view1.byteLength);
		return buffer_view{ view0.target, view0.buffer, byteStart, byteEnd - byteStart };
	}

	static buffer_view _Intersect(buffer_view view0, buffer_view view1) {
		if (view0.target != view1.target || view0.buffer != view1.buffer) 
			{ throw std::exception("_Intersect(...)"); }
		GLsizei byteStart = std::max(view0.byteOffset, view1.byteOffset);
		GLsizei byteEnd = std::max(byteStart, std::min(view0.byteOffset + view0.byteLength, view1.byteOffset + view1.byteLength));
		return buffer_view{ view0.target, view0.buffer, byteStart, byteEnd - byteStart };
	}

	static buffer_view _Difference(buffer_view view0, buffer_view view1) {
		if (view0.target != view1.target || view0.buffer != view1.buffer) 
			{ throw std::exception("_Difference(...)"); }
		GLsizei byteStart = std::max(view0.byteOffset, view1.byteOffset);
		GLsizei byteEnd = std::max(byteStart, std::min(view0.byteOffset + view0.byteLength, view1.byteOffset + view1.byteLength));
		if (byteStart <= view0.byteOffset) {
			GLsizei byteDiffEnd = view0.byteOffset + view0.byteLength;
			GLsizei byteDiffStart = std::min(byteEnd, byteDiffEnd);
			return buffer_view{ view0.target, view0.buffer, byteDiffStart, byteDiffEnd - byteDiffStart };
		} else if (byteEnd >= view0.byteOffset+view0.byteLength) {
			GLsizei byteDiffStart = view0.byteOffset;
			GLsizei byteDiffEnd = std::max(byteStart, byteDiffStart);
			return buffer_view{ view0.target, view0.buffer, byteDiffStart, byteDiffEnd - byteDiffStart };
		} else {
			throw std::exception("_Difference(...)");
		}
	}

	static ptrdiff_t _Distance(buffer_view view0, buffer_view view1) {
		if (_Cross_range(view0, view1)) {
			return 0;
		} else if (view0.byteOffset < view1.byteOffset) {
			return ptrdiff_t(view1.byteOffset - (view0.byteOffset + view0.byteLength));
		} else {
			return -ptrdiff_t(view0.byteOffset - (view1.byteOffset + view1.byteLength));
		}
	}

	/// Get cache dataset assosiated 'view[0,..)'. used for insert|remove variables.
	dataset_type& dataset(buffer_view view) {
		assert( view.byteOffset == 0 );
		auto nextf = std::lower_bound(_My_datatable.begin(), _My_datatable.end(), view, _Keycomp_with_view());
		auto found = _My_datatable.end();
		if (nextf != _My_datatable.begin()) {
			found = std::prev(nextf); 
		}

		/// dataset not found.
		assert( found != _My_datatable.end() );
		/// not inplace view.
		assert( nextf == _My_datatable.end() || nextf->origin.buffer != found->origin.buffer );
		return found->cache;
	}

	/// Get cache dataset assosiated 'buffer'. used for insert|remove variables.
	dataset_type& dataset(buffer& bufferX) {
		GLbuffer_view view;
		view.buffer = &bufferX;
		view.byteOffset = 0;
		return dataset(view);
	}

	/// Get data in buffer[view].
	void* get(GLbuffer_view view) {
		auto nextf = std::lower_bound(_My_datatable.begin(), _My_datatable.end(), view, _Keycomp_with_view());
		if (nextf != _My_datatable.begin()) {
			auto found = std::prev(nextf);
			if (_In_range(view, found->origin)) {
				return &(found->cache[view.byteOffset - found->origin.byteOffset]);
			}
		}

		return nullptr;
	}
	
	/// Get data in buffer[view].
	const void* get(GLbuffer_view view) const {
		return const_cast<GLbuffer_maps&>(*this).get(view);
	}

	/// Whether the 'view' is included.
	bool included(GLbuffer_view view) const {
		auto nextf = std::lower_bound(_My_datatable.begin(), _My_datatable.end(), view, _Keycomp_with_view());
		if (nextf != _My_datatable.begin()) {
			auto found = std::prev(nextf);
			if (_In_range(view, found->origin)) {
				return true;
			}
		}

		return false;
	}

	void readback(GLlibrary2& context, GLbuffer_view view) {
		auto nextf = std::lower_bound(_My_datatable.begin(), _My_datatable.end(), view, _Keycomp_with_view());
		auto found = _My_datatable.end();
		if (nextf != _My_datatable.begin()) {
			found = std::prev(nextf); 
		}

		buffer_view total_view;
		buffer_view beyond_view;
		if (found != _My_datatable.end() && _Cross_range(found->origin, view)) {
			/// combination of [found, ...) into found.
			total_view = _Union(found->origin, view);
		} else if (nextf != _My_datatable.end() && _Cross_range(nextf->origin, view)) {
			/// combination of [nextf, ...) into nextf.
			total_view = _Union(nextf->origin, view);
			found = nextf;
			nextf = std::next(nextf);
		} else {
			/// Readback entire view.
			dataset_type data(view.byteLength);
			if (view.byteLength != 0) {
				context.BindBuffer(GL_ARRAY_BUFFER, *view.buffer);
				context.ReadbackBufferRange(GL_ARRAY_BUFFER, view.byteOffset, view.byteLength, data.data());
			}
			_My_datatable.insert(nextf, {view,data});
			return;
		}
			
		auto lastf = nextf;
		for ( ; lastf != _My_datatable.end() && _Cross_range(total_view, lastf->origin); ++lastf) { 
			if ((beyond_view = _Difference(total_view, lastf->origin)).byteLength != 0)
				{ break; }
			total_view = _Union(total_view, lastf->origin); assert(lastf->origin.byteLength != lastf->cache.size());
		}

		// 1. Map readback part.
		found->cache.resize(total_view.byteLength);
		context.BindBuffer(GL_ARRAY_BUFFER, *view.buffer);
		context.ReadbackBufferRange(GL_ARRAY_BUFFER, view.byteOffset, view.byteLength,
			found->cache.data() + (view.byteOffset - found->origin.byteOffset) );// assert(_Difference(total_view, found->origin) == view);

		// 2. Map beyond part.
		if (beyond_view.byteLength != 0) {
			size_t offset_in_cache = beyond_view.byteOffset - lastf->origin.byteOffset;
			found->cache.insert(found->cache.end(), std::next(lastf->cache.begin(), offset_in_cache), lastf->cache.end());

			total_view = _Union(total_view, lastf->origin); assert(lastf->origin.byteLength != lastf->cache.size());
			++lastf;
		}

		// 3. Update _My_datatable.
		found->origin = total_view;
		_My_datatable.erase(nextf, lastf);
	}

			///
			/// if found->first.buffer == view.buffer
			///		has four cases.
			///		1. independent {found}, {view}, {std::next(found)}.
			///		2. combination {found,view}, {std::next(found)}.
			///		3. combination {found}, {view,std::next(found)}.
			///		4. combination {found,view,std::next(found)}.
			/// else
			///		found is not relation, so remains two cases.
			///		1. independent {view}, {std::next(found)}.
			///		2. combination {view,std::next(found)}.
			/// 
		//	bool combinate_found = found != _My_datatable.end()
		//		&& found->origin.buffer == view.buffer && _Cross_range(view.byteOffset,view.byteLength, found->origin.byteOffset,found->origin.byteLength);
		//	bool combinate_next = nextf != _My_datatable.end()
		//		&& nextf->origin.buffer == view.buffer && _Cross_range(view.byteOffset,view.byteLength, nextf->origin.byteOffset, nextf->origin.byteLength);

		//	if (combinate_found || combinate_next) {
		//		if (combinate_found) {
		//			/// Case2. combination {found,view}, {std::next(found)}.
		//			view = _Union(found->origin, view);
		//			nextf = found;
		//		} else if (combinate_next) {
		//			/// Case3. {found}, {view,std::next(found)}.
		//			GLsizei byteLength = nextf->first.byteOffset - view.byteOffset;
		//			nextf->second.insert(nextf->second.begin(), byteLength, {});
		//			context.BindBuffer(GL_ARRAY_BUFFER, *view.buffer);
		//			context.ReadbackBufferRange(GL_ARRAY_BUFFER, view.byteOffset, byteLength, nextf->second.data());
		//			nextf->first.byteOffset -= byteLength;
		//			nextf->first.byteLength += byteLength;
		//		}

		//		/// Combination {...,std::after(found)...}.
		//		auto afterf = std::next(nextf);
		//		while (afterf != _My_datatable.end() 
		//			&& afterf->first.buffer == view.buffer
		//			&& afterf->first.byteOffset <= view.byteOffset + view.byteLength)
		//		{
		//			GLsizei byteOffset = nextf->first.byteOffset + nextf->first.byteLength;
		//			GLsizei byteLength = afterf->first.byteOffset - byteOffset;
		//			nextf->second.insert(nextf->second.end(), byteLength, {});
		//			context.BindBuffer(GL_ARRAY_BUFFER, *view.buffer);
		//			context.ReadbackBufferRange(GL_ARRAY_BUFFER, byteOffset, byteLength, nextf->second.data() + byteOffset);
		//			nextf->first.byteLength += byteLength;
		//			nextf->second.insert(nextf->second.end(), afterf->second.begin(), afterf->second.end());
		//			nextf->first.byteLength += (GLsizei)afterf->second.size();

		//			++afterf;
		//		}
		//		_My_datatable.erase(std::next(nextf), afterf);
		//		
		//		// Readback part of view's tailing.
		//		if (nextf->first.byteOffset + nextf->first.byteLength < view.byteOffset + view.byteLength) {
		//			GLsizei byteOffset = nextf->first.byteOffset + nextf->first.byteLength;
		//			GLsizei byteLength = view.byteOffset + view.byteLength - byteOffset;
		//			nextf->second.insert(nextf->second.end(), byteLength, {});
		//			context.BindBuffer(GL_ARRAY_BUFFER, *view.buffer);
		//			context.ReadbackBufferRange(GL_ARRAY_BUFFER, byteOffset, byteLength, nextf->second.data() + byteOffset);
		//			nextf->first.byteLength += byteLength;
		//		}

		//		return;
		//	}
		//	else {
		//		/// 1. independent {found}, {view}, {std::next(found)}
		//	}

		//// Readback entire view
		//dataset_type data(view.byteLength);
		//if (view.byteLength != 0) {
		//	context.BindBuffer(GL_ARRAY_BUFFER, *view.buffer);
		//	context.ReadbackBufferRange(GL_ARRAY_BUFFER, view.byteOffset, view.byteLength, data.data());
		//}
		//_My_datatable.insert(nextf, {view,data});
	//}

	void upload(GLlibrary2& context, GLbuffer_view view, bool clear = false) {
		auto nextf = std::lower_bound(_My_datatable.begin(), _My_datatable.end(), view, _Keycomp_with_view());
		auto found = _My_datatable.end();
		if (nextf != _My_datatable.begin()) {
			found = std::prev(nextf); 
		}

		for (auto first = (found != _My_datatable.end() ? found : nextf); 
		first != _My_datatable.end() && _Cross_range(first->origin, view); ++first) {
			assert( first->origin.byteLength == first->cache.size() );
			auto temp = _Intersect(first->origin, view);
			context.BindBuffer(GL_ARRAY_BUFFER, *view.buffer);
			context.UploadBufferRange(GL_ARRAY_BUFFER, temp.byteOffset, temp.byteLength,
				first->cache.data() + (temp.byteOffset - first->origin.byteOffset));
		}

		//GLsizei byteOffset = found->first.byteOffset + found->first.byteLength;
		//GLsizei byteLength;
		//if (found != _My_datatable.end()) {
		//	if (byteOffset > view.byteOffset) {
		//		if (byteOffset >= view.byteOffset + view.byteLength) {
		//			context.BindBuffer(GL_ARRAY_BUFFER, *view.buffer);
		//			context.UploadBufferRange(GL_ARRAY_BUFFER, view.byteOffset, view.byteLength,
		//				found->second.data() + (view.byteOffset - found->first.byteOffset));
		//			// do clear, with seperate.
		//			return;
		//		} else {
		//			context.BindBuffer(GL_ARRAY_BUFFER, *view.buffer);
		//			context.UploadBufferRange(GL_ARRAY_BUFFER, view.byteOffset, byteOffset - view.byteOffset, 
		//				found->second.data() + (view.byteOffset - found->first.byteOffset));
		//			// do clear.
		//		}
		//	}
		//}

		//auto afterf = nextf;
		//while (afterf != _My_datatable.end()
		//	&& afterf->first.buffer == view.buffer
		//	&& afterf->first.byteOffset + afterf->first.byteLength <= view.byteOffset + view.byteLength) {
		//	context.BindBuffer(GL_ARRAY_BUFFER, *view.buffer);
		//	context.UploadBufferRange(GL_ARRAY_BUFFER, afterf->first.byteOffset, afterf->first.byteLength,
		//		afterf->second.data());
		//	++afterf;
		//}
		//// do clear.

		//if (afterf != _My_datatable.end()
		//	&& afterf->first.buffer == view.buffer
		//	&& afterf->first.byteOffset < view.byteOffset + view.byteLength) {
		//	byteLength = view.byteOffset + view.byteLength - afterf->first.byteOffset;
		//	context.BindBuffer(GL_ARRAY_BUFFER, *view.buffer);
		//	context.UploadBufferRange(GL_ARRAY_BUFFER, afterf->first.byteOffset, byteLength,
		//		afterf->second.data());
		//	// do clear, with cut.
		//}
	}

	void upload(GLlibrary2& context, std::shared_ptr<buffer>& resource, GLbuffer_view& view, bool clear = false) {
		if (view.byteOffset != 0) {
			upload(context, view, clear);
			return;// not need realloc.
		}

		auto nextf = std::lower_bound(_My_datatable.begin(), _My_datatable.end(), view, _Keycomp_with_view());
		if (nextf == _My_datatable.begin()) {
			return;// not included.
		}

		auto found = std::prev(nextf);
		if (found->origin.byteLength == found->cache.size()) {
			upload(context, view, clear);
			return;// not need realloc.
		}

		if (context.IsBuffer(*resource)) {
			GLbuffer_info info;
			context.GetBufferInfo(*resource, info);
			assert(found->origin.byteOffset == 0 && found->origin.byteLength == info.byteLength);
		
			GLbuffer_info newinfo = { info.usage, static_cast<GLsizei>(found->cache.size()) };
			context.BindBuffer(GL_ARRAY_BUFFER, *resource);
			context.ReallocBuffer(GL_ARRAY_BUFFER, newinfo, found->cache.data());
			view.byteLength          = newinfo.byteLength;
			found->origin.byteLength = newinfo.byteLength;
			assert( found->origin.buffer == resource.get() );
		} else {
			assert(found->origin.byteOffset == 0);
			GLbuffer_info newinfo = { GL_STATIC_DRAW, static_cast<GLsizei>(found->cache.size()) };
			resource = context.CreateBuffer(newinfo, found->cache.data());
			view.byteLength = newinfo.byteLength;
			view.buffer = resource.get();
			found->origin = view;
			std::sort(_My_datatable.begin(), _My_datatable.end(), _Keycomp());
		}

		//bool need_realloc = false;
		//bool can_realloc = true;
		//GLsizei byteLength = 0;
		//for (auto first = (found != _My_datatable.end() ? (found->origin.buffer == view.buffer ? found : nextf) : nextf); 
		//first != _My_datatable.end() && first->origin.buffer == view.buffer; ++first) {
		//	if ( first->origin.byteLength != first->cache.size() ) 
		//		{ need_realloc = true; }
		//	can_realloc &= (_Cross_range(first->origin, view));
		//	byteLength += first->cache.size();
		//}

		//if (!need_realloc) {
		//	upload(context, view, clear);
		//	return;
		//} else if (can_realloc) {
		//	
		//} else {
		//	abort();
		//}

		//if (found != _My_datatable.end()) {
		//	if (found->first.byteLength != found->second.size()) {
		//		if (!context.IsBuffer(*found->first.buffer)) {
		//			context.CreateBuffer({ GL_STATIC_DRAW, (GLsizei)(found->first.byteOffset + found->second.size()) }, nullptr, std::ref(buffer));
		//			context
		//		}
		//		//assert(view.byteOffset <= found->first.byteOffset);
		//		std::shared_ptr<buffer_type> realloc;
		//		GLbuffer_info                realloc_info = { GL_STATIC_DRAW, (GLsizei)(found->first.byteOffset + found->second.size()) };
		//		if (context.IsBuffer(*found->first.buffer)) {
		//			GLbuffer_info temp;
		//			context.GetBufferInfo(*found->first.buffer, std::ref(temp));
		//			realloc_info.usage = temp.usage;
		//		}
		//		context.CreateBuffer(realloc_info, nullptr, std::ref(realloc));
		//		if (found->first.byteOffset != 0) {
		//			context.CopyBufferRange(*found->first.buffer, 0, *realloc, 0, found->first.byteOffset); }
		//		context.UploadBufferRange(GL_ARRAY_BUFFER, found->first.byteOffset, (GLsizei)found->second.size(), found->second.data());
		//		
		//		buffer.swap(realloc);
		//		view.buffer     = buffer.get();
		//		view.byteLength = realloc_info.byteLength - view.byteOffset;
		//		found->first.buffer     = buffer.get();
		//		found->first.byteLength = realloc_info.byteLength - found->first.byteOffset;
		//		//assert(found->first.byteLength != found->second.size());
		//	} else {
		//		upload(context, view, clear);
		//	}
		//}
	}

	void clear() {
		_My_datatable.clear();
		_My_datatable.shrink_to_fit();
	}

	///!New
	bool have_null() const {
		for (const auto& pair : _My_datatable)
			if (pair.origin.buffer == nullptr)
				return true;
		return false;
	}
};

/// Old Version1, a series template structure, 
///		1.easy use.
///		2.static constraint.
///		but they are unnecessary.
/// 
/// Version2, more simpler and necessary, and 
///		same as modern interface of graphics.
/// 
///	image reousrce 
///		+----------+------------+------------------+----------------+-----------------------+
///		| readonly | read_write |    readonly      |    readonly    |       read_write      |
///		| sampler* |   image*   | uniform variable | uniform buffer | shader storage buffer |
///namespace GLdocument {
///	struct Buffer {
///		char* impl;
///
///		Buffer(size_t length, const void* data) {
///			size_t sizeof_head = sizeof(size_t);
///			impl = new char[sizeof_head + length];
///			memcpy(impl, &length, sizeof_head);
///			if (data != nullptr) {
///				memcpy(impl+sizeof_head, data, length);
///			}
///		}
///
///		/** 
///		 * if we implement Buffer in this form. 
///				 char* buffer;
///				 size_t length;
///				 Buffer(size_t length, const void* data) {
///					 // ...
///				 }
///		 * for concurrency, we must pack another pointer on the outside. 
///		 * then we need one more address search when accessing the buffer. 
///		*/
///	};
///}