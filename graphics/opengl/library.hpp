#pragma once

/// Library of OpenGL
///@license Free 
///@review 2022-5-29 
///@contact Jiang1998Nan@outlook.com 
#define _OPENGL_LIBRARY_

#include "platform.hpp"

#ifndef GLAPIENTRY
#ifdef _WIN64
#define GLAPIENTRY
#else
#define GLAPIENTRY __stdcall
#endif
#endif

using GLvoid     = void ;
using GLboolean  = unsigned char ;
using GLbyte     = signed char ;
using GLubyte    = unsigned char ;
using GLchar     = char ;
using GLshort    = short ;
using GLushort   = unsigned short ;
using GLint      = int ;
using GLuint     = unsigned int ;
using GLfixed    = int;
using GLint64    = long long ;
using GLuint64   = unsigned long long ;
using GLsizei    = unsigned int ;
using GLenum     = unsigned int ;
#ifdef _WIN64
using GLintptr   = /*ptrdiff_t*/__int64;
using GLsizeiptr = /*ptrdiff_t*/__int64;
#else
using GLintptr = /*ptrdiff_t*/int;
using GLsizeiptr = /*ptrdiff_t*/int;
#endif
typedef struct __GLsync* GLsync ;
using GLbitfield = unsigned int ;
using GLfloat    = float ;
using GLclampf   = float ;
using GLdouble   = double ;
using GLclampd   = double ;

enum GL_GetString_names 
{	GL_VENDOR = 0x1F00,
	GL_RENDERER = 0x1F01,
	GL_VERSION = 0x1F02,
	GL_EXTENSIONS = 0x1F03,
	GL_SHADING_LANGUAGE_VERSION = 0x8B8C,
	GL_SPIR_V_EXTENSIONS = 0x9553 /* 'glGetStringi' */ };

enum GL_errors 
{	GL_NO_ERROR = 0,
	GL_INVALID_ENUM = 0x0500,
	GL_INVALID_VALUE = 0x0501,
	GL_INVALID_OPERATION = 0x0502,
	GL_STACK_OVERFLOW = 0x0503,
	GL_STACK_UNDERFLOW = 0x0504,
	GL_OUT_OF_MEMORY = 0x0505,
	GL_INVALID_FRAMEBUFFER_OPERATION = 0x0506 };

enum GL_hit_targets 
{	GL_FRAGMENT_SHADER_DERIVATIVE_HINT = 0x8B8B,
	GL_PERSPECTIVE_CORRECTION_HINT = 0x0C50,
	GL_POINT_SMOOTH_HINT = 0x0C51,
	GL_LINE_SMOOTH_HINT = 0x0C52,
	GL_POLYGON_SMOOTH_HINT = 0x0C53 };

enum GL_context_profiles 
{	GL_CONTEXT_CORE_PROFILE_BIT = 0x00000001,
	GL_CONTEXT_COMPATIBILITY_PROFILE_BIT = 0x00000002 };
enum GL_context_flags 
{	GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT = 0x0001,
	GL_CONTEXT_FLAG_DEBUG_BIT = 0x00000002,
	GL_CONTEXT_FLAG_ROBUST_ACCESS_BIT = 0x00000004,//#version 450
	GL_CONTEXT_FLAG_NO_ERROR_BIT = 0x00000008      /*#version 460*/ };

enum GL_basic_types 
{	GL_BYTE = 0x1400,                     /// #version 110
	GL_UNSIGNED_BYTE = 0x1401,            /// #version 110
	GL_SHORT = 0x1402,                    /// #version 110
	GL_UNSIGNED_SHORT = 0x1403,           /// #version 110
	GL_INT = 0x1404,                      /// #version 110
	GL_UNSIGNED_INT = 0x1405,             /// #version 110
	GL_FLOAT = 0x1406,                    /// #version 110
	GL_R11F_G11F_B10F = 0x8C3A,           /// #version 300
	GL_FLOAT_32_UNSIGNED_INT_24_8_REV = 0x8DAD,/// #version ARB
	GL_DOUBLE = 0x140A,                   /// #version 410
	GL_UNSIGNED_BYTE_3_3_2 = 0x8032,      /// #version 120
	GL_UNSIGNED_BYTE_2_3_3_REV = 0x8362,  /// #version 130
	GL_UNSIGNED_SHORT_4_4_4_4 = 0x8033,   /// #version 120
	GL_UNSIGNED_SHORT_5_5_5_1 = 0x8034,   /// #version 130
	GL_UNSIGNED_SHORT_5_6_5 = 0x8363,     /// #version 130
	GL_UNSIGNED_SHORT_5_6_5_REV = 0x8364, /// #version 130
	GL_UNSIGNED_SHORT_4_4_4_4_REV= 0x8365,/// #version 130
	GL_UNSIGNED_SHORT_1_5_5_5_REV= 0x8366,/// #version 130
	GL_UNSIGNED_INT_8_8_8_8 = 0x8035,     /// #version 120
	GL_UNSIGNED_INT_8_8_8_8_REV = 0x8367, /// #version 130
	GL_UNSIGNED_INT_10_10_10_2 = 0x8036,  /// #version 120
	GL_UNSIGNED_INT_10F_11F_11F_REV = 0x8C3B,/// #version 300
	GL_UNSIGNED_INT_5_9_9_9_REV = 0x8C3E, /// #version 300
	GL_RGB9_E5 = 0x8C3D,                  /// #version 300
	GL_RGB10_A2UI = 0x906F,               /// #version 330
	GL_UNSIGNED_INT_24_8 = 0x84FA,        /// #version ARB
	GL_FIXED = 0x140C,                    /// #version 400
	/// only for GLSL
	GL_FLOAT_VEC2 = 0x8B50, GL_FLOAT_VEC3 = 0x8B51, GL_FLOAT_VEC4 = 0x8B52, 
	GL_FLOAT_MAT2 = 0x8B5A, GL_FLOAT_MAT2x3 = 0x8B65, GL_FLOAT_MAT2x4 = 0x8B66, 
	GL_FLOAT_MAT3x2 = 0x8B67, GL_FLOAT_MAT3 = 0x8B5B, GL_FLOAT_MAT3x4 = 0x8B68,
	GL_FLOAT_MAT4 = 0x8B5C, GL_FLOAT_MAT4x2 = 0x8B69, GL_FLOAT_MAT4x3 = 0x8B6A,
	GL_DOUBLE_VEC2 = 0x8FFC, GL_DOUBLE_VEC3 = 0x8FFD, GL_DOUBLE_VEC4 = 0x8FFE,
	GL_DOUBLE_MAT2 = 0x8F46, GL_DOUBLE_MAT2x3 = 0x8F49, GL_DOUBLE_MAT2x4 = 0x8F4A,
	GL_DOUBLE_MAT3x2 = 0x8F4B, GL_DOUBLE_MAT3 = 0x8F47, GL_DOUBLE_MAT3x4 = 0x8F4C,
	GL_DOUBLE_MAT4x2 = 0x8F4D, GL_DOUBLE_MAT4x3 = 0x8F4E, GL_DOUBLE_MAT4 = 0x8F48,
	GL_BOOL = 0x8B56,
	GL_BOOL_VEC2 = 0x8B57, GL_BOOL_VEC3 = 0x8B58, GL_BOOL_VEC4 = 0x8B59,
	GL_INT_VEC2 = 0x8B53, GL_INT_VEC3 = 0x8B54, GL_INT_VEC4 = 0x8B55,
	GL_UNSIGNED_INT_VEC2 = 0x8DC6,        /// #version 300
	GL_UNSIGNED_INT_VEC3 = 0x8DC7,        /// #version 300
	GL_UNSIGNED_INT_VEC4 = 0x8DC8,        /// #version 300
	/// only for Texture
	GL_SAMPLER_1D = 0x8B5D,               /// #version 200
	GL_SAMPLER_1D_SHADOW = 0x8B61,        /// #version 200
	GL_SAMPLER_1D_ARRAY = 0x8DC0,         /// #version 300
	GL_SAMPLER_1D_ARRAY_SHADOW = 0x8DC3,  /// #version 300
	GL_SAMPLER_2D = 0x8B5E,               /// #version 200
	GL_SAMPLER_2D_SHADOW = 0x8B62,        /// #version 200
	GL_SAMPLER_2D_ARRAY = 0x8DC1,         /// #version 300
	GL_SAMPLER_2D_ARRAY_SHADOW = 0x8DC4,  /// #version 300
	GL_SAMPLER_2D_MULTISAMPLE = 0x9108,   /// GL_ARB_texture_multisample
	GL_SAMPLER_2D_MULTISAMPLE_ARRAY = 0x910B,/// GL_ARB_texture_multisample
	GL_SAMPLER_2D_RECT = 0x8B63,          /// #version 310
	GL_SAMPLER_2D_RECT_SHADOW = 0x8B64,   /// #version 310
	GL_SAMPLER_3D = 0x8B5F,               /// #version 200
	GL_SAMPLER_CUBE = 0x8B60,             /// #version 200
	GL_SAMPLER_CUBE_SHADOW = 0x8DC5,      /// #version 300
	GL_SAMPLER_CUBE_MAP_ARRAY = 0x900C,   /// #version 400
	GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW = 0x900D,/// #version 400
	GL_SAMPLER_BUFFER = 0x8DC2,
	GL_IMAGE_1D = 0x904C,
	GL_IMAGE_1D_ARRAY = 0x9052,
	GL_IMAGE_2D = 0x904D,
	GL_IMAGE_2D_ARRAY = 0x9053,
	GL_IMAGE_2D_MULTISAMPLE = 0x9055,
	GL_IMAGE_2D_MULTISAMPLE_ARRAY = 0x9056,
	GL_IMAGE_2D_RECT = 0x904F,
	GL_IMAGE_3D = 0x904E,
	GL_IMAGE_CUBE = 0x9050,
	GL_IMAGE_CUBE_MAP_ARRAY = 0x9054,
	GL_IMAGE_BUFFER = 0x9051,
	GL_INT_SAMPLER_1D = 0x8DC9,           /// #version 300
	GL_INT_SAMPLER_1D_ARRAY = 0x8DCE,     /// #version 300
	GL_INT_SAMPLER_2D = 0x8DCA,           /// #version 300
	GL_INT_SAMPLER_2D_ARRAY = 0x8DCF,     /// #version 300
	GL_INT_SAMPLER_2D_MULTISAMPLE = 0x9109,/// GL_ARB_texture_multisample
	GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY = 0x910C,/// GL_ARB_texture_multisample
	GL_INT_SAMPLER_2D_RECT = 0x8DCD,
	GL_INT_SAMPLER_3D = 0x8DCB,           /// #version 300
	GL_INT_SAMPLER_CUBE = 0x8DCC,         /// #version 300
	GL_INT_SAMPLER_CUBE_MAP_ARRAY = 0x900E,/// #version 400
	GL_INT_SAMPLER_BUFFER = 0x8DD0,
	GL_INT_IMAGE_1D = 0x9057,
	GL_INT_IMAGE_1D_ARRAY = 0x905D,
	GL_INT_IMAGE_2D = 0x9058,
	GL_INT_IMAGE_2D_ARRAY = 0x905E,
	GL_INT_IMAGE_2D_MULTISAMPLE = 0x9060,
	GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY = 0x9061,
	GL_INT_IMAGE_2D_RECT = 0x905A,
	GL_INT_IMAGE_3D = 0x9059,
	GL_INT_IMAGE_CUBE = 0x905B,
	GL_INT_IMAGE_CUBE_MAP_ARRAY = 0x905F,
	GL_INT_IMAGE_BUFFER = 0x905C,
	GL_UNSIGNED_INT_SAMPLER_1D = 0x8DD1,      /// #version 300
	GL_UNSIGNED_INT_SAMPLER_1D_ARRAY = 0x8DD6,/// #version 300
	GL_UNSIGNED_INT_SAMPLER_2D = 0x8DD2,      /// #version 300
	GL_UNSIGNED_INT_SAMPLER_2D_ARRAY = 0x8DD7,/// #version 300
	GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE = 0x910A,/// GL_ARB_texture_multisample
	GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY = 0x910D,/// GL_ARB_texture_multisample
	GL_UNSIGNED_INT_SAMPLER_2D_RECT = 0x8DD5,
	GL_UNSIGNED_INT_SAMPLER_3D = 0x8DD3,      /// #version 300
	GL_UNSIGNED_INT_SAMPLER_CUBE = 0x8DD4,    /// #version 300
	GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY = 0x900F,/// #version 400
	GL_UNSIGNED_INT_SAMPLER_BUFFER = 0x8DD8,
	GL_UNSIGNED_INT_IMAGE_1D = 0x9062,
	GL_UNSIGNED_INT_IMAGE_1D_ARRAY = 0x9068,
	GL_UNSIGNED_INT_IMAGE_2D = 0x9063,
	GL_UNSIGNED_INT_IMAGE_2D_ARRAY = 0x9069,
	GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE = 0x906B,
	GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY = 0x906C,
	GL_UNSIGNED_INT_IMAGE_2D_RECT = 0x9065,
	GL_UNSIGNED_INT_IMAGE_3D = 0x9064,
	GL_UNSIGNED_INT_IMAGE_CUBE = 0x9066,
	GL_UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY = 0x906A,
	GL_UNSIGNED_INT_IMAGE_BUFFER = 0x9067,
	GL_UNSIGNED_INT_ATOMIC_COUNTER = 0x92DB };

enum GL_constants 
{	GL_NONE          = 0,
	GL_INVALID_INDEX = 0xFFFFFFFFu,
	GL_FALSE         = 0,
	GL_TRUE          = 1 };

enum GL_compare_ops 
{	GL_LEQUAL   = 0x0203,
	GL_GEQUAL   = 0x0206,
	GL_LESS     = 0x0201,
	GL_GREATER  = 0x0204,
	GL_EQUAL    = 0x0202,
	GL_NOTEQUAL = 0x0205,
	GL_ALWAYS   = 0x0207,
	GL_NEVER    = 0x0200 };

enum GL_logical_ops 
{	GL_CLEAR       = 0x1500,
	GL_AND         = 0x1501,
	GL_AND_REVERSE = 0x1502,
	GL_COPY        = 0x1503,
	GL_AND_INVERTED = 0x1504,
	GL_NOOP        = 0x1505,
	GL_XOR         = 0x1506,
	GL_OR          = 0x1507,
	GL_NOR         = 0x1508,
	GL_EQUIV       = 0x1509,
	GL_INVERT      = 0x150A,
	GL_OR_REVERSE  = 0x150B,
	GL_COPY_INVERTED = 0x150C,
	GL_OR_INVERTED = 0x150D,
	GL_NAND        = 0x150E,
	GL_SET         = 0x150F };

enum GL_stencil_ops 
{ GL_KEEP   = 0x1E00,
	/*GL_ZERO = 0,*/
	GL_REPLACE = 0x1E01,
	GL_INCR   = 0x1E02,
	GL_DECR   = 0x1E03,
	/*GL_INVERT = 0x150A*/ };

enum GL_accum_ops 
{	GL_ACCUM  = 0x0100,
	GL_LOAD   = 0x0101,
	GL_RETURN = 0x0102,
	GL_MULT   = 0x0103,
	GL_ADD    = 0x0104 };

enum GL_faces 
{	GL_BACK  = 0x0405,
	GL_FRONT = 0x0404,
	GL_FRONT_AND_BACK = 0x0408 };

enum GL_origin 
{	GL_LOWER_LEFT = 0x8CA1,
	GL_UPPER_LEFT = 0x8CA2 };

enum GL_access 
{	GL_READ_ONLY = 0x88B8,
	GL_WRITE_ONLY = 0x88B9,
	GL_READ_WRITE = 0x88BA };

enum GL_range_access 
{	GL_MAP_READ_BIT = 0x0001,
	GL_MAP_WRITE_BIT = 0x0002,
	GL_MAP_INVALIDATE_RANGE_BIT = 0x0004,
	GL_MAP_INVALIDATE_BUFFER_BIT = 0x0008,
	GL_MAP_FLUSH_EXPLICIT_BIT = 0x0010,
	GL_MAP_UNSYNCHRONIZED_BIT = 0x0020,
	GL_MAP_PERSISTENT_BIT = 0x00000040,//#version 440
	GL_MAP_COHERENT_BIT = 0x00000080   /*#version 440*/ }; 

enum GLcapability 
{	GL_DEBUG_OUTPUT = 0x92E0,
	GL_DEBUG_OUTPUT_SYNCHRONOUS = 0x8242,
/* Clipping */
	GL_CLIP_PLANE0 = 0x3000,
	GL_CLIP_PLANE1 = 0x3001,
	GL_CLIP_PLANE2 = 0x3002,
	GL_CLIP_PLANE3 = 0x3003,
	GL_CLIP_PLANE4 = 0x3004,
	GL_CLIP_PLANE5 = 0x3005,
/* Rasterization */
	GL_CULL_FACE   = 0x0B44, GL_VERTEX_PROGRAM_TWO_SIDE = 0x8643,
	GL_ALPHA_TEST  = 0x0BC0,
	GL_DEPTH_TEST  = 0x0B71,
	GL_DEPTH_CLAMP = 0x864F,
	GL_STENCIL_TEST= 0x0B90,
	GL_SCISSOR_TEST= 0x0C11,
	GL_LINE_SMOOTH = 0x0B20,
	GL_POINT_SPRITE = 0x8861,
	GL_POINT_SMOOTH = 0x0B10,
	GL_PROGRAM_POINT_SIZE = 0x8642, GL_VERTEX_PROGRAM_POINT_SIZE = GL_PROGRAM_POINT_SIZE,
	GL_POLYGON_SMOOTH      = 0x0B41,
	GL_POLYGON_OFFSET_FILL = 0x8037,
	GL_POLYGON_OFFSET_POINT = 0x2A01,
	GL_POLYGON_OFFSET_LINE = 0x2A02,
	GL_MULTISAMPLE = 0x809D,
	GL_SAMPLE_SHADING = 0x8C36,
	GL_RASTERIZER_DISCARD = 0x8C89,
/* Final */
	GL_BLEND = 0x0BE2,
	GL_INDEX_LOGIC_OP = 0x0BF1,
	GL_COLOR_LOGIC_OP = 0x0BF2,
	GL_TEXTURE_CUBE_MAP_SEAMLESS = 0x884F /* GL_ARB_seamless_cube_map */ };

enum GLpname /* : public GLcapability, GLpixelstoreParameterName */ 
{	GL_MAJOR_VERSION                  = 0x821B,
	GL_MINOR_VERSION                  = 0x821C,
	GL_NUM_EXTENSIONS                 = 0x821D,
	GL_NUM_SHADING_LANGUAGE_VERSIONS  = 0x82E9,
	GL_CONTEXT_FLAGS                  = 0x821E,
	GL_CONTEXT_PROFILE_MASK           = 0x9126,
	GL_SHADER_COMPILER                = 0x8DFA,
	GL_NUM_SPIR_V_EXTENSIONS          = 0x9554,

/* Debug */

	GL_MAX_DEBUG_GROUP_STACK_DEPTH    = 0x826C,
	GL_DEBUG_GROUP_STACK_DEPTH        = 0x826D,
	GL_VIEWPORT_BOUNDS_RANGE          = 0x825D,
	GL_MAX_ELEMENTS_VERTICES          = 0x80E8,/* <= end - start + 1 in glDrawRangeElements(..,start,end,..), then the call may operate at reduced performance. */
	GL_MAX_ELEMENTS_INDICES           = 0x80E9,/* <= count in glDrawRangeElements(..,count,..), then the call may operate at reduced performance. */
	GL_MAX_TRANSFORM_FEEDBACK_BUFFERS = 0x8E70,
	GL_MAX_VERTEX_STREAMS             = 0x8E71,
	GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS = 0x8C80,
	GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS = 0x8C8A,
	GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS = 0x8C8B,
	GL_PRIMITIVE_RESTART              = 0x8F9D,/* not response */
	GL_PRIMITIVE_RESTART_INDEX        = 0x8F9E,/* inverse of gl.PrimitiveRestartIndex(index) */
	GL_PRIMITIVE_RESTART_FOR_PATCHES_SUPPORTED = 0x8221,
	GL_MAX_SERVER_WAIT_TIMEOUT        = 0x9111,/* gl.WaitSync(..,MAX timeout) */

/* Buffer */

	GL_ARRAY_BUFFER_BINDING           = 0x8894,/* inverse of gl.BindBuffer(..,object) */

	GL_ELEMENT_ARRAY_BUFFER_BINDING   = 0x8895,/* inverse of gl.BindBuffer(..,object) */

	GL_PIXEL_PACK_BUFFER_BINDING      = 0x88ED,/* inverse of gl.BindBuffer(..,object) */

	GL_PIXEL_UNPACK_BUFFER_BINDING    = 0x88EF,/* inverse of gl.BindBuffer(..,object) */

	GL_TRANSFORM_FEEDBACK_BUFFER_BINDING = 0x8C8F,/* inverse of gl.BindBuffer(..,object) */
	GL_TRANSFORM_FEEDBACK_BUFFER_START = 0x8C84,
	GL_TRANSFORM_FEEDBACK_BUFFER_SIZE = 0x8C85,
	GL_TEXTURE_BUFFER_DATA_STORE_BINDING = 0x8C2D,/* inverse of gl.TexBuffer(...,object) */

	GL_UNIFORM_BUFFER_BINDING         = 0x8A28,/* inverse of gl.BindBuffer(..,object) */
	GL_UNIFORM_BUFFER_START           = 0x8A29,
	GL_UNIFORM_BUFFER_SIZE            = 0x8A2A,
	GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT = 0x8A34,

	GL_MAX_COMBINED_SHADER_OUTPUT_RESOURCES = 0x8F39,
	GL_SHADER_STORAGE_BUFFER_BINDING = 0x90D3,
	GL_SHADER_STORAGE_BUFFER_START = 0x90D4,
	GL_SHADER_STORAGE_BUFFER_SIZE = 0x90D5,
	GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS = 0x90DD,
	GL_MAX_SHADER_STORAGE_BLOCK_SIZE = 0x90DE,
	GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT = 0x90DF,

//GL_ATOMIC_COUNTER_BUFFER_BINDING  = 0x92C1,/* inverse of gl.BindBuffer(..,object) */
	GL_ATOMIC_COUNTER_BUFFER_START    = 0x92C2,
	GL_ATOMIC_COUNTER_BUFFER_SIZE     = 0x92C3,
	GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS = 0x92DC,

	GL_DISPATCH_INDIRECT_BUFFER_BINDING = 0x90EF,/* inverse of gl.BindBuffer(..,object) */

	GL_QUERY_BUFFER_BINDING           = 0x9193,

	GL_MIN_MAP_BUFFER_ALIGNMENT       = 0x90BC,
	GL_COPY_READ_BUFFER_BINDING       = 0x8F36,/* inverse of gl.BindBuffer(..,object) */
	GL_COPY_WRITE_BUFFER_BINDING      = 0x8F37,/* inverse of gl.BindBuffer(..,object) */

/* Texture */

	GL_MAX_TEXTURE_SIZE               = 0x0D33,
	GL_MAX_3D_TEXTURE_SIZE            = 0x8073,
	GL_MAX_TEXTURE_BUFFER_SIZE        = 0x8C2B,
	GL_MAX_RECTANGLE_TEXTURE_SIZE     = 0x84F8,
	GL_MAX_CUBE_MAP_TEXTURE_SIZE      = 0x851C,
	GL_MAX_TEXTURE_UNITS              = 0x84E2,
	GL_MAX_IMAGE_UNITS                = 0x8F38,
	GL_MAX_TEXTURE_LOD_BIAS           = 0x84FD,
	GL_MAX_ARRAY_TEXTURE_LAYERS       = 0x88FF,
	GL_NUM_COMPRESSED_TEXTURE_FORMATS = 0x86A2,
	GL_COMPRESSED_TEXTURE_FORMATS     = 0x86A3,
	GL_MAX_TEXTURE_COORDS             = 0x8871,
	GL_MAX_TEXTURE_IMAGE_UNITS        = 0x8872,
	GL_ACTIVE_TEXTURE                 = 0x84E0,/* inverse of gl.ActiveTexture(texture) */
	GL_CLIENT_ACTIVE_TEXTURE          = 0x84E1,/* inverse of gl.ActiveClientTexture(texture) */
	GL_TEXTURE_BINDING_1D             = 0x8068,/* inverse of gl.BindTexture(..,object) */
	GL_TEXTURE_BINDING_2D             = 0x8069,/* inverse of gl.BindTexture(..,object) */
	GL_TEXTURE_BINDING_1D_ARRAY       = 0x8C1C,/* inverse of gl.BindTexture(..,object) */
	GL_TEXTURE_BINDING_BUFFER         = 0x8C2C,/* inverse of gl.BindTexture(GL_TEXTURE_BUFFER, object); */
	GL_TEXTURE_BINDING_2D_ARRAY       = 0x8C1D,/* inverse of gl.BindTexture(..,object) */
	GL_TEXTURE_BINDING_3D             = 0x806A,/* inverse of gl.BindTexture(..,object) */
	GL_TEXTURE_BINDING_CUBE_MAP       = 0x8514,/* inverse of gl.BindTexture(..,object) */
	GL_TEXTURE_BINDING_RECTANGLE      = 0x84F6,/* inverse of gl.BindTexture(..,object) */
	GL_TEXTURE_BINDING_CUBE_MAP_ARRAY = 0x900A,/* inverse of gl.BindTexture(..,object) */
	GL_TEXTURE_BINDING_2D_MULTISAMPLE = 0x9104,/* inverse of gl.BindTexture(..,object) */
	GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY = 0x9105,/* inverse of gl.BindTexture(..,object) */
	GL_TEXTURE_BUFFER_BINDING         = 0x8C2A,
	GL_SAMPLER_BINDING                = 0x8919,/* inverse of gl.BindSampler(..,object) */
	GL_SAMPLE_POSITION                = 0x8E50,
	GL_SAMPLE_MASK                    = 0x8E51,
	GL_SAMPLE_MASK_VALUE              = 0x8E52,
	GL_MAX_SAMPLE_MASK_WORDS          = 0x8E59,
	GL_TEXTURE_FIXED_SAMPLE_LOCATIONS = 0x9107,
	GL_MAX_SAMPLES                    = 0x8D57,
	GL_MAX_COLOR_TEXTURE_SAMPLES      = 0x910E,
	GL_MAX_DEPTH_TEXTURE_SAMPLES      = 0x910F,
	GL_MAX_INTEGER_SAMPLES            = 0x9110,

	GL_IMAGE_BINDING_NAME = 0x8F3A,
	GL_IMAGE_BINDING_LEVEL = 0x8F3B,
	GL_IMAGE_BINDING_LAYERED = 0x8F3C,
	GL_IMAGE_BINDING_LAYER = 0x8F3D,
	GL_IMAGE_BINDING_ACCESS = 0x8F3E,
	GL_IMAGE_BINDING_FORMAT = 0x906E,
	GL_MAX_IMAGE_SAMPLES = 0x906D,

/* Render Target */

	GL_MAX_RENDERBUFFER_SIZE          = 0x84E8,
	GL_MAX_COLOR_ATTACHMENTS          = 0x8CDF,
	GL_MAX_FRAMEBUFFER_WIDTH          = 0x9315,
	GL_MAX_FRAMEBUFFER_HEIGHT         = 0x9316,
	GL_MAX_FRAMEBUFFER_LAYERS         = 0x9317,
	GL_MAX_FRAMEBUFFER_SAMPLES        = 0x9318,
	GL_FRAMEBUFFER_BINDING            = 0x8CA6,/* inverse of gl.BindFramebuffer(GL_FRAMEBUFFER_BINDING, object) */
	GL_DRAW_FRAMEBUFFER_BINDING       = GL_FRAMEBUFFER_BINDING,/* inverse of gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER_BINDING, object) */
	GL_READ_FRAMEBUFFER_BINDING       = 0x8CAA,/* inverse of gl.BindFramebuffer(GL_READ_FRAMEBUFFER_BINDING, object) */
	GL_RENDERBUFFER_BINDING           = 0x8CA7,/* inverse of gl.BindRenderbuffer(GL_RENDERBUFFER_BINDING, object) */\
	GL_MAX_DRAW_BUFFERS               = 0x8824,

/* PipelineStage */

	GL_SCISSOR_BOX                    = 0x0C10,/* inverse of gl.Scissor(box) */
	GL_VIEWPORT                       = 0x0BA2,/* inverse of gl.Viewport(viewport) */
	GL_MAX_VIEWPORT_DIMS              = 0x0D3A,
	GL_LOGIC_OP_MODE                  = 0x0BF0,/* inverse of gl.glLogicOp(op) */
	GL_COLOR_CLEAR_VALUE              = 0x0C22,/* inverse of gl.ClearColor(value) */
	GL_COLOR_WRITEMASK                = 0x0C23,/* inverse of gl.ColorMask(mask) */
	GL_ALPHA_TEST_FUNC                = 0x0BC1,
	GL_ALPHA_TEST_REF                 = 0x0BC2,
	GL_DEPTH_WRITEMASK                = 0x0B72,/* inverse of gl.DepthWriteMask(mask) */
	GL_DEPTH_CLEAR_VALUE              = 0x0B73,/* inverse of gl.ClearDepth(value) */
	GL_DEPTH_FUNC                     = 0x0B74,/* inverse of gl.DepthFunc(compare_op) */
	GL_DEPTH_RANGE                    = 0x0B70,/* inverse of gl.DepthClamp(range), need gl.Enable(GL_DEPTH_CLAMP) */
	GL_STENCIL_CLEAR_VALUE            = 0x0B91,/* inverse of gl.ClearStencil(value) */
	GL_STENCIL_FUNC                   = 0x0B92,/* inverse of gl.StencilFunc(func,...) */
	GL_STENCIL_REF                    = 0x0B97,/* inverse of gl.StencilFunc(..,ref,...) */
	GL_STENCIL_BACK_FUNC              = 0x8800,
	GL_STENCIL_BACK_REF               = 0x8CA3,
	GL_STENCIL_BACK_FAIL              = 0x8801,
	GL_STENCIL_BACK_PASS_DEPTH_FAIL   = 0x8802,
	GL_STENCIL_BACK_PASS_DEPTH_PASS   = 0x8803,
	GL_STENCIL_VALUE_MASK             = 0x0B93,/* inverse of gl.StencilFunc(..,mask) */
	GL_STENCIL_WRITEMASK              = 0x0B98,/* inverse of gl.StencilMask(mask) */
	GL_STENCIL_BACK_VALUE_MASK        = 0x8CA4,
	GL_STENCIL_BACK_WRITEMASK         = 0x8CA5,
	GL_BLEND_COLOR                    = 0x8005,/* inverse of gl.BlendColor(color) */
	GL_BLEND_DST_RGB                  = 0x80C8,/* inverse of gl.BlendFunc(...,dfactor) */
	GL_BLEND_SRC_RGB                  = 0x80C9,/* inverse of gl.BlendFunc(sfactor,...) */
	GL_BLEND_DST_ALPHA                = 0x80CA,
	GL_BLEND_SRC_ALPHA                = 0x80CB,
	GL_BLEND_EQUATION                 = 0x8009,/* inverse of gl.BlendEquation(equation) */
	GL_BLEND_EQUATION_RGB             = GL_BLEND_EQUATION,
	GL_BLEND_EQUATION_ALPHA           = 0x883D,
	GL_SAMPLE_COVERAGE_VALUE          = 0x80AA,/* inverse of gl.SampleCoverage(value,..) */
	GL_SAMPLE_COVERAGE_INVERT         = 0x80AB,/* inverse of gl.SampleCoverage(..,inverst) */
	GL_FRONT_FACE                     = 0x0B46,/* inverse of gl.FrontFace(mode) */
	GL_MAX_CLIP_PLANES                = 0x0D32,
	GL_MAX_CLIP_DISTANCES             = GL_MAX_CLIP_PLANES,
	GL_CLIP_DISTANCE0                 = GL_CLIP_PLANE0,/* inverse of gl.ClipPlane(GL_CLIP_DISTANCE0, distance) */
	GL_CLIP_DISTANCE1                 = GL_CLIP_PLANE1,/* inverse of gl.ClipPlane(GL_CLIP_DISTANCE1, distance) */
	GL_CLIP_DISTANCE2                 = GL_CLIP_PLANE2,/* inverse of gl.ClipPlane(GL_CLIP_DISTANCE2, distance) */
	GL_CLIP_DISTANCE3                 = GL_CLIP_PLANE3,/* inverse of gl.ClipPlane(GL_CLIP_DISTANCE3, distance) */
	GL_CLIP_DISTANCE4                 = GL_CLIP_PLANE4,/* inverse of gl.ClipPlane(GL_CLIP_DISTANCE4, distance) */
	GL_CLIP_DISTANCE5                 = GL_CLIP_PLANE5,/* inverse of gl.ClipPlane(GL_CLIP_DISTANCE5, distance) */
	GL_CULL_FACE_MODE                 = 0x0B45,/* inverse of gl.CullFace(mode) */
	GL_POINT_SIZE                     = 0x0B11,/* inverse of gl.PointSize(size) */
	GL_POINT_SIZE_RANGE               = 0x0B12,
	GL_POINT_SIZE_GRANULARITY         = 0x0B13,
	GL_POINT_SIZE_MIN                 = 0x8126,/* inverse of gl.PointParameterf(GL_POINT_SIZE_MIN,value) */
	GL_POINT_SIZE_MAX                 = 0x8127,/* inverse of gl.PointParameterf(GL_POINT_SIZE_MAX,value) */
	GL_POINT_FADE_THRESHOLD_SIZE      = 0x8128,/* inverse of gl.PointParameterf(GL_POINT_FADE_THRESHOLD_SIZE,value) */
	GL_POINT_DISTANCE_ATTENUATION     = 0x8129,/* inverse of gl.PointParameterf(GL_POINT_DISTANCE_ATTENUATION,value) */
	GL_POINT_SPRITE_COORD_ORIGIN      = 0x8CA0,/* inverse of gl.PointParameteri(GL_POINT_SPRITE_COORD_ORIGIN,value) */
	GL_LINE_WIDTH                     = 0x0B21,/* inverse of gl.LineWidth(width) */
	GL_LINE_WIDTH_RANGE               = 0x0B22,
	GL_LINE_WIDTH_GRANULARITY         = 0x0B23,
	GL_ALIASED_LINE_WIDTH_RANGE       = 0x846E,
	GL_SMOOTH_LINE_WIDTH_RANGE        = GL_LINE_WIDTH_RANGE,
	GL_SMOOTH_LINE_WIDTH_GRANULARITY  = GL_LINE_WIDTH_GRANULARITY,
	GL_POLYGON_MODE                   = 0x0B40,/* inverse of gl.PolygonMode(..,mode) */
	GL_POLYGON_OFFSET_FACTOR          = 0x8038,/* inverse of gl.PolygonOffset(factor,..) */
	GL_POLYGON_OFFSET_UNITS           = 0x2A00,/* inverse of gl.PolygonOffset(..,units) */
	GL_TRANSFORM_FEEDBACK_BUFFER_PAUSED = 0x8E23,
	GL_TRANSFORM_FEEDBACK_BUFFER_ACTIVE = 0x8E24,
	GL_TRANSFORM_FEEDBACK_PAUSED      = GL_TRANSFORM_FEEDBACK_BUFFER_PAUSED,
	GL_TRANSFORM_FEEDBACK_ACTIVE      = GL_TRANSFORM_FEEDBACK_BUFFER_ACTIVE,
	GL_TRANSFORM_FEEDBACK_BINDING       = 0x8E25,/* inverse of gl.BindTransformFeedback(..,object) */
	GL_IMPLEMENTATION_COLOR_READ_TYPE = 0x8B9A,/* type by gl.RealPixel(...) from currently bound framebuffer */
	GL_IMPLEMENTATION_COLOR_READ_FORMAT = 0x8B9B,/* format by gl.RealPixel(...) from currently bound framebuffer */

/* Program */

	GL_CURRENT_PROGRAM                = 0x8B8D,/* inverse of gl.UseProgram(program) */
	GL_VERTEX_ARRAY_BINDING           = 0x85B5,/* inverse of gl.BindArray(array) */
	GL_NUM_PROGRAM_BINARY_FORMATS     = 0x87FE,
	GL_PROGRAM_BINARY_FORMATS         = 0x87FF,
	GL_SHADER_BINARY_FORMATS          = 0x8DF8,
	GL_NUM_SHADER_BINARY_FORMATS      = 0x8DF9,
	GL_MAX_UNIFORM_LOCATIONS          = 0x826E,
	GL_MAX_VARYING_VECTORS            = 0x8DFC,
	GL_MAX_VERTEX_ATTRIBS             = 0x8869,
	GL_MAX_VERTEX_ATTRIB_STRIDE       = 0x82E5,
	GL_MAX_VERTEX_ATTRIB_RELATIVE_OFFSET = 0x82D9,
	GL_MAX_VERTEX_ATTRIB_BINDINGS     = 0x82DA,
	GL_MAX_VERTEX_UNIFORM_VECTORS     = 0x8DFB,
	GL_MAX_VERTEX_UNIFORM_COMPONENTS  = 0x8B4A,
	GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS = 0x8B4C,
	GL_MAX_VERTEX_OUTPUT_COMPONENTS   = 0x9122,
	GL_MAX_VERTEX_ATOMIC_COUNTER_BUFFERS = 0x92CC,
	GL_MAX_VERTEX_ATOMIC_COUNTERS = 0x92D2,
	GL_MAX_PATCH_VERTICES             = 0x8E7D,
	GL_MAX_TESS_GEN_LEVEL             = 0x8E7E,
	GL_MAX_TESS_PATCH_COMPONENTS          = 0x8E84,
	GL_MAX_TESS_CONTROL_UNIFORM_COMPONENTS = 0x8E7F,
	GL_MAX_TESS_CONTROL_UNIFORM_BLOCKS     = 0x8E89,
	GL_MAX_TESS_CONTROL_TEXTURE_IMAGE_UNITS = 0x8E81,
	GL_MAX_TESS_CONTROL_INPUT_COMPONENTS = 0x886C,
	GL_MAX_TESS_CONTROL_OUTPUT_COMPONENTS = 0x8E83,
	GL_MAX_TESS_CONTROL_TOTAL_OUTPUT_COMPONENTS = 0x8E85,
	GL_MAX_TESS_CONTROL_ATOMIC_COUNTER_BUFFERS = 0x92CD,
	GL_MAX_TESS_CONTROL_ATOMIC_COUNTERS = 0x92D3,
	GL_MAX_TESS_EVALUATION_UNIFORM_COMPONENTS = 0x8E80,
	GL_MAX_TESS_EVALUATION_UNIFORM_BLOCKS = 0x8E8A,
	GL_MAX_TESS_EVALUATION_TEXTURE_IMAGE_UNITS = 0x8E82,
	GL_MAX_TESS_EVALUATION_INPUT_COMPONENTS = 0x886D,
	GL_MAX_TESS_EVALUATION_OUTPUT_COMPONENTS = 0x8E86,
	GL_MAX_TESS_EVALUATION_ATOMIC_COUNTER_BUFFERS = 0x92CE,
	GL_MAX_TESS_EVALUATION_ATOMIC_COUNTERS = 0x92D4,
	GL_MAX_COMBINED_TESS_CONTROL_UNIFORM_COMPONENTS = 0x8E1E,
	GL_MAX_COMBINED_TESS_EVALUATION_UNIFORM_COMPONENTS = 0x8E1F,
	GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS = 0x8C29,
	GL_MAX_GEOMETRY_UNIFORM_COMPONENTS = 0x8DDF,
	GL_MAX_GEOMETRY_OUTPUT_VERTICES   = 0x8DE0,
	GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS = 0x8DE1,
	GL_MAX_GEOMETRY_INPUT_COMPONENTS  = 0x9123,
	GL_MAX_GEOMETRY_OUTPUT_COMPONENTS =0x9124,
	GL_MAX_GEOMETRY_ATOMIC_COUNTER_BUFFERS = 0x92CF,
	GL_MAX_GEOMETRY_ATOMIC_COUNTERS = 0x92D5,
	GL_MAX_FRAGMENT_UNIFORM_VECTORS   = 0x8DFD,
	GL_MAX_FRAGMENT_UNIFORM_COMPONENTS= 0x8B49,
	GL_MAX_FRAGMENT_INPUT_COMPONENTS  = 0x9125,
	GL_MAX_FRAGMENT_ATOMIC_COUNTER_BUFFERS = 0x92D0,
	GL_MAX_FRAGMENT_ATOMIC_COUNTERS = 0x92D6,
	GL_MAX_COMBINED_ATOMIC_COUNTER_BUFFERS = 0x92D1,
	GL_MAX_COMBINED_ATOMIC_COUNTERS = 0x92D7,
	GL_MAX_ATOMIC_COUNTER_BUFFER_SIZE = 0x92D8,
	GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS = 0x8B4D,
	GL_MAX_COMBINED_IMAGE_UNITS_AND_FRAGMENT_OUTPUTS = 0x8F39,
	GL_MAX_VERTEX_IMAGE_UNIFORMS = 0x90CA,
	GL_MAX_TESS_CONTROL_IMAGE_UNIFORMS = 0x90CB,
	GL_MAX_TESS_EVALUATION_IMAGE_UNIFORMS = 0x90CC,
	GL_MAX_GEOMETRY_IMAGE_UNIFORMS = 0x90CD,
	GL_MAX_FRAGMENT_IMAGE_UNIFORMS = 0x90CE,
	GL_MAX_COMBINED_IMAGE_UNIFORMS = 0x90CF,
	GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS = 0x90D6,
	GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS = 0x90D7,
	GL_MAX_TESS_CONTROL_SHADER_STORAGE_BLOCKS = 0x90D8,
	GL_MAX_TESS_EVALUATION_SHADER_STORAGE_BLOCKS = 0x90D9,
	GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS = 0x90DA,
	GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS = 0x90DB,
	GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS = 0x90DC,
	GL_MAX_VARYING_FLOATS             = 0x8B4B,
	GL_MAX_VARYING_COMPONENTS         = GL_MAX_VARYING_FLOATS,
	GL_MIN_PROGRAM_TEXEL_OFFSET       = 0x8904,
	GL_MAX_PROGRAM_TEXEL_OFFSET       = 0x8905,
	GL_MIN_SAMPLE_SHADING_VALUE       = 0x8C37,
	GL_MIN_PROGRAM_TEXTURE_GATHER_OFFSET = 0x8E5E,
	GL_MAX_PROGRAM_TEXTURE_GATHER_OFFSET = 0x8E5F,
	GL_MAX_PROGRAM_TEXTURE_GATHER_COMPONENTS = 0x8F9F,
	GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH = 0x8A35,
	GL_MAX_SUBROUTINES = 0x8DE7,
	GL_MAX_SUBROUTINE_UNIFORM_LOCATIONS = 0x8DE8,
/* GL_ARB_compute_shader */
	GL_MAX_COMPUTE_SHARED_MEMORY_SIZE          = 0x8262,
	GL_MAX_COMPUTE_UNIFORM_COMPONENTS          = 0x8263,
	GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS      = 0x8264,
	GL_MAX_COMPUTE_ATOMIC_COUNTERS             = 0x8265,
	GL_MAX_COMBINED_COMPUTE_UNIFORM_COMPONENTS = 0x8266,
/* GL_ARB_compute_shader */
	GL_COMPUTE_WORK_GROUP_SIZE                 = 0x8267,
	GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS      = 0x90EB,
	GL_UNIFORM_BLOCK_REFERENCED_BY_COMPUTE_SHADER = 0x90EC,
/* GL_ARB_compute_shader */
	GL_MAX_COMPUTE_UNIFORM_BLOCKS = 0x91BB,
	GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS = 0x91BC,
	GL_MAX_COMPUTE_IMAGE_UNIFORMS = 0x91BD,
	GL_MAX_COMPUTE_WORK_GROUP_COUNT = 0x91BE,
	GL_MAX_COMPUTE_WORK_GROUP_SIZE = 0x91BF,
	GL_MAX_COMPUTE_FIXED_GROUP_SIZE_ARB = 0x91BF,
	GL_MAX_COMPUTE_VARIABLE_GROUP_INVOCATIONS_ARB = 0x9344,
	GL_MAX_COMPUTE_VARIABLE_GROUP_SIZE_ARB = 0x9345,
/* GL_ARB_uniform_buffer_object */
	GL_MAX_VERTEX_UNIFORM_BLOCKS               = 0x8A2B,
	GL_MAX_GEOMETRY_UNIFORM_BLOCKS             = 0x8A2C,
	GL_MAX_FRAGMENT_UNIFORM_BLOCKS             = 0x8A2D,
	GL_MAX_COMBINED_UNIFORM_BLOCKS             = 0x8A2E,
	GL_MAX_UNIFORM_BUFFER_BINDINGS             = 0x8A2F,
	GL_MAX_UNIFORM_BLOCK_SIZE                  = 0x8A30,
	GL_MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS  = 0x8A31,
	GL_MAX_COMBINED_GEOMETRY_UNIFORM_COMPONENTS = 0x8A32,
	GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS = 0x8A33 };



enum GLbuffer_targets 
{	GL_ARRAY_BUFFER              = 0x8892,//#version 150
	GL_ELEMENT_ARRAY_BUFFER      = 0x8893,//#version 150
	GL_PIXEL_PACK_BUFFER         = 0x88EB,//#version 210
	GL_PIXEL_UNPACK_BUFFER       = 0x88EC,//#version 210
	GL_TRANSFORM_FEEDBACK_BUFFER = 0x8C8E,//#version 300
	GL_UNIFORM_BUFFER            = 0x8A11,//#version 310
	GL_ATOMIC_COUNTER_BUFFER     = 0x92C0,//#version 420
	GL_DISPATCH_INDIRECT_BUFFER  = 0x90EE,//#version 430, GL_ARB_compute_shader
	GL_SHADER_STORAGE_BUFFER     = 0x90D2,//#version 430, "https://www.khronos.org/opengl/wiki/Shader_Storage_Buffer_Object"
	GL_QUERY_BUFFER              = 0x9192,//#version 440, "https://www.khronos.org/opengl/wiki/Query_Object"
	GL_COPY_READ_BUFFER          = 0x8F36,//'glCopyBufferSubData'
	GL_COPY_WRITE_BUFFER         = 0x8F37 /*'glCopyBufferSubData'*/ }; 
enum GLbuffer_usages 
{	GL_STREAM_DRAW = 0x88E0,
	GL_STREAM_READ = 0x88E1,
	GL_STREAM_COPY = 0x88E2,
	GL_STATIC_DRAW = 0x88E4,
	GL_STATIC_READ = 0x88E5,
	GL_STATIC_COPY = 0x88E6,
	GL_DYNAMIC_DRAW = 0x88E8,
	GL_DYNAMIC_READ = 0x88E9,
	GL_DYNAMIC_COPY = 0x88EA };
enum GLbuffer_pnames 
{	GL_BUFFER_SIZE = 0x8764,/// #version 150
	GL_BUFFER_USAGE = 0x8765,
	GL_BUFFER_ACCESS = 0x88BB,
	GL_BUFFER_MAPPED = 0x88BC,
	GL_BUFFER_MAP_POINTER = 0x88BD,
	GL_BUFFER_ACCESS_FLAGS = 0x911F,/// #version 310
	GL_BUFFER_MAP_LENGTH = 0x9120,
	GL_BUFFER_MAP_OFFSET = 0x9121 }; 

enum GL_GetActiveAtomicCounterBuffer_pnames
{	GL_ATOMIC_COUNTER_BUFFER_BINDING = 0x92C1,
	GL_ATOMIC_COUNTER_BUFFER_DATA_SIZE = 0x92C4,
	GL_ATOMIC_COUNTER_BUFFER_ACTIVE_ATOMIC_COUNTERS = 0x92C5,
	GL_ATOMIC_COUNTER_BUFFER_ACTIVE_ATOMIC_COUNTER_INDICES = 0x92C6,
	GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_VERTEX_SHADER = 0x92C7,
	GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_TESS_CONTROL_SHADER = 0x92C8,
	GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_TESS_EVALUATION_SHADER = 0x92C9,
	GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_GEOMETRY_SHADER = 0x92CA,
	GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_FRAGMENT_SHADER = 0x92CB,
	GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_COMPUTE_SHADER = 0x90ED }; 

enum GLtransformfeedback_targets 
{	GL_TRANSFORM_FEEDBACK = 0x8E22 };
enum GLtransformfeedback_modes 
{	GL_INTERLEAVED_ATTRIBS = 0x8C8C, 
	GL_SEPARATE_ATTRIBS = 0x8C8D };



enum GLtexture_units /// textureunit = ID of texture register
{	GL_TEXTURE0  = 0x84C0,
	GL_TEXTURE1  = 0x84C1,
	GL_TEXTURE2  = 0x84C2,
	GL_TEXTURE3  = 0x84C3,
	GL_TEXTURE4  = 0x84C4,
	GL_TEXTURE5  = 0x84C5,
	GL_TEXTURE6  = 0x84C6,
	GL_TEXTURE7  = 0x84C7,
	GL_TEXTURE8  = 0x84C8,
	GL_TEXTURE9  = 0x84C9,
	GL_TEXTURE10 = 0x84CA,
	GL_TEXTURE11 = 0x84CB,
	GL_TEXTURE12 = 0x84CC,
	GL_TEXTURE13 = 0x84CD,
	GL_TEXTURE14 = 0x84CE,
	GL_TEXTURE15 = 0x84CF,
	GL_TEXTURE16 = 0x84D0,
	GL_TEXTURE17 = 0x84D1,
	GL_TEXTURE18 = 0x84D2,
	GL_TEXTURE19 = 0x84D3,
	GL_TEXTURE20 = 0x84D4,
	GL_TEXTURE21 = 0x84D5,
	GL_TEXTURE22 = 0x84D6,
	GL_TEXTURE23 = 0x84D7,
	GL_TEXTURE24 = 0x84D8,
	GL_TEXTURE25 = 0x84D9,
	GL_TEXTURE26 = 0x84DA,
	GL_TEXTURE27 = 0x84DB,
	GL_TEXTURE28 = 0x84DC,
	GL_TEXTURE29 = 0x84DD,
	GL_TEXTURE30 = 0x84DE,
	GL_TEXTURE31 = 0x84DF };
enum GLtexture_targets 
{	GL_TEXTURE_1D = 0x0DE0,                 /// #version 110 /
	GL_TEXTURE_2D = 0x0DE1,
	GL_PROXY_TEXTURE_1D = 0x8063,
	GL_PROXY_TEXTURE_2D = 0x8064,
	GL_TEXTURE_3D = 0x806F,                 /// #version 120 /
	GL_PROXY_TEXTURE_3D = 0x8070,
	GL_TEXTURE_CUBE_MAP = 0x8513,
	GL_TEXTURE_CUBE_MAP_POSITIVE_X = 0x8515,/// #version 130 /
	GL_TEXTURE_CUBE_MAP_NEGATIVE_X = 0x8516,
	GL_TEXTURE_CUBE_MAP_POSITIVE_Y = 0x8517,
	GL_TEXTURE_CUBE_MAP_NEGATIVE_Y = 0x8518,
	GL_TEXTURE_CUBE_MAP_POSITIVE_Z = 0x8519,
	GL_TEXTURE_CUBE_MAP_NEGATIVE_Z = 0x851A,
	GL_TEXTURE_CUBE_MAP_RIGHT = GL_TEXTURE_CUBE_MAP_POSITIVE_X,
	GL_TEXTURE_CUBE_MAP_LEFT = GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
	GL_TEXTURE_CUBE_MAP_TOP = GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
	GL_TEXTURE_CUBE_MAP_BOTTOM = GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
	GL_TEXTURE_CUBE_MAP_FRONT = GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
	GL_TEXTURE_CUBE_MAP_BACK = GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
	GL_PROXY_TEXTURE_CUBE_MAP = 0x851B,
	GL_TEXTURE_1D_ARRAY = 0x8C18,           /// #version 300 /
	GL_TEXTURE_2D_ARRAY = 0x8C1A,
	GL_PROXY_TEXTURE_1D_ARRAY = 0x8C19,
	GL_PROXY_TEXTURE_2D_ARRAY = 0x8C1B,
	GL_TEXTURE_BUFFER = 0x8C2A,             /// #version 310 /
	GL_TEXTURE_RECTANGLE = 0x84F5,
	GL_PROXY_TEXTURE_RECTANGLE = 0x84F7,
	GL_TEXTURE_CUBE_MAP_ARRAY = 0x9009,     /// #version 400 /
	GL_PROXY_TEXTURE_CUBE_MAP_ARRAY = 0x900B,
	GL_TEXTURE_2D_MULTISAMPLE = 0x9100,     /// GL_ARB_texture_multisample /
	GL_PROXY_TEXTURE_2D_MULTISAMPLE = 0x9101,
	GL_TEXTURE_2D_MULTISAMPLE_ARRAY = 0x9102,
	GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY = 0x9103 };
enum GLtexture_formats /// internalformat = texture::_Mydata[N]'s typeid
{	GL_RGB4            = 0x804F,/// #version 110 /
	GL_RGB5            = 0x8050,
	GL_RGB8            = 0x8051,
	GL_RGB10           = 0x8052,
	GL_RGB12           = 0x8053,
	GL_RGB16           = 0x8054,
	GL_RGB565          = 0x8D62,
	GL_RGBA2           = 0x8055,
	GL_RGBA4           = 0x8056,
	GL_RGB5_A1         = 0x8057,
	GL_RGBA8           = 0x8058,
	GL_RGB10_A2        = 0x8059,
	GL_RGBA12          = 0x805A,
	GL_RGBA16          = 0x805B,
	GL_DEPTH24_STENCIL8  = 0x88F0,/// #version 130
	GL_DEPTH_COMPONENT16 = 0x81A5,/// #version 140 /
	GL_DEPTH_COMPONENT24 = 0x81A6,
	GL_DEPTH_COMPONENT32 = 0x81A7,
	GL_SRGB8           = 0x8C41,  /// #version 210 /
	GL_SRGB8_ALPHA8    = 0x8C43,
	GL_RGBA32F           = 0x8814,/// #version 300 /
	GL_RGB32F            = 0x8815,
	GL_RGBA16F           = 0x881A,
	GL_RGB16F            = 0x881B,
	GL_RGBA32UI          = 0x8D70,
	GL_RGB32UI           = 0x8D71,
	GL_RGBA16UI          = 0x8D76,
	GL_RGB16UI           = 0x8D77,
	GL_RGBA8UI           = 0x8D7C,
	GL_RGB8UI            = 0x8D7D,
	GL_RGBA32I           = 0x8D82,
	GL_RGB32I            = 0x8D83,
	GL_RGBA16I           = 0x8D88,
	GL_RGB16I            = 0x8D89,
	GL_RGBA8I            = 0x8D8E,
	GL_RGB8I             = 0x8D8F,
	GL_R8_SNORM          = 0x8F94,/// #version 310 /
	GL_RG8_SNORM         = 0x8F95,
	GL_RGB8_SNORM        = 0x8F96,
	GL_RGBA8_SNORM       = 0x8F97,
	GL_R16_SNORM         = 0x8F98,
	GL_RG16_SNORM        = 0x8F99,
	GL_RGB16_SNORM       = 0x8F9A,
	GL_RGBA16_SNORM      = 0x8F9B,
	GL_DEPTH_COMPONENT32F = 0x8CAC,/// GL_ARB_depth_buffer_float /
	GL_DEPTH32F_STENCIL8  = 0x8CAD,
	GL_DEPTH_STENCIL      = 0x84F9,/// GL_ARB_framebuffer_object /
	GL_STENCIL_INDEX1     = 0x8D46,
	GL_STENCIL_INDEX4     = 0x8D47,
	GL_STENCIL_INDEX8     = 0x8D48,
	GL_STENCIL_INDEX16    = 0x8D49,
	GL_R8                 = 0x8229,/// GL_ARB_texture_rg /
	GL_R16                = 0x822A,
	GL_RG8                = 0x822B,
	GL_RG16               = 0x822C,
	GL_R16F               = 0x822D,
	GL_R32F               = 0x822E,
	GL_RG16F              = 0x822F,
	GL_RG32F              = 0x8230,
	GL_R8I                = 0x8231,
	GL_R8UI               = 0x8232,
	GL_R16I               = 0x8233,
	GL_R16UI              = 0x8234,
	GL_R32I               = 0x8235,
	GL_R32UI              = 0x8236,
	GL_RG8I               = 0x8237,
	GL_RG8UI              = 0x8238,
	GL_RG16I              = 0x8239,
	GL_RG16UI             = 0x823A,
	GL_RG32I              = 0x823B,
	GL_RG32UI             = 0x823C,
	GL_COMPRESSED_RED_RGTC1        = 0x8DBB,/// GL_ARB_texture_compression_rgtc /
	GL_COMPRESSED_SIGNED_RED_RGTC1 = 0x8DBC,
	GL_COMPRESSED_RG_RGTC2         = 0x8DBD,
	GL_COMPRESSED_SIGNED_RG_RGTC2  = 0x8DBE,
	GL_COMPRESSED_RGBA_BPTC_UNORM  = 0x8E8C,/// #version 420 /
	GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM = 0x8E8D,
	GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT = 0x8E8E,
	GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT = 0x8E8F };
enum GLtexture_channels 
{	GL_STENCIL_INDEX   = 0x1901,/// #version 110 /
	GL_DEPTH_COMPONENT = 0x1902,
	GL_RED             = 0x1903,
	GL_GREEN           = 0x1904,
	GL_BLUE            = 0x1905,
	GL_ALPHA           = 0x1906,
	GL_RGB             = 0x1907,
	GL_RGBA            = 0x1908,
	GL_BGR             = 0x80E0,/// #version 130 /
	GL_BGRA            = 0x80E1,
	GL_SRGB            = 0x8C40,/// #version 210 /
	GL_SRGB_ALPHA      = 0x8C42,
	GL_RED_INTEGER     = 0x8D94,/// #version 300 /
	GL_GREEN_INTEGER   = 0x8D95,
	GL_BLUE_INTEGER    = 0x8D96,
	GL_ALPHA_INTEGER   = 0x8D97,
	GL_RGB_INTEGER     = 0x8D98,
	GL_RGBA_INTEGER    = 0x8D99,
	GL_BGR_INTEGER     = 0x8D9A,
	GL_BGRA_INTEGER    = 0x8D9B,
	GL_RED_SNORM       = 0x8F90,/// #version 310 /
	GL_RG_SNORM        = 0x8F91,
	GL_RGB_SNORM       = 0x8F92,
	GL_RGBA_SNORM      = 0x8F93,
	GL_COMPRESSED_RED  = 0x8225,/// GL_ARB_texture_rg /
	GL_COMPRESSED_RG   = 0x8226,
	GL_RG              = 0x8227,
	GL_RG_INTEGER      = 0x8228 };
enum GLtexture_address_modes 
{	GL_CLAMP                = 0x2900,//#version 110
	GL_REPEAT               = 0x2901,
	GL_CLAMP_TO_EDGE        = 0x812F,//#version 120
	GL_CLAMP_TO_BORDER      = 0x812D,//#version 130
	GL_MIRRORED_REPEAT      = 0x8370,//#version 140
	GL_MIRROR_CLAMP_TO_EDGE = 0x8743 /*GL_ARB_texture_mirror_clamp_to_edge*/ };
enum GLtexture_filters 
{	GL_NEAREST                = 0x2600,/// #version 110 /
	GL_LINEAR                 = 0x2601,
	GL_NEAREST_MIPMAP_NEAREST = 0x2700,
	GL_LINEAR_MIPMAP_NEAREST  = 0x2701,
	GL_NEAREST_MIPMAP_LINEAR  = 0x2702,
	GL_LINEAR_MIPMAP_LINEAR   = 0x2703,
	/// glTexParameterf(..., GL_TEXTURE_MAX_ANISOTROPY, glGetFloatv(.GL_MAX_TEXTURE_MAX_ANISOTROPY.)) /
	GL_TEXTURE_MAX_ANISOTROPY = 0x84FE,/// #version 460 /
	GL_MAX_TEXTURE_MAX_ANISOTROPY = 0x84FF };
enum GLtexture_pnames
{	GL_TEXTURE_WRAP_S       = 0x2802,/// #version 110
	GL_TEXTURE_WRAP_T       = 0x2803,/// #version 110
	GL_TEXTURE_WRAP_R       = 0x8072,/// #version 120
	GL_TEXTURE_MAG_FILTER   = 0x2800,/// #version 110
	GL_TEXTURE_MIN_FILTER   = 0x2801,/// #version 110
	/// GL_ARB_stencil_texturing
	GL_DEPTH_STENCIL_TEXTURE_MODE = 0x90EA,
	/// LOD
	GL_TEXTURE_MIN_LOD      = 0x813A,/// #version 120 /
	GL_TEXTURE_MAX_LOD      = 0x813B,
	GL_TEXTURE_LOD_BIAS     = 0x8501,
	GL_TEXTURE_BASE_LEVEL   = 0x813C,
	GL_TEXTURE_MAX_LEVEL    = 0x813D,
	/// Depth Mode
	GL_DEPTH_TEXTURE_MODE   = 0x884B,/// #version 140 /
	GL_TEXTURE_COMPARE_MODE = 0x884C,
	GL_TEXTURE_COMPARE_FUNC = 0x884D,
	GL_COMPARE_REF_TO_TEXTURE = 0x884E,/// #version 300
	/// GL_ARB_texture_swizzle
	GL_TEXTURE_SWIZZLE_R    = 0x8E42,
	GL_TEXTURE_SWIZZLE_G    = 0x8E43,
	GL_TEXTURE_SWIZZLE_B    = 0x8E44,
	GL_TEXTURE_SWIZZLE_A    = 0x8E45,
	/// gl.TexParameter*v
	GL_TEXTURE_BORDER_COLOR = 0x1004,
	GL_TEXTURE_SWIZZLE_RGBA = 0x8E46,
	/// unkown
	GL_IMAGE_FORMAT_COMPATIBILITY_TYPE = 0x90C7,
	GL_IMAGE_FORMAT_COMPATIBILITY_BY_SIZE = 0x90C8,
	GL_IMAGE_FORMAT_COMPATIBILITY_BY_CLASS = 0x90C9,
	// #version 450
	GL_TEXTURE_TARGET = 0x1006 };
enum GLtexture_level_pnames 
{	GL_TEXTURE_WIDTH = 0x1000,                /// #version 110 /
	GL_TEXTURE_HEIGHT = 0x1001,
	GL_TEXTURE_DEPTH = 0x8071,                /// #version 120
	GL_TEXTURE_INTERNAL_FORMAT = 0x1003,
	GL_TEXTURE_RED_SIZE = 0x805C,
	GL_TEXTURE_GREEN_SIZE = 0x805D,
	GL_TEXTURE_BLUE_SIZE = 0x805E,
	GL_TEXTURE_ALPHA_SIZE = 0x805F,
	GL_TEXTURE_DEPTH_SIZE = 0x884A,           /// #version 140
	GL_TEXTURE_RED_TYPE = 0x8C10,             /// #version 300 /
	GL_TEXTURE_GREEN_TYPE = 0x8C11,
	GL_TEXTURE_BLUE_TYPE = 0x8C12,
	GL_TEXTURE_ALPHA_TYPE = 0x8C13,
	GL_TEXTURE_DEPTH_TYPE = 0x8C16,
	GL_TEXTURE_SHARED_SIZE = 0x8C3,
	/// Compressed Image
	GL_TEXTURE_COMPRESSED_IMAGE_SIZE = 0x86A0,/// #version 130
	GL_TEXTURE_COMPRESSED = 0x86A1,
	/// GL_ARB_texture_buffer_range
	GL_TEXTURE_BUFFER_OFFSET = 0x919D,        /// #version 430 /
	GL_TEXTURE_BUFFER_SIZE = 0x919E,
	GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT = 0x919F,
	/// GL_ARB_texture_storage
	GL_TEXTURE_IMMUTABLE_FORMAT = 0x912F,
	/// GL_ARB_texture_multisample
	GL_TEXTURE_SAMPLES = 0x9106 };

enum GLpixelstore_pnames 
{	GL_UNPACK_SWAP_BYTES = 0x0CF0,
	GL_UNPACK_LSB_FIRST = 0x0CF1,
	GL_UNPACK_ROW_LENGTH = 0x0CF2,
	GL_UNPACK_SKIP_ROWS = 0x0CF3,
	GL_UNPACK_SKIP_PIXELS = 0x0CF4,
	GL_UNPACK_ALIGNMENT = 0x0CF5,
	GL_PACK_SWAP_BYTES = 0x0D00,
	GL_PACK_LSB_FIRST = 0x0D01,
	GL_PACK_ROW_LENGTH = 0x0D02,
	GL_PACK_SKIP_ROWS = 0x0D03,
	GL_PACK_SKIP_PIXELS = 0x0D04,
	GL_PACK_ALIGNMENT = 0x0D05 }; 



enum GLframebuffer_targets 
{	GL_FRAMEBUFFER = 0x8D40,
	GL_READ_FRAMEBUFFER = 0x8CA8,
	GL_DRAW_FRAMEBUFFER = 0x8CA9 };
enum GLframebuffer_status 
{	GL_FRAMEBUFFER_UNDEFINED = 0x8219,
	GL_FRAMEBUFFER_COMPLETE = 0x8CD5,
	GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT = 0x8CD6,
	GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT = 0x8CD7,
	GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER = 0x8CDB,
	GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER = 0x8CDC,
	GL_FRAMEBUFFER_UNSUPPORTED = 0x8CDD,
	GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE = 0x8D56,
	GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS = 0x8DA8 }; 
enum GLframebuffer_pnames 
{	GL_FRAMEBUFFER_DEFAULT_WIDTH = 0x9310,
	GL_FRAMEBUFFER_DEFAULT_HEIGHT = 0x9311,
	GL_FRAMEBUFFER_DEFAULT_LAYERS = 0x9312,
	GL_FRAMEBUFFER_DEFAULT_SAMPLES = 0x9313,
	GL_FRAMEBUFFER_DEFAULT_FIXED_SAMPLE_LOCATIONS = 0x9314,
	/* boolean value indicating whether double buffering is supported. */
	GL_DOUBLEBUFFER = 0x0C32 };

enum GLrenderbuffer_targets 
{	GL_RENDERBUFFER = 0x8D41 };
enum GLrenderbuffer_pnames 
{	GL_RENDERBUFFER_WIDTH = 0x8D42,
	GL_RENDERBUFFER_HEIGHT = 0x8D43,
	GL_RENDERBUFFER_INTERNAL_FORMAT = 0x8D44,
	GL_RENDERBUFFER_SAMPLES = 0x8CAB,
	GL_RENDERBUFFER_RED_SIZE = 0x8D50,
	GL_RENDERBUFFER_GREEN_SIZE = 0x8D51,
	GL_RENDERBUFFER_BLUE_SIZE = 0x8D52,
	GL_RENDERBUFFER_ALPHA_SIZE = 0x8D53,
	GL_RENDERBUFFER_DEPTH_SIZE = 0x8D54,
	GL_RENDERBUFFER_STENCIL_SIZE = 0x8D55 };

enum GLattachment 
{	GL_COLOR_ATTACHMENT0  = 0x8CE0,
	GL_COLOR_ATTACHMENT1  = 0x8CE1,
	GL_COLOR_ATTACHMENT2  = 0x8CE2,
	GL_COLOR_ATTACHMENT3  = 0x8CE3,
	GL_COLOR_ATTACHMENT4  = 0x8CE4,
	GL_COLOR_ATTACHMENT5  = 0x8CE5,
	GL_COLOR_ATTACHMENT6  = 0x8CE6,
	GL_COLOR_ATTACHMENT7  = 0x8CE7,
	GL_COLOR_ATTACHMENT8  = 0x8CE8,
	GL_COLOR_ATTACHMENT9  = 0x8CE9,
	GL_COLOR_ATTACHMENT10 = 0x8CEA,
	GL_COLOR_ATTACHMENT11 = 0x8CEB,
	GL_COLOR_ATTACHMENT12 = 0x8CEC,
	GL_COLOR_ATTACHMENT13 = 0x8CED,
	GL_COLOR_ATTACHMENT14 = 0x8CEE,
	GL_COLOR_ATTACHMENT15 = 0x8CEF,
	GL_DEPTH_ATTACHMENT = 0x8D00,
	GL_STENCIL_ATTACHMENT = 0x8D20,
	GL_DEPTH_STENCIL_ATTACHMENT = 0x821A }; 
enum GLattachment_types 
{	//GL_NONE = 0,
	GL_FRAMEBUFFER_DEFAULT = 0x8218,
	GL_TEXTURE = 0x1702,
	/*GL_RENDERBUFFER = 0x8D41*/ };
enum GLattachment_pnames 
{	GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING = 0x8210,
	GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE = 0x8211,
	GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE = 0x8212,
	GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE = 0x8213,
	GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE = 0x8214,
	GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE = 0x8215,
	GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE = 0x8216,
	GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE = 0x8217,
	GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE = 0x8CD0,
	GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME = 0x8CD1,
	/// below requires(GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE == GL_TEXTURE).
	GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL = 0x8CD2,
	GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE = 0x8CD3,
	GL_FRAMEBUFFER_ATTACHMENT_LAYERED = 0x8DA7, 
	GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER = 0x8CD4 }; 



enum GLSL_precision_types 
{	GL_LOW_FLOAT = 0x8DF0,
	GL_MEDIUM_FLOAT = 0x8DF1,
	GL_HIGH_FLOAT = 0x8DF2,
	GL_LOW_INT = 0x8DF3,
	GL_MEDIUM_INT = 0x8DF4,
	GL_HIGH_INT = 0x8DF5 }; 

enum GLshader_types 
{ GL_VERTEX_SHADER = 0x8B31,//#version 200
	GL_TESS_CONTROL_SHADER = 0x8E88,//#version 400
	GL_TESS_EVALUATION_SHADER = 0x8E87,//#version 400
	GL_GEOMETRY_SHADER = 0x8DD9,//#version 320 
	GL_FRAGMENT_SHADER = 0x8B30,//#version 200
	GL_COMPUTE_SHADER = 0x91B9/* #version 430, GL_ARB_compute_shader */ };
enum GLshader_binaryformats 
{ GL_SHADER_BINARY_FORMAT_SPIR_V = 0x9551 };
enum GLshader_pnames 
{	GL_SHADER_TYPE = 0x8B4F,
	GL_DELETE_STATUS = 0x8B80,
	GL_COMPILE_STATUS = 0x8B81,
	GL_INFO_LOG_LENGTH = 0x8B84,
	GL_SHADER_SOURCE_LENGTH = 0x8B88,
	GL_SPIR_V_BINARY = 0x9552 };

enum GLprogram_pnames /* : public GLshader_pnames */ 
{ GL_LINK_STATUS = 0x8B82,
	GL_VALIDATE_STATUS = 0x8B83,
	GL_ATTACHED_SHADERS = 0x8B85,
	GL_ACTIVE_UNIFORMS = 0x8B86,
	GL_ACTIVE_UNIFORM_MAX_LENGTH = 0x8B87,
	GL_ACTIVE_UNIFORM_BLOCKS = 0x8A36,
	GL_ACTIVE_ATTRIBUTES = 0x8B89,
	GL_ACTIVE_ATTRIBUTE_MAX_LENGTH = 0x8B8A,
	GL_PROGRAM_BINARY_LENGTH = 0x8741,
	GL_GEOMETRY_VERTICES_OUT = 0x8916,
	GL_GEOMETRY_INPUT_TYPE = 0x8917,
	GL_GEOMETRY_OUTPUT_TYPE = 0x8918,
	GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH = 0x8C76,
	GL_TRANSFORM_FEEDBACK_BUFFER_MODE = 0x8C7F,
	GL_TRANSFORM_FEEDBACK_VARYINGS = 0x8C83,
	GL_TESS_CONTROL_OUTPUT_VERTICES = 0x8E75,
	GL_TESS_GEN_MODE = 0x8E76,
	GL_TESS_GEN_SPACING = 0x8E77,
	GL_TESS_GEN_VERTEX_ORDER = 0x8E78,
	GL_TESS_GEN_POINT_MODE = 0x8E79,
	GL_ACTIVE_ATOMIC_COUNTER_BUFFERS = 0x92D9,
	// set.
	GL_PROGRAM_BINARY_RETRIEVABLE_HINT = 0x8257,
	GL_PROGRAM_SEPARABLE = 0x8258 };
enum GL_tess_gen_spacing 
{ //GL_EQUAL = 0x0202,
	GL_FRACTIONAL_ODD = 0x8E7B,
	GL_FRACTIONAL_EVEN = 0x8E7C };

enum GLprogram_stage_pnames 
{	GL_ACTIVE_SUBROUTINES = 0x8DE5,
	GL_ACTIVE_SUBROUTINE_UNIFORMS = 0x8DE6,
	GL_ACTIVE_SUBROUTINE_UNIFORM_LOCATIONS = 0x8E47,
	GL_ACTIVE_SUBROUTINE_MAX_LENGTH = 0x8E48,
	GL_ACTIVE_SUBROUTINE_UNIFORM_MAX_LENGTH = 0x8E49 };

enum GL_vertexattrib_pnames 
{	GL_VERTEX_ATTRIB_ARRAY_ENABLED = 0x8622,
	GL_VERTEX_ATTRIB_ARRAY_SIZE    = 0x8623,
	GL_VERTEX_ATTRIB_ARRAY_NORMALIZED = 0x886A,
	GL_VERTEX_ATTRIB_ARRAY_STRIDE  = 0x8624,
	GL_VERTEX_ATTRIB_ARRAY_TYPE    = 0x8625,
	GL_CURRENT_VERTEX_ATTRIB       = 0x8626,
	GL_VERTEX_ATTRIB_ARRAY_POINTER = 0x8645,
	GL_VERTEX_ATTRIB_ARRAY_INTEGER = 0x88FD,
	GL_VERTEX_ATTRIB_ARRAY_DIVISOR = 0x88FE };

enum GL_uniform_pnames 
{	GL_UNIFORM_TYPE = 0x8A37,
	GL_UNIFORM_SIZE = 0x8A38,
	GL_UNIFORM_NAME_LENGTH = 0x8A39,
	GL_UNIFORM_BLOCK_INDEX = 0x8A3A,
	GL_UNIFORM_OFFSET = 0x8A3B,
	GL_UNIFORM_ARRAY_STRIDE = 0x8A3C,
	GL_UNIFORM_MATRIX_STRIDE = 0x8A3D,
	GL_UNIFORM_IS_ROW_MAJOR = 0x8A3E,
	GL_UNIFORM_ATOMIC_COUNTER_BUFFER_INDEX = 0x92DA };

enum GL_uniformblock_pnames 
{	GL_UNIFORM_BLOCK_BINDING = 0x8A3F,
	GL_UNIFORM_BLOCK_DATA_SIZE = 0x8A40,
	GL_UNIFORM_BLOCK_NAME_LENGTH = 0x8A41,
	GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS = 0x8A42,
	GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES = 0x8A43,
	GL_UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER = 0x8A44,
	GL_UNIFORM_BLOCK_REFERENCED_BY_TESS_CONTROL_SHADER = 0x84F0,
	GL_UNIFORM_BLOCK_REFERENCED_BY_TESS_EVALUATION_SHADER = 0x84F1,
	GL_UNIFORM_BLOCK_REFERENCED_BY_GEOMETRY_SHADER = 0x8A45,
	GL_UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER = 0x8A46 };

enum GL_subroutine_pnames 
{	GL_NUM_COMPATIBLE_SUBROUTINES = 0x8E4A,
	GL_COMPATIBLE_SUBROUTINES = 0x8E4B
	//GL_UNIFORM_SIZE = 0x8A38,
	/*GL_UNIFORM_NAME_LENGTH = 0x8A39*/ };



enum GL_clear_masks 
{	GL_COLOR_BUFFER_BIT = 0x00004000,
	GL_DEPTH_BUFFER_BIT = 0x00000100,
	GL_STENCIL_BUFFER_BIT = 0x00000400 };

enum GL_blend_func_factors 
{	GL_ZERO                = 0,
	GL_ONE                 = 1,
	GL_SRC_COLOR           = 0x0300,
	GL_ONE_MINUS_SRC_COLOR = 0x0301,
	GL_SRC_ALPHA           = 0x0302,
	GL_ONE_MINUS_SRC_ALPHA = 0x0303,
	GL_DST_ALPHA           = 0x0304,
	GL_ONE_MINUS_DST_ALPHA = 0x0305,
	GL_DST_COLOR           = 0x0306,
	GL_ONE_MINUS_DST_COLOR = 0x0307,
	GL_SRC_ALPHA_SATURATE  = 0x0308 };
enum GL_blend_equation_modes 
{	GL_FUNC_ADD              = 0x8006,//RGB:Rr=RssR+RddR  Gr=GssG+GddG Br=BssB+BddB, Alpha:Ar=AssA+AddA
	GL_FUNC_SUBTRACT         = 0x800A,//RGB:Rr=RssR-RddR  Gr=GssG-GddG Br=BssB-BddB, Alpha:Ar=AssA-AddA
	GL_FUNC_REVERSE_SUBTRACT = 0x800B,//RGB:Rr=RddR-RssR Gr=GddG-GssG Br=BddB-BssB, Alpha:Ar=AddA-AssA
	GL_MIN                   = 0x8007,//RGB:Rr=min(Rs,Rd) Gr=min(Gs,Gd) Br=min(Bs,Bd), Alpha:Ar=min(As,Ad)
	GL_MAX                   = 0x8008 /*RGB:Rr=max(Rs,Rd) Gr=max(Gs,Gd) Br=max(Bs,Bd), Alpha:Ar=max(As,Ad)*/ };

enum GLprimitive_types 
{	GL_POINTS        = 0x0000,
	GL_LINES         = 0x0001,
	GL_LINE_LOOP     = 0x0002,
	GL_LINE_STRIP    = 0x0003,
	GL_TRIANGLES     = 0x0004,
	GL_TRIANGLE_STRIP= 0x0005,
	GL_TRIANGLE_FAN  = 0x0006,
	GL_QUADS         = 0x0007,
	GL_QUAD_STRIP    = 0x0008,
	GL_LINES_ADJACENCY = 0x000A,/* #version 320 */
	GL_LINE_STRIP_ADJACENCY = 0x000B,
	GL_TRIANGLES_ADJACENCY = 0x000C,
	GL_TRIANGLE_STRIP_ADJACENCY = 0x000D,
	GL_PATCHES = 0xE,/* #version 400 */
	GL_ISOLINES = 0x8E7A /* not use in gl.DrawX */ };

enum GLprimitive_patch_pnames 
{	GL_PATCH_VERTICES = 0x8E72,
	GL_PATCH_DEFAULT_INNER_LEVEL = 0x8E73,
	GL_PATCH_DEFAULT_OUTER_LEVEL = 0x8E74 };

enum GLprimitive_polygon_modes 
{	GL_POINT = 0x1B00,
	GL_LINE = 0x1B01,
	GL_FILL = 0x1B02 };

enum GLprimitive_windings 
{	GL_CW = 0x0900,
	GL_CCW = 0x0901 };

enum GLprimitive_cull_modes 
{	GL_CULL_MODE_NONE = GL_NONE,
	GL_CULL_MODE_FRONT = GL_FRONT,
	GL_CULL_MODE_BACK = GL_BACK,
	GL_CULL_MODE_FRONT_AND_BACK = GL_FRONT_AND_BACK };

enum GL_clampcolor_targets 
{	GL_CLAMP_READ_COLOR = 0x891C };

enum GL_drawbuffers 
{	GL_DRAW_BUFFER0 = 0x8825,
	GL_DRAW_BUFFER1 = 0x8826,
	GL_DRAW_BUFFER2 = 0x8827,
	GL_DRAW_BUFFER3 = 0x8828,
	GL_DRAW_BUFFER4 = 0x8829,
	GL_DRAW_BUFFER5 = 0x882A,
	GL_DRAW_BUFFER6 = 0x882B,
	GL_DRAW_BUFFER7 = 0x882C,
	GL_DRAW_BUFFER8 = 0x882D,
	GL_DRAW_BUFFER9 = 0x882E,
	GL_DRAW_BUFFER10 = 0x882F,
	GL_DRAW_BUFFER11 = 0x8830,
	GL_DRAW_BUFFER12 = 0x8831,
	GL_DRAW_BUFFER13 = 0x8832,
	GL_DRAW_BUFFER14 = 0x8833,
	GL_DRAW_BUFFER15 = 0x8834 };


enum GLquery_targets 
{	GL_SAMPLES_PASSED = 0x8914,
	GL_ANY_SAMPLES_PASSED = 0x8C2F,
	GL_ANY_SAMPLES_PASSED_CONSERVATIVE = 0x8D6A,
	GL_PRIMITIVES_GENERATED = 0x8C87,
	GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN = 0x8C88,
	GL_TIME_ELAPSED = 0x88BF, /* glGet(pname) */
	GL_TIMESTAMP = 0x8E28 };
enum GLquery_pnames 
{	GL_CURRENT_QUERY = 0x8865,
	GL_QUERY_COUNTER_BITS = 0x8864,
	GL_QUERY_RESULT_NO_WAIT = 0x9194 };

enum GLquery_object_pnames 
{	GL_QUERY_RESULT = 0x8866,
	GL_QUERY_RESULT_AVAILABLE = 0x8867 };

enum GL_condition_render_modes 
{	GL_QUERY_WAIT = 0x8E13,
	GL_QUERY_NO_WAIT = 0x8E14,
	GL_QUERY_BY_REGION_WAIT = 0x8E15,
	GL_QUERY_BY_REGION_NO_WAIT = 0x8E16 };

enum GLsync_conditions 
{	GL_SYNC_GPU_COMMANDS_COMPLETE = 0x9117 };
enum GLsync_types 
{	GL_SYNC_FENCE = 0x9116 };
enum GLsync_status 
{	GL_UNSIGNALED = 0x9118, 
	GL_SIGNALED = 0x9119 };
enum GLsync_pnames 
{	GL_OBJECT_TYPE = 0x9112, 
	GL_SYNC_CONDITION = 0x9113,
	GL_SYNC_STATUS = 0x9114,
	GL_SYNC_FLAGS = 0x9115 };
enum GLsync_wait_status 
{	GL_ALREADY_SIGNALED = 0x911A, 
	GL_TIMEOUT_EXPIRED = 0x911B,
	GL_CONDITION_SATISFIED = 0x911C, 
	GL_WAIT_FAILED = 0x911D };
enum GLsync_wait_flags 
{	GL_SYNC_FLUSH_COMMANDS_BIT = 0x00000001 };
enum GLsync_wait_timeout 
{	GL_TIMEOUT_IGNORED = 0xFFFFFFFFFFFFFFFFull };


/// Old GLlibrary<100> is a series of Proc in "opengl32.dll", but is very small part of OpenGL,
/// other parts depend on hardware, so only wglGetProcAddress() known where they are in . 
/// 
///   opengl32.dll = GLbasePart{GLlibrary<100>} + GLplatformPart{WGLlibrary|...}.
///   OpenGL = GLbasePart{GLlibrary<100>} + GLotherParts{GLlibrary<110> + GLlibrary<120> + ... + GLlibrary<300> + ...} .
/// 
/// GLlibrary is apply with a context,
/// 
///   get: wglGetCurrentContext().
///   bind: wglMakeCurrent(...,context).
///   unbind: wglMakeCurrent(...,nullptr).
class GLlibrary : public GLplatform {
#pragma warning(disable: 26495)
public:

/// Basic

	const GLubyte* (GLAPIENTRY* GetString)(GLenum name);
	const GLubyte* (GLAPIENTRY* GetStringi)(GLenum name, GLuint index);
	GLenum (GLAPIENTRY* GetError)(void);
	void (GLAPIENTRY* Hint)(GLenum target, GLenum mode);
	void (GLAPIENTRY* GetBooleanv)(GLenum pname, GLboolean* params);
	void (GLAPIENTRY* GetDoublev)(GLenum pname, GLdouble* params);
	void (GLAPIENTRY* GetFloatv)(GLenum pname, GLfloat* params);
	void (GLAPIENTRY* GetIntegerv)(GLenum pname, GLint* params);
	void (GLAPIENTRY* GetInteger64v)(GLenum pname, GLint64* params);
	void (GLAPIENTRY* GetBooleani_v)(GLenum pname, GLuint index, GLboolean* data);
	void (GLAPIENTRY* GetIntegeri_v)(GLenum target, GLuint index, GLint* data);
	void (GLAPIENTRY* GetFloati_v)(GLenum target, GLuint index, GLfloat* data);
	void (GLAPIENTRY* GetDoublei_v)(GLenum target, GLuint index, GLdouble* data);
	void (GLAPIENTRY* GetInteger64i_v)(GLenum pname, GLuint index, GLint64* data);
	void (GLAPIENTRY* Enable)(GLenum cap);
	void (GLAPIENTRY* Enablei)(GLenum cap, GLuint index);
	void (GLAPIENTRY* Disable)(GLenum cap);
	void (GLAPIENTRY* Disablei)(GLenum cap, GLuint index);
	GLboolean (GLAPIENTRY* IsEnabled)(GLenum cap);
	GLboolean (GLAPIENTRY* IsEnabledi)(GLenum cap, GLuint index);

/// Buffer

	void (GLAPIENTRY* GenBuffers)(GLsizei n, GLuint* buffers);
	void (GLAPIENTRY* DeleteBuffers)(GLsizei n, const GLuint* buffers);
	GLboolean (GLAPIENTRY* IsBuffer)(GLuint buffer);
	void (GLAPIENTRY* BindBuffer)(GLenum target, GLuint buffer);
	void (GLAPIENTRY* BindBufferBase)(GLenum target, GLuint index, GLuint buffer);
	void (GLAPIENTRY* BindBuffersBase)(GLenum target, GLuint first, GLsizei count, const GLuint* buffers);
	void (GLAPIENTRY* BindBufferRange)(GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size);
	void (GLAPIENTRY* BindBuffersRange)(GLenum target, GLuint first, GLsizei count, const GLuint* buffers, const GLintptr* offsets, const GLsizeiptr* sizes);
	void (GLAPIENTRY* BufferData)(GLenum target, GLsizeiptr size, const void* data, GLenum usage);
	void (GLAPIENTRY* BufferSubData)(GLenum target, GLintptr offset, GLsizeiptr size, const void* data);
	void (GLAPIENTRY* GetBufferSubData)(GLenum target, GLintptr offset, GLsizeiptr size, void* data);
	void (GLAPIENTRY* GetBufferPointerv)(GLenum target, GLenum pname, void** params);
	void (GLAPIENTRY* CopyBufferSubData)(GLenum readtarget, GLenum writetarget, GLintptr readoffset, GLintptr writeoffset, GLsizeiptr size);
	void (GLAPIENTRY* ClearBufferData)(GLenum target, GLenum internalformat, GLenum format, GLenum type, const void* data);
	void (GLAPIENTRY* ClearBufferSubData)(GLenum target, GLenum internalformat, GLintptr offset, GLsizeiptr size, GLenum format, GLenum type, const void* data);
	void (GLAPIENTRY* GetBufferParameteriv)(GLenum target, GLenum pname, GLint* params);
	void (GLAPIENTRY* GetBufferParameteri64v)(GLenum target, GLenum value, GLint64* data);
 
	void* (GLAPIENTRY* MapBuffer)(GLenum target, GLenum access);
	GLboolean (GLAPIENTRY* UnmapBuffer)(GLenum target);
	void* (GLAPIENTRY* MapBufferRange)(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access);
	void (GLAPIENTRY* FlushMappedBufferRange)(GLenum target, GLintptr offset, GLsizeiptr length);

	void (GLAPIENTRY* GetActiveAtomicCounterBufferiv)(GLuint program, GLuint bufferIndex, GLenum pname, GLint* params);
	
	void (GLAPIENTRY* GenTransformFeedbacks) (GLsizei n, GLuint* ids);
	void (GLAPIENTRY* DeleteTransformFeedbacks) (GLsizei n, const GLuint* ids);
	GLboolean (GLAPIENTRY* IsTransformFeedback) (GLuint id);
	void (GLAPIENTRY* BindTransformFeedback) (GLenum target, GLuint id);
	void (GLAPIENTRY* PauseTransformFeedback) (void);
	void (GLAPIENTRY* ResumeTransformFeedback) (void);
	void (GLAPIENTRY* BeginTransformFeedback) (GLenum primitiveMode);
	void (GLAPIENTRY* EndTransformFeedback) (void);
	void (GLAPIENTRY* TransformFeedbackVaryings) (GLuint program, GLsizei count, const GLchar* const* varyings, GLenum bufferMode);
	void (GLAPIENTRY* GetTransformFeedbackVarying) (GLuint program, GLuint index, GLsizei bufSize, GLsizei* length, GLsizei* size, GLenum* type, GLchar* name);

/// Image&Texture

	void (GLAPIENTRY* GenTextures)(GLsizei n, GLuint* textures);
	void (GLAPIENTRY* DeleteTextures)(GLsizei n, const GLuint* textures);
	GLboolean (GLAPIENTRY* IsTexture)(GLuint texture);
	GLboolean (GLAPIENTRY* AreTexturesResident)(GLsizei n, const GLuint* textures, GLboolean* residences);
	void (GLAPIENTRY* ActiveTexture)(GLenum texture);
	void (GLAPIENTRY* BindTexture)(GLenum target, GLuint texture);
	void (GLAPIENTRY* BindTextures)(GLuint first, GLsizei count, const GLuint* textures);
	void (GLAPIENTRY* GenerateMipmap)(GLenum target);
	void (GLAPIENTRY* TexStorage1D)(GLenum target, GLsizei levels, GLenum internalformat, GLsizei width);
	void (GLAPIENTRY* TexStorage2D)(GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height);
	void (GLAPIENTRY* TexStorage3D)(GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth);
	void (GLAPIENTRY* GetTexImage)(GLenum target, GLint level, GLenum format, GLenum type, void* pixels);
	void (GLAPIENTRY* TexImage1D)(GLenum target, GLint level, GLint internalformat, GLsizei width, GLint border, GLenum format, GLenum type, const void* pixels);
	void (GLAPIENTRY* TexImage2D)(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void* pixels);
	void (GLAPIENTRY* TexImage3D)(GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void* pixels);
	void (GLAPIENTRY* TexSubImage1D)(GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void* pixels);
	void (GLAPIENTRY* TexSubImage2D)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void* pixels);
	void (GLAPIENTRY* TexSubImage3D)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void* pixels);
	void (GLAPIENTRY* CopyTexImage1D)(GLenum target, GLint level, GLenum internalFormat, GLint x, GLint y, GLsizei width, GLint border);
	void (GLAPIENTRY* CopyTexImage2D)(GLenum target, GLint level, GLenum internalFormat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border);
	void (GLAPIENTRY* CopyTexSubImage1D)(GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width);
	void (GLAPIENTRY* CopyTexSubImage2D)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height);
	void (GLAPIENTRY* CopyTexSubImage3D)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);
	void (GLAPIENTRY* CompressedTexImage1D)(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const void* data);
	void (GLAPIENTRY* CompressedTexImage2D)(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const void* data);
	void (GLAPIENTRY* CompressedTexImage3D)(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void* data);
	void (GLAPIENTRY* GetCompressedTexImage)(GLenum target, GLint lod, void* img);
	void (GLAPIENTRY* CompressedTexSubImage1D)(GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void* data);
	void (GLAPIENTRY* CompressedTexSubImage2D)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void* data);
	void (GLAPIENTRY* CompressedTexSubImage3D)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void* data);
	void (GLAPIENTRY* BindImageTexture)(GLuint unit, GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum format);
	void (GLAPIENTRY* CopyImageSubData)(GLuint srcName, GLenum srcTarget, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ, GLuint dstName, GLenum dstTarget, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ, GLsizei srcWidth, GLsizei srcHeight, GLsizei srcDepth);
	void (GLAPIENTRY* TexStorage2DMultisample)(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations);
	void (GLAPIENTRY* TexStorage3DMultisample)(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations);
	void (GLAPIENTRY* ClearTexImage)(GLuint texture, GLint level, GLenum format, GLenum type, const void* data);
	void (GLAPIENTRY* ClearTexSubImage)(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void* data);
	void (GLAPIENTRY* TexBuffer)(GLenum target, GLenum internalFormat, GLuint buffer);
	void (GLAPIENTRY* TexParameteri)(GLenum target, GLenum pname, GLint param);
	void (GLAPIENTRY* TexParameteriv)(GLenum target, GLenum pname, const GLint* params);
	void (GLAPIENTRY* TexParameterf)(GLenum target, GLenum pname, GLfloat param);
	void (GLAPIENTRY* TexParameterfv)(GLenum target, GLenum pname, const GLfloat* params);
	void (GLAPIENTRY* TexParameterIiv)(GLenum target, GLenum pname, const GLint* params);
	void (GLAPIENTRY* TexParameterIuiv)(GLenum target, GLenum pname, const GLuint* params);
	void (GLAPIENTRY* GetTexParameterfv)(GLenum target, GLenum pname, GLfloat* params);
	void (GLAPIENTRY* GetTexParameteriv)(GLenum target, GLenum pname, GLint* params);
	void (GLAPIENTRY* GetTexParameterIiv)(GLenum target, GLenum pname, GLint* params);
	void (GLAPIENTRY* GetTexParameterIuiv)(GLenum target, GLenum pname, GLuint* params);
	void (GLAPIENTRY* GetTexLevelParameteriv)(GLenum target, GLint level, GLenum pname, GLint* params);
	void (GLAPIENTRY* GetTexLevelParameterfv)(GLenum target, GLint level, GLenum pname, GLfloat* params);
	void (GLAPIENTRY* PixelStorei)(GLenum pname, GLint param);
	void (GLAPIENTRY* PixelStoref)(GLenum pname, GLfloat param);
	void (GLAPIENTRY* BindImageTextures)(GLuint first, GLsizei count, const GLuint* textures);

	void (GLAPIENTRY* DeleteSamplers)(GLsizei count, const GLuint* samplers);
	void (GLAPIENTRY* GenSamplers)(GLsizei count, GLuint* samplers);
	GLboolean (GLAPIENTRY* IsSampler)(GLuint sampler);
	void (GLAPIENTRY* BindSampler)(GLuint unit, GLuint sampler);
	void (GLAPIENTRY* BindSamplers)(GLuint first, GLsizei count, const GLuint* samplers);
	void (GLAPIENTRY* GetSamplerParameterIiv)(GLuint sampler, GLenum pname, GLint* params);
	void (GLAPIENTRY* GetSamplerParameterIuiv)(GLuint sampler, GLenum pname, GLuint* params);
	void (GLAPIENTRY* GetSamplerParameterfv)(GLuint sampler, GLenum pname, GLfloat* params);
	void (GLAPIENTRY* GetSamplerParameteriv)(GLuint sampler, GLenum pname, GLint* params);
	void (GLAPIENTRY* SamplerParameterIiv)(GLuint sampler, GLenum pname, const GLint* params);
	void (GLAPIENTRY* SamplerParameterIuiv)(GLuint sampler, GLenum pname, const GLuint* params);
	void (GLAPIENTRY* SamplerParameterf)(GLuint sampler, GLenum pname, GLfloat param);
	void (GLAPIENTRY* SamplerParameterfv)(GLuint sampler, GLenum pname, const GLfloat* params);
	void (GLAPIENTRY* SamplerParameteri)(GLuint sampler, GLenum pname, GLint param);
	void (GLAPIENTRY* SamplerParameteriv)(GLuint sampler, GLenum pname, const GLint* params);

/// Framebuffer

	void (GLAPIENTRY* GenFramebuffers)(GLsizei n, GLuint* framebuffers);
	void (GLAPIENTRY* DeleteFramebuffers)(GLsizei n, const GLuint* framebuffers);
	GLboolean (GLAPIENTRY* IsFramebuffer)(GLuint framebuffer);
	void (GLAPIENTRY* BindFramebuffer)(GLenum target, GLuint framebuffer);
	void (GLAPIENTRY* BlitFramebuffer)(GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);
	GLenum (GLAPIENTRY* CheckFramebufferStatus)(GLenum target);
	void (GLAPIENTRY* FramebufferParameteri)(GLenum target, GLenum pname, GLint param);
	void (GLAPIENTRY* GetFramebufferParameteriv)(GLenum target, GLenum pname, GLint* params);

	void (GLAPIENTRY* GenRenderbuffers)(GLsizei n, GLuint* renderbuffers);
	void (GLAPIENTRY* DeleteRenderbuffers)(GLsizei n, const GLuint* renderbuffers);
	GLboolean (GLAPIENTRY* IsRenderbuffer)(GLuint renderbuffer);
	void (GLAPIENTRY* BindRenderbuffer)(GLenum target, GLuint renderbuffer);
	void (GLAPIENTRY* RenderbufferStorage)(GLenum target, GLenum internalformat, GLsizei width, GLsizei height);
	void (GLAPIENTRY* RenderbufferStorageMultisample)(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height);
	void (GLAPIENTRY* GetRenderbufferParameteriv)(GLenum target, GLenum pname, GLint* params);
	
	void (GLAPIENTRY* FramebufferRenderbuffer)(GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer);
	void (GLAPIENTRY* FramebufferTexture) (GLenum target, GLenum attachment, GLuint texture, GLint level);
	void (GLAPIENTRY* FramebufferTexture1D)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level);
	void (GLAPIENTRY* FramebufferTexture2D)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level);
	void (GLAPIENTRY* FramebufferTexture3D)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level, GLint layer);
	void (GLAPIENTRY* FramebufferTextureLayer)(GLenum target, GLenum attachment, GLuint texture, GLint level, GLint layer);
	void (GLAPIENTRY* GetFramebufferAttachmentParameteriv)(GLenum target, GLenum attachment, GLenum pname, GLint* params);

/// Shader

	void (GLAPIENTRY* GetShaderPrecisionFormat)(GLenum shadertype, GLenum precisiontype, GLint* range, GLint* precision);
	void (GLAPIENTRY* ReleaseShaderCompiler)(void);

	GLuint (GLAPIENTRY* CreateShader)(GLenum type);
	void (GLAPIENTRY* DeleteShader)(GLuint shader);
	void (GLAPIENTRY* ShaderSource)(GLuint shader, GLsizei count, const GLchar* const* string, const GLint* length);
	void (GLAPIENTRY* CompileShader)(GLuint shader);
	void (GLAPIENTRY* ShaderBinary)(GLsizei count, const GLuint* shaders, GLenum binaryformat, const void* binary, GLsizei length);
	void (GLAPIENTRY* SpecializeShader)(GLuint shader, const GLchar* pEntryPoint, GLuint numSpecializationConstants, const GLuint* pConstantIndex, const GLuint* pConstantValue);
	GLboolean (GLAPIENTRY* IsShader)(GLuint shader);
	void (GLAPIENTRY* GetShaderiv)(GLuint shader, GLenum pname, GLint* param);
	void (GLAPIENTRY* GetShaderSource)(GLuint obj, GLsizei maxLength, GLsizei* length, GLchar* source);
	void (GLAPIENTRY* GetShaderInfoLog)(GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* infoLog);

	GLuint (GLAPIENTRY* CreateProgram)(void);
	void (GLAPIENTRY* DeleteProgram)(GLuint program);
	void (GLAPIENTRY* AttachShader)(GLuint program, GLuint shader);
	void (GLAPIENTRY* DetachShader)(GLuint program, GLuint shader);
	void (GLAPIENTRY* LinkProgram)(GLuint program);
	void (GLAPIENTRY* GetAttachedShaders)(GLuint program, GLsizei maxCount, GLsizei* count, GLuint* shaders);
	void (GLAPIENTRY* GetProgramBinary)(GLuint program, GLsizei bufSize, GLsizei* length, GLenum* binaryFormat, void* binary);
	void (GLAPIENTRY* ProgramBinary)(GLuint program, GLenum binaryFormat, const void* binary, GLsizei length);
	GLboolean (GLAPIENTRY* IsProgram)(GLuint program);
	void (GLAPIENTRY* UseProgram)(GLuint program);
	void (GLAPIENTRY* ValidateProgram)(GLuint program);
	void (GLAPIENTRY* GetProgramiv)(GLuint program, GLenum pname, GLint* param);
	void (GLAPIENTRY* GetProgramInfoLog)(GLuint program, GLsizei bufSize, GLsizei* length, GLchar* infoLog);
	void (GLAPIENTRY* ProgramParameteri)(GLuint program, GLenum pname, GLint value);
	void (GLAPIENTRY* GetProgramStageiv)(GLuint program, GLenum shadertype, GLenum pname, GLint* values);

	void (GLAPIENTRY* DisableVertexAttribArray)(GLuint index);
	void (GLAPIENTRY* EnableVertexAttribArray)(GLuint index);
	void (GLAPIENTRY* BindAttribLocation)(GLuint program, GLuint index, const GLchar* name);
	GLint (GLAPIENTRY* GetAttribLocation)(GLuint program, const GLchar* name);
	void (GLAPIENTRY* GetActiveAttrib)(GLuint program, GLuint index, GLsizei maxLength, GLsizei* length, GLint* size, GLenum* type, GLchar* name);
	void (GLAPIENTRY* GetVertexAttribPointerv) (GLuint index, GLenum pname, void** pointer);
	void (GLAPIENTRY* GetVertexAttribdv) (GLuint index, GLenum pname, GLdouble* params);
	void (GLAPIENTRY* GetVertexAttribfv) (GLuint index, GLenum pname, GLfloat* params);
	void (GLAPIENTRY* GetVertexAttribiv) (GLuint index, GLenum pname, GLint* params);
	void (GLAPIENTRY* VertexAttrib1d) (GLuint index, GLdouble x);
	void (GLAPIENTRY* VertexAttrib1dv) (GLuint index, const GLdouble* v);
	void (GLAPIENTRY* VertexAttrib1f) (GLuint index, GLfloat x);
	void (GLAPIENTRY* VertexAttrib1fv) (GLuint index, const GLfloat* v);
	void (GLAPIENTRY* VertexAttrib1s) (GLuint index, GLshort x);
	void (GLAPIENTRY* VertexAttrib1sv) (GLuint index, const GLshort* v);
	void (GLAPIENTRY* VertexAttrib2d) (GLuint index, GLdouble x, GLdouble y);
	void (GLAPIENTRY* VertexAttrib2dv) (GLuint index, const GLdouble* v);
	void (GLAPIENTRY* VertexAttrib2f) (GLuint index, GLfloat x, GLfloat y);
	void (GLAPIENTRY* VertexAttrib2fv) (GLuint index, const GLfloat* v);
	void (GLAPIENTRY* VertexAttrib2s) (GLuint index, GLshort x, GLshort y);
	void (GLAPIENTRY* VertexAttrib2sv) (GLuint index, const GLshort* v);
	void (GLAPIENTRY* VertexAttrib3d) (GLuint index, GLdouble x, GLdouble y, GLdouble z);
	void (GLAPIENTRY* VertexAttrib3dv) (GLuint index, const GLdouble* v);
	void (GLAPIENTRY* VertexAttrib3f) (GLuint index, GLfloat x, GLfloat y, GLfloat z);
	void (GLAPIENTRY* VertexAttrib3fv) (GLuint index, const GLfloat* v);
	void (GLAPIENTRY* VertexAttrib3s) (GLuint index, GLshort x, GLshort y, GLshort z);
	void (GLAPIENTRY* VertexAttrib3sv) (GLuint index, const GLshort* v);
	void (GLAPIENTRY* VertexAttrib4Nbv) (GLuint index, const GLbyte* v);
	void (GLAPIENTRY* VertexAttrib4Niv) (GLuint index, const GLint* v);
	void (GLAPIENTRY* VertexAttrib4Nsv) (GLuint index, const GLshort* v);
	void (GLAPIENTRY* VertexAttrib4Nub) (GLuint index, GLubyte x, GLubyte y, GLubyte z, GLubyte w);
	void (GLAPIENTRY* VertexAttrib4Nubv) (GLuint index, const GLubyte* v);
	void (GLAPIENTRY* VertexAttrib4Nuiv) (GLuint index, const GLuint* v);
	void (GLAPIENTRY* VertexAttrib4Nusv) (GLuint index, const GLushort* v);
	void (GLAPIENTRY* VertexAttrib4bv) (GLuint index, const GLbyte* v);
	void (GLAPIENTRY* VertexAttrib4d) (GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
	void (GLAPIENTRY* VertexAttrib4dv) (GLuint index, const GLdouble* v);
	void (GLAPIENTRY* VertexAttrib4f) (GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
	void (GLAPIENTRY* VertexAttrib4fv) (GLuint index, const GLfloat* v);
	void (GLAPIENTRY* VertexAttrib4iv) (GLuint index, const GLint* v);
	void (GLAPIENTRY* VertexAttrib4s) (GLuint index, GLshort x, GLshort y, GLshort z, GLshort w);
	void (GLAPIENTRY* VertexAttrib4sv) (GLuint index, const GLshort* v);
	void (GLAPIENTRY* VertexAttrib4ubv) (GLuint index, const GLubyte* v);
	void (GLAPIENTRY* VertexAttrib4uiv) (GLuint index, const GLuint* v);
	void (GLAPIENTRY* VertexAttrib4usv) (GLuint index, const GLushort* v);
	void (GLAPIENTRY* VertexAttribPointer) (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void* pointer);
	void (GLAPIENTRY* GetVertexAttribIiv) (GLuint index, GLenum pname, GLint* params);
	void (GLAPIENTRY* GetVertexAttribIuiv) (GLuint index, GLenum pname, GLuint* params);
	void (GLAPIENTRY* VertexAttribI1i) (GLuint index, GLint v0);
	void (GLAPIENTRY* VertexAttribI1iv) (GLuint index, const GLint* v0);
	void (GLAPIENTRY* VertexAttribI1ui) (GLuint index, GLuint v0);
	void (GLAPIENTRY* VertexAttribI1uiv) (GLuint index, const GLuint* v0);
	void (GLAPIENTRY* VertexAttribI2i) (GLuint index, GLint v0, GLint v1);
	void (GLAPIENTRY* VertexAttribI2iv) (GLuint index, const GLint* v0);
	void (GLAPIENTRY* VertexAttribI2ui) (GLuint index, GLuint v0, GLuint v1);
	void (GLAPIENTRY* VertexAttribI2uiv) (GLuint index, const GLuint* v0);
	void (GLAPIENTRY* VertexAttribI3i) (GLuint index, GLint v0, GLint v1, GLint v2);
	void (GLAPIENTRY* VertexAttribI3iv) (GLuint index, const GLint* v0);
	void (GLAPIENTRY* VertexAttribI3ui) (GLuint index, GLuint v0, GLuint v1, GLuint v2);
	void (GLAPIENTRY* VertexAttribI3uiv) (GLuint index, const GLuint* v0);
	void (GLAPIENTRY* VertexAttribI4bv) (GLuint index, const GLbyte* v0);
	void (GLAPIENTRY* VertexAttribI4i) (GLuint index, GLint v0, GLint v1, GLint v2, GLint v3);
	void (GLAPIENTRY* VertexAttribI4iv) (GLuint index, const GLint* v0);
	void (GLAPIENTRY* VertexAttribI4sv) (GLuint index, const GLshort* v0);
	void (GLAPIENTRY* VertexAttribI4ubv) (GLuint index, const GLubyte* v0);
	void (GLAPIENTRY* VertexAttribI4ui) (GLuint index, GLuint v0, GLuint v1, GLuint v2, GLuint v3);
	void (GLAPIENTRY* VertexAttribI4uiv) (GLuint index, const GLuint* v0);
	void (GLAPIENTRY* VertexAttribI4usv) (GLuint index, const GLushort* v0);
	void (GLAPIENTRY* VertexAttribIPointer) (GLuint index, GLint size, GLenum type, GLsizei stride, const void* pointer);
	void (GLAPIENTRY* GetVertexAttribLdv) (GLuint index, GLenum pname, GLdouble* params);
	void (GLAPIENTRY* VertexAttribL1d) (GLuint index, GLdouble x);
	void (GLAPIENTRY* VertexAttribL1dv) (GLuint index, const GLdouble* v);
	void (GLAPIENTRY* VertexAttribL2d) (GLuint index, GLdouble x, GLdouble y);
	void (GLAPIENTRY* VertexAttribL2dv) (GLuint index, const GLdouble* v);
	void (GLAPIENTRY* VertexAttribL3d) (GLuint index, GLdouble x, GLdouble y, GLdouble z);
	void (GLAPIENTRY* VertexAttribL3dv) (GLuint index, const GLdouble* v);
	void (GLAPIENTRY* VertexAttribL4d) (GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
	void (GLAPIENTRY* VertexAttribL4dv) (GLuint index, const GLdouble* v);
	void (GLAPIENTRY* VertexAttribLPointer) (GLuint index, GLint size, GLenum type, GLsizei stride, const void* pointer);

	void (GLAPIENTRY* GetActiveUniformsiv)(GLuint program, GLsizei uniformCount, const GLuint* uniformIndices, GLenum pname, GLint* params);
	void (GLAPIENTRY* GetActiveUniform)(GLuint program, GLuint index, GLsizei maxLength, GLsizei * length, GLint * size, GLenum * type, GLchar * name);
	void (GLAPIENTRY* GetActiveUniformName)(GLuint program, GLuint uniformIndex, GLsizei bufSize, GLsizei* length, GLchar* uniformName);
	GLint (GLAPIENTRY* GetUniformLocation)(GLuint program, const GLchar* name);
	void (GLAPIENTRY* GetUniformIndices)(GLuint program, GLsizei uniformCount, const GLchar* const* uniformNames, GLuint* uniformIndices);
	void (GLAPIENTRY* GetUniformiv) (GLuint program, GLint location, GLint* params);
	void (GLAPIENTRY* GetUniformuiv) (GLuint program, GLint location, GLuint* params);
	void (GLAPIENTRY* GetUniformfv) (GLuint program, GLint location, GLfloat* params);
	void (GLAPIENTRY* GetUniformdv) (GLuint program, GLint location, GLdouble* params);
	void (GLAPIENTRY* Uniform1f) (GLint location, GLfloat v0);
	void (GLAPIENTRY* Uniform1fv) (GLint location, GLsizei count, const GLfloat* value);
	void (GLAPIENTRY* Uniform1i) (GLint location, GLint v0);
	void (GLAPIENTRY* Uniform1ui) (GLint location, GLuint v0);
	void (GLAPIENTRY* Uniform1iv) (GLint location, GLsizei count, const GLint* value);
	void (GLAPIENTRY* Uniform1uiv) (GLint location, GLsizei count, const GLuint* value);
	void (GLAPIENTRY* Uniform2f) (GLint location, GLfloat v0, GLfloat v1);
	void (GLAPIENTRY* Uniform2fv) (GLint location, GLsizei count, const GLfloat* value);
	void (GLAPIENTRY* Uniform2i) (GLint location, GLint v0, GLint v1);
	void (GLAPIENTRY* Uniform2ui) (GLint location, GLuint v0, GLuint v1);
	void (GLAPIENTRY* Uniform2iv) (GLint location, GLsizei count, const GLint* value);
	void (GLAPIENTRY* Uniform2uiv) (GLint location, GLsizei count, const GLuint* value);
	void (GLAPIENTRY* Uniform3f) (GLint location, GLfloat v0, GLfloat v1, GLfloat v2);
	void (GLAPIENTRY* Uniform3fv) (GLint location, GLsizei count, const GLfloat* value);
	void (GLAPIENTRY* Uniform3i) (GLint location, GLint v0, GLint v1, GLint v2);
	void (GLAPIENTRY* Uniform3ui) (GLint location, GLuint v0, GLuint v1, GLuint v2);
	void (GLAPIENTRY* Uniform3iv) (GLint location, GLsizei count, const GLint* value);
	void (GLAPIENTRY* Uniform3uiv) (GLint location, GLsizei count, const GLuint* value);
	void (GLAPIENTRY* Uniform4f) (GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
	void (GLAPIENTRY* Uniform4fv) (GLint location, GLsizei count, const GLfloat* value);
	void (GLAPIENTRY* Uniform4i) (GLint location, GLint v0, GLint v1, GLint v2, GLint v3);
	void (GLAPIENTRY* Uniform4ui) (GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3);
	void (GLAPIENTRY* Uniform4iv) (GLint location, GLsizei count, const GLint* value);
	void (GLAPIENTRY* Uniform4uiv) (GLint location, GLsizei count, const GLuint* value);
	void (GLAPIENTRY* Uniform1d) (GLint location, GLdouble x);
	void (GLAPIENTRY* Uniform1dv) (GLint location, GLsizei count, const GLdouble* value);
	void (GLAPIENTRY* Uniform2d) (GLint location, GLdouble x, GLdouble y);
	void (GLAPIENTRY* Uniform2dv) (GLint location, GLsizei count, const GLdouble* value);
	void (GLAPIENTRY* Uniform3d) (GLint location, GLdouble x, GLdouble y, GLdouble z);
	void (GLAPIENTRY* Uniform3dv) (GLint location, GLsizei count, const GLdouble* value);
	void (GLAPIENTRY* Uniform4d) (GLint location, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
	void (GLAPIENTRY* Uniform4dv) (GLint location, GLsizei count, const GLdouble* value);
	void (GLAPIENTRY* UniformMatrix2fv) (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
	void (GLAPIENTRY* UniformMatrix3fv) (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
	void (GLAPIENTRY* UniformMatrix4fv) (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
	void (GLAPIENTRY* UniformMatrix2x3fv) (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
	void (GLAPIENTRY* UniformMatrix2x4fv) (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
	void (GLAPIENTRY* UniformMatrix3x2fv) (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
	void (GLAPIENTRY* UniformMatrix3x4fv) (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
	void (GLAPIENTRY* UniformMatrix4x2fv) (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
	void (GLAPIENTRY* UniformMatrix4x3fv) (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
	void (GLAPIENTRY* UniformMatrix2dv) (GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
	void (GLAPIENTRY* UniformMatrix2x3dv) (GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
	void (GLAPIENTRY* UniformMatrix2x4dv) (GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
	void (GLAPIENTRY* UniformMatrix3dv) (GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
	void (GLAPIENTRY* UniformMatrix3x2dv) (GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
	void (GLAPIENTRY* UniformMatrix3x4dv) (GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
	void (GLAPIENTRY* UniformMatrix4dv) (GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
	void (GLAPIENTRY* UniformMatrix4x2dv) (GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
	void (GLAPIENTRY* UniformMatrix4x3dv) (GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);

	GLuint (GLAPIENTRY* GetUniformBlockIndex) (GLuint program, const GLchar* uniformBlockName);
	void (GLAPIENTRY* UniformBlockBinding) (GLuint program, GLuint uniformBlockIndex, GLuint uniformBlockBinding);
	void (GLAPIENTRY* ShaderStorageBlockBinding) (GLuint program, GLuint storageBlockIndex, GLuint storageBlockBinding);
	void (GLAPIENTRY* GetActiveUniformBlockiv) (GLuint program, GLuint uniformBlockIndex, GLenum pname, GLint* params);
	void (GLAPIENTRY* GetActiveUniformBlockName) (GLuint program, GLuint uniformBlockIndex, GLsizei bufSize, GLsizei* length, GLchar* uniformBlockName);

	GLuint (GLAPIENTRY* GetSubroutineIndex) (GLuint program, GLenum shadertype, const GLchar* name);
	void (GLAPIENTRY* GetActiveSubroutineName) (GLuint program, GLenum shadertype, GLuint index, GLsizei bufsize, GLsizei* length, GLchar* name);
	GLint (GLAPIENTRY* GetSubroutineUniformLocation) (GLuint program, GLenum shadertype, const GLchar* name);
	void (GLAPIENTRY* GetActiveSubroutineUniformName) (GLuint program, GLenum shadertype, GLuint index, GLsizei bufsize, GLsizei* length, GLchar* name);
	void (GLAPIENTRY* GetActiveSubroutineUniformiv) (GLuint program, GLenum shadertype, GLuint index, GLenum pname, GLint* values);
	void (GLAPIENTRY* GetUniformSubroutineuiv) (GLenum shadertype, GLint location, GLuint* params);
	void (GLAPIENTRY* UniformSubroutinesuiv) (GLenum shadertype, GLsizei count, const GLuint* indices);// similar of glUniformMatrix4fv

	void (GLAPIENTRY* GenVertexArrays)(GLsizei n, GLuint* arrays);
	void (GLAPIENTRY* DeleteVertexArrays)(GLsizei n, const GLuint* arrays);
	GLboolean (GLAPIENTRY* IsVertexArray)(GLuint array);
	void (GLAPIENTRY* BindVertexArray)(GLuint array);
	void (GLAPIENTRY* BindVertexBuffers) (GLuint first, GLsizei count, const GLuint* buffers, const GLintptr* offsets, const GLsizei* strides);
	void (GLAPIENTRY* VertexAttribDivisor) (GLuint index, GLuint divisor);/* used in gl.DrawInstance* */
	void (GLAPIENTRY* BindFragDataLocation) (GLuint program, GLuint colorNumber, const GLchar* name);
	GLint (GLAPIENTRY* GetFragDataLocation) (GLuint program, const GLchar* name);

/// PipelineState

	void (GLAPIENTRY* Scissor)(GLint x, GLint y, GLsizei width, GLsizei height);
	void (GLAPIENTRY* Viewport)(GLint x, GLint y, GLsizei width, GLsizei height);
	void (GLAPIENTRY* WindowPos2d) (GLdouble x, GLdouble y);
	void (GLAPIENTRY* WindowPos2dv) (const GLdouble* p);
	void (GLAPIENTRY* WindowPos3d) (GLdouble x, GLdouble y, GLdouble z);
	void (GLAPIENTRY* WindowPos3dv) (const GLdouble* p);

	void (GLAPIENTRY* LogicOp)(GLenum opcode);//gl.Eneble(GL_COLOR_LOGIC_OP).
	
	void (GLAPIENTRY* Clear)(GLbitfield mask);
	void (GLAPIENTRY* ClearColor)(GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);
	void (GLAPIENTRY* ClearDepth)(GLclampd depth);
	void (GLAPIENTRY* ClearDepthf)(GLfloat d);
	void (GLAPIENTRY* ClearStencil)(GLint s);

	void (GLAPIENTRY* ColorMask)(GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha);
	void (GLAPIENTRY* ColorMaski)(GLuint buf, GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha);
	void (GLAPIENTRY* DepthMask)(GLboolean flag);
	void (GLAPIENTRY* DepthRange)(GLclampd zNear, GLclampd zFar);
	void (GLAPIENTRY* DepthRangef)(GLfloat n, GLfloat f);
	void (GLAPIENTRY* StencilMask)(GLuint mask);

	void (GLAPIENTRY* AlphaFunc)(GLenum func, GLclampf ref);//gl.Eneble(GL_ALPHA_TEST).
	void (GLAPIENTRY* DepthFunc)(GLenum func);//gl.Eneble(GL_DEPTH_TEST).
	void (GLAPIENTRY* StencilFunc)(GLenum func, GLint ref, GLuint mask);//gl.Eneble(GL_STENCIAL_TEST).
	void (GLAPIENTRY* StencilOp)(GLenum fail, GLenum zfail, GLenum zpass);
	void (GLAPIENTRY* StencilFuncSeparate) (GLenum frontfunc, GLenum backfunc, GLint ref, GLuint mask);
	void (GLAPIENTRY* StencilMaskSeparate) (GLenum face, GLuint mask);
	void (GLAPIENTRY* StencilOpSeparate) (GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass);
	
	void (GLAPIENTRY* BlendFunc)(GLenum sfactor, GLenum dfactor);
	void (GLAPIENTRY* BlendFunci)(GLuint buf, GLenum src, GLenum dst);
	void (GLAPIENTRY* BlendColor)(GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);
	void (GLAPIENTRY* BlendFuncSeparate)(GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha);
	void (GLAPIENTRY* BlendFuncSeparatei)(GLuint buf, GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha);
	void (GLAPIENTRY* BlendEquation)(GLenum mode);
	void (GLAPIENTRY* BlendEquationi)(GLuint buf, GLenum mode);
	void (GLAPIENTRY* BlendEquationSeparate)(GLenum modeRGB, GLenum modeAlpha);
	void (GLAPIENTRY* BlendEquationSeparatei)(GLuint buf, GLenum modeRGB, GLenum modeAlpha);

	void (GLAPIENTRY* SampleCoverage)(GLclampf value, GLboolean invert);
	void (GLAPIENTRY* MinSampleShading)(GLclampf value);

	void (GLAPIENTRY* PointSize)(GLfloat size);
	void (GLAPIENTRY* PointParameterf)(GLenum pname, GLfloat param);
	void (GLAPIENTRY* PointParameterfv)(GLenum pname, const GLfloat* params);
	void (GLAPIENTRY* PointParameteri)(GLenum pname, GLint param);
	void (GLAPIENTRY* PointParameteriv)(GLenum pname, const GLint* params);

	void (GLAPIENTRY* LineWidth)(GLfloat width);

	void (GLAPIENTRY* PolygonMode)(GLenum face, GLenum mode);
	void (GLAPIENTRY* PolygonOffset)(GLfloat factor, GLfloat units);
	void (GLAPIENTRY* PolygonStipple)(const GLubyte* mask);
	void (GLAPIENTRY* FrontFace)(GLenum mode);
	void (GLAPIENTRY* CullFace)(GLenum mode);

	void (GLAPIENTRY* ClipPlane)(GLenum plane, const GLdouble* equation);
	void (GLAPIENTRY* GetClipPlane)(GLenum plane, GLdouble* equation);

	void (GLAPIENTRY* PatchParameterfv)(GLenum pname, const GLfloat* values);
	void (GLAPIENTRY* PatchParameteri)(GLenum pname, GLint value);

	void (GLAPIENTRY* DrawArrays)(GLenum mode, GLint first, GLsizei count);
	void (GLAPIENTRY* DrawElements)(GLenum mode, GLsizei count, GLenum type, const void* indices);
	void (GLAPIENTRY* DrawPixels)(GLsizei width, GLsizei height, GLenum format, GLenum type, const void* pixels);
	void (GLAPIENTRY* DrawRangeElements)(GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void* indices);
	void (GLAPIENTRY* DrawArraysInstanced)(GLenum mode, GLint first, GLsizei count, GLsizei primcount);
	void (GLAPIENTRY* DrawElementsInstanced)(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei primcount);
	void (GLAPIENTRY* DrawElementsBaseVertex)(GLenum mode, GLsizei count, GLenum type, void* indices, GLint basevertex);
	void (GLAPIENTRY* DrawElementsInstancedBaseVertex)(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei primcount, GLint basevertex);
	void (GLAPIENTRY* DrawRangeElementsBaseVertex)(GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, void* indices, GLint basevertex);
	void (GLAPIENTRY* DrawTransformFeedback)(GLenum mode, GLuint id);
	void (GLAPIENTRY* DrawTransformFeedbackStream)(GLenum mode, GLuint id, GLuint stream);
	void (GLAPIENTRY* DrawElementsInstancedBaseVertexBaseInstance)(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei primcount, GLint basevertex, GLuint baseinstance);
	void (GLAPIENTRY* DrawElementsInstancedBaseInstance)(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei primcount, GLuint baseinstance);
	void (GLAPIENTRY* DrawArraysInstancedBaseInstance)(GLenum mode, GLint first, GLsizei count, GLsizei primcount, GLuint baseinstance);
	void (GLAPIENTRY* DrawTransformFeedbackInstanced)(GLenum mode, GLuint id, GLsizei primcount);
	void (GLAPIENTRY* DrawTransformFeedbackStreamInstanced)(GLenum mode, GLuint id, GLuint stream, GLsizei primcount);
	void (GLAPIENTRY* MultiDrawElementsBaseVertex)(GLenum mode, GLsizei* count, GLenum type, void** indices, GLsizei primcount, GLint* basevertex);
	void (GLAPIENTRY* MultiDrawArrays)(GLenum mode, const GLint* first, const GLsizei* count, GLsizei drawcount);
	void (GLAPIENTRY* MultiDrawElements)(GLenum mode, const GLsizei* count, GLenum type, const void* const* indices, GLsizei drawcount);
	void (GLAPIENTRY* MultiDrawArraysIndirectCount)(GLenum mode, const GLvoid* indirect, GLintptr drawcount, GLsizei maxdrawcount, GLsizei stride);
	void (GLAPIENTRY* MultiDrawElementsIndirectCount)(GLenum mode, GLenum type, const GLvoid* indirect, GLintptr drawcount, GLsizei maxdrawcount, GLsizei stride);
	void (GLAPIENTRY* PrimitiveRestartIndex)(GLuint buffer);

	void (GLAPIENTRY* ClampColor)(GLenum target, GLenum clamp);
	void (GLAPIENTRY* ReadPixels)(GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, void* pixels);
	void (GLAPIENTRY* DrawBuffer)(GLenum mode);
	void (GLAPIENTRY* DrawBuffers)(GLsizei n, const GLenum* bufs);
	void (GLAPIENTRY* ReadBuffer)(GLenum mode);
	void (GLAPIENTRY* ClearBufferfi)(GLenum buffer, GLint drawBuffer, GLfloat depth, GLint stencil);
	void (GLAPIENTRY* ClearBufferfv)(GLenum buffer, GLint drawBuffer, const GLfloat* value);
	void (GLAPIENTRY* ClearBufferiv)(GLenum buffer, GLint drawBuffer, const GLint* value);
	void (GLAPIENTRY* ClearBufferuiv)(GLenum buffer, GLint drawBuffer, const GLuint* value);

	void (GLAPIENTRY* DispatchCompute)(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z);
	void (GLAPIENTRY* DispatchComputeIndirect)(GLintptr indirect);
	//void (GLAPIENTRY* DispatchComputeGroupSizeARB) (GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z, GLuint group_size_x, GLuint group_size_y, GLuint group_size_z);

/// Query&Sync

	void (GLAPIENTRY* Finish)(void);
	void (GLAPIENTRY* Flush)(void);

	void (GLAPIENTRY* GenQueries)(GLsizei n, GLuint* ids);
	void (GLAPIENTRY* DeleteQueries)(GLsizei n, const GLuint* ids);
	GLboolean (GLAPIENTRY* IsQuery)(GLuint id);
	void (GLAPIENTRY* QueryCounter)(GLuint id, GLenum target);
	void (GLAPIENTRY* BeginQuery)(GLenum target, GLuint id);
	void (GLAPIENTRY* BeginQueryIndexed)(GLenum target, GLuint index, GLuint id);
	void (GLAPIENTRY* EndQuery)(GLenum target);
	void (GLAPIENTRY* EndQueryIndexed)(GLenum target, GLuint index);
	void (GLAPIENTRY* GetQueryiv)(GLenum target, GLenum pname, GLint* params);
	void (GLAPIENTRY* GetQueryIndexediv)(GLenum target, GLuint index, GLenum pname, GLint* params);

	void (GLAPIENTRY* GetQueryObjectiv)(GLuint id, GLenum pname, GLint* params);
	void (GLAPIENTRY* GetQueryObjectuiv)(GLuint id, GLenum pname, GLuint* params);
	void (GLAPIENTRY* GetQueryObjecti64v)(GLuint id, GLenum pname, GLint64* params);
	void (GLAPIENTRY* GetQueryObjectui64v)(GLuint id, GLenum pname, GLuint64* params);

	void (GLAPIENTRY* BeginConditionalRender)(GLuint id, GLenum mode);
	void (GLAPIENTRY* EndConditionalRender)(void);

	GLsync (GLAPIENTRY* FenceSync)(GLenum condition, GLbitfield flags);
	void (GLAPIENTRY* DeleteSync)(GLsync GLsync);
	GLboolean (GLAPIENTRY* IsSync)(GLsync GLsync);
	void (GLAPIENTRY* GetSynciv)(GLsync GLsync, GLenum pname, GLsizei bufSize, GLsizei* length, GLint* values);
	void (GLAPIENTRY* WaitSync)(GLsync GLsync, GLbitfield flags, GLuint64 timeout);
	void (GLAPIENTRY* MemoryBarrier_)(GLbitfield barriers);
	//GLenum (GLAPIENTRY* ClientWaitSync) (GLsync GLsync, GLbitfield flags, GLuint64 timeout);

	GLlibrary() = default;

	explicit GLlibrary(GLplatformArg arg) : GLplatform(arg) {
		GetString = reinterpret_cast<decltype(GetString)>( GetProcAddress("glGetString") );
		GetStringi = reinterpret_cast<decltype(GetStringi)>( GetProcAddress("glGetStringi") );
		GetError = reinterpret_cast<decltype(GetError)>( GetProcAddress("glGetError") );
		Hint = reinterpret_cast<decltype(Hint)>( GetProcAddress("glHint") );
		GetBooleanv = reinterpret_cast<decltype(GetBooleanv)>( GetProcAddress("glGetBooleanv") );
		GetDoublev = reinterpret_cast<decltype(GetDoublev)>( GetProcAddress("glGetDoublev") );
		GetFloatv = reinterpret_cast<decltype(GetFloatv)>( GetProcAddress("glGetFloatv") );
		GetIntegerv = reinterpret_cast<decltype(GetIntegerv)>( GetProcAddress("glGetIntegerv") );
		GetInteger64v = reinterpret_cast<decltype(GetInteger64v)>( GetProcAddress("glGetInteger64v") );
		GetBooleani_v = reinterpret_cast<decltype(GetBooleani_v)>( GetProcAddress("glGetBooleani_v") );
		GetIntegeri_v = reinterpret_cast<decltype(GetIntegeri_v)>( GetProcAddress("glGetIntegeri_v") );
		GetFloati_v = reinterpret_cast<decltype(GetFloati_v)>( GetProcAddress("glGetFloati_v") );
		GetDoublei_v = reinterpret_cast<decltype(GetDoublei_v)>( GetProcAddress("glGetDoublei_v") );
		GetInteger64i_v = reinterpret_cast<decltype(GetInteger64i_v)>( GetProcAddress("glGetInteger64i_v") );
		Enable = reinterpret_cast<decltype(Enable)>( GetProcAddress("glEnable") );
		Enablei = reinterpret_cast<decltype(Enablei)>( GetProcAddress("glEnablei") );
		Disable = reinterpret_cast<decltype(Disable)>( GetProcAddress("glDisable") );
		Disablei = reinterpret_cast<decltype(Disablei)>( GetProcAddress("glDisablei") );
		IsEnabled = reinterpret_cast<decltype(IsEnabled)>( GetProcAddress("glIsEnabled") );
		IsEnabledi = reinterpret_cast<decltype(IsEnabledi)>( GetProcAddress("glIsEnabledi") );
		GenBuffers = reinterpret_cast<decltype(GenBuffers)>( GetProcAddress("glGenBuffers") );
		DeleteBuffers = reinterpret_cast<decltype(DeleteBuffers)>( GetProcAddress("glDeleteBuffers") );
		IsBuffer = reinterpret_cast<decltype(IsBuffer)>( GetProcAddress("glIsBuffer") );
		BindBuffer = reinterpret_cast<decltype(BindBuffer)>( GetProcAddress("glBindBuffer") );
		BindBufferBase = reinterpret_cast<decltype(BindBufferBase)>( GetProcAddress("glBindBufferBase") );
		BindBuffersBase = reinterpret_cast<decltype(BindBuffersBase)>( GetProcAddress("glBindBuffersBase") );
		BindBufferRange = reinterpret_cast<decltype(BindBufferRange)>( GetProcAddress("glBindBufferRange") );
		BindBuffersRange = reinterpret_cast<decltype(BindBuffersRange)>( GetProcAddress("glBindBuffersRange") );
		BufferData = reinterpret_cast<decltype(BufferData)>( GetProcAddress("glBufferData") );
		BufferSubData = reinterpret_cast<decltype(BufferSubData)>( GetProcAddress("glBufferSubData") );
		GetBufferSubData = reinterpret_cast<decltype(GetBufferSubData)>( GetProcAddress("glGetBufferSubData") );
		GetBufferPointerv = reinterpret_cast<decltype(GetBufferPointerv)>( GetProcAddress("glGetBufferPointerv") );
		CopyBufferSubData = reinterpret_cast<decltype(CopyBufferSubData)>( GetProcAddress("glCopyBufferSubData") );
		ClearBufferData = reinterpret_cast<decltype(ClearBufferData)>( GetProcAddress("glClearBufferData") );
		ClearBufferSubData = reinterpret_cast<decltype(ClearBufferSubData)>( GetProcAddress("glClearBufferSubData") );
		GetBufferParameteriv = reinterpret_cast<decltype(GetBufferParameteriv)>( GetProcAddress("glGetBufferParameteriv") );
		GetBufferParameteri64v = reinterpret_cast<decltype(GetBufferParameteri64v)>( GetProcAddress("glGetBufferParameteri64v") );
		MapBuffer = reinterpret_cast<decltype(MapBuffer)>( GetProcAddress("glMapBuffer") );
		UnmapBuffer = reinterpret_cast<decltype(UnmapBuffer)>( GetProcAddress("glUnmapBuffer") );
		MapBufferRange = reinterpret_cast<decltype(MapBufferRange)>( GetProcAddress("glMapBufferRange") );
		FlushMappedBufferRange = reinterpret_cast<decltype(FlushMappedBufferRange)>( GetProcAddress("glFlushMappedBufferRange") );
		GetActiveAtomicCounterBufferiv = reinterpret_cast<decltype(GetActiveAtomicCounterBufferiv)>( GetProcAddress("glGetActiveAtomicCounterBufferiv") );
		GenTransformFeedbacks = reinterpret_cast<decltype(GenTransformFeedbacks)>( GetProcAddress("glGenTransformFeedbacks") );
		DeleteTransformFeedbacks = reinterpret_cast<decltype(DeleteTransformFeedbacks)>( GetProcAddress("glDeleteTransformFeedbacks") );
		IsTransformFeedback = reinterpret_cast<decltype(IsTransformFeedback)>( GetProcAddress("glIsTransformFeedback") );
		BindTransformFeedback = reinterpret_cast<decltype(BindTransformFeedback)>( GetProcAddress("glBindTransformFeedback") );
		PauseTransformFeedback = reinterpret_cast<decltype(PauseTransformFeedback)>( GetProcAddress("glPauseTransformFeedback") );
		ResumeTransformFeedback = reinterpret_cast<decltype(ResumeTransformFeedback)>( GetProcAddress("glResumeTransformFeedback") );
		BeginTransformFeedback = reinterpret_cast<decltype(BeginTransformFeedback)>( GetProcAddress("glBeginTransformFeedback") );
		EndTransformFeedback = reinterpret_cast<decltype(EndTransformFeedback)>( GetProcAddress("glEndTransformFeedback") );
		TransformFeedbackVaryings = reinterpret_cast<decltype(TransformFeedbackVaryings)>( GetProcAddress("glTransformFeedbackVaryings") );
		GetTransformFeedbackVarying = reinterpret_cast<decltype(GetTransformFeedbackVarying)>( GetProcAddress("glGetTransformFeedbackVarying") );
		GenTextures = reinterpret_cast<decltype(GenTextures)>( GetProcAddress("glGenTextures") );
		DeleteTextures = reinterpret_cast<decltype(DeleteTextures)>( GetProcAddress("glDeleteTextures") );
		IsTexture = reinterpret_cast<decltype(IsTexture)>( GetProcAddress("glIsTexture") );
		AreTexturesResident = reinterpret_cast<decltype(AreTexturesResident)>( GetProcAddress("glAreTexturesResident") );
		ActiveTexture = reinterpret_cast<decltype(ActiveTexture)>( GetProcAddress("glActiveTexture") );
		BindTexture = reinterpret_cast<decltype(BindTexture)>( GetProcAddress("glBindTexture") );
		BindTextures = reinterpret_cast<decltype(BindTextures)>( GetProcAddress("glBindTextures") );
		GenerateMipmap = reinterpret_cast<decltype(GenerateMipmap)>( GetProcAddress("glGenerateMipmap") );
		TexStorage1D = reinterpret_cast<decltype(TexStorage1D)>( GetProcAddress("glTexStorage1D") );
		TexStorage2D = reinterpret_cast<decltype(TexStorage2D)>( GetProcAddress("glTexStorage2D") );
		TexStorage3D = reinterpret_cast<decltype(TexStorage3D)>( GetProcAddress("glTexStorage3D") );
		GetTexImage = reinterpret_cast<decltype(GetTexImage)>( GetProcAddress("glGetTexImage") );
		TexImage1D = reinterpret_cast<decltype(TexImage1D)>( GetProcAddress("glTexImage1D") );
		TexImage2D = reinterpret_cast<decltype(TexImage2D)>( GetProcAddress("glTexImage2D") );
		TexImage3D = reinterpret_cast<decltype(TexImage3D)>( GetProcAddress("glTexImage3D") );
		TexSubImage1D = reinterpret_cast<decltype(TexSubImage1D)>( GetProcAddress("glTexSubImage1D") );
		TexSubImage2D = reinterpret_cast<decltype(TexSubImage2D)>( GetProcAddress("glTexSubImage2D") );
		TexSubImage3D = reinterpret_cast<decltype(TexSubImage3D)>( GetProcAddress("glTexSubImage3D") );
		CopyTexImage1D = reinterpret_cast<decltype(CopyTexImage1D)>( GetProcAddress("glCopyTexImage1D") );
		CopyTexImage2D = reinterpret_cast<decltype(CopyTexImage2D)>( GetProcAddress("glCopyTexImage2D") );
		CopyTexSubImage1D = reinterpret_cast<decltype(CopyTexSubImage1D)>( GetProcAddress("glCopyTexSubImage1D") );
		CopyTexSubImage2D = reinterpret_cast<decltype(CopyTexSubImage2D)>( GetProcAddress("glCopyTexSubImage2D") );
		CopyTexSubImage3D = reinterpret_cast<decltype(CopyTexSubImage3D)>( GetProcAddress("glCopyTexSubImage3D") );
		CompressedTexImage1D = reinterpret_cast<decltype(CompressedTexImage1D)>( GetProcAddress("glCompressedTexImage1D") );
		CompressedTexImage2D = reinterpret_cast<decltype(CompressedTexImage2D)>( GetProcAddress("glCompressedTexImage2D") );
		CompressedTexImage3D = reinterpret_cast<decltype(CompressedTexImage3D)>( GetProcAddress("glCompressedTexImage3D") );
		GetCompressedTexImage = reinterpret_cast<decltype(GetCompressedTexImage)>( GetProcAddress("glGetCompressedTexImage") );
		CompressedTexSubImage1D = reinterpret_cast<decltype(CompressedTexSubImage1D)>( GetProcAddress("glCompressedTexSubImage1D") );
		CompressedTexSubImage2D = reinterpret_cast<decltype(CompressedTexSubImage2D)>( GetProcAddress("glCompressedTexSubImage2D") );
		CompressedTexSubImage3D = reinterpret_cast<decltype(CompressedTexSubImage3D)>( GetProcAddress("glCompressedTexSubImage3D") );
		BindImageTexture = reinterpret_cast<decltype(BindImageTexture)>( GetProcAddress("glBindImageTexture") );
		CopyImageSubData = reinterpret_cast<decltype(CopyImageSubData)>( GetProcAddress("glCopyImageSubData") );
		TexStorage2DMultisample = reinterpret_cast<decltype(TexStorage2DMultisample)>( GetProcAddress("glTexStorage2DMultisample") );
		TexStorage3DMultisample = reinterpret_cast<decltype(TexStorage3DMultisample)>( GetProcAddress("glTexStorage3DMultisample") );
		ClearTexImage = reinterpret_cast<decltype(ClearTexImage)>( GetProcAddress("glClearTexImage") );
		ClearTexSubImage = reinterpret_cast<decltype(ClearTexSubImage)>( GetProcAddress("glClearTexSubImage") );
		TexBuffer = reinterpret_cast<decltype(TexBuffer)>( GetProcAddress("glTexBuffer") );
		TexParameteri = reinterpret_cast<decltype(TexParameteri)>( GetProcAddress("glTexParameteri") );
		TexParameteriv = reinterpret_cast<decltype(TexParameteriv)>( GetProcAddress("glTexParameteriv") );
		TexParameterf = reinterpret_cast<decltype(TexParameterf)>( GetProcAddress("glTexParameterf") );
		TexParameterfv = reinterpret_cast<decltype(TexParameterfv)>( GetProcAddress("glTexParameterfv") );
		TexParameterIiv = reinterpret_cast<decltype(TexParameterIiv)>( GetProcAddress("glTexParameterIiv") );
		TexParameterIuiv = reinterpret_cast<decltype(TexParameterIuiv)>( GetProcAddress("glTexParameterIuiv") );
		GetTexParameterfv = reinterpret_cast<decltype(GetTexParameterfv)>( GetProcAddress("glGetTexParameterfv") );
		GetTexParameteriv = reinterpret_cast<decltype(GetTexParameteriv)>( GetProcAddress("glGetTexParameteriv") );
		GetTexParameterIiv = reinterpret_cast<decltype(GetTexParameterIiv)>( GetProcAddress("glGetTexParameterIiv") );
		GetTexParameterIuiv = reinterpret_cast<decltype(GetTexParameterIuiv)>( GetProcAddress("glGetTexParameterIuiv") );
		GetTexLevelParameteriv = reinterpret_cast<decltype(GetTexLevelParameteriv)>( GetProcAddress("glGetTexLevelParameteriv") );
		GetTexLevelParameterfv = reinterpret_cast<decltype(GetTexLevelParameterfv)>( GetProcAddress("glGetTexLevelParameterfv") );
		PixelStorei = reinterpret_cast<decltype(PixelStorei)>( GetProcAddress("glPixelStorei") );
		PixelStoref = reinterpret_cast<decltype(PixelStoref)>( GetProcAddress("glPixelStoref") );
		BindImageTextures = reinterpret_cast<decltype(BindImageTextures)>( GetProcAddress("glBindImageTextures") );
		DeleteSamplers = reinterpret_cast<decltype(DeleteSamplers)>( GetProcAddress("glDeleteSamplers") );
		GenSamplers = reinterpret_cast<decltype(GenSamplers)>( GetProcAddress("glGenSamplers") );
		IsSampler = reinterpret_cast<decltype(IsSampler)>( GetProcAddress("glIsSampler") );
		BindSampler = reinterpret_cast<decltype(BindSampler)>( GetProcAddress("glBindSampler") );
		BindSamplers = reinterpret_cast<decltype(BindSamplers)>( GetProcAddress("glBindSamplers") );
		GetSamplerParameterIiv = reinterpret_cast<decltype(GetSamplerParameterIiv)>( GetProcAddress("glGetSamplerParameterIiv") );
		GetSamplerParameterIuiv = reinterpret_cast<decltype(GetSamplerParameterIuiv)>( GetProcAddress("glGetSamplerParameterIuiv") );
		GetSamplerParameterfv = reinterpret_cast<decltype(GetSamplerParameterfv)>( GetProcAddress("glGetSamplerParameterfv") );
		GetSamplerParameteriv = reinterpret_cast<decltype(GetSamplerParameteriv)>( GetProcAddress("glGetSamplerParameteriv") );
		SamplerParameterIiv = reinterpret_cast<decltype(SamplerParameterIiv)>( GetProcAddress("glSamplerParameterIiv") );
		SamplerParameterIuiv = reinterpret_cast<decltype(SamplerParameterIuiv)>( GetProcAddress("glSamplerParameterIuiv") );
		SamplerParameterf = reinterpret_cast<decltype(SamplerParameterf)>( GetProcAddress("glSamplerParameterf") );
		SamplerParameterfv = reinterpret_cast<decltype(SamplerParameterfv)>( GetProcAddress("glSamplerParameterfv") );
		SamplerParameteri = reinterpret_cast<decltype(SamplerParameteri)>( GetProcAddress("glSamplerParameteri") );
		SamplerParameteriv = reinterpret_cast<decltype(SamplerParameteriv)>( GetProcAddress("glSamplerParameteriv") );
		GenFramebuffers = reinterpret_cast<decltype(GenFramebuffers)>( GetProcAddress("glGenFramebuffers") );
		DeleteFramebuffers = reinterpret_cast<decltype(DeleteFramebuffers)>( GetProcAddress("glDeleteFramebuffers") );
		IsFramebuffer = reinterpret_cast<decltype(IsFramebuffer)>( GetProcAddress("glIsFramebuffer") );
		BindFramebuffer = reinterpret_cast<decltype(BindFramebuffer)>( GetProcAddress("glBindFramebuffer") );
		BlitFramebuffer = reinterpret_cast<decltype(BlitFramebuffer)>( GetProcAddress("glBlitFramebuffer") );
		CheckFramebufferStatus = reinterpret_cast<decltype(CheckFramebufferStatus)>( GetProcAddress("glCheckFramebufferStatus") );
		FramebufferParameteri = reinterpret_cast<decltype(FramebufferParameteri)>( GetProcAddress("glFramebufferParameteri") );
		GetFramebufferParameteriv = reinterpret_cast<decltype(GetFramebufferParameteriv)>( GetProcAddress("glGetFramebufferParameteriv") );
		GenRenderbuffers = reinterpret_cast<decltype(GenRenderbuffers)>( GetProcAddress("glGenRenderbuffers") );
		DeleteRenderbuffers = reinterpret_cast<decltype(DeleteRenderbuffers)>( GetProcAddress("glDeleteRenderbuffers") );
		IsRenderbuffer = reinterpret_cast<decltype(IsRenderbuffer)>( GetProcAddress("glIsRenderbuffer") );
		BindRenderbuffer = reinterpret_cast<decltype(BindRenderbuffer)>( GetProcAddress("glBindRenderbuffer") );
		RenderbufferStorage = reinterpret_cast<decltype(RenderbufferStorage)>( GetProcAddress("glRenderbufferStorage") );
		RenderbufferStorageMultisample = reinterpret_cast<decltype(RenderbufferStorageMultisample)>( GetProcAddress("glRenderbufferStorageMultisample") );
		GetRenderbufferParameteriv = reinterpret_cast<decltype(GetRenderbufferParameteriv)>( GetProcAddress("glGetRenderbufferParameteriv") );
		FramebufferRenderbuffer = reinterpret_cast<decltype(FramebufferRenderbuffer)>( GetProcAddress("glFramebufferRenderbuffer") );
		FramebufferTexture = reinterpret_cast<decltype(FramebufferTexture)>( GetProcAddress("glFramebufferTexture") );
		FramebufferTexture1D = reinterpret_cast<decltype(FramebufferTexture1D)>( GetProcAddress("glFramebufferTexture1D") );
		FramebufferTexture2D = reinterpret_cast<decltype(FramebufferTexture2D)>( GetProcAddress("glFramebufferTexture2D") );
		FramebufferTexture3D = reinterpret_cast<decltype(FramebufferTexture3D)>( GetProcAddress("glFramebufferTexture3D") );
		FramebufferTextureLayer = reinterpret_cast<decltype(FramebufferTextureLayer)>( GetProcAddress("glFramebufferTextureLayer") );
		GetFramebufferAttachmentParameteriv = reinterpret_cast<decltype(GetFramebufferAttachmentParameteriv)>( GetProcAddress("glGetFramebufferAttachmentParameteriv") );
		GetShaderPrecisionFormat = reinterpret_cast<decltype(GetShaderPrecisionFormat)>( GetProcAddress("glGetShaderPrecisionFormat") );
		ReleaseShaderCompiler = reinterpret_cast<decltype(ReleaseShaderCompiler)>( GetProcAddress("glReleaseShaderCompiler") );
		CreateShader = reinterpret_cast<decltype(CreateShader)>( GetProcAddress("glCreateShader") );
		DeleteShader = reinterpret_cast<decltype(DeleteShader)>( GetProcAddress("glDeleteShader") );
		ShaderSource = reinterpret_cast<decltype(ShaderSource)>( GetProcAddress("glShaderSource") );
		CompileShader = reinterpret_cast<decltype(CompileShader)>( GetProcAddress("glCompileShader") );
		ShaderBinary = reinterpret_cast<decltype(ShaderBinary)>( GetProcAddress("glShaderBinary") );
		SpecializeShader = reinterpret_cast<decltype(SpecializeShader)>( GetProcAddress("glSpecializeShader") );
		IsShader = reinterpret_cast<decltype(IsShader)>( GetProcAddress("glIsShader") );
		GetShaderiv = reinterpret_cast<decltype(GetShaderiv)>( GetProcAddress("glGetShaderiv") );
		GetShaderSource = reinterpret_cast<decltype(GetShaderSource)>( GetProcAddress("glGetShaderSource") );
		GetShaderInfoLog = reinterpret_cast<decltype(GetShaderInfoLog)>( GetProcAddress("glGetShaderInfoLog") );
		CreateProgram = reinterpret_cast<decltype(CreateProgram)>( GetProcAddress("glCreateProgram") );
		DeleteProgram = reinterpret_cast<decltype(DeleteProgram)>( GetProcAddress("glDeleteProgram") );
		AttachShader = reinterpret_cast<decltype(AttachShader)>( GetProcAddress("glAttachShader") );
		DetachShader = reinterpret_cast<decltype(DetachShader)>( GetProcAddress("glDetachShader") );
		LinkProgram = reinterpret_cast<decltype(LinkProgram)>( GetProcAddress("glLinkProgram") );
		GetAttachedShaders = reinterpret_cast<decltype(GetAttachedShaders)>( GetProcAddress("glGetAttachedShaders") );
		GetProgramBinary = reinterpret_cast<decltype(GetProgramBinary)>( GetProcAddress("glGetProgramBinary") );
		ProgramBinary = reinterpret_cast<decltype(ProgramBinary)>( GetProcAddress("glProgramBinary") );
		IsProgram = reinterpret_cast<decltype(IsProgram)>( GetProcAddress("glIsProgram") );
		UseProgram = reinterpret_cast<decltype(UseProgram)>( GetProcAddress("glUseProgram") );
		ValidateProgram = reinterpret_cast<decltype(ValidateProgram)>( GetProcAddress("glValidateProgram") );
		GetProgramiv = reinterpret_cast<decltype(GetProgramiv)>( GetProcAddress("glGetProgramiv") );
		GetProgramInfoLog = reinterpret_cast<decltype(GetProgramInfoLog)>( GetProcAddress("glGetProgramInfoLog") );
		ProgramParameteri = reinterpret_cast<decltype(ProgramParameteri)>( GetProcAddress("glProgramParameteri") );
		GetProgramStageiv = reinterpret_cast<decltype(GetProgramStageiv)>( GetProcAddress("glGetProgramStageiv") );
		DisableVertexAttribArray = reinterpret_cast<decltype(DisableVertexAttribArray)>( GetProcAddress("glDisableVertexAttribArray") );
		EnableVertexAttribArray = reinterpret_cast<decltype(EnableVertexAttribArray)>( GetProcAddress("glEnableVertexAttribArray") );
		BindAttribLocation = reinterpret_cast<decltype(BindAttribLocation)>( GetProcAddress("glBindAttribLocation") );
		GetAttribLocation = reinterpret_cast<decltype(GetAttribLocation)>( GetProcAddress("glGetAttribLocation") );
		GetActiveAttrib = reinterpret_cast<decltype(GetActiveAttrib)>( GetProcAddress("glGetActiveAttrib") );
		GetVertexAttribPointerv = reinterpret_cast<decltype(GetVertexAttribPointerv)>( GetProcAddress("glGetVertexAttribPointerv") );
		GetVertexAttribdv = reinterpret_cast<decltype(GetVertexAttribdv)>( GetProcAddress("glGetVertexAttribdv") );
		GetVertexAttribfv = reinterpret_cast<decltype(GetVertexAttribfv)>( GetProcAddress("glGetVertexAttribfv") );
		GetVertexAttribiv = reinterpret_cast<decltype(GetVertexAttribiv)>( GetProcAddress("glGetVertexAttribiv") );
		VertexAttrib1d = reinterpret_cast<decltype(VertexAttrib1d)>( GetProcAddress("glVertexAttrib1d") );
		VertexAttrib1dv = reinterpret_cast<decltype(VertexAttrib1dv)>( GetProcAddress("glVertexAttrib1dv") );
		VertexAttrib1f = reinterpret_cast<decltype(VertexAttrib1f)>( GetProcAddress("glVertexAttrib1f") );
		VertexAttrib1fv = reinterpret_cast<decltype(VertexAttrib1fv)>( GetProcAddress("glVertexAttrib1fv") );
		VertexAttrib1s = reinterpret_cast<decltype(VertexAttrib1s)>( GetProcAddress("glVertexAttrib1s") );
		VertexAttrib1sv = reinterpret_cast<decltype(VertexAttrib1sv)>( GetProcAddress("glVertexAttrib1sv") );
		VertexAttrib2d = reinterpret_cast<decltype(VertexAttrib2d)>( GetProcAddress("glVertexAttrib2d") );
		VertexAttrib2dv = reinterpret_cast<decltype(VertexAttrib2dv)>( GetProcAddress("glVertexAttrib2dv") );
		VertexAttrib2f = reinterpret_cast<decltype(VertexAttrib2f)>( GetProcAddress("glVertexAttrib2f") );
		VertexAttrib2fv = reinterpret_cast<decltype(VertexAttrib2fv)>( GetProcAddress("glVertexAttrib2fv") );
		VertexAttrib2s = reinterpret_cast<decltype(VertexAttrib2s)>( GetProcAddress("glVertexAttrib2s") );
		VertexAttrib2sv = reinterpret_cast<decltype(VertexAttrib2sv)>( GetProcAddress("glVertexAttrib2sv") );
		VertexAttrib3d = reinterpret_cast<decltype(VertexAttrib3d)>( GetProcAddress("glVertexAttrib3d") );
		VertexAttrib3dv = reinterpret_cast<decltype(VertexAttrib3dv)>( GetProcAddress("glVertexAttrib3dv") );
		VertexAttrib3f = reinterpret_cast<decltype(VertexAttrib3f)>( GetProcAddress("glVertexAttrib3f") );
		VertexAttrib3fv = reinterpret_cast<decltype(VertexAttrib3fv)>( GetProcAddress("glVertexAttrib3fv") );
		VertexAttrib3s = reinterpret_cast<decltype(VertexAttrib3s)>( GetProcAddress("glVertexAttrib3s") );
		VertexAttrib3sv = reinterpret_cast<decltype(VertexAttrib3sv)>( GetProcAddress("glVertexAttrib3sv") );
		VertexAttrib4Nbv = reinterpret_cast<decltype(VertexAttrib4Nbv)>( GetProcAddress("glVertexAttrib4Nbv") );
		VertexAttrib4Niv = reinterpret_cast<decltype(VertexAttrib4Niv)>( GetProcAddress("glVertexAttrib4Niv") );
		VertexAttrib4Nsv = reinterpret_cast<decltype(VertexAttrib4Nsv)>( GetProcAddress("glVertexAttrib4Nsv") );
		VertexAttrib4Nub = reinterpret_cast<decltype(VertexAttrib4Nub)>( GetProcAddress("glVertexAttrib4Nub") );
		VertexAttrib4Nubv = reinterpret_cast<decltype(VertexAttrib4Nubv)>( GetProcAddress("glVertexAttrib4Nubv") );
		VertexAttrib4Nuiv = reinterpret_cast<decltype(VertexAttrib4Nuiv)>( GetProcAddress("glVertexAttrib4Nuiv") );
		VertexAttrib4Nusv = reinterpret_cast<decltype(VertexAttrib4Nusv)>( GetProcAddress("glVertexAttrib4Nusv") );
		VertexAttrib4bv = reinterpret_cast<decltype(VertexAttrib4bv)>( GetProcAddress("glVertexAttrib4bv") );
		VertexAttrib4d = reinterpret_cast<decltype(VertexAttrib4d)>( GetProcAddress("glVertexAttrib4d") );
		VertexAttrib4dv = reinterpret_cast<decltype(VertexAttrib4dv)>( GetProcAddress("glVertexAttrib4dv") );
		VertexAttrib4f = reinterpret_cast<decltype(VertexAttrib4f)>( GetProcAddress("glVertexAttrib4f") );
		VertexAttrib4fv = reinterpret_cast<decltype(VertexAttrib4fv)>( GetProcAddress("glVertexAttrib4fv") );
		VertexAttrib4iv = reinterpret_cast<decltype(VertexAttrib4iv)>( GetProcAddress("glVertexAttrib4iv") );
		VertexAttrib4s = reinterpret_cast<decltype(VertexAttrib4s)>( GetProcAddress("glVertexAttrib4s") );
		VertexAttrib4sv = reinterpret_cast<decltype(VertexAttrib4sv)>( GetProcAddress("glVertexAttrib4sv") );
		VertexAttrib4ubv = reinterpret_cast<decltype(VertexAttrib4ubv)>( GetProcAddress("glVertexAttrib4ubv") );
		VertexAttrib4uiv = reinterpret_cast<decltype(VertexAttrib4uiv)>( GetProcAddress("glVertexAttrib4uiv") );
		VertexAttrib4usv = reinterpret_cast<decltype(VertexAttrib4usv)>( GetProcAddress("glVertexAttrib4usv") );
		VertexAttribPointer = reinterpret_cast<decltype(VertexAttribPointer)>( GetProcAddress("glVertexAttribPointer") );
		GetVertexAttribIiv = reinterpret_cast<decltype(GetVertexAttribIiv)>( GetProcAddress("glGetVertexAttribIiv") );
		GetVertexAttribIuiv = reinterpret_cast<decltype(GetVertexAttribIuiv)>( GetProcAddress("glGetVertexAttribIuiv") );
		VertexAttribI1i = reinterpret_cast<decltype(VertexAttribI1i)>( GetProcAddress("glVertexAttribI1i") );
		VertexAttribI1iv = reinterpret_cast<decltype(VertexAttribI1iv)>( GetProcAddress("glVertexAttribI1iv") );
		VertexAttribI1ui = reinterpret_cast<decltype(VertexAttribI1ui)>( GetProcAddress("glVertexAttribI1ui") );
		VertexAttribI1uiv = reinterpret_cast<decltype(VertexAttribI1uiv)>( GetProcAddress("glVertexAttribI1uiv") );
		VertexAttribI2i = reinterpret_cast<decltype(VertexAttribI2i)>( GetProcAddress("glVertexAttribI2i") );
		VertexAttribI2iv = reinterpret_cast<decltype(VertexAttribI2iv)>( GetProcAddress("glVertexAttribI2iv") );
		VertexAttribI2ui = reinterpret_cast<decltype(VertexAttribI2ui)>( GetProcAddress("glVertexAttribI2ui") );
		VertexAttribI2uiv = reinterpret_cast<decltype(VertexAttribI2uiv)>( GetProcAddress("glVertexAttribI2uiv") );
		VertexAttribI3i = reinterpret_cast<decltype(VertexAttribI3i)>( GetProcAddress("glVertexAttribI3i") );
		VertexAttribI3iv = reinterpret_cast<decltype(VertexAttribI3iv)>( GetProcAddress("glVertexAttribI3iv") );
		VertexAttribI3ui = reinterpret_cast<decltype(VertexAttribI3ui)>( GetProcAddress("glVertexAttribI3ui") );
		VertexAttribI3uiv = reinterpret_cast<decltype(VertexAttribI3uiv)>( GetProcAddress("glVertexAttribI3uiv") );
		VertexAttribI4bv = reinterpret_cast<decltype(VertexAttribI4bv)>( GetProcAddress("glVertexAttribI4bv") );
		VertexAttribI4i = reinterpret_cast<decltype(VertexAttribI4i)>( GetProcAddress("glVertexAttribI4i") );
		VertexAttribI4iv = reinterpret_cast<decltype(VertexAttribI4iv)>( GetProcAddress("glVertexAttribI4iv") );
		VertexAttribI4sv = reinterpret_cast<decltype(VertexAttribI4sv)>( GetProcAddress("glVertexAttribI4sv") );
		VertexAttribI4ubv = reinterpret_cast<decltype(VertexAttribI4ubv)>( GetProcAddress("glVertexAttribI4ubv") );
		VertexAttribI4ui = reinterpret_cast<decltype(VertexAttribI4ui)>( GetProcAddress("glVertexAttribI4ui") );
		VertexAttribI4uiv = reinterpret_cast<decltype(VertexAttribI4uiv)>( GetProcAddress("glVertexAttribI4uiv") );
		VertexAttribI4usv = reinterpret_cast<decltype(VertexAttribI4usv)>( GetProcAddress("glVertexAttribI4usv") );
		VertexAttribIPointer = reinterpret_cast<decltype(VertexAttribIPointer)>( GetProcAddress("glVertexAttribIPointer") );
		GetVertexAttribLdv = reinterpret_cast<decltype(GetVertexAttribLdv)>( GetProcAddress("glGetVertexAttribLdv") );
		VertexAttribL1d = reinterpret_cast<decltype(VertexAttribL1d)>( GetProcAddress("glVertexAttribL1d") );
		VertexAttribL1dv = reinterpret_cast<decltype(VertexAttribL1dv)>( GetProcAddress("glVertexAttribL1dv") );
		VertexAttribL2d = reinterpret_cast<decltype(VertexAttribL2d)>( GetProcAddress("glVertexAttribL2d") );
		VertexAttribL2dv = reinterpret_cast<decltype(VertexAttribL2dv)>( GetProcAddress("glVertexAttribL2dv") );
		VertexAttribL3d = reinterpret_cast<decltype(VertexAttribL3d)>( GetProcAddress("glVertexAttribL3d") );
		VertexAttribL3dv = reinterpret_cast<decltype(VertexAttribL3dv)>( GetProcAddress("glVertexAttribL3dv") );
		VertexAttribL4d = reinterpret_cast<decltype(VertexAttribL4d)>( GetProcAddress("glVertexAttribL4d") );
		VertexAttribL4dv = reinterpret_cast<decltype(VertexAttribL4dv)>( GetProcAddress("glVertexAttribL4dv") );
		VertexAttribLPointer = reinterpret_cast<decltype(VertexAttribLPointer)>( GetProcAddress("glVertexAttribLPointer") );
		GetActiveUniformsiv = reinterpret_cast<decltype(GetActiveUniformsiv)>( GetProcAddress("glGetActiveUniformsiv") );
		GetActiveUniform = reinterpret_cast<decltype(GetActiveUniform)>( GetProcAddress("glGetActiveUniform") );
		GetActiveUniformName = reinterpret_cast<decltype(GetActiveUniformName)>( GetProcAddress("glGetActiveUniformName") );
		GetUniformLocation = reinterpret_cast<decltype(GetUniformLocation)>( GetProcAddress("glGetUniformLocation") );
		GetUniformIndices = reinterpret_cast<decltype(GetUniformIndices)>( GetProcAddress("glGetUniformIndices") );
		GetUniformiv = reinterpret_cast<decltype(GetUniformiv)>( GetProcAddress("glGetUniformiv") );
		GetUniformuiv = reinterpret_cast<decltype(GetUniformuiv)>( GetProcAddress("glGetUniformuiv") );
		GetUniformfv = reinterpret_cast<decltype(GetUniformfv)>( GetProcAddress("glGetUniformfv") );
		GetUniformdv = reinterpret_cast<decltype(GetUniformdv)>( GetProcAddress("glGetUniformdv") );
		Uniform1f = reinterpret_cast<decltype(Uniform1f)>( GetProcAddress("glUniform1f") );
		Uniform1fv = reinterpret_cast<decltype(Uniform1fv)>( GetProcAddress("glUniform1fv") );
		Uniform1i = reinterpret_cast<decltype(Uniform1i)>( GetProcAddress("glUniform1i") );
		Uniform1ui = reinterpret_cast<decltype(Uniform1ui)>( GetProcAddress("glUniform1ui") );
		Uniform1iv = reinterpret_cast<decltype(Uniform1iv)>( GetProcAddress("glUniform1iv") );
		Uniform1uiv = reinterpret_cast<decltype(Uniform1uiv)>( GetProcAddress("glUniform1uiv") );
		Uniform2f = reinterpret_cast<decltype(Uniform2f)>( GetProcAddress("glUniform2f") );
		Uniform2fv = reinterpret_cast<decltype(Uniform2fv)>( GetProcAddress("glUniform2fv") );
		Uniform2i = reinterpret_cast<decltype(Uniform2i)>( GetProcAddress("glUniform2i") );
		Uniform2ui = reinterpret_cast<decltype(Uniform2ui)>( GetProcAddress("glUniform2ui") );
		Uniform2iv = reinterpret_cast<decltype(Uniform2iv)>( GetProcAddress("glUniform2iv") );
		Uniform2uiv = reinterpret_cast<decltype(Uniform2uiv)>( GetProcAddress("glUniform2uiv") );
		Uniform3f = reinterpret_cast<decltype(Uniform3f)>( GetProcAddress("glUniform3f") );
		Uniform3fv = reinterpret_cast<decltype(Uniform3fv)>( GetProcAddress("glUniform3fv") );
		Uniform3i = reinterpret_cast<decltype(Uniform3i)>( GetProcAddress("glUniform3i") );
		Uniform3ui = reinterpret_cast<decltype(Uniform3ui)>( GetProcAddress("glUniform3ui") );
		Uniform3iv = reinterpret_cast<decltype(Uniform3iv)>( GetProcAddress("glUniform3iv") );
		Uniform3uiv = reinterpret_cast<decltype(Uniform3uiv)>( GetProcAddress("glUniform3uiv") );
		Uniform4f = reinterpret_cast<decltype(Uniform4f)>( GetProcAddress("glUniform4f") );
		Uniform4fv = reinterpret_cast<decltype(Uniform4fv)>( GetProcAddress("glUniform4fv") );
		Uniform4i = reinterpret_cast<decltype(Uniform4i)>( GetProcAddress("glUniform4i") );
		Uniform4ui = reinterpret_cast<decltype(Uniform4ui)>( GetProcAddress("glUniform4ui") );
		Uniform4iv = reinterpret_cast<decltype(Uniform4iv)>( GetProcAddress("glUniform4iv") );
		Uniform4uiv = reinterpret_cast<decltype(Uniform4uiv)>( GetProcAddress("glUniform4uiv") );
		Uniform1d = reinterpret_cast<decltype(Uniform1d)>( GetProcAddress("glUniform1d") );
		Uniform1dv = reinterpret_cast<decltype(Uniform1dv)>( GetProcAddress("glUniform1dv") );
		Uniform2d = reinterpret_cast<decltype(Uniform2d)>( GetProcAddress("glUniform2d") );
		Uniform2dv = reinterpret_cast<decltype(Uniform2dv)>( GetProcAddress("glUniform2dv") );
		Uniform3d = reinterpret_cast<decltype(Uniform3d)>( GetProcAddress("glUniform3d") );
		Uniform3dv = reinterpret_cast<decltype(Uniform3dv)>( GetProcAddress("glUniform3dv") );
		Uniform4d = reinterpret_cast<decltype(Uniform4d)>( GetProcAddress("glUniform4d") );
		Uniform4dv = reinterpret_cast<decltype(Uniform4dv)>( GetProcAddress("glUniform4dv") );
		UniformMatrix2fv = reinterpret_cast<decltype(UniformMatrix2fv)>( GetProcAddress("glUniformMatrix2fv") );
		UniformMatrix3fv = reinterpret_cast<decltype(UniformMatrix3fv)>( GetProcAddress("glUniformMatrix3fv") );
		UniformMatrix4fv = reinterpret_cast<decltype(UniformMatrix4fv)>( GetProcAddress("glUniformMatrix4fv") );
		UniformMatrix2x3fv = reinterpret_cast<decltype(UniformMatrix2x3fv)>( GetProcAddress("glUniformMatrix2x3fv") );
		UniformMatrix2x4fv = reinterpret_cast<decltype(UniformMatrix2x4fv)>( GetProcAddress("glUniformMatrix2x4fv") );
		UniformMatrix3x2fv = reinterpret_cast<decltype(UniformMatrix3x2fv)>( GetProcAddress("glUniformMatrix3x2fv") );
		UniformMatrix3x4fv = reinterpret_cast<decltype(UniformMatrix3x4fv)>( GetProcAddress("glUniformMatrix3x4fv") );
		UniformMatrix4x2fv = reinterpret_cast<decltype(UniformMatrix4x2fv)>( GetProcAddress("glUniformMatrix4x2fv") );
		UniformMatrix4x3fv = reinterpret_cast<decltype(UniformMatrix4x3fv)>( GetProcAddress("glUniformMatrix4x3fv") );
		UniformMatrix2dv = reinterpret_cast<decltype(UniformMatrix2dv)>( GetProcAddress("glUniformMatrix2dv") );
		UniformMatrix2x3dv = reinterpret_cast<decltype(UniformMatrix2x3dv)>( GetProcAddress("glUniformMatrix2x3dv") );
		UniformMatrix2x4dv = reinterpret_cast<decltype(UniformMatrix2x4dv)>( GetProcAddress("glUniformMatrix2x4dv") );
		UniformMatrix3dv = reinterpret_cast<decltype(UniformMatrix3dv)>( GetProcAddress("glUniformMatrix3dv") );
		UniformMatrix3x2dv = reinterpret_cast<decltype(UniformMatrix3x2dv)>( GetProcAddress("glUniformMatrix3x2dv") );
		UniformMatrix3x4dv = reinterpret_cast<decltype(UniformMatrix3x4dv)>( GetProcAddress("glUniformMatrix3x4dv") );
		UniformMatrix4dv = reinterpret_cast<decltype(UniformMatrix4dv)>( GetProcAddress("glUniformMatrix4dv") );
		UniformMatrix4x2dv = reinterpret_cast<decltype(UniformMatrix4x2dv)>( GetProcAddress("glUniformMatrix4x2dv") );
		UniformMatrix4x3dv = reinterpret_cast<decltype(UniformMatrix4x3dv)>( GetProcAddress("glUniformMatrix4x3dv") );
		GetUniformBlockIndex = reinterpret_cast<decltype(GetUniformBlockIndex)>( GetProcAddress("glGetUniformBlockIndex") );
		UniformBlockBinding = reinterpret_cast<decltype(UniformBlockBinding)>( GetProcAddress("glUniformBlockBinding") );
		ShaderStorageBlockBinding = reinterpret_cast<decltype(ShaderStorageBlockBinding)>( GetProcAddress("glShaderStorageBlockBinding") );
		GetActiveUniformBlockiv = reinterpret_cast<decltype(GetActiveUniformBlockiv)>( GetProcAddress("glGetActiveUniformBlockiv") );
		GetActiveUniformBlockName = reinterpret_cast<decltype(GetActiveUniformBlockName)>( GetProcAddress("glGetActiveUniformBlockName") );
		GetSubroutineIndex = reinterpret_cast<decltype(GetSubroutineIndex)>( GetProcAddress("glGetSubroutineIndex") );
		GetActiveSubroutineName = reinterpret_cast<decltype(GetActiveSubroutineName)>( GetProcAddress("glGetActiveSubroutineName") );
		GetSubroutineUniformLocation = reinterpret_cast<decltype(GetSubroutineUniformLocation)>( GetProcAddress("glGetSubroutineUniformLocation") );
		GetActiveSubroutineUniformName = reinterpret_cast<decltype(GetActiveSubroutineUniformName)>( GetProcAddress("glGetActiveSubroutineUniformName") );
		GetActiveSubroutineUniformiv = reinterpret_cast<decltype(GetActiveSubroutineUniformiv)>( GetProcAddress("glGetActiveSubroutineUniformiv") );
		GetUniformSubroutineuiv = reinterpret_cast<decltype(GetUniformSubroutineuiv)>( GetProcAddress("glGetUniformSubroutineuiv") );
		UniformSubroutinesuiv = reinterpret_cast<decltype(UniformSubroutinesuiv)>( GetProcAddress("glUniformSubroutinesuiv") );
		GenVertexArrays = reinterpret_cast<decltype(GenVertexArrays)>( GetProcAddress("glGenVertexArrays") );
		DeleteVertexArrays = reinterpret_cast<decltype(DeleteVertexArrays)>( GetProcAddress("glDeleteVertexArrays") );
		IsVertexArray = reinterpret_cast<decltype(IsVertexArray)>( GetProcAddress("glIsVertexArray") );
		BindVertexArray = reinterpret_cast<decltype(BindVertexArray)>( GetProcAddress("glBindVertexArray") );
		BindVertexBuffers = reinterpret_cast<decltype(BindVertexBuffers)>( GetProcAddress("glBindVertexBuffers") );
		VertexAttribDivisor = reinterpret_cast<decltype(VertexAttribDivisor)>( GetProcAddress("glVertexAttribDivisor") );
		BindFragDataLocation = reinterpret_cast<decltype(BindFragDataLocation)>( GetProcAddress("glBindFragDataLocation") );
		GetFragDataLocation = reinterpret_cast<decltype(GetFragDataLocation)>( GetProcAddress("glGetFragDataLocation") );
		Scissor = reinterpret_cast<decltype(Scissor)>( GetProcAddress("glScissor") );
		Viewport = reinterpret_cast<decltype(Viewport)>( GetProcAddress("glViewport") );
		WindowPos2d = reinterpret_cast<decltype(WindowPos2d)>( GetProcAddress("glWindowPos2d") );
		WindowPos2dv = reinterpret_cast<decltype(WindowPos2dv)>( GetProcAddress("glWindowPos2dv") );
		WindowPos3d = reinterpret_cast<decltype(WindowPos3d)>( GetProcAddress("glWindowPos3d") );
		WindowPos3dv = reinterpret_cast<decltype(WindowPos3dv)>( GetProcAddress("glWindowPos3dv") );
		LogicOp = reinterpret_cast<decltype(LogicOp)>( GetProcAddress("glLogicOp") );
		Clear = reinterpret_cast<decltype(Clear)>( GetProcAddress("glClear") );
		ClearColor = reinterpret_cast<decltype(ClearColor)>( GetProcAddress("glClearColor") );
		ClearDepth = reinterpret_cast<decltype(ClearDepth)>( GetProcAddress("glClearDepth") );
		ClearDepthf = reinterpret_cast<decltype(ClearDepthf)>( GetProcAddress("glClearDepthf") );
		ClearStencil = reinterpret_cast<decltype(ClearStencil)>( GetProcAddress("glClearStencil") );
		ColorMask = reinterpret_cast<decltype(ColorMask)>( GetProcAddress("glColorMask") );
		ColorMaski = reinterpret_cast<decltype(ColorMaski)>( GetProcAddress("glColorMaski") );
		DepthMask = reinterpret_cast<decltype(DepthMask)>( GetProcAddress("glDepthMask") );
		DepthRange = reinterpret_cast<decltype(DepthRange)>( GetProcAddress("glDepthRange") );
		DepthRangef = reinterpret_cast<decltype(DepthRangef)>( GetProcAddress("glDepthRangef") );
		StencilMask = reinterpret_cast<decltype(StencilMask)>( GetProcAddress("glStencilMask") );
		AlphaFunc = reinterpret_cast<decltype(AlphaFunc)>( GetProcAddress("glAlphaFunc") );
		DepthFunc = reinterpret_cast<decltype(DepthFunc)>( GetProcAddress("glDepthFunc") );
		StencilFunc = reinterpret_cast<decltype(StencilFunc)>( GetProcAddress("glStencilFunc") );
		StencilOp = reinterpret_cast<decltype(StencilOp)>( GetProcAddress("glStencilOp") );
		StencilFuncSeparate = reinterpret_cast<decltype(StencilFuncSeparate)>( GetProcAddress("glStencilFuncSeparate") );
		StencilMaskSeparate = reinterpret_cast<decltype(StencilMaskSeparate)>( GetProcAddress("glStencilMaskSeparate") );
		StencilOpSeparate = reinterpret_cast<decltype(StencilOpSeparate)>( GetProcAddress("glStencilOpSeparate") );
		BlendFunc = reinterpret_cast<decltype(BlendFunc)>( GetProcAddress("glBlendFunc") );
		BlendFunci = reinterpret_cast<decltype(BlendFunci)>( GetProcAddress("glBlendFunci") );
		BlendColor = reinterpret_cast<decltype(BlendColor)>( GetProcAddress("glBlendColor") );
		BlendFuncSeparate = reinterpret_cast<decltype(BlendFuncSeparate)>( GetProcAddress("glBlendFuncSeparate") );
		BlendFuncSeparatei = reinterpret_cast<decltype(BlendFuncSeparatei)>( GetProcAddress("glBlendFuncSeparatei") );
		BlendEquation = reinterpret_cast<decltype(BlendEquation)>( GetProcAddress("glBlendEquation") );
		BlendEquationi = reinterpret_cast<decltype(BlendEquationi)>( GetProcAddress("glBlendEquationi") );
		BlendEquationSeparate = reinterpret_cast<decltype(BlendEquationSeparate)>( GetProcAddress("glBlendEquationSeparate") );
		BlendEquationSeparatei = reinterpret_cast<decltype(BlendEquationSeparatei)>( GetProcAddress("glBlendEquationSeparatei") );
		SampleCoverage = reinterpret_cast<decltype(SampleCoverage)>( GetProcAddress("glSampleCoverage") );
		MinSampleShading = reinterpret_cast<decltype(MinSampleShading)>( GetProcAddress("glMinSampleShading") );
		PointSize = reinterpret_cast<decltype(PointSize)>( GetProcAddress("glPointSize") );
		PointParameterf = reinterpret_cast<decltype(PointParameterf)>( GetProcAddress("glPointParameterf") );
		PointParameterfv = reinterpret_cast<decltype(PointParameterfv)>( GetProcAddress("glPointParameterfv") );
		PointParameteri = reinterpret_cast<decltype(PointParameteri)>( GetProcAddress("glPointParameteri") );
		PointParameteriv = reinterpret_cast<decltype(PointParameteriv)>( GetProcAddress("glPointParameteriv") );
		LineWidth = reinterpret_cast<decltype(LineWidth)>( GetProcAddress("glLineWidth") );
		PolygonMode = reinterpret_cast<decltype(PolygonMode)>( GetProcAddress("glPolygonMode") );
		PolygonOffset = reinterpret_cast<decltype(PolygonOffset)>( GetProcAddress("glPolygonOffset") );
		PolygonStipple = reinterpret_cast<decltype(PolygonStipple)>( GetProcAddress("glPolygonStipple") );
		FrontFace = reinterpret_cast<decltype(FrontFace)>( GetProcAddress("glFrontFace") );
		CullFace = reinterpret_cast<decltype(CullFace)>( GetProcAddress("glCullFace") );
		ClipPlane = reinterpret_cast<decltype(ClipPlane)>( GetProcAddress("glClipPlane") );
		GetClipPlane = reinterpret_cast<decltype(GetClipPlane)>( GetProcAddress("glGetClipPlane") );
		PatchParameterfv = reinterpret_cast<decltype(PatchParameterfv)>( GetProcAddress("glPatchParameterfv") );
		PatchParameteri = reinterpret_cast<decltype(PatchParameteri)>( GetProcAddress("glPatchParameteri") );
		DrawArrays = reinterpret_cast<decltype(DrawArrays)>( GetProcAddress("glDrawArrays") );
		DrawElements = reinterpret_cast<decltype(DrawElements)>( GetProcAddress("glDrawElements") );
		DrawPixels = reinterpret_cast<decltype(DrawPixels)>( GetProcAddress("glDrawPixels") );
		DrawRangeElements = reinterpret_cast<decltype(DrawRangeElements)>( GetProcAddress("glDrawRangeElements") );
		DrawArraysInstanced = reinterpret_cast<decltype(DrawArraysInstanced)>( GetProcAddress("glDrawArraysInstanced") );
		DrawElementsInstanced = reinterpret_cast<decltype(DrawElementsInstanced)>( GetProcAddress("glDrawElementsInstanced") );
		DrawElementsBaseVertex = reinterpret_cast<decltype(DrawElementsBaseVertex)>( GetProcAddress("glDrawElementsBaseVertex") );
		DrawElementsInstancedBaseVertex = reinterpret_cast<decltype(DrawElementsInstancedBaseVertex)>( GetProcAddress("glDrawElementsInstancedBaseVertex") );
		DrawRangeElementsBaseVertex = reinterpret_cast<decltype(DrawRangeElementsBaseVertex)>( GetProcAddress("glDrawRangeElementsBaseVertex") );
		DrawTransformFeedback = reinterpret_cast<decltype(DrawTransformFeedback)>( GetProcAddress("glDrawTransformFeedback") );
		DrawTransformFeedbackStream = reinterpret_cast<decltype(DrawTransformFeedbackStream)>( GetProcAddress("glDrawTransformFeedbackStream") );
		DrawElementsInstancedBaseVertexBaseInstance = reinterpret_cast<decltype(DrawElementsInstancedBaseVertexBaseInstance)>( GetProcAddress("glDrawElementsInstancedBaseVertexBaseInstance") );
		DrawElementsInstancedBaseInstance = reinterpret_cast<decltype(DrawElementsInstancedBaseInstance)>( GetProcAddress("glDrawElementsInstancedBaseInstance") );
		DrawArraysInstancedBaseInstance = reinterpret_cast<decltype(DrawArraysInstancedBaseInstance)>( GetProcAddress("glDrawArraysInstancedBaseInstance") );
		DrawTransformFeedbackInstanced = reinterpret_cast<decltype(DrawTransformFeedbackInstanced)>( GetProcAddress("glDrawTransformFeedbackInstanced") );
		DrawTransformFeedbackStreamInstanced = reinterpret_cast<decltype(DrawTransformFeedbackStreamInstanced)>( GetProcAddress("glDrawTransformFeedbackStreamInstanced") );
		MultiDrawElementsBaseVertex = reinterpret_cast<decltype(MultiDrawElementsBaseVertex)>( GetProcAddress("glMultiDrawElementsBaseVertex") );
		MultiDrawArrays = reinterpret_cast<decltype(MultiDrawArrays)>( GetProcAddress("glMultiDrawArrays") );
		MultiDrawElements = reinterpret_cast<decltype(MultiDrawElements)>( GetProcAddress("glMultiDrawElements") );
		MultiDrawArraysIndirectCount = reinterpret_cast<decltype(MultiDrawArraysIndirectCount)>( GetProcAddress("glMultiDrawArraysIndirectCount") );
		MultiDrawElementsIndirectCount = reinterpret_cast<decltype(MultiDrawElementsIndirectCount)>( GetProcAddress("glMultiDrawElementsIndirectCount") );
		PrimitiveRestartIndex = reinterpret_cast<decltype(PrimitiveRestartIndex)>( GetProcAddress("glPrimitiveRestartIndex") );
		ClampColor = reinterpret_cast<decltype(ClampColor)>( GetProcAddress("glClampColor") );
		ReadPixels = reinterpret_cast<decltype(ReadPixels)>( GetProcAddress("glReadPixels") );
		DrawBuffer = reinterpret_cast<decltype(DrawBuffer)>( GetProcAddress("glDrawBuffer") );
		DrawBuffers = reinterpret_cast<decltype(DrawBuffers)>( GetProcAddress("glDrawBuffers") );
		ReadBuffer = reinterpret_cast<decltype(ReadBuffer)>( GetProcAddress("glReadBuffer") );
		ClearBufferfi = reinterpret_cast<decltype(ClearBufferfi)>( GetProcAddress("glClearBufferfi") );
		ClearBufferfv = reinterpret_cast<decltype(ClearBufferfv)>( GetProcAddress("glClearBufferfv") );
		ClearBufferiv = reinterpret_cast<decltype(ClearBufferiv)>( GetProcAddress("glClearBufferiv") );
		ClearBufferuiv = reinterpret_cast<decltype(ClearBufferuiv)>( GetProcAddress("glClearBufferuiv") );
		DispatchCompute = reinterpret_cast<decltype(DispatchCompute)>( GetProcAddress("glDispatchCompute") );
		DispatchComputeIndirect = reinterpret_cast<decltype(DispatchComputeIndirect)>( GetProcAddress("glDispatchComputeIndirect") );
		Finish = reinterpret_cast<decltype(Finish)>( GetProcAddress("glFinish") );
		Flush = reinterpret_cast<decltype(Flush)>( GetProcAddress("glFlush") );
		GenQueries = reinterpret_cast<decltype(GenQueries)>( GetProcAddress("glGenQueries") );
		DeleteQueries = reinterpret_cast<decltype(DeleteQueries)>( GetProcAddress("glDeleteQueries") );
		IsQuery = reinterpret_cast<decltype(IsQuery)>( GetProcAddress("glIsQuery") );
		QueryCounter = reinterpret_cast<decltype(QueryCounter)>( GetProcAddress("glQueryCounter") );
		BeginQuery = reinterpret_cast<decltype(BeginQuery)>( GetProcAddress("glBeginQuery") );
		BeginQueryIndexed = reinterpret_cast<decltype(BeginQueryIndexed)>( GetProcAddress("glBeginQueryIndexed") );
		EndQuery = reinterpret_cast<decltype(EndQuery)>( GetProcAddress("glEndQuery") );
		EndQueryIndexed = reinterpret_cast<decltype(EndQueryIndexed)>( GetProcAddress("glEndQueryIndexed") );
		GetQueryiv = reinterpret_cast<decltype(GetQueryiv)>( GetProcAddress("glGetQueryiv") );
		GetQueryIndexediv = reinterpret_cast<decltype(GetQueryIndexediv)>( GetProcAddress("glGetQueryIndexediv") );
		GetQueryObjectiv = reinterpret_cast<decltype(GetQueryObjectiv)>( GetProcAddress("glGetQueryObjectiv") );
		GetQueryObjectuiv = reinterpret_cast<decltype(GetQueryObjectuiv)>( GetProcAddress("glGetQueryObjectuiv") );
		GetQueryObjecti64v = reinterpret_cast<decltype(GetQueryObjecti64v)>( GetProcAddress("glGetQueryObjecti64v") );
		GetQueryObjectui64v = reinterpret_cast<decltype(GetQueryObjectui64v)>( GetProcAddress("glGetQueryObjectui64v") );
		BeginConditionalRender = reinterpret_cast<decltype(BeginConditionalRender)>( GetProcAddress("glBeginConditionalRender") );
		EndConditionalRender = reinterpret_cast<decltype(EndConditionalRender)>( GetProcAddress("glEndConditionalRender") );
		FenceSync = reinterpret_cast<decltype(FenceSync)>( GetProcAddress("glFenceSync") );
		DeleteSync = reinterpret_cast<decltype(DeleteSync)>( GetProcAddress("glDeleteSync") );
		IsSync = reinterpret_cast<decltype(IsSync)>( GetProcAddress("glIsSync") );
		GetSynciv = reinterpret_cast<decltype(GetSynciv)>( GetProcAddress("glGetSynciv") );
		WaitSync = reinterpret_cast<decltype(WaitSync)>( GetProcAddress("glWaitSync") );
		MemoryBarrier_ = reinterpret_cast<decltype(MemoryBarrier_)>( GetProcAddress("glMemoryBarrier") );
	}
#pragma warning(default: 26495)
};


/** 
 * @code 
glfwInit();
GLFWwindow* window = glfwCreateWindow(1024, 1024, "Example", nullptr, nullptr);
wglMakeCurrent(GetDC(glfwGetWin32Window(window)), wglCreateContext(GetDC(glfwGetWin32Window(window))));
GLlibrary gl = GLlibrary("opengl32.dll"); 
*/

/// Buffer not have target.
///@code
/// GLuint buffer1;
/// gl.GenBuffers(1, &buffer1);
/// gl.BindBuffer(GL_ARRAY_BUFFER, buffer1);
/// GLuint buffer2;
/// gl.GenBuffers(1, &buffer2);
/// gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, buffer2);
/// GLenum error = gl.GetError();
///
/// GLint binding;
/// gl.GetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &binding);
/// gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, buffer1);
/// GLenum error2 = gl.GetError();
/// gl.GetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &binding);
/// 

/// Get binding buffer
///@code
/// switch (target) {
/// case GL_ARRAY_BUFFER: 
/// 	gl.GetIntegerv(GL_ARRAY_BUFFER_BINDING, (GLint*)(&buffer)); break;
/// case GL_ELEMENT_ARRAY_BUFFER: 
/// 	gl.GetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, (GLint*)(&buffer)); break;
/// case GL_PIXEL_PACK_BUFFER:
/// 	gl.GetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, (GLint*)(&buffer)); break;
/// case GL_PIXEL_UNPACK_BUFFER:
/// 	gl.GetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, (GLint*)(&buffer)); break;
/// case GL_TRANSFORM_FEEDBACK_BUFFER:
/// 	gl.GetIntegerv(GL_TRANSFORM_FEEDBACK_BUFFER_BINDING, (GLint*)(&buffer)); break;
/// case GL_UNIFORM_BUFFER:
/// 	gl.GetIntegerv(GL_UNIFORM_BUFFER_BINDING, (GLint*)(&buffer)); break;
/// case GL_ATOMIC_COUNTER_BUFFER:
/// 	gl.GetIntegerv(GL_ATOMIC_COUNTER_BUFFER_BINDING, (GLint*)(&buffer)); break;
/// case GL_DISPATCH_INDIRECT_BUFFER:
/// 	gl.GetIntegerv(GL_DISPATCH_INDIRECT_BUFFER_BINDING, (GLint*)(&buffer)); break;
/// case GL_SHADER_STORAGE_BUFFER:
/// 	gl.GetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, (GLint*)(&buffer)); break;
/// case GL_QUERY_BUFFER:
/// 	gl.GetIntegerv(GL_QUERY_BUFFER_BINDING, (GLint*)(&buffer)); break;
/// case GL_COPY_READ_BUFFER:
/// 	gl.GetIntegerv(GL_COPY_READ_BUFFER_BINDING, (GLint*)(&buffer)); break;
/// case GL_COPY_WRITE_BUFFER:
/// 	gl.GetIntegerv(GL_COPY_WRITE_BUFFER_BINDING, (GLint*)(&buffer)); break;
/// default:
/// 	break;
/// }
/// 

/// GLsampler initial values
/// "https://docs.gl/gl4/glGetSamplerParameter"
/// "https://docs.gl/gl4/glSamplerParameter"
///@code
/// GLuint buffer;
/// GLuint texture;
/// GLuint sampler;
/// gl.GenBuffers(1, &buffer);
/// gl.GenTextures(1, &texture);
/// gl.GenSamplers(1, &sampler);
/// if (gl.IsSampler(sampler)) {
/// 	GLint magFilter;
/// 	GLint minFilter;
/// 	GLint minLod;
/// 	GLint maxLod;
/// 	GLint mipLodBias;
/// 	GLint addressModeU;
/// 	GLint addressModeV;
/// 	GLint addressModeW;
/// 	GLint compareEnable;
/// 	GLint compareOp;
/// 	GLfloat borderColor[4];
/// 	gl.GetSamplerParameteriv(sampler, GL_TEXTURE_MAG_FILTER, &magFilter);
/// 	gl.GetSamplerParameteriv(sampler, GL_TEXTURE_MIN_FILTER, &minFilter);
/// 	gl.GetSamplerParameteriv(sampler, GL_TEXTURE_MIN_LOD, &minLod);
/// 	gl.GetSamplerParameteriv(sampler, GL_TEXTURE_MAX_LOD, &maxLod);
/// 	gl.GetSamplerParameteriv(sampler, GL_TEXTURE_LOD_BIAS, &mipLodBias);
/// 	gl.GetSamplerParameteriv(sampler, GL_TEXTURE_WRAP_S, &addressModeU);
/// 	gl.GetSamplerParameteriv(sampler, GL_TEXTURE_WRAP_T, &addressModeV);
/// 	gl.GetSamplerParameteriv(sampler, GL_TEXTURE_WRAP_R, &addressModeW);
/// 	gl.GetSamplerParameteriv(sampler, GL_TEXTURE_COMPARE_MODE, &compareEnable);
/// 	gl.GetSamplerParameteriv(sampler, GL_TEXTURE_COMPARE_FUNC, &compareOp);
/// 	gl.GetSamplerParameterfv(sampler, GL_TEXTURE_BORDER_COLOR, borderColor);
/// 	GLenum error = gl.GetError();
/// 	std::cin.get();
/// }
///

/// Enum GLuniform-constants and GLuniformbuffer-members
///@code
/// auto& gl = this->gl.gl;
/// GLuint program = *reinterpret_cast<GLuint*>(this->pipeline);
/// for (GLuint index = 0; true; ++index) {
/// 	GLsizei maxlength = 0;
/// 	gl.GetActiveUniformsiv(program, 1, &index, GL_UNIFORM_NAME_LENGTH, (GLint*)(&maxlength));
/// 	if (maxlength == 0)
/// 		break;
/// 	GLint size;
/// 	GLenum type;
/// 	GLsizei length;
/// 	std::string name(maxlength, '\0');
/// 	gl.GetActiveUniform(program, index, maxlength,
/// 		&length,
/// 		&size,
/// 		&type,
/// 		&name[0]);
/// 	if (length != maxlength)
/// 		name.resize(length);
/// 	if (size != 1)//assert(name.find("[0]") != std::string::npos);
/// 		name.erase(name.size() - 3, 3);
/// 
/// 	std::cout << index << ": layout(location = "<<gl.GetUniformLocation(program, name.c_str())<<") uniform " << type << " " << name << "["<<size<<"]" << std::endl;
/// }
/// 

/// Enum GLuniformbuffers
///@code
/// auto& gl = this->gl.gl;
/// GLuint program = *reinterpret_cast<GLuint*>(this->pipeline);
/// for (GLuint index = 0; true; ++index) {
/// 	GLsizei maxlength;
/// 	gl.GetActiveUniformBlockiv(program, index, GL_UNIFORM_BLOCK_NAME_LENGTH, (GLint*)(&maxlength));
/// 	GLsizei length;
/// 	std::string name(maxlength, '\0');
/// 	gl.GetActiveUniformBlockName(program, index, maxlength, &length, &name[0]);
/// 	if (length != maxlength)
/// 		name.resize(length);
/// 	if (gl.GetUniformBlockIndex(program, name.c_str()) != index)
/// 		break;
/// 
/// 	GLsizei datasize;
/// 	gl.GetActiveUniformBlockiv(program, index, GL_UNIFORM_BLOCK_DATA_SIZE, (GLint*)(&datasize));
/// 	GLuint binding;
/// 	gl.GetActiveUniformBlockiv(program, index, GL_UNIFORM_BLOCK_BINDING, (GLint*)(&binding));
/// 	std::cout << index << ": layout(binding = "<<binding<<") uniform " << name << "{\n";
/// 
/// 	GLsizei uniforms;
/// 	gl.GetActiveUniformBlockiv(program, index, GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS, (GLint*)(&uniforms));
/// 	if (uniforms != 0) {
/// 		std::vector<GLuint> indices(uniforms, -1);
/// 		std::vector<GLenum> types(uniforms, -1);
/// 		std::vector<GLenum> sizes(uniforms, -1);
/// 		std::vector<GLsizei> offsets(uniforms, -1);
/// 		std::vector<GLsizei> namelengths(uniforms, -1);
/// 		gl.GetActiveUniformBlockiv(program, index, GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES, (GLint*)(&indices[0]));
/// 		gl.GetActiveUniformsiv(program, uniforms, &indices[0], GL_UNIFORM_TYPE, (GLint*)(&types[0]));
/// 		gl.GetActiveUniformsiv(program, uniforms, &indices[0], GL_UNIFORM_SIZE, (GLint*)(&sizes[0]));
/// 		gl.GetActiveUniformsiv(program, uniforms, &indices[0], GL_UNIFORM_OFFSET, (GLint*)(&offsets[0]));
/// 		gl.GetActiveUniformsiv(program, uniforms, &indices[0], GL_UNIFORM_NAME_LENGTH, (GLint*)(&namelengths[0]));
/// 		for (GLsizei i = 0; i != uniforms; ++i) {
/// 			std::cout << "  layout(offset = "<<offsets[i]<<") "<<types[i]<<" NAME["<<sizes[i]<<"]\n";
/// 		}
/// 		//GLuint testBlockIndex;
/// 		//gl.GetActiveUniformsiv(program, 1, &indices[0], GL_UNIFORM_BLOCK_INDEX, (GLint*)(&testBlockIndex));
/// 		//assert(testBlockIndex == index);
/// 	}
/// 	std::cout << "}" << std::endl;
/// }
/// 

/// gl.DrawBuffers(n,bufs) cannot used to '0'(default)
///@code
/// GLenum drawbuffers[1] = { GL_COLOR_ATTACHMENT0 };
/// gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
/// bool compelte = gl.CheckFramebufferStatus() == GL_FRAMEBUFFER_COMPLETE;
/// 
/// GLuint framebuffer;
/// gl.GenFramebuffers(1, &framebuffer);
/// gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
/// compelte = gl.CheckFramebufferStatus() == GL_FRAMEBUFFER_COMPLETE;
/// 
/// GLuint current;
/// gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
/// gl.GetIntegerv(GL_FRAMEBUFFER_BINDING, (GLint*)&current);
/// gl.DrawBuffers(1, drawbuffers);
/// GLenum error = gl.GetError();
/// 
/// gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
/// gl.GetIntegerv(GL_FRAMEBUFFER_BINDING, (GLint*)&current);
/// gl.DrawBuffers(1, drawbuffers);
/// GLenum error2 = gl.GetError();
/// 