#pragma once

/// Platform of OpenGL. 
///@license Free 
///@review 2022-6-9 
///@contact Jiang1998Nan@outlook.com 
#define _OPENGL_PLATFORM_

#ifdef _WIN32
#include <windows.h>
#undef min
#undef max
#undef near
#undef far
#include <wingdi.h>
//#pragma comment(lib, "opengl32.lib") // copy to "main.cpp"

/// Library of opengl32.dll.
class WGLlibrary {
	HMODULE opengl32dll;
public:
	enum class Errors {
		NoError,
		EmptyLibrary,
		ErrorLibrary,
		NotHaveContext
	} error;

	WGLlibrary() : opengl32dll(nullptr),
		error(Errors::EmptyLibrary){}

	WGLlibrary(LPCWSTR path) : opengl32dll(LoadLibrary(path)),
		error(opengl32dll != nullptr ? Errors::NoError : Errors::ErrorLibrary) {}

	virtual ~WGLlibrary() {
		FreeLibrary(opengl32dll);
		opengl32dll = nullptr;
		error = Errors::EmptyLibrary;
	}

	bool operator==(nullptr_t) const {
		return opengl32dll == nullptr;
	}

	bool operator!=(nullptr_t) const {
		return opengl32dll != nullptr;
	}

	PROC GetProcAddress(LPCSTR name) {
		// Check library
		if (error != Errors::NoError) {
			return nullptr;
		}

		// Get from GLbasePart
		PROC baseproc = ::GetProcAddress(opengl32dll, name);
		if (baseproc != nullptr) {
			return baseproc;
		}
		if (GetLastError() == 127) {
			SetLastError(0);
		}

		// try Get from GLotherParts
		if (wglGetCurrentContext() == nullptr) {
			/**
				* Why we requires wgl.MakeCurrent() ?
				*   varient machine has varient implementations of opengl. opengl32.dll not known that implementations.
				*   known only for Context.     for example: nvoglv32.dll, amdXXX.dll.
			*/
			error = Errors::NotHaveContext;
			return nullptr;
		}

		// Get from GLotherParts
		return wglGetProcAddress(name);
	}
};
using GLplatform = WGLlibrary;
using GLplatformArg = LPCWSTR;
#endif