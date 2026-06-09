#pragma once

// TO KEEP THE INCLUDE DEPENDENCY CLEAN, PLEASE DO NOT INCLUDE ANY OTHER
// TAICHI HEADERS INTO THIS ONE.
//
// TODO(#2196): Once we can slim down "taichi/common/core.h", consider moving
// the contents back to core.h and delete this file.
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

// https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined _WIN64 || defined __CYGWIN__
#if defined(TI_WITH_SPLIT_PYTHON_RUNTIME) && \
    !defined(TI_BUILDING_PYTHON_RUNTIME)
#ifdef __GNUC__
#define TI_DLL_EXPORT __attribute__((dllimport))
#else
#define TI_DLL_EXPORT __declspec(dllimport)
#endif  //  __GNUC__
#else
#ifdef __GNUC__
#define TI_DLL_EXPORT __attribute__((dllexport))
#else
#define TI_DLL_EXPORT __declspec(dllexport)
#endif  //  __GNUC__
#endif  // split Python runtime import/export
#else
#define TI_DLL_EXPORT __attribute__((visibility("default")))
#endif  // defined _WIN32 || defined _WIN64 || defined __CYGWIN__

// Windows
#if defined(_WIN64)
#define TI_PLATFORM_WINDOWS
#endif

#if defined(_WIN32) && !defined(_WIN64)
static_assert(false, "32-bit Windows systems are not supported")
#endif

// Linux
#if defined(__linux__)
#if defined(ANDROID)
#define TI_PLATFORM_ANDROID
#else
#define TI_PLATFORM_LINUX
#endif
#endif

// OSX
#if defined(__APPLE__)
#define TI_PLATFORM_OSX
#endif

#if (defined(TI_PLATFORM_LINUX) || defined(TI_PLATFORM_OSX) || \
     defined(__unix__))
#define TI_PLATFORM_UNIX
#endif
