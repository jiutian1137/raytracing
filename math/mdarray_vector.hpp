#pragma once

///@brief Vector Function of Numbers 
///@license Free 
///@review 2023-01-03 
///@contact Jiang1998Nan@outlook.com 
#define _MATH_MDARRAY_VECTOR_
#include "mdarray.hpp"

#include <cmath>///basic-type not have ADL.
#include <algorithm>
using std::min, std::max, std::clamp;

#include <intrin.h>
#ifndef _mm_neg_ps
#define _mm_neg_ps(x) _mm_mul_ps(_mm_set1_ps(-1.0f), x)
#endif
#ifndef _mm_abs_ps
#define _mm_abs_ps(x) _mm_andnot_ps(_mm_set1_ps(-0.0f), x)
#endif
#ifndef _mm_copysign_ps
#define _mm_copysign_ps(mag,sgn) _mm_or_ps(_mm_and_ps(sgn, _mm_set1_ps(-0.0f)), _mm_andnot_ps(_mm_set1_ps(-0.0f), mag))
#endif
#ifndef _mm_round_even_ps
#define _mm_round_even_ps(x) _mm_round_ps(x, _MM_ROUND_MODE_NEAREST) 
#endif
#ifndef _mm_neg_pd
#define _mm_neg_pd(x) _mm_mul_pd(_mm_set1_pd(-1.0), x)
#endif
#ifndef _mm_abs_pd
#define _mm_abs_pd(x) _mm_andnot_pd(_mm_set1_pd(-0.0), x)
#endif
#ifndef _mm_roundeven_pd
#define _mm_roundeven_pd(x) _mm_round_pd(x, _MM_ROUND_MODE_NEAREST) 
#endif
#ifndef _mm256_neg_pd
#define _mm256_neg_pd(x) _mm256_mul_pd(_mm256_set1_pd(-1.0), x)
#endif
#ifndef _mm256_abs_pd
#define _mm256_abs_pd(x) _mm256_andnot_pd(_mm256_set1_pd(-0.0), x)
#endif
#ifndef _mm256_copysign_pd
#define _mm256_copysign_pd(mag,sgn) _mm256_or_pd(_mm256_and_pd(sgn, _mm_set1_pd(-0.0)), _mm_andnot_pd(_mm_set1_pd(-0.0), mag))
#endif
#ifndef _mm256_round_even_pd
#define _mm256_round_even_pd(x) _mm256_round_pd(x, _MM_ROUND_MODE_NEAREST) 
#endif
#ifndef _mm_neg_epi32
/// @note who should divide the __m128i to 
///		a series { __m128i8, __m128i16, __m128i32, __m128i64, ... 
/// Why not have those types ? below is my thoughts 
///		The simd integer instruction set
///			only epi32 has complete instruction set.
///		Although many other bit integer instruction sets have been gradually added to newer CPUs, 
///			but they are not complete or the CPU requirements are very new.
///		So I only implement epi32 now.
#define _mm_neg_epi32(x) _mm_mullo_epi32(x,_mm_set1_epi32(-1))
#endif

namespace math {
	/// Next Version, Expr may optimize Insufficient Registers. (Especially in scalar operations)
	/*template<typename scalar, typename sizety, typename arithmetic, sizety size, typename functy>
	struct smdarray_expr {
		const mdarray<scalar, sizety, arithmetic, size>& left;
		const functy& func;
		
		operator mdarray<scalar, sizety, arithmetic, size>() const {
			mdarray<scalar, sizety, arithmetic, size> result;
			for (decltype(result.package_length()) i = 0; i != result.package_length(); ++i) 
				result.__data[i] = func(left.__data[i]);
			return std::move(result);
		}
	};*/
	
	#define __smdarray_operator_x(T,D,P, OP) \
	smdarray<T,D,P> operator##OP(const smdarray<T,D,P>& x) { \
		smdarray<T,D,P> z; \
		iterator_based_unroll4x_for(constexpr, N, xi, \
			epconj(constexpr auto N = z.package_length(); auto* zi = z.__data; const auto* xi = x.__data), \
				epconj(zi+=4, xi+=4), \
					epconj(++zi, ++xi), \
			zi[i] = ( OP xi[i]); ) \
		return std::move( z ); \
	}
#ifdef __math_mdarray_version_4
	#define __mdarray_operator_x_y(T,D,A, OP) \
	mdarray<T,D,A> operator##OP(const mdarray_view<T,D,A>& x) { \
		assert( !x.empty() ); \
		mdarray<T,D,A> z( x.size() ); \
		iterator_based_unroll4x_for(, N, xi, \
			epconj(const auto N = z.package_length(); auto* zi = z.__data; const auto* xi = x.__data), \
				epconj(zi+=4, xi+=4), \
					epconj(++zi, ++xi), \
			zi[i] = ( OP xi[i]); ) \
		return std::move( z ); \
	}
#endif

	#define __smdarray_function_x(T,D,P, NAME, FN) \
	smdarray<T,D,P> NAME(const smdarray<T,D,P>& x) { \
		smdarray<T,D,P> z; \
		iterator_based_unroll4x_for(constexpr, N, xi, \
			epconj(constexpr auto N = z.package_length(); auto* zi = z.__data; const auto* xi = x.__data), \
				epconj(zi+=4, xi+=4), \
					epconj(++zi, ++xi), \
			zi[i] = FN( xi[i] ); ) \
		return std::move( z ); \
	}
#ifdef __math_mdarray_version_4
	#define __mdarray_function_x(T,D,A, NAME, FN) \
	mdarray<T,D,A> NAME(const mdarray_view<T,D,A>& x) { \
		assert( !x.empty() ); \
		mdarray<T,D,A> z( x.size() ); \
		iterator_based_unroll4x_for(, N, xi, \
			epconj(const auto N = z.package_length(); auto* zi = z.__data; const auto* xi = x.__data), \
				epconj(zi+=4, xi+=4), \
					epconj(++zi, ++xi), \
			zi[i] = FN( xi[i] ); ) \
		return std::move( z ); \
	}
#endif


	#define __smdarray_assign_operator_x_y(T,D,P, AOP, D2) \
	smdarray<T,D,P>& operator##AOP(smdarray<T,D,P>& x, const smdarray<T,D2,P>& y) { \
		iterator_based_unroll4x_for(constexpr, N, xi, \
			epconj(constexpr auto N = x.package_length(); auto* xi = x.__data; const auto* yi = y.__data), \
				epconj(xi+=4, yi+=4), \
					epconj(++xi, ++yi), \
			(xi[i] AOP yi[i]); ) \
		return x; \
	}
#ifdef __math_mdarray_version_4
	#define __mdarray_assign_operator_x_y(T,D,A, AOP, D2) \
	mdarray_view<T,D,A>& operator##AOP(mdarray_view<T,D,A>& x, const mdarray_view<T,D2,A>& y) { \
		assert( !x.empty() ); \
		assert( x.dimension() == y.dimension() ); \
		assert( x.size() == y.size() ); \
		iterator_based_unroll4x_for(, N, xi, \
			epconj(const auto N = x.package_length(); auto* xi = x.__data; const auto* yi = y.__data), \
				epconj(xi+=4, yi+=4), \
					epconj(++xi, ++yi), \
			(xi[i] AOP yi[i]); ) \
		return x; \
	}
#endif

	#define __smdarray_assign_function_x_y(T,D,P, NAME, FN, D2) \
	smdarray<T,D,P>& NAME(smdarray<T,D,P>& x, const smdarray<T,D2,P>& y) { \
		iterator_based_unroll4x_for(constexpr, N, xi, \
			epconj(constexpr auto N = x.package_length(); auto* xi = x.__data; const auto* yi = y.__data), \
				epconj(xi+=4, yi+=4), \
					epconj(++xi, ++yi), \
			xi[i] = FN(xi[i], yi[i]); ) \
		return x; \
	}
#ifdef __math_mdarray_version_4
	#define __mdarray_assign_function_x_y(T,D,A, NAME, FN, D2) \
	mdarray_view<T,D,A>& NAME(mdarray_view<T,D,A>& x, const mdarray_view<T,D2,A>& y) { \
		assert( !x.empty() ); \
		assert( x.dimension() == y.dimension() ); \
		assert( x.size() == y.size() ); \
		iterator_based_unroll4x_for(, N, xi, \
			epconj(const auto N = x.package_length(); auto* xi = x.__data; const auto* yi = y.__data), \
				epconj(xi+=4, yi+=4), \
					epconj(++xi, ++yi), \
			xi[i] = FN(xi[i], yi[i]); ) \
		return x; \
	}
#endif

	#define __smdarray_operator_x_y(T,D,P, OP, D2) \
	smdarray<T,D,P> operator##OP(const smdarray<T,D,P>& x, const smdarray<T,D2,P>& y) { \
		smdarray<T,D,P> z; \
		iterator_based_unroll4x_for(constexpr, N, xi, \
			epconj(constexpr auto N = z.package_length(); auto* zi = z.__data; const auto* xi = x.__data; const auto* yi = y.__data), \
				epconj(zi+=4, xi+=4, yi+=4), \
					epconj(++zi, ++xi, ++yi), \
			zi[i] = (xi[i] OP yi[i]); ) \
		return std::move( z ); \
	}
#ifdef __math_mdarray_version_4
	#define __mdarray_operator_x_y(T,D,A, OP, D2) \
	mdarray<T,D,A> operator##OP(const mdarray_view<T,D,A>& x, const mdarray_view<T,D2,A>& y) { \
		assert( !x.empty() ); \
		assert( x.dimension() == y.dimension() ); \
		assert( x.size() == y.size() ); \
		mdarray<T,D,A> z( x.size() ); \
		iterator_based_unroll4x_for(, N, xi, \
			epconj(const auto N = z.package_length(); auto* zi = z.__data; const auto* xi = x.__data; const auto* yi = y.__data), \
				epconj(zi+=4, xi+=4, yi+=4), \
					epconj(++zi, ++xi, ++yi), \
			zi[i] = (xi[i] OP yi[i]); ) \
		return std::move( z ); \
	}
#endif

	#define __smdarray_function_x_y(T,D,P, NAME, FN, D2) \
	smdarray<T,D,P> NAME(const smdarray<T,D,P>& x, const smdarray<T,D2,P>& y) { \
		smdarray<T,D,P> z; \
		iterator_based_unroll4x_for(constexpr, N, xi, \
			epconj(constexpr auto N = z.package_length(); auto* zi = z.__data; const auto* xi = x.__data; const auto* yi = y.__data), \
				epconj(zi+=4, xi+=4, yi+=4), \
					epconj(++zi, ++xi, ++yi), \
			zi[i] = FN(xi[i], yi[i]); ) \
		return std::move( z ); \
	}
#ifdef __math_mdarray_version_4
	#define __mdarray_function_x_y(T,D,A, NAME, FN, D2) \
	mdarray<T,D,A> NAME(const mdarray_view<T,D,A>& x, const mdarray_view<T,D2,A>& y) { \
		assert( !x.empty() ); \
		assert( x.dimension() == y.dimension() ); \
		assert( x.size() == y.size() ); \
		mdarray<T,D,A> z( x.size() ); \
		iterator_based_unroll4x_for(, N, xi, \
			epconj(const auto N = z.package_length(); auto* zi = z.__data; const auto* xi = x.__data; const auto* yi = y.__data), \
				epconj(zi+=4, xi+=4, yi+=4), \
					epconj(++zi, ++xi, ++yi), \
			zi[i] = FN(xi[i], yi[i]); ) \
		return std::move( z ); \
	}
#endif

	#define __smdarray_assign_operator_x_yval(T,D,P, AOP, T2,setP) \
	smdarray<T,D,P>& operator##AOP(smdarray<T,D,P>& x, const T2& yval) { const auto __yval = setP(static_cast<T>(yval)); \
		iterator_based_unroll4x_for(constexpr, N, xi, \
			epconj(constexpr auto N = x.package_length(); auto* xi = x.__data), \
				xi+=4, \
					++xi, \
			(xi[i] AOP __yval); ) \
		return x; \
	}
#ifdef __math_mdarray_version_4
	#define __mdarray_assign_operator_x_yval(T,D,A, AOP, T2,setP) \
	mdarray_view<T,D,A>& operator##AOP(const mdarray_view<T,D,A>& x, const T2& yval) { const auto __yval = setP(static_cast<T>(yval)); \
		assert( !x.empty() ); \
		iterator_based_unroll4x_for(, N, xi, \
			epconj(const auto N = x.package_length(); auto* xi = x.__data), \
				xi+=4, \
					++xi, \
			(xi[i] AOP __yval); ) \
		return x; \
	}
#endif

	#define __smdarray_assign_function_x_yval(T,D,P, NAME, FN, T2,setP) \
	smdarray<T,D,P>& NAME(smdarray<T,D,P>& x, const T2& yval) { const auto __yval = setP(static_cast<T>(yval)); \
		iterator_based_unroll4x_for(constexpr, N, xi, \
			epconj(constexpr auto N = x.package_length(); auto* xi = x.__data), \
				xi+=4, \
					++xi, \
			xi[i] = FN(xi[i], __yval); ) \
		return x; \
	}
#ifdef __math_mdarray_version_4
	#define __mdarray_assign_function_x_yval(T,D,A, NAME, FN, T2,setP) \
	mdarray_view<T,D,A>& NAME(const mdarray_view<T,D,A>& x, const T2& yval) { const auto __yval = setP(static_cast<T>(yval)); \
		assert( !x.empty() ); \
		iterator_based_unroll4x_for(, N, xi, \
			epconj(const auto N = x.package_length(); auto* xi = x.__data), \
				xi+=4, \
					++xi, \
			xi[i] = FN(xi[i], __yval); ) \
		return x; \
	}
#endif

	#define __smdarray_operator_x_yval(T,D,P, OP, T2,setP) \
	smdarray<T,D,P> operator##OP(const smdarray<T,D,P>& x, const T2& yval) { const auto __yval = setP(static_cast<T>(yval)); \
		smdarray<T,D,P> z; \
		iterator_based_unroll4x_for(constexpr, N, xi, \
			epconj(constexpr auto N = z.package_length(); auto* zi = z.__data; const auto* xi = x.__data), \
				epconj(zi+=4, xi+=4), \
					epconj(++zi, ++xi), \
			zi[i] = (xi[i] OP __yval); ) \
		return std::move( z ); \
	}
#ifdef __math_mdarray_version_4
	#define __mdarray_operator_x_yval(T,D,A, OP, T2,setP) \
	mdarray<T,D,A> operator##OP(const mdarray_view<T,D,A>& x, const T2& yval) { const auto __yval = setP(static_cast<T>(yval)); \
		assert( !x.empty() ); \
		mdarray<T,D,A> z( x.size() ); \
		iterator_based_unroll4x_for(, N, xi, \
			epconj(const auto N = z.package_length(); auto* zi = z.__data; const auto* xi = x.__data), \
				epconj(zi+=4, xi+=4), \
					epconj(++zi, ++xi), \
			zi[i] = (xi[i] OP __yval); ) \
		return std::move( z ); \
	}
#endif

	#define __smdarray_function_x_yval(T,D,P, NAME, FN, T2,setP) \
	smdarray<T,D,P> NAME(const smdarray<T,D,P>& x, const T2& yval) { const auto __yval = setP(static_cast<T>(yval)); \
		smdarray<T,D,P> z; \
		iterator_based_unroll4x_for(constexpr, N, xi, \
			epconj(constexpr auto N = z.package_length(); auto* zi = z.__data; const auto* xi = x.__data), \
				epconj(zi+=4, xi+=4), \
					epconj(++zi, ++xi), \
			zi[i] = FN(xi[i], __yval); ) \
		return std::move( z ); \
	}
#ifdef __math_mdarray_version_4
	#define __mdarray_function_x_yval(T,D,A, NAME, FN, T2,setP) \
	mdarray<T,D,A> NAME(const mdarray_view<T,D,A>& x, const T2& yval) { const auto __yval = setP(static_cast<T>(yval)); \
		assert( !x.empty() ); \
		mdarray<T,D,A> z( x.size() ); \
		iterator_based_unroll4x_for(, N, xi, \
			epconj(const auto N = z.package_length(); auto* zi = z.__data; const auto* xi = x.__data), \
				epconj(zi+=4, xi+=4), \
					epconj(++zi, ++xi), \
			zi[i] = FN(xi[i], __yval); ) \
		return std::move( z ); \
	}
#endif

	#define __smdarray_operator_xval_y(T,D,P, OP, T2,setP) \
	smdarray<T,D,P> operator##OP(const T2& xval, const smdarray<T,D,P>& y) { const auto __xval = setP(static_cast<T>(xval)); \
		smdarray<T,D,P> z; \
		iterator_based_unroll4x_for(constexpr, N, yi, \
			epconj(constexpr auto N = z.package_length(); auto* zi = z.__data; const auto* yi = y.__data), \
				epconj(zi+=4, yi+=4), \
					epconj(++zi, ++yi), \
			zi[i] = (__xval OP yi[i]); ) \
		return std::move( z ); \
	}
#ifdef __math_mdarray_version_4
	#define __mdarray_operator_xval_y(T,D,A, OP, T2,setP) \
	mdarray<T,D,A> operator##OP(const T2& xval, const mdarray_view<T,D,A>& y) { const auto __xval = setP(static_cast<T>(xval)); \
		assert( !y.empty() ); \
		mdarray<T,D,A> z( y.size() ); \
		iterator_based_unroll4x_for(, N, yi, \
			epconj(const auto N = z.package_length(); auto* zi = z.__data; const auto* yi = y.__data), \
				epconj(zi+=4, yi+=4), \
					epconj(++zi, ++yi), \
			zi[i] = (__xval OP yi[i]); ) \
		return std::move( z ); \
	}
#endif

	#define __smdarray_function_xval_y(T,D,P, NAME, FN, T2,setP) \
	smdarray<T,D,P> NAME(const T2& xval, const smdarray<T,D,P>& y) { const auto __xval = setP(static_cast<T>(xval)); \
		smdarray<T,D,P> z; \
		iterator_based_unroll4x_for(constexpr, N, yi, \
			epconj(constexpr auto N = z.package_length(); auto* zi = z.__data; const auto* yi = y.__data), \
				epconj(zi+=4, yi+=4), \
					epconj(++zi, ++yi), \
			zi[i] = FN(__xval, yi[i]); ) \
		return std::move( z ); \
	}
#ifdef __math_mdarray_version_4
	#define __mdarray_function_xval_y(T,D,A, NAME, FN, T2,setP) \
	mdarray<T,D,A> NAME(const T2& xval, const mdarray_view<T,D,A>& y) { const auto __xval = setP(static_cast<T>(xval)); \
		assert( !y.empty() ); \
		mdarray<T,D,A> z( y.size() ); \
		iterator_based_unroll4x_for(, N, yi, \
			epconj(const auto N = z.package_length(); auto* zi = z.__data; const auto* yi = y.__data), \
				epconj(zi+=4, yi+=4), \
					epconj(++zi, ++yi), \
			zi[i] = FN(__xval, yi[i]); ) \
		return std::move( z ); \
	}
#endif

	using _CSTD modf, _CSTD frexp, _CSTD ldexp;///basic-type not have ADL.

	template<typename scalar, auto dimension, typename package>
	constexpr auto modf(const smdarray<scalar,dimension,package>& x, scalar* yi) -> smdarray<scalar,dimension,package> {
		smdarray<scalar,dimension,package> z;
		iterator_based_unroll4x_for(constexpr, N, xi,
			epconj(constexpr auto N = z.length(); scalar* zi = z.data(); const scalar* xi = x.data()),
				epconj(zi+=4, xi+=4, yi+=4),
					epconj(++zi, ++xi, ++yi),
			zi[i] = modf(xi[i], yi+i); )
		return std::move( z );
	}

	template<typename scalar, auto dimension, typename package>
	constexpr auto frexp(const smdarray<scalar,dimension,package>& x, int* yi) -> smdarray<scalar,dimension,package> {
		smdarray<scalar,dimension,package> z;
		iterator_based_unroll4x_for(constexpr, N, xi,
			epconj(constexpr auto N = z.length(); scalar* zi = z.data(); const scalar* xi = x.data()),
				epconj(zi+=4, xi+=4, yi+=4),
					epconj(++zi, ++xi, ++yi),
			zi[i] = frexp(xi[i], yi+i); )
		return std::move( z );
	}

	template<typename scalar, auto dimension, typename package>
	constexpr auto ldexp(const smdarray<scalar,dimension,package>& x, const int* yi) -> smdarray<scalar,dimension,package> {
		smdarray<scalar,dimension,package> z;
		iterator_based_unroll4x_for(constexpr, N, xi,
			epconj(constexpr auto N = z.length(); scalar* zi = z.data(); const scalar* xi = x.data()),
				epconj(zi+=4, xi+=4, yi+=4),
					epconj(++zi, ++xi, ++yi),
			zi[i] = ldexp(xi[i], yi[i]); )
		return std::move( z );
	}

	/// ref(x) = { x[i] += y[i] } = { x.__data[i] += y.__data[i] }.
	template<typename T, auto D, typename P, auto D2> requires (D == D2)
	constexpr __smdarray_assign_operator_x_y(T,D,P, +=, D2)

	/// ref(x) = { x[i] += yval } = { x.__data[i] += set1(static_cast<scalar>(yval)) }.
	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_assign_operator_x_yval(T,D,P, +=, T2,static_cast<P>)

	/// z = { x[i] + y[i] } = { x.__data[i] + y.__data[i] }.
	template<typename T, auto D, typename P, auto D2> requires (D == D2)
	constexpr __smdarray_operator_x_y(T,D,P, +, D2)

	/// z = { x[i] + yval } = { x.__data[i] + set1(static_cast<scalar>(yval)) }.
	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_operator_x_yval(T,D,P, +, T2,static_cast<P>)

	/// z = { xval + y[i] } = { set1(static_cast<scalar>(xval)) + y.__data[i] }.
	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_operator_xval_y(T,D,P, +, T2,static_cast<P>)

	/// ref(x) = { x[i] -= y[i] } = { x.__data[i] -= y.__data[i] }.
	template<typename T, auto D, typename P, auto D2> requires (D == D2)
	constexpr __smdarray_assign_operator_x_y(T,D,P, -=, D2)

	/// ref(x) = { x[i] -= yval } = { x.__data[i] -= set1(static_cast<scalar>(yval)) }.
	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_assign_operator_x_yval(T,D,P, -=, T2,static_cast<P>)

	/// z = { x[i] - y[i] } = { x.__data[i] - y.__data[i] }.
	template<typename T, auto D, typename P, auto D2> requires (D == D2)
	constexpr __smdarray_operator_x_y(T,D,P, -, D2)

	/// z = { x[i] - yval } = { x.__data[i] - set1(static_cast<scalar>(yval)) }.
	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_operator_x_yval(T,D,P, -, T2,static_cast<P>)

	/// z = { xval - y[i] } = { set1(static_cast<scalar>(xval)) - y.__data[i] }.
	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_operator_xval_y(T,D,P, -, T2,static_cast<P>)

	/// ref(x) = { x[i] *= y[i] } = { x.__data[i] *= y.__data[i] }.
	template<typename T, auto D, typename P, auto D2> requires (D == D2)
	constexpr __smdarray_assign_operator_x_y(T,D,P, *=, D2)

	/// ref(x) = { x[i] *= yval } = { x.__data[i] *= set1(static_cast<scalar>(yval)) }.
	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_assign_operator_x_yval(T,D,P, *=, T2,static_cast<P>)

	/// z = { x[i] * y[i] } = { x.__data[i] * y.__data[i] }.
	template<typename T, auto D, typename P, auto D2> requires (D == D2)
	constexpr __smdarray_operator_x_y(T,D,P, *, D2)

	/// z = { x[i] * yval } = { x.__data[i] * set1(static_cast<scalar>(yval)) }.
	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_operator_x_yval(T,D,P, *, T2,static_cast<P>)

	/// z = { xval * y[i] } = { set1(static_cast<scalar>(xval)) * y.__data[i] }.
	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_operator_xval_y(T,D,P, *, T2,static_cast<P>)

	/// ref(x) = { x[i] /= y[i] } = { x.__data[i] /= y.__data[i] }.
	template<typename T, auto D, typename P, auto D2> requires (D == D2)
	constexpr __smdarray_assign_operator_x_y(T,D,P, /=, D2)

	/// ref(x) = { x[i] /= yval } = { x.__data[i] /= set1(static_cast<scalar>(yval)) }.
	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_assign_operator_x_yval(T,D,P, /=, T2,static_cast<P>)

	/// z = { x[i] / y[i] } = { x.__data[i] / y.__data[i] }.
	template<typename T, auto D, typename P, auto D2> requires (D == D2)
	constexpr __smdarray_operator_x_y(T,D,P, /, D2)

	/// z = { x[i] / yval } = { x.__data[i] / set1(static_cast<scalar>(yval)) }.
	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_operator_x_yval(T,D,P, /, T2,static_cast<P>)

	/// z = { xval / y[i] } = { set1(static_cast<scalar>(xval)) / y.__data[i] }.
	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_operator_xval_y(T,D,P, /, T2,static_cast<P>)

	using _CSTD fmod, _CSTD copysign;///basic-type not have ADL.
	using std::min, std::max, std::clamp;///general functions, 
	///note:these general functions not work for mdarray<D>,mdarray<D2> but can match them, so we need add a function.

	template<typename T, auto D, typename P, auto D2> requires (D == D2)
	constexpr __smdarray_function_x_y(T,D,P, fmod, fmod, D2)

	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_function_x_yval(T,D,P, fmod, fmod, T2,static_cast<P>)

	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_function_xval_y(T,D,P, fmod, fmod, T2,static_cast<P>)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x_y(T,D,P, min, min, D)

	template<typename T, auto D, typename P, auto D2> requires (D == D2)
	constexpr __smdarray_function_x_y(T,D,P, min, min, D2)

	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_function_x_yval(T,D,P, min, min, T2,static_cast<P>)

	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_function_xval_y(T,D,P, min, min, T2,static_cast<P>)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x_y(T,D,P, max, max, D)

	template<typename T, auto D, typename P, auto D2> requires (D == D2)
	constexpr __smdarray_function_x_y(T,D,P, max, max, D2)

	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_function_x_yval(T,D,P, max, max, T2,static_cast<P>)

	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_function_xval_y(T,D,P, max, max, T2,static_cast<P>)

	template<typename T, auto D, typename P, auto D2> requires (D == D2)
	constexpr __smdarray_function_x_y(T,D,P, copysign, copysign, D2)

	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_function_x_yval(T,D,P, copysign, copysign, T2,static_cast<P>)

	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_function_xval_y(T,D,P, copysign, copysign, T2,static_cast<P>)

	using _CSTD abs,
		_CSTD trunc,
		_CSTD floor,
		_CSTD ceil,
		_CSTD round,
		_CSTD sqrt,
		_CSTD cbrt,
		_CSTD exp,
		_CSTD exp2,
		_CSTD log,
		_CSTD log2,
		_CSTD pow,
		_CSTD cosh,
		_CSTD sinh,
		_CSTD tanh,
		_CSTD acosh,
		_CSTD asinh,
		_CSTD atanh,
		_CSTD cos,
		_CSTD sin,
		_CSTD tan,
		_CSTD acos,
		_CSTD asin,
		_CSTD atan,
		_CSTD atan2;///basic-type not have ADL.

	template<typename T, auto D, typename P>
	constexpr __smdarray_operator_x(T,D,P, -)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, abs, abs)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, trunc, trunc)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, floor, floor)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, ceil, ceil)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, round, round)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, sqrt, sqrt)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, cbrt, cbrt)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, exp, exp)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, exp2, exp2)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, log, log)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, log2, log2)

	template<typename T, auto D, typename P, auto D2> requires (D == D2)
	constexpr __smdarray_function_x_y(T,D,P, pow, pow, D2)

	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_function_x_yval(T,D,P, pow, pow, T2,static_cast<P>)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, cosh, cosh)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, sinh, sinh)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, tanh, tanh)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, acosh, acosh)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, asinh, asinh)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, atanh, atanh)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, cos, cos)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, sin, sin)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, tan, tan)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, acos, acos)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, asin, asin)

	template<typename T, auto D, typename P>
	constexpr __smdarray_function_x(T,D,P, atan, atan)

	template<typename T, auto D, typename P, auto D2> requires (D == D2)
	constexpr __smdarray_function_x_y(T,D,P, atan2, atan2, D2)

	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_function_x_yval(T,D,P, atan2, atan2, T2,static_cast<P>)

	template<typename T, auto D, typename P, std::convertible_to<T> T2>
	constexpr __smdarray_function_xval_y(T,D,P, atan2, atan2, T2,static_cast<P>)

	template<typename T, auto D, typename A>
	constexpr auto vsize(const smdarray<T,D,A>& x) {
		auto N = x.size(0);
		if (N == 1) {
			auto iend = x.dimension();
			if (iend != 1) {
				auto i = decltype(iend)(1);
				do {
					N = x.size(i);
				} while (i != iend && N == 1);
			}
		}
		return N;
	}

	template<typename T, auto D, typename P, auto D2> /* impilicit requires(vsize(x) == vsize(y)) */
	constexpr T dot(const smdarray<T,D,P>& x, const smdarray<T,D2,P>& y) {
		static_assert( x.simdlen == 1 );
		T _dp = x[0] * y[0];
		iterator_based_unroll4x_for(constexpr, N, xi, 
			epconj(constexpr auto N = vsize(smdarray<T,D,P>{}) - 1; const auto* xi = std::next(x.data()), * yi = std::next(y.data())),
				epconj(xi+=4, yi+=4),
					epconj(++xi, ++yi),
			_dp += xi[i] * yi[i]; )
		return _dp;
	}

	template<typename T, auto D, typename P>
	constexpr T dot(const smdarray<T,D,P>& x) {
		static_assert( x.simdlen == 1 );
		T _dp = x[0] * x[0];
		iterator_based_unroll4x_for(constexpr, N, xi, 
			epconj(constexpr auto N = vsize(smdarray<T,D,P>{}) - 1; const auto *xi = std::next(x.data())),
				xi+=4,
					++xi,
			_dp += xi[i] * xi[i]; )
		return _dp;
	}

	template<typename T, auto D, typename P>
	constexpr T length(const smdarray<T,D,P>& x) {
		return sqrt(dot(x));
	}

	template<typename T, auto D, typename P>
	constexpr smdarray<T,D,P> normalize(const smdarray<T,D,P>& x, T* length = nullptr) {
		T s2 = dot(x);
		if (s2 == 0/* || s2 == 1*/) {
			return smdarray<T,D,P>{ x };
		} else {
			T s = sqrt(s2);
			if (length) { (*length) = s; }
			return x/s;
		}
	}

	template<typename scalar, auto D, typename P, auto D2> /* impilicit requires(vsize(x) == vsize(y)) */
	constexpr smdarray<scalar,D,P> cross(const smdarray<scalar,D,P>& x, const smdarray<scalar,D2,P>& y) {
		///@thory 
		/// solve [ dot([i,j,k],v1] = 0 ]
		///       [ dot([i,j,k],v2) = 0 ]
		/// 
		///    | v1.x, v1.y, v1.z |   | i |   | 0 |
		/// => | v2.x, v2.y, v2.z | * | j | = | 0 |
		///    |   0 ,   0 ,   0  |   | k |   | 0 |
		/// 
		///    | v1.x,          v1.y,                 v1.z          |   | i |   | 0 |
		/// => |  0  , v2.y - v2.x/v1.x*v1.y, v2.z - v2.x/v1.x*v1.z | * | j | = | 0 |
		///    |  0  ,            0 ,                   0           |   | k |   | 0 |
		/// 
		/// => | v1.x,          v1.y,                 v1.z          |   | i |   | 0 |
		///    |  0  , v2.y*v1.x - v2.x*v1.y, v2.z*v1.x - v2.x*v1.z | * | j | = | 0 |
		///    |  0  ,            0 ,                   0           |   | k |   | 0 |
		/// 
		///  j =   v2.z*v1.x - v2.x*v1.z     : (v2.y*v1.x - v2.x*v1.y)*j + (v2.z*v1.x - v2.x*v1.z)*k = 0
		///  k = -(v2.y*v1.x - v2.x*v1.y)
		///  i = -(v1.y*j + v1.z*k)/v1.x
		///    = -(v1.y*v2.z*v1.x - v1.y*v2.x*v1.z - v1.z*v2.y*v1.x + v1.z*v2.x*v1.y)/v1.x
		///    = -(v1.y*v2.z - v1.y*v2.x*v1.z/v1.x - v1.z*v2.y + v1.z*v2.x*v1.y/v1.x)
		///    = -(v1.y*v2.z - v1.z*v2.y)
		/// 
		///  j = -(v2.z*v1.x - v2.x*v1.z)    : (v2.y*v1.x - v2.x*v1.y)*j + (v2.z*v1.x - v2.x*v1.z)*k = 0
		///  k =   v2.y*v1.x - v2.x*v1.y
		///  i = ...
		/// 
		///@thory 
		///            |  i  ,  j  ,  k   |
		/// solve det( | v1.x, v1.y, v1.z | ) = +-1    :error, v1 and v2 are not necessarily orthogonal .
		///            | v2.x, v2.y, v2.z |
		/// 
		/// magnitude can be any nonzero
		///            |  i    j    k   |
		/// solve det( | v1.x v1.y v1.z | ) = +-?
		///            | v2.x v2.y v2.z |
		///   i*det(minor(0,0))         - j*det(minor(0,1))         + k*det(minor(0,2))         = +-?
		///   i*(v1.y*v2.z - v1.z*v2.y) - j*(v1.x*v2.z - v1.z*v2.x) + k*(v1.x*v2.y - v1.y*v2.x) = +-?
		///   i*(v1.y*v2.z - v1.z*v2.y) + j*(v1.z*v2.x - v1.x*v2.z) + k*(v1.x*v2.y - v1.y*v2.x) = +-?
		/// 
		///    i = v1.y*v2.z - v1.z*v2.y
		///     j = v1.z*v2.x - v1.x*v2.z  for positive determinant
		///      k = v1.x*v2.y - v1.y*v2.x
		/// 
		///    i = -(v1.y*v2.z - v1.z*v2.y)
		///     j = -(v1.z*v2.x - v1.x*v2.z)  for negative determinant
		///      k = -(v1.x*v2.y - v1.y*v2.x)
		/// 
		///@summary 
		/// We cannot say which is good, but we like positive.
		/// So usually cross product between 'v1' and 'v2' is not meaning orthogonal bewteen 'v1' and 'v2'
		///                                                is meaning positive orthogonal between 'v1' and 'v2'.
		return smdarray<scalar,D,P>{
			x[1]*y[2] - x[2]*y[1], 
			x[2]*y[0] - x[0]*y[2], 
			x[0]*y[1] - x[1]*y[0] 
		};
	}

	template<typename scalar, auto D, typename P, auto D2, auto D3> /* impilicit requires(vsize(x) == vsize(y)) */
	constexpr smdarray<scalar,D,P> cross(const smdarray<scalar,D,P>& v1, const smdarray<scalar,D2,P>& v2, const smdarray<scalar,D3,P>& v3) {
		///@diagram
		/// |   i ,     j ,    k ,   u  |
		/// | v1.x,   v1.y,  v1.z, v1.w |
		/// | v2.x,   v2.y,  v2.z, v2.w | = i*1*det(minor(0,0)) + j*-1*det(minor(0,1)) + k*1*det(minor(0,2)) + u*-1*det(minor(0,3)), 1.determinat expand
		/// | v3.x,   v3.y,  v3.z, v3.w |
		///     |      | |    |      |    = vector{ +(v1.y*detC - v1.z*detE + v1.w*detB),
		///     +-detA-+-detB-+-detC-+              -(v1.x*detC - v1.z*detF + v1.w*detD),
		///     |        |    |      |              +(v1.x*detE - v1.y*detF + v1.w*detA),
		///     +---detD-+----+      |              -(v1.x*detB - v1.y*detD + v1.z*detA) }
		///     |        |           |
		///     |        +----detE---+
		///     |                    |
		///     +-----detF-----------+
		scalar detA = v2[0] * v3[1] - v2[1] * v3[0];
		scalar detB = v2[1] * v3[2] - v2[2] * v3[1];
		scalar detC = v2[2] * v3[3] - v2[3] * v3[2];
		scalar detD = v2[0] * v3[2] - v2[2] * v3[0];
		scalar detE = v2[1] * v3[3] - v2[3] * v3[1];
		scalar detF = v2[0] * v3[3] - v2[3] * v3[0];
		return smdarray<scalar,D,P>{
				v1[1]*detC - v1[2]*detE + v1[3]*detB,
			-(v1[0]*detC - v1[2]*detF + v1[3]*detD),
				v1[0]*detE - v1[1]*detF + v1[3]*detA,
			-(v1[0]*detB - v1[1]*detD + v1[2]*detA) 
		};
	}


/// SIMD.......

	template<size_t... index, auto size> 
		requires (smdarray<float, size, __m128>::length() <= 4)
	constexpr smdarray<float, size, __m128> permute(const smdarray<float, size, __m128>& src) {
		return { _mm_permute_ps(src.__data[0], []() consteval { size_t i = 0; return ((index<<(i++ * 2)) | ...); }()) };
	}

	template<size_t... index, auto size> 
		requires (smdarray<int, size, __m128i>::length() <= 4)
	constexpr smdarray<int, size, __m128i> permute(const smdarray<int, size, __m128i>& src) {
		return { _mm_shuffle_epi32(src.__data[0], []() consteval { size_t i = 0; return ((index<<(i++ * 2)) | ...); }()) };
	}

	template<size_t... index, auto size> 
		requires (smdarray<float, size, __m128>::length() <= 4)
	constexpr smdarray<float, size, __m128> permute(const smdarray<float, size, __m128>& src0, const smdarray<float, size, __m128>& src1) {
		static constexpr size_t indices[4] = { index... };
		if constexpr (((index < 4) && ...)) {
			return { _mm_permute_ps(src0.__data[0], []() consteval { size_t imm8 = 0; for (size_t i = 0; i != sizeof...(index); ++i) { 
				imm8 |= (indices[i] << (i*2)); } return imm8; }()) };
		} else if constexpr (((index >= 4) && ...)) {
			return { _mm_permute_ps(src1.__data[0], []() consteval { size_t imm8 = 0; for (size_t i = 0; i != sizeof...(index); ++i) { 
				imm8 |= ((indices[i] - 4) << (i*2)); } return imm8; }()) };
		} else if constexpr (
			indices[0] == 0 && (sizeof...(index) == 1 || (
				indices[1] == 4+0 && (sizeof...(index) == 2 || (
					indices[2] == 1 && (sizeof...(index) == 3 || 
						indices[3] == 4+1)))))) {
			return { _mm_unpacklo_ps(src0.__data[0], src1.__data[0]) };
		} else if constexpr (
			indices[0] == 2 && (sizeof...(index) == 1 || (
				indices[1] == 4+2 && (sizeof...(index) == 2 || (
					indices[2] == 3 && (sizeof...(index) == 3 || 
						indices[3] == 4+3)))))) {
			return { _mm_unpackhi_ps(src0.__data[0], src1.__data[0]) };
		} else if constexpr (
			indices[0] < 4 && (sizeof...(index) == 1 || (
				indices[1] < 4 && (sizeof...(index) == 2 || (
					indices[2] >= 4 && (sizeof...(index) == 3 || 
						indices[3] >= 4)))))) {
			return { _mm_shuffle_ps(src0.__data[0], src1.__data[0], []() consteval { size_t imm8 = 0; for (size_t i = 0; i != sizeof...(index); ++i) { 
				imm8 |= ((indices[i] - (i>=2?4:0)) << (i*2)); } return imm8; }()) };
		} else {
			constexpr size_t i0 = (index, ...);
			if constexpr (((index == i0) && ...)) {
				if constexpr (i0 < 4) {
					return _mm_shuffle_ps(src0.__data[0], src0.__data[0], static_cast<int>(i0|(i0<<2)|(i0<<4)|(i0<<6)));
				} else {
					constexpr int iX = static_cast<int>(i0 - 4);
					return _mm_shuffle_ps(src0.__data[0], src0.__data[0], iX|(iX<<2)|(iX<<4)|(iX<<6));
				}
			} else {
				const auto select = [&src0,&src1](size_t i) { 
					return i < src0.length() ? src0[i] : src1[i - src0.length()]; };
				return { select(index)... };
			}
		}
	}

	template<size_t... index, auto size> 
		requires (smdarray<int, size, __m128i>::length() <= 4)
	constexpr smdarray<int, size, __m128i> permute(const smdarray<int, size, __m128i>& src0, const smdarray<int, size, __m128i>& src1) {
		static constexpr size_t indices[4] = { index... };
		if constexpr (((index < 4) && ...)) {
			return { _mm_shuffle_epi32(src0.__data[0], []() consteval { size_t imm8 = 0; for (size_t i = 0; i != sizeof...(index); ++i) { 
				imm8 |= (indices[i] << (i*2)); } return imm8; }()) };
		} else if constexpr (((index >= 4) && ...)) {
			return { _mm_shuffle_epi32(src1.__data[0], []() consteval { size_t imm8 = 0; for (size_t i = 0; i != sizeof...(index); ++i) { 
				imm8 |= ((indices[i] - 4) << (i*2)); } return imm8; }()) };
		} else if constexpr (
			indices[0] == 0 && (sizeof...(index) == 1 || (
				indices[1] == 4+0 && (sizeof...(index) == 2 || (
					indices[2] == 1 && (sizeof...(index) == 3 || 
						indices[3] == 4+1)))))) {
			return { _mm_unpacklo_epi32(src0.__data[0], src1.__data[0]) };
		} else if constexpr (
			indices[0] == 2 && (sizeof...(index) == 1 || (
				indices[1] == 4+2 && (sizeof...(index) == 2 || (
					indices[2] == 3 && (sizeof...(index) == 3 || 
						indices[3] == 4+3)))))) {
			return { _mm_unpackhi_epi32(src0.__data[0], src1.__data[0]) };
		} else if constexpr (
			indices[0] < 4 && (sizeof...(index) == 1 || (
				indices[1] < 4 && (sizeof...(index) == 2 || (
					indices[2] >= 4 && (sizeof...(index) == 3 || 
						indices[3] >= 4)))))) {///"some"CPU will use more cycles.
			return { _mm_castps_si128( _mm_shuffle_ps(_mm_castsi128_ps(src0.__data[0]), _mm_castsi128_ps(src1.__data[0]), []() consteval { size_t imm8 = 0; for (size_t i = 0; i != sizeof...(index); ++i) { 
				imm8 |= ((indices[i] - (i>=2?4:0)) << (i*2)); } return imm8; }()) ) };
		} else {
			constexpr size_t i0 = (index, ...);
			if constexpr (((index == i0) && ...)) {
				if constexpr (i0 < 4) {
					return _mm_castps_si128( _mm_shuffle_ps(_mm_castsi128_ps(src0.__data[0]), _mm_castsi128_ps(src0.__data[0]), static_cast<int>(i0|(i0<<2)|(i0<<4)|(i0<<6))) );
				} else {
					constexpr int iX = static_cast<int>(i0 - 4);
					return _mm_castps_si128( _mm_shuffle_ps(_mm_castsi128_ps(src0.__data[0]), _mm_castsi128_ps(src0.__data[0]), iX|(iX<<2)|(iX<<4)|(iX<<6)) );
				}
			} else {
				const auto select = [&src0,&src1](size_t i) { 
					return i < src0.length() ? src0[i] : src1[i - src0.length()]; };
#ifndef __simd_type_clean
				smdarray<int,size,__m128i> dst;
				size_t i = 0;
				((dst[i++] = select(index)), ...);
				return std::move(dst);
#else
				return { select(index)... };
#endif
			}
		}
	}

#if 0
	template<typename dst_array, auto size>
		requires (smdarray<float, size, __m128>::length() <= 4 &&  std::same_as<value_t<dst_array>, int>)
	constexpr dst_array static_vector_cast(const smdarray<float, size, __m128>& src) {
		if constexpr (math::uses_package_v<dst_array, __m128i>) {
			return { _mm_cvtps_epi32(src.__data[0]) };
		} else {
			__m128i tmp = _mm_cvtps_epi32(src.__data[0]);
			if constexpr (static_cast<size_t>(std::size(dst_array{})) == 1) {
				return { _mm_extract_epi32(tmp, 0) };
			} else if constexpr (static_cast<size_t>(std::size(dst_array{})) == 2) {
				return { _mm_extract_epi32(tmp, 0), _mm_extract_epi32(tmp, 1) };
			} else if constexpr (static_cast<size_t>(std::size(dst_array{})) == 3) {
				return { _mm_extract_epi32(tmp, 0), _mm_extract_epi32(tmp, 1), _mm_extract_epi32(tmp, 2) };
			} else /*if constexpr (dst_array::length() == 4)*/ {
				return { _mm_extract_epi32(tmp, 0), _mm_extract_epi32(tmp, 1), _mm_extract_epi32(tmp, 2), _mm_extract_epi32(tmp, 3) };
			}
		}
	}

	template<typename dst_array, auto size>
		requires (smdarray<float, size, __m128>::length() <= 4 &&  std::same_as<value_t<dst_array>, double>)
	constexpr dst_array static_vector_cast(const smdarray<float, size, __m128>& src) {
		if constexpr (math::uses_package_v<dst_array, __m256d>) {
			return { _mm256_cvtps_pd(src.__data[0]) };
		} else {
			__m256d tmp = _mm256_cvtps_pd(src.__data[0]);
			if constexpr (static_cast<size_t>(std::size(dst_array{})) == 1) {
				return { tmp.m256d_f64[0] };
			} else if constexpr (static_cast<size_t>(std::size(dst_array{})) == 2) {
				return { tmp.m256d_f64[0], tmp.m256d_f64[1] };
			} else if constexpr (static_cast<size_t>(std::size(dst_array{})) == 3) {
				return { tmp.m256d_f64[0], tmp.m256d_f64[1], tmp.m256d_f64[2] };
			} else /*if constexpr (dst_array::length() == 4)*/ {
				return { tmp.m256d_f64[0], tmp.m256d_f64[1], tmp.m256d_f64[2], tmp.m256d_f64[3] };
			}
		}
	}
#endif

	template<auto size>
	constexpr bool operator==(const smdarray<float, size, __m128>& x, const smdarray<float, size, __m128>& y) {
		using smdarray_type = smdarray<float, size, __m128>;
		if constexpr (smdarray_type::dimension() == 1) {
			// Compare each element.
			constexpr auto N = smdarray_type::size(0);
			constexpr auto Ns = N%smdarray_type::simdlen;
			constexpr auto Nb = N/smdarray_type::simdlen;
			iterator_based_unroll4x_for(constexpr, Nb, xi,
				epconj(const auto* xi = x.__data; const auto* yi = y.__data),
					epconj(xi+=4, yi+=4),
						epconj(++xi, ++yi),
				if (_mm_movemask_ps(_mm_cmpneq_ps(xi[i], yi[i]))){ return false; })
			constexpr int Nsmask = ((1 << Ns) - 1);
			if (_mm_movemask_ps(_mm_cmpneq_ps(*xi, *yi)) & Nsmask){ return false; }
		} else {
			///next version...
			abort();
		}

		return true;
	}

	template<auto size, std::convertible_to<float> T2>
	constexpr bool operator==(const smdarray<float, size, __m128>& x, const T2& yval) { const __m128 __yval = _mm_set1_ps(static_cast<float>(yval));
		using smdarray_type = smdarray<float, size, __m128>;
		if constexpr (smdarray_type::dimension() == 1) {
			// Compare each element.
			constexpr auto N = smdarray_type::size(0);
			constexpr auto Ns = N%smdarray_type::simdlen;
			constexpr auto Nb = N/smdarray_type::simdlen;
			iterator_based_unroll4x_for(constexpr, Nb, xi,
				epconj(const auto* xi = x.__data),
					epconj(xi+=4),
						epconj(++xi),
				if (_mm_movemask_ps(_mm_cmpneq_ps(xi[i], __yval))){ return false; })
			constexpr int Nsmask = ((1 << Ns) - 1);
			if (_mm_movemask_ps(_mm_cmpneq_ps(*xi, __yval)) & Nsmask){ return false; }
		} else {
			///next version...
			abort();
		}

		return true;
	}

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_assign_function_x_y(float,D,__m128, operator+=, _mm_add_ps, D2)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_assign_function_x_yval(float,D,__m128, operator+=, _mm_add_ps, T2,_mm_set1_ps)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_function_x_y(float,D,__m128, operator+, _mm_add_ps, D2)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_function_x_yval(float,D,__m128, operator+, _mm_add_ps, T2,_mm_set1_ps)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_function_xval_y(float,D,__m128, operator+, _mm_add_ps, T2,_mm_set1_ps)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_assign_function_x_y(float,D,__m128, operator-=, _mm_sub_ps, D2)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_assign_function_x_yval(float,D,__m128, operator-=, _mm_sub_ps, T2,_mm_set1_ps)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_function_x_y(float,D,__m128, operator-, _mm_sub_ps, D2)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_function_x_yval(float,D,__m128, operator-, _mm_sub_ps, T2,_mm_set1_ps)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_function_xval_y(float,D,__m128, operator-, _mm_sub_ps, T2,_mm_set1_ps)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_assign_function_x_y(float,D,__m128, operator*=, _mm_mul_ps, D2)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_assign_function_x_yval(float,D,__m128, operator*=, _mm_mul_ps, T2,_mm_set1_ps)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_function_x_y(float,D,__m128, operator*, _mm_mul_ps, D2)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_function_x_yval(float,D,__m128, operator*, _mm_mul_ps, T2,_mm_set1_ps)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_function_xval_y(float,D,__m128, operator*, _mm_mul_ps, T2,_mm_set1_ps)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_assign_function_x_y(float,D,__m128, operator/=, _mm_div_ps, D2)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_assign_function_x_yval(float,D,__m128, operator/=, _mm_div_ps, T2,_mm_set1_ps)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_function_x_y(float,D,__m128, operator/, _mm_div_ps, D2)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_function_x_yval(float,D,__m128, operator/, _mm_div_ps, T2,_mm_set1_ps)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_function_xval_y(float,D,__m128, operator/, _mm_div_ps, T2,_mm_set1_ps)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_function_x_y(float,D,__m128, fmod, _mm_fmod_ps, D2)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_function_x_yval(float,D,__m128, fmod, _mm_fmod_ps, T2,_mm_set1_ps)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_function_xval_y(float,D,__m128, fmod, _mm_fmod_ps, T2,_mm_set1_ps)

	template<auto D>
	inline __smdarray_function_x_y(float,D,__m128, min, _mm_min_ps, D)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_function_x_y(float,D,__m128, min, _mm_min_ps, D2)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_function_x_yval(float,D,__m128, min, _mm_min_ps, T2,_mm_set1_ps)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_function_xval_y(float,D,__m128, min, _mm_min_ps, T2,_mm_set1_ps)

	template<auto D>
	inline __smdarray_function_x_y(float,D,__m128, max, _mm_max_ps, D)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_function_x_y(float,D,__m128, max, _mm_max_ps, D2)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_function_x_yval(float,D,__m128, max, _mm_max_ps, T2,_mm_set1_ps)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_function_xval_y(float,D,__m128, max, _mm_max_ps, T2,_mm_set1_ps)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_function_x_y(float,D,__m128, copysign, _mm_copysign_ps, D2)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_function_x_yval(float,D,__m128, copysign, _mm_copysign_ps, T2,_mm_set1_ps)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_function_xval_y(float,D,__m128, copysign, _mm_copysign_ps, T2,_mm_set1_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, operator-, _mm_neg_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, abs,   _mm_abs_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, trunc, _mm_trunc_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, floor, _mm_floor_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, ceil,  _mm_ceil_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, round, _mm_round_even_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, sqrt,  _mm_sqrt_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, cbrt,  _mm_cbrt_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, exp,   _mm_exp_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, exp2,  _mm_exp2_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, log,   _mm_log_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, log2,  _mm_log2_ps)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_function_x_y(float,D,__m128, pow, _mm_pow_ps, D2)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_function_x_yval(float,D,__m128, pow, _mm_pow_ps, T2,_mm_set1_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, cosh, _mm_cosh_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, sinh, _mm_sinh_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, tanh, _mm_tanh_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, acosh, _mm_acosh_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, asinh, _mm_asinh_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, atanh, _mm_atanh_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, cos, _mm_cos_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, sin, _mm_sin_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, tan, _mm_tan_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, acos, _mm_acos_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, asin, _mm_asin_ps)

	template<auto D>
	inline __smdarray_function_x(float,D,__m128, atan, _mm_atan_ps)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_function_x_y(float,D,__m128, atan2, _mm_atan2_ps, D2)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_function_x_yval(float,D,__m128, atan2, _mm_atan2_ps, T2,_mm_set1_ps)

	template<auto D, std::convertible_to<float> T2>
	inline __smdarray_function_xval_y(float,D,__m128, atan2, _mm_atan2_ps, T2,_mm_set1_ps)

	template<auto D, auto D2>
	inline float dot(const smdarray<float,D,__m128>& x, const smdarray<float,D2,__m128>& y) {
		//if constexpr ( static_mdarray<mdarray_view<float,dimension,allocator>> ) {
			constexpr size_t N = vsize(smdarray<float,D,__m128>{});
			constexpr size_t Ns = N%x.simdlen;
			constexpr size_t Nb = N/x.simdlen;
			if constexpr (Nb == 0) {
				return _mm_cvtss_f32( _mm_dp_ps(x.__data[0], y.__data[0], ((1<<(Ns+4)) - 1)/*_mm_dp_ps_arg(Ns)*/) );
			} else {
				__m128 _dp/*4*/ = _mm_mul_ps(x.__data[0], y.__data[0]); constexpr size_t Nbm1 = Nb-1;
				iterator_based_unroll4x_for(constexpr, Nbm1, xi, 
					epconj(const auto *xi = std::next(x.__data), *yi = std::next(y.__data)),
						epconj(xi+=4, yi+=4), 
							epconj(++xi, ++yi), 
					_dp/*4*/ = _mm_add_ps(_dp/*4*/, _mm_mul_ps(xi[i], yi[i])); )
				_dp/*4*/ = _mm_hadd_ps(_dp, _dp);
				_dp/*4*/ = _mm_hadd_ps(_dp, _dp);
				if constexpr (Ns != 0)
					_dp = _mm_add_ss(_dp, _mm_dp_ps(*xi, *yi, ((1<<(Ns+4)) - 1)));
				return _mm_cvtss_f32(_dp);
			}
		//} else {
		//	assert( !x.empty() );
		//	size_t       Nb = vsize(x);
		//	const size_t Ns = Nb%x.simdlen;
		//	Nb /= x.simdlen;
		//	if (Nb == 0) {
		//		switch (Ns) {
		//		case 3:	return _mm_cvtss_f32( _mm_dp_ps(x._Data[0], y._Data[0], _mm_dp_ps_arg(3)) );
		//		case 2:	return _mm_cvtss_f32( _mm_dp_ps(x._Data[0], y._Data[0], _mm_dp_ps_arg(2)) );
		//		case 1:	return _mm_cvtss_f32( _mm_dp_ps(x._Data[0], y._Data[0], _mm_dp_ps_arg(1)) );
		//		default: abort(); 
		//		}
		//	} else {
		//		__m128 _dp/*4*/ = _mm_mul_ps(x._Data[0], y._Data[0]); --Nb;
		//		iterator_based_unroll4x_for(, Nb, xi, 
		//			epconj(const auto *xi = std::next(x._Data), *yi = std::next(y._Data)),
		//				epconj(xi+=4, yi+=4),
		//					epconj(++xi, ++yi),
		//			_dp/*4*/ = _mm_add_ps(_dp/*4*/, _mm_mul_ps(xi[i], yi[i])); )
		//		_dp/*4*/ = _mm_hadd_ps(_dp, _dp);
		//		_dp/*4*/ = _mm_hadd_ps(_dp, _dp);
		//		if (Ns != 0)
		//			switch (Ns) {
		//			case 3:	_dp = _mm_add_ss(_dp, _mm_dp_ps(*xi, *yi, _mm_dp_ps_arg(3))); break;
		//			case 2:	_dp = _mm_add_ss(_dp, _mm_dp_ps(*xi, *yi, _mm_dp_ps_arg(2))); break;
		//			case 1:	_dp = _mm_add_ss(_dp, _mm_dp_ps(*xi, *yi, _mm_dp_ps_arg(1))); break;
		//			default: abort(); 
		//			}
		//		return _mm_cvtss_f32(_dp);
		//	}
		//}
	}
	
	template<auto D>
	inline float dot(const smdarray<float,D,__m128>& x) {
		//if constexpr ( static_mdarray<mdarray_view<float,dimension,allocator>> ) {
			constexpr size_t N = vsize(smdarray<float,D,__m128>{});
			constexpr size_t Ns = N%x.simdlen;
			constexpr size_t Nb = N/x.simdlen;
			if constexpr (Nb == 0) {
				return _mm_cvtss_f32( _mm_dp_ps(x.__data[0], x.__data[0], ((1<<(Ns+4)) - 1)) );
			} else {
				__m128 _dp/*4*/ = _mm_mul_ps(x.__data[0], x.__data[0]); constexpr size_t Nbm1 = Nb-1;
				iterator_based_unroll4x_for(constexpr, Nbm1, xi, const auto *xi = std::next(x.__data), xi+=4, ++xi,
					_dp/*4*/ = _mm_add_ps(_dp/*4*/, _mm_mul_ps(xi[i], xi[i])); )
				_dp/*4*/ = _mm_hadd_ps(_dp, _dp);
				_dp/*4*/ = _mm_hadd_ps(_dp, _dp);
				if constexpr (Ns != 0)
					_dp = _mm_add_ss(_dp, _mm_dp_ps(*xi, *xi, ((1<<(Ns+4)) - 1)));
				return _mm_cvtss_f32(_dp);
			}
		//} else {
		//	assert( !x.empty() );
		//	size_t       Nb = vsize(x);
		//	const size_t Ns = Nb%x.simdlen;
		//	Nb /= x.simdlen;
		//	if (Nb == 0) {
		//		switch (Ns) {
		//		case 3:	return _mm_cvtss_f32( _mm_dp_ps(x._Data[0], x._Data[0], _mm_dp_ps_arg(3)) );
		//		case 2:	return _mm_cvtss_f32( _mm_dp_ps(x._Data[0], x._Data[0], _mm_dp_ps_arg(2)) );
		//		case 1:	return _mm_cvtss_f32( _mm_dp_ps(x._Data[0], x._Data[0], _mm_dp_ps_arg(1)) );
		//		default: abort(); 
		//		}
		//	} else {
		//		__m128 _dp/*4*/ = _mm_mul_ps(x._Data[0], x._Data[0]); --Nb;
		//		iterator_based_unroll4x_for(, Nb, xi, const auto *xi = std::next(x._Data), xi+=4, ++xi, 
		//			_dp/*4*/ = _mm_add_ps(_dp/*4*/, _mm_mul_ps(xi[i], xi[i])); )
		//		_dp/*4*/ = _mm_hadd_ps(_dp, _dp);
		//		_dp/*4*/ = _mm_hadd_ps(_dp, _dp);
		//		if (Ns != 0)
		//			switch (Ns) {
		//			case 3:	_dp = _mm_add_ss(_dp, _mm_dp_ps(*xi, *xi, _mm_dp_ps_arg(3))); break;
		//			case 2:	_dp = _mm_add_ss(_dp, _mm_dp_ps(*xi, *xi, _mm_dp_ps_arg(2))); break;
		//			case 1:	_dp = _mm_add_ss(_dp, _mm_dp_ps(*xi, *xi, _mm_dp_ps_arg(1))); break;
		//			default: abort(); 
		//			}
		//		return _mm_cvtss_f32(_dp);
		//	}
		//}
	}
	
	template<auto D>
	inline smdarray<float,D,__m128> cross(const smdarray<float,D,__m128>& x, const smdarray<float,D,__m128>& y) {
		return smdarray<float,D,__m128>{
			_mm_sub_ps( 
				_mm_mul_ps( _mm_permute_ps(x.__data[0],_MM_PERM_AACB), _mm_permute_ps(y.__data[0],_MM_PERM_ABAC)),
				_mm_mul_ps( _mm_permute_ps(x.__data[0],_MM_PERM_ABAC), _mm_permute_ps(y.__data[0],_MM_PERM_AACB) )
			)
		};
	}


#if 0
#include <iostream>
#include <fstream>
#include <string>

int main() {
	std::fstream fin("temp.txt", std::ios::in);
	while (fin.good())
	{
		std::string line;
		std::getline(fin, line);

		size_t pos = line.find("float");
		while (pos != std::string::npos)
		{
			line.replace(pos, strlen("float"), "double");
			pos = line.find("float", pos + 1);
		}

		 pos = line.find("__m128");
		while (pos != std::string::npos)
		{
			line.replace(pos, strlen("__m128"), "__m256d");
			pos = line.find("__m128", pos + 1);
		}

		pos = line.find("_mm_");
		while (pos != std::string::npos)
		{
			line.replace(pos, strlen("_mm_"), "_mm256_");
			pos = line.find("_mm_", pos + 1);
		}

		pos = line.find("_ps");
		while (pos != std::string::npos)
		{
			line.replace(pos, strlen("_ps"), "_pd");
			pos = line.find("_ps", pos + 1);
		}

		std::cout << line << "\n";
	}

	return 0;
}
#endif
	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_assign_function_x_y(double,D,__m256d, operator+=, _mm256_add_pd, D2)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_assign_function_x_yval(double,D,__m256d, operator+=, _mm256_add_pd, T2,_mm256_set1_pd)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_function_x_y(double,D,__m256d, operator+, _mm256_add_pd, D2)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_function_x_yval(double,D,__m256d, operator+, _mm256_add_pd, T2,_mm256_set1_pd)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_function_xval_y(double,D,__m256d, operator+, _mm256_add_pd, T2,_mm256_set1_pd)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_assign_function_x_y(double,D,__m256d, operator-=, _mm256_sub_pd, D2)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_assign_function_x_yval(double,D,__m256d, operator-=, _mm256_sub_pd, T2,_mm256_set1_pd)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_function_x_y(double,D,__m256d, operator-, _mm256_sub_pd, D2)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_function_x_yval(double,D,__m256d, operator-, _mm256_sub_pd, T2,_mm256_set1_pd)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_function_xval_y(double,D,__m256d, operator-, _mm256_sub_pd, T2,_mm256_set1_pd)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_assign_function_x_y(double,D,__m256d, operator*=, _mm256_mul_pd, D2)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_assign_function_x_yval(double,D,__m256d, operator*=, _mm256_mul_pd, T2,_mm256_set1_pd)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_function_x_y(double,D,__m256d, operator*, _mm256_mul_pd, D2)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_function_x_yval(double,D,__m256d, operator*, _mm256_mul_pd, T2,_mm256_set1_pd)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_function_xval_y(double,D,__m256d, operator*, _mm256_mul_pd, T2,_mm256_set1_pd)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_assign_function_x_y(double,D,__m256d, operator/=, _mm256_div_pd, D2)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_assign_function_x_yval(double,D,__m256d, operator/=, _mm256_div_pd, T2,_mm256_set1_pd)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_function_x_y(double,D,__m256d, operator/, _mm256_div_pd, D2)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_function_x_yval(double,D,__m256d, operator/, _mm256_div_pd, T2,_mm256_set1_pd)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_function_xval_y(double,D,__m256d, operator/, _mm256_div_pd, T2,_mm256_set1_pd)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_function_x_y(double,D,__m256d, fmod, _mm256_fmod_pd, D2)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_function_x_yval(double,D,__m256d, fmod, _mm256_fmod_pd, T2,_mm256_set1_pd)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_function_xval_y(double,D,__m256d, fmod, _mm256_fmod_pd, T2,_mm256_set1_pd)

	template<auto D>
	inline __smdarray_function_x_y(double,D,__m256d, min, _mm256_min_pd, D)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_function_x_y(double,D,__m256d, min, _mm256_min_pd, D2)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_function_x_yval(double,D,__m256d, min, _mm256_min_pd, T2,_mm256_set1_pd)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_function_xval_y(double,D,__m256d, min, _mm256_min_pd, T2,_mm256_set1_pd)

	template<auto D>
	inline __smdarray_function_x_y(double,D,__m256d, max, _mm256_max_pd, D)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_function_x_y(double,D,__m256d, max, _mm256_max_pd, D2)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_function_x_yval(double,D,__m256d, max, _mm256_max_pd, T2,_mm256_set1_pd)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_function_xval_y(double,D,__m256d, max, _mm256_max_pd, T2,_mm256_set1_pd)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_function_x_y(double,D,__m256d, copysign, _mm256_copysign_pd, D2)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_function_x_yval(double,D,__m256d, copysign, _mm256_copysign_pd, T2,_mm_set1_ps)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_function_xval_y(double,D,__m256d, copysign, _mm256_copysign_pd, T2,_mm_set1_ps)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, operator-, _mm256_neg_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, abs,   _mm256_abs_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, trunc, _mm256_trunc_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, floor, _mm256_floor_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, ceil,  _mm256_ceil_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, round, _mm256_round_even_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, sqrt,  _mm256_sqrt_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, cbrt,  _mm256_cbrt_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, exp,   _mm256_exp_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, exp2,  _mm256_exp2_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, log,   _mm256_log_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, log2,  _mm256_log2_pd)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_function_x_y(double,D,__m256d, pow, _mm256_pow_pd, D2)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_function_x_yval(double,D,__m256d, pow, _mm256_pow_pd, T2,_mm256_set1_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, cosh, _mm256_cosh_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, sinh, _mm256_sinh_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, tanh, _mm256_tanh_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, acosh, _mm256_acosh_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, asinh, _mm256_asinh_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, atanh, _mm256_atanh_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, cos, _mm256_cos_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, sin, _mm256_sin_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, tan, _mm256_tan_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, acos, _mm256_acos_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, asin, _mm256_asin_pd)

	template<auto D>
	inline __smdarray_function_x(double,D,__m256d, atan, _mm256_atan_pd)

	template<auto D, auto D2> requires(D == D2)
	inline __smdarray_function_x_y(double,D,__m256d, atan2, _mm256_atan2_pd, D2)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_function_x_yval(double,D,__m256d, atan2, _mm256_atan2_pd, T2,_mm256_set1_pd)

	template<auto D, std::convertible_to<double> T2>
	inline __smdarray_function_xval_y(double,D,__m256d, atan2, _mm256_atan2_pd, T2,_mm256_set1_pd)
	
	template<size_t n>
	inline __m128d _mm256_dp_pd(const __m256d& ymm1, const __m256d& ymm2) {
		if constexpr(n==4) {
			__m256d xyzw = _mm256_mul_pd(ymm1, ymm2);
			__m128d xy = _mm256_extractf128_pd(xyzw, 0);
			__m128d zw = _mm256_extractf128_pd(xyzw, 1);
			__m128d xz_yw = _mm_add_pd(xy, zw);
			__m128d yw_xz = _mm_shuffle_pd(xz_yw,xz_yw,0b01);
			return _mm_add_sd(xz_yw, yw_xz);
		} else if constexpr(n==3) {
			__m256d xyzw = _mm256_mul_pd(ymm1, ymm2);
			__m128d xy = _mm256_extractf128_pd(xyzw, 0);
			__m128d yx = _mm_shuffle_pd(xy, xy, 0b01);
			__m128d zw = _mm256_extractf128_pd(xyzw, 1);
			return _mm_add_sd(_mm_add_sd(xy,yx),zw);
		} else if constexpr(n==2) {
			__m128d xy = _mm_mul_pd(_mm256_extractf128_pd(ymm1,0),_mm256_extractf128_pd(ymm2,0));
			__m128d yx = _mm_shuffle_pd(xy, xy, 0b01);
			return _mm_add_sd(xy,yx);
		} else if constexpr(n==1) {
			return _mm_mul_sd(_mm256_extractf128_pd(ymm1,0),_mm256_extractf128_pd(ymm2,0));
		} else {
			abort();
		}
	}

#if 0
	template<size_t dimension, typename allocator>
		requires std::is_same_v<typename allocator::value_type, __m256d>
	inline double dot(const mdarray_view<double,dimension,allocator> &x, const mdarray_view<double,dimension,allocator> &y) {
		if constexpr ( static_mdarray<mdarray_view<double,dimension,allocator>> ) {
			constexpr size_t N = vsize<double,dimension,allocator>();
			constexpr size_t Ns = N%x.simdlen;
			constexpr size_t Nb = N/x.simdlen;
			if constexpr (Nb == 0) {
				return _mm_cvtsd_f64( _mm256_dp_pd<Ns>(x._Data[0], y._Data[0]) );
			} else {
				__m256d _dp4 = _mm256_mul_pd(x._Data[0], y._Data[0]); constexpr size_t Nbm1 = Nb-1;
				iterator_based_unroll4x_for(constexpr, Nbm1, xi, 
					epconj(const auto *xi = std::next(x._Data), *yi = std::next(y._Data)), 
						epconj(xi+=4, yi+=4), 
							epconj(++xi, ++yi), 
					_dp4 = _mm256_add_pd(_dp4, _mm256_mul_pd(xi[i], yi[i])); )
				_dp4 = _mm256_hadd_pd(_dp4, _dp4);
				__m128d _dp = _mm_add_sd(_mm256_extractf128_pd(_dp4, 0), _mm256_extractf128_pd(_dp4, 1));
				if constexpr (Ns != 0)
					_dp = _mm_add_sd(_dp, _mm256_dp_pd<Ns>(*xi, *yi));
				return _mm_cvtsd_f64(_dp);
			}
		} else {
			assert( !x.empty() );
			size_t       Nb = vsize(x);
			const size_t Ns = Nb%x.simdlen;
			Nb /= x.simdlen;
			if (Nb == 0) {
				switch (Ns) {
				case 3:	return _mm_cvtsd_f64( _mm256_dp_pd<3>(x._Data[0], y._Data[0]) );
				case 2:	return _mm_cvtsd_f64( _mm256_dp_pd<2>(x._Data[0], y._Data[0]) );
				case 1:	return _mm_cvtsd_f64( _mm256_dp_pd<1>(x._Data[0], y._Data[0]) );
				default: abort(); 
				}
			} else {
				__m256d _dp4 = _mm256_mul_pd(x._Data[0], y._Data[0]); --Nb;
				iterator_based_unroll4x_for(, Nb, xi, 
					epconj(const auto *xi = std::next(x._Data), *yi = std::next(y._Data)),
						epconj(xi+=4, yi+=4),
							epconj(++xi, ++yi),
					_dp4 = _mm256_add_pd(_dp4, _mm256_mul_pd(xi[i], yi[i])); )
				_dp4 = _mm256_hadd_pd(_dp4, _dp4);
				__m128d _dp = _mm_add_sd(_mm256_extractf128_pd(_dp4, 0), _mm256_extractf128_pd(_dp4, 1));
				if (Ns != 0)
					switch (Ns) {
					case 3:	_dp = _mm_add_sd(_dp, _mm256_dp_pd<3>(*xi, *yi)); break;
					case 2:	_dp = _mm_add_sd(_dp, _mm256_dp_pd<2>(*xi, *yi)); break;
					case 1:	_dp = _mm_add_sd(_dp, _mm256_dp_pd<1>(*xi, *yi)); break;
					default: abort();
					}
				return _mm_cvtsd_f64(_dp);
			}
		}
	}

	template<size_t dimension, typename allocator>
		requires std::is_same_v<typename allocator::value_type, __m256d>
	inline double dot(const mdarray_view<double,dimension,allocator> &x) {
		if constexpr ( static_mdarray<mdarray_view<double,dimension,allocator>> ) {
			constexpr size_t N = vsize<double,dimension,allocator>();
			constexpr size_t Ns = N%x.simdlen;
			constexpr size_t Nb = N/x.simdlen;
			if constexpr (Nb == 0) {
				return _mm_cvtsd_f64( _mm256_dp_pd<Ns>(x._Data[0], x._Data[0]) );
			} else {
				__m256d _dp4 = _mm256_mul_pd(x._Data[0], x._Data[0]); constexpr size_t Nbm1 = Nb-1;
				iterator_based_unroll4x_for(constexpr, Nbm1, xi, const auto *xi = std::next(x._Data), xi+=4, ++xi, 
					_dp4 = _mm256_add_pd(_dp4, _mm256_mul_pd(xi[i], xi[i])); )
				_dp4 = _mm256_hadd_pd(_dp4, _dp4);
				__m128d _dp = _mm_add_sd(_mm256_extractf128_pd(_dp4, 0), _mm256_extractf128_pd(_dp4, 1));
				if constexpr (Ns != 0)
					_dp = _mm_add_sd(_dp, _mm256_dp_pd<Ns>(*xi, *xi));
				return _mm_cvtsd_f64(_dp);
			}
		} else {
			assert( !x.empty() );
			size_t       Nb = vsize(x);
			const size_t Ns = Nb%x.simdlen;
			Nb /= x.simdlen;
			if (Nb == 0) {
				switch (Ns) {
				case 3:	return _mm_cvtsd_f64( _mm256_dp_pd<3>(x._Data[0], x._Data[0]) );
				case 2:	return _mm_cvtsd_f64( _mm256_dp_pd<2>(x._Data[0], x._Data[0]) );
				case 1:	return _mm_cvtsd_f64( _mm256_dp_pd<1>(x._Data[0], x._Data[0]) );
				default: abort(); 
				}
			} else {
				__m256d _dp4 = _mm256_mul_pd(x._Data[0], x._Data[0]); --Nb;
				iterator_based_unroll4x_for(, Nb, xi, const auto *xi = std::next(x._Data), xi+=4, ++xi,
					_dp4 = _mm256_add_pd(_dp4, _mm256_mul_pd(xi[i], xi[i])); )
				_dp4 = _mm256_hadd_pd(_dp4, _dp4);
				__m128d _dp = _mm_add_sd(_mm256_extractf128_pd(_dp4, 0), _mm256_extractf128_pd(_dp4, 1));
				if (Ns != 0)
					switch (Ns) {
					case 3:	_dp = _mm_add_sd(_dp, _mm256_dp_pd<3>(*xi, *xi)); break;
					case 2:	_dp = _mm_add_sd(_dp, _mm256_dp_pd<2>(*xi, *xi)); break;
					case 1:	_dp = _mm_add_sd(_dp, _mm256_dp_pd<1>(*xi, *xi)); break;
					default: abort(); 
					}
				return _mm_cvtsd_f64(_dp);
			}
		}
	}
#endif


	template<>
	struct smdsize_type_deduce<__m128i> {
		using value_type = int;
		using package_type = __m128i;
	};

  template<auto D, auto D2> requires(D == D2)
  inline __smdarray_assign_function_x_y(int,D,__m128i, operator+=, _mm_add_epi32, D2)

  template<auto D, std::convertible_to<int> T2>
  inline __smdarray_assign_function_x_yval(int,D,__m128i, operator+=, _mm_add_epi32, T2,_mm_set1_epi32)

  template<auto D, auto D2> requires(D == D2)
  inline __smdarray_function_x_y(int,D,__m128i, operator+, _mm_add_epi32, D2)

  template<auto D, std::convertible_to<int> T2>
  inline __smdarray_function_x_yval(int,D,__m128i, operator+, _mm_add_epi32, T2,_mm_set1_epi32)

  template<auto D, std::convertible_to<int> T2>
  inline __smdarray_function_xval_y(int,D,__m128i, operator+, _mm_add_epi32, T2,_mm_set1_epi32)

  template<auto D, auto D2> requires(D == D2)
  inline __smdarray_assign_function_x_y(int,D,__m128i, operator-=, _mm_sub_epi32, D2)

  template<auto D, std::convertible_to<int> T2>
  inline __smdarray_assign_function_x_yval(int,D,__m128i, operator-=, _mm_sub_epi32, T2,_mm_set1_epi32)

  template<auto D, auto D2> requires(D == D2)
  inline __smdarray_function_x_y(int,D,__m128i, operator-, _mm_sub_epi32, D2)

  template<auto D, std::convertible_to<int> T2>
  inline __smdarray_function_x_yval(int,D,__m128i, operator-, _mm_sub_epi32, T2,_mm_set1_epi32)

  template<auto D, std::convertible_to<int> T2>
  inline __smdarray_function_xval_y(int,D,__m128i, operator-, _mm_sub_epi32, T2,_mm_set1_epi32)

  template<auto D, auto D2> requires(D == D2)
  inline __smdarray_assign_function_x_y(int,D,__m128i, operator*=, _mm_mullo_epi32, D2)

  template<auto D, std::convertible_to<int> T2>
  inline __smdarray_assign_function_x_yval(int,D,__m128i, operator*=, _mm_mullo_epi32, T2,_mm_set1_epi32)

  template<auto D, auto D2> requires(D == D2)
  inline __smdarray_function_x_y(int,D,__m128i, operator*, _mm_mullo_epi32, D2)

  template<auto D, std::convertible_to<int> T2>
  inline __smdarray_function_x_yval(int,D,__m128i, operator*, _mm_mullo_epi32, T2,_mm_set1_epi32)

  template<auto D, std::convertible_to<int> T2>
  inline __smdarray_function_xval_y(int,D,__m128i, operator*, _mm_mullo_epi32, T2,_mm_set1_epi32)

  template<auto D, auto D2> requires(D == D2)
  inline __smdarray_assign_function_x_y(int,D,__m128i, operator/=, _mm_div_epi32, D2)

  template<auto D, std::convertible_to<int> T2>
  inline __smdarray_assign_function_x_yval(int,D,__m128i, operator/=, _mm_div_epi32, T2,_mm_set1_epi32)

  template<auto D, auto D2> requires(D == D2)
  inline __smdarray_function_x_y(int,D,__m128i, operator/, _mm_div_epi32, D2)

  template<auto D, std::convertible_to<int> T2>
  inline __smdarray_function_x_yval(int,D,__m128i, operator/, _mm_div_epi32, T2,_mm_set1_epi32)

  template<auto D, std::convertible_to<int> T2>
  inline __smdarray_function_xval_y(int,D,__m128i, operator/, _mm_div_epi32, T2,_mm_set1_epi32)

	template<auto D, auto D2> requires(D == D2)
  inline __smdarray_assign_function_x_y(int,D,__m128i, operator%=, _mm_rem_epi32, D2)

  template<auto D, std::convertible_to<int> T2>
  inline __smdarray_assign_function_x_yval(int,D,__m128i, operator%=, _mm_rem_epi32, T2,_mm_set1_epi32)

  template<auto D, auto D2> requires(D == D2)
  inline __smdarray_function_x_y(int,D,__m128i, operator%, _mm_rem_epi32, D2)

  template<auto D, std::convertible_to<int> T2>
  inline __smdarray_function_x_yval(int,D,__m128i, operator%, _mm_rem_epi32, T2,_mm_set1_epi32)

  template<auto D, std::convertible_to<int> T2>
  inline __smdarray_function_xval_y(int,D,__m128i, operator%, _mm_rem_epi32, T2,_mm_set1_epi32)

  template<auto D>
  inline __smdarray_function_x_y(int,D,__m128i, min, _mm_min_epi32, D)

  template<auto D, auto D2> requires(D == D2)
  inline __smdarray_function_x_y(int,D,__m128i, min, _mm_min_epi32, D2)

  template<auto D, std::convertible_to<int> T2>
  inline __smdarray_function_x_yval(int,D,__m128i, min, _mm_min_epi32, T2,_mm_set1_epi32)

  template<auto D, std::convertible_to<int> T2>
  inline __smdarray_function_xval_y(int,D,__m128i, min, _mm_min_epi32, T2,_mm_set1_epi32)

  template<auto D>
  inline __smdarray_function_x_y(int,D,__m128i, max, _mm_max_epi32, D)

  template<auto D, auto D2> requires(D == D2)
  inline __smdarray_function_x_y(int,D,__m128i, max, _mm_max_epi32, D2)

  template<auto D, std::convertible_to<int> T2>
  inline __smdarray_function_x_yval(int,D,__m128i, max, _mm_max_epi32, T2,_mm_set1_epi32)

  template<auto D, std::convertible_to<int> T2>
  inline __smdarray_function_xval_y(int,D,__m128i, max, _mm_max_epi32, T2,_mm_set1_epi32)

  template<auto D>
  inline __smdarray_function_x(int,D,__m128i, operator-, _mm_neg_epi32)

  template<auto D>
  inline __smdarray_function_x(int,D,__m128i, abs,   _mm_abs_epi32)

	/*template<size_t N>
	inline smdsize<__m128i,N> fridx(const int _Idx, const smdsize<__m128i,N>& _Mdstride) {
		/// Factorize index'_Idx' into result with strides'_Mdstride'.
		smdsize<__m128i,N> _Mdidx;
		_Mdidx[N - 1] = _Idx;
		for (size_t i = N - 1; i != 0; --i) {
			_Mdidx[i-1] = _Mdidx[i] % _Mdstride[i];
			_Mdidx[i] /= _Mdstride[i];
		}

		return _Mdidx;
	}*/
}// end of namespace math