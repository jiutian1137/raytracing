#pragma once

///@brief Fraction 
///@license Free 
///@review 2022-8-9 
///@contact Jiang1998Nan@outlook.com 
#define _MATH_FRACTION_

#include <cassert>
#include <math/concepts.hpp>

namespace math {
	template<typename valuety1, typename valuety2, typename valuety3>
	constexpr valuety1 remap(const valuety1& value, const valuety2& lower, const valuety2& upper, const valuety3& new_lower, const valuety3& new_upper) {
		return (value - lower)/(upper - lower) * (new_upper - new_lower) + new_lower;
	}

	template<typename valuety, typename factorty>
	constexpr valuety lerp(const valuety& x0, const valuety& x1, const factorty& t) {
		return x0 + (x1 - x0)*t;
		/// lerp(A+B,C+D,t) = (C+D - (A+B))*t + A+B = lerp(A,C,t)+lerp(B,D,t) = lerp(B,C,t)+lerp(A,D,t).
	}
 
	template<math::ndimensional<2> container2, typename vector2>
	constexpr auto lerp(const container2& X, const typename container2::size_type& i0, const typename container2::size_type& i1, const vector2& t) {
		auto i0x = i0[0];
		auto i0y = i0[1] * X.stride(1);
		auto i1x = i1[0];
		auto i1y = i1[1] * X.stride(1);
		return lerp(
			lerp(X[i0x+i0y], X[i1x+i0y], t[0]),
			lerp(X[i0x+i1y], X[i1x+i1y], t[0]),
			t[1]
			);
	}
	
	template<math::ndimensional<3> container3, typename vector3>
	constexpr auto lerp(const container3& X, const typename container3::size_type& i0, const typename container3::size_type& i1, const vector3& t) {
		auto vi0 = i0 * X.stride();
		auto vi1 = i1 * X.stride();
		/// n0+n1+n2 = A0+n1 = C0
		/// t0+n1+n2 = B0+n1 = C1
		/// n0+t1+n2 = A0+t1 = D0
		/// t0+t1+n2 = B0+t1 = D1
		/// n0+n1+t2 = B1+n1 = C3
		/// t0+n1+t2 = A1+n1 = C2
		/// n0+t1+t2 = B1+t1 = D2
		/// t0+t1+t2 = A1+t1 = D3
		///        A = {n0,t0,n1,t1}+{n2,t2,...} = unpacklo(n,t) + unpackhi(n,t)
		///        B = {t0,n0,t1,n1}+{n2,t2,...} = unpacklo(t,n) + unpackhi(n,t)
		///        C = {A0,B0,A1,B1}+{n1}
		///        D = {A0,B0,A1,B1}+{t1}
		auto tmp = permute<2,4+2,3,4+3>(vi0, vi1);
		auto A = prrmute<0,4+0,1,4+1>(vi0,vi1) + tmp;
		auto B = prrmute<0,4+0,1,4+1>(vi1,vi0) + tmp;
		tmp = permute<0,4+0,1,4+1>(A,B);
		auto C = tmp + permute<1,1,1,1>(vi0);
		auto D = tmp + permute<1,1,1,1>(vi1);
		return lerp(
			lerp(
				lerp(X[extract<0>(C)], X[extract<1>(C)], t[0]),
				lerp(X[extract<0>(D)], X[extract<1>(D)], t[0]),
				t[1]
				),
			lerp(
				lerp(X[extract<3>(C)], X[extract<2>(C)], t[0]),
				lerp(X[extract<2>(D)], X[extract<3>(D)], t[0]),
				t[1]
				),
			t[2]
			);
	}

	template<math::ndimensional<4> container4, typename vector4>
	constexpr auto lerp(const container4& X, const typename container4::size_type& _i0, const typename container4::size_type& _i1, const vector4& t) {
#if 0
		auto i0x = _i0[0]/* * data.stride(0)*/;
		auto i0y = _i0[1] * X.stride(1);
		auto i0z = _i0[2] * X.stride(2);
		auto i0w = _i0[3] * X.stride(3);
		auto i1x = _i1[0]/* * data.stride(0)*/;
		auto i1y = _i1[1] * X.stride(1);
		auto i1z = _i1[2] * X.stride(2);
		auto i1w = _i1[3] * X.stride(3);

		auto i0z_i0w = i0z + i0w;
		auto i0y_i0z_i0w = i0y + i0z_i0w;
		auto i1y_i0z_i0w = i1y + i0z_i0w;
		auto result = lerp(
			lerp(X[i0x + i0y_i0z_i0w], X[i1x + i0y_i0z_i0w], t[0]),
			lerp(X[i0x + i1y_i0z_i0w], X[i1x + i1y_i0z_i0w], t[0]),
			t[1]
		);
		//std::cout << i0x + i0y_i0z_i0w << ' ' << i1x + i0y_i0z_i0w << ' ' << i0x + i1y_i0z_i0w << ' ' << i1x + i1y_i0z_i0w << std::endl;

		auto i1z_i0w = i1z + i0w;
		auto i0y_i1z_i0w = i0y + i1z_i0w;
		auto i1y_i1z_i0w = i1y + i1z_i0w;
		result = lerp(
			result,
			lerp(
				lerp(X[i0x + i0y_i1z_i0w], X[i1x + i0y_i1z_i0w], t[0]),
				lerp(X[i0x + i1y_i1z_i0w], X[i1x + i1y_i1z_i0w], t[0]),
				t[1]
			),
			t[2]
		);
		//std::cout << i0x + i0y_i1z_i0w << ' ' << i1x + i0y_i1z_i0w << ' ' << i0x + i1y_i1z_i0w << ' ' << i1x + i1y_i1z_i0w << std::endl;

		auto i0z_i1w = i0z + i1w;
		auto i0y_i0z_i1w = i0y + i0z_i1w;
		auto i1y_i0z_i1w = i1y + i0z_i1w;
		auto result2 = lerp(
			lerp(X[i0x + i0y_i0z_i1w], X[i1x + i0y_i0z_i1w], t[0]),
			lerp(X[i0x + i1y_i0z_i1w], X[i1x + i1y_i0z_i1w], t[0]),
			t[1]
		);
		//std::cout << i0x + i0y_i0z_i1w << ' ' << i1x + i0y_i0z_i1w << ' ' << i0x + i1y_i0z_i1w << ' ' << i1x + i1y_i0z_i1w << std::endl;

		auto i1z_i1w = i1z + i1w;
		auto i0y_i1z_i1w = i0y + i1z_i1w;
		auto i1y_i1z_i1w = i1y + i1z_i1w;
		result2 = lerp(
			result2,
			lerp(
				lerp(X[i0x + i0y_i1z_i1w], X[i1x + i0y_i1z_i1w], t[0]),
				lerp(X[i0x + i1y_i1z_i1w], X[i1x + i1y_i1z_i1w], t[0]),
				t[1]
			),
			t[2]
		);
		//std::cout << i0x + i0y_i1z_i1w << ' ' << i1x + i0y_i1z_i1w << ' ' << i0x + i1y_i1z_i1w << ' ' << i1x + i1y_i1z_i1w << std::endl;

		return lerp(result, result2, t[3]);
#elif 1
		auto i0 = _i0 * X.stride();
		auto i1 = _i1 * X.stride();
		/// n0+n1+n2+n3 = A0+A2 = C0
		/// t0+n1+n2+n3 = A2+B0 = D0
		/// n0+t1+n2+n3 = A0+B2 = D1
		/// t0+t1+n2+n3 = B0+B2 = C1
		/// n0+n1+t2+n3 = A2+B1 = F0
		/// t0+n1+t2+n3 = A1+A2 = E0
		/// n0+t1+t2+n3 = B1+B2 = F1
		/// t0+t1+t2+n3 = A1+B2 = E1
		/// n0+n1+n2+t3 = A0+B3 = E3
		/// t0+n1+n2+t3 = B0+B3 = F3
		/// n0+t1+n2+t3 = A0+A3 = E2
		/// t0+t1+n2+t3 = A3+B0 = F2
		/// n0+n1+t2+t3 = B1+B3 = C3
		/// t0+n1+t2+t3 = A1+B3 = D3
		/// n0+t1+t2+t3 = A3+B1 = D2
		/// t0+t1+t2+t3 = A1+A3 = C2
		///           A = {n0,t0,n1,t1}+{n2,t2,n3,t3} = unpacklo(n,t) + unpackhi(n,t)
		///           B = {t0,n0,t1,n1}+{n2,t2,n3,t3} = unpacklo(t,n) + unpackhi(n,t)
		/// 
		///           C = {A0,B0,A1,B1}+{A2,B2,A3,B3} = unpacklo(A,B) + unpackhi(A,B)
		///           D = {A2,B2,A3,B3}+{B0,A0,B1,A1} = unpackhi(A,B) + unpacklo(B,A)
		///           E = {A1,A1,A0,A0}+{A2,B2,A3,B3} = permute1100(A) + unpackhi(A,B)
		///           F = {B1,B1,B0,B0}+{A2,B2,A3,B3} = permute1100(B) + unpackhi(A,B)
		/// 
		auto tmp = permute<2, 4 + 2, 3, 4 + 3>(i0, i1);
		auto A = permute<0, 4 + 0, 1, 4 + 1>(i0, i1) + tmp;
		auto B = permute<0, 4 + 0, 1, 4 + 1>(i1, i0) + tmp;
		tmp = permute<2, 4 + 2, 3, 4 + 3>(A, B);
		auto C = permute<0, 4 + 0, 1, 4 + 1>(A, B) + tmp;
		auto D = permute<0, 4 + 0, 1, 4 + 1>(B, A) + tmp;
		auto E = permute<1, 1, 0, 0>(A) + tmp;
		auto F = permute<1, 1, 0, 0>(B) + tmp;
		/*std::cout << C[0] << ' ' << D[0] << ' ' << D[1] << ' ' << C[1] << std::endl;
		std::cout << F[0] << ' ' << E[0] << ' ' << F[1] << ' ' << E[1] << std::endl;
		std::cout << E[3] << ' ' << F[3] << ' ' << E[2] << ' ' << F[2] << std::endl;
		std::cout << C[3] << ' ' << D[3] << ' ' << D[2] << ' ' << C[2] << std::endl;*/
		return lerp(
			lerp(
				lerp(
					lerp(X[extract<0>(C)], X[extract<0>(D)], t[0]),
					lerp(X[extract<1>(D)], X[extract<1>(C)], t[0]),
					t[1]
				),
				lerp(
					lerp(X[extract<0>(F)], X[extract<0>(E)], t[0]),
					lerp(X[extract<1>(F)], X[extract<1>(E)], t[0]),
					t[1]
				),
				t[2]
			),
			lerp(
				lerp(
					lerp(X[extract<3>(E)], X[extract<3>(F)], t[0]),
					lerp(X[extract<2>(E)], X[extract<2>(F)], t[0]),
					t[1]
				),
				lerp(
					lerp(X[extract<3>(C)], X[extract<3>(D)], t[0]),
					lerp(X[extract<2>(D)], X[extract<2>(C)], t[0]),
					t[1]
				),
				t[2]
			),
			t[3]
		);
#endif
	}
	

	namespace sample_borders {
		struct __clamp {
			template<typename container, typename vector>
			inline void operator()(const container& cont, vector& xi, typename container::size_type& i0, typename container::size_type& i1) const {
				if constexpr (math::signed_vector<vector>) {
					xi = max(xi, 0);
				}
				if constexpr (std::same_as<std::remove_cvref_t<decltype(cont.edge())>, typename container::size_type>) {
					i0 = static_vcast<typename container::size_type>(xi);
					i1 = i0 + 1;
					i0 = min(i0, cont.edge());
					i1 = min(i1, cont.edge());
				} else {
					xi = min(xi, cont.edge());
					i0 = static_vcast<typename container::size_type>(xi);
					xi += 1;
					xi = min(xi, cont.edge());
					i1 = static_vcast<typename container::size_type>(xi);
					/*i0 = static_vcast<typename container::size_type>(vmin(xi, cont.edge()));
					i1 = static_vcast<typename container::size_type>(vmin(xi+1, cont.edge()));*/
				}
			}

			template<typename container, typename vector, size_t size, size_t anchor>
			inline void operator()(const container& cont, typename container::size_type (&i)[size], std::integral_constant<size_t,anchor>) const {
				if constexpr (math::signed_vector<typename container::size_type>) {
					i[anchor] = vmax(i[anchor], 0);
				}
				constexpr size_t radius = (size - 1) - anchor;
				for (size_t k = 1, kend = std::min(radius,anchor) + 1; k != kend; ++k) {
					i[anchor+k] = i[anchor] + k;
					i[anchor-k] = i[anchor] - k;
				}
				if constexpr (radius > anchor) {
					for (size_t k = anchor + 1, kend = radius + 1; k != kend; ++k) 
						i[anchor+k] = i[anchor] + k;
				} else if constexpr (anchor > radius) {
					for (size_t k = radius + 1, kend = anchor + 1; k != kend; ++k) 
						i[anchor-k] = i[anchor] - k;
				}
				for (size_t k = 0; k != size; ++k) {
					i[k] = vmin(i[k], cont.edge());
				}
			}
		} clamp;

		struct __period {
			template<typename container, typename vector>
			inline void operator()(const container& cont, vector& xi, typename container::size_type& i0, typename container::size_type& i1) const {
				if constexpr (math::signed_vector<vector>) {
					xi = vabs(xi);
				}
				if constexpr (std::same_as<std::remove_cvref_t<decltype(cont.edge())>, typename container::size_type>) {
					i0 = static_vcast<typename container::size_type>(xi);
					i1 = i0 + 1;
					i0 = vmod(i0, cont.edge());
					i1 = vmod(i1, cont.edge());
				} else {
					xi *= cont.unit();
					xi = xi - vtrunc(xi);
					i0 = static_vcast<typename container::size_type>(vround(xi * cont.edge()));
					xi += cont.unit();
					xi = xi - vtrunc(xi);
					i1 = static_vcast<typename container::size_type>(vround(xi * cont.edge()));
				}
			}
		} period;

		struct __repeat {
			template<typename container, typename vector>
			inline void operator()(const container& cont, vector& xi, typename container::size_type& i0, typename container::size_type& i1) const {
				if constexpr (std::same_as<std::remove_cvref_t<decltype(cont.edge())>, typename container::size_type>) {
					abort();
				} else {
					xi *= cont.unit();
					xi = xi - floor(xi);
					i0 = static_vcast<typename container::size_type>(round(xi * cont.edge()));
					xi += cont.unit();
					xi = xi - trunc(xi);
					i1 = static_vcast<typename container::size_type>(round(xi * cont.edge()));
				}
			}
		} repeat;
	}

	namespace sample_curves {
		struct __linear { template<typename Ty> constexpr const Ty& operator()(const Ty& t) const { return t; } } linear;
		struct __scurve { template<typename Ty> constexpr Ty operator()(const Ty& t) const { return t*t*(3 - 2*t); } } scurve;
		struct __fade { template<typename Ty> constexpr Ty operator()(const Ty& t) const { return t*t*t*(t*(t*6 - 15) + 10); } } fade;
	}

	template<typename container, typename vector, typename border_type = sample_borders::__clamp, typename curve_type = sample_curves::__linear>
	constexpr auto sample(const container& X, const vector& x, const border_type& border = sample_borders::clamp, const curve_type& curve = sample_curves::linear) {
		vector xi = floor(x);
		vector t = curve(x - xi);
		typename container::size_type i0, i1;
		border(X, xi, i0, i1);
		return lerp(X, i0, i1, t);
		/*typename container::size_type i[2];
		border(X, xi, i, std::integral_constant<size_t,0>{});
		return lerp(X, i[0], i[1], t);*/
	}

	template<typename container, typename vector, typename border_type = sample_borders::__clamp, typename curve_type = sample_curves::__linear>
	constexpr auto tex_sample(const container& X, const vector& x, const border_type& border = sample_borders::clamp, const curve_type& curve = sample_curves::linear) {
		return sample(X, x*X.edge(), border, curve);
	}

///
///		sample(container, x, ?border, ?curve)
///		tex_sample(container, x, ?border, ?curve)
/// 
/// Because computation of indices need some constants but not literal. So we can 
/// (1) compute those constants every sample. 
/// (2) precompute those constants every stream samples. 
/// (3) precompute those constants after container resized. 
/// The case(1) equal precompute those constants every sample. Unified, those constants need to
/// precomputed in all cases.
/// 
/// How place those constants?
/// (1) Embedded in function 'sample(container, x, ?border, ?curve, constants)' as arguments.
/// (2) Embedded in container rely-on concrete structure as members. 
/// (3) Embedded in a new view or wrapper.
/// Because case(1) conflicts with other cases, and the function becomes more complex. So we use 
/// case(2) and (3).
/// Because case(2) already always supports by template parameter. So we only use/process case(3).

	template<math::container array_type_or_view, typename vector>
	struct samplable_array : public array_type_or_view {
		using size_type = typename array_type_or_view::size_type;

		vector __boundary, __edge, __unit;

		void update() {
			__boundary = static_vcast<vector>(array_type_or_view::size());
			__edge = __boundary - 1;
			__unit = 1 / __edge;
		}

		const vector& bound() const {
			return __boundary;
		}

		const vector& edge() const {
			return __edge;
		}

		const vector& unit() const {
			return __unit;
		}
	};

	template<math::dynamic_container array_type, typename vector>
	struct samplable_array<array_type, vector> : array_type {
		using size_type = typename array_type::size_type;

		vector __boundary, __edge, __unit;

		constexpr samplable_array() = default;

		template<typename... types>
		samplable_array(const size_type& size, types&&... args) : array_type(size, std::forward<types&&>(args)...) {
			__boundary = static_vcast<vector>( size );
			__edge = __boundary - 1;
			__unit = 1 / __edge;
			/// 0 -- unit -- unit*2 -- unit*3 -- ... -- 1
			/// 0 --  1   --    2   --    3   -- ... -- edge
		}

		template<typename... types>
		void resize(const size_type& size, types&&... args) {
			array_type::resize(size, std::forward<types&&>(args)...);
			__boundary = static_vcast<vector>(size);
			__edge = __boundary - 1;
			__unit = 1 / __edge;
		}

		const vector& bound() const {
			return __boundary;
		}

		const vector& edge() const {
			return __edge;
		}

		const vector& unit() const {
			return __unit;
		}
	};

/// That case(1) can support by embedded in border.

#if 0
	/// sample lattice not around many cells, 
	///  so we not need window function for each cell,
	///  only window bound of neighbors, then enum all cells.
	struct samplmodes {
		struct nobound {
			/// for function lattice, requires(same( decltype(x), decltype(upper) )).
			template<typename vectorN>
			inline void operator()(vectorN& i0, vectorN& i1) const {
				i1 = i0 + 1;
			}

			/*template<typename vectorN, size_t N>
			inline void operator()(vectorN& xi, smdsize_t<N>& i0, smdsize_t<N>& i1) const {

			}*/

			/// for array lattice, may !same(decltype(x), decltype(upper)).
			template<typename vectorN>
			inline void operator()(vectorN& xi,
				size_t& ix, size_t& iy, size_t& ix1, size_t& iy1) const {
				ix = static_cast<size_t>(xi[0]);
				iy = static_cast<size_t>(xi[1]);
				ix1 = ix + 1;
				iy1 = iy + 1;
			}

			/// for array lattice, may !same(decltype(x), decltype(upper)).
			template<typename vectorN>
			inline void operator()(vectorN& xi,
				size_t& ix, size_t& iy, size_t& iz, size_t& ix1, size_t& iy1, size_t& iz1) const {
				ix = static_cast<size_t>(xi[0]);
				iy = static_cast<size_t>(xi[1]);
				iz = static_cast<size_t>(xi[2]);
				ix1 = ix + 1;
				iy1 = iy + 1;
				iz1 = iz + 1;
			}

			/// for array lattice, may !same(decltype(x), decltype(upper)).
			template<typename vectorN>
			inline void operator()(vectorN& xi,
				size_t& ix, size_t& iy, size_t& iz, size_t& iw, size_t& ix1, size_t& iy1, size_t& iz1, size_t& iw1) const {
				ix = static_cast<size_t>(xi[0]);
				iy = static_cast<size_t>(xi[1]);
				iz = static_cast<size_t>(xi[2]);
				iw = static_cast<size_t>(xi[3]);
				ix1 = ix + 1;
				iy1 = iy + 1;
				iz1 = iz + 1;
				iw1 = iw + 1;
			}
		};

		template<typename ivectorN>
		struct clamp_on_positive {
			const ivectorN& upper;

			/// assert( 0 <= upper ).
			explicit clamp_on_positive(const ivectorN& arg0) : upper(arg0) {}

			inline void operator()(ivectorN& xi, ivectorN& xi1) const {
				xi = vmin(xi, upper);
				xi1 = vmin(xi + 1, upper);
			}

			template<typename vectorN, /*size_t N*/ typename vectorNi>
			inline void operator()(vectorN& xi, vectorNi& i0, vectorNi& i1) const {
				/// if SIMD, the number of registers enough, or a little memory move. but
				/// no SIMD vector, the number of registers not enough, will have many memory move. 
				/// so, is_same<vectorN,ivectorN> when SIMD and "cvtss2si" many simpler than "minps" (addition N times cvtss2si).
				if constexpr (std::is_same_v<vectorN, ivectorN>) {
					xi = vmax(vmin(xi, upper), 0);
					i0 = static_vcast<vectorNi>(xi);
					/*if (N == 4) {
						__m128i tmp = _mm_cvtps_epi32(xi.__data[0]);
						i0[0] = _mm_extract_epi32(tmp, 0);
						i0[1] = _mm_extract_epi32(tmp, 1);
						i0[2] = _mm_extract_epi32(tmp, 2);
						i0[3] = _mm_extract_epi32(tmp, 3);
					} else {*/
						/*for (size_t i = 0; i != N; ++i) {
							assert( xi[i] >= 0 );
							i0[i] = static_cast<size_t>(xi[i]);
						}*/
					//}
					xi = vmin(xi + 1, upper);
					i1 = static_vcast<vectorNi>(xi);
				/*	if (N == 4) {
						__m128i tmp = _mm_cvtps_epi32(xi.__data[0]);
						i1[0] = _mm_extract_epi32(tmp, 0);
						i1[1] = _mm_extract_epi32(tmp, 1);
						i1[2] = _mm_extract_epi32(tmp, 2);
						i1[3] = _mm_extract_epi32(tmp, 3);
					} else {*/
						/*for (size_t i = 0; i != N; ++i) {
							assert( xi[i] >= 0 );
							i1[i] = static_cast<size_t>(xi[i]);
						}*/
					//}
				}/* else {
				/// else, no SIMD.
					for (size_t i = 0; i != N; ++i) {
						assert( xi[i] >= 0 );
						i0[i] = min(static_cast<size_t>(max(xi[i],value_t<vectorN>(0))), upper[i]);
						i1[i] = min(i0[i] + 1, upper[i]);
					}
				}*/
			}
		};

		template<typename ivectorN>
		struct clamp2 {
			const ivectorN& lower;
			const ivectorN& upper;

			/// requires lower <= upper.
			explicit clamp2(const ivectorN& arg0, const ivectorN& arg1) : lower(arg0), upper(arg1) {}

			inline void operator()(ivectorN& xi, ivectorN& xi1) const {
				xi = min(max(xi, lower), upper);
				xi1 = min(xi + 1, upper);
			}

			template<typename vectorN>
			inline void operator()(vectorN& xi, size_t& ix, size_t& iy, size_t& ix1, size_t& iy1) const {
				assert(lower[0] >= 0 && lower[1] >= 0);
				if constexpr (std::is_same_v<vectorN, ivectorN>) {
					xi = min(max(xi, lower), upper);
					ix = static_cast<size_t>(xi[0]);
					iy = static_cast<size_t>(xi[1]);
					xi = min(xi + 1, upper);
					ix1 = static_cast<size_t>(xi[0]);
					iy1 = static_cast<size_t>(xi[1]);
				} else {
					ix = xi[0] < lower[0] ? lower[0] : std::min(std::max(static_cast<size_t>(xi[0]), lower[0]), upper[0]);
					iy = xi[1] < lower[1] ? lower[1] : std::min(std::max(static_cast<size_t>(xi[1]), lower[1]), upper[1]);
					ix1 = std::min(ix + 1, upper[0]);
					iy1 = std::min(iy + 1, upper[1]);
				}
			}

			template<typename vectorN>
			inline void operator()(vectorN& xi, size_t& ix, size_t& iy, size_t& iz, size_t& ix1, size_t& iy1, size_t& iz1) const {
				assert(lower[0] >= 0 && lower[1] >= 0 && lower[2] >= 0);
				if constexpr (std::is_same_v<vectorN, ivectorN>) {
					xi = min(max(xi, lower), upper);
					ix = static_cast<size_t>(xi[0]);
					iy = static_cast<size_t>(xi[1]);
					iz = static_cast<size_t>(xi[2]);
					xi = min(xi + 1, upper);
					ix1 = static_cast<size_t>(xi[0]);
					iy1 = static_cast<size_t>(xi[1]);
					iz1 = static_cast<size_t>(xi[2]);
				} else {
					ix = xi[0] < lower[0] ? lower[0] : std::min(std::max(static_cast<size_t>(xi[0]), lower[0]), upper[0]);
					iy = xi[1] < lower[1] ? lower[1] : std::min(std::max(static_cast<size_t>(xi[1]), lower[1]), upper[1]);
					iz = xi[2] < lower[2] ? lower[2] : std::min(std::max(static_cast<size_t>(xi[2]), lower[2]), upper[2]);
					ix1 = std::min(ix + 1, upper[0]);
					iy1 = std::min(iy + 1, upper[1]);
					iz1 = std::min(iz + 1, upper[2]);
				}
			}

			template<typename vectorN>
			inline void operator()(vectorN& xi, size_t& ix, size_t& iy, size_t& iz, size_t& iw, size_t& ix1, size_t& iy1, size_t& iz1, size_t& iw1) const {
				assert(lower[0] >= 0 && lower[1] >= 0 && lower[2] >= 0);
				if constexpr (std::is_same_v<vectorN, ivectorN>) {
					xi = min(max(xi, lower), upper);
					ix = static_cast<size_t>(xi[0]);
					iy = static_cast<size_t>(xi[1]);
					iz = static_cast<size_t>(xi[2]);
					iw = static_cast<size_t>(xi[3]);
					xi = min(xi + 1, upper);
					ix1 = static_cast<size_t>(xi[0]);
					iy1 = static_cast<size_t>(xi[1]);
					iz1 = static_cast<size_t>(xi[2]);
					iw1 = static_cast<size_t>(xi[3]);
				} else {
					ix = xi[0] < lower[0] ? lower[0] : std::min(std::max(static_cast<size_t>(xi[0]), lower[0]), upper[0]);
					iy = xi[1] < lower[1] ? lower[1] : std::min(std::max(static_cast<size_t>(xi[1]), lower[1]), upper[1]);
					iz = xi[2] < lower[2] ? lower[2] : std::min(std::max(static_cast<size_t>(xi[2]), lower[2]), upper[2]);
					iw = xi[3] < lower[3] ? lower[3] : std::min(std::max(static_cast<size_t>(xi[3]), lower[3]), upper[3]);
					ix1 = std::min(ix + 1, upper[0]);
					iy1 = std::min(iy + 1, upper[1]);
					iz1 = std::min(iz + 1, upper[2]);
					iw1 = std::min(iw + 1, upper[3]);
				}
			}
		};

		template<typename ivectorN>
		struct bidirect_clamp_on_positive {
			const ivectorN& upper;

			/// requires 0 <= upper.
			explicit bidirect_clamp_on_positive(const ivectorN& arg0) : upper(arg0) {}

			inline void operator()(ivectorN& xip1, ivectorN& xi, ivectorN& xi1) const {
				xi = min(xi, upper);
				xi1 = min(xi + 1, upper);
				xip1 = max(xi - 1, 0);
			}

			template<typename vectorN, size_t dimension>
			inline void operator()(vectorN& xi, smdsize_t<dimension>& xip1, smdsize_t<dimension>& xi0, smdsize_t<dimension>& xi1) const {
				//assert(xi[i] >= 0);
				if constexpr (std::is_same_v<vectorN, ivectorN>) {
					xi = min(xi, upper);
					for (size_t i = 0; i != dimension; ++i)
						xi0[i] = static_cast<size_t>(xi[i]);
					vectorN tmp = max(xi - 1, 0);
					xi = min(xi + 1, upper);
					for (size_t i = 0; i != dimension; ++i)
						xi1[i] = static_cast<size_t>(xi[i]);
					for (size_t i = 0; i != dimension; ++i)
						xip1[i] = static_cast<size_t>(tmp[i]);
				} else {
					for (size_t i = 0; i != dimension; ++i)
						xi0[i] = std::min(static_cast<size_t>(xi[i]), upper[i]);
					for (size_t i = 0; i != dimension; ++i)
						xi1[i] = std::min(xi0[i] + 1, upper[i]);
					for (size_t i = 0; i != dimension; ++i)
						xip1[i] = xi0[i] < 0 ? size_t(0) : xi0[i] - 1;
				}
			}
		};

		template<typename ivectorN>
		struct period_on_positive {
			const ivectorN& bound;

			explicit period_on_positive(const ivectorN& arg0) : bound(arg0) {}

			inline void operator()(ivectorN& xi, ivectorN& xi1) const {
				xi = fmod(xi, bound);
				xi1 = fmod(xi + 1, bound);
			}

			template<typename vectorN>
			inline void operator()(vectorN& xi, size_t& ix, size_t& iy, size_t& ix1, size_t& iy1) const {
				assert(xi[0] >= 0 && xi[1] >= 0);
				if constexpr (std::is_same_v<vectorN, ivectorN>) {
					xi = fmod(xi, bound);
					ix = static_cast<size_t>(xi[0]);
					iy = static_cast<size_t>(xi[1]);
					xi = fmod(xi + 1, bound);
					ix1 = static_cast<size_t>(xi[0]);
					iy1 = static_cast<size_t>(xi[1]);
				} else {
					ix = static_cast<size_t>(xi[0]) % bound[0];
					iy = static_cast<size_t>(xi[1]) % bound[1];
					ix1 = (ix + 1) % bound[0];
					iy1 = (iy + 1) % bound[1];
				}
			}

			template<typename vectorN>
			inline void operator()(vectorN& xi, size_t& ix, size_t& iy, size_t& iz, size_t& ix1, size_t& iy1, size_t& iz1) const {
				assert(xi[0] >= 0 && xi[1] >= 0 && xi[2] >= 0);
				if constexpr (std::is_same_v<vectorN, ivectorN>) {
					xi = fmod(xi, bound);
					ix = static_cast<size_t>(xi[0]);
					iy = static_cast<size_t>(xi[1]);
					iz = static_cast<size_t>(xi[2]);
					xi = fmod(xi + 1, bound);
					ix1 = static_cast<size_t>(xi[0]);
					iy1 = static_cast<size_t>(xi[1]);
					iz1 = static_cast<size_t>(xi[2]);
				} else {
					ix = static_cast<size_t>(xi[0]) % bound[0];
					iy = static_cast<size_t>(xi[1]) % bound[1];
					iz = static_cast<size_t>(xi[2]) % bound[2];
					ix1 = (ix + 1) % bound[0];
					iy1 = (iy + 1) % bound[1];
					iz1 = (iz + 1) % bound[2];
				}
			}

			template<typename vectorN>
			inline void operator()(vectorN& xi, size_t& ix, size_t& iy, size_t& iz, size_t& iw, size_t& ix1, size_t& iy1, size_t& iz1, size_t& iw1) const {
				assert(xi[0] >= 0 && xi[1] >= 0 && xi[2] >= 0 && xi[3] >= 0);
				if constexpr (std::is_same_v<vectorN, ivectorN>) {
					xi = fmod(xi, bound);
					ix = static_cast<size_t>(xi[0]);
					iy = static_cast<size_t>(xi[1]);
					iz = static_cast<size_t>(xi[2]);
					iw = static_cast<size_t>(xi[3]);
					xi = fmod(xi + 1, bound);
					ix1 = static_cast<size_t>(xi[0]);
					iy1 = static_cast<size_t>(xi[1]);
					iz1 = static_cast<size_t>(xi[2]);
					iw1 = static_cast<size_t>(xi[3]);
				} else {
					ix = static_cast<size_t>(xi[0]) % bound[0];
					iy = static_cast<size_t>(xi[1]) % bound[1];
					iz = static_cast<size_t>(xi[2]) % bound[2];
					iw = static_cast<size_t>(xi[3]) % bound[3];
					ix1 = (ix + 1) % bound[0];
					iy1 = (iy + 1) % bound[1];
					iz1 = (iz + 1) % bound[2];
					iw1 = (iw + 1) % bound[3];
				}
			}
		};

		template<typename ivectorN>
		struct repeat_on_positive : public period_on_positive<ivectorN> {
			explicit repeat_on_positive(const ivectorN& arg0) : period_on_positive<ivectorN>(arg0) {}
		};

		template<typename ivectorN>
		struct bidirect_period_on_positive {
			const ivectorN& bound;

			explicit bidirect_period_on_positive(const ivectorN& arg0) : bound(arg0) {}

			inline void operator()(ivectorN& xip1, ivectorN& xi, ivectorN& xi1) const {
				xi = fmod(xi, bound);
				xi1 = fmod(xi + 1, bound);
				xip1 = fmod(xi - 1 + bound, bound);
			}

			template<typename vectorN, size_t dimension>
			inline void operator()(vectorN& xi, smdsize_t<dimension>& xip1, smdsize_t<dimension>& xi0, smdsize_t<dimension>& xi1) const {
				//assert(xi[i] >= 0);
				if constexpr (std::is_same_v<vectorN, ivectorN>) {
					xi = fmod(xi, bound);
					for (size_t i = 0; i != dimension; ++i)
						xi0[i] = static_cast<size_t>(xi[i]);
					vectorN tmp = fmod(xi - 1 + bound, bound);
					xi = fmod(xi + 1, bound);
					for (size_t i = 0; i != dimension; ++i)
						xi1[i] = static_cast<size_t>(xi[i]);
					for (size_t i = 0; i != dimension; ++i)
						xip1[i] = static_cast<size_t>(tmp[i]);
				} else {
					for (size_t i = 0; i != dimension; ++i)
						xi0[i] = static_cast<size_t>(xi[i]) % bound[i];
					for (size_t i = 0; i != dimension; ++i)
						xi1[i] = (xi0[i] + 1) % bound[i];
					for (size_t i = 0; i != dimension; ++i)
						xip1[i] = xi0[i] == 0 ? (bound[i] - 1) : xi0[i] - 1;
				}
			}
		};

		template<typename ivectorN>
		struct period {
			const ivectorN& bound;

			explicit period(const ivectorN& arg0) : bound(arg0) {}

			inline void operator()(ivectorN& xi, ivectorN& xi1) const {
				ivectorN direction;
				for(size_t i = 0; i != static_array_size<ivectorN>; ++i)
					direction[i] = xi[i]<0?-1:+1;
				xi = fmod(abs(xi), bound);
				xi = fmod(xi + direction, bound);
			}

			template<typename vectorN>
			inline void operator()(vectorN& xi, size_t& ix, size_t& iy, size_t& ix1, size_t& iy1) const {
				if constexpr (std::is_same_v<vectorN, ivectorN>) {
					vectorN direction = vectorN(xi[0]<0?-1:+1, xi[1]<0?-1:+1);
					xi = fmod(abs(xi), bound);
					ix = static_cast<size_t>(xi[0]);
					iy = static_cast<size_t>(xi[1]);
					xi = fmod(xi + direction, bound);
					ix1 = static_cast<size_t>(xi[0]);
					iy1 = static_cast<size_t>(xi[1]);
				} else {
					ix = static_cast<size_t>(abs(xi[0])) % bound[0];
					iy = static_cast<size_t>(abs(xi[1])) % bound[1];
					ix1 = xi[0] < 0 ? (ix - 1)%bound[0] : (ix + 1)%bound[0];
					iy1 = xi[1] < 0 ? (iy - 1)%bound[1] : (iy + 1)%bound[1];
				}
			}

			template<typename vectorN>
			inline void operator()(vectorN& xi, size_t& ix, size_t& iy, size_t& iz, size_t& ix1, size_t& iy1, size_t& iz1) const {
				if constexpr (std::is_same_v<vectorN, ivectorN>) {
					vectorN direction = vectorN(xi[0]<0?-1:+1, xi[1]<0?-1:+1, xi[2]<0?-1:+1);
					xi = fmod(abs(xi), bound);
					ix = static_cast<size_t>(xi[0]);
					iy = static_cast<size_t>(xi[1]);
					iz = static_cast<size_t>(xi[2]);
					xi = fmod(xi + direction, bound);
					ix1 = static_cast<size_t>(xi[0]);
					iy1 = static_cast<size_t>(xi[1]);
					iz1 = static_cast<size_t>(xi[2]);
				} else {
					ix = static_cast<size_t>(abs(xi[0])) % bound[0];
					iy = static_cast<size_t>(abs(xi[1])) % bound[1];
					iz = static_cast<size_t>(abs(xi[2])) % bound[2];
					ix1 = xi[0] < 0 ? (ix - 1)%bound[0] : (ix + 1)%bound[0];
					iy1 = xi[1] < 0 ? (iy - 1)%bound[1] : (iy + 1)%bound[1];
					iz1 = xi[2] < 0 ? (iz - 1)%bound[2] : (iz + 1)%bound[2];
				}
			}

			template<typename vectorN>
			inline void operator()(vectorN& xi, size_t& ix, size_t& iy, size_t& iz, size_t& iw, size_t& ix1, size_t& iy1, size_t& iz1, size_t& iw1) const {
				if constexpr (std::is_same_v<vectorN, ivectorN>) {
					vectorN direction = vectorN(xi[0]<0?-1:+1, xi[1]<0?-1:+1, xi[2]<0?-1:+1, xi[3]<0?-1:+1);
					xi = fmod(abs(xi), bound);
					ix = static_cast<size_t>(xi[0]);
					iy = static_cast<size_t>(xi[1]);
					iz = static_cast<size_t>(xi[2]);
					iw = static_cast<size_t>(xi[3]);
					xi = fmod(xi + direction, bound);
					ix1 = static_cast<size_t>(xi[0]);
					iy1 = static_cast<size_t>(xi[1]);
					iz1 = static_cast<size_t>(xi[2]);
					iw1 = static_cast<size_t>(xi[3]);
				} else {
					ix = static_cast<size_t>(abs(xi[0])) % bound[0];
					iy = static_cast<size_t>(abs(xi[1])) % bound[1];
					iz = static_cast<size_t>(abs(xi[2])) % bound[2];
					iw = static_cast<size_t>(abs(xi[3])) % bound[3];
					ix1 = xi[0] < 0 ? (ix - 1)%bound[0] : (ix + 1)%bound[0];
					iy1 = xi[1] < 0 ? (iy - 1)%bound[1] : (iy + 1)%bound[1];
					iz1 = xi[2] < 0 ? (iz - 1)%bound[2] : (iz + 1)%bound[2];
					iw1 = xi[3] < 0 ? (iw - 1)%bound[3] : (iw + 1)%bound[3];
				}
			}
		};

		template<typename ivectorN>
		struct repeat {
			const ivectorN& bound;

			explicit repeat(const ivectorN& arg0) : bound(arg0) {}

			inline void operator()(ivectorN& xi, ivectorN& xi1) const {
				xi = fmod(fmod(xi, bound) + bound, bound);
				xi1 = fmod(xi + 1, bound);
			}

			template<typename vectorN>
			inline void operator()(vectorN& xi, size_t& ix, size_t& iy, size_t& ix1, size_t& iy1) const {
				if constexpr (std::is_same_v<vectorN, ivectorN>) {
					xi = fmod(fmod(xi, bound) + bound, bound);
					ix = static_cast<size_t>(xi[0]);
					iy = static_cast<size_t>(xi[1]);
					xi = fmod(xi + 1, bound);
					ix1 = static_cast<size_t>(xi[0]);
					iy1 = static_cast<size_t>(xi[1]);
				} else {
					ix = xi[0] >= 0 ? (static_cast<size_t>(xi[0]) % bound[0]) : (bound[0] - (static_cast<size_t>(-xi[0]) % bound[0])) % bound[0];
					iy = xi[1] >= 0 ? (static_cast<size_t>(xi[1]) % bound[1]) : (bound[1] - (static_cast<size_t>(-xi[1]) % bound[1])) % bound[1];
					ix1 = (ix + 1) % bound[0];
					iy1 = (iy + 1) % bound[1];
				}
			}

			template<typename vectorN>
			inline void operator()(vectorN& xi, size_t& ix, size_t& iy, size_t& iz, size_t& ix1, size_t& iy1, size_t& iz1) const {
				if constexpr (std::is_same_v<vectorN, ivectorN>) {
					xi = fmod(fmod(xi, bound) + bound, bound);
					ix = static_cast<size_t>(xi[0]);
					iy = static_cast<size_t>(xi[1]);
					iz = static_cast<size_t>(xi[2]);
					xi = fmod(xi + 1, bound);
					ix1 = static_cast<size_t>(xi[0]);
					iy1 = static_cast<size_t>(xi[1]);
					iz1 = static_cast<size_t>(xi[2]);
				} else {
					ix = xi[0] >= 0 ? (static_cast<size_t>(xi[0]) % bound[0]) : (bound[0] - (static_cast<size_t>(-xi[0]) % bound[0])) % bound[0];
					iy = xi[1] >= 0 ? (static_cast<size_t>(xi[1]) % bound[1]) : (bound[1] - (static_cast<size_t>(-xi[1]) % bound[1])) % bound[1];
					iz = xi[2] >= 0 ? (static_cast<size_t>(xi[2]) % bound[2]) : (bound[2] - (static_cast<size_t>(-xi[2]) % bound[2])) % bound[2];
					ix1 = (ix + 1) % bound[0];
					iy1 = (iy + 1) % bound[1];
					iz1 = (iz + 1) % bound[2];
				}
			}

			template<typename vectorN>
			inline void operator()(vectorN& xi, size_t& ix, size_t& iy, size_t& iz, size_t& iw, size_t& ix1, size_t& iy1, size_t& iz1, size_t& iw1) const {
				if constexpr (std::is_same_v<vectorN, ivectorN>) {
					xi = fmod(fmod(xi, bound) + bound, bound);
					ix = static_cast<size_t>(xi[0]);
					iy = static_cast<size_t>(xi[1]);
					iz = static_cast<size_t>(xi[2]);
					iw = static_cast<size_t>(xi[3]);
					xi = fmod(xi + 1, bound);
					ix1 = static_cast<size_t>(xi[0]);
					iy1 = static_cast<size_t>(xi[1]);
					iz1 = static_cast<size_t>(xi[2]);
					iw1 = static_cast<size_t>(xi[3]);
				} else {
					ix = xi[0] >= 0 ? (static_cast<size_t>(xi[0]) % bound[0]) : (bound[0] - (static_cast<size_t>(-xi[0]) % bound[0])) % bound[0];
					iy = xi[1] >= 0 ? (static_cast<size_t>(xi[1]) % bound[1]) : (bound[1] - (static_cast<size_t>(-xi[1]) % bound[1])) % bound[1];
					iz = xi[2] >= 0 ? (static_cast<size_t>(xi[2]) % bound[2]) : (bound[2] - (static_cast<size_t>(-xi[2]) % bound[2])) % bound[2];
					iw = xi[3] >= 0 ? (static_cast<size_t>(xi[3]) % bound[3]) : (bound[3] - (static_cast<size_t>(-xi[3]) % bound[3])) % bound[3];
					ix1 = (ix + 1) % bound[0];
					iy1 = (iy + 1) % bound[1];
					iz1 = (iz + 1) % bound[2];
					iw1 = (iw + 1) % bound[3];
				}
			}
		};
	};

	struct sample_lattice2d {
		template<typename lattice2d, typename vector2, typename samplemode, typename intpfunc, typename smoothstepfunc>
		inline auto operator()(const lattice2d& F, const vector2& xi, const vector2& xf, const samplemode& mode, 
			const intpfunc& intp, const smoothstepfunc& smoothstep) const {
			vector2 i0 = xi;
			vector2 w = smoothstep(xf);
			if constexpr (std::invocable<lattice2d, vector2>) {
				vector2 i1;
				mode(i0, i1);
				return intp( 
					intp(F(i0), F(vector2{i1[0],i0[1]}), w[0]),
					intp(F(vector2{i0[0],i1[1]}), F(i1), w[0]), 
					w[1]
					);
			} else {
				smdsize_t<2> I0, i1;
				mode(i0, I0,i1);
				return intp( 
					intp(F[I0], F[smdsize_t<2>{i1[0],I0[1]}], w[0]),
					intp(F[smdsize_t<2>{I0[0],i1[1]}], F[i1], w[0]), 
					w[1]
					);
			}
		}

		template<typename lattice2d, typename vector2, typename samplemode, typename intpfunc, typename smoothstepfunc>
		inline auto operator()(const lattice2d& F, const vector2& x, const samplemode& mode, 
			const intpfunc& intp, const smoothstepfunc& smoothstep) const {
			vector2 i0 = vfloor(x);
			vector2 w = smoothstep(x - i0);
			if constexpr (std::invocable<lattice2d, vector2>) {
				vector2 i1;
				mode(i0, i1);
				return intp( 
					intp(F(i0), F(vector2{i1[0],i0[1]}), w[0]),
					intp(F(vector2{i0[0],i1[1]}), F(i1), w[0]),
					w[1]
					);
			} else {
				smdsize_t<2> I0, i1;
				mode(i0, I0,i1);
				return intp( 
					intp(F[I0], F[smdsize_t<2>{i1[0],I0[1]}], w[0]),
					intp(F[smdsize_t<2>{I0[0],i1[1]}], F[i1], w[0]),
					w[1]
					);
				/*size_t i0x,i0y, i1x,i1y;
				mode(i0, i0x,i0y, i1x,i1y);
				return intp( 
					intp(F[smdsize_t<2>{i0x,i0y}], F[smdsize_t<2>{i1x,i0y}], w[0]),
					intp(F[smdsize_t<2>{i0x,i1y}], F[smdsize_t<2>{i1x,i1y}], w[0]), w[1] );*/
			}
		}

		template<typename lattice2d, typename vector2, typename samplemode>
		inline auto operator()(const lattice2d& F, const vector2& x, const samplemode& mode) const {
			return (*this)(F, x, mode, math::lerp, math::smoothstep0);
		}
	};

	struct sample_lattice2d_reduce {
		template<typename lattice2d, typename vector2, typename samplemode, typename reducefunc>
		inline auto operator()(const lattice2d& F, const vector2& x, const samplemode& mode, const reducefunc& reduce) const {
			vector2 i0 = floor(x);
			if constexpr (std::invocable<lattice2d, vector2>) {
				vector2 i1;
				mode(i0, i1);
				return reduce(
					reduce(F(i0), F(vector2{i1[0],i0[1]})),
					reduce(F(vector2{i0[0],i1[1]}), F(i1)) );
			} else {
				smdsize_t<2> I0, i1;
				mode(i0, I0,i1);
				return reduce(
					reduce(F[I0],F[smdsize_t<2>{i1[0],I0[1]}]),
					reduce(F[smdsize_t<2>{I0[0],i1[1]}],F[i1]) );
			}
		}
	};

	struct sample_lattice3d {
		template<typename lattice3d, typename vector3, typename samplemode, typename intpfunc, typename smoothstepfunc>
		inline auto operator()(const lattice3d& F, const vector3& x, const samplemode& mode, 
			const intpfunc& intp, const smoothstepfunc& smoothstep) const {
			vector3 i0 = vfloor(x);
			vector3 w = smoothstep(x - i0);
			if constexpr (std::invocable<lattice3d, vector3>) {
				vector3 i1;
				mode(i0, i1);
				return intp( 
					intp(
						intp(F(i0),                         F(vector3{i1[0],i0[1],i0[2]}), w[0]),
						intp(F(vector3{i0[0],i1[1],i0[2]}), F(vector3{i1[0],i1[1],i0[2]}), w[0]),
						w[1]
						),
					intp(
						intp(F(vector3{i0[0],i0[1],i1[2]}), F(vector3{i1[0],i0[1],i1[2]}), w[0]),
						intp(F(vector3{i0[0],i1[1],i1[2]}), F(i1),                         w[0]),
						w[1]
						),
					w[2]
					);
			} else {
				smdsize_t<3> I0,i1;
				mode(i0, I0,i1);
				return intp(
					intp(
						intp(F[I0],                             F[smdsize_t<3>{i1[0],I0[1],I0[2]}], w[0]),
						intp(F[smdsize_t<3>{I0[0],i1[1],I0[2]}], F[smdsize_t<3>{i1[0],i1[1],I0[2]}], w[0]),
						w[1]
						),
					intp(
						intp(F[smdsize_t<3>{I0[0],I0[1],i1[2]}], F[smdsize_t<3>{i1[0],I0[1],i1[2]}], w[0]),
						intp(F[smdsize_t<3>{I0[0],i1[1],i1[2]}], F[i1],                             w[0]),
						w[1]
						),
					w[2]
					);
			}
		}

		template<typename lattice3d, typename vector3, typename samplemode>
		inline auto operator()(const lattice3d& F, const vector3& x, const samplemode& mode) const {
			return (*this)(F, x, mode, math::lerp, math::smoothstep0);
		}
	};

	struct sample_lattice4d {
		template<typename lattice4d, typename vector4, typename samplemode, typename intpfunc, typename smoothstepfunc>
		inline auto operator()(const lattice4d& F, const vector4& x, const samplemode& mode,
			const intpfunc& intp, const smoothstepfunc& smoothstep) const {
			vector4 i0 = vfloor(x);
			vector4 w = smoothstep(x - i0);
			if constexpr (std::invocable<lattice4d, vector4>) {
				vector4 i1;
				mode(i0, i1);
				return intp(
					intp(
						intp(
							intp(F(i0),                               F(vector4{i1[0],i0[1],i0[2],i0[3]}), w[0]),
							intp(F(vector4{i0[0],i1[1],i0[2],i0[3]}), F(vector4{i1[0],i1[1],i0[2],i0[3]}), w[0]), 
							w[1]
							),
						intp(
							intp(F(vector4{i0[0],i0[1],i1[2],i0[3]}), F(vector4{i1[0],i0[1],i1[2],i0[3]}), w[0]),
							intp(F(vector4{i0[0],i1[1],i1[2],i0[3]}), F(vector4{i1[0],i1[1],i1[2],i0[3]}), w[0]),
							w[1]
							),
						w[2]
						),
					intp(
						intp(
							intp(F(vector4{i0[0],i0[1],i0[2],i1[3]}), F(vector4{i1[0],i0[1],i0[2],i1[3]}), w[0]),
							intp(F(vector4{i0[0],i1[1],i0[2],i1[3]}), F(vector4{i1[0],i1[1],i0[2],i1[3]}), w[0]),
							w[1]
							),
						intp(
							intp(F(vector4{i0[0],i0[1],i1[2],i1[3]}), F(vector4{i1[0],i0[1],i1[2],i1[3]}), w[0]),
							intp(F(vector4{i0[0],i1[1],i1[2],i1[3]}), F(i1),                               w[0]),
							w[1]
							),
						w[2]
						),
					w[3]
					);
			} else {
				math::smdsize<int,4> I0, i1;
				mode(i0, I0,i1);

				auto i0x = I0[0]/* * data.stride(0)*/;
				auto i0y = I0[1] * F.stride(1);
				auto i0z = I0[2] * F.stride(2);
				auto i0w = I0[3] * F.stride(3);
				auto i1x = i1[0]/* * data.stride(0)*/;
				auto i1y = i1[1] * F.stride(1);
				auto i1z = i1[2] * F.stride(2);
				auto i1w = i1[3] * F.stride(3);

#if 0
				using size_type = typename lattice4d::size_type;
				return intp(
					intp(
						intp(
							intp(F[I0],                                 F[size_type{i1[0],I0[1],I0[2],I0[3]}], w[0]),
							intp(F[size_type{I0[0],i1[1],I0[2],I0[3]}], F[size_type{i1[0],i1[1],I0[2],I0[3]}], w[0]),
							w[1]
							),
						intp(
							intp(F[size_type{I0[0],I0[1],i1[2],I0[3]}], F[size_type{i1[0],I0[1],i1[2],I0[3]}], w[0]),
							intp(F[size_type{I0[0],i1[1],i1[2],I0[3]}], F[size_type{i1[0],i1[1],i1[2],I0[3]}], w[0]),
							w[1]
							),
						w[2]
						),
					intp(
						intp(
							intp(F[size_type{I0[0],I0[1],I0[2],i1[3]}], F[size_type{i1[0],I0[1],I0[2],i1[3]}], w[0]),
							intp(F[size_type{I0[0],i1[1],I0[2],i1[3]}], F[size_type{i1[0],i1[1],I0[2],i1[3]}], w[0]),
							w[1]
							),
						intp(
							intp(F[size_type{I0[0],I0[1],i1[2],i1[3]}], F[size_type{i1[0],I0[1],i1[2],i1[3]}], w[0]),
							intp(F[size_type{I0[0],i1[1],i1[2],i1[3]}], F[i1],                                 w[0]),
							w[1]
							),
						w[2]
						),
					w[3]
					);
#elif 1
				auto i0z_i0w     = i0z+i0w;
				auto i0y_i0z_i0w = i0y+i0z_i0w;
				auto i1y_i0z_i0w = i1y+i0z_i0w;
				auto result = intp(
					intp(F[i0x + i0y_i0z_i0w], F[i1x + i0y_i0z_i0w], w[0]),
					intp(F[i0x + i1y_i0z_i0w], F[i1x + i1y_i0z_i0w], w[0]),
					w[1]
					);

				auto i1z_i0w     = i1z+i0w;
				auto i0y_i1z_i0w = i0y+i1z_i0w;
				auto i1y_i1z_i0w = i1y+i1z_i0w;
				result = intp(
					result,
					intp(
						intp(F[i0x + i0y_i1z_i0w], F[i1x + i0y_i1z_i0w], w[0]),
						intp(F[i0x + i1y_i1z_i0w], F[i1x + i1y_i1z_i0w], w[0]),
						w[1]
						),
					w[2]
					);
		
				auto i0z_i1w     = i0z+i1w;
				auto i0y_i0z_i1w = i0y+i0z_i1w;
				auto i1y_i0z_i1w = i1y+i0z_i1w;
				auto result2 = intp(
					intp(F[i0x + i0y_i0z_i1w], F[i1x + i0y_i0z_i1w], w[0]),
					intp(F[i0x + i1y_i0z_i1w], F[i1x + i1y_i0z_i1w], w[0]),
					w[1]
					);

				auto i1z_i1w     = i1z+i1w;
				auto i0y_i1z_i1w = i0y+i1z_i1w;
				auto i1y_i1z_i1w = i1y+i1z_i1w;
				result2 = intp(
					result2,
					intp(
						intp(F[i0x + i0y_i1z_i1w], F[i1x + i0y_i1z_i1w], w[0]),
						intp(F[i0x + i1y_i1z_i1w], F[i1x + i1y_i1z_i1w], w[0]),
						w[1]
						),
					w[2]
					);

				return intp(result, result2, w[3]);
#elif 1
	///lerp(x0, x1, t)
	/// = x0*(1-t) + x1*t
	/// = conv({x0,x1}, {1-t,t})
	/// 
	///lerp({x00,x10}, {x01,x11}, tx, ty)
	///	= lerp( lerp(x00,x10,tx), 
	///	        lerp(x01,x11,tx), ty )
	///	= lerp( x00*(1-tx) + x10*tx, 
	///         x01*(1-tx) + x11*tx, ty )
	/// = ( x00*(1-tx) + x10*tx )*(1-ty) + ( x01*(1-tx) + x11*tx )*ty
	/// 
	///lerp({{x000,x100},{x010,x110}}, {{x001,x101},{x011,x111}}, tx, ty, tz)
	/// 
	/// = lerp( ( x000*(1-tx) + x100*tx )*(1-ty) + ( x010*(1-tx) + x110*tx )*ty,
	///         ( x001*(1-tx) + x101*tx )*(1-ty) + ( x011*(1-tx) + x111*tx )*ty, tz )
	/// 
	/// = ( ( x000*(1-tx) + x100*tx )*(1-ty) + ( x010*(1-tx) + x110*tx )*ty )*(1-tz) +
	///   ( ( x001*(1-tx) + x101*tx )*(1-ty) + ( x011*(1-tx) + x111*tx )*ty )*tz
	/// 
	/// 
	///lerp({{{x0000,x1000},{x0100,x1100}},{{x0010,x1010},{x0110,x1110}}},                (16+8+4+2) mul
	///     {{{x0001,x1001},{x0101,x1101}},{{x0011,x1011},{x0111,x1111}}}, 
	///     tx, ty, tz, tw)
	/// = lerp( ( ( x0000*(1-tx) + x1000*tx )*(1-ty) + ( x0100*(1-tx) + x1100*tx )*ty )*(1-tz) +
	///         ( ( x0010*(1-tx) + x1010*tx )*(1-ty) + ( x0110*(1-tx) + x1110*tx )*ty )*tz,
	///         ( ( x0001*(1-tx) + x1001*tx )*(1-ty) + ( x0101*(1-tx) + x1101*tx )*ty )*(1-tz) +
	///         ( ( x0011*(1-tx) + x1011*tx )*(1-ty) + ( x0111*(1-tx) + x1111*tx )*ty )*tz, tw )
	/// 
	/// = ( ( ( x0000*(1-tx) + x1000*tx )*(1-ty) + ( x0100*(1-tx) + x1100*tx )*ty )*(1-tz) +
	///     ( ( x0010*(1-tx) + x1010*tx )*(1-ty) + ( x0110*(1-tx) + x1110*tx )*ty )*tz )*(1-tw) + 
	///   ( ( ( x0001*(1-tx) + x1001*tx )*(1-ty) + ( x0101*(1-tx) + x1101*tx )*ty )*(1-tz) +
	///     ( ( x0011*(1-tx) + x1011*tx )*(1-ty) + ( x0111*(1-tx) + x1111*tx )*ty )*tz )*tw
	/// 
	/// (1 - t), t, 8 scalars,
	/// result 16 kinds,
	/// most case is 4 vectors by 4 kinds is 16 
	/// 
	const auto& _t = w;
	const auto  nt = (1 - _t);

	/// t0,t1,t2,t3  n0,t1,t2,t3    t0t1,t2,t3  n0t1,t2,t3 <--       t0t1,t2,t3  n0t1,t2,t3
	/// t0,n1,t2,t3  n0,n1,t2,t3    t0n1,t2,t3  n0n1,t2,t3 <--       t0t2,n1,t3  n0t2,n1,t3
	/// t0,t1,n2,t3  n0,t1,n2,t3    t0t1,n2,t3  n0t1,n2,t3           t0n2,t1,t3  n0n2,t1,t3
	/// t0,n1,n2,t3  n0,n1,n2,t3    t0n1,n2,t3  n0n1,n2,t3           t0n1,n2,t3  n0n1,n2,t3
	/// t0,t1,t2,n3  n0,t1,t2,n3    t0t1,t2,n3  n0t1,t2,n3           t0t1,t2,n3  n0t1,t2,n3
	/// t0,n1,t2,n3  n0,n1,t2,n3    t0n1,t2,n3  n0n1,t2,n3           t0t2,n1,n3  n0t2,n1,n3
	/// t0,t1,n2,n3  n0,t1,n2,n3    t0t1,n2,n3  n0t1,n2,n3           t0n2,t1,n3  n0n2,t1,n3
	/// t0,n1,n2,n3  n0,n1,n2,n3    t0n1,n2,n3  n0n1,n2,n3           t0n1,n2,n3  n0n1,n2,n3
	///                               {t0,t0,n0,n0}*{t1,n1,t1,n1}      A={t0,t0,n0,n0}*{t1,t2,t1,t2}
	///                                the right cannot shuffle.       B={t0,t0,n0,n0}*{n2,n1,n2,n1}
	/// 
	/// A0,t2,t3  A2,t2,t3                A0t2,t3  A2t2,t3
	/// A1,n1,t3  A3,n1,t3                A1n1,t3  A3n1,t3
	/// B0,t1,t3  B2,t1,t3                B0t1,t3  B2t1,t3
	/// B1,n2,t3  B3,n2,t3                B1n2,t3  B3n2,t3
	/// A0,t2,n3  A2,t2,n3                A0t2,n3  A2t2,n3
	/// A1,n1,n3  A3,n1,n3                A1n1,n3  A3n1,n3
	/// B0,t1,n3  B2,t1,n3                B0t1,n3  B2t1,n3
	/// B1,n2,n3  B3,n2,n3                B1n2,n3  B3n2,n3
	///   A={t0,t0,n0,n0}*{t1,t2,t1,t2}     A={t0,t0,n0,n0}*{t1,t2,t1,t2}
	///   B={t0,t0,n0,n0}*{n2,n1,n2,n1}     B={t0,t0,n0,n0}*{n2,n1,n2,n1}
	///                                     C={A0,A2,A1,A3}*{t2,t2,n1,n1}
	///                                     D={B0,B2,B1,B3}*{t1,t1,n2,n2}
	/// 
	/// C0,t3  C1,t3
	/// C2,t3  C3,t3
	/// D0,t3  D1,t3
	/// D2,t3  D3,t3
	/// C0,n3  C1,n3
	/// C2,n3  C3,n3
	/// D0,n3  D1,n3
	/// D2,n3  D3,n3
	///   A={t0,t0,n0,n0}*{t1,t2,t1,t2}
	///   B={t0,t0,n0,n0}*{n2,n1,n2,n1}
	///   C={A0,A2,A1,A3}*{t2,t2,n1,n1}
	///   D={B0,B2,B1,B3}*{t1,t1,n2,n2}
	/// 
	/*const auto tmp = vector4{nt[0],nt[0],_t[0],_t[0]};
	const auto A = tmp * vector4{nt[1],nt[2],nt[1],nt[2]};
	const auto B = tmp * vector4{_t[2],_t[1],_t[2],_t[1]};
	const auto C = vector4{A[0],A[2],A[1],A[3]} * vector4{nt[2],nt[2],_t[1],_t[1]};
	const auto D = vector4{B[0],B[2],B[1],B[3]} * vector4{nt[1],nt[1],_t[2],_t[2]};*/
	const auto tmp = _mm_shuffle_ps(nt.__data[0], _t.__data[0], _MM_SHUFFLE(0,0,0,0));
	const auto A = _mm_mul_ps(tmp, _mm_shuffle_ps(nt.__data[0], nt.__data[0], _MM_SHUFFLE(2,1,2,1)));
	const auto B = _mm_mul_ps(tmp, _mm_shuffle_ps(_t.__data[0], _t.__data[0], _MM_SHUFFLE(1,2,1,2)));
	const auto C = vector4{ _mm_mul_ps(_mm_shuffle_ps(A,A, _MM_SHUFFLE(3,1,2,0)), _mm_shuffle_ps(nt.__data[0], _t.__data[0], _MM_SHUFFLE(1,1,2,2))) };
	const auto D = vector4{ _mm_mul_ps(_mm_shuffle_ps(B,B, _MM_SHUFFLE(3,1,2,0)), _mm_shuffle_ps(nt.__data[0], _t.__data[0], _MM_SHUFFLE(2,2,1,1))) };
	
	//__m128 wX = _mm_mul_ps(C, _mm_set1_ps(nt[3]));
	auto wX = C*nt[3];
	auto i0z_i0w     = i0z+i0w;
	auto i0y_i0z_i0w = i0y+i0z_i0w;
	auto i1y_i0z_i0w = i1y+i0z_i0w;
	/*[0.0, 6.5)*/auto   vx = F[i0x + i0y_i0z_i0w];
	/*[0.0, 7.0)*/auto   vy = F[i1x + i0y_i0z_i0w];
	///*[0.0, 1.3)*/__m128 wx = _mm_shuffle_ps(wX, wX, _MM_SHUFFLE(0,0,0,0));
	///*[0.0, 1.6)*/__m128 wy = _mm_shuffle_ps(wX, wX, _MM_SHUFFLE(1,1,1,1));
	/*[0.0, 7.5)*/auto   vz = F[i0x + i1y_i0z_i0w];
	/*[0.0, 8.0)*/auto   vw = F[i1x + i1y_i0z_i0w];
	///*[0.0, 2.0)*/__m128 wz = _mm_shuffle_ps(wX, wX, _MM_SHUFFLE(2,2,2,2));
	///*[0.0, 2.3)*/__m128 ww = _mm_shuffle_ps(wX, wX, _MM_SHUFFLE(3,3,3,3));
	/*[6.5, ...)*/auto result = vx*wX[0];
	/*[7.0, ...)*/auto result1 = vy*wX[1];
	/*[7.5, ...)*/auto result2 = vz*wX[2];
	/*[8.0, ...)*/auto result3 = vw*wX[3];

		//wX = _mm_mul_ps(D, _mm_set1_ps(nt[3]));
	wX = D*nt[3];
	auto i1z_i0w     = i1z+i0w;
	auto i0y_i1z_i0w = i0y+i1z_i0w;
	auto i1y_i1z_i0w = i1y+i1z_i0w;
		vx = F[i0x + i0y_i1z_i0w];
		vy = F[i1x + i0y_i1z_i0w];
		/*wx = _mm_shuffle_ps(wX, wX, _MM_SHUFFLE(0,0,0,0));
		wy = _mm_shuffle_ps(wX, wX, _MM_SHUFFLE(1,1,1,1));*/
		vz = F[i0x + i1y_i1z_i0w];
		vw = F[i1x + i1y_i1z_i0w];
		/*wz = _mm_shuffle_ps(wX, wX, _MM_SHUFFLE(2,2,2,2));
		ww = _mm_shuffle_ps(wX, wX, _MM_SHUFFLE(3,3,3,3));*/
	result += vx*wX[0];
	result1 += vy*wX[1];
	result2 += vz*wX[2];
	result3 += vw*wX[3];

		//wX = _mm_mul_ps(C, _mm_set1_ps(_t[3]));
	wX = C*_t[3];
	auto i0z_i1w     = i0z+i1w;
	auto i0y_i0z_i1w = i0y+i0z_i1w;
	auto i1y_i0z_i1w = i1y+i0z_i1w;
		vx = F[i0x + i0y_i0z_i1w];
		vy = F[i1x + i0y_i0z_i1w];
		/*wx = _mm_shuffle_ps(wX, wX, _MM_SHUFFLE(0,0,0,0));
		wy = _mm_shuffle_ps(wX, wX, _MM_SHUFFLE(1,1,1,1));*/
		vz = F[i0x + i1y_i0z_i1w];
		vw = F[i1x + i1y_i0z_i1w];
		/*wz = _mm_shuffle_ps(wX, wX, _MM_SHUFFLE(2,2,2,2));
		ww = _mm_shuffle_ps(wX, wX, _MM_SHUFFLE(3,3,3,3));*/
	result += vx*wX[0];
	result1 += vy*wX[1];
	result2 += vz*wX[2];
	result3 += vw*wX[3];

		//wX = _mm_mul_ps(D, _mm_set1_ps(_t[3]));
	wX = D*_t[3];
	auto i1z_i1w     = i1z+i1w;
	auto i0y_i1z_i1w = i0y+i1z_i1w;
	auto i1y_i1z_i1w = i1y+i1z_i1w;
		vx = F[i0x + i0y_i1z_i1w];
		vy = F[i1x + i0y_i1z_i1w];
		/*wx = _mm_shuffle_ps(wX, wX, _MM_SHUFFLE(0,0,0,0));
		wy = _mm_shuffle_ps(wX, wX, _MM_SHUFFLE(1,1,1,1));*/
		vz = F[i0x + i1y_i1z_i1w];
		vw = F[i1x + i1y_i1z_i1w];
		/*wz = _mm_shuffle_ps(wX, wX, _MM_SHUFFLE(2,2,2,2));
		ww = _mm_shuffle_ps(wX, wX, _MM_SHUFFLE(3,3,3,3));*/
	result += vx*wX[0];
	result1 += vy*wX[1];
	result2 += vz*wX[2];
	result3 += vw*wX[3];

	return std::move(result+result1+result2+result3);
	/* very fast!!!!, but too many shuffles for !std::same_as<value_t<vector4>,result_t<lattice4d>> */
#endif
			}
		}

		template<typename lattice4d, typename vector4, typename samplemode>
		inline auto operator()(const lattice4d& F, const vector4& x, const samplemode& mode) const {
			return (*this)(F, x, mode, math::lerp, math::smoothstep0);
		}
	};

	constexpr sample_lattice2d sample_lattice2;
	constexpr sample_lattice3d sample_lattice3;
	constexpr sample_lattice4d sample_lattice4;
	constexpr sample_lattice2d_reduce reduce_lattice2;
#endif

#if 0
	enum class samplmodes {
		normal = 0,
		positive = 1,
		clamp = 2,
		repeat = 4,
		period = 8,
		clamp_posi = 2 | positive,
		repeat_posi = 4 | positive,
		period_posi = 8 | positive
	};
	constexpr int operator&(samplmodes a, samplmodes b) { return static_cast<int>(a) & static_cast<int>(b); }

	template<samplmodes mode, typename vector4, typename ivector4>
	inline void samplindex4(vector4& xi, const ivector4& bound/* is edge if mode&samplmodes::clamp */,
		size_t& ix, size_t& iy, size_t& iz, size_t& iw, size_t& ix1, size_t& iy1, size_t& iz1, size_t& iw1) {
		if constexpr (std::is_same_v<ivector4, vector4>) 
		{/// more speed. but multi_array::edge is not same vector4.
			if constexpr (mode == samplmodes::clamp_posi) {
				assert(xi[0] >= 0 && xi[1] >= 0 && xi[2] >= 0 && xi[3] >= 0);
				xi = min(xi, bound);
			} else if constexpr (mode == samplmodes::clamp) {
				xi = min(max(xi, 0), bound);
			} else if constexpr (mode == samplmodes::period_posi) {
				assert(xi[0] >= 0 && xi[1] >= 0 && xi[2] >= 0 && xi[3] >= 0);
				xi = fmod(xi, bound);
			} else if constexpr (mode == samplmodes::period) {
				xi = fmod(abs(xi), bound);
			} else if constexpr (mode == samplmodes::repeat_posi) {
				assert(xi[0] >= 0 && xi[1] >= 0 && xi[2] >= 0 && xi[3] >= 0);
				xi = fmod(xi, bound);
			} else if constexpr (mode == samplmodes::repeat) {
				xi = fmod(fmod(xi, bound) + bound, bound);
			} else {
				// do nothing.
			}
			ix = static_cast<size_t>(xi[0]);
			iy = static_cast<size_t>(xi[1]);
			iz = static_cast<size_t>(xi[2]);
			iw = static_cast<size_t>(xi[3]);

			if constexpr (mode == samplmodes::clamp_posi || mode == samplmodes::clamp) {
				xi = min(xi + 1, bound);
			} else if constexpr (mode == samplmodes::period_posi || mode == samplmodes::period
				|| mode == samplmodes::repeat_posi || mode == samplmodes::repeat) {
				xi = fmod(xi + 1, bound);
			} else {
				xi = xi + 1;
			}
			ix1 = static_cast<size_t>(xi[0]);
			iy1 = static_cast<size_t>(xi[1]);
			iz1 = static_cast<size_t>(xi[2]);
			iw1 = static_cast<size_t>(xi[3]);
		} 
		else
		{/// slight slower when simdlen != 1, avoid meaningless conversion. (the conversion is expensive higher)
			if constexpr (mode == samplmodes::clamp_posi) {
				assert(xi[0] >= 0 && xi[1] >= 0 && xi[2] >= 0 && xi[3] >= 0);
				ix = std::min(static_cast<size_t>(xi[0]), bound[0]);
				iy = std::min(static_cast<size_t>(xi[1]), bound[1]);
				iz = std::min(static_cast<size_t>(xi[2]), bound[2]);
				iw = std::min(static_cast<size_t>(xi[3]), bound[3]);
			} else if constexpr (mode == samplmodes::clamp) {
				if (xi[0] < 0) { ix = 0; } else { ix = std::min(static_cast<size_t>(xi[0]), bound[0]); }
				if (xi[1] < 0) { iy = 0; } else { iy = std::min(static_cast<size_t>(xi[1]), bound[1]); }
				if (xi[2] < 0) { iz = 0; } else { iz = std::min(static_cast<size_t>(xi[2]), bound[2]); }
				if (xi[3] < 0) { iw = 0; } else { iw = std::min(static_cast<size_t>(xi[3]), bound[3]); }
			} else if constexpr (mode == samplmodes::period_posi || mode == samplmodes::repeat_posi) {
				assert(xi[0] >= 0 && xi[1] >= 0 && xi[2] >= 0 && xi[3] >= 0);
				ix = static_cast<size_t>(xi[0]) % bound[0];
				iy = static_cast<size_t>(xi[1]) % bound[1];
				iz = static_cast<size_t>(xi[2]) % bound[2];
				iw = static_cast<size_t>(xi[3]) % bound[3];
			} else if constexpr (mode == samplmodes::period) {
				xi = abs(xi);
				ix = static_cast<size_t>(xi[0]) % bound[0];
				iy = static_cast<size_t>(xi[1]) % bound[1];
				iz = static_cast<size_t>(xi[2]) % bound[2];
				iw = static_cast<size_t>(xi[3]) % bound[3];
			} else if constexpr (mode == samplmodes::repeat) {
				ix = xi[0] >= 0 ? (static_cast<size_t>(xi[0]) % bound[0]) : (bound[0] - (static_cast<size_t>(-xi[0]) % bound[0])) % bound[0];
				iy = xi[1] >= 0 ? (static_cast<size_t>(xi[1]) % bound[1]) : (bound[1] - (static_cast<size_t>(-xi[1]) % bound[1])) % bound[1];
				iz = xi[2] >= 0 ? (static_cast<size_t>(xi[2]) % bound[2]) : (bound[2] - (static_cast<size_t>(-xi[2]) % bound[2])) % bound[2];
				iw = xi[3] >= 0 ? (static_cast<size_t>(xi[3]) % bound[3]) : (bound[3] - (static_cast<size_t>(-xi[3]) % bound[3])) % bound[3];
			} else {
				ix = static_cast<size_t>(xi[0]);
				iy = static_cast<size_t>(xi[1]);
				iz = static_cast<size_t>(xi[2]);
				iw = static_cast<size_t>(xi[3]);
			}

			if constexpr (mode == samplmodes::clamp_posi || mode == samplmodes::clamp) {
				ix1 = std::min(ix + 1, bound[0]);
				iy1 = std::min(iy + 1, bound[1]);
				iz1 = std::min(iz + 1, bound[2]);
				iw1 = std::min(iw + 1, bound[3]);
			} else if constexpr (mode == samplmodes::period_posi || mode == samplmodes::period
				|| mode == samplmodes::repeat_posi || mode == samplmodes::repeat) {
				ix1 = (ix + 1) % bound[0];
				iy1 = (iy + 1) % bound[1];
				iz1 = (iz + 1) % bound[2];
				iw1 = (iw + 1) % bound[3];
			} else {
				ix1 = (ix + 1);
				iy1 = (iy + 1);
				iz1 = (iz + 1);
				iw1 = (iw + 1);
			}
		}
	}

	template<samplmodes mode, typename md4array, typename vector4, typename ivector4>
		requires requires(md4array F) { F[{size_t(),size_t(),size_t(),size_t()}]; }
	auto sample4(const md4array& F, const vector4& x, const ivector4& bound) {
	#if 0
		/// This method detects the sampling point'x' on the edge
		///  to avoid additional calculation and collision boundary.
		/// (in fact, this method don't process out of boundary only process collision boundary "only case: extents() - 1".)
		///		... all type-cast ...
		///		cmp
		///		jb
		///		cmp
		///		jb
		///		cmp
		///		jb
		///		... all get and arithmetic ...
		/// 
		/// Originally I thought this method was simplest, I don't process out of boundary to avoid many control-flows,
		///  but in fact, the sampling point'x' almost not on the edge therefor almost cannot reduced any calculation.
		/// So this method is "failed".
		const size_t ix = static_cast<size_t>(xi[0]),
			iy = static_cast<size_t>(xi[1]),
			iz = static_cast<size_t>(xi[2]),
			iw = static_cast<size_t>(xi[3]);
		auto sample = F[{ ix,iy,iz,iw }];
		if ( xf[0] != 0 ) {
			const size_t ix1 = ix + 1;
			sample = lerp(sample, F[{ ix1,iy,iz,iw }], xf[0]);
			if ( xf[1] != 0 ) {
				const size_t iy1 = iy + 1;
				sample = lerp(sample, lerp(F[{ ix,iy1,iz,iw }],F[{ ix1,iy1,iz,iw }],xf[0]), xf[1]);
				if ( xf[2] != 0 ) {
					const size_t iz1 = iz + 1;
					sample = lerp(sample, lerp(lerp(F[{ ix,iy,iz1,iw }],F[{ ix1,iy,iz1,iw }],xf[0]),lerp(F[{ ix,iy1,iz1,iw }],F[{ ix1,iy1,iz1,iw }],xf[0]),xf[1]), xf[2]);
					if ( xf[3] != 0 ) {
						const size_t iw1 = iw + 1;
						sample = lerp(sample, lerp(
							lerp(lerp(F[{ ix,iy,iz, iw1 }],F[{ ix1,iy,iz, iw1 }],xf[0]),lerp(F[{ ix,iy1,iz, iw1 }],F[{ ix1,iy1,iz, iw1 }],xf[0]),xf[1]),
							lerp(lerp(F[{ ix,iy,iz1,iw1 }],F[{ ix1,iy,iz1,iw1 }],xf[0]),lerp(F[{ ix,iy1,iz1,iw1 }],F[{ ix1,iy1,iz1,iw1 }],xf[0]),xf[1]), xf[2]), xf[3]);
					}
				} else {
					if ( xf[3] != 0 ) {
						const size_t iw1 = iw + 1;
						sample = lerp(sample, lerp(lerp(F[{ ix,iy,iz,iw1 }],F[{ ix1,iy,iz,iw1 }],xf[0]),lerp(F[{ ix,iy1,iz,iw1 }],F[{ ix1,iy1,iz,iw1 }],xf[0]),xf[1]), xf[3]);
					}
				}
			} else {
				if ( xf[2] != 0 ) {
					const size_t iz1 = iz + 1;
					sample = lerp(sample, lerp(F[{ ix,iy,iz1,iw }],F[{ ix1,iy,iz1,iw }],xf[0]), xf[2]);
					if ( xf[3] != 0 ) {
						const size_t iw1 = iw + 1;
						sample = lerp(sample, lerp(lerp(F[{ ix,iy,iz,iw1 }],F[{ ix1,iy,iz,iw1 }],xf[0]),lerp(F[{ ix,iy,iz1,iw1 }],F[{ ix1,iy,iz1,iw1 }],xf[0]),xf[2]), xf[3]);
					}
				} else {// X-W
					if ( xf[3] != 0 ) {
						const size_t iw1 = iw + 1;
						sample = lerp(sample, lerp(F[{ ix,iy,iz,iw1 }],F[{ ix1,iy,iz,iw1 }],xf[0]), xf[3]);
					}
				}
			}
		} else {

			if ( xf[1] != 0 ) {
				const size_t iy1 = iy + 1;
				sample = lerp(sample, F[{ ix,iy1,iz,iw }], xf[1]);
				if ( xf[2] != 0 ) {
					const size_t iz1 = iz + 1;
					sample = lerp(sample, lerp(F[{ ix,iy,iz1,iw }],F[{ ix,iy1,iz1,iw }],xf[1]), xf[2]);
					if ( xf[3] != 0 ) {
						const size_t iw1 = iw + 1;
						sample = lerp(sample, lerp(lerp(F[{ ix,iy,iz,iw1 }],F[{ ix,iy1,iz,iw1 }],xf[1]),lerp(F[{ ix,iy,iz1,iw1 }],F[{ ix,iy1,iz1,iw1 }],xf[1]),xf[2]), xf[3]);
					}
				} else {// Y-W
					if ( xf[3] != 0 ) {
						const size_t iw1 = iw + 1;
						sample = lerp(sample, lerp(F[{ ix,iy,iz,iw1 }],F[{ ix,iy1,iz,iw1 }],xf[1]), xf[3]);
					}
				}
			} else {
				if ( xf[2] != 0 ) {
					const size_t iz1 = iz + 1;
					sample = lerp(sample, F[{ ix,iy,iz1,iw }], xf[2]);
					if ( xf[3] != 0 ) {
						const size_t iw1 = iw + 1;
						sample = lerp(sample, lerp(F[{ ix,iy,iz,iw1 }],F[{ ix,iy,iz1,iw1 }],xf[2]), xf[3]);
					}
				} else {
					if ( xf[3] != 0 ) {
						sample = lerp(sample, F[{ ix,iy,iz,iw+1 }], xf[3]);
					}
				}
			}

		}

		return sample;
	#else
		/// First I try morden out of boundary, (each dimension can be processed differently)
		///  but very expensive, and not often use.
		/// I am doing one way for all dimension, is
		///		maxps xi,0
		///		minps xi,edge
		///		... half of all type-cast.
		///		addps xi,1
		///		minps xi,edge
		///		... half of all type-cast.
		///		... all get and arithmetic, but I can easy switch optimizetions.
		/// 
		/// Let's compare the previous methods, excluding type-cast, get and arithmetic.
		///		maxps xi,0        cmp xf[0],0
		///		minps xi,edge     jb
		///		addps xi,1        cmp xf[1],0
		///		minps xi,edge     jb
		///		                  cmp xf[2],0
		///		                  jb
		/// The new method seems to be more simpler.
		/// And new method can only "one arithmetic" for "all control-flow",
		///  therefor can be easy to speed up the artihmetic part. ( previous method almost not any speed up. )
		///
		vector4 xi = floor(x);
		vector4 xf = x - xi;
		size_t ix, iy, iz, iw,
			ix1, iy1, iz1, iw1;
		math::samplindex4<mode>(xi, bound, ix, iy, iz, iw, ix1, iy1, iz1, iw1);
		//std::cout << ix<<','<<iy<<','<<iz<<','<<iw<<' '<<ix1<<','<<iy1<<','<<iz1<<','<<iw1<<'\n'; return F[{0,0,0,0}];
		if constexpr (std::is_same_v<float, std::remove_cvref_t<decltype(F[{ix,iy,iz,iw}])>>) {
			__m128 wwww1 = _mm_setr_ps(F[{ix ,iy ,iz ,iw }],F[{ix1,iy ,iz ,iw }],F[{ix ,iy1,iz ,iw }],F[{ix1,iy1,iz ,iw }]), wwww2 = _mm_setr_ps(F[{ix ,iy ,iz1,iw }],F[{ix1,iy ,iz1,iw }],F[{ix ,iy1,iz1,iw }],F[{ix1,iy1,iz1,iw }]);
			__m128 wwww3 = _mm_setr_ps(F[{ix ,iy ,iz ,iw1}],F[{ix1,iy ,iz ,iw1}],F[{ix ,iy1,iz ,iw1}],F[{ix1,iy1,iz ,iw1}]), wwww4 = _mm_setr_ps(F[{ix ,iy ,iz1,iw1}],F[{ix1,iy ,iz1,iw1}],F[{ix ,iy1,iz1,iw1}],F[{ix1,iy1,iz1,iw1}]);
			auto _mm_lerp_ps = [](const __m128& a, const __m128& b, const float& t) { return _mm_fmadd_ps( _mm_sub_ps(b, a), _mm_set1_ps(t), a ); };
			__m128 zw = _mm_lerp_ps(_mm_lerp_ps(wwww1,wwww3,xf[3]), _mm_lerp_ps(wwww2,wwww4,xf[3]), xf[2]);
			__m128 yzw = _mm_lerp_ps(_mm_shuffle_ps(zw,zw,_MM_SHUFFLE(0,0,1,0)), _mm_shuffle_ps(zw,zw,_MM_SHUFFLE(0,0,3,2)), xf[1]);
			return lerp(yzw.m128_f32[0], yzw.m128_f32[1], xf[0]);
		} else {
			/*std::cout << F[{ix ,iy ,iz ,iw }]<< ',' <<F[{ix1,iy ,iz ,iw }] << ',' <<F[{ix ,iy1,iz ,iw }]<< ',' <<F[{ix1,iy1,iz ,iw }] << std::endl;
			std::cout << F[{ix ,iy ,iz1,iw }]<< ',' <<F[{ix1,iy ,iz1,iw }] << ',' <<F[{ix ,iy1,iz1,iw }]<< ',' <<F[{ix1,iy1,iz1,iw }] << std::endl;
			std::cout << F[{ix ,iy ,iz ,iw1}]<< ',' <<F[{ix1,iy ,iz ,iw1}] << ',' <<F[{ix ,iy1,iz ,iw1}]<< ',' <<F[{ix1,iy1,iz ,iw1}] << std::endl;
			std::cout << F[{ix ,iy ,iz1,iw1}]<< ',' <<F[{ix1,iy ,iz1,iw1}] << ',' <<F[{ix ,iy1,iz1,iw1}]<< ',' <<F[{ix1,iy1,iz1,iw1}] << std::endl;*/
			return lerp(
				lerp( 
					lerp(lerp(F[{ix ,iy ,iz ,iw }],F[{ix1,iy ,iz ,iw }],xf[0]),lerp(F[{ix ,iy1,iz ,iw }],F[{ix1,iy1,iz ,iw }],xf[0]),xf[1]),
					lerp(lerp(F[{ix ,iy ,iz1,iw }],F[{ix1,iy ,iz1,iw }],xf[0]),lerp(F[{ix ,iy1,iz1,iw }],F[{ix1,iy1,iz1,iw }],xf[0]),xf[1]), xf[2]),
				lerp( 
					lerp(lerp(F[{ix ,iy ,iz ,iw1}],F[{ix1,iy ,iz ,iw1}],xf[0]),lerp(F[{ix ,iy1,iz ,iw1}],F[{ix1,iy1,iz ,iw1}],xf[0]),xf[1]),
					lerp(lerp(F[{ix ,iy ,iz1,iw1}],F[{ix1,iy ,iz1,iw1}],xf[0]),lerp(F[{ix ,iy1,iz1,iw1}],F[{ix1,iy1,iz1,iw1}],xf[0]),xf[1]), xf[2]),
				xf[3]);
		}
	#endif
	}
#endif
}