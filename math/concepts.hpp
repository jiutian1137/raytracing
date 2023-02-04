#pragma once

/// number = cmath + complex + ADLnumber
/// concept number
/// concept real_number
/// concept complex_number
/// 
/// vector = number + container
/// concept vector
/// concept real_vector
/// concept complex_vector
/// static_vcast
/// 
/// container = utility + random_access_iterator
/// concept container
/// concept static_container
/// concept dynamic_container
/// using value_t<>
/// 
/// multidimensional
/// concept multidimensinal
/// struct smdsize_t<>
/// struct smdintptr_t<>
/// struct smdptrdiff_t<>
#define _MATH_CONCEPTS_

#include <concepts>

#include <xutility>
#include <cmath>///basic-type not have ADL.
#include <algorithm>
using std::min, std::max, std::clamp;

namespace math {
	template<typename __number>
	concept number = requires(const __number& x) {
		{ x + x } -> std::convertible_to<__number>;
		{ x - x } -> std::convertible_to<__number>;
		{ x * x } -> std::convertible_to<__number>;
		{ x / x } -> std::convertible_to<__number>;
		//{ x % x || fmod(x,x) } -> std::convertible_to<__number>;
		{ x == x } -> std::convertible_to<bool>;
		{ x != x } -> std::convertible_to<bool>;
		{ x <  x } -> std::convertible_to<bool>;
		{ x <= x } -> std::convertible_to<bool>;
		{ x >  x } -> std::convertible_to<bool>;
		{ x >= x } -> std::convertible_to<bool>;
		{ min(x,x) } -> std::convertible_to<__number>;
		{ max(x,x) } -> std::convertible_to<__number>;
		{ clamp(x,x,x) } -> std::convertible_to<__number>;
	};

	template<typename __number>
	concept signed_number = 
		number<__number> && 
		requires(const __number& x) {
			{ -x } -> std::convertible_to<__number>;
			{ abs(x) } -> std::convertible_to<__number>;
			{ copysign(x,x) } -> std::convertible_to<__number>;
		};

	template<typename __number>
	concept real_number = 
		signed_number<__number> &&
		requires(const __number& x) {
			{ trunc(x) } -> std::convertible_to<__number>;
			{ floor(x) } -> std::convertible_to<__number>;
			{ ceil(x)  } -> std::convertible_to<__number>;
			{ round(x) } -> std::convertible_to<__number>;
			/// Transcendental Functions
			{ pow(x,2) } -> std::convertible_to<__number>;
			{ pow(x,x) } -> std::convertible_to<__number>;
			{ sqrt(x) } -> std::convertible_to<__number>;
			{ cbrt(x) } -> std::convertible_to<__number>;
			{ exp(x)  } -> std::convertible_to<__number>;
			{ exp2(x) } -> std::convertible_to<__number>;
			{ log(x)  } -> std::convertible_to<__number>;
			{ log2(x) } -> std::convertible_to<__number>;
			/// Transcendental Functions (Hyperbolic)
			{ cosh(x) } -> std::convertible_to<__number>;
			{ sinh(x) } -> std::convertible_to<__number>;
			{ tanh(x) } -> std::convertible_to<__number>;
			{ acosh(x) } -> std::convertible_to<__number>;
			{ asinh(x) } -> std::convertible_to<__number>;
			{ atanh(x) } -> std::convertible_to<__number>;
			/// Transcendental Functions (Trigonometric)
			{ cos(x) } -> std::convertible_to<__number>;
			{ sin(x) } -> std::convertible_to<__number>;
			{ tan(x) } -> std::convertible_to<__number>;
			{ acos(x) } -> std::convertible_to<__number>;
			{ asin(x) } -> std::convertible_to<__number>;
			{ atan(x) } -> std::convertible_to<__number>;
			{ atan2(x,x) } -> std::convertible_to<__number>;
		};

	/*template<typename _Complexty>
	concept complex_number = */


	template<typename _Ty>
	concept container = requires(_Ty x) {/* math::container always allow random_access */
		{ std::begin(x) }/* -> std::random_access_iterator*/;
		{ std::end(x) }/* -> std::random_access_iterator*/;
			std::size(x);/* may be smdsize_t */
		{ std::empty(x) } -> std::convertible_to<bool>;
			std::data(x);
	};

	template<typename _Ty>
	concept static_container = container<_Ty> && requires(const _Ty& x) { 
			_Ty::size();/* may be smdsize_t */
			//get<0>(x);/* mdarray also as 1darray, _Ty::at<index>() */
	};

	template<typename _Ty>
	concept dynamic_container = container<_Ty> && requires(_Ty x) {
		x.resize(typename _Ty::size_type());
	};

	template<typename _Ty>
	using value_t = typename std::iterator_traits<decltype(std::begin(std::declval<_Ty>()))>::value_type;

	template<typename _Ty, typename _Pack, typename = void>
	struct uses_package : std::is_same<value_t<_Ty>, _Pack>::type {};

	template<typename _Ty, typename _Pack>
	struct uses_package<_Ty, _Pack, std::void_t<typename _Ty::package_type>>
		: std::is_same<_Pack, typename _Ty::package_type>::type {};

	template<typename _Ty, typename _Pack>
	constexpr bool uses_package_v = uses_package<_Ty, _Pack>::value;


	template<typename __vector/*, not constraint dimension */>
	concept vector = 
		number<value_t<__vector>> &&
		requires(const __vector& x, const value_t<__vector>& s) {
			__vector{ s/*, ...*/ };
			{ x + x } -> std::convertible_to<__vector>;
			{ x - x } -> std::convertible_to<__vector>;
			{ x * x } -> std::convertible_to<__vector>;
			{ x / x } -> std::convertible_to<__vector>;
			///{ x%x || fmod(x,x) } -> std::convertible_to<__vector>;
			{ x + s } -> std::convertible_to<__vector>;
			{ x - s } -> std::convertible_to<__vector>;
			{ x * s } -> std::convertible_to<__vector>;
			{ x / s } -> std::convertible_to<__vector>;
			///{ x%s || fmod(x,s) } -> std::convertible_to<__vector>;
			{ s + x } -> std::convertible_to<__vector>;
			{ s - x } -> std::convertible_to<__vector>;
			{ s * x } -> std::convertible_to<__vector>;
			{ s / x } -> std::convertible_to<__vector>;
			///{ s%x || fmod(s,x) } -> std::convertible_to<__vector>;
			/*{ x == x } -> std::convertible_to<bool>;
			{ x == s } -> std::convertible_to<bool>;
			{ s == x } -> std::convertible_to<bool>;
			{ x != x } -> std::convertible_to<bool>;
			{ x != s } -> std::convertible_to<bool>;
			{ s != x } -> std::convertible_to<bool>;*/
			/* ... compare ... */
			{ min(x,x) } -> std::convertible_to<__vector>;
			{ min(x,s) } -> std::convertible_to<__vector>;
			{ min(s,x) } -> std::convertible_to<__vector>;
			{ max(x,x) } -> std::convertible_to<__vector>;
			{ max(x,s) } -> std::convertible_to<__vector>;
			{ max(s,x) } -> std::convertible_to<__vector>;
			/*{ clamp(x,x,x) } -> std::convertible_to<__vector>;
			{ clamp(x,s,s) } -> std::convertible_to<__vector>;
			{ clamp(x,s,x) } -> std::convertible_to<__vector>;
			{ clamp(x,x,s) } -> std::convertible_to<__vector>;*/
		};

	template<typename __vector/*, not constraint dimension */>
	concept signed_vector = 
		vector<__vector> &&
		signed_number<value_t<__vector>> &&
		requires(const __vector& x, const value_t<__vector>& s) {
			{ -x } -> std::convertible_to<__vector>;
			{ abs(x) } -> std::convertible_to<__vector>;
			{ copysign(x,x) } -> std::convertible_to<__vector>;
			{ copysign(x,s) } -> std::convertible_to<__vector>;
			{ copysign(s,x) } -> std::convertible_to<__vector>;
		};

	template<typename __vector/*, not constraint dimension */>
	concept real_vector = 
		signed_vector<__vector> &&
		real_number<value_t<__vector>> &&
		requires(const __vector& x, const value_t<__vector>& s) {
			{ trunc(x) } -> std::convertible_to<__vector>;
			{ floor(x) } -> std::convertible_to<__vector>;
			{ ceil(x)  } -> std::convertible_to<__vector>;
			{ round(x) } -> std::convertible_to<__vector>;
			/// Transcendental Functions
			{ pow(x,2) } -> std::convertible_to<__vector>;
			{ pow(x,x) } -> std::convertible_to<__vector>;
			{ sqrt(x) } -> std::convertible_to<__vector>;
			{ cbrt(x) } -> std::convertible_to<__vector>;
			{ exp(x)  } -> std::convertible_to<__vector>;
			{ exp2(x) } -> std::convertible_to<__vector>;
			{ log(x)  } -> std::convertible_to<__vector>;
			{ log2(x) } -> std::convertible_to<__vector>;
			/// Transcendental Functions (Hyperbolic)
			{ cosh(x) } -> std::convertible_to<__vector>;
			{ sinh(x) } -> std::convertible_to<__vector>;
			{ tanh(x) } -> std::convertible_to<__vector>;
			{ acosh(x) } -> std::convertible_to<__vector>;
			{ asinh(x) } -> std::convertible_to<__vector>;
			{ atanh(x) } -> std::convertible_to<__vector>;
			/// Transcendental Functions (Trigonometric)
			{ cos(x) } -> std::convertible_to<__vector>;
			{ sin(x) } -> std::convertible_to<__vector>;
			{ tan(x) } -> std::convertible_to<__vector>;
			{ acos(x) } -> std::convertible_to<__vector>;
			{ asin(x) } -> std::convertible_to<__vector>;
			{ atan(x) } -> std::convertible_to<__vector>;
			{ atan2(x,x) } -> std::convertible_to<__vector>;
			/// Vector Functions
			{ dot(x,x) } -> std::convertible_to<value_t<__vector>>;
			{ dot(x) } -> std::convertible_to<value_t<__vector>>;
			{ length(x) } -> std::convertible_to<value_t<__vector>>;
			/*implicit cross(x, x);*/
			/*implicit cross(x, x, x);*/
		};

	template<typename _Vecty>
	consteval std::pair<size_t,bool> _Get_static_vector_size() noexcept {
		if constexpr (static_container<_Vecty>) {
			return std::pair(static_cast<size_t>(_Vecty::size()), true);
		} else {
			return std::pair(size_t(1), false);
		}
	}

	template<size_t dst_offset, typename dst_array>
	constexpr void static_vector_from(dst_array& dst) {}

	template<size_t dst_offset, typename dst_array, typename src_type_i, typename... src_type_N>
	constexpr void static_vector_from(dst_array& dst, const src_type_i& src_i, src_type_N&&... src_N) {
		using                            dst_scalar = std::remove_cvref_t< decltype(std::declval<dst_array>()[0]) >;
		constexpr std::pair<size_t,bool> src_i_size = _Get_static_vector_size<src_type_i>();
		if constexpr (src_i_size.second) 
			for (size_t i = 0; i != src_i_size.first; ++i)
				dst[dst_offset + i] = static_cast<dst_scalar>( src_i[i] );
		else dst[dst_offset] = static_cast<dst_scalar>( src_i );
		static_vector_from<dst_offset + src_i_size.first>(dst, std::forward<src_type_N&&>(src_N)...);
	}

	template<size_t src_offset, typename src_array>
	constexpr void static_vector_to(const src_array& src) {}

	template<size_t src_offset, typename src_array, typename dst_type_i, typename... dst_type_N>
	constexpr void static_vector_to(const src_array& src, dst_type_i& dst_i, dst_type_N&... dst_N) {
		constexpr std::pair<size_t,bool> dst_i_size = _Get_static_vector_size<dst_type_i>();
		if constexpr (dst_i_size.second)
			for (size_t i = 0; i != dst_i_size.first; ++i)
				dst_i[i] = static_cast< std::remove_cvref_t<decltype(std::declval<dst_type_i>()[0])> >( src[src_offset + i] );
		else dst_i = static_cast< dst_type_i >( src[src_offset] );
		static_vector_to<src_offset + dst_i_size.first>(src, std::forward<dst_type_N&>(dst_N)...);
	}

	template<typename src_array, typename dst_type_0, typename... dst_type_N, size_t... dst_index>
	constexpr void static_vector_to(const src_array& src, std::tuple<dst_type_0,dst_type_N...>& dst, std::index_sequence<dst_index...>) {
		static_vector_to<0>(src, std::get<dst_index>(dst)...);
	}
	
	/// {src_0, src_1, ...} to dst.
	template<typename dst_array, typename src_type_0, typename... src_type_N>
	constexpr dst_array static_vector_cast(const src_type_0& src_0, src_type_N&&... src_N) {
		dst_array dst;
		static_vector_from<0>(dst, src_0, std::forward<src_type_N&&>(src_N)...);
		return dst;
	}

	/// trunc_size(src) to dst.
	template<typename dst_type, typename src_type>
	constexpr dst_type static_vector_cast(const src_type& src) {
		constexpr std::pair<size_t,bool> dst_size = _Get_static_vector_size<dst_type>();
		constexpr std::pair<size_t,bool> src_size = _Get_static_vector_size<src_type>();
		if constexpr (dst_size.second) {
			if constexpr (src_size.second) {
				static_assert( dst_size.first <= src_size.first );
				dst_type dst;
				for (size_t i = 0; i != dst_size.first; ++i) {
					dst[i] = static_cast<value_t<dst_type>>( src[i] );
				}
				return dst;
			} else {
				static_assert( dst_size.first == 1 );
				return dst_type{ static_cast<value_t<dst_type>>(src) };
			}
		} else {
			if constexpr (src_size.second) {
				return static_cast<dst_type>( src[0] );
			} else {
				return static_cast<dst_type>( src );
			}
		}
	}

#if 0
	/// load(ptr) to dst.
	template<typename dst_type, typename src_scalar>
	constexpr dst_type static_vector_cast(const src_scalar* ptr) {
		constexpr std::pair<size_t,bool> dst_size = _Get_static_vector_size<dst_type>();
		if constexpr (dst_size.second) {
			dst_type dst;
			for (size_t i = 0; i != dst_size.first; ++i) {
				dst[i] = static_cast<value_t<dst_type>>( ptr[i] );
			}
			return dst;
		} else {
			return static_cast<dst_type>( ptr[0] );
		}
	}
#endif

	/// src to {dst_0, dst_1, ...}.
	template<typename dst_type_0, typename dst_type_1, typename... dst_type_N, typename src_array>
	constexpr std::tuple<dst_type_0,dst_type_1,dst_type_N...> static_vector_cast(const src_array& src) {
		std::tuple<dst_type_0,dst_type_1,dst_type_N...> dst;
		static_vector_to(src, dst, std::make_index_sequence<std::tuple_size_v<decltype(dst)>>());
		return dst;
	}

#ifndef static_vcast
	#define static_vcast ::math::static_vector_cast
#endif

	template<typename static_array, typename scalar>
	constexpr static_array ones(const scalar& val) {
		constexpr std::pair<size_t,bool> ones_array_size = _Get_static_vector_size<static_array>();
		if constexpr (ones_array_size.second) {
			static_array ones_array;
			for (size_t i = 0; i != ones_array_size.first; ++i)
				ones_array[i] = static_cast<value_t<static_array>>(val);
			return ones_array;
		} else {
			return static_array{ static_cast<value_t<static_array>>(val) };
		}
	}


	template<typename _Ty>
	concept multidimensional = requires(const _Ty& a) {
		{ a.dimension() } -> std::convertible_to<size_t>;
	};

	template<typename _Ty, size_t _Nd>
	concept ndimensional = multidimensional<_Ty> && (_Ty::dimension() == _Nd);

#if 0
	template<typename size_type, std::invocable<const size_type&> function>
	constexpr void for_each(const size_t dimension, const size_type& first, const size_type& last, const function& fn) {
		if (dimension == 1) {
			fn(first);
		} else {
			size_type pos = first;
			bool nextline = true;
			while (nextline) {
				fn(pos);

				size_t carry = 1;
				while (++pos[carry] == last[carry]) {
					if (carry+1 == dimension) {
						nextline = false;
						break;
					}
					pos[carry++] = first[0];
				}
			}
		}
	}
#endif
//namespace std {
//#ifndef __algorithm_mirror
//	template<typename _Ty>
//	constexpr _Ty mirror(const _Ty& val, const _Ty& minval, const _Ty& maxval) {
//		std::complex<
//	if (val < minval) {
//		return minval + (minval - val);
//	}
//
//	if (val > maxval) {
//		return maxval - (val - maxval);
//	}
//
//	return val;
//}
//#endif
//}
}// end of namespace math


#include <limits>
#include <vector>/// default container.

/// std extension.
_STD_BEGIN
template<typename _Ty1, size_t _Ty2idx, 
  typename _Ty2 = std::remove_cvref_t<decltype(std::declval<_Ty1>()[0])>>
struct jointed_pair {
  _Ty1 first;

  inline _Ty2& second() {
    return reinterpret_cast<_Ty2&>(first[_Ty2idx]);
  }

  inline const _Ty2& second() const {
    return reinterpret_cast<const _Ty2&>(first[_Ty2idx]);
  }
};

template<typename _Ty1, size_t _Ty2idx>
struct jointed_pair<_Ty1, _Ty2idx, std::remove_cvref_t<decltype(std::declval<_Ty1>()[0])>> {
  _Ty1 first;

  constexpr auto& second() {
    return first[_Ty2idx];
  }

  constexpr const auto& second() const {
    return first[_Ty2idx];
  }
};

template <typename _Ty1, size_t _Ty2idx, typename _Ty2>
struct tuple_size<jointed_pair<_Ty1, _Ty2idx, _Ty2>> : integral_constant<size_t, 2> {}; // size of joint_pair

template <size_t _Idx, typename _Ty1, size_t _Ty2idx, typename _Ty2>
struct tuple_element<_Idx, jointed_pair<_Ty1, _Ty2idx, _Ty2>> {
  static_assert(_Idx < 2, "pair index out of bounds");

  using type = conditional_t<_Idx == 0, _Ty1, _Ty2>;
};

template <class _Ret, class _Jpair>
constexpr _Ret _Jpair_get(_Jpair& _Pr, integral_constant<size_t, 0>) noexcept {
  // get reference to element 0 in pair _Pr
  return _Pr.first;
}

template <class _Ret, class _Jpair>
constexpr _Ret _Jpair_get(_Jpair& _Pr, integral_constant<size_t, 1>) noexcept {
  // get reference to element 1 in pair _Pr
  return _Pr.second();
}

template <size_t _Idx, typename _Ty1, size_t _Ty2idx, typename _Ty2>
_NODISCARD constexpr tuple_element_t<_Idx, jointed_pair<_Ty1, _Ty2idx, _Ty2>>& get(jointed_pair<_Ty1, _Ty2idx, _Ty2>& _Pr) noexcept {
  // get reference to element at _Idx in pair _Pr
  using _Rtype = tuple_element_t<_Idx, jointed_pair<_Ty1, _Ty2idx, _Ty2>>&;
  return _Jpair_get<_Rtype>(_Pr, integral_constant<size_t, _Idx>{});
}

template <size_t _Idx, typename _Ty1, size_t _Ty2idx, typename _Ty2>
_NODISCARD constexpr const tuple_element_t<_Idx, jointed_pair<_Ty1, _Ty2idx, _Ty2>>& get(const jointed_pair<_Ty1, _Ty2idx, _Ty2>& _Pr) noexcept {
  // get reference to element at _Idx in pair _Pr
  using _Ctype = const tuple_element_t<_Idx, jointed_pair<_Ty1, _Ty2idx, _Ty2>>&;
  return _Jpair_get<_Ctype>(_Pr, integral_constant<size_t, _Idx>{});
}
_STD_END