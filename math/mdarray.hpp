#pragma once

///@brief Structure of Multi-dimensional Array.
///@license Free 
///@review 2022-11-09 
///@contact Jiang1998Nan@outlook.com 
#define _MATH_MDARRAY_

#include <cassert>

#include <initializer_list>
#include <concepts>
#include <memory>

#ifndef __simd_type_clean
#include <intrin.h>
#endif

namespace math {
///1. Array Structure
///
/// 	template<typename value_type, size_type size>
/// 	struct array {
/// 		value_type __data[ size ];
/// 	};
/// 
/// For single instruction multiple data, we have to align '__data'.
/// "https://en.wikipedia.org/wiki/Single_instruction,_multiple_data"
/// 
/// 	template<typename value_type, size_type size, size_t simdlen>
/// 	struct aligns(simdlen * alignof(value_type)) array {
/// 		value_type __data[ std::align(simdlen, size) ];
/// 	};
/// 
/// 	array<float,N,4> x, y;
/// 	_mm_store_ps(x,__data, _mm_add_ps(_mm_load_ps(x.__data), _mm_load_ps(y.__data)));
/// 
/// But some problems about explicit loading data:
/// (1) (Although the compiler will eliminate some superfluous things, it will not miss anything.)
///     Because we explicit done "loading data", so the compiler may not be able to eliminate the 
///     things that need to be done, and the data may already be in the register.
/// (2) Because the compiler does "inline" and "inline" is important, so we cannot manage registe-
///     rs by ourselves.
/// So we need a new way not explicit loading data. we add a new type 'package_type' instead of 
/// 'value_type' as structure core, and we still need to construct the structure by 'value_type'. 
/// 
/// 	template<typename value_type, size_type size, size_t package_type>
/// 	struct array { 
/// 		static constexpr size_t simdlen = sizeof(package_type)/sizeof(value_type);
/// 		union {
/// 			package_type __data[ std::align(simdlen, size)/simdlen ];
/// 			value_type __data_as_value[ std::align(simdlen, size) ];
/// 		};
/// 	};
/// 
/// 	array<float,N,__m128> x, y;
/// 	x.__data[0] = _mm_add_ps(x.__data[0], y.__data[0]);
/// 
/// 	/// In this way, we can also do the "return value optimization" without "named return value 
/// 	/// optimization".
/// 	if constexpr ( std::align(simdlen, size)/simdlen == ? )
///  		return array<float,N,__m128>{ _mm_add_ps(x.__data[0], y.__data[0]), ... };
/// 
/// 	/// In this way, we can also do inplace construct by 'package_type'.
/// 	__m128 xmm0 = ...;
/// 	auto x = array<float,N,__m128>{ xmm0 };
/// 
/// Because the "union" cannot running at constexpr, so we cannot use two types union. Fortunately 
/// aggregate has a standard of "brase elision", so we can omit '__data_as_value'.
/// 
/// 	template<typename value_type, size_type size, size_t package_type>
/// 	struct array { 
/// 		static constexpr size_t simdlen = sizeof(package_type)/sizeof(value_type);
/// 		package_type __data[ std::align(simdlen, size)/simdlen ];
/// 	};
///
/// 
///2. Multi-dimensional
/// 
/// 	template<typename value_type, size_type mdsize>
/// 	struct mdarray { 
/// 		static constexpr auto __length = (mdsize[0] * ...);
/// 		value_type __data[ __length ];///always continuous no matter what dimension.
/// 	};
/// 
/// About the '__length':
/// (1) Align first non-one dimension, for vector operation.
/// (2) Align first two non-one dimension, for matrix operation. but hardly implement.
/// (.) ...
/// (N) Align all dimensions, for any operation.
/// Because one or all of them may be used, so cannot be embedded in the structure. 
///		
/// 	template<typename value_type, size_type mdsize, size_t package_type, typename alignment>
/// 	struct mdarray { 
/// 		static constexpr size_t simdlen = sizeof(package_type)/sizeof(value_type);
/// 		static constexpr auto __length = alignment::align(simdlen, mdsize);
/// 		package_type __data[ __length/simdlen ];
/// 	};
/// 
/// 	template<typename value_type, size_type mdsize, size_t package_type, typename alignment>
/// 		requires support_simd<alignment>/// The algorithm becomes more complex.
/// 	auto operator+(...) { ... }
/// 
/// This is a problem, the algorithm becomes more complex and hardly implement the various alignme-
/// nts. Let's go back to the starting point "Why need (vector) align?"
/// (1) one-step loading value. (may be many instructions for big 'value_type')
/// (2) some Operation System unsupport unaligned loading.
/// Why need matrix align?
/// (1) one-step loading a matrix values.
/// 
/// Think:
/// (1) But no error will occur if there is vector-alignment but no matrix-alignment.
/// (2) And no hardware for matrix alignment optimization. (the SIMD is instruction parallel, the 
///     Wave Operation is parallel shared register, both no for matrix-alignment.)
/// (3) And hardly implement matrix(or higher order)-alignment optimization. (Hardware and Software)
/// (4) And users can explicit resize the aligned-size for higher order alignment. (may indeed opti-
///     mized.)
/// So we only use vector-alignment, and embedded in the structure.

	/// Align first non-one dimension.
	template<typename size_type, 
		typename length_type = std::remove_cvref_t< decltype(std::declval<size_type>()[0]) >,
		typename dimension_type = std::remove_cvref_t< decltype(std::size(std::declval<size_type>())) > >
	constexpr length_type __vector_align(const length_type& bound, const size_type& mdsize, size_type& stride) {
		length_type length = 1;
		dimension_type i = 0;
		for ( ; i != mdsize.size() && length == 1; ++i) {
			stride[i] = 1;
			length = mdsize[i];
		}
 
		if (length != 1) {
			length = ( (length + (bound - 1)) & (~(bound - 1)) );
			for ( ; i != mdsize.size(); ++i) {
				stride[i] = length;
				length *= mdsize[i];
			}
			return length;
		} else {
			return bound;
		}
	}

/// 	template<typename value_type, size_type mdsize, size_t package_type>
/// 	struct mdarray { 
/// 		static constexpr size_t simdlen = sizeof(package_type)/sizeof(value_type);
/// 		static constexpr auto __length = __vector_align(simdlen, mdsize);
/// 		package_type __data[ __length/simdlen ];
/// 	};
/// 
/// In SIMD Alignment, Vector Alignment vs Nearest Bound Alignment(error)
/// 
/// 	mdarray<float,{1,3,4},__m128> x;
/// 	/** x.size()[2] is nearest bound'simdlen' dimension
/// 	    so Nearest Bound Alignment is mdarray<float,{1,3,4},__m128> x; 
/// 	    |{0,0,0}|{0,1,0}|{0,2,0}| ? | <-- error will occur.
/// 	    |{0,0,1}|{0,1,1}|{0,2,1}|
/// 	    |{0,0,2}|{0,1,2}|{0,2,2}|
/// 	    |{0,0,3}|{0,1,3}|{0,2,3}| */
/// 	/** x.size()[1] is first non-one dimension 
/// 	    so Vector Alignment is mdarray<float,{1,4,4},__m128> x; 
/// 	    |{0,0,0}|{0,1,0}|{0,2,0}|{0,3,0}| <-- no error.
/// 	    |{0,0,1}|{0,1,1}|{0,2,1}|{0,3,1}|
/// 	    |{0,0,2}|{0,1,2}|{0,2,2}|{0,3,2}|
/// 	    |{0,0,3}|{0,1,3}|{0,2,3}|{0,3,3}| */
///
/// 
///3. Multi-dimensional Array Structure 
///
/// 	template<typename value_type, size_type size, typename package_type>
/// 	struct mdarray{ package_type __data[ ... ]; };
/// 
/// We may use various allocators, 
/// 
/// 	template<typename value_type, size_type size, typename package_type, typename allocator_type = std::allocator<package_type>>
/// 	struct mdarray{ package_type __data[ ... ]; };
/// 
/// 'allocator_type::value_type' is 'package_type', so simplified as
/// 
/// 	template<typename value_type, size_type size, typename allocator_type = std::allocator<value_type>>
/// 	struct mdarray{ allocator_type::value_type __data[ ... ]; };
/// 
/// For dynamic mdarray we cannot known the 'size' at compile-time but known the 'dimensions',
/// And the static mdarray unified with a static allocator. 
/// 
/// 	/// Dynamic Multi-dimensional Array.
/// 	template<typename value_type, dimension_type dimension, typename allocator_type = std::allocator<value_type>>
/// 	struct mdarray { allocator_type::value_type *__data; size_type __size; };
/// 
/// 	/// Static Multi-dimensional Allocator.
/// 	template<typename package_type, size_type mdsize>
/// 	struct smdallocator { ... };
/// 
/// 	/// Static Multi-dimensional Array.
/// 	template<typename value_type, dimension_type dimension, typename allocator_type = std::allocator<value_type>>
/// 		requires static_allocator<allocator_type> 
/// 	struct mdarray<value_type, dimensions, ...> { 
/// 		static_assert(allocator_type::dimension() == dimension);
/// 		allocator_type::value_type __data[ ... ];
/// 	};
/// 
/// 	/// Example.
/// 	mdarray<float, 2> A;
/// 	mdarray<float, 2, smdallocator<float,{4,4}>> x;
/// 
/// This is a good way.
/// 
///3.1 Multi-dimensional Array Structure, Another way
/// 
/// *. Because static-mdarray is different from dynamic-mdarray even though they can be unified. 
/// *. Because static-mdarray can be simplified when not unified. 
/// *. Because operations of static-mdarray may be completely different from dynamic-mdarray.
/// 	o. one use "Math Symbol" with constexpr optimization.
/// 	o. another use "Basic Linear Algebra Subroutine" optimization by users themselves.
/// We use another way to simplify the static-mdarray.
/// 
/// 	/// Dynamic Multi-dimensional Array.
/// 	template<typename value_type, dimension_type dimension, typename allocator = std::allocator<value_type>>
/// 	struct mdarray;
///
/// 	/// Static Multi-dimensional Array.
/// 	template<typename value_type, size_type mdsize, typename package_type = value_type>
/// 	struct mdarray;
/// 
/// We may use various size_type for high-performance.
///
/// 	/// Dynamic Multi-dimensional Array, dimensiona = size_type::size().
/// 	template<typename value_type, typename size_type, typename allocator = std::allocator<value_type>>
/// 	struct mdarray;
///
/// 	/// Static Multi-dimensional Array.
/// 	template<typename value_type, auto mdsize, typename package_type = value_type>
/// 	struct smdarray;
///
///3.2. Multi-dimensional Array Structure with View
///
/// We may use a part of Multi-dimensional Array, 

	/// Dynamic Multi-dimensional Array View.
	template<typename __value_type, typename __size_type, typename allocator = std::allocator<__value_type>>
	struct mdarray_view {
		static_assert( sizeof(typename allocator::value_type) == sizeof(__value_type)
			|| sizeof(typename allocator::value_type) % sizeof(__value_type) == 0,
			"Mismatch allocator<T|SIMD,...> and T" );

		using value_type      = __value_type;
		using allocator_type  = allocator;
		using package_type    = typename allocator::value_type;
		using size_type       = __size_type;
		using length_type     = std::remove_cvref_t< decltype(std::declval<size_type>()[0]) >;
		using dimension_type  = std::remove_cvref_t< decltype(std::declval<size_type>().size()) >;
		static constexpr length_type simdlen = 
			static_cast<length_type>( sizeof(package_type) / sizeof(value_type) );

		using pointer         = __value_type *;
		using const_pointer   = const __value_type *;
		using reference       = __value_type &;
		using const_reference = const __value_type &;
		using iterator        = value_type *;
		using const_iterator  = const value_type *;

		package_type* __data;
		length_type __package_length;/// o(1) determine number of forloop.
		size_type __size;  /// O(1) check dimension (is vector, is matrix, ...).
		size_type __stride;/// O(1) fetch element.

		constexpr explicit mdarray_view(package_type* data = nullptr, const length_type& package_length = 0, const size_type& size = {}, const size_type& stride = {}) noexcept
			: __data(data), __package_length(package_length), __size(size), __stride(stride) {}

		static constexpr dimension_type dimension() {
			return std::size(size_type());
		}
		
		constexpr bool empty() const {
			return __data == nullptr;
		}
		
		constexpr const size_type& size() const {
			return __size;
		}
		
		constexpr const size_type& stride() const {
			return __stride;
		}

		constexpr length_type length() const {
#if 0
			constexpr dimension_type back_i = std::size(size_type()) - 1;
			const length_type __length =
				__stride[back_i] * (__size[back_i] == 1 ? 1 : std::align(simdlen, __size[back_i]));
			return __length == 1 ? simdlen : __length;
#else
			return __package_length * simdlen;
#endif
		}
		
		constexpr length_type package_length() const {
			return __package_length;/* length()/simdlen */;
		}

		constexpr length_type size(const size_t dim) const {
			return __size[dim];
		}
		
		constexpr length_type stride(const size_t dim) const {
			return __stride[dim];
		}

		constexpr pointer data() {
			if constexpr (std::same_as<package_type, value_type>)
				return __data;
			else return reinterpret_cast<pointer>(__data);
		}

		constexpr const_pointer data() const {
			if constexpr (std::same_as<package_type, value_type>)
				return __data;
			else return reinterpret_cast<const_pointer>(__data);
		}
		
		constexpr iterator begin() {
			if constexpr (std::same_as<package_type, value_type>)
				return __data;
			else return reinterpret_cast<iterator>(__data);
		}
		
		constexpr const_iterator begin() const {
			if constexpr (std::same_as<package_type, value_type>)
				return __data;
			else return reinterpret_cast<const_iterator>(__data);
		}
		
		constexpr iterator end() {
			return begin() + length();
		}
		
		constexpr const_iterator end() const {
			return begin() + length();
		}
		
		template<typename size_type2>
		constexpr reference operator[](const size_type2& i) {
			if constexpr (std::integral<size_type2>) {
				assert( i < length() );
				if constexpr (std::same_as<package_type, value_type>) {
					return __data[i];
				} else if (std::is_constant_evaluated()) {
#ifndef __simd_type_clean
					if constexpr (std::same_as<package_type, __m128>)
						return __data[i/simdlen].m128_f32[i%simdlen];
					else if constexpr (std::same_as<package_type, __m128d>)
						return __data[i/simdlen].m128d_f64[i%simdlen];
					else if constexpr (std::same_as<package_type, __m128i>)
						return __data[i/simdlen].m128i_i32[i%simdlen];
					else if constexpr (std::same_as<package_type, __m256>)
						return __data[i/simdlen].m256_f32[i%simdlen];
					else if constexpr (std::same_as<package_type, __m256d>)
						return __data[i/simdlen].m256d_f64[i%simdlen];
					else if constexpr (std::same_as<package_type, __m256i>)
						return __data[i/simdlen].m256i_i32[i%simdlen];
					else 
#endif
						return __data[i/simdlen][i%simdlen];
				} else {
					return reinterpret_cast<iterator>(__data)[i];
				}
			} else {
				auto index = i[0];
				if constexpr (dimension() != 1) {
					constexpr dimension_type dend = dimension();
					dimension_type d = 1;
					do {
						index += i[d] * stride(d);
					} while (++d != dend);
				}

				if constexpr (std::same_as<package_type, value_type>) {
					return __data[index];
				} else if (std::is_constant_evaluated()) {
					#ifndef __simd_type_clean
					if constexpr (std::same_as<package_type, __m128>)
						return __data[index/simdlen].m128_f32[index%simdlen];
					else if constexpr (std::same_as<package_type, __m128d>)
						return __data[index/simdlen].m128d_f64[index%simdlen];
					else if constexpr (std::same_as<package_type, __m128i>)
						return __data[index/simdlen].m128i_i32[index%simdlen];
					else if constexpr (std::same_as<package_type, __m256>)
						return __data[index/simdlen].m256_f32[index%simdlen];
					else if constexpr (std::same_as<package_type, __m256d>)
						return __data[index/simdlen].m256d_f64[index%simdlen];
					else if constexpr (std::same_as<package_type, __m256i>)
						return __data[index/simdlen].m256i_i32[index%simdlen];
					else 
#endif
						return __data[index/simdlen][index%simdlen];
				} else {
					return reinterpret_cast<iterator>(__data)[index];
				}
			}
		}
		
		template<typename size_type2>
		constexpr const_reference operator[](const size_type2& i) const {
			return const_cast<mdarray_view&>(*this)[i];
		}

		length_type unchanged_resize(const size_type& new_size) {
			size_type   new_stride;
			length_type new_length = __vector_align(simdlen, new_size, new_stride);
			length_type new_package_length = new_length / simdlen;
			if (new_package_length != __package_length)
				throw ;
			__size   = new_size;
			__stride = new_stride;
			return new_length;
		}

		mdarray_view& operator=(std::initializer_list<value_type> ilist)  {
			assert( ilist.size() == length() );
			pointer __scalar_i = begin();
			for (const value_type& each_scalar : ilist)
				(*__scalar_i++) = each_scalar;
			return *this;
		}
	};

/// 	template<typename value_type, auto size, typename package_type>
/// 	struct smdarray_view {
/// 		/// Cannot define.
/// 	};
/// 
/// Not have static multi-dimensional array view.
/// *. Because if smdarray_view defines a pointer, then smdarray cannot as smdarray_view. 
///    ('package_type __data[__length]' cannot as 'package_type *__data'.)
/// *. Because if smdarray_view defines a array, then the view will copy by value.
///    (view should copy by reference.)
/// 
/// Note: there is no better way to define the view which can maintain "mdarray as/is mdarray_view"
///       unless implicit-convertion can be match template parameters.

	/// Dynamic Multi-dimensional Array, 'mdarray' "as" 'mdarray_view'.
	template<typename __value_type, typename __size_type, typename allocator = std::allocator<__value_type>>
	struct mdarray : mdarray_view<__value_type, __size_type, allocator> {
		using __base = mdarray_view<__value_type, __size_type, allocator>;
		using __base::simdlen;
		using __base::__data;
		using __base::__package_length;
		using __base::__size;
		using __base::__stride;

		using value_type      = __value_type;
		using allocator_type  = allocator;
		using package_type    = typename allocator::value_type;
		using pointer         = __value_type *;
		using const_pointer   = const __value_type *;
		using reference       = __value_type &;
		using const_reference = const __value_type &;
		using size_type       = __size_type;
		using length_type     = std::remove_cvref_t< decltype(std::declval<size_type>()[0]) >;
		using dimension_type  = std::remove_cvref_t< decltype(std::declval<size_type>().size()) >;

		using iterator        = value_type *;
		using const_iterator  = const value_type *;

		allocator _Al;

/// Safe construct by any thing

		template<std::invocable<package_type*,length_type> uninitialized_constructor>
		void _Construct(const size_type& the_size, const uninitialized_constructor& uninitialized_construct) {
			size_type   the_stride;
			length_type the_length = __vector_align(simdlen, the_size, the_stride); assert(the_length != 0);
			length_type the_package_length = the_length/simdlen;
			__data = _Al.allocate(the_package_length);
			try {
				uninitialized_construct(__data, the_package_length);
			} catch (const std::exception&) {
				_Al.deallocate(__data, the_package_length);
				__data = nullptr;
				throw;
			}

			__package_length = the_package_length;
			__size   = the_size;
			__stride = the_stride;
		}

		constexpr mdarray() noexcept = default;
	
		explicit mdarray(const size_type& size) {
			_Construct(size, [](package_type* data, const length_type package_length) {
				std::uninitialized_default_construct_n(data, package_length);
			});
		}

		mdarray(const size_type& size, const package_type& value) {
			_Construct(size, [&value](package_type* data, const length_type package_length) {
				std::uninitialized_fill_n(data, package_length, value);
			});
		}

		mdarray(const size_type& size, std::initializer_list<value_type> ilist) {
			//assert( length == ilist.size() );
			_Construct(size, [ilist](package_type* data, const length_type package_length) {
				std::uninitialized_copy_n(reinterpret_cast<value_type*>(data), package_length * simdlen, ilist.begin());
			});
		}
		
		template<typename input_iterator>
		mdarray(const size_type& size, input_iterator first, const input_iterator last) {
			//assert( length == std::distance(first,last) );
			_Construct(size, [first,last](package_type* data, const length_type package_length) {
				std::uninitialized_copy_n(reinterpret_cast<value_type*>(data), package_length * simdlen, first);
			});
		}

		template<typename... size_types>
			requires std::conjunction_v<std::is_convertible<size_types, length_type>...>
		explicit mdarray(length_type size_0, size_types... size_N)
			: mdarray( size_type{size_0,static_cast<length_type>(size_N)...} ) {}

		mdarray(mdarray&& right) noexcept
			: __base(right.__data, right.__package_length, right.__size, right.__stride) {
			right.__data   = nullptr;
			right.__package_length = 0;
			right.__size   = size_type;
			right.__stride = size_type();
		}

		mdarray(const mdarray& right) {
			if (right.__package_length != 0) {
				__data = _Al.allocate(right.__package_length);
				try {
					std::uninitialized_copy_n(right.__data, right.__package_length, __data);
				} catch (const std::exception&) {
					_Al.deallocate(__data, right.__package_length);
					__data = nullptr;
					throw;
				}
				__package_length = right.__package_length;
				__size   = right.__size;
				__stride = right.__stride;
			}
		}

		~mdarray() noexcept {
			if (__data != nullptr) {
				std::destroy_n(__data, __package_length);
				_Al.deallocate(__data, __package_length);
				__data   = nullptr;
				__package_length = 0;
				__size   = size_type();
				__stride = size_type();
			}
		}

		void clear() noexcept { 
			this->~mdarray();
		}

/// Safe resize

		void _Resize(const size_type& new_size, const package_type& initval) {
			size_type   new_stride;
			length_type new_length = __vector_align(simdlen, new_size, new_stride); assert(new_length != 0);
			length_type new_package_length = new_length / simdlen;
			if (new_package_length != __package_length) {
				package_type* new_data = _Al.allocate(new_package_length);
				try {
					auto new_data_backout = std::_Uninitialized_backout(new_data, 
						std::uninitialized_fill_n(new_data, new_package_length, initval));

					if constexpr (mdarray::dimension() == 1) {
						if constexpr (std::is_nothrow_move_constructible_v<package_type> || !std::is_copy_constructible_v<package_type>) {
							std::move(__data, std::next(__data, std::min(__package_length, new_package_length)), new_data);
						} else {
							std::copy(__data, std::next(__data, std::min(__package_length, new_package_length)), new_data);
						}
					} else if (__package_length != 0) {
						size_type pos;
						size_type copy_size;
						for (size_t i = 0; i != mdarray::dimension(); ++i) {
							pos[i]       = 0;
							copy_size[i] = std::min(__size[i], new_size[i]);
						}

						bool nextline = true;
						while (nextline) {
							size_t __data_i_offset = 0;
							size_t new_data_i_offset = 0;
							for (size_t i = 1; i != mdarray::dimension(); ++i) {
								__data_i_offset += pos[i] * __stride[i];
								new_data_i_offset += pos[i] * new_stride[i];
							}
							if constexpr (std::is_nothrow_move_constructible_v<package_type> || !std::is_copy_constructible_v<package_type>) {
								std::move(std::next(__data, (__data_i_offset + (simdlen - 1))/simdlen), 
									std::next(__data, (__data_i_offset + copy_size[0] + (simdlen - 1))/simdlen),
									std::next(new_data, (new_data_i_offset + (simdlen - 1))/simdlen));
							} else {
								std::copy(std::next(__data, (__data_i_offset + (simdlen - 1))/simdlen), 
									std::next(__data, (__data_i_offset + copy_size[0] + (simdlen - 1))/simdlen),
									std::next(new_data, (new_data_i_offset + (simdlen - 1))/simdlen));
							}

							size_t carry = 1;
							while (++pos[carry] == copy_size[carry]) {
								if (carry+1 == mdarray::dimension()) {
									nextline = false;
									break;
								}
								pos[carry++] = 0;
							}
						}
					}

					new_data_backout._Release();
				} catch (const std::exception&) {
					_Al.deallocate(new_data, new_package_length);
					throw;
				}

				if (__data != nullptr) {
					std::destroy_n(__data, __package_length);
					_Al.deallocate(__data, __package_length);
				}
				__data = new_data;
				__package_length = new_package_length;
			}
			__size   = new_size;
			__stride = new_stride;
		}

		void resize(const size_type& new_size, const package_type& value = {}) {
			_Resize(new_size, value);
		}

		template<typename... size_types>
			requires std::conjunction_v<std::is_convertible<size_types, length_type>...>
		void resize(length_type size_0, size_types... size_N) { 
			resize(size_type{size_0, static_cast<length_type>(size_N)...});
		}

/// Safe assign by any thing

		mdarray& operator=(mdarray&& right) noexcept {
			if (this != (&right)) {
				this->clear();
				static_cast<__base&>(*this) = static_cast<const __base&>(right);
				static_cast<__base&>(right) = __base();
			}
			return *this;
		}
		
		mdarray& operator=(const mdarray& right) {
			if (this != (&right)) 
				(*this) = std::move(mdarray(right));
			return *this;
		}
		
		explicit mdarray(const __base& view) : mdarray(static_cast<const mdarray&>(view)) {}

		mdarray& operator=(const __base& view) { return (*this) = static_cast<const mdarray&>(view); }
	};

	/// Static Multi-dimensional Array, 'smdarray' "not as" any view.
	template<typename __value_type, auto __size, typename __package_type = __value_type>
	struct smdarray {
		static_assert( !__size.empty() );
		static_assert( sizeof(__value_type) == sizeof(__package_type)
			|| sizeof(__package_type) % sizeof(__value_type) == 0,
			"Mismatch allocator<T|SIMD,...> and T" );

		using value_type      = __value_type;
		using package_type    = __package_type;
		using allocator_type  = void;
		using size_type2       = decltype(__size);
		using size_type       = std::remove_cvref_t< decltype(__size) >;
		using length_type     = std::remove_cvref_t< decltype(std::declval<size_type>()[0]) >;
		using dimension_type  = std::remove_cvref_t< decltype(std::size(std::declval<size_type>())) >;
		static constexpr length_type simdlen = 
			static_cast<length_type>( sizeof(package_type) / sizeof(value_type) );

		using pointer         = __value_type*;
		using const_pointer   = const __value_type*;
		using reference       = __value_type&;
		using const_reference = const __value_type&;
		using iterator        = pointer;
		using const_iterator  = const_pointer;
		
		///Note
		/// If we defined as
		///		static consteval size_t size(){ return __size; }
		/// 
		/// Then call from std::size(..), 
		///		call a consteval function by a (maybe|not)non-constexpr object.
		/// 
		/// because 
		///		call a consteval function by a maybe non-constexpr object.
		///		(The compiler may allow this case, but the cost is not small. 
		///		 For example, it must check whether all objects calling this function are constexpr)
		/// 
		/// so
		///		syntax error! "We not define consteval on public functions".
		///  

		static consteval size_type __stride() {
			size_type stride;
			__vector_align(simdlen, __size, stride);
			return stride;
		}

		static consteval length_type __length() {
			size_type ignored;
			return __vector_align(simdlen, __size, ignored);
		}
		
		static consteval length_type __package_length() {
			return __length()/simdlen;
		}

		package_type __data[ __package_length() ];

		static constexpr dimension_type dimension() {
			return std::size(__size);
		}

		static constexpr bool empty() {
			return false;
		}
		
		static constexpr size_type size() {
			return __size;
		}
		
		static constexpr size_type stride() {
			return __stride();
		}

		static constexpr length_type length() {
			return __length();
		}

		static constexpr length_type package_length() {
			return __package_length();
		}

		static constexpr length_type size(dimension_type d) {
			assert( d < dimension() );
			return __size[d];
		}
		
		static constexpr length_type stride(dimension_type d) {
			assert( d < dimension() );
			return __stride()[d];
		}

		constexpr package_type* data() {
			return __data;
		}

		constexpr const package_type* data() const {
			return __data;
		}
		
		constexpr iterator begin() {
			if constexpr (std::same_as<package_type, value_type>)
				return __data;
			else return reinterpret_cast<iterator>(__data);
		}
		
		constexpr const_iterator begin() const {
			if constexpr (std::same_as<package_type, value_type>)
				return __data;
			else return reinterpret_cast<const_iterator>(__data);
		}
		
		constexpr iterator end() {
			return begin() + length();
		}
		
		constexpr const_iterator end() const {
			return begin() + length();
		}

		template<typename size_type2>
		constexpr reference operator[](const size_type2& i) {
			if constexpr (std::integral<size_type2>) {
				assert( i < length() );
				if constexpr (std::same_as<package_type, value_type>) {
					return __data[i];
				} else if (std::is_constant_evaluated()) {
#ifndef __simd_type_clean
					if constexpr (std::same_as<package_type, __m128>)
						return __data[i/simdlen].m128_f32[i%simdlen];
					else if constexpr (std::same_as<package_type, __m128d>)
						return __data[i/simdlen].m128d_f64[i%simdlen];
					else if constexpr (std::same_as<package_type, __m128i>)
						return __data[i/simdlen].m128i_i32[i%simdlen];
					else if constexpr (std::same_as<package_type, __m256>)
						return __data[i/simdlen].m256_f32[i%simdlen];
					else if constexpr (std::same_as<package_type, __m256d>)
						return __data[i/simdlen].m256d_f64[i%simdlen];
					else if constexpr (std::same_as<package_type, __m256i>)
						return __data[i/simdlen].m256i_i32[i%simdlen];
					else 
#endif
						return __data[i/simdlen][i%simdlen];
				} else {
					return reinterpret_cast<iterator>(__data)[i];
				}
			} else {
				size_type2 index = i[0];
				if constexpr (dimension() != 1) {
					constexpr dimension_type dend = dimension();
					dimension_type d = 1;
					do {
						index += i[d] * stride(d);
					} while (d != dend);
				}

				if constexpr (std::same_as<package_type, value_type>) {
					return __data[index];
				} else if (std::is_constant_evaluated()) {
					#ifndef __simd_type_clean
					if constexpr (std::same_as<package_type, __m128>)
						return __data[index/simdlen].m128_f32[index%simdlen];
					else if constexpr (std::same_as<package_type, __m128d>)
						return __data[index/simdlen].m128d_f64[index%simdlen];
					else if constexpr (std::same_as<package_type, __m128i>)
						return __data[index/simdlen].m128i_i32[index%simdlen];
					else if constexpr (std::same_as<package_type, __m256>)
						return __data[index/simdlen].m256_f32[index%simdlen];
					else if constexpr (std::same_as<package_type, __m256d>)
						return __data[index/simdlen].m256d_f64[index%simdlen];
					else if constexpr (std::same_as<package_type, __m256i>)
						return __data[index/simdlen].m256i_i32[index%simdlen];
					else 
#endif
						return __data[index/simdlen][index%simdlen];
				} else {
					return reinterpret_cast<iterator>(__data)[index];
				}
			}
		}
		
		template<typename size_type2>
		constexpr const_reference operator[](const size_type2& i) const {
			return const_cast<smdarray&>(*this)[i];
		}

		template<typename value_type2> requires std::convertible_to<value_type, value_type2>
		constexpr explicit operator value_type2() const {
			return static_cast<value_type2>((*this)[0]);
		}
	};

	/// Static Multi-dimensional Vector(or Size).
	template<typename __value_type, auto __size, typename __package_type>
		requires std::integral< std::remove_cvref_t<decltype(__size)> >
	struct smdarray<__value_type, __size, __package_type> {
		static_assert( __size != 0 );
		static_assert( sizeof(__value_type) == sizeof(__package_type)
			|| sizeof(__package_type) % sizeof(__value_type) == 0,
			"Mismatch allocator<T|SIMD,...> and T" );

		using value_type      = __value_type;
		using package_type    = __package_type;
		using allocator_type  = void;
		using size_type       = std::remove_cvref_t< decltype(__size) >;
		using length_type     = size_type;
		using dimension_type  = size_t;
		static constexpr length_type simdlen = 
			static_cast<length_type>( sizeof(package_type) / sizeof(value_type) );

		using pointer         = __value_type*;
		using const_pointer   = const __value_type*;
		using reference       = __value_type&;
		using const_reference = const __value_type&;
		using iterator        = pointer;
		using const_iterator  = const_pointer;

		static consteval length_type __length() {
			return ( (__size + (simdlen - 1)) & (~(simdlen - 1)) );
		}
		
		static consteval length_type __package_length() {
			return __length()/simdlen;
		}

		package_type __data[ __package_length() ];

		static constexpr dimension_type dimension() {
			return 1;
		}

		static constexpr bool empty() {
			return false;
		}
		
		static constexpr size_type size() {
			return __size;
		}
		
		static constexpr size_type stride() {
			return 1 ;
		}

		static constexpr length_type length() {
			return __length();
		}

		static constexpr length_type package_length() {
			return __package_length();
		}

		static constexpr length_type size(dimension_type d) {
			assert( d == 0 );
			return __size;
		}
		
		static constexpr length_type stride(dimension_type d) {
			assert( d < 0 );
			return 1;
		}

		constexpr package_type* data() {
			return __data;
		}

		constexpr const package_type* data() const {
			return __data;
		}
		
		constexpr iterator begin() {
			if constexpr (std::same_as<package_type, value_type>)
				return __data;
			else return reinterpret_cast<iterator>(__data);
		}
		
		constexpr const_iterator begin() const {
			if constexpr (std::same_as<package_type, value_type>)
				return __data;
			else return reinterpret_cast<const_iterator>(__data);
		}
		
		constexpr iterator end() {
			return begin() + length();
		}
		
		constexpr const_iterator end() const {
			return begin() + length();
		}

		template<typename size_type2>
		constexpr reference operator[](const size_type2& i) {
			if constexpr (std::integral<size_type2>) {
				assert( i < length() );
				if constexpr (std::same_as<package_type, value_type>) {
					return __data[i];
				} else if (std::is_constant_evaluated()) {
#ifndef __simd_type_clean
					if constexpr (std::same_as<package_type, __m128>)
						return __data[i/simdlen].m128_f32[i%simdlen];
					else if constexpr (std::same_as<package_type, __m128d>)
						return __data[i/simdlen].m128d_f64[i%simdlen];
					else if constexpr (std::same_as<package_type, __m128i>)
						return __data[i/simdlen].m128i_i32[i%simdlen];
					else if constexpr (std::same_as<package_type, __m256>)
						return __data[i/simdlen].m256_f32[i%simdlen];
					else if constexpr (std::same_as<package_type, __m256d>)
						return __data[i/simdlen].m256d_f64[i%simdlen];
					else if constexpr (std::same_as<package_type, __m256i>)
						return __data[i/simdlen].m256i_i32[i%simdlen];
					else 
#endif
						return __data[i/simdlen][i%simdlen];
				} else {
					return reinterpret_cast<iterator>(__data)[i];
				}
			} else {
				if constexpr (std::same_as<package_type, value_type>) {
					return __data[i[0]];
				} else if (std::is_constant_evaluated()) {
					#ifndef __simd_type_clean
					if constexpr (std::same_as<package_type, __m128>)
						return __data[i[0]/simdlen].m128_f32[i[0]%simdlen];
					else if constexpr (std::same_as<package_type, __m128d>)
						return __data[i[0]/simdlen].m128d_f64[i[0]%simdlen];
					else if constexpr (std::same_as<package_type, __m128i>)
						return __data[i[0]/simdlen].m128i_i32[i[0]%simdlen];
					else if constexpr (std::same_as<package_type, __m256>)
						return __data[i[0]/simdlen].m256_f32[i[0]%simdlen];
					else if constexpr (std::same_as<package_type, __m256d>)
						return __data[i[0]/simdlen].m256d_f64[i[0]%simdlen];
					else if constexpr (std::same_as<package_type, __m256i>)
						return __data[i[0]/simdlen].m256i_i32[i[0]%simdlen];
					else 
#endif
						return __data[i[0]/simdlen][i[0]%simdlen];
				} else {
					return reinterpret_cast<iterator>(__data)[i[0]];
				}
			}
		}
		
		template<typename size_type2>
		constexpr const_reference operator[](const size_type2& i) const {
			return const_cast<smdarray&>(*this)[i];
		}

		template<typename value_type2> requires std::convertible_to<value_type, value_type2>
		constexpr explicit operator value_type2() const {
			return static_cast<value_type2>((*this)[0]);
		}
	};


///4. Optimization

/// Swizzle for high-performance

	template<size_t index, typename value, auto size, typename package>
	constexpr value& get(smdarray<value, size, package>& src) {
		return src[index];
	}

	template<size_t index, typename value, auto size, typename package>
	constexpr const value& get(const smdarray<value, size, package>& src) {
		return src[index];
	}

	template<size_t index, typename value, auto size, typename package>
	constexpr value extract(const smdarray<value, size, package>& src) {
		if constexpr (std::same_as<package, __m128i>) {
			return _mm_extract_epi32(src.__data[0], index);
		} else
		return src[index];
	}

	template<size_t... index, typename value, auto size, typename package>
	constexpr smdarray<value,size,package> permute(const smdarray<value,size,package>& src) {
#ifndef __simd_type_clean
		if constexpr (std::same_as<package, __m128i>) {
			smdarray<value,size,package> dst;
			size_t i = 0;
			((dst[i++] = src[index]), ...);
			return std::move(dst);
		} else
#endif
		return { src[index]... };
	}

	template<size_t... index, typename value, auto size, typename package>
	constexpr smdarray<value,size,package> permute(const smdarray<value,size,package>& src0, const smdarray<value,size,package>& src1) {
		const auto select = [&src0,&src1](size_t i) { 
			return i < src0.length() ? src0[i] : src1[i - src0.length()]; };
#ifndef __simd_type_clean
		if constexpr (std::same_as<package, __m128i>) {
			smdarray<value,size,package> dst;
			size_t i = 0;
			((dst[i++] = select(index)), ...);
			return std::move(dst);
		} else
#endif
		return { select(index)... };
	}

	template<typename value, auto size, typename package>
	constexpr void load(const value* x, smdarray<value,size,package>& y) {
		for (size_t i = 0, iend = static_cast<size_t>(size); i != iend; ++i) {
			y[i] = x[i];
		}
	}

	template<typename value, auto size, typename package>
	constexpr void store(value* y, const smdarray<value,size,package>& x) {
		for (size_t i = 0, iend = static_cast<size_t>(size); i != iend; ++i) {
			y[i] = x[i];
		}
	}

#if 0
	template<auto mdindex, typename value, auto size, typename package>
		requires static_mdarray<smdarray_view<value, size, package>>
	consteval size_t _Consteval_get_index() {
		using mdarray_type = mdarray<scalar, dimension, allocator>;
		size_t index = mdindex[0] * mdarray_type::stride(0);
		for (size_t i = 1; i != mdindex.size(); ++i) {
			index += mdindex[i] * mdarray_type::stride(i);
		}

		return index;
	}

	template<smdsize mdindex, typename scalar, size_t dimension, typename allocator>
		requires static_mdarray<mdarray_view<scalar, dimension, allocator>>
	constexpr const scalar& get(const mdarray_view<scalar, dimension, allocator>& _Arr) {
		return _Arr[ _Consteval_get_index<mdindex, scalar, dimension, allocator>() ];
	}

	template<smdsize mdindex, typename scalar, size_t dimension, typename allocator>
		requires static_mdarray<mdarray_view<scalar, dimension, allocator>>
	constexpr scalar& get(mdarray_view<scalar, dimension, allocator>& _Arr) {
		return _Arr[ _Consteval_get_index<mdindex, scalar, dimension, allocator>() ];
	}
#endif

/// Simplify template parameter for 'mdsize'.

	template<typename type>
	struct smdsize_type_deduce {
		using value_type = type;
		using package_type = type;
	};

	template<typename type, size_t dimension>
	using smdsize = smdarray<typename smdsize_type_deduce<type>::value_type, dimension,
		typename smdsize_type_deduce<type>::package_type>;

	template<size_t dimension>
	using smdsize_t = smdsize<size_t, dimension>;

	template<size_t dimension>
	using smdintptr_t = smdsize<size_t, dimension>;

 
///Summary the mdarray
/// *. Accelerated index computation, by using faster tparam'size_type'.
/// *. Single instruction multiple data, by tparam 'package_type' or 'allocator_type::value_type'.
/// *. (Dynamic only) Customized memory, by tparam 'allocator_type'.
/// *. (Dynamic only) Saft memory.
/// *. Deeper optimization, by swizzle functions.
/// *. Deeper optimization, by convert functions.
/// *. Get view directly, by mdarray as mdarray_view. @note cannot modify the view before a copy.
/// 
///Note 
/// "array_view" is descriptor, "array" is descriptor + memory.
///	 so "array" as "array_view", but "array_view" not as "array".
/// 
/// "size"   is number of all elements.
/// 
/// "length" is number of all elements entire continues memory.
/// 
/// mdarray is "vector" with
///  ( dimension == 1 || max(size(dim)) == size() ).
/// 
/// mdarray is "matrix" with 
///  ( dimension == 2 || (dimension == 1) ).
/// 
/// difference between static_mdarray and dynamic_mdarray
///		1. size
///			static_mdarray.size is constexpr
///			dynamic_mdarray.size is variable
///		2. algorithm loop count
///			static_mdarray.agloop may be constexpr
///			dynamic_mdarray.agloop is variable
/// 
///Reference
#if 0
struct best_simd_optimized {
	__m128 data[2];

	best_simd_optimized() = default;

/// Only this constructor must inplace .
	best_simd_optimized(const __m128& d1, const __m128& d2) : data{ d1, d2 } {}

/// Other constructors are usually not need inplace .
	best_simd_optimized(const float& value) : data{ _mm_set1_ps(value), _mm_set1_ps(value) } {}
	best_simd_optimized(std::initializer_list<float> ilist) : data{ _mm_loadu_ps(ilist.begin()), _mm_loadu_ps(ilist.begin() + 4)} {}
};

inline best_simd_optimized operator+(const best_simd_optimized& a, const best_simd_optimized& b) {
	return best_simd_optimized(_mm_add_ps(a.data[0], b.data[0]), _mm_add_ps(a.data[1], b.data[1]));
}

inline best_simd_optimized operator-(const best_simd_optimized& a, const best_simd_optimized& b) {
	return best_simd_optimized(_mm_sub_ps(a.data[0], b.data[0]), _mm_sub_ps(a.data[1], b.data[1]));
}

inline best_simd_optimized operator*(const best_simd_optimized& a, const best_simd_optimized& b) {
	return best_simd_optimized(_mm_mul_ps(a.data[0], b.data[0]), _mm_mul_ps(a.data[1], b.data[1]));
}

inline best_simd_optimized operator/(const best_simd_optimized& a, const best_simd_optimized& b) {
	return best_simd_optimized(_mm_div_ps(a.data[0], b.data[0]), _mm_div_ps(a.data[1], b.data[1]));
}

int main() {
	best_simd_optimized a = { (float)rand(),(float)rand(),(float)rand(),(float)rand(),(float)rand(),(float)rand(),(float)rand() },
		b = { (float)rand(),(float)rand(),(float)rand(),(float)rand(),(float)rand(),(float)rand(),(float)rand() },
		c = { (float)rand(),(float)rand(),(float)rand(),(float)rand(),(float)rand(),(float)rand(),(float)rand() },
		d = { (float)rand(),(float)rand(),(float)rand(),(float)rand(),(float)rand(),(float)rand(),(float)rand() },
		e = { (float)rand(),(float)rand(),(float)rand(),(float)rand(),(float)rand(),(float)rand(),(float)rand() };

	auto start_time = std::chrono::steady_clock::now();
	for (size_t i = 0; i < 100000000; i++) {
		a = a + b;
		a = a * c;
		a = a / d;
		a = a - e;
		a = a * best_simd_optimized(5.5f);
	}
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time) << std::endl;

	std::cout << ((float*)a.data)[0] << ((float*)a.data)[1];
	std::cout << ((float*)a.data)[2] << ((float*)a.data)[3];
	std::cout << ((float*)a.data)[4] << ((float*)a.data)[5];
	std::cout << ((float*)a.data)[6] << ((float*)a.data)[7];
	return 0;
}
#endif

	template<typename value_type, typename size_type, typename allocator>
	constexpr bool operator==(const mdarray_view<value_type, size_type, allocator>& x, const mdarray_view<value_type, size_type, allocator>& y) {
		// Compare dimension.(structure)
		if (x.dimension() != y.dimension())
			return false;
		else if (x.dimension() == 1) {
			// Compare size.(structure)
			if (x.size(0) != y.size(0))
				return false;
			// Compare each element.
			for (size_t i = 0, iend = x.size(0); i != iend; ++i)
				if (x[i] != y[i])
					return false;
		} else {
			// Compare size.(structure)
			for (size_t i = 0; i != x.dimension(); ++i)
				if (x.size(i) != y.size(i))
					return false;

			// Compare each element.
			std::remove_cvref_t<decltype(x.size())> pos;
			for (size_t i = 0; i != x.dimension(); ++i) 
				pos[i] = 0;
			bool nextline = true;
			while (nextline) {
				size_t linebegin = pos[0];
				for (size_t i = 1; i != x.dimension(); ++i) 
					linebegin += pos[i] * x.stride(i);
				for (size_t i = 0, iend = x.size(0); i != iend; ++i) {
					size_t index = linebegin + i;
					if (x[index] != y[index])
						return false;
				}

				size_t carry = 1;
				while (++pos[carry] == x.size(carry)) {
					if (carry+1 == x.dimension()) {
						nextline = false;
						break;
					}
					pos[carry++] = 0;
				}
			}
		}

		return true;
	}

	template<typename value_type, auto size, typename package_type>
	constexpr bool operator==(const smdarray<value_type, size, package_type>& x, const smdarray<value_type, size, package_type>& y) {
		using smdarray_type = smdarray<value_type, size, package_type>;
		if constexpr (smdarray_type::dimension() == 1) {
			// Compare each element.
			for (size_t i = 0, iend = smdarray_type::size(0); i != iend; ++i)
				if (x[i] != y[i])
					return false;
		} else {
			// Compare each element.
			std::remove_cvref_t<decltype(x.size())> pos;
			for (size_t i = 0; i != smdarray_type::dimension(); ++i)
				pos[i] = 0;
			bool nextline = true;
			while (nextline) {
				size_t linebegin = pos[0];
				for (size_t i = 1; i != smdarray_type::dimension(); ++i)
					linebegin += pos[i] * x.stride(i);
				for (size_t i = 0, iend = x.size(0); i != iend; ++i) {
					size_t index = linebegin + i;
					if (x[index] != y[index])
						return false;
				}

				size_t carry = 1;
				while (++pos[carry] == x.size(carry)) {
					if (carry+1 == smdarray_type::dimension()) {
						nextline = false;
						break;
					}
					pos[carry++] = 0;
				}
			}
		}

		return true;
	}

	template<typename value_type, typename size_type, typename allocator, /*std::equality_comparable_with<value_type>*/typename value_type2>
	constexpr bool operator==(const mdarray_view<value_type, size_type, allocator>& x, const value_type2& yval) {
		if (x.empty())
			return false;
		else if (x.dimension() == 1) {
			// Compare each element.
			for (size_t i = 0, iend = x.size(0); i != iend; ++i)
				if (x[i] != yval)
					return false;
		} else {
			// Compare each element.
			std::remove_cvref_t<decltype(x.size())> pos;
			for (size_t i = 0; i != x.dimension(); ++i) 
				pos[i] = 0;
			bool nextline = true;
			while (nextline) {
				size_t linebegin = pos[0];
				for (size_t i = 1; i != x.dimension(); ++i) 
					linebegin += pos[i] * x.stride(i);
				for (size_t i = 0, iend = x.size(0); i != iend; ++i) {
					size_t index = linebegin + i;
					if (x[index] != yval)
						return false;
				}

				size_t carry = 1;
				while (++pos[carry] == x.size(carry)) {
					if (carry+1 == x.dimension()) {
						nextline = false;
						break;
					}
					pos[carry++] = 0;
				}
			}
		}

		return true;
	}

	template<typename value_type, auto size, typename package_type, /*std::equality_comparable_with<value_type>*/typename value_type2>
	constexpr bool operator==(const smdarray<value_type, size, package_type>& x, const value_type2& yval) {
		using smdarray_type = smdarray<value_type, size, package_type>;
		if constexpr (smdarray_type::dimension() == 1) {
			// Compare each element.
			for (size_t i = 0, iend = smdarray_type::size(0); i != iend; ++i)
				if (x[i] != yval)
					return false;
		} else {
			// Compare each element.
			std::remove_cvref_t<decltype(x.size())> pos;
			for (size_t i = 0; i != smdarray_type::dimension(); ++i)
				pos[i] = 0;
			bool nextline = true;
			while (nextline) {
				size_t linebegin = pos[0];
				for (size_t i = 1; i != smdarray_type::dimension(); ++i)
					linebegin += pos[i] * x.stride(i);
				for (size_t i = 0, iend = x.size(0); i != iend; ++i) {
					size_t index = linebegin + i;
					if (x[index] != yval)
						return false;
				}

				size_t carry = 1;
				while (++pos[carry] == x.size(carry)) {
					if (carry+1 == smdarray_type::dimension()) {
						nextline = false;
						break;
					}
					pos[carry++] = 0;
				}
			}
		}

		return true;
	}

	template<typename elem, typename traits,
		typename value_type, auto size, typename package_type>
	std::basic_ostream<elem,traits>& operator<<(std::basic_ostream<elem,traits>& ostr, const smdarray<value_type, size, package_type>& x) {
		using smdarray_type = smdarray<value_type, size, package_type>;
		if constexpr (smdarray_type::dimension() == 1) {
			ostr << elem('[');
			for (size_t i = 0, iend = smdarray_type::size(0); i != iend; ++i) {
				ostr << x[i];
				if (i + 1 != iend)
					ostr << elem(',');
			}
			return (ostr << elem(']'));
		} else {
			ostr << elem('[');
			std::remove_cvref_t<decltype(x.size())> pos;
			for (size_t i = 0; i != smdarray_type::dimension(); ++i)
				pos[i] = 0;
			bool nextline = true;
			while (nextline) {
				ostr << elem('[');
				size_t linebegin = pos[0];
				for (size_t i = 1; i != smdarray_type::dimension(); ++i)
					linebegin += pos[i] * x.stride(i);
				for (size_t i = 0, iend = x.size(0); i != iend; ++i) {
					size_t index = linebegin + i;
					ostr << x[index];
					if (i + 1 != iend)
						ostr << elem(',');
				}

				size_t carry = 1;
				while (++pos[carry] == x.size(carry)) {
					if (carry+1 == smdarray_type::dimension()) {
						nextline = false;
						break;
					}
					pos[carry++] = 0;
				}
				ostr << elem(']');
				if (nextline)
					ostr << elem(',');
			}
			return (ostr << elem(']'));
		}
	}

	template<typename value_type, typename size_type, typename allocator>
	constexpr bool operator!=(const mdarray_view<value_type, size_type, allocator>& x, const mdarray_view<value_type, size_type, allocator>& y) {
		return !(x == y);
	}

	template<typename value_type, auto size, typename package_type>
	constexpr bool operator!=(const smdarray<value_type, size, package_type>& x, const smdarray<value_type, size, package_type>& y) {
		return !(x == y);
	}

	template<typename value_type, typename size_type, typename allocator, /*std::equality_comparable_with<value_type>*/typename value_type2>
	constexpr bool operator!=(const mdarray_view<value_type, size_type, allocator>& x, const value_type2& yval) {
		return !(x == yval);
	}

	template<typename value_type, auto size, typename package_type, /*std::equality_comparable_with<value_type>*/typename value_type2>
	constexpr bool operator!=(const smdarray<value_type, size, package_type>& x, const value_type2& yval) {
		return !(x == yval);
	}
}// end of namespace math

template<size_t dimension>
using mdsize_t = math::smdsize_t<dimension>;

template<size_t dimension>
using mdintptr_t = math::smdintptr_t<dimension>;

#ifndef unroll4x_for
#define unroll4x_for(init_statement, condition4x, iteration4x, condition, iteration, statement) \
	init_statement; \
	for ( ; condition4x; iteration4x) { \
		{ constexpr int i = 0; statement } \
		{ constexpr int i = 1; statement } \
		{ constexpr int i = 2; statement } \
		{ constexpr int i = 3; statement } \
	} \
	for ( ; condition; iteration) { \
		{ constexpr int i = 0; statement } \
	}
#endif

#ifndef iterator_based_unroll4x_for
#define iterator_based_unroll4x_for(constexpr_, count, main_iterator, init_statement, iteration4x, iteration, statement) \
	init_statement; \
	constexpr_ auto PRE = ((count >> 2) << 2); \
	if constexpr_(PRE != 0) { \
		const auto main_iterator_end = std::next(main_iterator, PRE); \
		do { \
			{ constexpr size_t i = 0; statement } \
			{ constexpr size_t i = 1; statement } \
			{ constexpr size_t i = 2; statement } \
			{ constexpr size_t i = 3; statement } \
			iteration4x; \
		} while (main_iterator != main_iterator_end); \
	} \
	for(auto REM = count - PRE; REM != 0; --REM) { \
		{ constexpr size_t i = 0; statement } \
		iteration; \
	}
	/*if constexpr_(REM != 0) { \
		const auto main_iterator_end = std::next(main_iterator, REM); \
		do { \
			{ constexpr size_t i = 0; statement } \
			iteration; \
		} while (main_iterator != main_iterator_end); \
	}*/
	/// This REM method is not suitable for optimization.
#endif

#ifndef epconj
/// Expression Conjunction.
#define epconj(...) __VA_ARGS__
#endif