#pragma once

///@brief Multi-dimensional Pyramid Structure 
///@license Free 
///@review 2023-01-03 
///@contact Jiang1998Nan@outlook.com 
#define _MATH_MDARRAY_PYRAMID_
#include "mdarray.hpp"

namespace math {
	template<typename __value_type, typename __size_type, typename __allocator_type = std::allocator<__value_type>>
	struct mdarray_pyramid_view {
		using layer_view = mdarray_view<__value_type, __size_type, __allocator_type>;
		using value_type      = typename layer_view::value_type;
		using allocator_type  = typename layer_view::allocator_type;
		using package_type    = typename layer_view::package_type;
		using size_type       = typename layer_view::size_type;
		using length_type     = typename layer_view::length_type;
		using dimension_type  = typename layer_view::dimension_type;

		layer_view *__views;
		length_type __size;

		constexpr explicit mdarray_pyramid_view(layer_view* views = nullptr, length_type size = 0) noexcept : __views(views), __size(size) {}

		length_type size() const {
			return __size;
		}

		layer_view& operator[](length_type i) {
			return __views[i];
		}

		const layer_view& operator[](length_type i) const {
			return __views[i];
		}
	};

	template<typename __value_type, typename __size_type, typename __allocator_type = std::allocator<__value_type>>
	struct mdarray_pyramid : mdarray_pyramid_view<__value_type, __size_type, __allocator_type> {
		using __base = mdarray_pyramid_view<__value_type, __size_type, __allocator_type>;
		using __base::__views;
		using __base::__size;

		using layer_view = mdarray_view<__value_type, __size_type, __allocator_type>;
		using value_type      = typename layer_view::value_type;
		using allocator_type  = typename layer_view::allocator_type;
		using package_type    = typename layer_view::package_type;
		using size_type       = typename layer_view::size_type;
		using length_type     = typename layer_view::length_type;
		using dimension_type  = typename layer_view::dimension_type;

		allocator_type _Al;

		constexpr mdarray_pyramid() noexcept = default;

		explicit mdarray_pyramid(const size_type& the_layer0_size, const length_type& the_size = 1) : __base() {
			__views = new layer_view[the_size];

			__views[0].__size = the_layer0_size;
			__views[0].__package_length = __vector_align(layer_view::simdlen, the_layer0_size,
				__views[0].__stride) / layer_view::simdlen;
			length_type the_package_length = __views[0].__package_length;
			if (the_size > 1) {
				length_type i = 1;
				do {
					//__views[i].__data = reinterpret_cast<package_type*>(static_cast<size_t>(the_package_length));
					__views[i].__size = __views[i - 1].__size / 2;
					__views[i].__package_length = __vector_align(layer_view::simdlen, __views[i].__size,
						__views[i].__stride) / layer_view::simdlen;
					the_package_length += __views[i].__package_length;
				} while (++i != the_size);
			}

			__views[0].__data = _Al.allocate(the_package_length);
			try {
				if (the_size > 1) {
					length_type i = 1;
					do {
						__views[i].__data = __views[i-1].__data + __views[i-1].__package_length;
					} while (++i != the_size);
				}
			} catch (const std::exception&) {
				_Al.deallocate(__views[0].__data, the_package_length);
				delete[] __views;
				__views = nullptr;
				throw;
			}

			__size = the_size;
		}

		~mdarray_pyramid() noexcept {
			if (__views != nullptr && __views[0].__data != nullptr) {
				length_type the_package_length = 0;
				for (length_type i = 0; i != __size; ++i) {
					std::destroy_n(__views[i].__data, __views[i].__package_length);
					the_package_length += __views[i].__package_length;
				}
				_Al.deallocate(__views[0].__data, the_package_length);

				for (length_type i = 0; i != __size; ++i) {
					__views[i].__data   = nullptr;
					__views[i].__package_length = 0;
					__views[i].__size   = size_type();
					__views[i].__stride = size_type();
				}
			}
		}
	};
}// end of namespace math