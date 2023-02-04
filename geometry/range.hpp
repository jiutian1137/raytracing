#pragma once

///		namespace std { namespace ranges {
/// 
/// Because:
/// (1) The constraints of std::ranges::range is too strong, we cannot use it for 'numeric_range'.
/// (2) The constraints of std::ranges::range is necessary, it can't be weakened.
/// So we need to a new constraints, we don't need to rely on std::ranges. These operations should
/// originally be in geometry|topology when they not depend on std::ranges.
/// 
namespace geometry {
///		template<class _Rng>
///		concept range = requires(_Rng& __r) {
///			std::begin(__r);
///			std::end(__r);
///		};
///
///		template<range _Rng>
///		auto range_expand(const _Rng& _Left/*, [_Left)*/, const _Rng& _Right/*, [_Right)*/) {
///			...
///		}
/// 
/// 	std::string A = "Hellow World!";
///		std::cout << geometry::range_expand(
///			std::string_view(A.begin(), A.begin()+5), 
///			std::string_view(A.begin()+5, A.begin()+10) ) << std::endl;
/// 
///		!!!Abort "cannot compare incompatible string_view iterators".
/// 
/// A new constraints does not work, then because:
/// (1) A strict view must have such assertions, so we cannot use std::*_view or std::container do
///     range_operation.
/// (2) A strict structure is desirable, so we cannot use any view or any container do range_oper-
///     ation.
/// So we need a new structure for weak assertion, and only it can do range_operation.
/// 
	template<typename _Ty>
	struct range {
		_Ty l;
		_Ty u;

		_Ty begin() const {
			return l;
		}

		_Ty end() const {
			return u;
		}

		/*bool empty() const {
			return !(l < u);
		}*/

		auto size() const {
			return u - l;
		}

		static range empty_range() {
			return { static_cast<_Ty>(1), static_cast<_Ty>(0) };
			//return { std::numeric_limits<_Ty>::quiet_NaN(), std::numeric_limits<_Ty>::quiet_NaN()};
		}

		static range from(const _Ty& val0, const _Ty& val1) {
			using std::min, std::max;
			return { min(val0,val1), max(val0,val1) };
		}

		static range from(const _Ty& val0, const _Ty& val1, const _Ty& val2) {
			using std::min, std::max;
			return { min(min(val0,val1),val2), max(max(val0,val1),val2) };
		}
	};

	template<typename _Ty>
	bool empty(const range<_Ty>& _Rng) {
		return !(_Rng.l < _Rng.u);
	}

	template<typename _Ty>
	_Ty center(const range<_Ty>& _Rng) {
		return (std::begin(_Rng) + std::end(_Rng))/2;
	}

	template<typename _Ty>
	bool equals(const range<_Ty>& _Left/*, [_Left)*/, const range<_Ty> _Right/*, [_Right)*/) {
		return ( std::begin(_Left) == std::begin(_Right) && std::end(_Left) == std::end(_Right) ) ||
			( std::empty(_Left) && std::empty(_Right) );
	}

	template<typename _Ty>
	bool disjoint(const range<_Ty>& _Left/*, [_Left)*/, const range<_Ty>& _Right/*, [_Right)*/) {
		return ( std::end(_Left) <= std::begin(_Right) || std::end(_Right) <= std::begin(_Left) ) || 
			empty(_Left) || empty(_Right);
	}

	template<typename _Ty>
	bool disjoint(const range<_Ty>& _Left/*, [_Left)*/, const _Ty& _Right) {
		return ( std::end(_Left) <= _Right || _Right <= std::begin(_Left) ) ||
			empty(_Left);
	}

	template<typename _Ty>
	bool before(const _Ty& _Right, const range<_Ty>& _Left/*, [_Left)*/) {
		return (_Right < std::begin(_Left) ) ||
			empty(_Left);
	}

	template<typename _Ty>
	bool intersects(const range<_Ty>& _Left/*, [_Left)*/, const range<_Ty>& _Right/*, [_Right)*/) {
		return !disjoint(_Left, _Right);
	}

	template<typename _Ty>
	bool intersects(const range<_Ty>& _Left/*, [_Left)*/, const _Ty& _Right) {
		return !disjoint(_Left, _Right);
	}

	template<typename _Ty>
	bool contains(const range<_Ty>& _Left/*, [_Left)*/, const range<_Ty>& _Right/*, [_Right)*/) {
		return ( std::begin(_Left) <= std::begin(_Right) && std::end(_Right) <= std::end(_Left) ) ||
			std::empty(_Right);
	}

	template<typename _Ty>
	bool contains(const range<_Ty>& _Left/*, [_Left)*/, const _Ty& _Right) {
		return ( std::begin(_Left) <= _Right && _Right < std::end(_Left) );
	}

	template<typename _Ty>
	bool whthin(const range<_Ty>& _Left/*, [_Left)*/, const range<_Ty>& _Right/*, [_Right)*/) {
		return contains(_Right, _Left);
	}
	
	template<typename _Ty>
	bool whthin(const _Ty& _Left, const range<_Ty>& _Right/*, [_Right)*/) {
		return contains(_Right, _Left);
	}

	/*template<typename _Ty>
	bool crosses(const range<_Ty>& x, const range<_Ty>& y) {
		return (y.l <= x.l && x.l < y.u) || (y.l < x.u && x.u <= y.u);
	}*/

	template<typename _Ty>
	range<_Ty> intersection(const range<_Ty>& _Left/*, [_Left)*/, const range<_Ty>& _Right/*, [_Right)*/) {
		return { std::max(std::begin(_Left), std::begin(_Right)), std::min(std::end(_Left), std::end(_Right)) };
	}
	
	template<typename _Ty>
	range<_Ty> expand(const range<_Ty>& _Left/*, [_Left)*/, const range<_Ty>& _Right/*, [_Right)*/) {
		/*if (empty(_Left)) {
			return _Right;
		} else if (empty(_Right)) {
			return _Left;
		} else {*/
			using std::min, std::max;
			return { min(std::begin(_Left), std::begin(_Right)), max(std::end(_Left), std::end(_Right)) };
		//}
	}

	template<typename _Ty>
	range<_Ty> expand(const range<_Ty>& _Left/*, [_Left)*/, const _Ty& _Right) {
		using std::min, std::max;
		return { min(std::begin(_Left), _Right), max(std::end(_Left), _Right) };
	}

	template<typename _Ty>
	size_t union_(const range<_Ty>& _Left/*, [_Left)*/, const range<_Ty>& _Right/*, [_Right)*/, range<_Ty>* _Results) {
		if (std::empty(_Left)) {
			_Results[0] = _Right;
			return 1;
		} else if (std::empty(_Right)) {
			_Results[0] = _Left;
			return 1;
		} else {
			range<_Ty> _Expand = { std::min(std::begin(_Left), std::begin(_Right)), std::max(std::end(_Left), std::end(_Right)) };
			if (std::size(_Left) + std::size(_Right) < std::size(_Expand)) {
				if (std::begin(_Left) < std::begin(_Right)) {
					_Results[0] = _Left;
					_Results[1] = _Right;
				} else {
					_Results[0] = _Right;
					_Results[1] = _Left;
				}
				return 2;
			} else {
				_Results[0] = _Expand;
				return 1;
			}
		}
	}

	template<typename _Ty>
	size_t difference(const range<_Ty>& _Left/*, [_Left)*/, const range<_Ty>& _Right/*, [_Right)*/, range<_Ty>* _Results) {
		if (std::empty(_Right)) {
			_Results[0] = _Left;
			return 1;
		} else {
			if (std::begin(_Left) < std::begin(_Right)) {
				if (std::end(_Right) < std::end(_Left)) {
					_Results[0] = { std::begin(_Left), std::begin(_Right) };
					_Results[1] = { std::end(_Right), std::end(_Left) };
					return 2;
				} else {
					_Results[0] = { std::begin(_Left), std::min(std::end(_Left), std::begin(_Right)) };
					return 1;
				}
			} else {
				_Results[0] = { std::max(std::begin(_Left), std::end(_Right)), std::max(std::end(_Left), std::end(_Right)) };
				return 1;
			}
		}
	}

#if 0
	std::default_random_engine rng{ std::random_device{}() };
	std::uniform_int_distribution<int> dis(-9, 10);
	for (size_t i = 0; i != 10; ++i) {
		geometry::range range1{ dis(rng), dis(rng) };
		geometry::range range2{ dis(rng), dis(rng) };
		if (range1.l > range1.u) { std::swap(range1.l, range1.u); }
		if (range2.l > range2.u) { std::swap(range2.l, range2.u); }

		for (int k = -9; k < 10; ++k) {
			std::cout << abs(k);
		}
		std::cout << std::endl;

		for (int k = -9; k < 10; ++k) {
			if (geometry::contains(range1, k))
				std::cout << "1";
			else std::cout << "0";
		}
		std::cout << "\t[" << range1.begin() << "," << range1.end() << ")" << std::endl;

		for (int k = -9; k < 10; ++k) {
			if (geometry::contains(range2, k))
				std::cout << "2";
			else std::cout << "0";
		}
		std::cout << "\t[" << range2.begin() << "," << range2.end() << ")" << std::endl;

		for (int k = -9; k < 10; ++k) {
			if (geometry::contains(range1, k) || geometry::contains(range2,k))
				std::cout << "D";
			else std::cout << "0";
		}

		/*std::cout << (geometry::range_contains(range1, range2) ? "contains" : "whthout") << std::endl;*/

		/*auto range3 = geometry::range_intersection(range1, range2);
		std::cout << "\t[" << range3.begin() << "," << range3.end() << ")" << std::endl;*/
		
		decltype(range1) rangeN[2];
		if (geometry::union_(range1, range2, rangeN) == 2) {
			std::cout << "\t[" << rangeN[0].begin() << "," << rangeN[0].end() << ") & [" << rangeN[1].begin() << "," << rangeN[1].end() << ")" << std::endl;
		} else {
			std::cout << "\t[" << rangeN[0].begin() << "," << rangeN[0].end() << ")" << std::endl;
		}

		std::cout << std::endl;
	}
#endif

	template<typename _Ty>
	struct in_range : range<_Ty> {
		template<typename _Ty2>
		bool operator()(const _Ty2& _Right) const {
			return contains(*this, _Right);
		}
	};
}