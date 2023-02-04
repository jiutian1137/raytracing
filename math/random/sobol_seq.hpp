#pragma once

/// Random generation of Sobol Sequence.
///@license Free 
///@review 2022-10-9 
///@contact Jiang1998Nan@outlook.com 
#define _MATH_RANDOM_SOBOL_SEQUENCE_

#include <concepts>// for automatic type conversion in sobol_seq::generate(..).
#include <vector>/// dynamic_bitset
#include <bitset>/// static_bitset

namespace math {
	/// Low-discrepancy random, it's base 2.
	template<typename integer>
	struct sobol_seq {
		integer seed = 0;

		struct _Dirnum {
			static constexpr size_t size() {
				return sizeof(integer) * 8 - std::is_signed_v<integer>;
			}

			integer _Arr[size()];

			integer& operator[](size_t i) {
				return _Arr[i];
			}

			const integer& operator[](size_t i) const {
				return _Arr[i];
			}
		};
		std::vector<_Dirnum> direction_numbers;
		
		///@brief 
		/// Sobol's direction numbers recurrence, 
		/// computing sequence may not require this recursion.
		/// 
		///@theory
		/// polynomial is form of
		/// 
		///		polynomials[i] = pow(x,d)*polynomials[i][0] + pow(x,d-1)*polynomials[i][1] 
		///			+ ... + pow(x,1)*polynomials[i][d-1] + pow(x,0)*polynomials[i][d] .
		///   (example: polynomials[X] = {1, 0, 1, 1} = pow(x,3)*1 + pow(x,2)*0 + pow(x,1)*1 + pow(x,0)*1 .)
		/// 
		/// direction numbers is derivative from polynomial, is form of
		/// 
		///		...
		///		<<On the distribution of points in a cube and the approximate evaluation of integrals>>
		///		<<Algorithm 659 Implementing Sobol's Quasirandom Sequence Generator>>
		/// 
		/// direction numbers recurrence is form of
		/// 
		///		direction_numbers[i][X] = XORdot(polynomials, direction_numbers[i][[prev, X)]) .
		///		(initial values list:
		///		 joe-kuo-old.1111 contains the direction numbers up to dimension 1111.
		///		 joe-kuo-other-0.7600 contains some direction numbers up to dimension 7600.
		///		 joe-kuo-other-2.3900 contains some direction numbers up to dimension 3900.
		///		 joe-kuo-other-3.7300 contains some direction numbers up to dimension 7300.
		///		 joe-kuo-other-4.5600 contains some direction numbers up to dimension 5600.
		///		 new-joe-kuo-5.21201 contains the direction numbers obtained using the search criterion D(5) up to dimension 21201.
		///		 new-joe-kuo-6.21201 contains the direction numbers obtained using the search criterion D(6) up to dimension 21201.
		///			 polynomials[1] = {1,1}; direction_numbers.subrow(1, 0,1) = {1/*, 3, 5, 15, 17, 51, 85, 255, 257, 771, ...*/};
		///			 polynomials[2] = {1,1,1}; direction_numbers.subrow(2, 0,2) = {1, 3/*, 3, 9, 29, 23, 71, 197, 209, 627, ...*/};
		///			 polynomials[3] = {1,0,1,1}; direction_numbers.subrow(3, 0,3) = {1, 3, 1/*, 5, 31, 29, 81, 147, 433, 149, ...*/};
		///			 polynomials[4] = {1,1,0,1}; direction_numbers.subrow(4, 0,3) = {1, 1, 1/*, 11, 31, 55, 61, 157, 181, 191, ...*/};
		///			 polynomials[5] = {1,0,0,1,1}; direction_numbers.subrow(5, 0,4) = {1, 1, 3, 3/*, 25, 9, 43, 251, 449, 449, ...*/};
		///			 polynomials[6] = {1,1,0,0,1}; direction_numbers.subrow(6, 0,4) = {1, 3, 5, 13};
		///			 ...
		///		 new-joe-kuo-7.21201 contains the direction numbers obtained using the search criterion D(7) up to dimension 21201.)
		/// 
		///		<<Remark on Algorithm 659: Implementing Sobol's Quasirandom Sequence Generator>>
		///		<<Constructing Sobol sequences with better two-dimensional projections>>
		template<typename bitset>
		static void direction_numbers_recurrence(const std::vector<bitset>& polynomials, std::vector<_Dirnum>& direction_numbers, bool compute_v = true) {
				for (size_t i = 0, iend = direction_numbers[0].size(); i != iend; ++i) {
					direction_numbers[0][i] = 1;
				}
			for (size_t s = 1; s != direction_numbers.size(); ++s) {
				const bitset& a = polynomials[s];
				_Dirnum&      m = direction_numbers[s];
				/// m[i] = pow(2,1)*a[1]*m[i-1] XOR pow(2,2)*a[2]*m[i-2] XOR ... XOR pow(2,d-1)*a[d-1]*m[i-(d-1)] XOR
				///        pow(2,d)*m[i-d] XOR m[i-d] .
				const size_t degree = a.size() - 1;
				for (size_t i = degree; i < m.size(); ++i) {
					integer m_i = 0;
					integer l = 2;
					for (size_t k = 1; k < degree; ++k, l*=2)
						m_i ^= (l * a[k] * m[i-k]);
					m_i ^= (l * m[i-degree]);
					m_i ^= m[i-degree];
					m[i] = m_i;
				}
			}

			if (compute_v) {
				for (size_t s = 0; s != direction_numbers.size(); ++s) {
					_Dirnum& m = direction_numbers[s];
					/// v[i] = m[i]/pow(2,i).
					size_t bits = m.size();
					for (size_t i = 0; i != m.size(); ++i) {
						m[i] = m[i]*(integer(1)<<(bits-1-i));
					}
				}
			}
		}

		explicit sobol_seq(integer _seed = 0) : seed(_seed),
			direction_numbers{
				_Dirnum{},
				_Dirnum{1/*, 3, 5, 15, 17, 51, 85, 255, 257, 771, ...*/},
				_Dirnum{1, 3/*, 3, 9, 29, 23, 71, 197, 209, 627, ...*/},
				_Dirnum{1, 3, 1/*, 5, 31, 29, 81, 147, 433, 149, ...*/},
				_Dirnum{1, 1, 1/*, 11, 31, 55, 61, 157, 181, 191, ...*/},
				_Dirnum{1, 1, 3, 3/*, 25, 9, 43, 251, 449, 449, ...*/},
				_Dirnum{1, 3, 5, 13} } 
		{
			std::vector<std::vector<bool>> polynomials{
				std::vector<bool>{},
				std::vector<bool>{1,1},
				std::vector<bool>{1,1,1},
				std::vector<bool>{1,0,1,1},
				std::vector<bool>{1,1,0,1},
				std::vector<bool>{1,0,0,1,1},
				std::vector<bool>{1,1,0,0,1} };
			direction_numbers_recurrence(polynomials, direction_numbers, true);
		}

		sobol_seq(const std::vector<std::vector<bool>>& polynomials, const std::vector<_Dirnum>& direction_numbers_initial_value, integer _seed = 0)
			: direction_numbers(direction_numbers_initial_value), seed(_seed) {
			direction_numbers_recurrence(polynomials, direction_numbers, true);
		}

		template<typename result_type>
		void generate(size_t dims, result_type* results) {
			integer n = seed ^ (seed >> 1);
		
			for (size_t i = 0; i != dims; ++i) {
				integer seq = 0;
				for (size_t j = 0, jend = direction_numbers[i].size(); j != jend; ++j) {
					seq ^= ((n>>j)&1)*direction_numbers[i][j];
				}

				if constexpr (std::is_same_v<result_type, integer>) {
					results[i] = seq;
				} else if constexpr (std::is_floating_point_v<result_type>) {
					results[i] = seq/exp2(static_cast<result_type>(_Dirnum::size()));
				} else {
					abort();
				}
			}

			++seed;
		}

		template<typename result_type = integer>
		result_type generate() {
			result_type result;
			if constexpr (std::is_scalar_v<result_type>) {
				generate(1, &result);
			} else if constexpr (std::is_array_v<result_type>) {
				generate(sizeof(result)/sizeof(result[0]), result);
			} else {
				generate(result.length(), result.begin());
			}

			return result;
		}

		size_t dimensions() const {
			return direction_numbers.size();
		}
	};
}