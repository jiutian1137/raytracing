#pragma once

///@brief Integral of Number.
///@license Please Identify Mathmatician and Reference 
///@review 2022-7-21 
///@contact Jiang1998Nan@outlook.com 
///@reference
/// { "Book":"Numerical Analysis", "Author":"Timothy Sauer" }
/// { "Book":"Harmonia Mensurarum", "Author":"..." }
/// { "URL":"https://people.sc.fsu.edu/~sshanbhag/NumericalIntegration.pdf", "Author":"...", "Desc":"some example" }
/// { "URL":"http://homepage.math.uiowa.edu/~whan/3800.d/S5-2.pdf", "Author":"?", "Desc":"correct_polynomial_x" }
/// "Rimemann Sum":{ 
///		"Mathematician":"Georg Friedrich Bernhard Riemann"
/// }
///		
/// "Interpolating Polynomial Integration":{
///		"Mathematician":[
///			"Isaac Newton",
///			"Roger Cotes",
///			"Thomas Simpson",
///			"George Boole",
///			"Abramowitz ...",
///			"Stegun ...",
///			"Ueberhuber ..."
///		],
///		"Formula":"https://mathworld.wolfram.com/Newton-CotesFormulas.html"
/// }
///
/// "Gauss Integration":{
///		"Mathematician":[
///			"Johann Carl Friedrich Gauss"
///		],
///		"Formula":"https://mathworld.wolfram.com/GaussianQuadrature.html"
/// }
///
/// "MonteCarlo Integration":{ }
#define _MATH_NUMBER_INTEGRAL_

#include <cassert>

#include <numeric>
#include <math/concepts.hpp>

namespace math {
#ifndef __calculation_sum__
#define __calculation_sum__
	template<typename Integer, typename Function>
	inline auto sum(Integer start, Integer end, Function f) -> decltype(f(start) + f(end)) {
		auto result = f(start);
		for (Integer i = start + 1; i <= end; ++i) {
			result += f(i);
		}

		return std::move(result);
	}
#endif

	template<typename type, typename scalar>
	concept linearly_combinable = requires(type a, scalar s) {
		a += a*s;
		a -= a*s;
		a *= s;
		a /= s;
		(a*s + a*s - a*s)/s; 
	};

	/// |   |   |   |   |     |   |
	/// x   x   x   x   x ... x   |
	/// |   |   |   |   |     |   | 
	template<typename Real, typename Function>
		//requires linearly_combinable<decltype(Function{}(Real{})), Real>
	inline auto upper_sum(Real a, Real b, Function f, size_t n) -> decltype(f(a)) {
		Real dx = (b - a)/Real(n);
	
		auto result = static_cast<decltype(f(a))>(0);
		for (size_t i = 0; i != n; ++i) {
			result += f(a + Real(i)*dx);
		}
		return result * dx;
	}

	/// |   |   |   |   |     |   |
	/// |   x   x   x   x ... x   x
	/// |   |   |   |   |     |   | 
	template<typename Real, typename Function>
	inline auto lower_sum(Function f, Real a, Real b, size_t n) -> decltype(f(a)) {
		Real dx = (b - a) / n;
	
		auto result = static_cast<decltype(f(a))>(0);
		for (size_t i = 0; i != n; ++i) {
			result += f(a + (i+1)*dx) * dx;
		}
		return result;
	}

	/// |   |   |   |   |     |   |
	/// | x | x | x | x | ... | x | 
	/// |   |   |   |   |     |   | 
	template<typename Real, typename Function>
	inline auto middle_sum(Real a, Real b, Function f, size_t n) -> decltype(f(a)) {
		Real dx = (b - a) / n;

		auto result = static_cast<decltype(f(a))>(0);
		for (size_t i = 0; i != n; ++i) {
			result += f( a + (i*dx + (i+1)*dx)/2 ) * dx;
		}
		return result;
	}


	/// first order interpolating polynomial composite integration. ( Trapozoidal )
	///@formula
	///	integral( 'a', 'b', (y0 + y1)*(h/2) ) + Error ~= sum( '0', 'n-1', (y0 + y1)*(h/2) )
	///	                                              ~= ( f('a') + f('b') + sum(1, ('n'+1)-1, f('a'+dx*i)) * 2 ) * h/2
	///	a-----|-----|-----b
	///	|--dx-|--h--|
	/// 
	///@theory interpolating polynomial: in [x0,x1], f(x) = y0*(x-x1)/(x0-x1) + y1*(x-x0)/(x1-x0) + Error
	///	integral( x0,x1, f(x) ) = integral( x0,x1, y0*(x-x1)/(x0-x1) + y1*(x-x0)/(x1-x0) + Error )
	///	                        = y0*((x1-x0)/2) + y1*((x1-x0)/2) + Error
	///	                        = trapozoidal_area + Error
	template<typename Real, typename Function>
	inline auto p1quad(Real a, Real b, Function f, size_t segments = 128) -> decltype(f(a)) {
		Real dx = (b - a) / segments;
		return ( 
				( f(a) + f(b) )/2 
			+ sum(size_t(1),segments-1,[&](size_t i){ return f(a+dx*i); })/* sum_y0y2_overlap/2 */ ) * dx;
	}

	/// second order interpolating polynomial composite integration. ( Simpson )
	///@formula 
	///	integral<x=a,b>( f(x)*dx ) = sum<i=0,n-1>( (y0 + y1*4 + y2)*h/3 ) + Error.
	///		( y0 = f(x[i]), y1 = f(x[i] + h), y2 = f(x[i] + 2*h), h = dx/2. )
	///@diagram 
	///		                                   y0 + y1*4 + y2
	///		                             ... + y2
	///		                        y0 + ... 
	///		            y0 + y1*4 + y2
	///		y0 + y1*4 + y2
	///		*-----+-----*-----+-----* ... ...  *-----+-----*
	///		|- - -dx - -|
	///		|- h -|
	template<typename Real, typename Function>
	inline auto p2quad(Real a, Real b, Function f, size_t segments = 64) -> decltype( f(a) ) {
		Real dx = (b - a) / segments;
		Real h = dx / 2;
		return ( f(a) + f(b)
			+ sum( size_t(0), segments-1, [&](size_t i){ return f(a + h + dx*i); } ) * 4 /* sum_y1*4 */
			+ sum( size_t(1), segments-1, [&](size_t i){ return f(a + dx*i); } ) * 2 /* sum_y0y2_overlap */
			) * h/3;
	}

	/// third order interpolating polynomial composite integration. ( Simpson 3/8 )
	///@formula 
	/// integral<x=a,b>( f(x)*dx ) = sum<i=0,n-1>( (y0 + y1*3 + y2*3 + y3)*h*3/8 ) + Error.
	///		( y0 = f(x[i]), y1 = f(x[i] + h), y2 = f(x[i] + 2*h), y3 = f(x[i] + 3*h), h = dx/3. )
	///@diagram 
	///		                                               y0 + y1*3 + y2*3 +y3
	///		                                         ...  +y3
	///		                                    y0 + ... 
	///		                  y0 + y1*3 + y2*3 +y3
	///		y0 + y1*3 + y2*3 +y3
	///		*-----+-----+-----*-----+-----+-----* ... ...  *-----+-----+-----*
	///		|- - - - dx- - - -|
	///		|- h -|
	template<typename Real, typename Function>
	inline auto p3quad(Real a, Real b, Function f, size_t n = 42) -> decltype( f(a) ) {
		Real dx = (b - a) / n;
		Real h = dx / 3;
		return ( f(a) + f(b)
			+ sum( size_t(0), n-1, [&](size_t i){ return f(a + h + dx*i); } ) * 3 /* sum_y1*3 */
			+ sum( size_t(0), n-1, [&](size_t i){ return f(a + h*2 + dx*i); } ) * 3 /* sum_y2*3 */
			+ sum( size_t(1), n-1, [&](size_t i){ return f(a + dx*i); } ) * 2 /* sum_y0y2_overlap */ 
			) * h*3/8;
	}

	/// fourth order interpolating polynomial composite integration. ( Boole )
	///@formula 
	/// integral<x=a,b>( f(x)*dx ) = sum<i=0,n-1>( (y0*7 + y1*32 + y2*12 + y3*32 + y4*7)*h*2/45 ) + Error.
	///		( y0 = f(x[i]), y1 = f(x[i] + h), y2 = f(x[i] + 2*h), y3 = f(x[i] + 3*h), y3 = f(x[i] + 4*h), h = dx/4. )
	///@diagram 
	///		                                                           y0*7+y1*32+y2*12+y3*32+ y4*7
	///		                                                     ... + y4*7
	///		                                                y0 + ...
	///		                        y0*7+y1*32+y2*12+y3*32+ y4*7
	///		y0*7+y1*32+y2*12+y3*32+ y4*7
	///		*-----+-----+-----+-----*-----+-----+-----+-----* ... ...  *-----+-----+-----+-----*
	///		|- - - - - -dx - - - - -|
	///		|- h -|
	template<typename Real, typename Function>
	inline auto p4quad(Real a, Real b, Function f, size_t n = 32) -> decltype( f(a) ) {
		Real dx = (b - a) / n;
		Real h = dx / 4;
		return (  
				(f(a) + f(b)) * 7
			+ sum( size_t(0), n-1, [&](size_t i){ return f(a + h + dx*i); } ) * 32 /* sum_y1*32 */
			+ sum( size_t(0), n-1, [&](size_t i){ return f(a + h*2 + dx*i); } ) * 12 /* sum_y2*12 */
			+ sum( size_t(0), n-1, [&](size_t i){ return f(a + h*3 + dx*i); } ) * 32 /* sum_y3*32 */
			+ sum( size_t(1), n-1, [&](size_t i){ return f(a + dx*i); } ) * 14 /* sum_y0y4_overlap*7 */
			) * h*2/45;
	}

	/// seventh order interpolating polynomial composite integration. ( Weddle ) 
	/// multipliers are integers, the only divisor is also integers, and very stable.
	///@formula 
	/// integral<x=a,b>( f(x)*dx ) = sum<i=0,n-1>( (y0 + y1*5 + y2 + y3*6 + y4 + y5*5 + y6)*h*3/10 ) + Error.
	///		( ... )
	///@diagram
	///		                                                                                   y0 + y1*5 + y2 + y3*6 + y4 + y5*5 + y6
	///		                                                                             ... + y6
	///		                                                                        y0 + ...
	///		                                    y0 + y1*5 + y2 + y3*6 + y4 + y5*5 + y6
	///		y0 + y1*5 + y2 + y3*6 + y4 + y5*5 + y6
	///		*-----+-----+-----+-----+-----+-----*-----+-----+-----+-----+-----+-----* ... ...  *-----+-----+-----+-----+-----+-----*
	///		|- - - - - - - - -dx - - - - - - - -|
	///		|- h -|
	template<typename Real, typename Function>
	inline auto p7quad(Real a, Real b, Function f, size_t n = 24) -> decltype( f(a) ) {
		Real dx = (b - a) / n;
		Real h = dx / 6;
		return (  
				f(a) + f(b)
			+ sum( size_t(0), n-1, [&](size_t i){ return f(a + h + dx*i); } ) * 5 /* sum_y1*5 */
			+ sum( size_t(0), n-1, [&](size_t i){ return f(a + h*2 + dx*i); } ) /* sum_y2 */
			+ sum( size_t(0), n-1, [&](size_t i){ return f(a + h*3 + dx*i); } ) * 6 /* sum_y3*6 */
			+ sum( size_t(0), n-1, [&](size_t i){ return f(a + h*4 + dx*i); } ) /* sum_y4 */
			+ sum( size_t(0), n-1, [&](size_t i){ return f(a + h*5 + dx*i); } ) * 5 /* sum_y5*5 */
			+ sum( size_t(1), n-1, [&](size_t i){ return f(a + dx*i); } ) * 2 /* sum_y0y6_overlap */ 
			) * h*3/10;
	}

	/* integral['a' -> 'b']('f'(x), 'dx'|'n')
	
	* integral( u, dv )
		= u*v - integral( v, du )
	
	* integral( pow(x,N), dx )
		= pow(x, N + 1)/(N + 1) ,apply anti-derivative: derivative( pow(x,N) ) = N*pow(x,N-1)
	
	* integral( 1/x, dx )
		= ln(abs(x)) ,apply anti-derivative: derivative( ln(x) ) = 1/x
	
	* integral( sin(x), dx )
		= -cos(x) ,apply anti-derivative: derivative( cos(x) ) = -sin(x)

	* integral( cos(x), dx )
		= sin(x) ,apply anti-derivative: derivative( sin(x) ) = cos(x)

	* integral( sin(x*A), dx )
		= integral( sin(x*A)*-A/-A, dx )
		= integral( sin(x*A)*-A, dx ) / -A
		= cos(x*A) / -A ,apply anti-derivative: derivative( cos(x*A) ) = -sin(x*A)*A
	 
	* integral( 1/(x*A + B), dx )
		= ln(abs(x*A + B))/A
	
	* integral( 1/pow(x + A, 2), dx ) = - 1/(x + A)
	
	* integral( pow(x + A, N), dx ) = pow(x + A, N + 1)/(N + 1)

	* Taylor theorem
						 f(x)                 fx(x)               fxx(x)                     fx...x(x)                          dN-1f(u)
	f(x+h) = ---------*pow(h,0) + ---------*pow(h,1) + ---------*pow(h,2) + ... + -----------*pow(h,N-1) + integral( -----------*pow(x+h-u,N-1), du )
						fact(0)              fact(1)              fact(2)                    fact(N-1)                          fact(N-1)

	*/
	template<typename Real, typename Function>
	inline auto quad(Real a, Real b, Function f, size_t n = 24) {
		return p7quad(a, b, f, n);
	}


	template<math::number scalar, 
		math::container container = std::vector<std::pair<scalar,scalar>>>
	struct quadrature : public container {
		template<typename functionty>
		auto operator()(const functionty& f) const {
			assert( !std::empty(*this) );
			const auto& _0 = (*this)[0];
			auto result = f(std::get<0>(_0)) * std::get<1>(_0);
			for (size_t i = 1, iend = static_cast<size_t>(std::size(*this)); i != iend; ++i) {
				const auto& _i = (*this)[i];
				result += f(std::get<0>(_i)) * std::get<1>(_i);
			}

			return result;
		}

		template<typename functionty>
		auto operator()(const scalar& a, const scalar& b, const functionty& f) const {
			assert( !std::empty(*this) );
			const scalar scale  = (b - a) / 2;
			const scalar center = (a + b) / 2;
			const auto& _0 = (*this)[0];
			auto result = f(std::get<0>(_0) * scale + center) * std::get<1>(_0);
			for (size_t i = 1, iend = static_cast<size_t>(std::size(*this)); i != iend; ++i) {
				const auto& _i = (*this)[i];
				result += f(std::get<0>(_i) * scale + center) * std::get<1>(_i);
			}

			return result * scale;
		}

		template<typename functionty>
		void operator()(int, const scalar& a, const scalar& b, const functionty& f) const {
			assert( !std::empty(*this) );
			const scalar scale  = (b - a) / 2;
			const scalar center = (a + b) / 2;
			const auto& _0 = (*this)[0];
			f(std::get<0>(_0) * scale + center, std::get<1>(_0) * scale);
			for (size_t i = 1, iend = static_cast<size_t>(std::size(*this)); i != iend; ++i) {
				const auto& _i = (*this)[i];
				f(std::get<0>(_i) * scale + center, std::get<1>(_i) * scale);
			}
		}
	};

	template<typename vector3, 
		math::container container = std::vector<std::pair<vector3,value_t<vector3>>>>
	struct spherical_quadrature : public container {
		template<typename functionty>
		auto operator()(const functionty& f) const {
			assert( !std::empty(*this) );
			const auto& _0 = (*this)[0];
			auto result = f(std::get<0>(_0)) * std::get<1>(_0);
			for (size_t i = 1, iend = static_cast<size_t>(std::size(*this)); i != iend; ++i) {
				const auto& _i = (*this)[i];
				result += f(std::get<0>(_i)) * std::get<1>(_i);
			}

			return result;
		}

		template<typename functionty>
		void operator()(int, const functionty& f) const {
			assert( !std::empty(*this) );
			const auto& _0 = (*this)[0];
			f(std::get<0>(_0), std::get<1>(_0));
			for (size_t i = 1, iend = static_cast<size_t>(std::size(*this)); i != iend; ++i) {
				const auto& _i = (*this)[i];
				f(std::get<0>(_i), std::get<1>(_i));
			}
		}
	};

	struct sampling2d_iterator {
		size_t sample_count = 0;
		size_t subsample_count = 0;
		
		size_t sample_index = 0;
		size_t subsample_index = 0;
		size_t start_subsample_index = 0;

		bool complete() const {
			return sample_index == sample_count;
		}

		void start() {
			sample_index = 0;
			subsample_index = 0;
			start_subsample_index = 0;
		}

		void restart() {
			sample_index = 0;
			start_subsample_index = subsample_index;
		}

		void advance() {
			++subsample_index;
			if (subsample_index == subsample_count) {
				subsample_index = 0;
			}

			if (subsample_index == start_subsample_index) {
				++sample_index;
			}
		}

		bool skip_subsample(size_t i) const {
			return i != subsample_index;
		}
	};
}// end of namespace math