#pragma once

///@brief Functions of Numbers 
///@license Free 
///@review 2022-10-5 
///@contact Jiang1998Nan@outlook.com 
#define _MATH_FUNCTION_

#include <cassert>
#include <concepts>
//#include <iostream>

#include <limits>
#include <cmath>
#include <utility>//std::min,std::max
#include <algorithm>//std::clamp

namespace math {
#ifndef __calculation_sum__
#define __calculation_sum__
/// y = sum<i=start, end>( f(i) ).
template<typename Integer, typename Function>
auto sum(Integer start, Integer end, Function f) -> decltype( f(start)+f(end) ) {
	auto result = f( start );
	for (Integer i = start+1; i <= end; ++i) {
		result += f( i );
	}

	return result;
}
#endif

/// y = product<i=start, end>( f(i) ).
template<typename Integer, typename Function>
auto product(Integer start, Integer end, Function f) -> decltype( f(start)*f(end) ){
	auto result = f( start );
	for (Integer i = start+1; i <= end; ++i) {
		result *= f( i );
	}

	return std::move( result );
}

/// factorial is 1*2*3*...*n , zeroe-order is 1 .
///@param n is positive_integer
///@formula
/// factorial(n) = 1*2*3*...*n
/// factorial(n) = n == 0 ? 1 : product<k=1,n>( k )
template<typename Number = unsigned long long>
Number factorial(Number n) {
	assert( n >= 0 );
	assert( floor(n) == n );
	Number result = 1;
	for (size_t k = 2/*ignore'1'*/; k <= static_cast<size_t>(n); ++k) {
		result *= k;
	}

	return result;
}

/// shifted factorial is 1*(x + 0)*(x + 1)*(x + 2)*...*(x + n-1), zeroe-order is 1 .
///@param x is real
///@param n is positive_integer
///@formula
/// factorial(x,n) = 1*(x + 0)*(x + 1)*(x + 2)*...*(x + n-1)
/// factorial(x,n) = n == 0 ? 1 : product<k=0,n-1>( x + k )
/// factorial(1,n) = factorial(n)
/// factorial(x,n) = gamma(x+n)/gamma(x)
template<typename Number = unsigned long long>
Number factorial(Number x, Number n) {
	assert( n >= 0 );
	assert( floor(n) == n );
	Number result = 1;
	for (size_t k = 0; k != static_cast<size_t>(n); ++k) {
		result *= (x + k);
	}

	return result;
}

/// double factorial is 1*3*5*7*...*(n-2)*n when n is odd, is 2*4*6*8*...*(n-2)*n when n is even, is 1 when n == 0 .
///@param n is positive_integer
///@formula
/// factorial2(n) * factorial2(n-1) = factorial(n)
/// factorial2(2*n-1)/( pow(2,n) * sqrt(pi) ) = gamma(n+0.5)
template<typename Number = unsigned long long>
Number factorial2(Number n) {
	assert( n >= 0 );
	assert( floor(n) == n );
	if constexpr (std::is_same_v<Number, unsigned int>) {
		if( n > 20 ) { throw /*std::overflow_error("fact2(order)")*/; }
	} else if constexpr (std::is_same_v<Number, unsigned long long>) {
		if( n > 33 ) { throw /*std::overflow_error("fact2(order)")*/; }
	}
	bool   odd    = static_cast<size_t>(n) & 1;
	Number result = 1;
	for (size_t k = odd?1:2; k <= static_cast<size_t>(n); k+=2) {
		result *= k;
	}

	return result;
}

/// binomial coefficient is pow(a + b, 2) = binomial(2,0)*a*a + binomial(2,1)*a*b + binomial(2,2)*b*b .
///@param power is integer
///@param nth is integer
/// 
///@formula
///                                1*2*3*...*power                 power!
/// binomial(power,nth) = ------------------------------- = ---------------------
///                        1*2*3*nth * 1*2*3*(power-nth)     nth! * (power-nth)!
///
///@formula shift 'nth'
///                          (nth+1)*(nth+2)*...*power        factorial(nth+1,power-nth)
/// binomial(power,nth) = -------------------------------- = ----------------------------
///                                    1*2*3*(power-nth)            (power-nth)!
/// 
///@formula shift 'power-nth'
///                        (power-nth+1)*(power-nth+2)*...*power      factorial(power-nth+1,nth)
/// binomial(power,nth) = ---------------------------------------  = ----------------------------
///                                    1*2*3*nth                                nth!
/// 
///@formula binomial theorem (power is positive integer)
///        power                                            power-nth    nth
/// (a + b)^     = sum<nth=0,power>( binomial(power,nth) * a^         * b^   )
/// 
///@formula binomial series (power is real)
///        power                    power*(power-1)*...*(power-(nth-1))     nth
/// (1 + x)^     = sum<nth=0,inf>( ------------------------------------- * x^   ) + O(...)
///                                             1*2*3*nth
/// 
///                                 factorial(power-(nth-1), nth)     nth
///              = sum<nth=0,inf>( ------------------------------- * x^   ) + O(...)
///                                       factorial(nth)
template<typename Integer = unsigned long long>
Integer binomial(size_t power, size_t nth) {
#if 1

	Integer result_num = 1;
	Integer result_den = 1;

	assert( nth <= power );
	if ( nth > power - nth ) {
		Integer k = nth + 1;
		Integer n = power - nth; 
		for ( ; k <= power; ++k, --n) {
			result_num *= k;
			result_den *= n;
			Integer divsor = gcd(result_num, result_den);
			result_num /= divsor;
			result_den /= divsor;
		}
	} else {
		Integer k = power - nth + 1;
		Integer n = nth;
		for ( ; k <= power; ++k, --n) {
			result_num *= k;
			result_den *= n;
			Integer divsor = gcd(result_num, result_den);
			result_num /= divsor;
			result_den /= divsor;
		}
	}

	assert( result_num % result_den == 0 );
	return result_num / result_den;

#else

	// C(n,0) = C(n,n) = 1
	if (nth == 0 || nth == power) {
		return 1;
	}

	// C(n,1) = C(n,n-1) = n
	if (nth == 1 || nth + 1 == power) {
		return power;
	}

	/**
	 * 1
	 * 1 1
	 * ....
	 * ......
	 *       .. <----- k
	 * 1      ... cache[nth]    { 1 }      :for n == power-2
	 *             \     |   \    |
	 *              \    |    \   |
	 *               \   |     \  |
	 *       .. <----- k-1 <----- k
	 * 1      ... cache[nth-1] cache[nth]  :for n == power-1
	 *                      \     |
	 *                       \    |
	 *                        \   |
	 *                         (p,n)       :return
	*/
	std::vector<Integer> cache;
	cache.resize(nth + 1, 1);
	for (size_t n = 2; n != power; ++n) {
		// clip right{ ignore_terms }
		size_t k = min(size_t(n), nth/*cache.size()-1*/);
		// clip right{ C(n,n) }
		if (k == size_t(n)) {
			--k;
		}
		// clip left{ ignore_terms }
		size_t final_k = nth/*cache.size()-1*/ - min(power-n, nth);
		// clip left{ C(n,0) }
		size_t last_k = final_k == 0 ? 0 : final_k - 1;
						
		for ( ; k != last_k; --k) {
			cache[k] += cache[k - 1];
		}
	}

	return cache[nth] + cache[nth - 1];

#endif
}

#if 0
/**
 * bernoulli numbers: 
 *   power series 
 * 
 * @generating function
 *        x                        B[k]*pow(x,k)
 *   ------------ = Sum[k=0,inf]( --------------- )
 *    exp(x) - 1                        k!
 * 
 * @calculate formula
 *                         1                    r                n
 *   B[n] = Sum[k=0,n]( -------*Sum[r=0,k]( (-1)^*binomial(k,r)*r^ ) )
 *                       k + 1
 *
 * @relation for recurrence 
 * 
 *    Sum[k=0,n-1]( B[k]*binomial(n,k) ) == 0
 * 
 *    Sum[k=0,n-1]( B[k]*binomial(n+1,k) )
 *   -------------------------------------- == B[n]
 *             - binomial(n+1,n)
 * 
 * ...
*/
template<typename Rational/* = rational64_t*/>
std::vector<Rational> bernoulli_numbers(size_t index) {
/**
 * @relation for recurrence 
 *   ...
 * 
 * @relation for recurrence form2
 *
 *    Sum[k=0,n-1]( B[k]*binomial(n,k) )    == 0                      :using recurrence relation
 *
 *    Sum[k=0,n]( B[k]*binomial(n+1,k) )    == 0                      :replace 'n' by 'n+1'
 *
 *    Sum[k=0,n-1]( B[k]*binomial(n+1,k) )  == -B[n]*binomial(n+1,n)  :substract last nth
 *
 *    Sum[k=0,n-1]( B[k]*binomial(n+1,k) )
 *   -------------------------------------- == B[n]
 *              - binomial(n+1,n)
 *
 * ...
*/
	auto number_array = std::vector<Rational>(index + 1); 
			 number_array[0] = 1;
	for (size_t n = 1; n <= index; ++n) {
		number_array[n] = - sum(size_t(0), n-1, [n,&number_array](size_t k){ return number_array[k] * binomial(n+1, k); })
												/ binomial(n+1, n);
	}

	return std::move(number_array);
}
#endif


using std::modf;
using std::frexp;
using std::ldexp;
using std::trunc;
template<typename number> 
number frac(number x) { number unsed; return modf(x, &unsed); }

using std::min;
using std::max;
using std::clamp;
using std::abs;
using std::floor;
using std::ceil;
using std::round;

using std::pow;
using std::sqrt;
using std::cbrt;
using std::exp;

using std::atanh;
using std::log;
using std::asinh;
using std::acosh;
using std::cosh;
using std::sinh;
using std::tanh;

using std::cos;
using std::sin;
using std::tan;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;

/// Integer Exponential of Numbers (use sum_series).
///		y = pow(x,power), @param x in (-inf, inf), @param power is integer, @return y in (-inf, inf). 
template<typename Number> requires (std::integral<Number> || std::floating_point<Number>)
Number pow(const Number& x, int power) {
	if ( power < 0 ) {
		return pow(1/x, -power);
	}

	/** I write at 2021.5.20, LongJiangnan
	 * 1100 1011 = 203
	 *
	 * =  2^7 * 1
	 *  + 2^6 * 1
	 *  + 2^5 * 0
	 *  + 2^4 * 0
	 *  + 2^3 * 1
	 *  + 2^2 * 0
	 *  + 2^1 * 1
	 *  + 2^0 * 1
	 * 
	 * =  2^7 * (203 - 2^0 - 2^1 - 2^3 - 2^6)%256/128
	 *  + 2^6 * (203 - 2^0 - 2^1 - 2^3)%128/64
	 *  + 2^5 * (203 - 2^0 - 2^1 - 2^3)%64/32
	 *  + 2^4 * (203 - 2^0 - 2^1 - 2^3)%32/16
	 *  + 2^3 * (203 - 2^0 - 2^1)%16/8
	 *  + 2^2 * (203 - 2^0 - 2^1)%8/4
	 *  + 2^1 * (203 - 2^0)%4/2
	 *  + 2^0 * 203%2/1
	 *                i                                  i+1    i
	 * = sum<i=0,7>( 2^ * ( (203 - sum<k=0,i-1>(...)) % 2^   / 2^ ) )
	 * 
	 * 
	 * pow(x,203)
	 *           7    6    3    1    0
	 * = pow(x, 2^ + 2^ + 2^ + 2^ + 2^)
	 *          7           6           3           1           0
	 * = pow(x,2^) * pow(x,2^) * pow(x,2^) * pow(x,2^) * pow(x,2^)
	 *                           i                                  i+1    i
	 * = product<i=0,7>( pow(x, 2^ * ( (203 - sum<k=0,i-1>(...)) % 2^   / 2^ )) )
	 *                           i                                        i+1    i
	 * = product<i=0,7>( pow(x, 2^) * pow(x, (203 - sum<k=0,i-1>(...)) % 2^   / 2^) )
	 * 
	 * = product<i=0,7>( pow(x, 2^i) * has_bit )
	*/
	int seriesP = 0;
	int termP = 1;
	Number series = {1};
	Number term = x;
	for (int i = 0; seriesP != power; ) {
		if ( (power - seriesP) % (termP<<1) ) {
			seriesP += termP;
			series *= term;
		}

		termP <<= 1;
		term *= term;
	}

	return series;
}

/// Square Root of Numbers (use Babylonian's method or Newton's iterative formula). 
///		y = pow(x,1/2), @param x in [0, inf), @return y in [0, inf). 
/// derivative( sqrt(x) ) = 1/2 / sqrt(x), 
/// antiderivative( sqrt(x) ) = 3/2 * sqrt(x)*x 
template<typename Number> requires std::floating_point<Number>
Number sqrt(const Number& x) {
	assert( x >= 0 );
	if ( x > 0 ) {
		/// "x = y*y" use Newton's iterative formula "x[i+1] = x[i] - f(x[i])/dfdx(x[i])",
		/// x is not unknown, replace 'x' by 'y' and transform to form of "0 = f(y)":
		///		f(y) = y*y - x 
		///		dfdy(y) = 2*y 
		///		                 y[i]*y[i] - x
		///		y[i+1] = y[i] - ---------------
		///		                    2*y[i]
		/// 
		/// simply
		///		          y[i] + x/y[i] 
		///		y[i+1] = ---------------
		///		               2
		/// 
		/// "https://en.wikipedia.org/wiki/Methods_of_computing_square_roots"
		const Number prev_eps = pow(static_cast<Number>(2), - static_cast<int>(std::numeric_limits<Number>::digits/2 + 1));//@see cbrt(x)
		Number y = x;
		Number ym1;
		do {
			ym1 = y;
			y = (y + x/y)/2;
		} while ( abs(y-ym1) >= prev_eps*abs(y) );

		return (y + x/y)/2;
	}
	else /*( x == 0 )*/ {// not convergence
		return static_cast<Number>(0);
	}
}

/// Cubic Root of Numbers (use ...). 
/// 	y = pow(x,1/3), @param x in (-inf, inf), @return y in (-inf, inf). 
/// derivative( cbrt(x) ) = 1/3 * cbrt(x)/x, 
/// antiderivative( cbrt(x) ) = 4/3 * cbrt(x)*x 
template<typename Number> requires std::floating_point<Number>
Number cbrt(const Number& x) {
	if ( x != 0 ) {
		/// third order convergence, for example:
		///		exp2(-1) = 0.5   ~= error
		///		exp2(-1*3) = 0.125   ~= next error
		///		exp2(-1*3*3) = 0.001953125   ~= next next error
		///		exp2(-1*3*3*3) = 0.000000007450580596923828125
		///		...
		///		exp2(-digits/3) = ....
		///		exp2(-digits) = 'epsilon'
		/// 
		/// if convergence started, the error must pass (0.X|0.0X|...),
		/// then error alone the formula 'pow(error,3)' go to end.
		/// so only until error < prev digit 'epsilon', and can avoid loop at 'epsilon', notice we added a tolerance.
		const Number prev_eps = pow(static_cast<Number>(2), - static_cast<int>(std::numeric_limits<Number>::digits/3 + 2));
		Number y = x;
		Number factor;
		do {
			Number yyy = y * y * y;
			factor = (yyy + x + x) / (yyy + yyy + x);
			y *= factor;
		} while ( abs(1 - factor) >= prev_eps );

		Number yyy = y*y*y;
		factor = (yyy + x + x) / (yyy + yyy + x);
		return y * factor;
	}
	else /* x == 0 */ {// not convergence.
		return static_cast<Number>(0);
	}
}

/// Natural Exponential of Numbers (use maclaurin_series). 
///		y = limit(n=inf, pow(1 + x/n, n) ), @param x in (-inf, inf), @return y in (0, inf). 
/// exp(x) = sum(k=0, inf, pow(x,k)/fact(k)), 
/// derivative( exp(x) ) = exp(x), 
/// antiderivative( exp(x) ) = exp(x). 
template<typename Number> requires std::floating_point<Number>
Number exp(const Number& x) {
	if ( 0 < x && x < 1 ) {
		/// exp(x)
		///		                 dexp(0)     k
		///		= sum(k=0,inf,  --------- * x^ )    :expand use maclaurin_series
		///		                   k!
		///		                 1      k
		///		= sum(k=0,inf, ----- * x^ )
		///		                 k!
		/// 
		/// term[k+1]/term[k]
		///		   pow(x,k+1)/fact(k+1)
		///		= --------------------- = x/(k+1) 
		///		    pow(x,k)/fact(k)
		Number series = 0;
		const Number eps = static_cast<Number>(0.01);
		Number seriesR = 0;
		const Number epsR = std::numeric_limits<Number>::epsilon();
		Number k = 0;
		Number term = 1;
		do {
			series += term;
			term *= ( x/(k += 1) );
		} while ( term >= eps*series );
		do {
			seriesR += term;
			term *= ( x/(k += 1) );
		} while ( term >= epsR*seriesR );

		return series + seriesR;
	}
	else if (0 < x) {// bad-convergence, odd terms are negative, even terms are positive. 
		return 1 / exp(abs(x));//@note "abs(float)" faster than "neg(float)".
	}
	else if (x < 1) {// slow-convergence.
		Number xi = floor(x);
		return exp(x - xi) * pow(exp(static_cast<Number>(1)), static_cast<int>(xi));
	}
	else /*(x == 0)*/ {// seriesR not convergence.
		return static_cast<Number>(1);
	}
}


/// Inverse Hyperbolic Tangent of Numbers (use maclaurin_series).
///		y = atanh(x), @param x in (-1, 1), @return y in (-inf, inf). 
/// atanh(u) + atanh(v) = atanh((u+v)/(1 + u*v)), 
/// atanh(x) = (+ -)acosh( 1/sqrt(1-x*x) ), 
/// atanh(x) = asinh( x/sqrt(1-x*x) ), 
/// atanh(x) = log((1+x)/(1-x))/2, 
/// derivative( atanh(x) ) = 1/(1 - x*x), 
/// antiderivative( atanh(x) ) = log(1 - x*x)/2 + x*atanh(x) 
template<typename Number> requires std::floating_point<Number>
Number atanh(const Number& x) {
	assert( -1 < x && x < 1 );
	if (x != 0) {
		/// atanh(x)
		///		                    1
		///		= integral<0,x>( ---------, dx )
		///		                  1 - x*x
		///		                           -1
		///		= integral<0,x>( (1 + -x*x)^ , dx )                                  :1. binomial_series expansion
		///
		///		                                                      k
		///		= integral<0,x>( sum<k=0,inf>( binomial(-1,k) * (-x*x)^ ), dx )
		///
		///		                                                    k    2*k
		///		= integral<0,x>( sum<k=0,inf>( binomial(-1,k) * (-1)^ * x^   ), dx )
		///
		///		                                     k                  2*k
		///		= sum<k=0,inf>( binomial(-1,k) * (-1)^ * integral<0,x>(x^  , dx) )   :2. exchange sum<> and integral<>, and integrate
		///
		///		                                     k    x^(2*k+1)
		///		= sum<k=0,inf>( binomial(-1,k) * (-1)^ * ----------- )
		///		                                            2*k+1
		///		                 -1*-2*-3*...*(-1-k+1)        k    x^(2*k+1)
		///		= sum<k=0,inf>( ----------------------- * (-1)^ * ----------- )      :3. simplify
		///		                          k!                         2*k+1
		///		                 x^(2*k+1)
		///		= sum<k=0,inf>( ----------- )                                        :   -1^k * -1*-2*-3*...*(-1-k+1) = k!
		///		                   2*k+1
		/// 
		/// term[k+1]/term[k]
		///		   x^(2*(k+1)+1)     x^(2*k+1)     x^(2*k+3)     2*k+1
		///		= --------------- / ----------- = ----------- * -------
		///		     2*(k+1)+1         2*k+1       x^(2*k+1)     2*k+3
		/// 
		///		= x*x * (2*k+1)/(2*k+3)
		/// 
		/// "https://mathworld.wolfram.com/InverseHyperbolicTangent.html"
		/// "https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions"
		Number series = 0;
		const Number eps = static_cast<Number>(0.01);
		Number seriesR = 0;
		const Number epsR = std::numeric_limits<Number>::epsilon();
		Number term = x;
		Number k = 0;
		const Number xx = x * x;
		do {
			series += term;
			term *= xx * (2*k+1)/(2*k+3);
			k += 1;
		} while ( term >= eps*series );
		do {
			seriesR += term;
			term *= xx * (2*k+1)/(2*k+3);
			k += 1;
		} while ( term >= epsR*seriesR );

		return series + seriesR;
	}
	else /*(x == 0)*/ {// not convergence
		return static_cast<Number>(0);
	}
}

/// Natural Logarithm of Numbers (use maclaurin_series). 
///		y = log(x), @param x in (0, inf), @return y in (-inf, inf). 
/// log(a*b) = log(a) + log(b), 
/// log(a/b) = log(a) - log(b), 
/// log(x^b) = log(x) * b, 
/// logB(x) = logC(x)/logC(b), 
/// log(x) = atanh((x - 1)/(x + 1)) * 2, 
/// derivative( log(x) ) = 1/x, 
/// antiderivative( log(x) ) = x*(log(x) - 1). 
template<typename Number> requires std::floating_point<Number>
Number log(const Number& x) {
	/// log(x) = log( x.significand() * exp2(x.exponent()) ) = log( x,significand() ) + x.exponent()*log(2)
	int x_exponent2;
	Number x_significand = frexp(x, &x_exponent2);
	if ( x_significand == 0 ) {
		return - std::numeric_limits<Number>::infinity();
	}
	return atanh((x_significand - 1)/(x_significand + 1)) * 2
		+ x_exponent2 * atanh(static_cast<Number>(2-1)/(2+1)) * 2;
}

/// General Exponential of Numbers (use maclaurin_series).
///		y = pow(x,power), @param x in (-inf, inf), @param power in (-inf, inf), @return y in (-inf, inf). 
/// 
/// derivative( pow(x,power) ) = pow(x,power - 1)*power, 
/// 
///                                   pow(x,power + 1)
/// antiderivative( pow(x,power) ) = ------------------. 
///                                    (power + 1)
template<typename Number> requires std::floating_point<Number>
Number pow(const Number& x, const Number& power) {
	///  power      log(x)  power    log(x) * power
	/// x^     = ( e^      )^     = e^
	if ( floor(power) == power ) {
		return pow(x, static_cast<int>(power));
	}
	return exp(log(x) * power);
}

/// Inverse Hyperbolic Sine of Numbers (use maclaurin_series). 
///		y = asinh(x), @param x in (-inf, inf), @return y in (-inf, inf). 
/// 
/// asinh(x) = log(sqrt(x*x + 1) + x), 
/// 
///                                 1
/// derivative( asinh(x) ) = ---------------, 
///                           sqrt(x*x + 1)
/// 
/// antiderivative( asinh(x) ) = x*asinh(x) - sqrt(x*x + 1). 
/// 
template<typename Number> requires std::floating_point<Number>
Number asinh(const Number& x) {
	return log(sqrt(x*x + 1) + x);
}

/// Inverse Hyperbolic Cosine of Numbers (use maclaurin_series). 
///		y = asinh(x), @param x in [1, inf), @return y in [0, inf). 
/// 
/// acosh(x) = log(x + sqrt(x - 1)*sqrt(x + 1)), 
/// 
///             acosh(2*x^2 - 1)     acosh(8*x^4 - 8*x^2 + 1)
/// acosh(x) = ------------------ = --------------------------, 
///                     2                       4
///                                      1
/// derivative( acosh(x) ) = -------------------------, 
///                           sqrt(x - 1)*sqrt(x + 1)
/// 
/// antiderivative( acosh(x) ) = x*acosh(x) - sqrt(x - 1)*sqrt(x + 1). 
/// 
template<typename Number> requires std::floating_point<Number>
Number acosh(const Number& x) {
	return log(x + sqrt(x-1)*sqrt(x+1));
}


/**
 * @brief calculate pi: use Bailey-Borwein-Plouffe formula
 * @author David H. Bailey, Peter Borwein, Simon Plouff
 * @return ratio of the circumference of a circle to its diameter
 * @Bailey-Borwein-Plouffe formula for 'pi'
 *   part of derivation  ...
 *                        1          4           2           1           1
 *   pi = sum<k=0,inf>( ------*( --------- - --------- - --------- - --------- ) )
 *                       16^k     8*k + 1     8*k + 4     8*k + 5     8*k + 6
*/
template<typename Number> requires std::floating_point<Number>
Number calculate_pi() {
	Number series = 0;
	const Number eps = static_cast<Number>(0.01);
	Number seriesR = 0;
	const Number epsR = std::numeric_limits<Number>::epsilon();

	Number hex_base = 1;
	Number k8 = 0;
	Number term = hex_base * (4/(k8 + 1) - 2/(k8 + 4) - 1/(k8 + 5) - 1/(k8 + 6));
	do {
		series += term;
		hex_base /= 16; k8 += 8;
		term = hex_base * (4/(k8 + 1) - 2/(k8 + 4) - 1/(k8 + 5) - 1/(k8 + 6));
	} while ( abs(term) >= eps * abs(series) );
	do {
		seriesR += term;
		hex_base /= 16; k8 += 8;
		term = hex_base * (4/(k8 + 1) - 2/(k8 + 4) - 1/(k8 + 5) - 1/(k8 + 6));
	} while ( abs(term) >= epsR * abs(seriesR) );

	return series + seriesR;
}

/** 
 * @author 
 *   David H. Bailey, Peter Borwein, Simon Plouffe, Donald Knuth
 * @param digit 
 *   hexdigit - 1 = std::numeric_limit<Number>::digit/4 - 1
 * @return 
 *   pi.hex_significant[digit+1, ...) 
 * @exapmle 
 *   digit = std::numeric_limits<double>::digits/4 - 1;
 *   float256_t A = trunc((calculate_pi<double>()*2) * pow(16.0,digit)) * pow(16.0,-digit);
 *   float256_t B = trunc(calculate_pi<double>(digit,2) * pow(16.0,digit)) * pow(16.0,-digit-digit);
 *   float256_t C = trunc(calculate_pi<double>(digit+digit,2) * pow(16.0,digit)) * pow(16.0,-digit-digit-digit);
 *   std::cout << A << std::endl;
 *   std::cout << B << std::endl;
 *   std::cout << C << std::endl;
 *   std::cout << A + B + C << std::endl;
 *   std::cout << calculate_pi<float256_t>()*2 << std::endl;
*/
template<typename Number> requires std::floating_point<Number>
Number calculate_pi(int digit, Number multiplier = 1) {
	assert( multiplier >= 0 );
	Number series = 0;
	const Number eps = static_cast<Number>(0.01);
	Number seriesR = 0;
	const Number epsR = std::numeric_limits<Number>::epsilon();
	
	Number hex_base;
	Number k8;
	Number term;
	//if ( floor(multiplier) == multiplier && multiplier < std::numeric_limits<unsigned long long>::max()
	//  && digit <= 16
	//  && pow(16ULL,digit) <= std::numeric_limits<unsigned long long>::max()/static_cast<unsigned long long>(multiplier) )
	//{// optimize
	//  unsigned long long i_hex_base = pow(16ULL,digit) * static_cast<unsigned long long>(multiplier);
	//  unsigned long long i_k8 = 0;
	//  term = static_cast<Number>(4*(i_hex_base%(i_k8+1)))/static_cast<Number>(i_k8+1)
	//    - static_cast<Number>(2*(i_hex_base%(i_k8+4)))/static_cast<Number>(i_k8+4)
	//    - static_cast<Number>(i_hex_base%(i_k8+5))/static_cast<Number>(i_k8+5)
	//    - static_cast<Number>(i_hex_base%(i_k8+6))/static_cast<Number>(i_k8+6);
	//  if (i_hex_base != 1 && i_hex_base != multiplier) {
	//  do {
	//    series += term;
	//    i_hex_base /= 16; i_k8 += 8;
	//    term = static_cast<Number>(4*(i_hex_base%(i_k8+1)))/static_cast<Number>(i_k8+1)
	//      - static_cast<Number>(2*(i_hex_base%(i_k8+4)))/static_cast<Number>(i_k8+4)
	//      - static_cast<Number>(i_hex_base%(i_k8+5))/static_cast<Number>(i_k8+5)
	//      - static_cast<Number>(i_hex_base%(i_k8+6))/static_cast<Number>(i_k8+6);
	//  } while (i_hex_base != 1 && i_hex_base != multiplier);
	//  }
	//  hex_base = static_cast<Number>(i_hex_base);
	//  k8 = static_cast<Number>(i_k8);
	//  series = frac(series);
	//}
	//else 
	{
		hex_base = pow(Number(16), digit) * multiplier;
		k8 = 0;
		term = 4*fmod(hex_base, k8+1)/(k8+1)
			- 2*fmod(hex_base, k8+4)/(k8+4)
			- fmod(hex_base, k8+5)/(k8+5)
			- fmod(hex_base, k8+6)/(k8+6);
		do {
			series += term;
			hex_base /= 16; k8 += 8;
			term = 4*fmod(hex_base, k8+1)/(k8+1) 
				- 2*fmod(hex_base, k8+4)/(k8+4)
				- fmod(hex_base, k8+5)/(k8+5)
				- fmod(hex_base, k8+6)/(k8+6);
		} while ( hex_base >= 1 );
		series = frac(series);
	}
	do {
		series += term;
		hex_base /= 16; k8 += 8;
		term = hex_base*(4/(k8+1) - 2/(k8+4) - 1/(k8+5) - 1/(k8+6));
	} while ( abs(term) >= eps * abs(series) );
	do {
		seriesR += term;
		hex_base /= 16; k8 += 8;
		term = hex_base*(4/(k8+1) - 2/(k8+4) - 1/(k8+5) - 1/(k8+6));
	} while ( abs(term) >= epsR * abs(seriesR) );

	return series < 0 
		? 1-abs(frac(series+seriesR))
		: frac(series+seriesR);
	/** you need 
	* trunc(calculate_pi(digit) * pow(16.0,digit)) * pow(16.0,-digit-digit)
	* @see circular_constant::operator-
	*/
}

/**
 * @brief circular_constant = pi * multiplier + addend
 * @author LongJiangnan
*/
template<typename Number = double>
struct circular_constant {
	mutable Number approximation;
	mutable Number remainder1;
	mutable Number remainder2;
	mutable Number multiplier;
	Number addend;
	Number new_multiplier;
	 
	circular_constant() {
		static int digit = std::numeric_limits<Number>::digits / 4 - 1;
		static Number pi_appeoximation = trunc(calculate_pi<Number>() * pow(Number(16),digit)) * pow(Number(16),-digit);
		static Number pi_remainder1 = trunc(calculate_pi<Number>(digit) * pow(Number(16),digit)) * pow(Number(16),-digit-digit);
		static Number pi_remainder2 = trunc(calculate_pi<Number>(digit+digit) * pow(Number(16),digit)) * pow(Number(16),-digit-digit-digit);
		approximation = pi_appeoximation;
		remainder1 = pi_remainder1;
		remainder2 = pi_remainder2;
		multiplier = static_cast<Number>(1);
		addend     = static_cast<Number>(0);
		new_multiplier = multiplier;
	}

	circular_constant(Number _Approx, Number _Rem1, Number _Rem2, Number _Mulp, Number _Adde, Number _Mulp_new)
		: approximation(_Approx), remainder1(_Rem1), remainder2(_Rem2), multiplier(_Mulp), addend(_Adde), new_multiplier(_Mulp_new) {}

	void calculate() const {
		if ( new_multiplier != multiplier ) {
			Number scale = new_multiplier / multiplier;
			if ( scale == -1 || scale == 4 || scale == 2 || scale == 1 || scale == 0.5 || scale == 0.25 || scale == 0.125 ) {
				// significand not change
				approximation *= scale;
				remainder1 *= scale;
				remainder2 *= scale;
			} else {
				int digit = std::numeric_limits<Number>::digits / 4 - 1;

				Number sign = new_multiplier >= 0 ? 1 : -1;
				Number scale = abs(new_multiplier);
				Number binary_offset = 1;
				while (binary_offset < scale) {
					binary_offset *= 2;
				}
				approximation
					= trunc(calculate_pi<Number>() * scale/binary_offset * pow(Number(16),digit))
						* pow(Number(16),-digit) * binary_offset
							* sign;
				remainder1
					= trunc(calculate_pi<Number>(digit, scale/binary_offset)
						* pow(Number(16),digit)) * pow(Number(16),-digit-digit) * binary_offset
							* sign;
				remainder2
					= trunc(calculate_pi<Number>(digit+digit, scale/binary_offset)
						* pow(Number(16),digit)) * pow(Number(16),-digit-digit-digit) * binary_offset
							* sign;
			}
			multiplier = new_multiplier;
		}
	}

	bool calculated() const {
		return new_multiplier == multiplier;
	}

	circular_constant operator-() const {
		// - (pi*multiplier + addend) = pi*(-multiplier) + (-addend)
		return circular_constant(approximation, remainder1, remainder2, multiplier, 
			-addend, -new_multiplier);
	}

	circular_constant operator+(const Number& right) const {
		return circular_constant(approximation, remainder1, remainder2, multiplier,
			addend + right, new_multiplier);
	}

	circular_constant operator-(const Number& right) const {
		return circular_constant(approximation, remainder1, remainder2, multiplier,
			addend - right, new_multiplier);
	}

	circular_constant operator*(const Number& right) const {
		return circular_constant(approximation, remainder1, remainder2, multiplier,
			addend, new_multiplier*right);
	}

	circular_constant operator/(const Number& right) const {
		return circular_constant(approximation, remainder1, remainder2, multiplier,
			addend, new_multiplier/right);
	}

	Number operator%(const Number& right) const {
		const Number piA = approximation * multiplier;
		const Number piR1 = remainder1 * multiplier;
		const Number piR2 = remainder2 * multiplier;

		Number result = piA - trunc(piA/right) * right;
		result = result + piR1;
		if ( result > right ) {
			result = result - trunc(result/right) * right;
			assert( right >= std::numeric_limits<Number>::epsilon() );
			// .... while(..){ ... } for infinite small 'right'...
		}
		return result += piR2;
	}
	
	circular_constant operator+(const circular_constant& right) const {
		// pi * multiplier + addend  +  pi*r_multiplier + r_addend
		// pi * (multiplier + r_multiplier) + (addend + r_addend)
		return circular_constant(approximation, remainder1, remainder2, multiplier,
			this->addend + right.addend, this->new_multiplier + right.new_multiplier);
	}

	circular_constant operator-(const circular_constant& right) const {
		// pi * multiplier + addend  -  pi*r_multiplier - r_addend
		// pi * (multiplier - r_multiplier) + (addend - r_addend)
		return circular_constant(approximation, remainder1, remainder2, multiplier,
			this->addend - right.addend, this->new_multiplier - right.new_multiplier);
	}

	bool operator==(const Number& right) const {
		Number scale = new_multiplier / multiplier;
		if ( scale > 0 ) {
			return abs((addend/scale + approximation+remainder1+remainder2) - right/scale)
				< std::numeric_limits<Number>::epsilon();
		} else if ( scale == 0 ) {
			return addend == right;
		}
	}

	bool operator!=(const Number& right) const {
		return !(*this == right);
	}

	bool operator<(const Number& right) const {
		Number scale = new_multiplier / multiplier;
		if ( scale > 0 ) {
			return (addend/scale + approximation+remainder1+remainder2) < right/scale;
		} else if ( scale == 0 ) {
			return addend < right;
		} else /* ( scale  < 0 ) */ {
			return (addend/scale + approximation+remainder1+remainder2) > right/scale;
		}
	}

	bool operator>(const Number& right) const {
		Number scale = new_multiplier / multiplier;
		if ( scale > 0 ) {
			return (addend/scale + approximation+remainder1+remainder2) > right/scale;
		} else if ( scale == 0 ) {
			return addend > right;
		} else /* ( scale  < 0 ) */ {
			return (addend/scale + approximation+remainder1+remainder2) < right/scale;
		}
	}

	bool operator>=(const Number& right) const {
		return !((*this) < right);
	}

	bool operator<=(const Number& right) const {
		return !((*this) > right);
	}

	friend circular_constant operator+(const Number& left, const circular_constant& pi) {
		return pi + left;
	}

	friend circular_constant operator-(const Number& left, const circular_constant& pi) {
		return -(pi - left);
	}

	friend circular_constant operator*(const Number& left, const circular_constant& pi) {
		return pi * left;
	}

	friend Number operator/(const Number& left, const circular_constant& pi) {
		Number scale = pi.new_multiplier / pi.multiplier;
		Number discrim = pi.approximation * scale;
		if ( (discrim >= 0 && pi.addend >= 0) || (discrim < 0 && pi.addend < 0) ) {
			// left/(pi*multiplier + addend)
			// left*scale / ((pi*multiplier + addend)*scale)
			return (left/scale) / (pi.addend/scale + pi.approximation+pi.remainder1+pi.remainder2);
		} else {
			if ( !pi.calculated() ) {
				pi.calculate();
			}
			return left / (pi.addend + pi.approximation + pi.remainder1 + pi.remainder2);
		}
	}

	friend circular_constant operator%(const Number& left, const circular_constant& pi) {
		/*const Number piA = pi.approximation * pi.scale;
		const Number piR1 = pi.remainder1 * pi.scale;
		const Number piR2 = pi.remainder2 * pi.scale;
		Number result = (left/piA - trunc(left/piA))*piA;
		result += piR1 + piR2;
		return result < piA ? result : result - pi;*/
		return left - trunc(left/pi)*pi;
	}
	
	friend bool operator==(const Number& left, const circular_constant& pi) {
		return pi == left;
	}

	friend bool operator!=(const Number& left, const circular_constant& pi) {
		return pi != left;
	}

	friend bool operator<(const Number& left, const circular_constant& pi) {
		return pi > left;
	}

	friend bool operator>(const Number& left, const circular_constant& pi) {
		return pi < left;
	}

	friend bool operator<=(const Number& left, const circular_constant& pi) {
		return pi >= left;
	}

	friend bool operator>=(const Number& left, const circular_constant& pi) {
		return pi <= left;
	}

	circular_constant operator*(int times) const {
		return (*this) * static_cast<Number>(times);
	}

	circular_constant operator/(int times) const {
		return (*this) / static_cast<Number>(times);
	}

	operator Number() const {
		if ( multiplier == 0 || new_multiplier == 0 ) {// addend + pi*0*new_multiplier
			return addend;
		}
		if (addend == 0) {
			return approximation * new_multiplier / multiplier;
		}

		Number scale = new_multiplier / multiplier;
		Number discrim = approximation * scale;
		if ( (discrim >= 0 && addend >= 0) || (discrim < 0 && addend < 0) ) {
			// addend + pi*new_multiplier
			// addend/(new_multiplier/multiplier) + pi*new_multiplier/(new_multiplier/multiplier)
			// addend/(new_multiplier/multiplier) + pi*multiplier
			return (addend/scale + approximation+remainder1+remainder2) * scale;
		} else {
			if ( !this->calculated() ) {
				this->calculate();
			}
			return addend + approximation + remainder1 + remainder2;
		}
	}
};

template<typename Number> requires std::floating_point<Number>
Number cos(const Number& x, const circular_constant<Number>& pi);

/**
 * @brief 
 *   sine function: use maclaurin_series
 * @param x 
 *   [-pi*2, pi*2]
 * @return 
 *   [0, 1], max_error ~= epsilon, avg_error ~= epsilon/6
 * @derivative 
 *   differentiate(sin, x) = cos(x)
 * @integral 
 *   integrate(sin(x), dx) = -cos(x) + constant
 * @formula 
 *   sin(x) = -sin(x)              :odd function
 * 
 *   sin(x) = sin(x + pi*2 * n)    :period function
 * 
 *   sin(x*2) = 2 * sin(x) * cos(x)             :double angle law
 * 
 *   sin(x/2) = sqrt((1 - cos(x))/2)            :half angle law
 * 
 *   sin(a+b) = sin(a)*cos(b) + cos(a)*sin(b)   :sum law
 *  
 *   sin(a-b) = sin(a)*cos(b) - cos(a)*sin(b)   :difference law
 * 
 *   sin(a) + sin(b) = 2 * sin((a+b)/2) * cos((a-b)/2)   :sum to product law
 * 
 *   sin(a) * sin(b) = 1/2 * ( cos(a-b) - cos(a+b) )     :product to sum law
 * 
 *   sin(x) = sum<k=0,inf>( pow(-1,k)/fact(2*k+1) * pow(x,2*k+1) )   :maclaurin_series, only odd term, odd is 2*k+1
*/
template<typename Number> requires std::floating_point<Number>
Number sin(const Number& x, const circular_constant<Number>& pi = circular_constant<Number>()) {
	if ( x < 0 ) {//  convergence optimize, sin(x) = -sin(-x)
		return -sin(abs(x), pi);
	} else if ( x == 0 ) {// not convergence
		return static_cast<Number>(0);
	} else if ( x > pi/2 ) {// convergence optimize
		int n = static_cast<int>(trunc(x / (pi/2)));
		auto xn = x % (pi/2);
		switch ( n % 4 ) {
			case 0: // 0 -> 1
				return sin(static_cast<Number>(xn), pi);
			case 1: // 1 -> 0
				if ( Number(xn) - pi/2 < 0.125 ) {
					return sin(static_cast<Number>(pi/2 - xn), pi);
				}
				return cos(static_cast<Number>(xn), pi);
			case 2: // 0 -> -1
				return -sin(static_cast<Number>(xn), pi);
			case 3: // -1 -> 0
				if ( Number(xn) - pi/2 < 0.125 ) {
					return -sin(static_cast<Number>(pi/2 - xn), pi);
				}
				return -cos(static_cast<Number>(xn), pi);
		}
	}

	assert( x != 0 );
	// sum maclaurin_series
	const Number eps = static_cast<Number>(0.01);
	Number series = 0;
	const Number epsR = std::numeric_limits<Number>::epsilon();
	Number seriesR = 0;

	const Number neg_x_x = -(x*x);
	Number term = x;
	Number k = 0;
	do {
		series += term;
		/**
		 *  term[k+1]       -1^(k+1)       2*(k+1)+1      -1^k       2*k+1
		 * ----------- = -------------- * x^         / ---------- / x^
		 *  term[k]       (2*(k+1)+1)!                  (2*k+1)!
		 *                     (2*k+1)!     x^(2*k+3)
		 *             = -1 * ---------- * -----------
		 *                     (2*k+3)!     x^(2*k+1)
		 *                            1
		 *             = -1 * ----------------- * x*x
		 *                     (2*k+2)*(2*k+3)
		*/
		term *= ( neg_x_x / (2*k+2) / (2*k+3) );
		k += 1;
	} while ( abs(term) >= eps*abs(series) );
	do {
		seriesR += term;
		term *= ( neg_x_x / (2*k+2) / (2*k+3) );
		k += 1;
	} while ( abs(term) >= epsR*abs(seriesR) );

	return series + seriesR;
}

/**
 * cosine function:
 *   use maclaurin_series
 * 
 * @param x 
 *   [-pi*2, pi*2]
 * 
 * @return 
 *   [0, 1]
 * 
	* @derivative
 *   differentiate(cos, x) = -sin(x)
 * 
 * @integral
 *   integrate(cos(x), dx) = sin(x) + constant
 * 
 * @alternative
 *   cos(x) = cos(-x)             :even function
 * 
 *   cos(x) = cos(x + pi*2 * n)   :period function
 * 
 *   cos(x*2) = cos(x)*cos(x) - sin(x)*sin(x)   :double angle law
 * 
 *   cos(x/2) = sqrt((1 + cos(x))/2)            :half angle law
 * 
 *   cos(a+b) = cos(a)*cos(b) - sin(a)*sin(b)   :sum law
 * 
 *   cos(a-b) = cos(a)*cos(b) + sin(a)*sin(b)   :difference law
 * 
 *   cos(a) + cos(b) = 2 * cos((a+b)/2) * cos((a-b)/2)  :sum to product law
 * 
 *   cos(a) * cos(b) = 1/2 * ( cos(a-b) + cos(a+b) )    :product to sum law
 * 
 *   cos(x) = sum<k=0,inf>( pow(-1,k)/fact(2*k) * pow(x,2*k) )   :maclaurin_series, only even term, even is 2*k
*/
template<typename Number> requires std::floating_point<Number>
Number cos(const Number& x, const circular_constant<Number>& pi = circular_constant<Number>()) {
	if ( abs(x) > pi/2 ) {// convergence optimize
		int n = static_cast<int>(trunc(x / (pi/2)));
		auto xn = x % (pi/2);
		switch ( n % 4 ) {
			case 0: // 1 -> 0
				if ( abs(Number(xn)) - pi/2 < 0.125 ) {
					return sin(static_cast<Number>(pi/2 - xn), pi);
				}
				return cos(static_cast<Number>(xn), pi);
			case 1: // 0 -> -1
				return -sin(static_cast<Number>(xn), pi);
			case 2: // -1 -> 0
				if ( abs(Number(xn)) - pi/2 < 0.125 ) {
					return -sin(static_cast<Number>(pi/2 - xn), pi);
				}
				return -cos(static_cast<Number>(xn), pi);
			case 3: // 0 -> 1
				return  sin(static_cast<Number>(xn), pi);
		}
	} else if ( x == 0 ) {// seriesR not convergence
		return static_cast<Number>(1);
	}

	assert( x != 0 );
	// sum maclaurin_series
	const Number eps = static_cast<Number>(0.01);
	Number series = 0;
	const Number epsR = std::numeric_limits<Number>::epsilon();
	Number seriesR = 0;

	const Number neg_x_x = -(x*x);
	Number term = 1;
	Number k = 0;
	do {
		series += term;
		/**
		 *  term[k+1]      -1^(k+1)      2*(k+1)     -1^k      2*k
		 * ----------- = ------------ * x^       / -------- / x^
		 *  term[k]       (2*(k+1))!                (2*k)!
		 *                     (2*k)!       x^(2*k+2)
		 *             = -1 * ---------- * ----------
		 *                     (2*k+2)!     x^(2*k)
		 *                            1
		 *             = -1 * ----------------- * x*x
		 *                     (2*k+1)*(2*k+2)
		*/
		term *= ( neg_x_x / (2*k+1) / (2*k+2) ); 
		k += 1;
	} while ( abs(term) >= eps*abs(series) );
	do {
		seriesR += term;
		term *= ( neg_x_x / (2*k+1) / (2*k+2) ); 
		k += 1;
	} while ( abs(term) >= epsR*abs(seriesR) );

	return series + seriesR;
}

/**
 * tangent function
 *   ...
 * 
 * @param x 
 *   (-pi/2, pi/2)
 * 
 * @return
 *   (-inf, inf)
 * 
 * @derivative
 *   differentiate(tan,x) = sec(x)*sec(x)
 * 
 * @integral
 *   integrate(tan(x),dx) = -log(cos(x)) + constant
 * 
 * @alternative
 *   tan(x) = sin(x)/cos(x)
 * 
 *   tan(x) = sin(x*2)/(cos(x*2) + 1)
*/
template<typename Number> requires std::floating_point<Number>
Number tan(const Number& x, const circular_constant<Number>& pi = circular_constant<Number>()) {
	return sin(x,pi) / cos(x,pi);
/**
 * @alternative of 'maclaurin_series'
 *   tan(x) = ....
 * 
 * @link
 *   "https://math.stackexchange.com/questions/2098941/bernoulli-numbers-taylor-series-expansion-of-tan-x"
 *   "https://math.stackexchange.com/questions/1546539/maclaurin-polynomial-of-tanx"
*/ 
}

/**
 * inverse sine function:
 *   use maclaurin_series
 * 
 * @param x 
 *   [-1, 1]
 * 
 * @return 
 *   [-pi/2, pi/2]
 * 
 * @derivative 
 *   differentiate(asin,x) = 1/sqrt(1-x*x)
 * 
 * @integral
 *   integate(asin(x), dx) = x*asin(x) + sqrt(1 - x*x) + constant
 * 
 * @alternative
 *   asin(x) = -asin(-x)   :odd function
 * 
 *   asin(x) = ( pi/2 - asin(sqrt(1-x*x)) ) * sign(x)
 * 
 *   asin(x) = integrate<0,x>( 1/sqrt(1 - t*t), dt )   :integral form
 * 
 *   asin(x) = sum<k=0,inf>( fact2(2*k-1)/fact2(2*k) * pow(x,2*k+1)/(2*k+1) )  :maclaurin_series
*/
template<typename Number> requires std::floating_point<Number>
Number asin(const Number& x, const circular_constant<Number>& pi = circular_constant<Number>()) {
/** inverse sine function details
 * 
 * @alternative for 'maclaurin_series'
 *                                   1
 *   asin(x) = integral<0,x>( ---------------, dt )                                                      :   f(x) = integral( df(x), dx )
 *                             sqrt(1 - t*t)
 *
 *                                     -0.5
 *           = integral<0,x>( (1 + -t*t)^  , dt )                                                        :1. expansion by 'binomial-series'
 *
 *                                           -0.5 * -1.5 * ... * (-0.5 - k + 1)        k
 *           = integral<0,x>( sum<k=0,inf>( ------------------------------------*(-t*t)^ ), dt )
 *                                                          k!
 *
 *                            -0.5 * -1.5 * ... * (-0.5 - k + 1)                         k
 *           = sum<k=0,inf>( ------------------------------------ * integral<0,x>( (-t*t)^, dt ) )       :2. ... and simplify
 *                                           k!
 *
 *                            -0.5 * -1.5 * ... * (-0.5 - k + 1)        k                   2*k
 *           = sum<k=0,inf>( ------------------------------------ * (-1)^ * integral<0,x>( t^  , dt ) )  :   (-t*t)^k = (-1*t*t)^k
 *                                           k!
 *
 *                            -0.5 * -1.5 * ... * (-0.5 - k + 1)        k    x^(2*k+1)
 *           = sum<k=0,inf>( ------------------------------------ * (-1)^ * ----------- )
 *                                           k!                                2*k+1
 *
 *                             0.5 * 1.5 * ... * abs(-0.5 - k + 1)     x^(2*k+1)
 *           = sum<k=0,inf>( -------------------------------------- * ----------- )                      :   (-1)^k * sign(numerator,k) == 1
 *                                           k!                          2*k+1
 *
 * @alternative for 'maclaurin_series' another form
 * 
 *                            1.0 * 3.0 * ... * 2*k-1       x^(2*k+1)
 *           = sum<k=0,inf>( --------------------------- * ----------- )                                 :   numerator and denominator multiply by 2
 *                                2 * 4 * ... * 2*k           2*k+1
 *
 *                            (2*k-1)!!     x^(2*k+1)
 *           = sum<k=0,inf>( ----------- * ----------- )
 *                             (2*k)!!        2*k+1
*/
	assert( abs(x) <= 1 );
	if ( x < 0 ) {// convergence optimize
		return -asin(abs(x), pi);
	} else if ( x == 0 ) {// not convergence
		return static_cast<Number>(0);
	} else if ( x > 0.625 ) {// convergence slow, from Netlib or GCC
		return static_cast<Number>(pi/2 - asin( sqrt((1 - x)/2), pi )*2);
	}

	// sum maclaurin_series
	const Number eps = static_cast<Number>(0.01);
	Number series = 0;
	const Number epsR = std::numeric_limits<Number>::epsilon();
	Number seriesR = 0;

	const Number x_x = x * x;
	Number term = x;
	Number k2 = 0;
	do {
		series += term;
		/**
		 *  term[k+1]     1 * 3 * ... * (2*(k+1)-1)     x^(2*(k+1)+1)     1 * 3 * ... * (2*k-1)     x^(2*k+1)
		 * ----------- = --------------------------- * --------------- / ----------------------- / -----------
		 *  term[k]       2 * 4 * ... * (2*(k+1))         2*(k+1)+1       2 * 4 * ... * (2*k)         2*k+1
		 *           
		 *                1 * 3 * ... * (2*k+1)     2 * 4 * ... * (2*k)       x^(2*k+3)     2*n+1
		 *             = ----------------------- * ----------------------- * ----------- * -------
		 *                1 * 3 * ... * (2*k-1)     2 * 4 * ... * (2*k+2)     x^(2*k+1)     2*n+3         
		 *
		 *                2*k+1           2*k+1
		 *             = ------- * x^2 * -------
		 *                2*k+2           2*k+3
		*/
		term *= (k2+1)/(k2+2) * (k2+1)/(k2+3) * x_x;
		k2 += 2;
	} while ( term >= eps*series );
	do {
		seriesR += term;
		term *= (k2+1)/(k2+2) * (k2+1)/(k2+3) * x_x;
		k2 += 2;
	} while ( term >= epsR*seriesR );

	return series + seriesR;
}

/**
 * inverse cosine function:
 *   use maclaurin_series
 *
 * @param x 
 *   [-1, 1]
 *
 * @return 
 *   [0, pi]
 * 
 * @alternative
 *   acos(x) = pi/2 - asin(x)
*/
template<typename Number> requires std::floating_point<Number>
Number acos(const Number& x, const circular_constant<Number>& pi = circular_constant<Number>()) {
	return static_cast<Number>( pi/2 - asin(x,pi) );
}

/**
 * inverse tangent function
 * 
 * @param x 
 *   (-inf, inf)
 * 
 * @return
 *   [-pi/2, pi/2]
 * 
 * @Euler's XXXX-series
 *                            2^(2*k) * k! * k!       x^(2*k + 1)
 *   atan(x) = sum<k=0,inf>( ------------------- * ------------------- )
 *                                (2*k + 1)!        (1 + x*x)^(k + 1)
 * 
 * @line 
 *   "https://en.wikipedia.org/wiki/Inverse_trigonometric_functions"
*/
template<typename Number> requires std::floating_point<Number>
Number atan(const Number& x) {
	if ( x < 0 ) {// optimize convergence, remove abs(...)
		return -atan(abs(x));
	} else if ( x > 0.35 ) {// convergence slow
		return atan(x / (1 + sqrt(1 + x*x))) * 2;
	} else if ( x == 0 ) {// not convergence
		return static_cast<Number>(0);
	}

	assert( 0 < x && x <= 0.35 );
	const Number eps = std::numeric_limits<Number>::epsilon();
	const Number xx_div_1pxx = x*x/(1+x*x);
	Number series = 0;
	Number term = x/(1+x*x);
	Number k = 0;
	do {
		series += term;
		/**
		 *	nth[k+1]     2^(2*(k+1)) * (k+1)! * (k+1)!       x^(2*(k+1) + 1)       2^(2*k) * k! * k!       x^(2*k + 1)
		 * ----------- = ------------------------------- * --------------------- / ------------------- / -------------------
		 *	nth[k]              (2*(k+1) + 1)!             (1 + x*x)^(k+1 + 1)         (2*k + 1)!        (1 + x*x)^(k + 1)
		 *                                                                             
		 *                2^(2*k+2) * (k+1)! * (k+1)!     (2*k + 1)!     x^(2*k + 3)     (1+x*x)^(k + 1)
		 *             = ----------------------------- * ------------ * ------------- * -----------------
		 *                2^(2*k) * k! * k!               (2*k + 3)!     x^(2*k + 1)     (1+x*x)^(k + 2)
		 * 
		 *                                            1                     1
		 *             = 2^2 * (k+1) * (k+1) * ----------------- * x^2 * -------
		 *                                      (2*k+2)*(2*k+3)           1+x*x
		 * 
		 *                 4*(k+1)*(k+1)       x*x
		 *             = ----------------- * -------
		 *                (2*k+2)*(2*k+3)     1+x*x
		*/
		term *= ( 4*(k+1)*(k+1) / (2*k+2)/(2*k+3) * xx_div_1pxx );
		k += 1;
	} while ( term >= eps * series );
		
	return series;
}

/**
 * inverse tangent function for two arguments
 * @return (-pi, pi]
*/
template<typename Number> requires std::floating_point<Number>
Number atan2(const Number& y, const Number& x, const circular_constant<Number>& pi = circular_constant<Number>()) {
	if (x > 0) {
		return atan(y / x);
	} else if (x < 0 && y >= 0) {
		return atan(y / x) + pi;
	} else if (x < 0 && y < 0) {
		return atan(y / x) - pi;
	} else if (x == 0 && y > 0) {
		return static_cast<Number>( pi/2 );
	} else if (x == 0 && y < 0) {
		return static_cast<Number>( -pi/2 );
	} else  /* x == 0 && y == 0 */ {
		return std::numeric_limits<Number>::signaling_NaN();
	}

	/*if (x > 0) {
		return atan(y / (sqrt(x*x + y*y) + x)) * 2;
	} else if (x <= 0 && y != 0) {
		return atan((sqrt(x*x + y*Y) - x) / y) * 2;
	} else if (x < 0 && y == 0) {
		return pi;
	} else {
		return std::numeric_limits<Number>::signaling_NaN();
	}*/
}


#ifndef __remove_physics_function__
template<typename Number>
Number cyl_bessel_j(Number v, Number z, const circular_constant<Number>& pi = circular_constant<Number>()) {
	if ( z < 0 ) {// non-negative variable
		return std::numeric_limits<Number>::signaling_NaN();
	} else if ( z == 0 ) {// optimize
		return static_cast<Number>(1);
	}

	const Number desirable_iteration_count
		= ceil(log(std::numeric_limits<Number>::epsilon()) / log(Number(0.1))) + 2;

	/**
	 *      term[k+2]     term[k+1]         1            1         pow(z,2)      1        1       pow(z,2)
	 * abs(----------- / -----------) = --------- * ----------- * ---------- / ----- / ------- / ----------
	 *      term[k+1]     term[k]        (k+1)+1     v+(k+1)+1         4        k+1     v+k+1         4
	 *                                k+1         v+k+1
	 *                           = --------- * -----------
	 *                              (k+1)+1     v+(k+1)+1
	 * 
	 *                           = (k+1)/(k+2) * (v+k+1)/(v+k+2)
	 *    conv(v->inf)
	 * 1.0 |     _ ------- - - - - - -
	 *     |  -^
	 *     | ^
	 *     |^ 
	 * 0.5 |* 
	 *     |
	 *     |
	 *     |
	 *     |
	 * 0.0 +- -- -- -- -- -- -- -- -- -- k
	 *          20    40                100  
	*/
	Number zeroseries_ratio0th = abs( z*z/4/(v+1) );
	if ( zeroseries_ratio0th < 0.6 ) {
		/**
		 * 1. Apply relation for 'hypergeometric' function
		 *                                 (-1)^k          1            z
		 * bessel_j<v>(z) = sum<k=0,inf>( -------- * -------------- * (---)^(2*k+v) )
		 *                                   k!       gamma(v+k+1)      2
		 * 
		 *                                 (-1)^k          1           z^2         z
		 * 			          = sum<k=0,inf>( -------- * -------------- * -----^k ) * ---^v
		 *                                   k!       gamma(v+k+1)      4          2
		*/
		const Number eps = static_cast<Number>(0.01);
		Number series = 0;
		const Number epsR = std::numeric_limits<Number>::epsilon();
		Number seriesR = 0;

		const Number zzo4 = z*z/4;
		Number term = 1;
		Number k = 0;
		do {
			series += term;
			/**
			 *  term[k+1]     (-1)^(k+1)             1            z^2           (-1)^k          1           z^2
			 * ----------- = ------------ * ------------------ * -----^(k+1) / -------- / -------------- / -----^k
			 *  term[k]         (k+1)!       gamma(v+(k+1)+1)      4              k!       gamma(v+k+1)      4
			 * 
			 *                (-1)^(k+1)       k!       gamma(v+k+1)     (z^2 / 4)^(k+1)
			 *             = ------------ * -------- * -------------- * -----------------
			 *                 (-1)^k        (k+1)!     gamma(v+k+2)     (z^2 / 4)^k
			 * 
			 *                      1        1       z^2
			 *             = -1 * ----- * ------- * -----
			 *                     k+1     v+k+1      4
			*/
			term *= zzo4 / (k+1) / (v+k+1) * (-1);
			k += 1;
		} while ( abs(term) >= eps*abs(series) );
		assert( z != 0 );
		do {
			seriesR += term;
			term *= zzo4 / (k+1) / (v+k+1) * (-1);
			k += 1;
		} while ( abs(term) >= epsR*abs(seriesR) );
		
		return (series + seriesR) * exp( v*log(z/2) - lgamma(v + 1) );
	}
	
	/**
	 *      term[m+2]     term[m+1]                                                                                           1                     1                                                                           1                 1
	 * abs(----------- / -----------) = (v*v - ((m+1)*2+0.5)*((m+1)*2+0.5))*(v*v - ((m+1)*2+1.5)*((m+1)*2+1.5)) * ------------------------- * ------------- / (v*v - (m*2+0.5)*(m*2+0.5))*(v*v - (m*2+1.5)*(m*2+1.5)) / ----------------- / -------------
	 *      term[m+1]     term[m]                                                                                  ((m+1)*2+1)*((m+1)*2+2)     pow(2*z, 2)                                                               (m*2+1)*(m*2+2)     pow(2*z, 2)
	 *                          (v*v - ((m+1)*2+0.5)*((m+1)*2+0.5))*(v*v - ((m+1)*2+1.5)*((m+1)*2+1.5))         (m*2+1)*(m*2+2)         pow(2*z, 2)
	 *                       = ------------------------------------------------------------------------- * ------------------------- * -------------
	 *                          (v*v - (m*2+0.5)*(m*2+0.5))*(v*v - (m*2+1.5)*(m*2+1.5))                     ((m+1)*2+1)*((m+1)*2+2)     pow(2*z, 2)
	 *                          (v*v - pow(m*2+2.5,2))*(v*v - pow(m*2+3.5,2))     (m*2+1)*(m*2+2)
	 *                       = ----------------------------------------------- * -----------------
	 *                          (v*v - pow(m*2+0.5,2))*(v*v - pow(m*2+1.5,2))     (m*2+3)*(m*2+4)
	 *   conv(v->inf), the break at ( v*v < pow(m*2+2.5,2) ), m = (v-2.5)/2
	 * 10.0|                                ^_
	 *     |                                ^ _
	 *     |                                ^   - _ _ _ _
	 * 1.0 |     _ ------- - - - - - - - - -^
	 *     |   -^
	 *     |  ^
	 *     | ^ 
	 * 0.5 |^ 
	 *     |*
	 *     |
	 *     |
	 *     |
	 * 0.0 +- -- -- -- -- -- -- -- -- -- m
	 *          20    40                100
	*/
	Number infseries_max_iteration_count = (v-2.5)/2;
	Number infseries_ratio0th = abs( (v*v - 0.5*0.5)*(v*v - 1.5*1.5) / (z*z*8) );
	if ( (infseries_max_iteration_count >= desirable_iteration_count && infseries_ratio0th < 3.8)
		|| /* term[0th] time */infseries_ratio0th < std::numeric_limits<Number>::epsilon() )
	{
		/**
		 * cyl_bessel_j(v,z)*2
		 * 
		 * 1. Apply relation, cyl_bessel_j(v,z)*2 = cyl_hankel_1(v,z)+cyl_hankel_2(v,z)
		 * 
		 *              2                                                  factorial(v-0.5-(m-1), m)*factorial(v+0.5,m)        1i
		 *   ~= sqrt(------) * exp(1i*(z - v*pi/2 - pi/4)) * sum<m=0,p-1>(----------------------------------------------*pow(-----,m))
		 *            pi*z                                                                  factorial(m)                      2*z
		 *                2                                                   factorial(v-0.5-(m-1), m)*factorial(v+0.5,m)       -1i
		 *      + sqrt(------) * exp(-1i*(z - v*pi/2 - pi/4)) * sum<m=0,p-1>(----------------------------------------------*pow(-----,m))
		 *              pi*z                                                                   factorial(m)                      2*z
		 *
		 * Simplify, exp(1i*A)*Ar + exp(-1i*A)*Br = cos(A)*Ar+i*sin(A)*Ar + cos(A)*Br-i*sin(A)*Br = cos(A)*(Ar+Br) + i*sin(A)*(Ar-Br)
		 *
		 *              2                                                factorial(v-0.5-(m-1), m)*factorial(v+0.5,m)          1
		 *   ~= sqrt(------) * ( cos(z - v*pi/2 - pi/4) * sum<m=0,p-1>( ---------------------------------------------- * pow(-----,m) * (pow(1i,m) + pow(-1i,m)) )
		 *            pi*z                                                                factorial(m)                        2*z
		 *                                                                   factorial(v-0.5-(m-1), m)*factorial(v+0.5,m)          1
		 *                       + i*sin(z - v*pi/2 - pi/4) * sum<m=0,p-1>( ---------------------------------------------- * pow(-----,m) * (pow(1i,m) - pow(-1i,m)) )
		 *                                                                                    factorial(m)                        2*z
		 *                     )
		 * 
		 * Simplify, ...
		 *
		 *              2                                                    factorial(v-0.5-(m*2-1), m*2)*factorial(v+0.5,m*2)          1
		 *   ~= sqrt(------) * ( cos(z - v*pi/2 - pi/4) * sum<m=0,(p-1)/2>( ---------------------------------------------------- * pow(-----,m*2) * pow(-1,m) * 2 )
		 *            pi*z                                                                    factorial(m*2)                            2*z
		 *                                                                     factorial(v-0.5-(m*2+1-1), m*2+1)*factorial(v+0.5,m*2+1)          1
		 *                       - sin(z - v*pi/2 - pi/4) * sum<m=0,(p-1)/2>( ---------------------------------------------------------- * pow(-----,m*2+1) * pow(-1,m) * 2 )
		 *                                                                                         factorial(m*2+1)                               2*z
		 *                     )
		 * 
		 * Simplify, factorial(v-0.5-(m*2-1), m*2)*factorial(v+0.5,m*2) = 1*(v-0.5)*(v-1.5)*... * 1*(v+0.5)*(v*1.5)*... = 1*(v*v-0.5*0.5)*(v*v-1.5*1.5)*...
		 *
		 *              2                                                    1*(v*v-0.5*0.5)*(v*v-1.5*1.5)*...*(v*v-(m*2-0.5)*(m*2-0.5))          1
		 *   ~= sqrt(------) * ( cos(z - v*pi/2 - pi/4) * sum<m=0,(p-1)/2>( ------------------------------------------------------------- * pow(-----,m*2) * pow(-1,m) * 2 )
		 *            pi*z                                                                    factorial(m*2)                                     2*z
		 *                                                                     (v*v-0.5*0.5)*(v*v-1.5*1.5)*...*(v*v-(m*2+0.5)*(m*2+0.5))          1
		 *                       - sin(z - v*pi/2 - pi/4) * sum<m=0,(p-1)/2>( ----------------------------------------------------------- * pow(-----,m*2+1) * pow(-1,m) * 2 )
		 *                                                                                         factorial(m*2+1)                              2*z
		 *                     )
		 * 
		 * cyl_bessel_j(v,z)
		 * 
		 *              2                                                    1*(v*v-0.5*0.5)*(v*v-1.5*1.5)*...*(v*v-(m*2-0.5)*(m*2-0.5))          1
		 *   ~= sqrt(------) * ( cos(z - v*pi/2 - pi/4) * sum<m=0,(p-1)/2>( ------------------------------------------------------------- * pow(-----,m*2) * pow(-1,m) )
		 *            pi*z                                                                    factorial(m*2)                                     2*z
		 *                                                                     (v*v-0.5*0.5)*(v*v-1.5*1.5)*...*(v*v-(m*2+0.5)*(m*2+0.5))          1
		 *                       - sin(z - v*pi/2 - pi/4) * sum<m=0,(p-1)/2>( ----------------------------------------------------------- * pow(-----,m*2+1) * pow(-1,m) )
		 *                                                                                         factorial(m*2+1)                              2*z
		 *                     )
		*/
		const Number eps = std::numeric_limits<Number>::epsilon();
		Number jreal  = 0;
		Number jimag  = 0;

		const Number vv = v * v;
		const Number z2 = z*2;
		Number term = 1;
		Number m2 = 0;
		do {
			/**
			 * @jreal
			 *  term[0] = 1
			 * 
			 *  term[m+1]     1*(v*v-0.5*0.5)*(v*v-1.5*1.5)*...*(v*v-((m+1)*2-0.5)*((m+1)*2-0.5))          1                               1*(v*v-0.5*0.5)*(v*v-1.5*1.5)*...*(v*v-(m*2-0.5)*(m*2-0.5))          1
			 * ----------- = --------------------------------------------------------------------- * pow(-----,(m+1)*2) * pow(-1,(m+1)) / ------------------------------------------------------------- / pow(-----,m*2) / pow(-1,m)
			 *  term[m]                             factorial((m+1)*2)                                    2*z                                               factorial(m*2)                                     2*z
			 *                1*(v*v-0.5*0.5)*(v*v-1.5*1.5)*...*(v*v-((m+1)*2-0.5)*((m+1)*2-0.5))     factorial(m*2)         pow(2*z, m*2)         pow(-1,(m+1))
			 *             = --------------------------------------------------------------------- * -------------------- * ------------------- * ---------------
			 *                1*(v*v-0.5*0.5)*(v*v-1.5*1.5)*...*(v*v-(m*2-0.5)*(m*2-0.5))             factorial((m+1)*2)     pow(2*z, (m+1)*2)     pow(-1,m)
			 *                                                                                 1                 1
			 *             = (v*v - (m*2+0.5)*(m*2+0.5))*(v*v - (m*2+1.5)*(m*2+1.5)) * ----------------- * ------------- * -1
			 *                                                                          (m*2+1)*(m*2+2)     pow(2*z, 2)
			 * 
			 * 
			 * @jimag
			 *  term[m+1]     (v*v-0.5*0.5)*(v*v-1.5*1.5)*...*(v*v-((m+1)*2+0.5)*((m+1)*2+0.5))          1                                 (v*v-0.5*0.5)*(v*v-1.5*1.5)*...*(v*v-(m*2+0.5)*(m*2+0.5))          1
			 * ----------- = ------------------------------------------------------------------- * pow(-----,(m+1)*2+1) * pow(-1,(m+1)) / ----------------------------------------------------------- / pow(-----,m*2+1) / pow(-1,m) )
			 *  term[m]                             factorial((m+1)*2+1)                                2*z                                                   factorial(m*2+1)                               2*z
			 *                (v*v-0.5*0.5)*(v*v-1.5*1.5)*...*(v*v-((m+1)*2+0.5)*((m+1)*2+0.5))     factorial(m*2+1)         pow(2*z, m*2+1)         pow(-1,(m+1))
			 *             = ------------------------------------------------------------------- * ---------------------- * --------------------- * ---------------
			 *                (v*v-0.5*0.5)*(v*v-1.5*1.5)*...*(v*v-(m*2+0.5)*(m*2+0.5))             factorial((m+1)*2+1)     pow(2*z, (m+1)*2+1)     pow(-1,m)
			 *                                                                                  1                 1
			 *             =  (v*v - (m*2+1.5)*(m*2+1.5))*(v*v - (m*2+2.5)*(m*2+2.5)) * ----------------- * ------------- * -1
			 *                                                                           (m*2+2)*(m*2+3)     pow(2*z, 2)
			 * term[0] = (v*v-0.5*0.5)/1/(2*z)
			*/
			jreal += term;
			term *= ( vv - pow(m2+0.5,2) ) / (m2+1) / z2;
			jimag += term;
			term *= ( vv - pow(m2+1.5,2) ) / (m2+2) / z2 * (-1);
			m2 += 2;
		} while ( abs(term) >= eps*abs(jreal) );// convergence very fast, not need seriesR

		/** const Number phase = z - v*pi/2 - pi/4;
		 *  return (cos(phase)*(jreal) - sin(phase)*(jimag)) * sqrt(Number(2)/(pi*z));
		*/
		const Number phase_z = z;
		const Number phase_v = fmod(v/2 + 0.25, Number(2)) * pi;//(full precision)
		Number sz = sin(phase_z);
		Number cz = cos(phase_z);
		Number sv = sin(phase_v);
		Number cv = cos(phase_v);
		Number sin_phase = sz * cv - sv * cz;
		Number cos_phase = cz * cv + sz * sv;
		return (cos_phase*jreal - sin_phase*jimag) * sqrt(Number(2)/(pi*z));
	}

	// continued fraction ...
	if ( v > z ) {
		// ...
	} 
	
	{
		/** solve [zeroseries_ratio0th(z, desirable order 'v') = 0.5] */
		Number back_n = ceil(z*z/2 - 1) - floor(v);
		assert( back_n > 0 );
		
		/** solve [infseries_ratio0th(z, desirable order 'v') = 3.5]] */
		Number fward_n = ceil(v) - floor(pow(z*z*28, 0.25));
		if ( fward_n < 0 || (v-fward_n-1-2.5)/2 < desirable_iteration_count ) {
			fward_n = std::numeric_limits<Number>::quiet_NaN();
		}

		if ( fward_n < back_n )
		{
			/** forward_recur J[v](z)*(v*2/z) - J[v-1] = J[v+1] */
			Number Jvm1 = cyl_bessel_j(v-fward_n-1, z, pi);
			Number Jv = cyl_bessel_j(v-fward_n, z, pi);
			Number Jv1;
			for ( ; fward_n > 0; --fward_n) {
				Jv1 = Jv*(v-fward_n)*2/z - Jvm1;
				Jvm1 = Jv;
				Jv = Jv1;
			}
			return Jv1;
		}
		else 
		{
			/** backward_recur J[v](z)*(v*2/z) - J[v+1] = J[v-1] */
			Number Jv1 = cyl_bessel_j(v+back_n+1, z, pi);
			Number Jv = cyl_bessel_j(v+back_n, z, pi);
			Number Jvm1;
			for ( ; back_n > 0; --back_n) {
				Jvm1 = Jv*(v+back_n)*2/z - Jv1;
				Jv1 = Jv;
				Jv = Jvm1;
			}
			return Jvm1;
		}
	}


	//auto bessel_lvz = [pi](double v, double z) {
	//	using Number = std::complex<double>;
	//	//using Number = double;
	//	using namespace::std;

	//	Number y = z / v;
	//	//assert(y >= 0 && y != 1);

	//	Number zeta;
	//	Number sqrt1myy = 1.0/sqrt(1.0-y*y);
	//	if ( abs(y) > 1 ) {  
	//      zeta = - pow(1.5, 2/3.0) * pow(sqrt(y*y - 1.0) - acos(1.0/y), 2/3.0);
	//	} else if ( abs(y) <= 1 ) {
	//		zeta = pow(3.0/2, 2/3.0) * pow( log((1.0 + sqrt(1.0 - y*y))/y) - sqrt(1.0 - y*y), 2/3.0);
	//	}
	//	Number zeta32 = sqrt(zeta) * zeta;

	//	std::vector<Number> lamb = std::vector<Number>(50, Number(1));
	//	std::vector<Number> mu = std::vector<Number>(50, Number(1));
	//	for (size_t s = 0; s != 50-1; ++s) {
	//		lamb[s+1] = ( lamb[s]*(3*s + 0.5)*(3*s + 1.5)*(3*s + 2.5)/( 9*(2*s + 1)*(2*s + 2) ) );
	//	}
	//	for (int s = 0; s != 50; ++s) {
	//		mu[s] = - lamb[s]*(s*6+1)/(s*6-1);
	//	}
	//	std::vector<Number> U = std::vector<Number>(50, Number(0)); U[0] = 1;
	//	{
	//		std::vector<double> prev_p;
	//		std::vector<double> p = { 1 };
	//		for (size_t k = 1; k != 50; ++k) {
	//			/**
	//			 * debye[nth+1](x) = 1/2*pow(x,2)*(1 - pow(x,2)) * d/dx*debye(x) + 1/8 * integral<t,0,x>( (1 - pow(t,2))*debye(t) )
	//			 * 
	//			 * debye[nth+1](x) = 1/2*pow(x,2) * d/dx*debye[nth](x)
	//			 *                   - 1/2*pow(x,4) * d/dx*debye[nth](x)
	//			 *                   + 1/8*integral<t,0,x>( debye[nth](t) )
	//			 *                   - 1/8*5*integral<t,0,x>( pow(t,2)*debye[nth](t) )
	//			*/
	//			prev_p = p;
	//			p = std::vector<double>(prev_p.size() + 3, double(0));
	//			for (size_t i = 0; i != prev_p.size(); ++i) {
	//				size_t power = i;
	//				double multiplier = prev_p[i];

	//				if ( power != 0 ) {
	//					size_t dFdx_power = power - 1;
	//					double dFdx_multiplier = multiplier * power;
	//					// 1/2*pow(x,2) * d/dx*debye[nth](x)
	//					p[dFdx_power+2] += dFdx_multiplier * 0.5;
	//					// - 1/2*pow(x,4) * d/dx*debye[nth](x)
	//					p[dFdx_power+4] -= dFdx_multiplier * 0.5;
	//				}

	//				// 1/8*integral<t,0,x>( debye[nth](t) )
	//				p[power+1] += multiplier/(power + 1) * 0.125;
	//		
	//				// - 1/8*5*integral<t,0,x>( pow(t,2)*debye[nth](t) )
	//				p[power+2+1] -= multiplier/(power+2 + 1) * 0.625;
	//			}

	//			Number& result = U[k];
	//			auto multiplier = p.rbegin();
	//			auto first_multiplier = p.rend();
	//			for ( ; multiplier != first_multiplier; ++multiplier) {
	//				result = result*sqrt1myy + (*multiplier);
	//			}
	//		}
	//	}

	//	const double eps = std::numeric_limits<double>::epsilon();
	//	Number seriesAi = 0;
	//	Number seriesAiprime = 0;

	//	{
	//		size_t k = 0;
	//		Number ak = 0;
	//		for (int s = 0; s <= k*2; s += 1) {
	//			ak += mu[s] / std::pow(zeta32,s) * U[k*2-s];
	//		} 
	//		Number term = ak /* / pow(v,2*0)*/ ;
	//		do {
	//			seriesAi += term;

	//			k += 1;

	//			Number ak_prev = ak;
	//			ak = 0;
	//			for (int s = 0; s <= k*2; s += 1) {
	//				auto t = mu[s] / std::pow(zeta32,s) * U[k*2-s];
	//				ak += t;
	//			} 
	//			term *= (ak/ak_prev) / (v*v);
	//		} while ( abs(term) >= eps*abs(seriesAi) );
	//	}

	//	{
	//		size_t k = 0;
	//		Number bk = 0;
	//		for (int s = 0; s <= k*2+1; s += 1) {
	//			bk += lamb[s] / std::pow(zeta32,s) * U[k*2+1-s];
	//		} bk *= -pow(zeta, -0.5);
	//		Number term = bk /* / pow(v,2*0)*/ ;
	//		do {
	//			seriesAiprime += term;

	//			k += 1;

	//			Number bk_prev = bk;
	//			bk = 0;
	//			for (int s = 0; s <= k*2+1; s += 1) {
	//				bk += lamb[s] / std::pow(zeta32,s) * U[k*2+1-s];
	//			} bk *= -pow(zeta, -0.5);
	//			term *= (bk/bk_prev) / (v*v);
	//		} while ( abs(term) >= eps*abs(seriesAiprime) );
	//	}

	//	auto term1 = boost::math::airy_ai(pow(v,2.0/3) * real(zeta))/pow(v,1.0/3) * seriesAi;
	//	auto term2 = boost::math::airy_ai_prime(pow(v,2.0/3) * real(zeta))/pow(v,5.0/3) * seriesAiprime;
	//	auto term3 = pow(zeta*4.0/(1.0-y*y),0.25);
	//	return (term1 + term2) * term3;
	//};
}

// undeterminant

// hypergeometric series( Gauss )
template<typename Number>
Number hypergeometric_series(Number a, Number b, Number c, Number z, size_t n = 1024){
	/* hypergeometric series( Gauss )
									a*b     a*(a+1)*b*(b+1)       a*(a+1)*(a+2)*b*(b+1)*(b+2)               fact(i,a)*fact(i,b)
	* series = 1 + ---*z + ---------------*z*z + ---------------------------*z*z*z + ... + -------------------*pow(z,i)
									1*c       1*2*c*(c+1)             1*2*3*c*(c+1)*(c+2)                    fact(i)*fact(i,c)
		abs(z) < 1, absolutely converges
		abs(z) == 1, requires real(c-a-b) > 0

	* application:
										dd'y'                 d'y'
		solve [ z*(1-z)*----- + (c-(a+b+1)*z)*---- - a*b*y = 0 ], result'y' is hypergeometric_series(a,b,c,z)
											dzz                   dz
		asin(z) = z * hypergeometric_series(0.5, 0.5, 1.5, z*z)
		atan(z) = z * hypergeometric_series(0.5, 1.0, 1.5, -z*z)
	*/
	
	assert( abs(z) < 1 || (abs(z) == 1 && (c-a-b) > 0) );

	const Number eps = std::numeric_limits<Number>::epsilon();
	Number result = 0;
	Number term = 1;
	size_t i = 0;
	do {
		result += term;
		term = term * (a+i)*(b+i)/(1+i)/(c+i)*z;
	} while (++i != n && abs(term) >= eps*max(abs(result),eps));

	return result;
}

#if 0
// gamma function( Euler )
template<typename Number>
Number gamma(Number z, Number r, size_t n, Number pi, Number e) {
	/* gamma function approximation( Lanczos )
																 z   -t
	gamma(z+1) = integral[0,inf]( t^ * e^, dt )


	Substitution 't' = u*a, a != 0
													z   -u*a
	= integral[0,inf]( (u*a)^ * e^  , dt )              :t = u*a
															z   -u*a
	= integral[0/a,inf/a]( (u*a)^ * e^ * a, a*du )      :dt = a*du, u = t/a
											z    z   -u*a               z+1
	= integral[0,inf]( u^ * a^ * e^   * a, du ) * a^    :simplify
											z   -u*a           z+1
	= integral[0,inf]( u^ * e^  , du ) * a^             :simplify


	Replacing 'a' by z+r+1, since 'z' is a constant with process of integration
											z   -u*(z+r+1)                z+1
	= integral[0,inf]( u^ * e^        , du ) * (z+r+1)^


	Substituation 'u' = 1 - log(v)
															 z    -(1-log(v))*(z+r+1)                z+1
	= integral[0,inf]( (1-log(v))^ * e^                  , du ) * (z+r+1)^           :u = 1 - log(v)

															 z    -(z+r+1)    log(v)*(z+r+1)                z+1
	= integral[0,inf]( (1-log(v))^ * e^        * e^             , du ) * (z+r+1)^    :simplify exponential distribution

															 z    log(v)*(z+r+1)                z+1   -(z+r+1)
	= integral[0,inf]( (1-log(v))^ * e^             , du ) * (z+r+1)^   * e^         :simplify move out integration

															 z    z    r                    z+1   -(z+r+1)
	= integral[0,inf]( (1-log(v))^ * v^ * v^ * v, du ) * (z+r+1)^   * e^             :simplify

																						 z    z    r                         z+1   -(z+r+1)
	= integral[exp(1-0),exp(1-inf)]( (1-log(v))^ * v^ * v^ * v, -1/v*dv ) * (z+r+1)^   * e^         :du = -1/v * dv, v = exp(1 - u)

														 z    z    r                           z+1   -(z+r+1)
	= integral[e,0]( (1-log(v))^ * v^ * v^ * v * -1/v, dv ) * (z+r+1)^   * e^          :simplify

																 z    r                           z+1   -(z+r+1)
	= integral[e,0]( ((1-log(v))*v)^ * v^ * v * -1/v, dv ) * (z+r+1)^   * e^         :simplify

																 z    r                     z+1   -(z+r+1)
	= integral[e,0]( ((1-log(v))*v)^ * v^ * -1, dv ) * (z+r+1)^   * e^         :simplify

																 z    r                z+1   -(z+r+1)
	= integral[0,e]( ((1-log(v))*v)^ * v^, dv ) * (z+r+1)^   * e^         :simplify, note: r > 0 may be 1

	*/

	z = z - 1;
		
		/*
					k                                                                             k*pi*x                                  
alpha = -1^ * (2/pi) * Integral[-1,1]( ( alpha(0,r)*0.5 + sum[k=1,inf]( alpha(k,r)*cos(--------) ) )
																																												 pi/2
																															2*j
																			 * sum[j=0,k]( C2j2k * x^   )
																								1
																			 * ---------------
																					sqrt(1 - x*x)
																			,dx )

			k                                                                             k*pi*x
alpha = -1^ * (2/pi) * Integral[-1,1]( ( alpha(0,r)*0.5 + sum[k=1,inf]( alpha(k,r)*cos(--------) ) )
																						 pi/2
											 k                             2 j
										 * -1^ * sum[j=0,k]( C2j2k * (1 - x^)^ )
												1
										 * ---------------
											sqrt(1 - x*x)
										,dx )

....
							 k                 r                     j       fact(k+j-1)            e     j+0.5
alpha(k,r) = -1^ * sqrt(2/pi) * e^ * k * Sum[j=0,k]( -1^ * ------------------- * (---------)^     )
																														fact(k-j)*fact(j)      j+r+0.5
*/
	auto alpha = [pi,e](size_t k, Number r) {
		if (k == 0) {
			return sqrt(2*e / (pi*(r+0.5))) * exp(r);
		}
		return pow(-1,k) * sqrt(2/pi) * exp(r) * k
			* sum(size_t(0),k,[k,r,e](size_t j){ return fact(k+j-1)/fact(k-j)/fact(j) * pow(e/(j+r+0.5),j+0.5) * pow(-1,j); });
	};

	Number Fr = alpha(0,r) / 2;
	Number c = z/(z+1);
	for (size_t k = 1; k != n; ++k) {
		Fr += alpha(k,r) * c;
		c = c * (z-k) / (z+k+1);
	}

	return pow(z+r+0.5,z+0.5) * exp(-(z+r+0.5)) * sqrt(pi*2) * Fr;
}
#endif
#endif
}

/// 
///@readme
/// Template of these functions can match "any", will it conflict with other modules?
/// No, no conflict. basic-type{int,float,double,...} have typed functions, 
///  arguments of array-type's functions have least 1 type be array-container,
///  so only floating<..> can be matched.
/// Will such conflict in the future?
/// No, no conflict in the future. I pay great attention to practicality, 
///  faster speed or less space, in any case, 
///  newly defined structures for these purposes should have 
///  newly specific functions.
///@note
/// template function we have
/// 	sin(_Ty x) : least 1 times convertion
/// 	sin(float x)
/// 	sin(double x)
/// 	sin(complex<_Ty> x)
/// test 
/// 	input float : { 1 times, 0 times, 1 times, Notconvert }
/// 	input double : { 1 times, 1 times, 0 times, Notconvert }
/// 	input Myclass : { 1 times, Notconvert, Notconvert, Notconvert }
/// 	input complex<_Ty> : { 1 times, 0 times, 0 times, 0 times!!!!!!!!!!! }
/// so Container<_Ty>: This is Container idea, matching has a priority level.