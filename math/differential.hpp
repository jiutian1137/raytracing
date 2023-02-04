#pragma once
/*{ "clmagic/calculation/fundamental/differential":{ 
"Mathematician":[ 
  "Brook Taylor", 
  "C.J.F. Ridders" 
] ,
"Reference":{
  "Library": "boost::math::differentiation::differentiation" ,
  "Book": "Numerical Analysis" ,
  "Url":"http://www.holoborodko.com/pavel/about/"
},
"Author": "LongJiangnan",
  
"License": "Please identify Mathematician"
} }*/


#include <cmath>// abs, sqrt, pow, nextafter
#include <numeric>// std::numeric_limits<T>::epsilon, std::numeric_limits<T>::max
namespace math 
{

/* two point forward difference
* ...
*/
template<typename Function, typename Real>
Real f2_difference(Function f, Real x, Real* error = nullptr) {
  Real eps = std::numeric_limits<Real>::epsilon();
  Real y = f(x);
  Real h = sqrt((y==0 ? eps : abs(y)*eps) * 4);
  if ( (x + h) - x == 0 ) {
    h = nextafter(x, std::numeric_limits<Real>::max()) - x; 
  }

  Real y0 = f(x);
  Real yh = f(x + h);
  Real dfx = (yh - y0) / h;
  if (error) {
    // math_error: h/2*ddf(x)
    *error = h / 2 * abs(yh - h * dfx - y0/* + dddf(x) + round_error */);
    // round_error
    *error += (abs(yh) * eps + abs(y0) * eps) / h;
  }
  return dfx;
}

/* three point centered difference
* about 'error', we ignore low order terms in 'formula'
* about 'h', we remove relation between fxxx(x) and Error
  
* formula
                f(x)                fx(x)              fxx(x)              fxxx(x)
(1) f(x+h) = --------*pow(h,0) + --------*pow(h,1) + --------*pow(h,2) + --------*pow(h,3) +  ...    ,apply 'Taylor theorem'
              fact(0)             fact(1)             fact(2)             fact(3)
                f(x)                 fx(x)                fxx(x)              fxxx(x)
(2) f(x-h) = --------*pow(-h,0) + --------*pow(-h,1) + --------*pow(-h,2) + --------*pow(-h,3) +  ...    ,apply 'Taylor theorem'
              fact(0)              fact(1)               fact(2)             fact(3)
                                            fxx(x)              fxxx(x)                                      fxx(x)              fxxx(x)
                          f(x) + fx(x)*h + --------*pow(h,2) + --------*pow(h,3) +  ...  - f(x) + fx(x)*h - --------*pow(h,2) + --------*pow(h,3) + ...
      f(x+h) - f(x-h)                       fact(2)             fact(3)                                      fact(2)             fact(3)
(3) ----------------- = -------------------------------------------------------------------------------------------------------------------------------
            2*h                                                                 2*h
                                        fxxx(x)
                          2*fx(x)*h + 2*-------*pow(h,3) + ...
                                        fact(3)
                      = --------------------------------------
                                    2*h
                                  fxxx(x)
                      = fx(x) + --------*pow(h,2) + ...
                                  fact(3)
      f(x+h) - f(x-h)     fxxx(x)
    ----------------- - --------*pow(h,2) - ... = fx(x)
            2*h           fact(3)
* computation formula
                    f(x+h)*(1 +- eps) - f(x-h)*(1 +- eps)
c3_difference(x) = --------------------------------------
                                      2*h
                    f(x+h) +- f(x+h)*eps - f(x-h) +- f(x-h)*eps
                  = ---------------------------------------------
                                        2*h
                    f(x+h) - f(x-h)      f(x+h)*eps +- f(x-h)*eps
                  = ----------------- +- --------------------------
                          2*h                      2*h
* mathematical error
Error = abs(ErrorResult - NoErrorResult)
              f(x+h) - f(x-h)     f(x+h) - f(x-h)     fxxx(x)
      = abs(----------------- - ----------------- + --------*pow(h,2) + ...) 
                    2*h                 2*h           fact(3)
              fxxx(x)
      = abs( --------*pow(h,2) + ... )
              fact(3)
* computation error
Error = abs(ErrorResult - NoErrorResult)
              f(x+h) - f(x-h)     f(x+h) - f(x-h)      f(x+h)*eps +- f(x-h)*eps
      = abs( ----------------- - ----------------- +- -------------------------- )
                    2*h                 2*h                       2*h
          abs(f(x+h))*eps + abs(f(x-h))*eps
      <= -----------------------------------
                          2*h
          max( abs(f(x+h)),abs(f(x-h)) ) * (eps + eps)
      <= ----------------------------------------------
                            2*h
          abs(f(x)) * (eps + eps)
      ~<= -------------------------  ,apply 'intermediate theorem'
                    2*h
      ~<= abs(f(x)) * eps / h
* 'h'
                          fxxx(x)
(1) TotalError ~<= abs( --------*pow(h,2) + ... ) + abs(f(x))*eps/h
                          fact(3)
                          fxxx(x)
                ~<= abs( --------*pow(h,2) ) + abs(f(x))*eps/h   ,ignore low order terms
                          fact(3)
                          pow(h,2)
                ~<= abs( --------- ) + abs(f(x))*eps/h   ,remove relatin between fxxx(x) and Error
                          fact(3)
                ~<= pow(h,2)/6 + abs(f(x))*eps/h
(2)
    Error| +            +
          |  +        +
          |   +    +
          |     + 
          +--------------------- h
(2)            d/dh * TotalError == 0   ,find critial point
    h/3 - abs(f(x))*eps/pow(h,2) == 0
                              h/3 == abs(f(x))*eps/pow(h,2)  ,+ abs(f(x))*eps/pow(h,2)
                      pow(h,3)/3 == abs(f(x))*eps           ,* pow(h,2)
                        pow(h,3) == abs(f(x))*eps*3
                                h == pow(abs(f(x))*eps*3, 1/3) ,pow 1/3
* reference 
<<Numerical Analysis>> Timothy Sauer
"https://www.uio.no/studier/emner/matnat/math/MAT-INF1100/h08/kompendiet/diffint.pdf"
*/
template<typename Function, typename Real>
Real c3_difference(Function f, Real x, Real* error = nullptr) {
  Real eps = std::numeric_limits<Real>::epsilon();
  Real y = f(x);
  Real h = pow((y==0 ? eps : abs(y)*eps)*3, Real(1)/3);
  if ( (x + h) - x == 0 ) {
    h = nextafter(x, std::numeric_limits<Real>::max()) - x; 
  }

  Real y_h = f(x + h);
  Real y_m_h = f(x - h);
  if (error) {
    // math_error: pow(h,2)/6 * d3f(x), five_point_centered_difference_third_derivative
    Real y_two_h = f(x + 2 * h);
    Real y_m_two_h = f(x - 2 * h);
    *error = abs((y_two_h - y_m_two_h) - 2 * (y_h - y_m_h)) / (12 * h);
    // round_error: ...
    *error += (abs(y_h) * eps + abs(y_m_h) * eps) / (2 * h);
  }
  return (y_h - y_m_h) / (2 * h);
}

/* five point centered difference
*/
template<typename Function, typename Real>
Real c5_difference(Function f, Real x, Real* error = nullptr) {
  Real eps = std::numeric_limits<Real>::epsilon();
  Real y = f(x);
  Real h = pow((y == 0 ? eps : abs(y)*eps)*Real(11.25), Real(1)/5);
  if ( (x + h) - x == 0 ) {
    h = nextafter(x, std::numeric_limits<Real>::max()) - x; 
  }

  Real y_h = f(x + h);
  Real y_m_h = f(x - h);
  Real y_two_h = f(x + 2 * h);
  Real y_m_two_h = f(x - 2 * h);
  if (error) {
    // math_error: pow(h,4)/30 * dddddf(x), dddddfx=seven_point_centered_difference_fifth_derivative
    Real y_three_h = f(x + 3 * h);
    Real y_m_three_h = f(x - 3 * h);
    *error = abs((y_three_h - y_m_three_h) + 5 * (y_h - y_m_h) - 4 * (y_two_h - y_m_two_h)) / (60 * h);
    // round_error
    *error += (8 * (abs(y_h) + abs(y_m_h)) + (abs(y_two_h) + abs(y_m_two_h))) * eps / (12 * h);
  }
  return (8 * (y_h - y_m_h) - (y_two_h - y_m_two_h)) / (12 * h);
}

/* seven point centered difference
*/
template<typename Function, typename Real>
Real c7_difference(Function f, Real x, Real* error = nullptr) {
  Real eps = std::numeric_limits<Real>::epsilon();
  Real y = f(x);
  Real h = pow((y == 0 ? eps : abs(y)*eps)*385/9, Real(1.0)/7);
  if ( (x + h) - x == 0 ) {
    h = nextafter(x, std::numeric_limits<Real>::max()) - x; 
  }

  Real y_h = f(x + h);
  Real y_m_h = f(x - h);
  Real y_two_h = f(x + 2 * h);
  Real y_m_two_h = f(x - 2 * h);
  Real y_three_h = f(x + 3 * h);
  Real y_m_three_h = f(x - 3 * h);
  if (error) {
    // math_error: pow(h,6)/140 * d7fx, nine_point_centered_difference_seventh_derivative
    Real y_four_h = f(x + 4 * h);
    Real y_m_four_h = f(x - 4 * h);
    *error = abs((y_four_h - y_m_four_h) - 14 * (y_h - y_m_h) + 14 * (y_two_h - y_m_two_h) - 6 * (y_three_h - y_m_three_h)) / (280 * h);
    // round_error: ...
    *error += (abs(y_three_h) + abs(y_m_three_h) + 9 * (abs(y_two_h) + abs(y_m_two_h)) + 45 * (abs(y_h) + abs(y_m_h))) * eps / (60 * h);
  }
  return ((y_three_h - y_m_three_h) - 9 * (y_two_h - y_m_two_h) + 45 * (y_h - y_m_h)) / (60 * h);
}

/* nine point centered difference
* formula
                              fxx(x)            fxxx(x)            fxxxx(x)            fxxxxx(x)            fxxxxxx(x)            fxxxxxxx(x)            fxxxxxxxx(x)            fxxxxxxxxx(x)
(1) f(x+h) = f(x) + fx(x)*h + ------*pow(h,2) + -------*pow(h,3) + --------*pow(h,4) + ---------*pow(h,5) + ----------*pow(h,6) + -----------*pow(h,7) + ------------*pow(h,8) + -------------*pow(h,9) + ...
                                2                  6                  24                 120                   720                   5'040                  40'320                  362'880
(1) f(x-h) = ...
                                  fxx(x)*4          fxxx(x)*8          fxxxx(x)*16         fxxxxx(x)*32         fxxxxxx(x)*64         fxxxxxxx(x)*128        fxxxxxxxx(x)*256        fxxxxxxxxx(x)*512
(1) f(x+2*h) = f(x) + fx(x)*2*h + ------*pow(h,2) + -------*pow(h,3) + --------*pow(h,4) + ---------*pow(h,5) + ----------*pow(h,6) + -----------*pow(h,7) + ------------*pow(h,8) + -------------*pow(h,9) + ...
                                    2                  6                  24                 120                    720                  5'040                  40'320                  362'880
(1) f(x-2*h)) = ...
                                  fxx(x)*9          fxxx(x)*27         fxxxx(x)*81         fxxxxx(x)*243        fxxxxxx(x)*729        fxxxxxxx(x)*2'187      fxxxxxxxx(x)*8'748      fxxxxxxxxx(x)*19'683
(1) f(x+3*h) = f(x) + fx(x)*3*h + ------*pow(h,2) + -------*pow(h,3) + --------*pow(h,4) + ---------*pow(h,5) + ----------*pow(h,6) + -----------*pow(h,7) + ------------*pow(h,8) + -------------*pow(h,9) + ...
                                    2                  6                  24                 120                    720                  5'040                  40'320                  362'880		 
(1) f(x-3*h) = ...
                                  fxx(x)*16         fxxx(x)*64         fxxxx(x)*256        fxxxxx(x)*1'024      fxxxxxx(x)*4'096      fxxxxxxx(x)*16'384     fxxxxxxxx(x)*65'536     fxxxxxxxxx(x)*262'144
(1) f(x+4*h) = f(x) + fx(x)*4*h + ------*pow(h,2) + -------*pow(h,3) + --------*pow(h,4) + ---------*pow(h,5) + ----------*pow(h,6) + -----------*pow(h,7) + ------------*pow(h,8) + -------------*pow(h,9) + ...
                                    2                  6                  24                 120                    720                  5'040                  40'320                  362'880
(1) f(x-4*h) = ...
  
(2) 672*(f(x+h)-f(x-h)) - 168*(f(x+2*h)-f(x-2*h)) + 32*(f(x+3*h)-f(x-3*h)) - 3*(f(x+4*h)-f(x-4*h))
                              fxxx(x)                  fxxxxx(x)                  fxxxxxxx(x)                  fxxxxxxxxx(x)
    =  ( 1'344*fx(x)*h + 1'344*-------*pow(h,3) + 1'344*---------*pow(h,5) + 1'344*-----------*pow(h,7) + 1'344*-------------*pow(h,9) + ... )
                                  6                        120                        5'040                        362'880
                            fxxx(x)*8              fxxxxx(x)*32             fxxxxxxx(x)*128            fxxxxxxxxx(x)*512
    - ( 336*fx(x)*2*h + 336*-------*pow(h,3) + 336*---------*pow(h,5) + 336*-----------*pow(h,7) + 336*-------------*pow(h,9) + ... )
                                6                      120                      5'040                      362'880
                          fxxx(x)*27            fxxxxx(x)*243           fxxxxxxx(x)*2'187         fxxxxxxxxx(x)*19'683
    + ( 64*fx(x)*3*h + 64*-------*pow(h,3) + 64*---------*pow(h,5) + 64*-----------*pow(h,7) + 64*-------------*pow(h,9) + ... )
                              6                     120                     5'040                     362'880
                        fxxx(x)*64           fxxxxx(x)*1'024        fxxxxxxx(x)*16'384       fxxxxxxxxx(x)*262'144
    - ( 6*fx(x)*4*h + 6*-------*pow(h,3) + 6*---------*pow(h,5) + 6*-----------*pow(h,7) + 6*-------------*pow(h,9) + ... )
                            6                    120                    5'040                    362'880
                                            (1'344 - 336*8 + 64*27 - 6*64)                    (1'344 - 336*32 + 64*243 - 6*1'024)                      (1'344 - 336*128 + 64*2'187 - 6*16'384)                        (1'344 - 336*512 + 64*19'683 - 6*262'144)
    = (1'344 - 336*2 + 64*3 - 6*4)*fx(x)*h + ------------------------------*fxxx(x)*pow(h,3) + -----------------------------------*fxxxxx(x)*pow(h,5) + ---------------------------------------*fxxxxxxx(x)*pow(h,7) + -----------------------------------------*fxxxxxxxxx(x)*pow(h,9) + ...
                                                          6                                                    120                                                      5'040                                                         362'880
    = 840*fx(x)*h + 0 + 0 + 0 + (-4/3)*fxxxxxxxxx(x)*pow(h,9) + ...
          672*(f(x+h)-f(x-h)) - 168*(f(x+2*h)-f(x-2*h)) + 32*(f(x+3*h)-f(x-3*h)) - 3*(f(x+4*h)-f(x-4*h) + 4/3*fxxxxxxxxx(x)*pow(h,9) + ...
fx(x) = ----------------------------------------------------------------------------------------------------------------------------------
                                                                  840*h
          672*(f(x+h)-f(x-h)) - 168*(f(x+2*h)-f(x-2*h)) + 32*(f(x+3*h)-f(x-3*h)) - 3*(f(x+4*h)-f(x-4*h)      1
      = ----------------------------------------------------------------------------------------------- + -----*fxxxxxxxxx(x)*pow(h,8) + ...
                                                  840*h                                                    630
*/
template<typename Function, typename Real>
Real c9_difference(Function f, Real x, Real* error = nullptr) {
  Real eps = std::numeric_limits<Real>::epsilon();
  Real y = f(x);
  Real h = pow((y == 0 ? eps : abs(y)*eps)*2625/16, Real(1)/9);
  if ( (x + h) - x == 0 ) {
    h = nextafter(x, std::numeric_limits<Real>::max()) - x; 
  }

  Real y_h = f(x + h);
  Real y_m_h = f(x - h);
  Real y_two_h = f(x + 2 * h);
  Real y_m_two_h = f(x - 2 * h);
  Real y_three_h = f(x + 3 * h);
  Real y_m_three_h = f(x - 3 * h);
  Real y_four_h = f(x + 4 * h);
  Real y_m_four_h = f(x - 4 * h);
  if (error) {
    // round_error: ...
    *error = (
      672 * (abs(y_h) + abs(y_m_h)) +
      168 * (abs(y_two_h) + abs(y_m_two_h)) +
      32 * (abs(y_three_h) + abs(y_m_three_h)) +
      3 * (abs(y_four_h) + abs(y_m_four_h))) * eps / (840 * h);
    // math_error: pow(h,8)*(4/3)*d9f(x) 
  }
  return (672 * (y_h - y_m_h)
          - 168 * (y_two_h - y_m_two_h)
          + 32 * (y_three_h - y_m_three_h)
          - 3 * (y_four_h - y_m_four_h)) / (840 * h);
}

/* derivative of Function('f') respect to 'x'
  
* differentiate( sin, x )
  = cos(x)
* differentiate( cos, x )
  = -sin(x)
...
*/
template<typename Function, typename Real>
Real differentiate(Function f, Real x, Real* error = nullptr) {
  return c5_difference(f, x, error);
}

/* del(v)
*/
template<typename Function, typename Vector3>
Vector3 gradient(Function f, Vector3 v) {
  using Real = std::remove_cvref_t< decltype( v[0] ) >;
  Real x = v[0];
  Real y = v[1];
  Real z = v[2];
  auto fdx = [f,y,z](Real x){ return f(Vector3{x,y,z}); };
  auto fdy = [f,x,z](Real y){ return f(Vector3{x,y,z}); };
  auto fdz = [f,x,y](Real z){ return f(Vector3{x,y,z}); };
  Real dfdx = differentiate(fdx, x);
  Real dfdy = differentiate(fdy, y);
  Real dfdz = differentiate(fdz, z);
  return Vector3{ dfdx, dfdy, dfdz };
}

/* dot(del,v)
*/
template<typename Function, typename Vector3>
auto divergence(Function f, Vector3 v) {
  using Real = std::remove_cvref_t< decltype( v[0] ) >;
  Real x = v[0];
  Real y = v[1];
  Real z = v[2];
  Real dfdx_x = differentiate( [f,y,z](Real x){ return f(Vector3{x,y,z})[0]; }, x );
  Real dfdy_y = differentiate( [f,x,z](Real y){ return f(Vector3{x,y,z})[1]; }, y );
  Real dfdz_z = differentiate( [f,x,y](Real z){ return f(Vector3{x,y,z})[2]; }, z );
  return dfdx_x + dfdy_y + dfdz_z;
}

/* cross(del,v) 
*/
template<typename Function, typename Vector3>
Vector3 curl(Function f, Vector3 v) {
  using Real = std::remove_cvref_t< decltype(v[0]) >;
  Real x = v[0];
  Real y = v[1];
  Real z = v[2];
  Real dfdy_z = differentiate( [f,x,z](Real y){return f(Vector3{x,y,z})[2];}, y );
  Real dfdz_y = differentiate( [f,x,y](Real z){return f(Vector3{x,y,z})[1];}, z );
  Real dfdz_x = differentiate( [f,x,y](Real z){return f(Vector3{x,y,z})[0];}, z );
  Real dfdx_z = differentiate( [f,y,z](Real x){return f(Vector3{x,y,z})[2];}, x );
  Real dfdx_y = differentiate( [f,y,z](Real x){return f(Vector3{x,y,z})[1];}, x );
  Real dfdy_x = differentiate( [f,x,z](Real y){return f(Vector3{x,y,z})[0];}, y );
  return Vector3{ dfdy_z - dfdz_y, dfdz_x - dfdx_z, dfdx_y - dfdy_x };
}


// undeterminant ...

/* three point centered difference, second order derivative
*/
template<typename Function, typename Real>
Real c3_difference2nd(Function f, Real x, Real* error = nullptr) {
  Real eps = std::numeric_limits<Real>::epsilon();
  Real y = f(x);
  Real h = pow((y == 0 ? eps : abs(y)*eps) * 48, Real(1)/4);
  if ( (x + h) - x == 0 ) {
    h = nextafter(x, std::numeric_limits<Real>::max()) - x; 
  }

  Real y_h = f(x + h);
  Real y_m_h = f(x - h);
  if (error) {
    // math_error: pow(h,2)*(1/12)*d4f(x)
  }
  return (y_h + y_m_h - 2 * y) / pow(h, 2);
}

/* five point centered difference, fourth order derivative
*/
template<typename Function, typename Real>
Real c5_difference4th(Function f, Real x, Real* error = nullptr) {
  Real eps = std::numeric_limits<Real>::epsilon();

  // E = pow(h,2)*(1/6)*d6f(x) + 16*e/pow(h,4)
  // dE = h*(2/6)*d6f(x) - 4*16*e*pow(h,-5) = 0
  // h*(2/6)*d6f(x) = 4*16*e*pow(h,-5)
  // pow(h,6) = 64*(6/2)*e/d6f(x)
  // h approxi pow(192*e, 1.0/6.0)
  Real y = f(x);
  Real h = pow((y == 0 ? eps : abs(y)*eps) * 192, Real(1)/6);
  if ( (x + h) - x == 0 ) {
    h = nextafter(x, std::numeric_limits<Real>::max()) - x; 
  }

  Real y_h = f(x + h);
  Real y_m_h = f(x - h);
  Real y_two_h = f(x + 2 * h);
  Real y_m_two_h = f(x - 2 * h);
  if (error) {
    // math_error: pow(h,2)*(1/6)*d6f(x)
  }
  return ((y_two_h + y_m_two_h) - 4 * (y_h + y_m_h) + 6 * y) / pow(h, 4);
}

}// namespace calculation


/* // Test
//double x = 0.77777;
//auto f = [](double x) { return cos(x); };
//auto df = [](double x) { return -sin(x); };
//double x = 10.500;
//auto f = [](double x){ return pow(x,6) - 1'340'095.640625; };
//auto df = [](double x){ return 6*pow(x,5); };
double x = 10.51231;
auto f = [](double x){ return pow(x,3) - 110.25; };
auto df = [](double x){ return 3*x*x; };
std::cout << "res:" << df(x) << std::endl;
auto c3_diff = [&](double x, double h) { return (f(x+h) - f(x-h)) / h/2; };
for(double h = 1000.0; h >= std::numeric_limits<double>::epsilon(); h *= 0.5){
	std::cout << "h:" <<std::setw(20)<< h 
              << "\tres:" <<std::setw(20)<<c3_diff(x,h) 
              << "\terror:" << abs(c3_diff(x,h) - df(x)) << std::endl;
}
//abs(f(x))*eps / pow(h, 2) = -h
//abs(f(x))*eps = -pow(h, 3)
// -cbrt(abs(f(x))*eps) = h
double eps = std::numeric_limits<double>::epsilon();
std::cout << "idea_error:" << abs(c3_diff(x, -cbrt(abs(f(x)*eps))) - df(x)) << std::endl;
std::cout << "h:" <<std::setw(20)<<pow((f(x)==0 ? eps : abs(f(x))*eps)*3, 1.0/3)
          << "\tres:" <<std::setw(20)<<c3_difference(f,x) 
          << "\terror:" << abs(c3_difference(f,x) - df(x)) << std::endl;
*/