#pragma once

///@brief Microsurface Normal Distribution. 
///@license Free 
///@review 2023-01-31 
///@contact Jiang1998Nan@outlook.com 
#define _MATH_PHYSICS_MICROFACET_

#include <cassert>
#include <numbers>

namespace math { namespace physics {
///<p>
///Microfacet definition, the projected areas of macro surface and micro surface are equal.</p><pre>
/// 
///		 |---- projected area ----|
/// 
///		 |                        |
/// 
///		_|________________________|_ macro surface (perfect)
/// 
///		 |                        |
///		    /\      ____/\       /\
///		_|/   \____/      \____/     micro surface (rough)
/// 
///		intg( D(N,Nm)*cos(N,Nm) * dNm ) = 1[m*m] (is probability density function.)
///</pre>

	template<typename scalar>
	struct ggx_base {
		static scalar distribution(const scalar cos_N_Nm, const scalar r2) {
			///<p cite="Walter et al. 2007, Microfacet Models for Refraction through Rough Surfaces">
			///Ground Glass Microsurface Normal Distribution. </p><pre>
			///		                            r^2
			///		D(N,Nm) = --------------------------------------------
			///		           pi * cos(N,Nm)^4 * ( r^2 + tan(N,Nm)^2 )^2
			///</pre>
			///
			///<p>
			///Optimization for reduce <q>tan</q>. </p><pre>
			/// 
			///	Expand "binomial" and "distributive-law" in denominator.
			///		                      r^2
			///		----------------------------------------------------------------------------------------------
			///		 pi * ( cos(N,Nm)^4 * r^4 + cos(N,Nm)^4 * tan(N,Nm)^4 + cos(N,Nm)^4 * r^2 * tan(N,Nm)^2 * 2 )
			/// 
			///	Eliminate "tan(N,Nm)" by "tan = sin/cos".
			///		                      r^2
			///		--------------------------------------------------------------------------------
			///		 pi * ( cos(N,Nm)^4 * r^4 + sin(N,Nm)^4 + cos(N,Nm)^2 * r^2 * sin(N,Nm)^2 * 2 )
			/// 
			///	Inverse "distributive-law".
			///		                    r^2
			///		--------------------------------------------
			///		 pi * ( cos(N,Nm)^2 * r^2 + sin(N,Nm)^2 )^2
			///</pre>
			assert( cos_N_Nm >= 0 );
			scalar temp = (cos_N_Nm * r2 - cos_N_Nm) * cos_N_Nm + 1;
			return r2 / ( std::numbers::pi_v<scalar> * temp*temp );
			//"cos_N_Nm*r2 - cos_N_Nm" is absolute-error,
			//when the error is more important, the error is more less than "+ 1", so the error is relative-error.
		}
		
		static scalar masking_shadowing(const scalar cos_N_V, const scalar r2) {
			///<p cite="Eric Heitz. 2014, Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs">
			///What fraction of the microsurface with normal<q>N</q> is visible. </p> <pre>
			///		                          1
			///		P22(x,y) = ----------------------------------
			///		            pi * r^2 * (1 + (x^2+y^2)/r^2)^2
			/// 
			///		A(N,V) = intg(cot(N,V),inf, (y - cot(N,V)) * intg( P22(x,y) * dx ) * dy )
			/// 
			///		          -1 + sqrt(1 + tan(N,V)^2*r^2)
			///		A(N,V) = -------------------------------
			///		                       2
			///		               1
			///		G(N,V) = ------------
			///		          1 + A(N,V)
			///		                          1
			///		G(N,V) = ------------------------------------
			///		           2 - 1 + sqrt(1 + tan(N,V)^2*r^2) 
			///		          ----------------------------------
			///		                          2
			/// 
			///		                      2*cos(N,V)
			///		G(N,V) = ------------------------------------
			///		          cos(N,V) + sqrt(cos(N,V)^2 + (1 - cos(N,V))^2*r^2 = (1 - r^2)*cos(N,V)^2 + r^2) 
			///</pre>
			assert( cos_N_V > 0 );
			scalar sq_cos_N_V = cos_N_V * cos_N_V;
			scalar sq_sin_N_V = 1 - sq_cos_N_V;
			return 2/(
				1 + sqrt(1 + sq_sin_N_V/sq_cos_N_V*r2) );
#if 0
scalar mu = 0.2333f+0.2f;//=cot(cos_N_V);
scalar r = 0.125f+0.2f;

std::cout << math::p7quad(mu, mu + 10.0f, [mu,r](scalar q) {
	return (q - mu)*math::p7quad(-10.0f, +10.0f, [q,r](scalar p) {
		return 1 / ( pi*r*r * pow(1 + (p*p+q*q)/(r*r), 2.0f) );
	});
}) / mu << std::endl;

std::cout << (sqrt(1 + 1/pow(mu/r,2.0f)) - 1)/2 << std::endl;
#endif
		}

		static scalar height_correlated_masking_shadowing(const scalar cos_N_Vi, const scalar cos_N_Vo, const scalar r2) {
			///<pre>
			///		                       1
			///		G(N,Vi,Vo) = -----------------------
			///		              1 + A(N,Vi) + A(N,Vo)
			///		                                          1
			///		G(N,V) = ---------------------------------------------------------------------
			///		           2 - 1 + sqrt(1 + tan(N,Vi)^2*r^2) - 1 + sqrt(1 + tan(N,Vo)^2*r^2)
			///		          -------------------------------------------------------------------
			///		                                          2
			///</pre>
			assert( cos_N_Vi > 0 && cos_N_Vo > 0 );
			scalar sq_cos_N_Vi = cos_N_Vi * cos_N_Vi;
			scalar sq_cos_N_Vo = cos_N_Vo * cos_N_Vo;
			return 2 / (
				sqrt(1 + (1 - sq_cos_N_Vi)/sq_cos_N_Vi*r2) + sqrt(1 + (1 - sq_cos_N_Vo)/sq_cos_N_Vo*r2) );
		}

		static scalar direction_correlated_masking_shadowing(const scalar cos_N_Vi, const scalar cos_N_Vo, const scalar r2) {
			assert( cos_N_Vi > 0 && cos_N_Vo > 0 );
			abort();
		}
		
		static scalar visibility(const scalar cos_N_V, const scalar r2) {
			///		            G(N,V)
			///		V(N,V) = --------------
			///		          2 * cos(N,V)
			assert( cos_N_V > 0 );
			return 1 / (
				cos_N_V + sqrt((1 - r2)*cos_N_V*cos_N_V + r2) );
		}

		static scalar height_correlated_visibility(const scalar cos_N_Vi, const scalar cos_N_Vo, const scalar r2) {
			///		                    G(N,Vi,Vo)
			///		V(N,Vi,Vo) = ---------------------------
			///		              4 * cos(N,Vi) * cos(N,Vo)
			assert( cos_N_Vi > 0 && cos_N_Vo > 0 );
			scalar temp = 1 - r2;
			scalar for_Vo = cos_N_Vi * sqrt(temp * cos_N_Vo*cos_N_Vo + r2);
			scalar for_Vi = cos_N_Vo * sqrt(temp * cos_N_Vi*cos_N_Vi + r2);
			return scalar(0.5)/(for_Vo + for_Vi);
		}

		static scalar direction_correlated_visibility(const scalar cos_N_Vi, const scalar cos_N_Vo, const scalar r2) {
			assert( cos_N_Vi > 0 && cos_N_Vo > 0 );
			abort();
		}

	//anisotropic:

		///Anisotropic Roughness passby what?
		///pass by two scalar, need twice copy.
		///pass by scalar pointer, only once copy but cannot for temporary variables.
		///pass by vector, only once copy but requires SSE cpu support. (we used)
		/// 
		///But all functions do nonlinear operations on roughness, "... = r[0]*r[1]".
		///So pass by vector or pointer only increase coat, !!we use pass by two scalar!!.

		static scalar distribution(const scalar cos_N_Nm, const scalar cos_T0_Nm, const scalar cos_T1_Nm, const scalar r0, const scalar r1) {
			///<pre cite="Eric Heitz. 2014, Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs">
			///		                                                           1
			///		D(b,Nm,rx,ry) = ------------------------------------------------------------------------------------------
			///		                                                                   cos(azimuth)^2     sin(azimuth)^2
			///		                 pi * rx * ry * cos(N,Nm)^4 * (1 + tan(N,Nm)^2 * (---------------- + ----------------))^2
			///                                                                           rx^2               ry^2
			/// dot(T0,Nm) = sin(elevation) * cos(azimuth)
			/// dot(T1,Nm) = sin(elevation) * sin(azimuth)
			///</pre>
			/// 
			///<p>
			///Optimization. </p><pre>
			/// Expand "tan(N,Nm)^2" by "tan = sin/cos".
			///		                                                        1
			///		D(B,Nm,rx,ry) = ----------------------------------------------------------------------------------------
			///		                                                         1           cos(T0,Nm)^2     cos(T1,Nm)^2
			///		                 pi * rx * ry * cos(N,Nm)^4 * (1 + ------------- * (-------------- + --------------))^2
			///		                                                    cos(N,Nm)^2          rx^2             ry^2
			/// Apply "Distributive-law".
			///                                                  1
			///		D(B,Nm,rx,ry) = -------------------------------------------------------------------
			///		                                                cos(T0,Nm)^2     cos(T1,Nm)^2
			///		                 pi * rx * ry * (cos(N,Nm)^2 + -------------- + --------------))^2
			///		                                                    rx^2             ry^2
			///</pre>
			assert( r0 != 0 && r1 != 0 );
			assert( cos_N_Nm > 0 );
			scalar temp = cos_N_Nm*cos_N_Nm + cos_T0_Nm*cos_T0_Nm/(r0*r0) + cos_T1_Nm*cos_T1_Nm/(r1*r1); 
			return 1 / ( std::numbers::pi_v<scalar> * r0 * r1 * temp*temp );
		}

		static scalar masking_shadowing(const scalar cos_N_V, const scalar cos_T0_V, const scalar cos_T1_V, const scalar r0, const scalar r1) {
			///                         1                                                           1
			/// ggx_g(..) = ----------------------------, a = ---------------------------------------------------------------------------------
			///                   -1 + sqrt(1 + 1/a^2)         sqrt( (cos(T,V)/sin(N,V))^2 * rx^2 + (cos(B,V)/sin(N,V))^2 * ry^2 ) * tan(Nm,V)
			///              1 + ----------------------
			///                           2
			assert( cos_N_V > 0 );
			scalar sq_inv_a = (cos_T0_V*cos_T0_V * r0*r0 + cos_T1_V*cos_T1_V * r1*r1)/(cos_N_V*cos_N_V);
			return 2/(1 + sqrt(1 + sq_inv_a));
		}

		static scalar height_correlated_masking_shadowing(const scalar cos_N_Vi, const scalar cos_T0_Vi, const scalar cos_T1_Vi,
			const scalar cos_N_Vo,  const scalar cos_T0_Vo, const scalar cos_T1_Vo, const scalar r0, const scalar r1) {
			/// G(N,T0,T1,Vi,Vo)
			assert( cos_N_Vi > 0 && cos_N_Vo > 0 );
			return height_correlated_visibility(cos_N_Vi, cos_T0_Vi, cos_T1_Vi, cos_N_Vo, cos_T0_Vo, cos_T1_Vo, r0, r1) * (4 * cos_N_Vi * cos_N_Vo);
		}

		static scalar height_correlated_visibility(const scalar cos_N_Vi, const scalar cos_T0_Vi, const scalar cos_T1_Vi, 
			const scalar cos_N_Vo,  const scalar cos_T0_Vo, const scalar cos_T1_Vo, const scalar r0, const scalar r1) {
			///		                  G(N,T0,T1,Vi,Vo)
			///		V(N,Vi,Vo) = ---------------------------
			///		              4 * cos(N,Vi) * cos(N,Vo)
			assert( cos_N_Vi > 0 && cos_N_Vo > 0 );
			scalar r0sq = r0 * r0;
			scalar r1sq = r1 * r1;
			scalar lambdaV = cos_N_Vi * sqrt(r0sq*cos_T0_Vo*cos_T0_Vo + r1sq*cos_T1_Vo*cos_T1_Vo + cos_N_Vo*cos_N_Vo);
			scalar lambdaL = cos_N_Vo * sqrt(r0sq*cos_T0_Vi*cos_T0_Vi + r1sq*cos_T1_Vi*cos_T1_Vi + cos_N_Vi*cos_N_Vi);
			return scalar(0.5)/(lambdaV + lambdaL);
		}
	};

	template<typename vector3>
	struct ggx_normal : ggx_base<std::remove_cvref_t<decltype(std::declval<vector3>()[0])>> {
		using scalar = std::remove_cvref_t<decltype(std::declval<vector3>()[0])>;

		static scalar probability_density(const scalar cos_N_Nm, const scalar r2) {
			assert(cos_N_Nm >= 0);
			return ggx_base<scalar>::distribution(cos_N_Nm, r2) * cos_N_Nm;
		}

		scalar probability_density(const scalar cos_N_Nm) const {
			return probability_density(cos_N_Nm, this->r2);
		}

		template<typename vector2, typename _Basis3> 
		static vector3 generate(const vector2& Xi, const _Basis3& B, const scalar r2) {
			///<p cite="Walter et al. 2007, Microfacet Models for Refraction through Rough Surfaces">
			///Generate Microsurface Normal in TBN space. First we known generate number 
			///from distribution with uniform-variable<q>U</q>[0,1] may be derivated 
			///by the <q>Inverse of Comulative Distribution Function</q>. </p><pre>
			/// 
			///		C(X,Y) = intg(0,Y, intg(0,X, PDF(x,y) * sin(y)*dx*dy ))
			/// 
			///		C(X,Y) = intg(0,Y, intg(0,X, D(cos(N,Nm))*cos(N,Nm) * sin(y)*dx*dy ))
			///		    Nm = sph2cart(x,y,1)
			/// 
			/// Because the integrand associated only elevation<q>y</q>, so eliminate azimuth<q>x</q>[0,pi*2].
			/// 
			///		C(Y) = intg(0,Y, PDF(y) * sin(y)*dy ) * 2*pi
			/// 
			///		C(Y) = intg(0,Y, D(cos(y))*cos(y) * sin(y)*dy ) * 2*pi
			/// 
			///	Then do integrate.
			///		                                  r^2
			///		C(Y) = intg(0,Y, -------------------------------------- * cos(y) * sin(y)*dy ) * 2*pi
			///		                  pi * cos(y)^4 * ( r^2 + tan(y)^2 )^2
			///		                                  r^2
			///		C(Y) = intg(0,Y, --------------------------------------          * sin(y)*dy ) * 2
			///		                       cos(y)^3 * ( r^2 + tan(y)^2 )^2
			/// 
			///		                              2 * sec(y)^2 * tan(y)
			///		C(Y) = intg(0,Y, -------------------------------------- * dy ) * r^2 * 2
			///		                              2 * ( r^2 + tan(y)^2 )^2
			///		                                du
			///		C(u(Y)) = intg(0,U, ----------------------------------- ) * r^2 * 2
			///		                              2 * u^2
			///		u = r^2 + tan(y)^2
			///		du = 2 * sec(y)^2 * tan(y) * dy
			/// 
			///		               r^2                r^2
			///		C(Y) = - ---------------- + ---------------- = Xi
			///		          r^2 + tan(Y)^2     r^2 + tan(0)^2
			/// 
			/// Final solve it for Y.
			///		          r * sqrt(Xi)
			///		Y = atan(--------------)
			///		          sqrt(1 - Xi)
			///</pre>
			///
			///<p>
			///Optimization for elevation<q>Y</q> by <q>cos(atan(x)) = 1/sqrt(x^2 + 1)</q>. </p><pre>
			///		                       1
			///		cos(Y) = ------------------------------
			///		                 r * sqrt(Xi)
			///		          sqrt( -------------- ^ 2 + 1 )
			///		                 sqrt(1 - Xi)
			/// 
			///		                  sqrt(1)
			///		cos(Y) = ------------------------------
			///		                 r^2 * Xi + 1 - Xi
			///		          sqrt( ------------------- )
			///		                      1 - Xi
			/// 
			///		                     1 - Xi
			///		cos(Y) = sqrt( ------------------- )
			///		                r^2 * Xi + 1 - Xi
			///</pre>
			scalar azimuth = Xi[0] * (std::numbers::pi_v<scalar>*2);
			scalar temp = 1 - Xi[1];
			scalar cos_elevation = sqrt( temp/(r2*Xi[1] + temp) );//Not need clamp because 'r2*X[1] >= 0'.
			scalar sin_elevation = sqrt( 1 - cos_elevation*cos_elevation );
			
			//static_assert(elevation_0_pi);
			return B.normal() * cos_elevation +
				B.tangent0() * (sin_elevation * cos(azimuth)) +
				B.tangent1() * (sin_elevation * sin(azimuth));
		}

		template<typename vector2, typename _Basis3>
		vector3 operator()(const vector2& Xi, const _Basis3& B) const {
			return generate(Xi, B, this->r2);
		}

#if 0
		template<typename vector2>
		static scalar generate_for_reflectance(const vector2& Xi, const _Basis& B, const vector3& V, vector3& L, const scalar r2) {
			scalar cos_N_V = dot(B.normal(), V);
			if (cos_N_V <= 0) { return scalar(0); }

			vector3 Nm = generate(Xi, B, r2);
			scalar cos_Nm_V = dot(Nm, V);
			scalar cos_Nm_N = dot(Nm, B.normal());
			if (cos_Nm_V <= 0 || cos_Nm_N <= 0) { return scalar(0); }
			        L = math::geometry::mirror(V, Nm);
			if (dot(Nm, L) <= 0) { return scalar(0); }

			scalar cos_N_L = dot(B.normal(), L);
			if (cos_N_L <= 0) { return scalar(0); }

			///<pre>
			///       F(N,V,L)
			/// avg( ---------- )
			///         PDF
			/// 
			///      F(V,Nm) * D(Nm) * G2(V,L)                       Dm(Nm)        D(Nm) * cos(N,Nm)
			/// F = --------------------------- * cos(N,L), PDF = ------------- = -------------------
			///       4 * cos(N,V) * cos(N,L)                      4*cos(V,Nm)       4 * cos(V,Nm)
			/// 
			///          F(V,Nm) * D(Nm) * G2(V,L) * 4 * cos(V,Nm)            * cos(N,L)
			/// F/PDF = -----------------------------------------------------------------
			///                    D(Nm)           * 4 * cos(N,Nm) * cos(N,V) * cos(N,L)
			///</pre>
			return ggx_base<scalar>::height_correlated_masking_shadowing(cos_N_V, cos_N_L, r2) * cos_Nm_V / (cos_Nm_N * cos_N_V);
		}
		
		template<typename vector2>
		scalar operator()(const vector2& Xi, const _Basis& B, const vector3& Vi, vector3& Vo) const {
			return generate_for_reflectance(Xi, B, Vi, std::ref(Vo), this->r2);
		}
#endif

		static void set_roughness(const scalar r, scalar& r2) { r2 = r * r; }
		static scalar roughness(const scalar& r2) { return sqrt(r2); }

		void set_roughness(const scalar r) { this->r2 = r * r; }
		scalar roughness() const { return sqrt(this->r2); }

		scalar r2;
	};

	template<typename vector3>
	struct ggx_anisotropic_normal : ggx_base<std::remove_cvref_t<decltype(std::declval<vector3>()[0])>> {
		using scalar = std::remove_cvref_t<decltype(std::declval<vector3>()[0])>;

		//...

		scalar r[2];
	};
	
	template<typename vector3>
	struct ggx_visible_anisotropic_normal : ggx_base<std::remove_cvref_t<decltype(std::declval<vector3>()[0])>> {
		using scalar = std::remove_cvref_t<decltype(std::declval<vector3>()[0])>;

		static scalar probability_density(const scalar cos_N_Nm, const scalar cos_T0_Nm, const scalar cos_T1_Nm, 
			const scalar cos_N_V, const scalar cos_T0_V, const scalar cos_T1_V,  const scalar cos_Nm_V, const vector3 r) {
			assert(cos_N_Nm > 0 && cos_N_V > 0 && cos_Nm_V > 0);
			if (r[0] == r[1]) {
				scalar r2 = r[0] * r[0];
				return ggx_base<scalar>::distribution(cos_N_Nm, r2) 
					* ggx_base<scalar>::masking_shadowing(cos_N_V, r2) * cos_Nm_V / cos_N_V;
			}
			return ggx_base<scalar>::distribution(cos_N_Nm,cos_T0_Nm,cos_T1_Nm,r[0],r[1])
				* ggx_base<scalar>::masking_shadowing(cos_N_V,cos_T0_V,cos_T1_V,r[0],r[1]) * cos_Nm_V / cos_N_V;
		}

		scalar probability_density(const scalar cos_N_Nm, const scalar cos_T0_Nm, const scalar cos_T1_Nm, 
			const scalar cos_N_V, const scalar cos_T0_V, const scalar cos_T1_V,  const scalar cos_Nm_V) const {
			return probability_density(cos_N_Nm, cos_T0_Nm, cos_T1_Nm, cos_N_V, cos_T0_V, cos_T1_V,   cos_Nm_V, this->r);
		}
		
		template<typename vector2, typename _Basis3>
		static vector3 generate(const vector2& Xi, const _Basis3& B, const vector3& V, const vector3 r____) {
			scalar tangent0 = dot(V, B.tangent0());
			scalar tangent1 = dot(V, B.tangent1());
			scalar normal   = dot(V, B.normal());
			vector3 Ve = {tangent0, tangent1, normal};

/// Generate microsurface_normal in TBN space, 
/// published by [Heitz 2018, "Sampling the GGX Distribution of Visible Normals"].
			const vector3& elip2hemis = r____; // = {r, 1, 1};
			// Section 3.2: transforming the view direction to the hemisphere configuration
			vector3 Vh = normalize(Ve * elip2hemis);
			// Section 4.1: orthonormal basis (with special case if cross product is zero)
			scalar lensq = Vh[0] * Vh[0] + Vh[1] * Vh[1];
			vector3 T1 = lensq > 0 ? vector3{-Vh[1], Vh[0], 0}/sqrt(lensq) : vector3{1, 0, 0};
			vector3 T2 = cross(Vh, T1);
			// Section 4.2: parameterization of the projected area
			scalar phi = Xi[0] * static_cast<scalar>(6.283185307179586476925286766559);
			scalar r   = sqrt(Xi[1]);
			scalar t1  = r * cos(phi);
			scalar t2  = r * sin(phi);
			scalar s   = scalar(0.5) * (1 + Vh[2]);
			t2 = (1 - s)*sqrt(1 - t1*t1) + s*t2;
			// Section 4.3: reprojection onto hemisphere
			vector3 Nh = t1*T1 + t2*T2 + sqrt(max(scalar(0), 1 - t1*t1 - t2*t2))*Vh;
			assert(Nh[2] >= 0);
			// Section 3.4: transforming the normal back to the ellipsoid configuration
			vector3 Ne = normalize(Nh * elip2hemis);

			tangent0 = Ne[0];
			tangent1 = Ne[1];
			normal   = Ne[2];
			return B.normal()*normal + B.tangent0()*tangent0 + B.tangent1()*tangent1;
		}

		template<typename vector2, typename _Basis3>
		vector3 operator()(const vector2& Xi, const _Basis3& B, const vector3& V) const {
			return generate(Xi, B, V, this->r);
		}

#if 0
		template<typename vector2, typename _Basis3>
		static scalar generate_for_reflectance(const vector2& Xi, const _Basis3& B, const vector3& V, vector3& L, const vector3 r) {
			scalar cos_N_V = dot(B.normal(), V);
			if (cos_N_V <= 0) { return scalar(0); }
																																																																																																					
			vector3 Nm = generate(Xi, B, V, r);
			if (dot(Nm,V) <= 0 || dot(Nm,B.normal()) <= 0) { return scalar(0); }
			        L = math::geometry::mirror(V, Nm);
			if (dot(Nm,L) <= 0) { return scalar(0); }

			scalar cos_N_L = dot(B.normal(), L);
			if (cos_N_L <= 0) { return scalar(0); }

			///<pre>
			///       F(N,V,L)
			/// avg( ---------- )
			///         PDF
			/// 
			///      F(V,Nm) * D(Nm) * G2(V,L)                       DV(Nm)        G1(V) * cos(V,Nm) * D(Nm)
			/// F = --------------------------- * cos(N,L), PDF = ------------- = ---------------------------
			///       4 * cos(N,V) * cos(N,L)                      4*cos(V,Nm)       4*cos(V,Nm) * cos(V,N)
			/// 
			///          4 * cos(V,N) * cos(N,L) * G2(V,L) * cos(V,Nm) * D(Nm) * F(V,Nm)
			/// F/PDF = -------------------------------------------------------------------
			///          4 * cos(N,V) * cos(N,L) * G1(V)   * cos(V,Nm) * D(Nm)
			///</pre>
			if (r[0] == r[1]) {
				scalar r2 = r[0] * r[0];
				return ggx_base<scalar>::height_correlated_masking_shadowing(cos_N_V, cos_N_L, r2)
					/ ggx_base<scalar>::masking_shadowing(cos_N_V, r2);
			}
			scalar cos_T0_V = dot(B.tangent0(), V);
			scalar cos_T1_V = dot(B.tangent1(), V);
			return ggx_base<scalar>::height_correlated_masking_shadowing(cos_N_V, cos_T0_V, cos_T1_V, cos_N_L, dot(B.tangent0(), L), dot(B.tangent1(), L), r[0], r[1])
				/ ggx_base<scalar>::masking_shadowing(cos_N_V, cos_T0_V, cos_T1_V, r[0], r[1]);
		}

		template<typename vector2>
		scalar operator()(const vector2& Xi, const _Basis& B, const vector3& Vi, vector3& Vo) const {
			return generate_for_reflectance(Xi, B, Vi, std::ref(Vo), this->r);
		}
#endif

		//static scalar roughness(const vector3& elip2hemis) { return elip2hemis[0]; }

		void set_roughness(const scalar r) { this->r = {r,r,1}; }
		void set_roughness(const scalar r0, const scalar r1) { this->r = {r0,r1,1}; }
		scalar roughness(size_t i = 0) const { return this->r[i]; }

		vector3 r;
	};

	template<typename vector3>
	using ggx_visible_normal = ggx_visible_anisotropic_normal<vector3>;

	template<typename scalar>
	scalar ggx_specular(const scalar cos_N_Nm, const scalar cos_N_V, const scalar cos_N_L, const scalar r2) {
		//return ggx_base<scalar>::distribution(cos_N_Nm, r2) * ggx_base<scalar>::height_correlated_masking_shadowing(cos_N_L, cos_N_V, r2) / (4 * cos_N_V * cos_N_L);
		return ggx_base<scalar>::distribution(cos_N_Nm, r2) * ggx_base<scalar>::height_correlated_visibility(cos_N_L, cos_N_V, r2);
	}


#if 0
	/// Physically Based Microfacet Normals Distribution,
	///  published by [Beckmann 1963, "The Scattering of Electromagnetic Waves from Rough Surfaces"].
	///@theory
	///                    heaviside(cos(N,Nm))             tan(N,Nm)^2
	/// beckmann_d(..) = ------------------------ * exp( - ------------- )
	///                   pi * r^2 * cos(N,Nm)^4                r^2
	/// 
	///@param cos_N_Nm is cos(angle(micro_normal, geometric_normal)) in [-1,+1].
	///@param r is roughness in [0,1].
	template<typename scalar>
	scalar beckmann_d(scalar cos_N_Nm, scalar r) {
		if (cos_N_Nm > 0) {
			///@optimization 
			/// Eliminate "tan(N,Nm)^2" by "tan = sin/cos".
			///         sin(N,Nm)^2
			/// exp( - ------------------- ) * ...
			///         cos(N,Nm)^2 * r^2
			/// 
			/// Eliminate "sin(N,Nm)^2" by "sin^2 = 1 - cos^2".
			///          1 - cos(N,Nm)^2
			/// exp( - ------------------- ) * ...
			///         cos(N,Nm)^2 * r^2
			scalar sq_r = r * r;
			scalar sq_cos_N_Nm = cos_N_Nm * cos_N_Nm;
			return exp( (sq_cos_N_Nm - 1)/sq_cos_N_Nm / sq_r ) / ( static_cast<scalar>(3.1415926535897932384626433832795) * sq_r * sq_cos_N_Nm*sq_cos_N_Nm );
		} else {
			return 0;
		}
	}

	/// Generate microsurface_normal in tbn,
	///  published by [Beckmann 1963, "The Scattering of Electromagnetic Waves from Rough Surfaces"].
	///@theory
	/// 
	/// azimuth = 2*pi * U0, elevation = atan(sqrt( -r^2 * log(1 - U1) ))
	/// 
	/// 
	///@param r is roughness in [0,1].
	///@param U0 is uniform random var in [0,1].
	///@param U1 is uniform random var in [0,1].
	///@return normal sampled with pdf = ggx_d(Nm) * dot(N,Nm).
	template<typename vector3, typename scalar, typename configty>
	vector3 beckmann_ndf_sample(scalar r, scalar U0, scalar U1, configty) {
		abort();
	}

	/// Generate microsurface_normal.
	template<typename vector3, typename scalar, typename configty>
	vector3 beckmann_ndf_sample(vector3 N, scalar r, scalar U0, scalar U1, configty) {
		abort();
	}

	/// Physically Based Microfacet Masking,
	///  published by [Beckmann 1963, "The Scattering of Electromagnetic Waves from Rough Surfaces"].
	///@theory
	///                                   2                                1
	/// beckmann_g(..) = ---------------------------------------, a = ------------
	///                   1 + erf(a) + 1/(a*sqrt(pi))*exp(-a^2)        r*tan(N,V)
	/// 
	///@param cos_N_V cos(angle(geometric_normal, in_vector|out_vector)) in [-1,+1].
	///@param cos_Nm_V cos(angle(microsurface_normal, in_vector|out_vector)) in [-1,+1].
	///@param r is roughness in [0,1].
	template<typename scalar>
	scalar beckmann_g(scalar cos_N_V, scalar cos_Nm_V, scalar r) {
		if (cos_Nm_V > 0) {
			scalar a = 1/(r * tan(acos(cos_N_V)));
			return 2/(1 + erf(a) + 1/(a*static_cast<scalar>(1.7724538509055160272981674833411))*exp(-a*a));
		} else {
			return 0;
		}
	}

	/// Physically Based Microfacet Masking uncheck,
	///  approximated by [Walter et al. 2007], [Schlick 1994].
	template<typename scalar>
	scalar beckmann_g1(scalar cos_N_V, scalar r) {
		scalar a = 1 / (r * tan(acos(cos_N_V)));
		if (a < 1.6) {
			scalar sq_a = a * a;
			return static_cast<scalar>((3.535f*a + 2.181f*sq_a) / (1 + 2.276f*a + 2.577f*sq_a));
		} else {
			return 1;
		}
	}


	/// Physically Based Microfacet Normals Distribution,
	///  published by [Beckmann 1963, "The Scattering of Electromagnetic Waves from Rough Surfaces"].
	///@theory
	///                    heaviside(cos(N,Nm))                               (cos(T,Nm)/sin(N,Nm))^2     (cos(B,Nm)/sin(N,Nm))^2
	/// beckmann_d(..) = -------------------------- * exp( - tan(N,Nm)^2 * ( ------------------------- + ------------------------- ) )
	///                   pi * rT*rB * cos(N,Nm)^4                                     rT^2                        rB^2
	/// 
	///@param cos_N_Nm is cos(angle(geometric_normal, micro_normal)) in [-1,+1].
	///@param cos_T_Nm is cos(angle(geometric_tangent, micro_normal)) in [-1,+1].
	///@param cos_B_Nm is cos(angle(geometric_bitangent, micro_normal)) in [-1,+1].
	///@param rT is roughness for tangent in [0,1].
	///@param rB is roughness for bitangent in [0,1].
	template<typename scalar>
	scalar beckmann_d(scalar cos_N_Nm, scalar cos_T_Nm, scalar cos_B_Nm, scalar rT, scalar rB) {
		if (cos_N_Nm > 0) {
			scalar mu_t = cos_T_Nm * cos_T_Nm / (rT * rT);
			scalar mu_b = cos_B_Nm * cos_B_Nm / (rB * rB);
			scalar sigma = rT * rB;
			scalar sq_cos_N_Nm = cos_N_Nm * cos_N_Nm;
			return exp( -(mu_t + mu_b)/sq_cos_N_Nm ) / ( static_cast<scalar>(3.1415926535897932384626433832795) * sigma * sq_cos_N_Nm*sq_cos_N_Nm );
		} else {
			return 0;
		}
	}
#endif

#if 0
template<typename vector3>
struct beckmann_distribution {
	using scalar = std::remove_cvref_t< decltype(std::declval<vector3>()[0]) >;

	// BeckmannDistribution Public Methods
	static scalar RoughnessToAlpha(scalar roughness) {
		roughness = max(roughness, scalar(1e-3));
		scalar x = std::log(roughness);
		return 1.62142f + 0.819955f * x + 0.1734f * x * x +
						0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
	}

	beckmann_distribution(scalar alphax, scalar alphay, bool samplevis = true)
		: alphax(max(scalar(0.001), alphax)), alphay(max(scalar(0.001), alphay)) {}

	scalar pdf(const scalar& cos_N_Nm) const {
		assert(alphax == alphay);
		scalar sin_N_Nm = 1 - cos_N_Nm;
		scalar tan_N_Nm = sin_N_Nm/cos_N_Nm;
		scalar tan2_N_Nm = tan_N_Nm*tan_N_Nm;
		if (std::isinf(tan2_N_Nm)) return scalar(0);
		scalar cos4_N_Nm = cos_N_Nm * cos_N_Nm * cos_N_Nm * cos_N_Nm;
		return exp(-tan2_N_Nm * alphax*alphax)/
			( 3.1415926535897932f * alphax*alphax * cos4_N_Nm);
	}

	template<typename vector2, typename configty>
	vector3 operator()(const vector2& u, const configty& unused = math::geometry::sphconfig<>{}) const {
		//if (!sampleVisibleArea) {
				// Sample full distribution of normals for Beckmann distribution

				// Compute $\tan^2 \theta$ and $\phi$ for Beckmann distribution sample
				scalar tan2Theta, azimuth;
				if (alphax == alphay) {
						tan2Theta = -alphax * alphax * log(1 - u[0]); assert(!isinf(tan2Theta));
						azimuth   = u[1] * 2 * 3.1415926535897932f;
				} else {
						// Compute _tan2Theta_ and _phi_ for anisotropic Beckmann
						// distribution
						/*Float logSample = std::log(1 - u[0]);
						DCHECK(!std::isinf(logSample));
						phi = std::atan(alphay / alphax *
														std::tan(2 * Pi * u[1] + 0.5f * Pi));
						if (u[1] > 0.5f) phi += Pi;
						Float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
						Float alphax2 = alphax * alphax, alphay2 = alphay * alphay;
						tan2Theta = -logSample /
												(cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);*/
				}

				// Map sampled Beckmann angles to normal direction _wh_
				scalar cos_elevation = 1/sqrt(1 + tan2Theta);
				scalar sin_elevation = sqrt(max(1 - cos_elevation * cos_elevation, scalar(0)));

				decltype(cos(azimuth)) normal, tangent, bitangent;
				if constexpr (configty::elevation_domain == elevation_0_pi) {
					normal = abs(cos_elevation);
					tangent = cos(azimuth) * sin_elevation;
					bitangent = sin(azimuth) * sin_elevation;
				} else {
					normal = abs(sin_elevation);
					tangent = cos(azimuth) * cos_elevation;
					bitangent = sin(azimuth) * cos_elevation;
				}

				if constexpr (configty::normal_pos == 0) {
					return vector3{normal,tangent,bitangent};
				} else if constexpr (configty::normal_pos == 1) {
					return vector3{bitangent,normal,tangent};
				} else {
					return vector3{tangent,bitangent,normal};
				}
		//} else {
		//    // Sample visible area of normals for Beckmann distribution
		//    Vector3f wh;
		//    bool flip = wo.z < 0;
		//    wh = BeckmannSample(flip ? -wo : wo, alphax, alphay, u[0], u[1]);
		//    if (flip) wh = -wh;
		//    return wh;
		//}
	}

	template<typename vector2, typename configty>
	vector3 operator()(const vector3& N, const vector2& u, const configty& unused = math::geometry::sphconfig<>{}) const {
		vector3 T, B; 
		math::geometry::get_tangent(N, T, B, configty{});

		vector3 Nm = this->operator()(u, unused);
		if constexpr (configty::normal_pos == 0) {
			return N*Nm[0] + T*Nm[1] + B*Nm[2];
		} else if constexpr (configty::normal_pos == 1) {
			return B*Nm[0] + N*Nm[1] + T*Nm[2];
		} else {
			return T*Nm[0] + B*Nm[1] + N*Nm[2];
		}
	}
	 
private:
	// BeckmannDistribution Private Methods
	//scalar Lambda(const Vector3f& w) const {
	//	scalar absTanTheta = std::abs(TanTheta(w));
 //   if (std::isinf(absTanTheta)) return 0.;
 //   // Compute _alpha_ for direction _w_
 //   Float alpha =
 //       std::sqrt(Cos2Phi(w) * alphax * alphax + Sin2Phi(w) * alphay * alphay);
 //   Float a = 1 / (alpha * absTanTheta);
 //   if (a >= 1.6f) return 0;
 //   return (1 - 1.259f * a + 0.396f * a * a) / (3.535f * a + 2.181f * a * a);
	//}

	// BeckmannDistribution Private Data
	scalar alphax, alphay;
};
#endif
//auto alpha = beckmann_distribution<vector3>::RoughnessToAlpha(roughness);
//beckmann_distribution<vector3> NDF(alpha, alpha);
//
//auto V  = -rayIO.d;
//auto Nm = NDF(N,X, math::geometry::sphconfig<>{});
//auto L  = math::geometry::reflect(rayIO.d, Nm);

//scalar cosThetaO = abs(dot(N,V)), cosThetaI = abs(dot(N,L));
//if (cosThetaI == 0 || cosThetaO == 0) {
//	result.transmittance *= 0;
//} else {
//	auto f = reflectance(abs(dot(rayIO.d, Nm)), etaI, eta, etak) * NDF.pdf(dot(N,Nm)) /** distribution->G(wo, wi)*/ /
//        (4 * cosThetaI * cosThetaO);
//	auto pdf =  NDF.pdf(dot(N,Nm)) / (4 * max(0.0f,dot(V,Nm)));
//	result.transmittance *= vmin(f/pdf,1);
//}
} }//end of namespace math::physics