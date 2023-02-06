#pragma once

///@brief Transport  
///@license Free 
///@review 2023-01-31 
///@contact Jiang1998Nan@outlook.com 
#define _MATH_PHYSICS_OPTICS_TRANSPORT_

#include <math/concepts.hpp>
#include <cassert>
#include <iostream>
#include <span>

#include "microfacet.hpp"

#include <math/geometry/transform.hpp>
#include <math/geometry/shape.hpp>
#include <geometry/range.hpp>
#include <geometry/tree.hpp>

#include <math/fraction.hpp>

namespace math { namespace physics {
	template<typename Number>
	Number small_sphere_phase(Number cos_angle/* = dot(ray.direction, light.vector)*/) {
		return (cos_angle*cos_angle + 1) * 3/4;
	}

	template<typename Number>
	Number henyey_greenstein_phase(Number cos_angle, Number g) {
		Number gg = g*g;
		Number den = 1 + gg - 2*g*cos_angle;
		return (1 - gg) / (den * sqrt(den));
	}


	template<typename scalar, typename spectrum>
	spectrum reflectance(const scalar& costhetaI, const spectrum& etaI, const spectrum& etaT/*, scalar* T = (scalar*)nullptr*/) {
		assert(0 <= costhetaI && costhetaI <= 1);
		spectrum eta = etaT/etaI;
		spectrum eta2 = eta * eta;

		spectrum eta_costhetaT = sqrt(max(eta2 - (1 - costhetaI*costhetaI), scalar(0)));
		spectrum eta2_costhetaI = eta2 * costhetaI;

		spectrum rTE = (costhetaI - eta_costhetaT)/(costhetaI + eta_costhetaT);
		spectrum rTM = (eta_costhetaT - eta2_costhetaI)/(eta2_costhetaI + eta_costhetaT);
		/*if (T) {
			scalar tTE = 2*costhetaI/(costhetaI + eta_costhetaT);
			scalar tTM = 2*eta*costhetaI/(eta*eta*costhetaI + eta_costhetaT);
			(*T) = eta_costhetaT/costhetaI * (tTE*tTE + tTM*tTM)/2;
			/// T = 1 - R.
		}*/
		///
		///		rTE^2 * cos(alpha)^2 + rTM^2 * sin(alpha)^2
		/// 
		/// if alpha = real_uniform_distribution(0,pi/2).mean() = pi/4;
		/// 
		///		rTE^2 * 0.5 + rTM^2 * 0.5
		/// 
		return (rTE*rTE + rTM*rTM)*scalar(0.5);
	}

	template<typename scalar, typename spectrum>
	spectrum reflectance(const scalar& costhetaI, const spectrum& etaI, const spectrum& etaT, const spectrum& etakT) {
		if (costhetaI <= 0) { spectrum{ 0,0,0 }; }
		//assert(0 <= costhetaI && costhetaI <= 1);
		spectrum eta = etaT/etaI;
		spectrum etak = etakT/etaI;

		scalar costhetaI2 = costhetaI * costhetaI;
		scalar sinthetaI2 = 1 - costhetaI2;
		spectrum eta2 = eta * eta;
		spectrum etak2 = etak * etak;

		spectrum t0 = eta2 - etak2 - sinthetaI2;
		spectrum a2plusb2 = sqrt(t0 * t0 + 4 * eta2 * etak2);
		spectrum t1 = a2plusb2 + costhetaI2;
		spectrum a = sqrt(scalar(0.5) * (a2plusb2 + t0));
		spectrum t2 = 2 * costhetaI * a;
		spectrum Rs = (t1 - t2) / (t1 + t2);

		spectrum t3 = costhetaI2 * a2plusb2 + sinthetaI2 * sinthetaI2;
		spectrum t4 = t2 * sinthetaI2;
		spectrum Rp = Rs * (t3 - t4) / (t3 + t4);

		return (Rp + Rs) * scalar(0.5);
	}

	template<typename spectrum, typename scalar>
	spectrum reflectance(spectrum f0, scalar cos_nv) {
		return f0 + (1 - f0)*pow(1 - cos_nv,5);
	}


	//random transport {V,L} around TBN with interaction{etaI,mtl}.

	template<typename vector3, typename spectrum = vector3>
	struct conductor {
		spectrum eta;
		spectrum etak;
		vector3 roughness;
	};

	template<typename vector3, typename basis3, typename spectrum>
	spectrum transmission(const vector3& V, const vector3& L, const basis3& TBN, 
		const spectrum& etaI, const conductor<vector3,spectrum>& mtl) {
		using scalar = std::remove_cvref_t<decltype(std::declval<vector3>()[0])>;
		scalar cos_N_V = dot(TBN.normal(), V);
		if (cos_N_V <= 0) { return math::ones<spectrum>(0); }
		scalar cos_N_L = dot(TBN.normal(), L);
		if (cos_N_L <= 0) { return math::ones<spectrum>(0); }

		vector3 Nm = normalize(V + L);
		scalar cos_N_Nm = dot(TBN.normal(), Nm);
		if (cos_N_Nm <= 0) { return math::ones<spectrum>(0); }
		scalar cos_Nm_V = dot(Nm, V);
		if (cos_Nm_V <= 0) { return math::ones<spectrum>(0); }
		scalar cos_Nm_L = dot(Nm, L);
		if (cos_Nm_L <= 0) { return math::ones<spectrum>(0); }

		spectrum F = reflectance(cos_Nm_V, etaI, mtl.eta, mtl.etak);
		if (mtl.roughness[0] == mtl.roughness[1]) {
			scalar  r2 = mtl.roughness[0] * mtl.roughness[0];
			scalar   D = ggx_base<scalar>::distribution(cos_N_Nm, r2);
			scalar Vis = ggx_base<scalar>::height_correlated_visibility(cos_N_V, cos_N_L, r2);
			return (/*(1-F)/std::numbers::pi_v<scalar> + */F*D*Vis) * cos_N_L;
		} else {
			scalar   D = ggx_base<scalar>::distribution(cos_N_Nm, dot(TBN.tangent0(),Nm), dot(TBN.tangent1(),Nm),  mtl.roughness[0], mtl.roughness[1]);
			scalar Vis = ggx_base<scalar>::height_correlated_visibility(cos_N_V, dot(TBN.tangent0(),V), dot(TBN.tangent1(),V),  cos_N_L, dot(TBN.tangent0(),L), dot(TBN.tangent1(),L),  mtl.roughness[0], mtl.roughness[1]);
			return (/*(1-F)/std::numbers::pi_v<scalar> + */F*D*Vis) * cos_N_L;
		}
	}

	template<typename vector2, typename vector3, typename basis3, typename spectrum>
	spectrum transport(const vector2& Xi, const vector3& V, vector3& L, const basis3& TBN, 
		const spectrum& etaI, const conductor<vector3,spectrum>& mtl) {
		using scalar = std::remove_cvref_t<decltype(std::declval<vector3>()[0])>;
		scalar cos_N_V = dot(TBN.normal(), V);
		if (cos_N_V <= 0) { return math::ones<spectrum>(0); }
		
		vector3 Nm = ggx_visible_normal<vector3>::generate(Xi, TBN, V, mtl.roughness);
		scalar cos_N_Nm = dot(TBN.normal(), Nm);
		if (cos_N_Nm <= 0) { return math::ones<spectrum>(0); }
		scalar cos_Nm_V = dot(Nm, V);
		if (cos_Nm_V <= 0) { return math::ones<spectrum>(0); }

		L = math::geometry::mirror(V, Nm);
		scalar cos_Nm_L = dot(Nm, L);
		if (cos_Nm_L <= 0) { return math::ones<spectrum>(0); }
		scalar cos_N_L = dot(TBN.normal(), L);
		if (cos_N_L <= 0) { return math::ones<spectrum>(0); }

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
		spectrum F = reflectance(cos_Nm_V, etaI, mtl.eta, mtl.etak);
		if (mtl.roughness[0] == mtl.roughness[1]) {
			scalar r2 = mtl.roughness[0] * mtl.roughness[0];
			return F * (ggx_base<scalar>::height_correlated_masking_shadowing(cos_N_V, cos_N_L, r2)
				/ ggx_base<scalar>::masking_shadowing(cos_N_V, r2));
		}
		scalar cos_T0_V = dot(TBN.tangent0(), V);
		scalar cos_T1_V = dot(TBN.tangent1(), V);
		return F * (ggx_base<scalar>::height_correlated_masking_shadowing(cos_N_V, cos_T0_V, cos_T1_V, cos_N_L, dot(TBN.tangent0(), L), dot(TBN.tangent1(), L), mtl.roughness[0], mtl.roughness[1])
			/ ggx_base<scalar>::masking_shadowing(cos_N_V, cos_T0_V, cos_T1_V, mtl.roughness[0], mtl.roughness[1]));
	}
	
	template<typename vector3, typename spectrum = vector3>
	struct diffuse_brdf {
		spectrum color;
		//scalar roughness = 1.0; else difficult sampling.
	};

	template<typename vector3, typename basis3, typename spectrum>
	spectrum transmission(const vector3& V, const vector3& L, const basis3& TBN, 
		const spectrum& etaI, const diffuse_brdf<vector3,spectrum>& mtl) {
		using scalar = std::remove_cvref_t<decltype(std::declval<vector3>()[0])>;
		scalar cos_N_V = dot(TBN.normal(), V);
		if (cos_N_V <= 0) { return math::ones<spectrum>(0); }
		scalar cos_N_L = dot(TBN.normal(), L);
		if (cos_N_L <= 0) { return math::ones<spectrum>(0); }

		return mtl.color * (cos_N_L/std::numbers::pi_v<scalar>);
	}

	template<typename vector2, typename vector3, typename basis3, typename spectrum>
	spectrum transport(const vector2& Xi, const vector3& V, vector3& L, const basis3& TBN, 
		const spectrum& etaI, const diffuse_brdf<vector3,spectrum>& mtl) {
		using scalar = std::remove_cvref_t<decltype(std::declval<vector3>()[0])>;
		scalar cos_N_V = dot(TBN.normal(), V);
		if (cos_N_V <= 0) { return math::ones<spectrum>(0); }

		scalar cos_elevation = sqrt(1 - Xi[0]);
		scalar sin_elevation = sqrt(1 - cos_elevation*cos_elevation);
		scalar azimuth = Xi[1] * std::numbers::pi_v<scalar>*2;
		L = TBN.normal() * cos_elevation +
			TBN.tangent0() * (cos(azimuth) * sin_elevation) +
			TBN.tangent1() * (sin(azimuth) * sin_elevation);
		assert( dot(L, TBN.normal()) >= 0 );

		/// BSDF = color/pi
		/// PDF = cos(N,L)/pi
		///                      color/pi
		/// BSDF/PDF*cos(N,L) = ------------- * cos(N,L)
		///                      cos(N,L)/pi
		return mtl.color;
	}

	template<typename vector3, typename spectrum = vector3>
	struct principle_brdf {
		using scalar = std::remove_cvref_t<decltype(std::declval<vector3>()[0])>;
		spectrum color;
		scalar metallic;
		vector3 roughness;
	};

	template<typename vector3, typename basis3, typename spectrum>
	spectrum transmission(const vector3& V, const vector3& L, const basis3& TBN, 
		const spectrum& etaI, const principle_brdf<vector3,spectrum>& mtl) {
		using scalar = std::remove_cvref_t<decltype(std::declval<vector3>()[0])>;
		scalar cos_N_V = dot(TBN.normal(), V);
		if (cos_N_V <= 0) { return math::ones<spectrum>(0); }
		scalar cos_N_L = dot(TBN.normal(), L);
		if (cos_N_L <= 0) { return math::ones<spectrum>(0); }

		vector3 Nm = normalize(V + L);
		scalar cos_N_Nm = dot(TBN.normal(), Nm);
		if (cos_N_Nm <= 0) { return math::ones<spectrum>(0); }
		scalar cos_Nm_V = dot(Nm, V);
		if (cos_Nm_V <= 0) { return math::ones<spectrum>(0); }
		scalar cos_Nm_L = dot(Nm, L);
		if (cos_Nm_L <= 0) { return math::ones<spectrum>(0); }

		///		lerp( lerp(0.08, color, metallic), 1.0, f )
		///		 = ( 0.08*(1-metallic) + color*metallic )*(1-f) + f
		///		 = ( 0.08 + (color - 0.08)*metallic )*(1-f) + f
		///Cannot get a form of <q>metallic * X</q>.
		spectrum F = reflectance((mtl.color - 0.08)*mtl.metallic + 0.08, cos_Nm_V);
		spectrum Pss = mtl.color * (1 - mtl.metallic);
		if (mtl.roughness[0] == mtl.roughness[1]) {
			scalar  r2 = mtl.roughness[0] * mtl.roughness[0];
			scalar   D = ggx_base<scalar>::distribution(cos_N_Nm, r2);
			scalar Vis = ggx_base<scalar>::height_correlated_visibility(cos_N_V, cos_N_L, r2);
			return ((1-F)*Pss/std::numbers::pi_v<scalar> + F*D*Vis) * cos_N_L;
		} else {
			scalar   D = ggx_base<scalar>::distribution(cos_N_Nm, dot(TBN.tangent0(),Nm), dot(TBN.tangent1(),Nm),  mtl.roughness[0], mtl.roughness[1]);
			scalar Vis = ggx_base<scalar>::height_correlated_visibility(cos_N_V, dot(TBN.tangent0(),V), dot(TBN.tangent1(),V),  cos_N_L, dot(TBN.tangent0(),L), dot(TBN.tangent1(),L),  mtl.roughness[0], mtl.roughness[1]);
			return ((1-F)*Pss/std::numbers::pi_v<scalar> + F*D*Vis) * cos_N_L;
		}
	}

	template<typename vector2, typename vector3, typename basis3, typename spectrum>
	spectrum transport(const vector2& Xi, const vector3& V, vector3& L, const basis3& TBN, 
		const spectrum& etaI, const principle_brdf<vector3,spectrum>& mtl) {
		///Note: is biased, max(error) ~= 0.02.
		using scalar = std::remove_cvref_t<decltype(std::declval<vector3>()[0])>;
		scalar cos_N_V = dot(TBN.normal(), V);
		if (cos_N_V <= 0) { return math::ones<spectrum>(0); }
																																																																																																					
		vector3 Nm = ggx_visible_normal<vector3>::generate(Xi, TBN, V, mtl.roughness);
		scalar cos_N_Nm = dot(TBN.normal(), Nm);
		if (cos_N_Nm <= 0) { return math::ones<spectrum>(0); }
		scalar cos_Nm_V = dot(Nm, V);
		if (cos_Nm_V <= 0) { return math::ones<spectrum>(0); }

		spectrum F = reflectance((mtl.color - 0.08)*mtl.metallic + 0.08, cos_Nm_V);
		spectrum Pss = mtl.color * (1 - mtl.metallic);
		auto ff = (F[0]+F[1]+F[2])/3;
		if (Xi[0] < ff) {
			L = math::geometry::mirror(V, Nm);
			scalar cos_Nm_L = dot(Nm, L);
			if (cos_Nm_L <= 0) { return math::ones<spectrum>(0); }
			scalar cos_N_L = dot(TBN.normal(), L);
			if (cos_N_L <= 0) { return math::ones<spectrum>(0); }

			if (mtl.roughness[0] == mtl.roughness[1]) {
				scalar r2 = mtl.roughness[0] * mtl.roughness[0];
				return F/ff * (ggx_base<scalar>::height_correlated_masking_shadowing(cos_N_V, cos_N_L, r2)
					/ ggx_base<scalar>::masking_shadowing(cos_N_V, r2));
			}
			scalar cos_T0_V = dot(TBN.tangent0(), V);
			scalar cos_T1_V = dot(TBN.tangent1(), V);
			return F/ff * (ggx_base<scalar>::height_correlated_masking_shadowing(cos_N_V, cos_T0_V, cos_T1_V, cos_N_L, dot(TBN.tangent0(), L), dot(TBN.tangent1(), L), mtl.roughness[0], mtl.roughness[1])
				/ ggx_base<scalar>::masking_shadowing(cos_N_V, cos_T0_V, cos_T1_V, mtl.roughness[0], mtl.roughness[1]));
		} 
		else {
			scalar cos_elevation = sqrt(Xi[0]);
			scalar sin_elevation = sqrt(1 - cos_elevation*cos_elevation);
			scalar azimuth = Xi[1] * std::numbers::pi_v<scalar>*2;
			L = TBN.normal() * cos_elevation +
				TBN.tangent0() * (cos(azimuth) * sin_elevation) +
				TBN.tangent1() * (sin(azimuth) * sin_elevation);
			L = normalize(L);
			assert( dot(L, TBN.normal()) >= 0 );

			return (1-F)/(1-ff) * Pss;
		}
	}

	template<typename vector3, typename spectrum = vector3>
	struct translucent_bsdf {
		spectrum color;
	};

	template<typename vector3, typename basis3, typename spectrum>
	spectrum transmission(const vector3& V, const vector3& L, const basis3& TBN, 
		const spectrum& etaI, const translucent_bsdf<vector3,spectrum>& mtl) {
		using scalar = std::remove_cvref_t<decltype(std::declval<vector3>()[0])>;
		scalar cos_N_V = dot(TBN.normal(), V);
		if (cos_N_V <= 0) { return math::ones<spectrum>(0); }
		scalar cos_N_L = dot(TBN.normal(), L);
		if (cos_N_L >= 0) { return math::ones<spectrum>(0); }

		return mtl.color * (-cos_N_L/std::numbers::pi_v<scalar>);
	}

	template<typename vector2, typename vector3, typename basis3, typename spectrum>
	spectrum transport(const vector2& Xi, const vector3& V, vector3& L, const basis3& TBN, 
		const spectrum& etaI, const translucent_bsdf<vector3,spectrum>& mtl) {
		using scalar = std::remove_cvref_t<decltype(std::declval<vector3>()[0])>;
		scalar cos_N_V = dot(TBN.normal(), V);
		if (cos_N_V <= 0) { return math::ones<spectrum>(0); }

		scalar cos_elevation = sqrt(1 - Xi[0]);
		scalar sin_elevation = sqrt(1 - cos_elevation*cos_elevation);
		scalar azimuth = Xi[1] * std::numbers::pi_v<scalar>*2;
		L = TBN.normal() * cos_elevation +
			TBN.tangent0() * (cos(azimuth) * sin_elevation) +
			TBN.tangent1() * (sin(azimuth) * sin_elevation);
		assert( dot(L, TBN.normal()) >= 0 );
		L = -L;

		/// BSDF = color/pi
		/// PDF = cos(N,L)/pi
		///                      color/pi
		/// BSDF/PDF*cos(N,L) = ------------- * cos(N,L)
		///                      cos(N,L)/pi
		return mtl.color;
	}

/// Lightpath reflects until it nohit.
/// Besides, the path may pass through a lightsource, in which case it will straightly transmission
/// the lightsource to reflect on the next surface. this transmission will accumulate the radiance
/// of the lightsource. but this case is not transmission, so new named "reflect with lightsource".
/// 
///		p0                      \ p1
///		*-- --> -- -- -- -- -- --\
///		                        / \
///		         ----p3--      /   \
///		            / \       /
///		          |/_  *lit1 /
///		         *lit2  \   /
///		                 \ /
///		              ----p2---
/// 
///		L(p[1] -> p[0]) = product<j=0,...>( R(p[j],p[j+1]) ) * L[i], accumualte all lightsoucrs.
/// 
/// Perfect can also be refraction, so named "transmission".
/// 
///		p0                      \ p1
///		*-- --> -- -- -- -- -- --\
///		           _            / \
///		         /|  /         /   \
///		          |/p4        /
///		          /\     ---p2---
///		        /    \     /
///		               \ /
///		            ----p3---
/// 
///		L(p[1] -> p[0]) = product<j=0,...>( T(p[j],p[j+1]) ) * L[i], accumualte all lightsoucrs.
/// 
template<typename spectrum, typename intersection, typename transmission, typename ray,
	typename intersect_surface_and_lightsource_fn, 
	typename transmission_with_lightsource_fn,
	typename accumulate_nohit_radiance_fn>
auto perfect_surface_raytrace(const ray& start_ray, const intersect_surface_and_lightsource_fn& intersect_surface_and_lightsource, const transmission_with_lightsource_fn& transmission_with_lightsource, const accumulate_nohit_radiance_fn& accumulate_nohit_radiance) {
	ray          the_ray          = start_ray;
	intersection the_intersection;
	transmission the_transmission;
	size_t i = 0, max_iterations = 12;
	while (i++ != max_iterations && intersect_surface_and_lightsource(the_ray, the_intersection)) {
		transmission_with_lightsource(the_ray, the_intersection, the_transmission);
		if (the_transmission.transmittance == 0) {
			break;
		}
	}
	return accumulate_nohit_radiance(the_ray, the_transmission);
}

/// Material can also be rough, there are at least 10^8 >= 2^32 lightpaths. (example: 1 * 50 * 2 * 20 * ...)
/// 
///		p0                      \ p1
///		*-- --> -- -- -- -- -- --\
///		                        /|\
///		                     / / | \
///		                  /   /  |
///		               /     /   |
///		           |/_     \/_  \|/
/// 
///		L(p[1] -> p[0]) = integral<n[0]>( T(p[0],t(p[0],n[0])) * integral( ... integral<n[N]>( T(p[N],t(p[N],n[N])) ) ... )) * L[i], accumualte all lightsoucrs.
///	
/// Fortunately there is probability method, so we want to use the probability methods.
/// 
///		p0                      \ p1
///		*-- --> -- -- -- -- -- --\
///		                        /|\
///		                     / / | \
///		                  /   /  |
///		              2/    1/  3|
///		           |/_     \/_  \|/
/// 
///		L(p[1] -> p[0]) = avg<n[0]>( T(p[0],t(p[0],n[0]))/pdf(n[0]) * avg( ... *  avg<n[N]>( T(p[N],t(p[N],n[N]))/pdf(n[N]) ) ... )) * L[i], accumualte all lightsoucrs.
/// 
///			= sum<n[0]>( T(p[0],t(p[0],n[0]))/pdf(n[0]) * sum( ... *  sum<n[N]>( T(p[N],t(p[N],n[N]))/pdf(n[N]) ) ... )) / size(union(n[0], ..., n[N-1], n[N])) * L[i], accumualte all lightsoucrs.
/// 
///			                                       |<---------------------- lightpath ---------------------->|
///			   sum<union(n[0], ..., n[N-1], n[N])>( product<j=0,...>(T(p[j],t(p[j],n[j]))/pdf(n[j])) )
///			= ----------------------------------------------------------------------------------------- * L[i], accumualte all lightsoucrs.
///			                         size(union(n[0], ..., n[N-1], n[N]))
/// 
#if 0
template<typename spectrum, typename intersection, typename transmission, typename ray, typename random_vector,
	typename intersect_surface_and_lightsource_fn, 
	typename transmission_with_lightsource_fn, 
	typename accumulate_nohit_radiance_fn>
auto random_path_raytrace(const ray& start_ray, const random_vector& X, const intersect_surface_and_lightsource_fn& intersect_surface_and_lightsource, const transmission_with_lightsource_fn& transmission_with_lightsource, const accumulate_nohit_radiance_fn& accumulate_nohit_radiance) {
	ray          the_ray          = start_ray;
	intersection the_intersection;
	transmission the_transmission;
	size_t i = 0, max_iterations = 12;
	while (i++ != max_iterations && intersect_surface_and_lightsource(the_ray, the_intersection)) {
		transmission_with_lightsource(the_ray, the_intersection, X, the_transmission);
		if (the_transmission.transmittance == 0) {
			break;
		}
	}
	return accumulate_nohit_radiance(the_ray, the_transmission);
}
#endif
/// Very slow convergence for small lightsources.
///
///		p0                      \ p1
///		*-- --> -- -- -- -- -- --\
///		                        /|\
///		                     / / | \
///		                  /   /  |
///		              2/    1/  3|
///		           |/_     \/_  \|/
/// 
///		            *lit
/// 
/// In order to faster convergence we can use BRDF"Bidirectional Reflectance Distribution Function"
/// instead of intersection of ray-lightsource. 
/// And note that we still support transmission with lightsource for no extra cost usually the lig-
/// htsources in light-list requires shapeless, others named "emissive surface".
/// 
///		...
/// 
template<typename spectrum, typename intersection, typename transmission, typename ray, typename random_vector,
	typename intersect_surface_and_emissive_fn, 
	typename accumulate_lightsource_fn, 
	typename transmission_with_emissive_fn, 
	typename accumulate_nohit_radiance_fn>
auto random_path_raytrace(const ray& start_ray, const random_vector& X, const intersect_surface_and_emissive_fn& intersect_surface_and_emissive, const accumulate_lightsource_fn& accumulate_lightsource, const transmission_with_emissive_fn& transmission_with_emissive, const accumulate_nohit_radiance_fn& accumulate_nohit_radiance) {
	ray          the_ray          = start_ray;
	intersection the_intersection;
	transmission the_transmission;
	size_t i = 0, max_iterations = 24;
	while (i++ != max_iterations && intersect_surface_and_emissive(the_ray, the_intersection)) {
		//accumulate_lightsource(the_ray, the_intersection, X, the_transmission);
		transmission_with_emissive(the_ray, the_intersection, X, the_transmission);
		if (the_transmission.transmittance == 0) {
			break;
		}
	}
	return accumulate_nohit_radiance(the_ray, the_transmission);
}

#if 0
template<typename vector3>
struct henyeygreenstein_distribution {
	using scalar = std::remove_cvref_t< decltype(std::declval<vector3>()[0]) >;

	henyeygreenstein_distribution(scalar g) : g(g) {}

	template<typename vector2, typename configty>
	vector3 operator()(const vector2& u, const configty& unused /*= math::geometry::coordinate_traits<>{}*/) const {
		scalar cos_elevation;
		if (std::abs(g) < 1e-3)
			cos_elevation = 1 - 2 * u[0];
		else {
			scalar gg = g * g;
			scalar sqrTerm = (1 - gg) / (1 + g - 2*g*u[0]);
			cos_elevation = (1 + gg - sqrTerm*sqrTerm)/(2*g);
		}

		// Compute direction _wi_ for Henyey--Greenstein sample
		scalar sin_elevation = std::sqrt(std::max((scalar)0, 1 - cos_elevation * cos_elevation));
		scalar azimuth = 2 * 3.1415926535897932f * u[1];

		decltype(cos(azimuth)) normal, tangent, bitangent;
		if constexpr (configty::elevation_domain == 0/*elevation_0_pi*/) {
			normal = cos_elevation;
			tangent = cos(azimuth) * sin_elevation;
			bitangent = sin(azimuth) * sin_elevation;
		} else {
			normal = sin_elevation;
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
	}

	template<typename vector2, typename configty>
	vector3 operator()(const vector3& N, const vector2& u, const configty& unused = math::geometry::coordinate_traits<>{}) const {
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
	scalar g;
};
#endif

namespace raytracing {
	using namespace math::geometry;
	using namespace geometry;

/// Raytracing means: The ray hit something one after another, These things interact with each 
/// other and any affect the subsequent ray hits. 
/// 
/// Therefore we need known the ray, the hit_result (intersect_result), the interact_result. Thery
/// all have two definitions:
/// (1) As tparam, all concrete structures must use same ray, same hit_result, and same ....
/// (2) As base structure, all concrete structures can be use different ray, different ..., but 
///     some additional flags are required (because do not use dynamic_cast check derived structure).
/// 
/// Because (1) the flags checking wastes performance unless we use unsafe method without checking
/// anything. And (2) we always use only one ray, hit_result, ... different types are useless. So 
/// we use second definition.

	template<typename traits> 
	struct primitive {
		using scalar_type    = typename traits::scalar_type;
		using vector3_type   = typename traits::vector3_type;
		using spectrum_type  = typename traits::spectrum_type;
		using ray_type       = typename traits::ray_type;
		using arguments_type = typename traits::arguments_type;
		using intersect_result_type = typename traits::intersect_result_type;
		using interact_result_type  = typename traits::interact_result_type;
		using attribute_type = typename traits::attribute_type;

		/*virtual bool intersect(const ray_type&, const arguments_type&, intersect_result_type&) const = 0;*/

/// New meaning: The ray hit the boundary of something one after another, These boundaries interact
/// with each other in one interior and any effect the subsequent ray hits.
/// 
/// Because (1) interior detection may be more expensive than boundary detection, so used a stack
/// push primitive at enter and pop primitive at exit. And (2) the "exit" must return true for 
/// safety stack. Therefore the "intersect" is divide to { "enter" and "exit" }.
/// And (3) the "enter" can be optimized. (4) The new collision detection does not need to offset
/// (without unclosed geometry).

		virtual bool enter(const ray_type&, const arguments_type&, intersect_result_type&, bool is_self/*for avoid self-intersects*/) const = 0;

		virtual void exit(const ray_type&, const arguments_type&, intersect_result_type&) const = 0;

		virtual void get_attributes(const intersect_result_type&, const size_t normals_count, const size_t texcoords_count, const size_t others_count,
			attribute_type[]) const = 0;

		virtual bool intersects(const ray_type&, const range<scalar_type>&) const = 0;

		virtual spectrum_type get_transmittance(const ray_type&, const range<scalar_type>&) const = 0;

		virtual scalar_type march_interior(const ray_type&, const arguments_type&, const intersect_result_type&,
			interact_result_type&) const { abort(); }

		virtual void interact_interior(ray_type&, const arguments_type&, const intersect_result_type&, const scalar_type distance,
			interact_result_type&) const {}
		
		virtual void interact_boundary(ray_type&, const arguments_type&, const intersect_result_type&,
			interact_result_type&) const {}

		virtual bounds<vector3_type> boundary() const { abort(); }
	};

	template<typename traits>
	struct material {
		using scalar_type    = typename traits::scalar_type;
		using vector3_type   = typename traits::vector3_type;
		using spectrum_type  = typename traits::spectrum_type;
		using ray_type       = typename traits::ray_type;
		using arguments_type = typename traits::arguments_type;
		using intersect_result_type = typename traits::intersect_result_type;
		using interact_result_type  = typename traits::interact_result_type;
		using attribute_type = typename traits::attribute_type;

		virtual scalar_type march_interior(const ray_type&, const arguments_type&, const intersect_result_type& x,
			interact_result_type&) const { return x.distance; }

		virtual void interact_interior(ray_type&, const arguments_type&, const intersect_result_type&, const scalar_type distance,
			interact_result_type&) const {}
		
		virtual void interact_boundary(ray_type&, const arguments_type&, const intersect_result_type&,
			interact_result_type&) const {}

		virtual spectrum_type get_transmittance(const ray_type&, const range<scalar_type>&) const { return math::ones<spectrum_type>(0); }

		virtual spectrum_type get_transmittance(const ray_type&, const range<scalar_type>&, const intersect_result_type&) const { return math::ones<spectrum_type>(0); }

		bool doublesided = false;
		bool opaque = true;
	};

	template<typename traits>
	struct primitive2 : virtual primitive<traits> {
		using scalar_type    = typename traits::scalar_type;
		using vector3_type   = typename traits::vector3_type;
		using spectrum_type  = typename traits::spectrum_type;
		using ray_type       = typename traits::ray_type;
		using arguments_type = typename traits::arguments_type;
		using intersect_result_type = typename traits::intersect_result_type;
		using interact_result_type  = typename traits::interact_result_type;
		using attribute_type = typename traits::attribute_type;

		material<traits>* material;

		virtual scalar_type march_interior(const ray_type& x, const arguments_type& y, const intersect_result_type& z,
			interact_result_type& w) const override { return material->march_interior(x,y,z,w); }

		virtual void interact_interior(ray_type& x, const arguments_type& y, const intersect_result_type& z, const scalar_type distance,
			interact_result_type& w) const override { material->interact_interior(x,y,z,distance,w); }
		
		virtual void interact_boundary(ray_type& x, const arguments_type& y, const intersect_result_type& z,
			interact_result_type& w) const override { material->interact_boundary(x,y,z,w); }
	};

	template<typename intersect_result>
	concept requires_intersect_result = requires (intersect_result& result) {
		result.distance;
		result.primitive;
		result.element;
	};

	template<typename interact_result>
	concept requires_interact_result = requires (interact_result& result){
		result.radiance;
		result.transmittance;
	};

	template<typename arguments>
	concept requires_uniform_random = requires(const arguments& args) {
		args.random;
	};

	template<typename interact_result>
	concept requires_importance_sample_lightsource = requires(interact_result& result) {
		result.incident_ray;
		result.incident_distance;
		result.incident_visibility;
		result.incident_radiance;
	};


///Examples:

	template<typename traits, typename shape>
	struct shape_primitive : primitive2<traits>, shape {
		using ray_type       = typename traits::ray_type;
		using arguments_type = typename traits::arguments_type;
		using intersect_result_type = typename traits::intersect_result_type;
		using interact_result_type  = typename traits::interact_result_type;
		using attribute_type = typename traits::attribute_type;
		using spectrum = typename traits::spectrum_type;
		using vector3 = typename traits::vector3_type;
		using scalar  = typename traits::scalar_type;

		bool invert = false;

		virtual bool enter( const ray_type& ray, const arguments_type&, intersect_result_type& result, bool is_self ) const override {
			auto section = math::geometry::intersection(static_cast<const shape&>(*this), ray);
			if constexpr (std::is_same_v<decltype(section), range<scalar>>) 
			{
				if (invert) {
					if (!empty(section) && 0 <= std::end(section) && std::end(section) < result.distance) {
						result.distance = std::end(section);
						result.primitive = this;
						return true;
					}
				}

				///if( in range )
				if (!empty(section) && 0 <= std::begin(section) && std::begin(section) < result.distance) {///in_range
					result.distance  = std::begin(section);
					result.primitive = this;
					return true;
				}
			} 
			else 
			{
				///if( in range && faceforward )
				if (0 <= section && section < result.distance  && dot(ray.direction(), this->normal()) < 0 ) {
					result.distance  = section;
					result.primitive = this;
					return true;
				}
			}
			return false;
		}

		virtual void exit( const ray_type& ray, const arguments_type&, intersect_result_type& result ) const override {
			assert( !invert );
			auto section = math::geometry::intersection(static_cast<const shape&>(*this), ray);
			if constexpr (std::is_same_v<decltype(section), range<scalar>>) 
			{
				if (empty(section)) {
					throw std::exception("at exit(){ eject from self }");
				} else {
					scalar distance = std::end(section);
						distance = max(distance, scalar(0));///avoid self-intersection again.
					if (distance < result.distance) {
						result.distance  = distance;
						result.primitive = this;
						return;
					}
				}
			} 
			else 
			{
				scalar distance = max(section, scalar(0));
				if (distance < result.distance) {
					assert( dot(ray.direction(), this->normal()) >= 0 );
					result.distance  = section;
					result.primitive = this;
					return;
				}
			}
			throw std::exception("at exit(){ Unkown Exception }");
		}

		virtual bool intersects( const ray_type& ray, const range<scalar>& therange ) const override {
			auto section = math::geometry::intersection(static_cast<const shape&>(*this), ray);
			if constexpr (std::is_same_v<decltype(section), range<scalar>>) {
				if (::geometry::intersects(therange, section)) {
					return true;
				}
			} else if constexpr (std::is_same_v<decltype(section), scalar>) {
				if (contains(therange, section)) {
					return true;
				}
			} else {
				abort();
			}

			return false;
		}

		virtual spectrum get_transmittance( const ray_type& ray, const range<scalar>& therange ) const override {
			auto section = math::geometry::intersection(static_cast<const shape&>(*this), ray);
			if constexpr (std::is_same_v<decltype(section), range<scalar>>) {
				if (!empty(section) && (contains(therange,std::begin(section)) || contains(therange, std::end(section)))/*::geometry::intersects(therange, section)*/ ) {
					return this->material->get_transmittance(ray, intersection(therange, section));
				}
			} else if constexpr (std::is_same_v<decltype(section), scalar>) {
				if (contains(therange, section)) {
					return ones<spectrum>(0);
				}
			} else {
				abort();
			}

			return ones<spectrum>(1);
		}

		virtual void get_attributes( const intersect_result_type& result, const size_t normals_count, const size_t texcoords_count, const size_t others_count, 
			attribute_type attributes[]) const override {
			const auto& my_shape = static_cast<const shape&>(*this);
			size_t i = 0;

			auto& position = reinterpret_cast<vector3&>(attributes[i++]);
				position = result.ray(result.distance);

			auto& normal = reinterpret_cast<vector3&>(attributes[i++]);
			if constexpr (std::is_same_v<shape, math::geometry::sphere<vector3>>) {
				normal = normalize(position - my_shape.center());
			} else if constexpr (std::is_same_v<shape, math::geometry::box<vector3>>) {
				vector3 sign = (position - my_shape.center())/my_shape.halfextents();
				vector3 det  = abs(sign);
				if (det[0] > det[1] && det[0] > det[2]) {
					normal = {copysign(scalar(1),sign[0]), 0, 0};
				} else if (det[1] > det[0] && det[1] > det[2]) {
					normal = {0, copysign(scalar(1),sign[1]), 0};
				} else {
					normal = {0, 0, copysign(scalar(1),sign[2])};
				}
			} else if constexpr (std::is_same_v<shape, math::geometry::plane<vector3>>) {
				normal = my_shape.normal();
			} else {
				abort();
			}
			if (invert) {
				normal = -normal;
			}

			position = result.transformation.transform(position);
			normal = result.transformation.transform_for_normal(normal);

			if (normals_count == 3) {
				auto space = math::geometry::oriented_orthogonal_basis<vector3>::from_normal(normal);
				auto& tangent0 = reinterpret_cast<vector3&>(attributes[i++]);
				auto& tangent1 = reinterpret_cast<vector3&>(attributes[i++]);
				tangent0 = space.tangent0();
				tangent1 = space.tangent1();
				//get_tangent(normal, tangent0, tangent1, coordinate_traits<>{});
			}
		}

		virtual bounds<vector3> boundary() const override {
			//auto inf = std::numeric_limits<scalar>::infinity();
			//return bounds<vector3>{{-inf,0,-inf}, {inf,0,inf}};
			const auto& my_shape = static_cast<const shape&>(*this);
			return { my_shape.center() - my_shape.halfextents(), my_shape.center() + my_shape.halfextents() };
			//abort();
		}
	};

#if 0
	template<typename ray, typename arguments, typename intersect_result, typename interact_result, typename attribute, typename shape>
	struct shape_lightsource : shape_primitive<ray, arguments, intersect_result, interact_result, attribute, shape> {

		spectrum power;
		//ray_vector_t<ray> direction;

		virtual void interact_boundary( ray& ray, const arguments& arguments, const intersect_result& intersection, interact_result& result ) const override {
			result.radiance += power * result.transmittance/* * max(scalar(0),dot(-rayIO.d,direction))*/;
			//result.transmittance *= 0;
			ray.start_point() += ray.direction() * (intersection.distance + this->radius() * 2 + 1e-3f);
		}
	};
#endif

	template<typename traits, typename element_index_type>
	struct triangle_mesh : primitive2<traits> {
		using ray_type       = typename traits::ray_type;
		using arguments_type = typename traits::arguments_type;
		using intersect_result_type = typename traits::intersect_result_type;
		using interact_result_type  = typename traits::interact_result_type;
		using attribute_type = typename traits::attribute_type;
		using spectrum = typename traits::spectrum_type;
		using vector3 = typename traits::vector3_type;
		using scalar  = typename traits::scalar_type;

		std::vector<std::shared_ptr<std::vector<unsigned char>>> buffers;
		
		struct attribute_binding {
			struct type_info {
				size_t hash_code;
				bool normalied;
				size_t size;
				size_t alignment;
				size_t length;
			} type;
			struct buffer_type {
				unsigned char* data;
				size_t stride;
			} stream;
		};
		std::vector<attribute_binding> vertex_attributes;
		size_t num_vertices{0};
		std::span<element_index_type> elements;
		/*element_index_type* elements{nullptr};
		size_t num_elements{0};*/

		void _Tidy_buffers() {
			/// ...
		}

		void bind_vertex_stream(size_t k, std::shared_ptr<std::vector<unsigned char>> buffer, size_t offset, size_t stride) {
			assert( k < vertex_attributes.size() );
			if (std::find(buffers.begin(), buffers.end(), buffer) == buffers.end()) {
				buffers.push_back(buffer);
			}
			
			vertex_attributes[k].stream = { std::next(buffer->data(), offset), stride };
			_Tidy_buffers();
		}

		void bind_element_stream(std::shared_ptr<std::vector<unsigned char>> buffer, size_t offset, size_t length) {
			if (std::find(buffers.begin(), buffers.end(), buffer) == buffers.end()) {
				buffers.push_back(buffer);
			}

			elements = {reinterpret_cast<element_index_type*>(std::next(buffer->data(), offset)), length/sizeof(element_index_type)};
			_Tidy_buffers();
		}

		attribute_type get_attribute(size_t k, size_t i) const {
			const auto& __attrib = vertex_attributes[k];
			assert( __attrib.type.hash_code == typeid(float).hash_code() );
			assert( __attrib.type.normalied == false );
			static_assert( std::is_same_v<typename attribute_type::package_type, __m128> );
			return { _mm_loadu_ps((const float*)std::next(__attrib.stream.data, __attrib.stream.stride * i)) };
		}

		void set_attribute(size_t k, size_t i, const attribute_type& val) {
			const auto& __attrib = vertex_attributes[k];
			assert( __attrib.type.hash_code == typeid(float).hash_code() );
			assert( __attrib.type.normalied == false );
			static_assert( std::is_same_v<typename attribute_type::package_type, __m128> );
			auto* ptr = reinterpret_cast<float*>( std::next(__attrib.stream.data, __attrib.stream.stride * i) );
			for (size_t i = 0; i != __attrib.type.size; ++i) {
				ptr[i] = val[i];
			}
		}

		template<typename type, typename type2> static type& as(type2& x) { 
			static_assert(sizeof(type) <= sizeof(type2)); return reinterpret_cast<type&>(x); }

		template<typename type, typename type2> static const type& as(const type2& x) { 
			static_assert(sizeof(type) <= sizeof(type2)); return reinterpret_cast<const type&>(x); }

		/// classify.

		size_t normals_first{0};
		size_t texcoords_first{0};
		size_t others_first{0};

		size_t num_normals() const { return texcoords_first - normals_first; }
		size_t num_texcoords() const { return others_first - texcoords_first; }
		size_t num_others() const { return vertex_attributes.size() - others_first; }
		attribute_type get_position(size_t i) const { return get_attribute(0, i); }
		attribute_type get_normal(size_t k, size_t i) const { return get_attribute(normals_first + k, i); }
		attribute_type get_texcoord(size_t k, size_t i) const { return get_attribute(texcoords_first + k, i); }
		attribute_type get_other(size_t k, size_t i) const { return get_attribute(others_first + k, i); }

		void set_vertex_attributes(const std::vector<typename attribute_binding::type_info>& types, size_t num_normals, size_t num_texcoords, size_t num_others) {
			assert( types == 1 + num_normals + num_texcoords + num_others );
			for (size_t i = 0; i != types.size(); ++i) {
				vertex_attributes[i].type = types[i];
			}

			normals_first = 1;
			texcoords_first = normals_first + num_normals;
			others_first = texcoords_first + num_texcoords;
		}

		void set_position_attribute(const typename attribute_binding::type_info& type) {
			if (!vertex_attributes.empty()) {
				vertex_attributes[0].type = type;
			} else {
				vertex_attributes.push_back({type, nullptr, 0});
				normals_first = 1;
				texcoords_first = 1;
				others_first = 1;
			}
		}

		void set_normal_attribute(size_t k, const typename attribute_binding::type_info& type) {
			if (k < num_normals()) {
				vertex_attributes[normals_first + k].type = type;
			} else {
				assert(k == num_normals());
				vertex_attributes.insert(std::next(vertex_attributes.begin(), texcoords_first), {type, nullptr, 0});
				++texcoords_first;
				++others_first;
			}
		}

		void set_texcoord_attribute(size_t k, const typename attribute_binding::type_info& type) {
			if (k < num_texcoords()) {
				vertex_attributes[texcoords_first + k].type = type;
			} else {
				assert(k == num_texcoords());
				vertex_attributes.insert(std::next(vertex_attributes.begin(), others_first), {type, nullptr, 0});
				++others_first;
			}
		}

		void set_other_attribute(size_t k, const typename attribute_binding::type_info& type) {
			if (k < num_others()) {
				vertex_attributes[others_first + k].type = type;
			} else {
				assert(k == num_others());
				vertex_attributes.push_back({type, nullptr, 0});
			}
		}
		
		void bind_position_stream(std::shared_ptr<std::vector<unsigned char>> buffer, size_t offset, size_t stride) {
			bind_vertex_stream(0, buffer, offset, stride);
		}

		void bind_normal_stream(size_t k, std::shared_ptr<std::vector<unsigned char>> buffer, size_t offset, size_t stride) {
			bind_vertex_stream(normals_first + k, buffer, offset, stride);
		}

		void bind_texcoord_stream(size_t k, std::shared_ptr<std::vector<unsigned char>> buffer, size_t offset, size_t stride) {
			bind_vertex_stream(texcoords_first + k, buffer, offset, stride);
		}

		void bind_other_stream(size_t k, std::shared_ptr<std::vector<unsigned char>> buffer, size_t offset, size_t stride) {
			bind_vertex_stream(others_first + k, buffer, offset, stride);
		}

		template<typename position_type>
		void set_position_attribute() {
			using scalar_type = std::remove_cvref_t<decltype(std::declval<position_type>()[size_t()])>;
			set_position_attribute({typeid(scalar_type).hash_code(), false, sizeof(position_type)/sizeof(scalar_type), alignof(position_type), sizeof(position_type)});
		}

		template<typename normal_type>
		void set_normal_attribute(size_t k) {
			using scalar_type = std::remove_cvref_t<decltype(std::declval<normal_type>()[size_t()])>;
			set_normal_attribute(k, {typeid(scalar_type).hash_code(), false, sizeof(normal_type)/sizeof(scalar_type), alignof(normal_type), sizeof(normal_type)});
		}

		template<typename texcoord_type>
		void set_texcoord_attribute(size_t k) {
			using scalar_type = std::remove_cvref_t<decltype(std::declval<texcoord_type>()[size_t()])>;
			set_texcoord_attribute(k, {typeid(scalar_type).hash_code(), false, sizeof(texcoord_type)/sizeof(scalar_type), alignof(texcoord_type), sizeof(texcoord_type)});
		}

		template<typename other_type>
		void set_other_attribute(size_t k) {
			using scalar_type = std::remove_cvref_t<decltype(std::declval<other_type>()[size_t()])>;
			set_texcoord_attribute(k, {typeid(scalar_type).hash_code(), false, sizeof(other_type)/sizeof(scalar_type), alignof(other_type), sizeof(other_type)});
		}

		::geometry::adjacency_list<::geometry::tree_constraits, 
			std::pair<bounds<vector3>, std::span<element_index_type>>> bvh;

		void build_bvh() {
			bvh = ::geometry::make_boundary_volume_hierarchy<decltype(bvh)>(elements.begin(), elements.end(),
				[this](auto first, auto last) {
					attribute_type p0 = get_position((*first)[0]), p1 = get_position((*first)[1]), p2 = get_position((*first)[2]);
					auto boundary = bounds<vector3>::from(as<vector3>(p0), as<vector3>(p1), as<vector3>(p2));
					for (auto seek = std::next(first); seek != last; ++seek) {
						attribute_type p0 = get_position((*seek)[0]), p1 = get_position((*seek)[1]), p2 = get_position((*seek)[2]);
						boundary = expand(boundary, bounds<vector3>::from(as<vector3>(p0), as<vector3>(p1), as<vector3>(p2)));
					}

					for (size_t i = 0; i != 3; ++i) {
						if (boundary.l[i] == boundary.u[i]) {
							boundary.l[i] -= 1e-6f;
							boundary.u[i] += 1e-6f;
						}
					}

					return boundary;
				},
				[this](auto first, auto last, const bounds<vector3>& boundary) {
					vector3 sides = boundary.size();
					size_t max_side = (sides[0] > sides[1] && sides[0] > sides[2]) ? 0 : (sides[1] > sides[0] && sides[1] > sides[2]) ? 1 : 2;
					auto mid = std::partition(first, last,
						[&,this](const auto& i) {
							attribute_type p0 = get_position(i[0]), p1 = get_position(i[1]), p2 = get_position(i[2]);
							return center(bounds<vector3>::from(as<vector3>(p0), as<vector3>(p1), as<vector3>(p2)))[max_side] < center(boundary)[max_side];
						}
					);

					return (mid == first || mid == last) ? std::next(mid, std::distance(first,last)/2) : mid;
				}
			);
		}

		virtual bool enter( const ray_type& ray, const arguments_type&, intersect_result_type& result, bool is_self ) const override {
			typename decltype(bvh)::vertex_descriptor node = 0;
			while (node != bvh.null_vertex()) {
				bool can_skip = before(result.distance, intersection(bvh[node].first, ray));
				if (!can_skip && bvh.vertex(node).is_leaf()) {
					for (const auto& indices : bvh[node].second) {
						if (is_self && result.prev_primitive == this && result.prev_element == (&indices) - (&elements[0])) {
							//avoid self-intersection for doubleSided triangles.
							//(and another process in material when ray parallel the triangle.)
							continue;
						}
						attribute_type p0 = get_position(indices[0]), p1 = get_position(indices[1]), p2 = get_position(indices[2]);
						auto element_i = triangle<vector3>::from_points(as<vector3>(p0), as<vector3>(p1), as<vector3>(p2));
						if (this->material->doublesided || dot(ray.direction(), cross(element_i.ori.e[0], element_i.ori.e[1])) < 0) {
							scalar t = intersection(element_i, ray);
							if (0 <= t && t < result.distance) {
								result.primitive = this;
								result.element   = /*std::distance*/(&indices) - (&elements[0]);
								result.distance  = t;
							}
						}
					}
				}
				node = can_skip ? bvh.vertex(node).skip : bvh.vertex(node).next;

				//if (bvh.vertex(node).is_leaf()) {

				//	if (bvh[node].second.size() == 1) {
				//		const auto& indices = bvh[node].second[0];
				//		attribute_type p0 = get_position(indices[0]), p1 = get_position(indices[1]), p2 = get_position(indices[2]);
				//		auto element_i = triangle<vector3>::from_points(as<vector3>(p0), as<vector3>(p1), as<vector3>(p2));
				//		if (dot(ray.direction(), cross(element_i.ori.e[0], element_i.ori.e[1])) < 0) {
				//			scalar t = intersection(element_i, ray);
				//			if (0 <= t && t < result.distance) {
				//				result.primitive = this;
				//				result.element   = /*std::distance*/(&indices) - (&elements[0]);
				//				result.distance  = t;
				//			}
				//		}
				//	} else if (!before(result.distance, intersection(bvh[node].first, ray))) {
				//		for (const auto& indices : bvh[node].second) {
				//			attribute_type p0 = get_position(indices[0]), p1 = get_position(indices[1]), p2 = get_position(indices[2]);
				//			auto element_i = triangle<vector3>::from_points(as<vector3>(p0), as<vector3>(p1), as<vector3>(p2));
				//			if (dot(ray.direction(), cross(element_i.ori.e[0], element_i.ori.e[1])) < 0) {
				//				scalar t = intersection(element_i, ray);
				//				if (0 <= t && t < result.distance) {
				//					result.primitive = this;
				//					result.element   = /*std::distance*/(&indices) - (&elements[0]);
				//					result.distance  = t;
				//				}
				//			}
				//		}
				//	}

				//	node = bvh.vertex(node).next;
				//} else {
				//	auto section = intersection(bvh[node].first, ray);
				//	bool can_skip = (empty(section) || result.distance < section.begin());
				//	node = (can_skip ? bvh.vertex(node).skip : bvh.vertex(node).next);
				//}
			}

			return result.primitive == this;
		}

		virtual void exit( const ray_type& ray, const arguments_type&, intersect_result_type& result ) const override {
			typename decltype(bvh)::vertex_descriptor node = 0;
			while (node != bvh.null_vertex()) {
				bool can_skip = before(result.distance, intersection(bvh[node].first, ray));
				if (!can_skip && bvh.vertex(node).is_leaf()) {
					for (const auto& indices : bvh[node].second) {
						attribute_type p0 = get_position(indices[0]), p1 = get_position(indices[1]), p2 = get_position(indices[2]);
						auto element_i = triangle<vector3>::from_points(as<vector3>(p0), as<vector3>(p1), as<vector3>(p2));
						if (/*this->material->doublesided  ||*/ dot(ray.direction(), cross(element_i.ori.e[0], element_i.ori.e[1])) > 0) {
							scalar t = intersection(element_i, ray);
							if (0 <= t && t < result.distance) {
								result.primitive = this;
								result.element   = /*std::distance*/(&indices) - (&elements[0]);
								result.distance  = t;
							} else if (result.primitive != this && t < result.distance) {
								result.primitive = this;
								result.element   = /*std::distance*/(&indices) - (&elements[0]);
								result.distance  = max(t, scalar(0));
							}
						}
					}
				}
				node = can_skip ? bvh.vertex(node).skip : bvh.vertex(node).next;
			}

			if (result.primitive != this) {
				throw std::exception("at exit(){ Unkown Exception }");
			}
		}

		virtual void get_attributes( const intersect_result_type& result, const size_t normals_count, const size_t texcoords_count, const size_t others_count, 
			attribute_type attributes[]) const override {
			
			auto& position = reinterpret_cast<vector3&>(attributes[0]);
			position = result.ray(result.distance);

			const auto& element_i = elements[result.element];
			attribute_type 
				p0 = get_position(element_i[0]),
				p1 = get_position(element_i[1]),
				p2 = get_position(element_i[2]);
			auto u = inside_relation(triangle<vector3>::from_points(as<vector3>(p0), as<vector3>(p1), as<vector3>(p2)), position);
			
			position = result.transformation.transform(position);

			if (normals_count != 0) {
				auto& normal = reinterpret_cast<vector3&>(attributes[1]);
				if (num_normals() != 0) {
					attribute_type 
						n0 = get_normal(0, element_i[0]),
						n1 = get_normal(0, element_i[1]),
						n2 = get_normal(0, element_i[2]);
					//normal = normalize(as<vector3>(n0)+as<vector3>(n1)+as<vector3>(n2));
					normal = normalize(as<vector3>(n0)*u[0] + as<vector3>(n1)*u[1] + as<vector3>(n2)*u[2]);
				} else {
					normal = normalize(cross(as<vector3>(p1) - as<vector3>(p0), as<vector3>(p2) - as<vector3>(p0)));
				}
				normal = result.transformation.transform_for_normal(normal);
			
				if (normals_count != 1) {
					auto* tangent = reinterpret_cast<vector3*>(attributes + 2);
					auto space = math::geometry::oriented_orthogonal_basis<vector3>::from_normal((normal));
					tangent[0] = (space.tangent0());
					tangent[1] = (space.tangent1());
				}
			}

			auto* texcoords = reinterpret_cast<attribute_type*>(attributes + 4);
			for (size_t i = 0; i != texcoords_count; ++i) {
				attribute_type
					t0 = get_texcoord(i, element_i[0]),
					t1 = get_texcoord(i, element_i[1]),
					t2 = get_texcoord(i, element_i[2]);
				texcoords[i] = as<attribute_type>(t0)*u[0] + as<attribute_type>(t1)*u[1] + as<attribute_type>(t2)*u[2];
			}

			/*auto* others = reinterpret_cast<attributes*>(attributes + 4 + texcoords_count);
			for (size_t i = 0; i != texcoords_count; ++i) {
				attribute_type
					h0 = get_other(i, element_i[0]),
					h1 = get_other(i, element_i[1]),
					h2 = get_other(i, element_i[2]);
				others[i] = as<attribute_type>(h0)*u[0] + as<attribute_type>(h1)*u[1] + as<attribute_type>(h2)*u[2];
			}*/
		}

		virtual bool intersects( const ray_type& ray, const range<scalar>& therange ) const override {
			typename decltype(bvh)::vertex_descriptor node = 0;
			while (node != bvh.null_vertex()) {
				bool can_skip = disjoint(therange, intersection(bvh[node].first, ray));
				if (!can_skip && bvh.vertex(node).is_leaf()) {
					for (const auto& indices : bvh[node].second) {
						attribute_type p0 = get_position(indices[0]), p1 = get_position(indices[1]), p2 = get_position(indices[2]);
						auto element_i = triangle<vector3>::from_points(as<vector3>(p0), as<vector3>(p1), as<vector3>(p2));
						if (contains(therange, intersection(element_i, ray))) {
							return true;
						}
					}
				}
				node = can_skip ? bvh.vertex(node).skip : bvh.vertex(node).next;
			}

			return false;
		}

		virtual spectrum get_transmittance( const ray_type& ray, const range<scalar>& therange ) const override {
			//return ones<spectrum>(1);
			if (this->material->opaque) {
				return intersects(ray, therange) ? ones<spectrum>(0) : ones<spectrum>(1);
			}
			else {
				intersect_result_type result;
				result.distance  = std::numeric_limits<scalar>::max();
				result.primitive = nullptr;
				typename decltype(bvh)::vertex_descriptor node = 0;
				while (node != bvh.null_vertex()) {
					bool can_skip = disjoint(therange, intersection(bvh[node].first, ray));
					if (!can_skip && bvh.vertex(node).is_leaf()) {
						for (const auto& indices : bvh[node].second) {
							attribute_type p0 = get_position(indices[0]), p1 = get_position(indices[1]), p2 = get_position(indices[2]);
							auto element_i = triangle<vector3>::from_points(as<vector3>(p0), as<vector3>(p1), as<vector3>(p2));
							scalar t = intersection(element_i, ray);
							if (0 <= t && t < result.distance) {
								result.primitive = this;
								result.element   = /*std::distance*/(&indices) - (&elements[0]);
								result.distance  = t;
							}
						}
					}
					node = can_skip ? bvh.vertex(node).skip : bvh.vertex(node).next;
				}

				if (result.primitive == nullptr) {
					return math::ones<spectrum>(1);
				}

				result.ray = ray;
				return this->material->get_transmittance(ray, therange, result);
			}
		}

		virtual bounds<vector3> boundary() const { assert(!bvh.empty()); return bvh[0].first; }
	};


	template<typename primitive_type>
	struct scene {
		using vector3 = typename primitive_type::vector3_type;

		std::vector< std::vector<std::shared_ptr<primitive_type>> > geometries;

		enum class types { camera, light, geometry, skinned_geometry, node };
		struct node_type {
			types type;
			size_t index;
			flexibility<vector3> geom2local;
			flexibility<vector3> geom2world;
			std::string name;
		};
		std::vector<node_type> nodes;
		std::vector< std::vector<size_t> >      node_instances;
		std::vector< size_t >                   geometry_instances;
		std::vector< std::pair<size_t,size_t> > skinned_geometry_instances;
		std::vector< std::vector<flexibility<vector3>> > skin_instances;

		struct primitive_descriptor {
			size_t node;
			size_t primitive;
		};
		std::vector<primitive_descriptor> primitives;
		::geometry::adjacency_list<::geometry::tree_constraits, 
			std::pair<bounds<vector3>, std::span<primitive_descriptor>>> bvh;

		inline ::geometry::range<vector3> rotate(const bounds<vector3>& a, const flexibility<vector3>& T) {
			auto p0 = T.transform(vector3{a.l[0], a.l[1], a.l[2]});
			auto p1 = T.transform(vector3{a.u[0], a.l[1], a.l[2]});
			auto p2 = T.transform(vector3{a.l[0], a.u[1], a.l[2]});
			auto p3 = T.transform(vector3{a.u[0], a.u[1], a.l[2]});
			auto p4 = T.transform(vector3{a.l[0], a.l[1], a.u[2]});
			auto p5 = T.transform(vector3{a.u[0], a.l[1], a.u[2]});
			auto p6 = T.transform(vector3{a.l[0], a.u[1], a.u[2]});
			auto p7 = T.transform(vector3{a.u[0], a.u[1], a.u[2]});
			return { min(min(min(min(min(min(min(p0,p1),p2),p3),p4),p5),p6),p7), max(max(max(max(max(max(max(p0,p1),p2),p3),p4),p5),p6),p7) };
			/*if (isinf(a.l[0]) || isinf(a.l[1]) || isinf(a.l[2])) {
				return a;
			}
			vector3 minp = T.position(), maxp = T.position();
			for (size_t i = 0; i != 3; ++i) {
				for (size_t j = 0; j != 3; ++j) {
					auto e = T.without_translation.sr[j][i] * a.l[j];
					auto f = T.without_translation.sr[j][i] * a.u[j];
					if (e < f) {
						minp[i] += e;
						maxp[i] += f;
					} else {
						minp[i] += f;
						maxp[i] += e;
					}
				}
			}

			return {minp, maxp};*/
		}

		void build_bvh() {
			primitives.clear();
			for (size_t i = 0; i != nodes.size(); ++i) {
				if (nodes[i].type == types::geometry) {
					for (size_t j = 0; j != geometries[geometry_instances[nodes[i].index]].size(); ++j) {
						primitives.push_back({i,j});
					}
				}
			}

			bvh = ::geometry::make_boundary_volume_hierarchy<decltype(bvh)>(primitives.begin(), primitives.end(),
				[this](auto first, auto last) {
					bounds<vector3> boundary = rotate(geometries[geometry_instances[nodes[first->node].index]][first->primitive]->boundary(), nodes[first->node].geom2world);
					for (auto seek = std::next(first); seek != last; ++seek) {
						boundary = expand(boundary, rotate(geometries[geometry_instances[nodes[seek->node].index]][seek->primitive]->boundary(), nodes[seek->node].geom2world));
					}

					return boundary;
				},
				[this](auto first, auto last, const bounds<vector3>& boundary) {
					vector3 sides = boundary.size();
					size_t max_side = (sides[0] > sides[1] && sides[0] > sides[2]) ? 0 : (sides[1] > sides[0] && sides[1] > sides[2]) ? 1 : 2;
					auto mid = std::partition(first, last,
						[&,this](const auto& i) {
							return center(rotate(geometries[geometry_instances[nodes[i.node].index]][i.primitive]->boundary(), nodes[i.node].geom2world))[max_side] < center(boundary)[max_side];
						}
					);

					return (mid == first || mid == last) ? std::next(mid, std::distance(first,last)/2) : mid;
				}
			);
		}
	};

	template<typename traits>
	struct dielectric : public material<traits> {
		using ray_type              = typename traits::ray_type;
		using arguments_type        = typename traits::arguments_type;
		using intersect_result_type = typename traits::intersect_result_type;
		using interact_result_type  = typename traits::interact_result_type;
		using attribute_type        = typename traits::attribute_type;
		using spectrum = typename traits::spectrum_type;
		using vector3 = typename traits::vector3_type;
		using scalar  = typename traits::scalar_type;

		//scalar etaI;
		scalar eta;
		spectrum multiplier;

		scalar roughness;

		virtual void interact_boundary( ray_type& ray, const arguments_type& arguments, const intersect_result_type& intersection, interact_result_type& result ) const override {
			abort();
#if 0
			attribute_type attributes[4];
			intersection.primitive->get_attributes(intersection, 3, 0, 0, attributes);
			const vector3& position = reinterpret_cast<const vector3&>(attributes[0]);
			const auto&    basis    = reinterpret_cast</*const */oriented_orthogonal_basis<vector3>&>(attributes[1]);
			const vector3 V = -ray.direction();
			bool entering = true;
			if (dot(V,basis.normal()) < 0) { basis.flip(); entering = false; }
			const scalar etaI = entering ? result.enter_ior() : result.exit_ior();
			const vector3 N = basis.normal();

			if (result.incident_radiance != 0) {
				scalar cos_N_V = dot(N, V);
				if (entering && cos_N_V > 0) {
					vector3 L  = -result.incident_ray.direction();
					vector3 Nm = normalize(L + V);
					scalar cos_Nm_V = dot(Nm, V);
					scalar cos_N_L  = dot(N, L);
					scalar cos_Nm_L = dot(Nm, L);
					scalar 
						f  = reflectance(max(cos_Nm_V,scalar(0)), etaI, eta);
						f *= ggx_g(cos_N_L, cos_Nm_L, cos_N_V, cos_Nm_V, roughness);
						f *= ggx_d(dot(N, Nm), roughness);
						f /= (4 * cos_N_V);
					result.radiance += result.incident_radiance * result.transmittance * min(f,scalar(1));
				} else if (!entering && cos_N_V != 0) {
					vector3 L = -result.incident_ray.direction();
					vector3 Nm = -normalize(-eta*V - etaI*L);

					scalar Lterm = abs(dot(V,Nm)*dot(L,Nm) / (dot(V,N)/**dot(L,N)*/));
					scalar F = reflectance(max(dot(V,Nm), scalar(0.00001f)), eta, etaI);
					scalar G = math::physics::ggx_g(dot(N,V),dot(Nm,V),dot(N,L),dot(Nm,L),roughness);
					scalar D = math::physics::ggx_d(dot(N,Nm),roughness);
	
					scalar Rterm = pow(etaI,2.0f)/max(pow(eta*dot(V,Nm) + etaI*dot(L,Nm),2.0f), std::numeric_limits<scalar>::epsilon());
					/*if (isnan(Lterm*Rterm*(1-F)*G*D)){
						std::cout << "debug" << std::endl;
					}*/
					result.radiance += result.incident_radiance * result.transmittance * min(Lterm*Rterm*(1-F)*G*D,scalar(1));
				}
			}

			using microfacet = ggx_visible_normal<vector3>;
			vector3 Nm = microfacet::generate(arguments.random, basis, V, vector3{roughness,roughness,1});
			scalar costhetaI = abs(dot(V, Nm));
			//bool   entering   = costhetaI < 0;
			scalar reflectance = entering ? (::reflectance(/*-*/costhetaI, etaI, eta)) : (::reflectance(costhetaI, eta, etaI));
#if 0
			if (arguments.random[0] < reflectance) {
				ray.set_start_point(position);
				ray.set_direction(math::geometry::reflect(ray.direction(), normal));
				result.transmittance *= multiplier;
				/// f = reflectance * multiplier / cos(N,L)
				/// PDF = reflectance
			} else {
				ray.set_start_point(position);
				ray.set_direction(entering ? math::geometry::refract(ray.direction(), normal, etaI/eta) : math::geometry::refract(ray.direction(), normal, eta/etaI));
				result.transmittance *= multiplier;
				/// f = (1 - reflectance) * multiplier / cos(N,L)
				/// PDF = 1 - reflectance
				if (entering) { result.push(intersection.primitive, eta); }
				else { if (result.empty() || result.top() != intersection.primitive) { throw std::exception("dielectric::exit(...){ may be normal error. }"); } result.pop(); }
			}
#else
			if (arguments.random[0] < reflectance) {
				vector3 L  = reflect(ray.direction(), Nm);
				scalar cos_Nm_V = dot(Nm, V);
				scalar cos_N_L  = dot(N, L);
				scalar cos_Nm_L = dot(Nm, L);
				result.transmittance *= microfacet::masking_shadowing(  specular_distribution.masking(cos_N_L,cos_Nm_L);
				ray.set_start_point(position);
				ray.set_direction(L);
				/// f = reflectance * multiplier
				/// PDF = reflectance
			} else {
				vector3 L  = entering ? math::geometry::refract(ray.direction(), Nm, etaI/eta) : math::geometry::refract(ray.direction(), Nm, eta/etaI);
				scalar cos_Nm_V = dot(Nm, V);
				scalar cos_N_L  = dot(N, L);
				scalar cos_Nm_L = dot(Nm, L);
				//std::cout << normal << V << Nm << "\t" << (entering?"1":"0") << "\t" << specular_distribution.masking(cos_N_L, cos_Nm_L) << std::endl;
				result.transmittance *= specular_distribution.masking(cos_N_L,cos_Nm_L);
				ray.set_start_point(position);
				ray.set_direction(L);
				/// f = (1 - reflectance) * multiplier
				/// PDF = 1 - reflectance
				if (entering) { result.push(intersection.primitive, intersection.object, eta); }
				else { if (result.empty() || result.top() != intersection.primitive) { throw std::exception("dielectric::exit(...){ may be normal error. }"); } result.pop(); }
			}
#endif
#endif
		}
	};

	template<typename traits>
	struct conductor : material<traits>, math::physics::conductor<vector3,spectrum> {
		using ray_type              = typename traits::ray_type;
		using arguments_type        = typename traits::arguments_type;
		using intersect_result_type = typename traits::intersect_result_type;
		using interact_result_type  = typename traits::interact_result_type;
		using attribute_type        = typename traits::attribute_type;
		using spectrum = typename traits::spectrum_type;
		using vector3 = typename traits::vector3_type;
		using scalar  = typename traits::scalar_type;

		virtual void interact_boundary( ray_type& ray, const arguments_type& arguments, const intersect_result_type& intersection, interact_result_type& result ) const override {
			attribute_type attributes[4];
			intersection.primitive->get_attributes(intersection, 3, 0, 0, attributes);
			const vector3& position = reinterpret_cast<const vector3&>(attributes[0]);
			const auto&    basis    = reinterpret_cast<const oriented_orthogonal_basis<vector3>&>(attributes[1]);
			const spectrum etaI     = {result.enter_ior(),result.enter_ior(),result.enter_ior()};
			
			vector3 V = (-ray.direction());
			vector3 L;
			if (result.incident_radiance != 0) 
				result.radiance += math::physics::transmission(V, -result.incident_ray.direction(), basis, etaI, *this) * result.incident_radiance * result.transmittance;
			result.transmittance *= math::physics::transport(arguments.random, V, /*std::ref*/(L), basis, etaI, *this);
			if (result.transmittance != 0) {
				ray.set_start_point(position);
				ray.set_direction(L);
			}
		}
	};

	template<typename traits>
	struct principle_brdf : material<traits> {
		using ray_type              = typename traits::ray_type;
		using arguments_type        = typename traits::arguments_type;
		using intersect_result_type = typename traits::intersect_result_type;
		using interact_result_type  = typename traits::interact_result_type;
		using attribute_type        = typename traits::attribute_type;
		using spectrum = typename traits::spectrum_type;
		using vector3 = typename traits::vector3_type;
		using scalar  = typename traits::scalar_type;

		math::samplable_array<
			math::mdarray<vector4,mdsize_t<2>>, vector2> color;
		vector4 color_factor;
		size_t color_texture_index = 0;
		
		math::samplable_array<
			math::mdarray<vector2,mdsize_t<2>>, vector2> metallic_and_roughness;
		scalar metallic_factor;
		vector3 roughness_factor;
		size_t metallic_and_roughness_texture_index = 0;

		math::samplable_array<
			math::mdarray<spectrum,mdsize_t<2>>, vector2> emissive_texture;
		spectrum emissive_factor;
		size_t emissive_texture_index = 0;

		bool translucent = false;
		translucent_bsdf<vector3, spectrum> translucent_mtl = { spectrum{0.277955f,0.064603f,0.104813f} };

		virtual void interact_boundary( ray_type& ray, const arguments_type& arguments, const intersect_result_type& intersection, interact_result_type& result ) const override {
			attribute_type attributes[4+4];
			intersection.primitive->get_attributes(intersection, 3, color.empty()&&metallic_and_roughness.empty() ? 0 : (max(color_texture_index, metallic_and_roughness_texture_index) + 1), 0, attributes);
			const vector3& position = reinterpret_cast<const vector3&>(attributes[0]);
			/*const*/auto& basis    = reinterpret_cast</*const */oriented_orthogonal_basis<vector3>&>(attributes[1]);
			const spectrum etaI     = {result.enter_ior(),result.enter_ior(),result.enter_ior()};
			if (this->doublesided && dot(basis.normal(), -ray.direction()) < 0) {
				basis.flip();
			}
			
			math::physics::principle_brdf<vector3,spectrum> mtl;
			if (!color.empty()) {
				const auto& x = reinterpret_cast<const vector2&>(attributes[4 + color_texture_index]);
				auto c = math::tex_sample(color, x/*vector2{x[0],1-x[1]}*/) * color_factor;
				if (!this->opaque && arguments.random[0] >= c[3]) {
					ray.set_start_point(position);
					return;
				}
				mtl.color = reinterpret_cast<const spectrum&>(c);
			} else {
				if (!this->opaque && arguments.random[0] >= color_factor[3]) {
					ray.set_start_point(position);
					return;
				}
				mtl.color = reinterpret_cast<const spectrum&>(color_factor);
			}

			if (!metallic_and_roughness.empty()) {
				const auto& x = reinterpret_cast<const vector2&>(attributes[4 + metallic_and_roughness_texture_index]);
				const auto metrou = math::tex_sample(metallic_and_roughness, x/*vector2{x[0],1-x[1]}*/);
				mtl.metallic = metrou[0] * metallic_factor;
				mtl.roughness = vector3{metrou[1],metrou[1], 1} *roughness_factor;
			} else {
				mtl.metallic = metallic_factor;
				mtl.roughness = roughness_factor;
			}

			spectrum emissive;
			if (!emissive_texture.empty()) {
				const auto& x = reinterpret_cast<const vector2&>(attributes[4 + emissive_texture_index]);
				emissive = math::tex_sample(emissive_texture, x) * emissive_factor;
			} else {
				emissive = emissive_factor;
			}

			if (translucent) {
				vector3 V = (-ray.direction());
				vector3 L;
				if (result.incident_radiance != 0) 
					result.radiance += (math::physics::transmission(V, -result.incident_ray.direction(), basis, etaI, mtl)
						+ math::physics::transmission(V, -result.incident_ray.direction(), basis, etaI, translucent_mtl)) * result.incident_radiance * result.transmittance;
				result.radiance += emissive * result.transmittance;
				if (arguments.random[1] < 0.5) {
					result.transmittance *= min(scalar(1), math::physics::transport(arguments.random, V, /*std::ref*/(L), basis, etaI, mtl) * 2);
				} else {
					result.transmittance *= min(scalar(1), math::physics::transport(arguments.random, V, /*std::ref*/(L), basis, etaI, translucent_mtl) * 2);
				}
				if (result.transmittance != 0) {
					ray.set_direction(L);
					ray.set_start_point(position);
				}
			} else {
				vector3 V = (-ray.direction());
				vector3 L;
				if (result.incident_radiance != 0) 
					result.radiance += min( math::physics::transmission(V, -result.incident_ray.direction(), basis, etaI, mtl), scalar(1)) * result.incident_radiance * result.transmittance;
				result.radiance += emissive * result.transmittance;
				result.transmittance *= min( math::physics::transport(arguments.random, V, /*std::ref*/(L), basis, etaI, mtl), scalar(1) );
				if (result.transmittance != 0) {
					ray.set_direction(L);
					ray.set_start_point(position);
				}
			}
		}
	
		virtual spectrum get_transmittance(const ray_type& ray, const range<scalar>& therange, const intersect_result_type& intersection) const { 
			attribute_type attributes[2+4];
			intersection.primitive->get_attributes(intersection, 0, color.empty()&&metallic_and_roughness.empty() ? 0 : (max(color_texture_index, metallic_and_roughness_texture_index) + 1), 0, attributes);
			//const vector3& position = reinterpret_cast<const vector3&>(attributes[0]);
			//const spectrum etaI     = {result.enter_ior(),result.enter_ior(),result.enter_ior()};
			
			if (!color.empty()) {
				const auto& x = reinterpret_cast<const vector2&>(attributes[4 + color_texture_index]);
				auto c = math::tex_sample(color, x) * color_factor;
				return math::ones<spectrum>(1 - c[3]);
			} else {
				return math::ones<spectrum>(1 - color_factor[3]);
			}
		}
	};

	/*if (isnan(result.radiance[0]) || isnan(result.radiance[1]) || isnan(result.radiance[2])) {
					std::cout << "transmision(" << std::endl;
					std::cout << "auto V = " << V << std::endl;
					std::cout << "auto L = " << -result.incident_ray.direction() << std::endl;
					std::cout << "auto basis = {" << basis.normal() << "," << basis.tangent0() << "," << basis.tangent1() << "}" << std::endl;
					std::cout << "auto mtl = {" << this->eta << "," << this->etak << "," << this->roughness << "}" << std::endl;
					std::cin.get();
				}*/

	template<typename traits>
	struct empty_volume : public material<traits> {
		using ray_type              = typename traits::ray_type;
		using arguments_type        = typename traits::arguments_type;
		using intersect_result_type = typename traits::intersect_result_type;
		using interact_result_type  = typename traits::interact_result_type;
		using attribute_type        = typename traits::attribute_type;
		using spectrum = typename traits::spectrum_type;
		using vector3 = typename traits::vector3_type;
		using scalar  = typename traits::scalar_type;

		spectrum sigma_t;
		scalar g;
		
		virtual scalar march_interior( const ray_type& ray, const arguments_type& arguments, const intersect_result_type& intersection,
			interact_result_type& result) const override {
			int channel = min((int)(arguments.random[0] * 3), int(2));
			scalar distance = -log(1 - arguments.random[1]) / sigma_t[channel];
				distance = min(distance, intersection.distance);

			// Compute the transmittance and sampling density
			spectrum Tr = exp(-sigma_t * distance);

			result.transmittance *= Tr/((Tr[0]+Tr[1]+Tr[2])/3);
			return distance;
		}

		virtual void interact_interior( ray_type& ray, const arguments_type& arguments, const intersect_result_type& intersection, const scalar distance,
			interact_result_type& result ) const override {
/// integral<distance_i = intersection.distance>(
///		transmittance(ray, distance_i) *
///		integral<oray_j = {ray(distance_i), sph}>( radiance(oray_j) * scattering(ray(distance_i)) * phase(ray,oray_j) )
/// )
///         
/// 			   transmittance(ray,distance_i)           radiance(oray_j) * scattering(ray(distance_i)) * phase(ray,oray_j)
/// = avg( --------------------------------- * avg( -------------------------------------------------------------------- ) )
///         probability_density(distance_i)                             probability_density(oray_i)
/// 
///           transmittance(ray,distance_i)      radiance(oray_j) * scattering(ray(distance_i)) * phase(ray,oray_j)
/// = avgU( --------------------------------- * -------------------------------------------------------------------- )
/// 			   probability_density(distance_i)                        probability_density(oray_j)
/// 
/// (when raymarch, probability_density(distance_i) always 1.0)
/// (when single-scattering, probability_density(oray_j) always 1.0)
/// 
			if (result.incident_radiance != 0) {
				result.radiance += result.incident_radiance * result.transmittance *
					sigma_t * math::physics::henyey_greenstein_phase(dot(ray.direction(),-result.incident_ray.direction()),g)/(12.5663706143591f);
			}

			//result.transmittance *= sigma_t;
			ray.start_point() += ray.direction() * distance;
			//ray.set_direction(henyeygreenstein_distribution<vector3>(g)(ray.direction(), arguments.random, math::geometry::coordinate_traits<>{}));
			/*scalar zenith = arguments.random[0] * 3.141592653589793f;
			scalar azimuth = arguments.random[1] * 3.141592653589793f * 2;*/
			//ray.direction() = math::geometry::sph2cart(vector3{azimuth,zenith,1}, math::geometry::sphconfig<>{});
		}

		virtual void interact_boundary( ray_type& ray, const arguments_type& arguments, const intersect_result_type& intersection, 
			interact_result_type& result ) const override {
			attribute_type attributes[2];
			const vector3& position = reinterpret_cast<const vector3&>(attributes[0]);
			const vector3& normal   = reinterpret_cast<const vector3&>(attributes[1]);
			intersection.primitive->get_attributes(intersection, 1,0,0, attributes);
			
			ray.start_point() += ray.direction() * intersection.distance;
			if (dot(ray.direction(), normal) < 0) { result.push(intersection.primitive, intersection.object, 1.0f); }
			else { if (result.empty() || result.top() != intersection.primitive) { throw std::exception("dielectric::exit(...){ may be normal error. }"); } result.pop(); }
		}

		virtual spectrum get_transmittance(const ray_type&, const range<scalar>& therange) const override {
			return exp(- sigma_t * std::size(therange));
		}
	};

	template<typename traits>
	struct point_light : public material<traits> {
		using ray_type              = typename traits::ray_type;
		using arguments_type        = typename traits::arguments_type;
		using intersect_result_type = typename traits::intersect_result_type;
		using interact_result_type  = typename traits::interact_result_type;
		using attribute_type        = typename traits::attribute_type;
		using spectrum = typename traits::spectrum_type;
		using vector3 = typename traits::vector3_type;
		using scalar  = typename traits::scalar_type;

		vector3 position;
		spectrum color;
		scalar intensity;

		virtual void interact_boundary( ray_type& ray, const arguments_type& arguments, const intersect_result_type& intersection, interact_result_type& result ) const override {
#if 1
			scalar length;
			ray.set_direction(normalize(ray.start_point() - position, &length));
#if 1
			ray.set_start_point(position);
			result.incident_distance = length;
			result.incident_radiance = color*(intensity/(length*length));
#else
			scalar radius = 0.1f;
			if (length < radius) {
				result.incident_distance = 0;
				result.incident_radiance = {0,0,0};
			}
			scalar sinthetaMax2 = pow(radius/length, scalar(2));
			scalar costhetaMax = sqrt(1 - sinthetaMax2);
			scalar costheta = (1 - arguments.random[0]) + arguments.random[0] * costhetaMax;
			scalar sintheta = sqrt(1 - costheta*costheta);
			scalar phi = arguments.random[1] * 6.283185307179586f;

			scalar temp = max(scalar(0), length*costheta - sqrt(radius*radius - pow(length*sintheta,scalar(2))));

			scalar cosalpha = (length*length + radius*radius - temp*temp)/(2*length*radius);
			scalar sinalpha = sqrt(max(1 - cosalpha*cosalpha, scalar(0)));

			scalar normal, tangent, bitangent;
			normal = costheta;
			tangent = cos(phi) * sintheta;
			bitangent = sin(phi) * sintheta;
			//vector3 N = ray.d;
			auto TBN = math::geometry::oriented_orthogonal_basis<vector3>::from_normal(ray.d);
			/*vector3 T, B; 
			math::geometry::get_tangent(N, T, B, math::geometry::coordinate_traits<>{});*/
			vector3 P = position + (TBN.tangent1()*bitangent + TBN.normal()*normal + TBN.tangent0()*tangent);
			ray.set_direction(normalize(ray.s - P, &result.incident_distance));
			ray.s = P;
			
			result.incident_radiance = color*intensity*(6.2831853f * (1 - costhetaMax));
#endif
#else
			plane<vector3> pln{ normalize(vector3{1,1,1}), 200 };
			scalar t = geometry::intersection(pln, geometry::ray<vector3>::from_ray(ray.start_point(), pln.n));
			ray.set_direction(-pln.n);
			ray.set_start_point(ray.start_point() + pln.n*t);
			result.incident_distance = t;
			result.incident_radiance = color*(intensity);
#endif
		/*	rayIO.d = -normalize(vector3{1,1,1});
			rayIO.s += (-rayIO.d * 999);
			result.incident_distance = 999;
			result.incident_radiance = {1,1,1};*/

			//result.incident_pdf      = 1;
		}
	};
}

} }//end of namespace math::physics