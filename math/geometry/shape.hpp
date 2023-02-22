#pragma once

///@brief Geometry Shape
///@license Free 
///@review 2022-7-20 
///@contact Jiang1998Nan@outlook.com 
#define _MATH_GEOMETRY_SHAPE_

#include <math/concepts.hpp>
#include <cassert>
#include <array>
#ifndef _MATH_CONCEPTS_
namespace math { 
	template<typename _Ty>
	using value_t = typename std::iterator_traits<decltype(std::begin(std::declval<_Ty>()))>::value_type; }
#endif

#include <geometry/range.hpp>

namespace math { namespace geometry {
/// Desirable Bounding Volume Characteristics
///		Inexpensive intersection tests
///		Tight fitting
///		Inexpensive to compute
///		Easy to rotateand transform
///		Use little memory
///		                         <<Real-Time Collision Detection>>

///
///		struct object{
///			shape _My_geometry;
///			surf|volm _My_material;
///		}
/// 
/// We can known sometimes material working with geometry (procedural material, we don't known infinite geometry properties in finite kind of intersect interfaces),
///  so we must define interfaces at object, cannot only at geometry and material.
/// 
///		struct object{
///			virtual spectrum reflect(...) { ... }
///			virtual spectrum transmit(...) { ... }
///		}
/// 
/// We usually reused geometry shape only is std::vector<triangle>, other shape almost not reuse.
///  so we can define 
/// 
///		std::vector< std::vector<triangle> > geometries;
///		struct object1 : public object{
///			std::vector<triangle>* geometry;
///			what_material* material;
///		}
/// 
///		struct special_object1 : public object{
///			sphere geometry;
///			virtual spectrum transmit(...) { ... }
///		};
/// 
///		...
/// 
/// In other word, we not need virual shape.
///		1. reused shape only triangle mesh, other shape almost not reused.
///		2. sometimes material working with geometry, We can't deny this situation, so interfaces don't need to be defined in shapes in objects. 
///		3. infinite kind of shapes has infinite intersect results, we cannot use finite kind of intersection interfaces do well.
///		 if we only cover some easy cases, but also remains infinite cases, 
///		 so almost impossible to define interfaces for the purpose of shape, but only for other purposes. (render opaque lambert-surface only distance and normal, but like this interface should at object or lambert-shape not shape)
/// 

///
/// There is a assert is:
/// 
///		the "fullname" cannot be directly used in formulas.
///		(May be used in some simple cases, but most of the formulas are not so simple,
///		 so it cannot be done. example: 
///		   x*x*t^2 + y*y*t + z = 0, is symbols usage. 
///		   point*point*impack[0] + ... = 0, super complex!! Although directly known each meanings of
///		                         variable, but too many repetitions (char count of each variables). )
/// 
/// There are tow usages for members:
/// 
///		///(1) Fullname member usage:
///		struct ray { vector3 start_point, direction; } the_ray;
///		auto a = dot(the_ray.start_point, the_ray.direction);
/// 	/// Note: Although we direct use the fullname, but really 
///		///  we simplfied the fullnames in the formula.
/// 
///		///(2) Symbol member usage:
/// 	struct ray { vector3 o, d; } r;
///		auto a = dot(r.o, r.d);
/// 
/// Try to derive the best usage.(May be.)
/// Four usages ["object.symbol", "symbol.symbol", "object.fullname", "symbol.fullname"].
/// 
/// Because the object information has been included in the type, we can remove two uses.
/// Two usages ["symbol.symbol", "symbol.fullname"].
/// 
/// Because (1) "symbol.symbol" simplified in each structure. better than (2) "symbol.fullname"
/// simplified in each formula. 
/// Only usage "symbol.symbol".
/// 
/// Why does the "symbol.symbol" in each formula not need to be simplified?
/// (1) In multi-object formula, we 'must' need add another symbol as object for each symbol.
/// (2) In single-object formula, we can place it in the member-function then we can use 
///     only symbols.
/// 
/// 

/// linear shape.

#if 0
	template<typename vector3>
	struct raybeam : ray<vector3> {
		typedef std::remove_cvref_t<decltype(vector3{}[size_t()])> scalar;
		typedef vector3 vector_type;
		typedef scalar scalar_type;
		vector3 s;///raybeam::start_point.
		vector3 d;///raybeam::direction.
		scalar r;///raybeam::radius.

		vector3 operator()(scalar t, scalar s, scalar arg) const {
			abort();
			//cos(arg)*s, sin(arg)*s
		}
	};

	template<typename vector3>
	struct raypencil {
		typedef std::remove_cvref_t<decltype(vector3{}[size_t()]) > scalar;
		typedef vector3 vector_type;
		typedef scalar scalar_type;
		vector3 s;///raypencil::start_point.
		vector3 d;///raypencil::direction.
		scalar a;///raypencil::angle.
		
		vector3 operator()(scalar t, scalar s, scalar arg) const {
			abort();
			//cos(arg)*s, sin(arg)*s
		}
	};
#endif
	
	template<typename __vectorN>
	struct ray {
		using vector = __vectorN;
		using scalar = value_t<__vectorN>;
		vector s;
		vector d;
		vector invd;
		
		constexpr const vector& position() const { return s; }
		
		constexpr const vector& direction() const { return d; }

		constexpr const vector& inversed_direction() const { return invd; }

		constexpr vector operator()(const scalar t) const { return s + d * t; }

		constexpr ray() = default;

		constexpr ray(const vector& p, const vector& d)
			: s(p), d(d), invd(copysign(1/max(abs(d), std::numeric_limits<scalar>::min()), d)) {}

		static constexpr ray from_segment(const vector& p0, const vector& p1) {
			return ray{p0, normalize(p1 - p0)};
		}

		void reposit(const vector& p1) {
			s = p1;
		}

		void redirect(const vector& d1) {
			d = d1;
			invd = copysign(1/max(abs(d1), std::numeric_limits<scalar>::min()), d1);
		}
	};

	template<typename __vectorN>
	struct plane {
		using vector = __vectorN;
		using scalar = value_t<__vectorN>;
		///@note what difference between "dot(p-0,n) = d" and "dot(p-0,n)+d=0" ? 
		/// "dot(p-0,n) = d" means signed distance from '0' to 'p', named Plane, it aligns coordinate axis. @see transform.hpp.
		/// "dot(p-0,n)+d=0" means signed distance from 'p' to '0', named Equation. @see aeq.hpp.
		vector n;
		scalar d;
		
		constexpr const vector& normal() const { return n; }

		constexpr const vector& distance() const { return d; }

		static constexpr plane from_triangle(const vector& p, const vector& e01, const vector& e02) {
			plane pln;
			pln.n = cross(e01, e02);
			pln.d = dot(p, pln.n);
			return pln;
		}

		static constexpr plane from_points(const vector& p0, const vector& p1, const vector& p2) {
			plane pln;
			pln.n = cross(p1-p0,p2-p0);
			pln.d = dot(p0, pln.n);
			return pln;
		}
	};
	
	template<typename vectorN>
	auto intersection(const plane<vectorN>& pln, const ray<vectorN>& ray) {
///  -+------------------------+
///  /*c                      /pln
/// /        *a     *b       /
/// +-----------------------+
///      \  d| / /
///        \ |//
/// ---------O-------->
///         /
///        /
/// 
/// The distance from any point'X' on the plane'pln' to the origin'o = 0.0',
/// projected on the plane normal'pln.n' must be equal to perpendicular-distance'pln.d'.
/// 
///		dot(X - 0.0, pln.n) = pln.d 
/// 
/// Therefore, we can express intersection of ray'r.s + r.d*t' and plane as,
/// 
///		dot(r.s + r.d*t, pln.n) = pln.d 
/// 
/// Solve it by Distributive Law of Dot Product,
/// 
///		dot(r.s, pln.n) + dot(r.d, pln.n)*t = pln.d 
///		t = (pln.d - dot(r.s, pln.n))/dot(r.d, pln.n) 
/// 
#if 0
		return (pln.d - dot(ray.s, pln.n))/dot(ray.d, pln.n);///may NaN if 'ray' on the 'pln'.
#else
		auto dx = pln.d - dot(ray.s, pln.n);
		if (dx == 0) { return 0; }
		auto dxdt = dot(ray.d, pln.n);
		return dx/dxdt;
#endif
	}


	template<typename vectorN>
	struct origin_triangle {
		vectorN e[2];
	};

	template<typename __vectorN>
	struct triangle {
		using vector = __vectorN;
		using scalar = value_t<__vectorN>;
		vector p;
		origin_triangle<vector> base;

		constexpr const vector& pivot() const { return p; }

		constexpr const vector& edge(size_t i) const { return base.e[i]; }

		constexpr const vector point(size_t i) const { return i == 0 ? p : i == 1 ? (p + base.e[0]) : (p + base.e[1]); };

		template<typename vector2>
		constexpr vector operator()(const vector2& u) const { return p + base.e[0]*u[0] + base.e[1]*u[1]; }

		static triangle from_points(const vector& p0, const vector& p1, const vector& p2) {
			return {p0, p1-p0, p2-p0};
			///order not change.
			///		tri.p2 + (tri.p0 - tri.p2)*u + (tri.p1 - tri.p2)*v = X
			///		tri.p((2+1)%3) + (tri.p((0+1)%3) - tri.p((2+1)%3))*u + (tri.p((1+1)%3) - tri.p((2+1)%3))*v = X
			///		tri.p0 + (tri.p1 - tri.p0)*u + (tri.p2 - tri.p0)*v = X    
		}

		template<typename vector2>
		static vector lerp_from_points(const vector& p0, const vector& p1, const vector& p2, const vector2& u) {
			return p0 + (p1 - p0)*u[0] + (p2 - p0)*u[1];
		}
	};

	template<typename vectorN>
	auto intersection(const origin_triangle<vectorN>& tri, const vectorN raySp, const vectorN& rayDr) {
/// tri.p --- * --> tri.e0
///  \      / 
///   \    /
///    \  /
///     * 
///      \
///      _\| tri.e1
/// 
/// A triangle is sandwiched between two edges'tri.e' that have a unique intersection
/// pivot'tri.p = 0.0'.
/// 
///		tri.p0*u + tri.p1*v + tri.p2*(1 - u - v) = X
///		tri.p2 + (tri.p0 - tri.p2)*u + (tri.p1 - tri.p2)*v = X
///		0.0 + tri.e0*u + tri.e1*v = X 
/// 
/// Therefore, we can express intersection of ray'ray.s + ray.d*t' and triangle as,
/// 
///		tri.e0*u + tri.e1*v = ray.s + ray.d*t 
/// 
/// Solve it by Cramer's Rule, the first get the matrix equation,
/// 
///		tri.e0*u + tri.e1*v - ray.d*t = ray.s 
/// 
///		{ dot(tri.e0*u + tri.e1*v - ray.d*t, tri.e0) = dot(ray.s, tri.e0),
///		  dot(tri.e0*u + tri.e1*v - ray.d*t, tri.e1) = dot(ray.s, tri.e1),
///		  dot(tri.e0*u + tri.e1*v - ray.d*t, ray.d)  = dot(ray.s, ray.d)  }
/// 
///		solve ... 
/// 
/// Optimize by determinant with transpose
/// 
///		determinant(MatrixNxN) = determinant(transpose(MatrixNxN)) 
///		determinant(Matrix3x3) = dot(row(Matrix3x3,0), cross(row(Matrix3x3,1), row(Matrix3x3,2))) 
///		                                                    from <<Introduction to Linear Algebra>>
///		determinant(Matrix3x3) =
///		determinant(transpose(Matrix3x3)) = dot(column(Matrix3x3,0), cross(column(Matrix3x3,1), column(Matrix3x3,2))) 
/// 
		vectorN column0 = { dot(tri.e[0], tri.e[0]), dot(tri.e[0], tri.e[1]), dot(tri.e[0], rayDr)};
		vectorN column1 = { dot(tri.e[1], tri.e[0]), dot(tri.e[1], tri.e[1]), dot(tri.e[1], rayDr)};
		vectorN column2 = - vectorN{ dot(rayDr, tri.e[0]), dot(rayDr, tri.e[1]), dot(rayDr, rayDr) };
		auto determinant = dot(column0, cross(column1, column2));
		if ( determinant != 0 ) {
			vectorN column3 = { dot(raySp, tri.e[0]), dot(raySp, tri.e[1]), dot(raySp, rayDr) };
			auto u = dot(column3, cross(column1, column2))/determinant;
			if ( 0 <= u && u <= 1 ) {
				auto v = dot(column0, cross(column3, column2))/determinant;
				if ( 0 <= v && u + v <= 1 ) {
///		auto t0 = dot(column0, cross(column1, column3))/determinant;
///		if (t0 < tMin) {
///			return -1;
///		} else {
///			t = t0;
///			return 1;
///		}
/// 
/// These steps are all in vain, because:
/// (1) Anyone must still do comp and junp after return.
/// (2) Anyone can be get all informations{nohit, back, forward} by 't0' (because only one result).
/// 
					return dot(column0, cross(column1, column3))/determinant;
				}
			}
		}

		return std::numeric_limits<decltype(determinant)>::quiet_NaN();
	}

	template<typename vectorN>
	auto intersection(const origin_triangle<vectorN>& tri, const vectorN X/* = raySp + rayDr*t */) {
///		0.0 + tri.e[0]*u + tri.e[1]*v = X 
///
///		{ dot(tri.e[0]*u + tri.e[1]*v, tri.e[0]) = dot(X, tri.e[0]),
///		  dot(tri.e[0]*u + tri.e[1]*v, tri.e[1]) = dot(X, tri.e[1]) }
		auto m00 = dot(tri.e[0], tri.e[0]);
		auto m10 = dot(tri.e[1], tri.e[0]);
		auto m20 = dot(   X,     tri.e[0]);
		auto m01 = dot(tri.e[0], tri.e[1]);
		auto m11 = dot(tri.e[1], tri.e[1]);
		auto m21 = dot(   X,     tri.e[1]);
		auto determinant = m00*m11 - m10*m01;
		if (determinant == 0) { determinant = 1; }
		auto u0 = (m20*m11 - m10*m21)/determinant;
		auto u1 = (m00*m21 - m20*m01)/determinant;
		return std::array<value_t<vectorN>,2>{u0, u1};
	}

	template<typename vectorN>
	auto intersection(const triangle<vectorN>& tri, const ray<vectorN>& ray) {
		return intersection(tri.base, ray.s - tri.p, ray.d);
	}

	template<typename vectorN>
	auto intersection(const triangle<vectorN>& tri, const vectorN& X) {
		return intersection(tri.base, X - tri.p);
	}


	using ::geometry::range;

	template<typename vectorN>
	struct origin_box {
		vectorN r;
	};

	template<typename __vectorN>
	struct box {
		using vector = __vectorN;
		vector c;
		origin_box<vector> base;

		constexpr const vector& center() const { return c; }

		constexpr const vector& halfextents() const { return base.r; }
	};

	template<typename vectorN>
	auto intersection(const origin_box<vectorN>& box, const vectorN& raySp, const vectorN& rayDrInv) {
		using scalar = value_t<vectorN>;
		vectorN tmp = (-box.r - raySp)*rayDrInv;///@see intersect(ray, plane)
		vectorN tB  = ( box.r - raySp)*rayDrInv;///@see intersect(ray, plane)
		vectorN tF  = max(tmp, tB);///forward intersect results.
		        tB  = min(tmp, tB);///backward intersect results.
		scalar  t0  = max(max(tB[0], tB[1]), tB[2]);
		scalar  t1  = min(min(tF[0], tF[1]), tF[2]);
		if (t1 < t0) { return range<scalar>::empty_range(); }
		return range<scalar>{t0, t1};
	}

	template<typename vectorN>
	auto intersection(const box<vectorN>& box, const ray<vectorN>& ray) {
		return intersection(box.ori, ray.s - box.c, ray.invd);
	}

	template<typename vectorN>
	auto intersection(const range<vectorN>& box, const ray<vectorN> ray) {
		using scalar = value_t<vectorN>;
		vectorN tmp = (box.l - ray.s)*ray.invd;///@see intersect(ray, plane)
		vectorN tB  = (box.u - ray.s)*ray.invd;///@see intersect(ray, plane)
		vectorN tF  = max(tmp, tB);///forward intersect results.
		        tB  = min(tmp, tB);///backward intersect results.
		scalar  t0  = max(max(tB[0], tB[1]), tB[2]);
		scalar  t1  = min(min(tF[0], tF[1]), tF[2]);
		if (t1 < t0) { return range<scalar>::empty_range(); }
		return range<scalar>{t0, t1};
	}


	template<typename vectorN>
	struct origin_sphere { 
		value_t<vectorN> r;
	};

	template<typename __vectorN>
	struct sphere {
		using vector = __vectorN;
		using scalar = value_t<__vectorN>;
		vector c;
		origin_sphere<vector> base;

		constexpr const vector& center() const { return c; }

		constexpr const auto& radius() const { return base.r; }
	};

	template<typename vectorN>
	auto intersection(const origin_sphere<vectorN>& sph, const vectorN& raySp, const vectorN& rayDr) {
		using scalar = value_t<vectorN>;
		///             _ _
		///         -         - 
		///     .-               -.
		///   .                     .
		///  /                       \
		///  .                        .
		/// |            C-- -- r-- --|
		///  .                        .
		///  \                       /
		///    .                   .
		///      .               .
		///         - _ ... _ -
		/// 
		/// The distance from any point'X' on the Sphere surface, to the sphere center'sph.c = 0.0'
		/// must be equal to sphere radius'sph.r'.
		///		distance(X, 0.0) = sph.r.
		///		sqr(X - 0.0) = sqr(ph.r).
		/// 
		/// Therefore, we can express intersected point as,
		///		sqr(O_ray + D_ray*t) = sqr(r_sph).
		/// 
		/// Exapand it by Binomial Expansion of Dot Product.
		///		sqr(O_ray) + 2*dot(O_ray,D_ray*t) + sqr(D_ray*t) = sqr(r_sph).
		///		sqr(D_ray)*t*t + 2*dot(O_ray,D_ray)*t + sqr(O_ray) - sqr(r_sph) = 0.
		///		      1   *t*t +          b        *t +           c             = 0.
		/// 
		/// Solve this Quadratic Equation,
		///		t = ( -bh +- sqrt(bh*bh - a*c) )/a, bh is half b, a is 1.0
		/// 
		scalar bh = dot(raySp, rayDr);
		scalar c  = dot(raySp, raySp) - sph.r*sph.r;
		scalar discrim = bh*bh - c;
		if (discrim < 0) { 
			return range<scalar>::empty_range();
		} else if ( 0 < bh ) {///optimize more accuracy.
			scalar t0 = -bh - sqrt(discrim);
			scalar t1 = c/t0;
			return range{t0,t1};
		} else if ( bh < 0 ) {///optimize more accuracy.
			scalar t1 = (-bh + sqrt(discrim));
			scalar t0 = c/t1;
			return range{t0,t1};
		} else /*if (bh == 0)*/ {///avoid "0/0=nan".
/// Error case:
///		if ( bh <= 0 ) {
///			t1 = (-bh + sqrt(discrim)); //bh=0, discrim=0 
///			t0 = c/t1; //c=0, t1=0
///		}
			scalar t = sqrt(discrim);
			return range{t,t};
		}
	}

	template<typename vectorN>
	auto intersection(const sphere<vectorN>& sph, const ray<vectorN>& ray) {
		return intersection(sph.ori, ray.s - sph.c, ray.d);
	}

#if 0
	template<typename vector3>
	auto inside_ray_intersection(const origin_sphere<vector3>& sph, const vector3& raySp, const vector3& rayDr) {
		auto c  = dot(r.s, r.s) - sph.r*sph.r;
		auto bh = dot(r.s, r.d);
		auto sqrt_discrim = sqrt(max(bh*bh - c, decltype(bh)(0)));
		return 0 < bh ? c/(-bh - sqrt_discrim) : (-bh + sqrt_discrim);
	}

	template<typename vector3>
	bool ray_intersects(const origin_sphere<vector3>& sph, const vector3& raySp, const vector3& rayDr) {
		auto c  = dot(raySp, raySp) - sph.r*sph.r;
		if (c <= 0) { return true; }/// ray inside sphere.
		auto bh = dot(raySp, rayDr);
		if (bh > 0) { return false; }/// ray outside sphere and ray lookout sphere.
		if (bh*bh < c) { return false; }/// ray missing sphere.
		return true;
	}

	template<typename vector3>
	bool outside_ray_intersects(const origin_sphere<vector3>& sph, const vector3& raySp, const vector3& rayDr) {
		auto c  = dot(raySp, raySp) - sph.r*sph.r;
		auto bh = dot(raySp, rayDr);
		if (bh > 0) { return false; }/// ray outside sphere and ray lookout sphere.
		if (bh*bh < c) { return false; }/// ray missing sphere.
		return true;
	}

#endif

	/// slice linear shape.


	/// quadratic structure.






	template<typename vector3, typename scalar>
	scalar ray_sph_inside_intersect(const vector3& O_ray, const vector3& D_ray, const scalar r_sph) {
		scalar bh = dot(O_ray,D_ray);
		scalar c  = dot(O_ray,O_ray) - r_sph*r_sph;
		scalar sqrt_discrim = sqrt(std::max(bh*bh - c, scalar(0)));
		return 0 < bh ? c/(-bh - sqrt_discrim) : (-bh + sqrt_discrim);
	}

	template<typename vector3, typename scalar>
	bool ray_sph_test(const vector3& O_ray, const vector3& D_ray, const scalar r_sph) {
		scalar c = dot(O_ray,O_ray) - r_sph*r_sph;
		if (c <= 0) { return true; }/// ray inside sphere.
		scalar bh = dot(O_ray,D_ray);
		if (bh > 0) { return false; }/// ray outside sphere and ray lookout sphere.
		if (bh*bh < c) { return false; }/// ray missing sphere.
		return true;
	}

	template<typename vector3, typename scalar>
	bool ray_sph_outside_test(const vector3& O_ray, const vector3& D_ray, const scalar r_sph) {
		scalar bh = dot(O_ray,D_ray);
		scalar c  = dot(O_ray,O_ray) - r_sph*r_sph;
		if (bh > 0) { return false; }/// ray outside sphere and ray lookout sphere.
		if (bh*bh < c) { return false; }/// ray missing sphere.
		return true;
	}
	
	template<typename vector3, typename scalar>
	int ray_box_intersect2(const vector3& O_ray, const vector3& D_ray, bool isurf, const vector3& R_box, scalar t[]) {
		// Compute ray intersect six-planes (@see rayiplane).
		vector3 strides = 1/D_ray;
		vector3 t0_array = (-R_box - O_ray)*strides;
		vector3 t1_array = ( R_box - O_ray)*strides;
		vector3 t1_temp = max(t0_array, t1_array);
		t0_array = min(t0_array, t1_array);
		t1_array = t1_temp;
		scalar t0 = std::max(std::max(t0_array[0], t0_array[1]), t0_array[2]);
		scalar t1 = std::min(std::min(t1_array[0], t1_array[1]), t1_array[2]);
		if (t1 < t0) { return 0; }
		if (t1 < 0) { return -2; }

#ifdef __calculation_shrink_intersections__
		if (t1 == 0 || t0 == t1) { t[0] = t1; return 1; }/// ray at box radius and lookout box || ray outside box and intersect tap.
#endif
		if (t0 < 0) {
			if (isurf) { t[0] = t1; return 1; }
			else { 
			t[0] = 0; t[1] = t1; return 2; }
		} else {
			t[0] = t0; t[1] = t1; return 2;
		}
	}

	template<typename vector3>
	inline bool outof_box(const vector3& P, const vector3& MIN_box, const vector3& MAX_box) {
		return P[0] < MIN_box[0] || MAX_box[0] < P[0]
			|| P[1] < MIN_box[1] || MAX_box[1] < P[1]
			|| P[2] < MIN_box[2] || MAX_box[2] < P[2];
	}

	template<typename vector3, typename scalar>
	inline bool outof_sph(const vector3& P, const scalar r_sph) {
		return dot(P,P) > r_sph*r_sph;
	}

	template<typename vector3, typename scalar = std::remove_cvref_t<decltype(vector3{}[0])>>
	inline scalar sdfof_box(const vector3& P, const vector3& R_box) {
		vector3 q = abs(P) - R_box;
		return length(max(q,0)) + std::min(std::max(std::max(q[0],q[1]),q[2]),scalar(0));
	}

	template<typename vector3, typename scalar>
	inline scalar sdfof_sph(const vector3& P, const scalar r_sph) {
		return length(P) - r_sph;
	}

	template<typename vector3, typename scalar>
	inline scalar sdfof_pln(const vector3& P, const vector3& N_pln, const scalar d_pln) {
		return dot(P, N_pln) + d_pln;
	}

	

	///@param Oray is ray(Oray,Dray) 
	///@param Dray is ray(Oray,Dray) 
	///@param isurf is whether surface of volume 
	///@param irsph is sphere2({0,0,0},irsph,orsph) 
	///@param orsph is sphere2({0,0,0},irsph,orsph) 
	///@param t is output, assert(t.size() >= 4) 
	template<typename vector3, typename scalar>
	int ray_sph_intersect4(const vector3& Oray, const vector3& Dray, bool isurf, const scalar irsph, const scalar orsph, scalar t[]) {
		assert( !isurf );
		assert( irsph < orsph );
		///@diagram 
		///          .  --  .
		///        -||||||||||-
		///     .|||||- ~~~ - ||||.
		///    /|||/           \|||\
		///    |||.              .||.
		///   ||||       C--ir --||||
		///    |||.      |-- --or --|
		///    \|||\           /|||/
		///      .||||- _ _ -||||.
		///        .|||||||||||.
		///          -  --  -
		scalar a  = 1;
		scalar bh = dot(Oray,Dray);
		scalar ic = dot(Oray,Oray) - irsph*irsph;
		scalar oc = dot(Oray,Oray) - orsph*orsph;

		// if intersect inner, then must intersect outer, but the reverse is not true.
		// so, we first test outer.
		scalar od = bh*bh - a*oc;
		scalar ot0, ot1;
		if ( od < 0 ) {
			return 0;// not intersect any thing
		}
		if ( bh > 0 ) {
			ot0 = -bh - sqrt(od);
			ot1 = oc/ot0;/*(-bh + sqrt(od));*/
		} else {
			ot1 = -bh + sqrt(od);
			ot0 = oc/ot1;/*(-bh - sqrt(od));*/
		}
		if ( ot1 < 0 ) {
			return -1;/* back to all thing */
		}

		// then, test inner.
		scalar id = bh*bh - a*ic;
		scalar it0, it1;
		if ( id >= 0 ) {
			if ( bh > 0 ) {
				it0 = -bh - sqrt(id);
				it1 = ic/it0;/*(-bh + sqrt(id));*/
			} else {
				it1 = -bh + sqrt(id);
				it0 = ic/it1;/*(-bh - sqrt(id));*/
			}
		}
		if ( id < 0 || it1 < 0 ) {/* intersected outer-shell, but not inner-shell */
			assert( ot0 <= ot1 );
			assert( 0 <= ot1 );
			if /*..*/ (ot0 < 0) {
				t[0] = 0;
				t[1] = ot1;
				return 2;
			} else {
#ifdef __calculation_shrink_intersections__
				if (od == 0) {
					t[0] = ot1;
					return 1;
				}
#endif
				t[0] = ot0;
				t[1] = ot1;
				return 2;
			}
		}

		assert( it0 <= it1 );
		assert( it1 >= 0 );
		if /*..*/ (it0 < 0) {/* in inner-shell */
			assert(ot0 < 0);
			t[0] = it1;
			t[1] = ot1;
			return 2;
		} else if (ot0 < 0) {/* in outer-shell */
#ifdef __calculation_shrink_intersections__
			if (id == 0) {
				t[0] = 0;
				t[1] = it1;
				t[2] = ot1;
				return 3;
			}
#endif
			t[0] = 0;
			t[1] = it0;
			t[2] = it1;
			t[3] = ot1;
			return 4;
		} else {/* out outer-sheel */
			assert(od != 0);
#ifdef __calculation_shrink_intersections__
			if (id == 0) {
				t[0] = ot0;
				t[1] = it1;
				t[2] = ot1;
				return 3;
			}
#endif
			t[0] = ot0;
			t[1] = it0;
			t[2] = it1;
			t[3] = ot1;
			return 4;
		}
	}
	
	///@param h = length(ray.origin) 
	///@param mu = dot(ray.origin,ray.direction)/length(ray.origin) 
	///@param isurf is whether surface of volume 
	///@param r is sphere({0,0,0},r) 
	///@param t is output, assert(t.size() >= 2) 
	template<typename scalar>
	int ray1sphere(const scalar h, const scalar mu, bool isurf, const scalar r, scalar* t) {
		assert( h >= 0 && abs(mu) <= 1 );
		assert( r > 0 );
		/** solve [ length(o+d*t) == r ]
		 * sqrt( dot(o+d*t,o+d*t) ) == r
		 * sqrt( dot(o,o) + dot(d*t,d*t) + 2*dot(o,d*t) ) == r
		 *       dot(o,o) + dot(d*t,d*t) + 2*dot(o,d*t)   == r*r
		 *       dot(o,o) + dot(d,d)*t*t + dot(o,d)*2*t   == r*r
		 *           h*h  +     1 * t*t  + mu*h*2*t - r*r == 0
		 * 
		 * solve [ t*t  + (mu*h*2)*t + (h*h-r*r) == 0 ]
		 *   (-b +- sqrt(b*b - a*c*4)) / (a*2)
		 *   = ( -b/2 +- sqrt(b/2*b/2 - a*c) ) / a
		 *   = ( -mu*h +- sqrt(mu*h*mu*h - (h*h-r*r)) )
		 * 
		 * optimize
		 *   2*c / (-b - sqrt(b*b - a*c*4))
		 *   = 2*(h*h-r*r) / ( (-mu*h + sqrt(mu*h*mu*h - (h*h-r*r)))*(a*2) )
		 *   = (h*h-r*r) / ( -mu*h + sqrt(mu*h*mu*h - (h*h-r*r)) )
		 * 
		 * readme
		 *   large error in 'ray_origin close to sphere boundary'
		 * 
		 *@note
		 * float t[2];
		 * math::geometry::ray1sphere(false, 6376262.50f, -0.0713751018f, 6360000.0f, t).
		 * /// d = (mu*mu-1)*h*h + r*r
		 * ///   = -4.04495993e+13 + 4.04495993e+13, larger error.
		 * ///
		 * /// d = mu*h*mu*h - (h*h-r*r)
		 * ///   = 2.07121809e+11 - 2.07123120e+11, more accuracy, and more speed.
		*/
		scalar bh = mu*h;
		scalar c  = h*h - r*r;
		scalar discrim = bh*bh - c;
		if ( discrim < 0 ) {
			return 0;
		}
		scalar t0, t1;
		if (mu > 0) {
			t0 = -bh - sqrt(discrim);
			t1 = c/t0;//-bh + sqrt(discrim)
		} else {
			t1 = -bh + sqrt(discrim);
			t0 = c/t1;//-bh - sqrt(discrim)
		}

		assert(t0 <= t1);
		if /*..*/ (t1 < 0) {
			return -1;
		} else if (t0 < 0) {
			if (isurf) {
				t[0] = t1;
				return 1;
			} else {
#ifdef __calculation_shrink_intersections__
				if (t1 == 0) {
					t[0] = 0;
					return 1;
				}
#endif
				t[0] = 0;
				t[1] = t1;
				return 2;
			}
		} else {
#ifdef __calculation_shrink_intersections__
			if ( d == 0 ) {
				t[0] = t1;
				return 1;
			}
#endif
			t[0] = t0;
			t[1] = t1;
			return 2;
		}
	}

	///@param Oray is ray(Oray,Dray) 
	///@param Dray is ray(Oray,Dray) 
	///@param rsph is sphere({0,0,0},rsph) 
	///@param drdx (sqrt( (-2*r*r)/(cos(larg*2) - 1) - r*r ),ld,larg*2) is cone 
	///@param xshadow is shadow length, recommend xshadow = rsph/drdx 
	///@param t is output, assert(t.size() >= 2) 
	template<typename vector3, typename scalar>
	int ray1sphshadow(const vector3& Oray, const vector3& Dray, const scalar rsph, const vector3& Dlit, const scalar drdx, const scalar xshadow, scalar* t) {
		///@reference { title={Precomputed Atmospheric Scattering}, author={Eric Bruneton, Fabrice Neyret}, 
		/// doi={10.1111/j.1467-8659.2008.01245.x}, year={2008,2017} }
		///@diagram
		///                \  |  /
		///                      
		///                  \|/
		///                   * lit
		///                  /|\
		///                 / | \
		///                /  |  \
		///               /   |   \
		///              /   \|/   \
		///             /  - ~~~ -  \
		///            //           \\
		///           /.             .\
		///          /|       C-- r --|\
		/// ray     /  .      |      .  \
		///  *     /    \           /    \
		///      \/        - _|_ -        \
		///      /   \                     \
		///     /        \    |             \
		///    /             \               \
		///   /         dot(X-C,Dlit)         \
		///  /                |      \         \
		///                              \      \
		/// 	                |              \   \
		///                                      \\
		///                   |                    \ \   X = ray(t)
		///                   *---  --  --  -r2  -- \ ---*
		///                                          \      _\
		///                                                     _\
		///@theory
		/// The point'X' at distance't' from the ray is on the shadow pencil with sphere center'C = 0' only if
		/// shadowed radius'r' = radius'r' + rate of change of pencil radius(drdx = tan(light angular radius)) * project point'X' onto light direction'D'.
		///		( rsph + drdx*dot(X - 0, Dlit) ) = length( (X - 0) - proj(X - 0,Dlit) )
		///		( rsph + drdx*dot(X, Dlit) )^2   = sqr( X - proj(X, Dlit) )
		///		                                 = sqr(X) + sqr(proj(X, Dlit)) - dot(X, proj(X,Dlit))*2
		///		                                 = sqr(X) + dot(X, Dlit)^2*sqr(Dlit) - dot(X, Dlit)*dot(X, Dlit)*2
		///		                                 = sqr(X) - dot(X, Dlit)^2
		/// 
		/// Therefore, we can express intersected point as,
		///		( rsph + drdx*dot(Oray+Dray*t, Dlit) )^2 = sqr(Oray+Dray*t) - dot(Oray+Dray*t, Dlit)^2
		/// 
		/// Expand use Distributive-law,
		///		( rsph + drdx*dot(Oray,Dlit) + drdx*dot(Dray,Dlit)*t )^2
		///		= sqr(Oray) + sqr(Dray)*t*t + dot(Oray,Dray)*t*2 - ( dot(Oray,Dlit) + dot(Dray,Dlit)*t )^2
		/// 
		/// Expand use Binomial,
		///		( rsph + drdx*dot(Oray,Dlit) )^2 + ( drdx*dot(Dray,Dlit) )^2*t*t + ( rsph + drdx*dot(Oray,Dlit) )*drdx*dot(Dray,Dlit)*t*2
		///		= sqr(Oray) + sqr(Dray)*t*t + dot(Oray,Dray)*t*2 - dot(Oray,Dlit)^2 - dot(Dray,Dlit)^2*t*t - dot(Oray,Dlit)*dot(Dray,Dlit)*t*2 .
		/// 
		///		0 = ( ( drdx*dot(Dray,Dlit) )^2 - sqr(Dray) + dot(Dray,Dlit)^2 )*t*t
		///		  + ( ( rsph + drdx*dot(Oray,Dlit) )*drdx*dot(Dray,Dlit) - dot(Oray,Dray) + dot(Oray,Dlit)*dot(Dray,Dlit) )*t*2
		///		  + ( rsph + drdx*dot(Oray,Dlit) )^2 - sqr(Oray) + dot(Oray,Dlit)^2
		/// 
		/// This equation always have solution, but not shadowed always.
		/// So we need clipping test use boundary plane. (equation always has solution, boundary should always has solution, choose Plane)
		///		tbound = (distance - dot(Oray, Dlit))/dot(Dray, Dlit).
		/// 
		///@optimization 
		/// define relation'rel = dot(Dray,Dlit)', length'xray = dot(Oray,Dlit)', radius'rray = rsph + drdx*xray'
		///	
		/// a = ( drdx*dot(Dray,Dlit) )^2 - sqr(Dray) + dot(Dray,Dlit)^2
		///   = ( drdx*rel )^2            - sqr(Dray) + rel^2
		///   = (drdx*drdx + 1)*rel^2 - 1
		/// 
		/// bh = ( rsph + drdx*dot(Oray,Dlit) )*drdx*dot(Dray,Dlit) - dot(Oray,Dray) + dot(Oray,Dlit)*dot(Dray,Dlit)
		///	  =                            rray*drdx*rel            - dot(Oray,Dray) +           xray*rel
		///   = ( rray*drdx + xray )*rel - dot(Oray,Dray)
		/// 
		/// c = ( rsph + drdx*dot(Oray,Dlit) )^2 - sqr(Oray) + dot(Oray,Dlit)^2
		///   =               rray^2             - sqr(Oray) + xray^2
		/// 
		using std::min, std::max;
		scalar rel  = dot(Dray,Dlit);
		scalar xray = dot(Oray,Dlit);
		// boundary.
		scalar tbase = (-rsph - xray)/rel;         //scalar t_base = (-(sqrt( (-2*r*r)/(cos(larg*2) - 1) - r*r )) - ro_x)/cos_rarg;
		scalar tapex = (xshadow - xray)/rel;
		if (rel < 0) { assert(tbase > tapex);
			std::swap(tbase, tapex);
		}
		if (tapex < 0) {
			return 0;
		}
		// equation(!!! negate equation in @theory ).
		scalar rray = rsph + drdx*xray;
		scalar a    = 1 - (drdx*drdx + 1)*rel*rel;
		scalar bh   = dot(Oray,Dray) - (rray*drdx + xray)*rel;
		scalar c    = dot(Oray,Oray) - rray*rray - xray*xray;
		scalar discrim = bh*bh - a*c;
		if (discrim < 0) {
			return 0;
		}
		scalar t0, t1;
		if ( 0 < bh ) {
			scalar negbh_sub_sqrtdiscrim = -bh - sqrt(discrim);
			t0 = negbh_sub_sqrtdiscrim/a;
			t1 = c/negbh_sub_sqrtdiscrim;
		} else if (bh < 0) {
			scalar negbh_add_sqrtdiscrim = -bh + sqrt(discrim);
			t0 = c/negbh_add_sqrtdiscrim;
			t1 = negbh_add_sqrtdiscrim/a;
		} else {
			scalar negbh_add_sqrtdiscrim = sqrt(discrim);
			t1 = t0 = negbh_add_sqrtdiscrim/a;
		}
#if 0
		/// // Comment "if (tapex < 0) {...}"
		/// vector3 Oray = { 2, 2, 0 };
		/// vector3 Dray;
		/// vector3 Dlit = { 0, -1, 0 };
		/// scalar  alit = 0.7;
		/// scalar t[2];
		/// for (scalar aray = 0; aray <= 6.28; aray += 0.1f) {
		///		Dray = { cos(aray), sin(aray), 0 };
		///		std::cout << "aray=" << aray << '\t';
		///		math::geometry::ray1sphshadow_pencil(Oray, Dray, scalar(1), Dlit, tan(alit), scalar(1)/tan(alit), t);
		/// }
		///                                               aray=0          dot(Dray,Dlit)= 0               t0=-2.6845765      t1=-1.3154235   a= 1
		///          /\                                   aray=0.1        dot(Dray,Dlit)=-0.09983342      t0=-2.4878087      t1=-1.4440675   a= 0.9829624
		///         /  \            /_                    aray=0.2        dot(Dray,Dlit)=-0.19866933      t0=-2.3396983      t1=-1.6185243   a= 0.93252885
		///        /    \      /-   \  \                  aray=0.3        dot(Dray,Dlit)=-0.29552022      t0=-2.2292562      t1=-1.8620865   a= 0.85071
		///       /  -   \   \|/        |                 aray=0.4        dot(Dray,Dlit)=-0.38941833      t0=-2.2180169      t1=-2.1492872   a= 0.7407677
		///      /        \        * -----> Dray[0]       aray=0.5        dot(Dray,Dlit)=-0.47942555      t0=-2.7765129      t1=-2.0950394   a= 0.60708493
		///     /          \   \                          aray=0.6        dot(Dray,Dlit)=-0.5646425       t0=-3.7611063      t1=-2.0635865   a= 0.45499128
		///    |     C- -1 -|   \__\                      aray=0.7        dot(Dray,Dlit)=-0.6442177       t0=-5.9193296      t1=-2.0532777   a= 0.29055017
		///   /              \    /                       aray=0.8        dot(Dray,Dlit)=-0.71735615      t0=-14.222956      t1=-2.0635874   a= 0.12031746
		///  /                \                           aray=0.870796   dot(Dray,Dlit)=-0.7648422       t0= 14217332(-inf) t1=-2.0835943   a=-1.1920929e-07  (parallel)
		/// /        -         \                          aray=0.9        dot(Dray,Dlit)=-0.783327        t0= 34.45556(-inf) t1=-2.0950391   a=-0.048920393
		///                                               aray=1          dot(Dray,Dlit)=-0.8414711       t0= 7.808558(-inf) t1=-2.149272    a=-0.2104162
		/// aray=1.1        dot(Dray,Dlit)=-0.8912074       t0= 4.4281745(-inf) t1=-2.229253    a=-0.35773158
		/// aray=1.2        dot(Dray,Dlit)=-0.93203914      t0= 3.112042 (-inf) t1=-2.3396976   a=-0.48499382
		/// aray=1.3        dot(Dray,Dlit)=-0.96355826      t0= 2.4176342(-inf) t1=-2.4878092   a=-0.58712924
		/// aray=1.4        dot(Dray,Dlit)=-0.9854498       t0= 1.9928664(-inf) t1=-2.6845772   a=-0.6600659
		/// aray=1.5        dot(Dray,Dlit)=-0.997495        t0= 1.7095821(-inf) t1=-2.94712     a=-0.700896
		/// aray=1.6        dot(Dray,Dlit)=-0.9995736       t0= 1.5100213(-inf) t1=-3.3031614   a=-0.7079922
		/// aray=1.7        dot(Dray,Dlit)=-0.99166477      t0= 1.3643878(-inf) t1=-3.8002398   a=-0.6810713
		/// aray=1.8        dot(Dray,Dlit)=-0.97384757      t0= 1.2558185(-inf) t1=-4.5266676   a=-0.6212064
		/// aray=1.9        dot(Dray,Dlit)=-0.94629997      t0= 1.174121(-inf)  t1=-5.6664414   a=-0.53078437
		/// aray=2          dot(Dray,Dlit)=-0.90929735      t0= 1.1128438(-inf) t1=-7.675842    a=-0.4134102
		/// aray=2.1        dot(Dray,Dlit)=-0.8632093       t0= 1.0677854(-inf) t1=-12.080444   a=-0.27376282
		/// aray=2.2        dot(Dray,Dlit)=-0.80849636      t0= 1.036184(-inf)  t1=-29.026861   a=-0.117409825
		/// aray=2.270796   dot(Dray,Dlit)=-0.7648422       t0= 1.0209463(-inf) t1=-29015388    a=-1.1920929e-07  (parallel)
		/// aray=2.3        dot(Dray,Dlit)=-0.74570525      t0= 1.0162615       t1= 70.319046   a= 0.04941547
		/// aray=2.4        dot(Dray,Dlit)=-0.67546326      t0= 1.0069621       t1= 15.936106   a= 0.2200625
		/// aray=2.5        dot(Dray,Dlit)=-0.59847236      t0= 1.0078098       t1= 9.037241    a= 0.3877278
		/// aray=2.6        dot(Dray,Dlit)=-0.5155017       t0= 1.0188475       t1= 6.351209    a= 0.54572743
		/// aray=2.7        dot(Dray,Dlit)=-0.42738026      t0= 1.0406424       t1= 4.934026    a= 0.6877624
		/// aray=2.8        dot(Dray,Dlit)=-0.33498865      t0= 1.0743595       t1= 4.0671387   a= 0.80817
		/// aray=2.9        dot(Dray,Dlit)=-0.23924993      t0= 1.12192         t1= 3.4889975   a= 0.9021502
		/// aray=3          dot(Dray,Dlit)=-0.14112072      t0= 1.1862879       t1= 3.0817244   a= 0.9659562
		/// aray=3.1        dot(Dray,Dlit)=-0.04158147      t0= 1.2719743       t1= 2.7845087   a= 0.9970443
		/// aray=3.2        dot(Dray,Dlit)= 0.05837324      t0= 1.385929        t1= 2.5629344   a= 0.99417514
		/// aray=3.3        dot(Dray,Dlit)= 0.1577447       t0= 1.5392038       t1= 2.396201    a= 0.9574631
		/// aray=3.4        dot(Dray,Dlit)= 0.25554004      t0= 1.7502564       t1= 2.2711456   a= 0.8883717
		/// aray=3.5        dot(Dray,Dlit)= 0.35078213      t0= 2.0521562       t1= 2.179181    a= 0.78965545
		/// aray=3.6        dot(Dray,Dlit)= 0.44251928      t0= 2.1146958       t1= 2.510202    a= 0.6652499
		/// aray=3.7        dot(Dray,Dlit)= 0.529835        t0= 2.0740333       t1= 3.2736077   a= 0.52011454
		/// aray=3.8        dot(Dray,Dlit)= 0.6118567       t0= 2.0550547       t1= 4.772792    a= 0.36003566
		/// aray=3.9        dot(Dray,Dlit)= 0.687765        t0= 2.0567842       t1= 8.970617    a= 0.19139487
		/// aray=4          dot(Dray,Dlit)= 0.7568014       t0= 2.0793107       t1= 81.2        a= 0.020915389
		/// aray=4.1        dot(Dray,Dlit)= 0.8182762       t0= 2.12379         t1=-11.498513(inf)   a=-0.14460659
		/// aray=4.2        dot(Dray,Dlit)= 0.871575        t0= 2.1926014       t1=-5.3942766(inf)   a=-0.2985716
		/// aray=4.3        dot(Dray,Dlit)= 0.91616523      t0= 2.2896647       t1=-3.5468144(inf)   a=-0.4348415
		/// aray=4.4        dot(Dray,Dlit)= 0.9516015       t0= 2.4210293       t1=-2.661786(inf)    a=-0.54798436
		/// aray=4.5        dot(Dray,Dlit)= 0.9775297       t0= 2.595902        t1=-2.1474044(inf)   a=-0.63348925
		/// aray=4.6        dot(Dray,Dlit)= 0.9936908       t0= 2.828464        t1=-1.8148284(inf)   a=-0.6879473
		/// aray=4.7        dot(Dray,Dlit)= 0.9999232       t0= 3.1412723       t1=-1.5851663(inf)   a=-0.70918727
		/// aray=4.8        dot(Dray,Dlit)= 0.9961648       t0= 3.5719993       t1=-1.4196929(inf)   a=-0.69636285
		/// aray=4.9        dot(Dray,Dlit)= 0.98245305      t0= 4.188116        t1=-1.2972374(inf)   a=-0.64998484
		/// aray=5          dot(Dray,Dlit)= 0.95892495      t0= 5.12293         t1=-1.2053163(inf)   a=-0.5719024
		/// aray=5.1        dot(Dray,Dlit)= 0.92581564      t0= 6.6809034       t1=-1.1361607(inf)   a=-0.46522856
		/// aray=5.2        dot(Dray,Dlit)= 0.8834559       t0= 9.740482        t1=-1.0847608(inf)   a=-0.33421576
		/// aray=5.3        dot(Dray,Dlit)= 0.8322689       t0= 18.307476       t1=-1.0478266(inf)   a=-0.18408716
		/// aray=5.4        dot(Dray,Dlit)= 0.77276623      t0= 165.70119       t1=-1.0232117(inf)   a=-0.020828128
		/// aray=5.5        dot(Dray,Dlit)= 0.7055423       t0=-23.46706        t1=-1.0095826        a= 0.14905304
		/// aray=5.6        dot(Dray,Dlit)= 0.6312689       t0=-11.008974    t1=-1.0062335   a= 0.31878346
		/// aray=5.7        dot(Dray,Dlit)= 0.5506881       t0=-7.2385345    t1=-1.0129946   a= 0.48159677
		/// aray=5.8        dot(Dray,Dlit)= 0.46460497      t0=-5.432314     t1=-1.0302101   a= 0.63100195
		/// aray=5.9        dot(Dray,Dlit)= 0.37387967      t0=-4.3825336    t1=-1.0587832   a= 0.76104283
		/// aray=6          dot(Dray,Dlit)= 0.2794187       t0=-3.7037938    t1=-1.1002933   a= 0.866535
		/// aray=6.1        dot(Dray,Dlit)= 0.18216588      t0=-3.235086     t1=-1.157226    a= 0.94327295
		/// aray=6.2        dot(Dray,Dlit)= 0.08309292      t0=-2.8973784    t1=-1.2333676   a= 0.9881972
		std::cout << std::format("dot(D_ray,Dlit)={0}\tt0={1}\tt1={2}\ta={3}\n", rel, t0, t1, a);
#endif

#if 0
		/// t0 <= t1, intersected a continued range.
		/// t0 > t1, intersected two individe infinite-range.
		if (t0 <= t1) {
			if (tapex < t0) {
				return 0;
			} else if (t1 < tbase || t1 < 0) {
				return -2;
			} else if (t0 < 0) {
				t[0] = max(tbase, (scalar)0);
				t[1] = min(tapex, t1);
				return 2;
			} else {
				t[0] = max(tbase, t0);
				t[1] = min(tapex, t1);
				return 2;
			}
		} else {
			/// (-inf,t1], [t0,inf)
			if (t1 < tbase || t1 < 0) {
				if (tapex < t0) {
					return 0;
				} else if (t0 < 0) {
					t[0] = max(tbase, (scalar)0);
					t[1] = tapex;
					return 2;
				} else {
					t[0] = max(tbase, t0);
					t[1] = tapex;
					return 2;
				}
			} else {
				t[0] = max(tbase, (scalar)0);
				t[1] = min(tapex, t1);
				return 2;
			} 
		}
#else
		/// we only need positive range.
		if (a < 0) {
			if (rel > 0) {
				t1 = std::numeric_limits<scalar>::infinity();
			} else {
				t0 = -std::numeric_limits<scalar>::infinity();
			}
		}
		if (t1<tbase||tapex<t0 || t1 < 0) {
			return -2;
		} else if (t0 < 0) {
			t[0] = max((scalar)0, tbase);
			t[1] = min(t1, tapex);
			return 2;
		} else {
			t[0] = max(t0, tbase);
			t[1] = min(t1, tapex);
			return 2;
		}
#endif
	}






#if 0
	template<typename vector3, typename scalar>
	inline int intersect(bool surface, const ray<vector3>& the_ray, const sphere<vector3>& the_sphere, scalar* result) {
		return ray1sphere(surface, the_ray.get_start() - the_sphere.get_center(), 
			the_ray.get_direction(), the_sphere.get_radius(), result);
	}

	template<typename vector3, typename scalar>
	inline int intersect(bool surface, const ray<vector3>& ray, const sphere2<vector3>& sphere, scalar* t) {
		return ray1sphere(surface, ray.o - sphere.c, ray.d, sphere.r[0], sphere.r[1], t);
	}

	template<typename vector3, typename Scalar>
	int intersect(const Ray<vector3>& ray, const Sphere<vector3>& sphere, Scalar* t, int surface_tag) {
		return ray1sphsurf(ray.o - sphere.c, ray.d, sphere.r, t);
	}

	template<typename Vector>
	Vector closest(const Vector& point, const Triangle<Vector>& triangle) {
		using Scalar = decltype(dot(Vector(),Vector()));
		Vector prel = point - triangle.points[2];
		Vector e0 = triangle.points[0] - triangle.points[2];
		Vector e1 = triangle.points[1] - triangle.points[2];
		/** 
		 * We have a point'p' and a triangle{'p0','p1','p2'}, then get following eq: 
		 *       [ p0*u + p1*v + p2*(1-u-v) = p ]
		 *   p0*u + p1*v + p2 - p2*u - p2*v = p
		 *            (p0-p2)*u + (p1-p2)*v = p - p2
		 *              e0*u        e1*v    = prel
		 * 
		 * There are two variables, we're need to write a system of two equation:
		 *   =>[ dot(e0*u + e1*v = prel, e0),
		 *       dot(e0*u + e1*v = prel, e1) ]
		 *   =>[ dot(e0,e0) + dot(e1,e0) = dot(prel,e0),
		 *       dot(e0,e1) + dot(e1,e1) = dot(prel,e1) ]
		 * 
		 * or matrix:
		 *   { dot(e0,e0), dot(e1,e0), dot(prel,e0), 
		 *     dot(e0,e1), dot(e1,e1), dot(prel,e1) }
		*/
		Scalar m00 = dot(e0,e0), m01 = dot(e1,e0), m02 = dot(prel,e0);
		Scalar m10 = dot(e0,e1), m11 = dot(e1,e1), m12 = dot(prel,e1);
		Scalar det = m00 * m11 - m01 * m10;
		Scalar u = (m02 * m11 - m01 * m12) / det;
		Scalar v = (m00 * m12 - m02 * m10) / det;

		/**
		 * @figure
		 *            v
		 *           /|\
		 *   \ Region2|
		 *     \      |
		 *       \    |
		 *         \  |
		 *            *p1
		 *            |\
		 *            |  \        Region1
		 *            |    \
		 *    Region3 |      \
		 *            | Region0\
		 *            |          \
		 *   -- -- -- *p2 -- -- -- *p0 -- -- -- -- -->> u
		 *            |              \  Region6
		 *    Region4 |    Region5     \
		 *            |                  \
		 *            |                    \the line (u+v==1)
		 * 
		 *   Note: triangle winding. axis coordinate.
		 * 
		 * @reference 
		 *   "https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf"
		 *   "https://www.gamedev.net/forums/topic/552906-closest-point-on-triangle/"
		*/
		if (u + v < 1) 
		{
			if (u < 0) {
				if (v < 0) {
					// (u+v<1 && u<0&&v<0) is Region4
					if (m02 < 0) {
						assert(m12 >= 0);
						u = 0;
						v = std::clamp(-m02/m00, Scalar(0), Scalar(1));
					} else {
						assert(m02 >= 0);
						u = std::clamp(-m12/m11, Scalar(0), Scalar(1));
						v = 0;
					}
				} else {
					// (u+v<1 && u<0&&v>=0) is Region3
					u = 0;
					v = std::clamp(-m02/m00, Scalar(0), Scalar(1));
				}
			} else {
				if (v < 0) {
					// (u+v<1 && u>=0&&v<0) is Region5
				} else {
					// (u+v<1 && u>=0&&v>=0) is Region0
				}
			}
		} 
		else 
		{
			if (u < 0) {
				assert(v >= 0);
				// (u+v>=1 && u<0&&v>=0) is Region2
			} else {
				if (v < 0) {
					// (u+v>=1 && u>=0&&v<0) is Region6
				} else {
					// (u+v>=1 && u>=0&&v>=0) is Region1
				}
			}
		}
	}

	template<typename vector3>
	AxisAlignedBox<vector3> Union(const AxisAlignedBox<vector3>& box, const vector3& point) {
		vector3 pmin = min(box.c - box.r, point);
		vector3 pmax = max(box.c + box.r, point);
		return AxisAlignedBox<vector3>{(pmin+pmax)/2, (pmax-pmin)/2};
	}

	template<typename vector3>
	AxisAlignedBox<vector3> Union(const AxisAlignedBox<vector3>& box, const AxisAlignedBox<vector3>& rbox) {
		vector3 pmin = min(box.c - box.r, rbox.c - rbox.r);
		vector3 pmax = max(box.c + box.r, rbox.c + rbox.r);
		return AxisAlignedBox<vector3>{(pmin+pmax)/2, (pmax-pmin)/2};
	}

	template<typename vector3, typename Scalar>
	float test(const AxisAlignedBox<vector3,Scalar>& box, const Plane<vector3,Scalar>& plane) {
		Scalar e = dot(box.r, plane.n);
		Scalar s = dot(box.c, plane.n) - plane.d;
		if (s > e) {
			return 1.0f;
		} else if (s < -e) {
			return -1.0f;
		} else {
			return 0.5f;
		}
	}
#endif

	/*template<typename vector3>
	vector3 */

	//template<typename Vector>
	//Vector closest(const Vector& point, const Line<Vector>& line) {
	//  using Scalar = decltype(dot(Vector(),Vector()));
	//  Vector p = point;
	//  Vector i = line.points[0];
	//  Vector f = line.points[1];
	//  /**
	//   * @figure 
	//   *     * p
	//   *     |
	//   *     |_
	//   * *---*-|-------*
	//   * i   m         f
	//   * 
	//   * m = i + proj(p-i,f-i)
	//   *   = i + normalize(f-i) * dot(p-i,f-i)/length(f-i)
	//   *   = i + (f-i)/length(f-i) * dot(p-i,f-i)/length(f-i)
	//   *   = i + (f-i) * dot(p-i,f-i)/sqrlength(f-i)
	//   *                |-------------+-------------| <- this is ratio of 'p-i' on 'f-i'
	//  */
	//  Vector fmi = f - i;
	//  Scalar ratio = dot(p-i,fmi)/dot(fmi,fmi);
	//  if (ratio < 0) {
	//    return i;
	//  } else if (ratio > 1) {
	//    return f;
	//  } else {
	//    return i + fmi * ratio;
	//  }
	//}

} }// end of namespace math::geometry

// Test angular radius.
//
//double h = 1;
//
//for (h = 0.2; h < 1000.0; h *= 1.1) {
//	std::cout << "h:"<<h<<"\t";
//
//	double w = 0.001;
//	double solid_angle = p7quad(-w/2,w/2, [&](double x){ 
//		return p7quad(-w/2,w/2, [&](double z){
//			dvector3_t light = { 0,h,0 };
//			dvector3_t point = { x,0,z };
//			dvector3_t normal = { 0,1,0 };
//			return dot(normal, normalize(light - point)) / pow(length(light - point),2);
//			});
//		});
//	std::cout << solid_angle*(3.141592653589793/4) << "\t";
//	
//	double angle = atan((w / 2) / h);
//	std::cout << angle*angle*3.141592653589793 << std::endl;
//}
//
//std::cout << sqrt(6.78e-5/3.141592653589793) << std::endl;
///// requires{ (light is a point source | light radius/light distance = small)
/////            && (object is a point | object radius/light distance = small) }.
///// 
///// sidelength = 5
////h:0.690454      3.74494         5.32018
////h:0.7595        3.63407         5.11392
////h:0.83545       3.51457         4.89527
////h:0.918995      3.38627         4.66469
////h:1.01089       3.24915         4.42297
////h:1.11198       3.10337         4.17125
////h:1.22318       2.94933         3.91108
////h:1.3455        2.78768         3.64442
////h:1.48005       2.61938         3.37358
////h:1.62805       2.44569         3.10128
////h:1.79086       2.26817         2.83043
////h:1.96995       2.08867         2.56412
////h:2.16694       1.90923         2.30542
////h:2.38364       1.73203         2.05724
////h:2.622         1.55924         1.82216
////h:2.8842        1.39295         1.6023
////h:3.17262       1.23502         1.39924
////h:3.48988       1.08698         1.21395
////h:3.83887       0.949993        1.0468
////h:4.22276       0.824791        0.897653
////h:4.64503       0.711697        0.765874
////h:5.10953       0.610654        0.650491
////h:5.62049       0.521285        0.550284
////h:6.18254       0.442962        0.463883
////h:6.80079       0.374883        0.389856
////h:7.48087       0.31614         0.326781
////h:8.22896       0.265777        0.273294
////h:9.05185       0.222841        0.228122
////h:9.95704       0.186412        0.190107
////h:10.9527       0.155634        0.158209
////h:12.048        0.129722        0.13151
////h:13.2528       0.107973        0.109211
////h:14.5781       0.0897634       0.0906192
////h:16.0359       0.0745512       0.0751413
////h:17.6395       0.0618655       0.0622719
////h:19.4034       0.0513028       0.0515822
////h:21.3438       0.0425189       0.0427108
////h:23.4782       0.035222        0.0353537
////h:25.826        0.0291656       0.0292559
////h:28.4086       0.0241426       0.0242045
////h:31.2494       0.0199792       0.0200215
////h:34.3744       0.0165299       0.0165589
////h:37.8118       0.0136736       0.0136934
////h:41.593        0.011309        0.0113226
////h:45.7523       0.00935211      0.00936139
////h:50.3275       0.00773301      0.00773935
////h:55.3603       0.00639365      0.00639798
////h:60.8963       0.00528587      0.00528884
////h:66.986        0.00436976      0.00437179
////h:73.6846       0.00361225      0.00361363
////h:81.053        0.00298592      0.00298687
////h:89.1583       0.00246811      0.00246876
////h:98.0741       0.00204004      0.00204048
////h:107.882       0.00168617      0.00168648
////h:118.67        0.00139366      0.00139387
////h:130.537       0.00115188      0.00115202
////h:143.59        0.000952024     0.00095212
////h:157.949       0.000786838     0.000786903
////h:173.744       0.000650307     0.000650352
////h:191.119       0.000537463     0.000537494
////h:210.231       0.000444198     0.000444219
////h:231.254       0.000367115     0.000367129
////h:254.379       0.000303407     0.000303416
////h:279.817       0.000250754     0.00025076
////h:307.799       0.000207237     0.000207242
////h:338.579       0.000171272     0.000171275
////h:372.436       0.000141549     0.000141551
////h:409.68        0.000116983     0.000116985
////h:450.648       9.66811e-05     9.66821e-05
////h:495.713       7.99022e-05     7.99028e-05
////h:545.284       6.60351e-05     6.60356e-05
////h:599.813       5.45747e-05     5.4575e-05
////h:659.794       4.51032e-05     4.51034e-05
////h:725.773       3.72755e-05     3.72756e-05
////h:798.351       3.08062e-05     3.08063e-05
////h:878.186       2.54597e-05     2.54598e-05
////h:966.004       2.10411e-05     2.10412e-05
//// the larger distance than object's size, the smaller error 
////
//// sidelength = 0.001
////h:0.2           1.96348e-05     1.96349e-05
////h:0.22          1.62272e-05     1.62272e-05
////h:0.242         1.34109e-05     1.34109e-05
////h:0.2662        1.10834e-05     1.10834e-05
////h:0.29282       9.15982e-06     9.15983e-06
////h:0.322102      7.57011e-06     7.57011e-06
////h:0.354312      6.25629e-06     6.25629e-06
////h:0.389743      5.17049e-06     5.17049e-06
////h:0.428718      4.27313e-06     4.27313e-06
////h:0.47159       3.53152e-06     3.53152e-06
////h:0.518748      2.91861e-06     2.91861e-06
//// the smaller object's size, the smaller error 