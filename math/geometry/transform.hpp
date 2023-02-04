#pragma once

/// Geometry Transform.
///@license Free 
///@review 2023-01-23 
///@contact Jiang1998Nan@outlook.com 
#define _MATH_GEOMETRY_TRANSFORM_

#include <math/concepts.hpp>
#include <cassert>

namespace math { namespace geometry { 
///<h1>Orhogonal Spatial Coordinate System</h1>
///<p>
///Define the orthogonal spatial coordinate system use three unit-vectors {i,j,k}, they are orthogonal.
///and with a "order" is cross-product a vector and next vector equal next next vector.
///(modulus for index out of range, cross(1:j, 2:k) = 3:?, 3%3=0, ? is i. the determinant({i,j,k}) is +1 not -1)
/// they are may be any arrangement related the real world, example:
///</p><pre>
///		          + y:{0,1,0}                                   + y:{0,1,0}
///		         /|\    _                                      /|\
///		          |     +| z:{0,0,1}                            |     /
///		          |   /                                         |   /
///		          | /                                           | /
///		----------+----------->+                   -------------+----------->+
///		        / |            x:{1,0,0}                      / |            x:{1,0,0}
///		      /   |                                         /   |
///		    /     |                                      |+_ 
///		          |                                      z:{0,0,1}
///       Left-Hand-CS                                  Right-Hand-CS
///
///		          + z:{0,1,0}
///		         /|\    _ 
///		          |     +| x:{0,0,1}
///		          |   /
///		          | /
///		----------+----------->+ 
///		        / |            y:{1,0,0}
///		      /   |
///		    /     |
///		          |
///       Left-Hand-CS
///</pre><p>
///No matter how three unit-vectors are arranged relate "the real world",
///or arranged in "math" with orthogonal and cross-product order, or both.
///These orthogonal spatial coordinate systems must be identical in "math".
///Meaning relate real world can be not identical. example:
///</p><pre>
///		Cross-product will clockwise rotate 90 degrees in Left-Hand-CS.
///		(is like the 'left palm' rotating around the 'thumb', so named left hand.)
/// 
///		Cross-product will anti-clockwise rotate 90 degrees in Right-Hand-CS.
///		(is like the 'right palm' rotating around the 'thumb', so named right hand.)
/// 
///		cross(x:{1,0,0}, y:{0,0,1}) must be z:{0,0,1}, this is in the math.
///</pre>

	template<typename vector3>
	struct oriented_orthogonal_basis {
		using scalar = std::remove_cvref_t< decltype(std::declval<vector3>()[0]) >;
		vector3 N;
		vector3 T[2];

		static oriented_orthogonal_basis from_tangent(const vector3& T0, const vector3& T1) {
			return {cross(T0,T1), T0, T1};
		}

		static oriented_orthogonal_basis from_normal(const vector3& N, const size_t z = 2) {
			const auto Nz = N[z],
				Nx = N[(z+1)%3],
				Ny = N[(z+2)%3];

			///<p>"Building an Orthonormal Basis, Revisited" in "Journal of Computer Graphics Techniques Vol. 6"</p>
			auto sign = copysign(scalar(1), Nz);
			auto a = -1/(sign + Nz);
			auto b = Nx * Ny * a;
			return {N, vector3{1 + sign*Nx*Nx*a, sign*b, -sign*Nx}, vector3{b, sign + Ny*Ny*a, -Ny}};
		}

		template<typename vector3u>
		static oriented_orthogonal_basis from_tangent(const vector3u& T0u, const vector3u& T1u) {
			return from_tangent(static_vcast<vector3>(T0u), static_vcast<vector3>(T1u));
		}

		template<typename vector3u>
		static oriented_orthogonal_basis from_normal(const vector3u& Nu, const size_t z = 2) {
			return from_normal(static_vcast<vector3>(Nu), z);
		}

		static oriented_orthogonal_basis identity() {
			return {vector3{1,0,0}, vector3{0,1,0}, vector3{0,0,1}};
		}

		const vector3& tangent0() const noexcept {
			return T[0];
		}

		const vector3& tangent1() const noexcept {
			return T[1];
		}

		const vector3& normal() const noexcept {
			return N;
		}

		void flip() {
			N = -N;
			auto tmp = T[0];
			T[0] = T[1];
			T[1] = tmp;
		}

		scalar check() const {
			scalar a = (length(N) - 1), b = (length(T[0]) - 1), c = (length(T[1]) - 1),
				d = dot(N, T[0]), e = dot(N, T[1]), f = dot(T[0], T[1]);
			return (a*a + b*b + c*c + d*d + e*e + f*f)/6;
		}
	};

	template<math::vector vector3>
	vector3 sph2cart(const vector3& v, const oriented_orthogonal_basis<vector3>& V, const bool elevation_0_pi = true) {
		const auto azimuth = v[0],
			elevation = v[1],
			r = v[2];

		if (elevation_0_pi) {
			auto normal = cos(elevation) * r, 
				tangent = cos(azimuth) * sin(elevation) * r,
				bitangent = sin(azimuth) * sin(elevation) * r;
			return V.tangent0()*tangent + V.tangent1()*bitangent + V.normal()*normal;
		} else {
			auto normal = sin(elevation) * r,
				tangent = cos(azimuth) * cos(elevation) * r,
				bitangent = sin(azimuth) * cos(elevation) * r;
			return V.tangent0()*tangent + V.tangent1()*bitangent + V.normal()*normal;
		}
	}

	template<math::vector vector3>
	vector3 sph2cart(const vector3& v, const bool elevation_0_pi = true) {
		const auto azimuth = v[0],
			elevation = v[1],
			r = v[2];

		if (elevation_0_pi) {
			auto normal = cos(elevation) * r,
				tangent = cos(azimuth) * sin(elevation) * r,
				bitangent = sin(azimuth) * sin(elevation) * r;
			return {tangent, bitangent, normal};
		} else {
			auto normal = sin(elevation) * r,
				tangent = cos(azimuth) * cos(elevation) * r,
				bitangent = sin(azimuth) * cos(elevation) * r;
			return {tangent, bitangent, normal};
		}
	}

	template<math::vector vector3>
	vector3 cart2sph(const vector3& v, const oriented_orthogonal_basis<vector3>& V, const bool elevation_0_pi = true) {
		const auto tangent = dot(v, V.tangent0()),
			bitangent = dot(v, V.tangent1()),
			normal = dot(v, V.normal());

		auto azimuth = atan2(bitangent,tangent),
			elevation = elevation_0_pi
				? atan2(sqrt(tangent * tangent + bitangent * bitangent), normal)
				: atan2(normal, sqrt(tangent * tangent + bitangent * bitangent)),
			r = sqrt(normal*normal + tangent*tangent + bitangent*bitangent);
		return {azimuth, elevation, r};
	}

	template<math::vector vector3>
	vector3 cart2sph(const vector3& v, const bool elevation_0_pi = true) {
		const auto tangent = v[0],
			bitangent = v[1],
			normal = v[2];

		auto azimuth = atan2(bitangent,tangent),
			elevation = elevation_0_pi
				? atan2(sqrt(tangent * tangent + bitangent * bitangent), normal)
				: atan2(normal, sqrt(tangent * tangent + bitangent * bitangent)),
			r = sqrt(normal*normal + tangent*tangent + bitangent*bitangent);
		return {azimuth, elevation, r};
	}

	template<math::vector vector3, typename configty>
	[[deprecated("Slower than oriented_orthogonal_basis<...> and accuracy lower than that..")]]
	void get_tangent(const vector3& normal, vector3& tangent, vector3& bitangent, configty) {
	///@note 
	/// transpose{normal,tangent1,tangent2} from tbn to world (rotation from initial-normal to normal).
	/// {normal,tangent1,tangent2} from world to tbn.
	///@example 
	/// for (size_t i = 0; i != 100; ++i) {
	/// 	auto normal = normalize(uniform_normal_distribution(0.0, rng));
	///		decltype(normal) tangent1, tangent2;
	///		geom::get_tangent(normal, tangent1, tangent2, geom::sphconfig<0, geom::normal_at_x>{});
	///		decltype(normal) normal_in_tbn = { dot(normal,normal),dot(tangent1,normal),dot(tangent2,normal) };
	/// 
	///		/*geom::get_tangent(normal, tangent1, tangent2, geom::sphconfig<0,geom::normal_at_y>{});
	///		decltype(normal) normal_in_tbn = { dot(tangent1,normal),dot(normal,normal),dot(tangent2,normal) };*/
	/// 
	///		/*geom::get_tangent(normal, tangent1, tangent2, geom::sphconfig<0,geom::normal_at_z>{});
	///		decltype(normal) normal_in_tbn = { dot(tangent1,normal),dot(tangent2,normal),dot(normal,normal) };*/
	/// 
	///		std::cout << "normal:" << normal << "\tin tbn:" << normal_in_tbn << std::endl;
	/// }
	///@note 
	/// can direct rotation for isotropic|anisotropic, 
	/// not verify indirect rotation(animation for params) may be error at initial pivot('z','x','y').
		if constexpr (configty::normal_pos == 0) {
			auto sqlen = normal[1]*normal[1] + normal[2]*normal[2];
			tangent = sqlen > 0 ? cross(vector3{1,0,0},normal)/sqrt(sqlen) : vector3{0,1,0};
			bitangent = cross( normal, tangent );
		}
		else if constexpr (configty::normal_pos == 1) {
			auto sqlen = normal[0]*normal[0] + normal[2]*normal[2];
			tangent = sqlen > 0 ? cross(vector3{0,1,0},normal)/sqrt(sqlen) : vector3{0,0,1};
			bitangent = cross( normal, tangent );
		} 
		else {
			auto sqlen = normal[0]*normal[0] + normal[1]*normal[1];
			tangent = sqlen > 0 ? cross(vector3{0,0,1},normal)/sqrt(sqlen) : vector3{1,0,0};
			bitangent = cross( normal, tangent );
			//vector3 y = abs(normal[2]) == 1 ? vector3{0,1,0} : vector3{0,0,1};
			//tangent = normalize( cross( y, normal ) );
			//bitangent = cross( normal, tangent );
		}
	}


	template<math::vector vector3>
	vector3 scale(const vector3& x, const vector3& ds) {
		return x*ds;
	}

	template<math::vector vector3, math::number scalar>
	vector3 shear(const vector3& x, size_t i, size_t j, scalar ds) {
		vector3 y = x;
		y[i] = x[j]*ds + x[i];
		return y;
	}

	template<math::vector vector3>
	vector3 translate(const vector3& x, const vector3& dx) {
		return x + dx;
	}

	template<math::vector vector3>
	vector3 proj(const vector3& x, const vector3& ref) {
///@diagram
///           x
///         *
///       / |
///     /   |
///   /     |
/// /-------+--- ref
/// |---+---|
/// proj(x,ref) = dot(x,normalize(ref)) * normalize(ref)
/// 
///@optimization
/// There is a costly sqrt(Number) in normalize(Vector), which can be eliminated.
///		                             ref                     ref
///		proj(x,ref) = dot(x, --------------------) * --------------------
///		                      sqrt(dot(ref,ref))      sqrt(dot(ref,ref))
/// 
///		                          dot(x, ref)                ref
///		            =        -------------------- * --------------------
///		                      sqrt(dot(ref,ref))     sqrt(dot(ref,ref))
/// 
///		                          dot(x, ref)
///		            =           --------------- * ref
///		                          dot(ref,ref)
		return dot(x,ref)/dot(ref,ref) * ref;
///<h1>Vector Decomposition</h1>
///<pre>
///		        *---
///		   x  / | |
///		    /     +- (x - proj(x,ref)) is vertical component.
///		  /     | |
///		o- - - -+--- ref(or unit-vector n) is orthogonal vector for x.
///		|---+---|
///		proj(x,ref) is horizontal component.
///</pre><p>
///Decompose <em>x</em> into horizontal component and vertical component is
///</p><pre>
///		x = proj(x,ref) + (x - proj(x,ref))
///		x = cos(x,ref)*ref + sin(x,ref)*cross(ref,cross(x,ref))
///</pre><p>
///If 'ref' is unit-vector, then
///</p><pre>
///		x = dot(x,ref)*ref + (x - dot(x,ref)*ref)
///</pre>
	}

	template<math::vector vector3, math::number scalar>
	vector3 rotate3(const vector3& x, const vector3& n, scalar arg) {
///@article { title={Rodrigues's rotation formula}, author={Benjamin-Olinde-Rodrigues}, year={1840} }
///@diagram
///                x - proj(x,n)
/// o---------------------------+
///  \ \  ) arg                 | proj(x,n)
///   \   \                 \   |
///    \     \                 \|
///     \       \            x
///    n           +
///            |  /
///             |/ proj(rotate(x,n,arg),n)
///      rotate(x,n,arg)
/// 
///@theory
/// Decompose rotate(x,n,arg) is
/// 
///		rotate(x,n,arg) = cos(rotate(x,n,arg),n)*n + sin(rotate(x,n,arg),n)*cross(n,cross(rotate(x,n,arg),n)).
/// 
/// Because the rotation is around the unit-vector 'n', so
/// 1. the horizontal component is identity.
/// 2. the vertical is perpendicular 'n', vertical is rotate in a plane(2D).
/// 
///		rotate(x,n,arg) = proj(x,n) + rotate(x - proj(x,n),n,arg).
///		rotate(x,n,arg) = proj(x,n) + cos(arg)*(x - proj(x,n)) + sin(arg)*cross(n,x - proj(x,n)).
/// 
///@optimization
/// 'cross(n,x - proj(x,n)) = cross(n,x) - cross(n,proj(x,n))' the 'cross(n,proj(x,n))' is 0,
/// 
///		rotate(x,n,arg) = proj(x,n) + cos(arg)*(x - proj(x,n)) + sin(arg)*cross(n,x).
/// 
/// replace a vector operation by scalar operation,
/// 
///		rotate(x,n,arg) = proj(x,n)*(1 - cos(arg)) + cos(arg)*x + sin(arg)*cross(n,x).
/// 
		scalar cos_arg = cos(arg),
			sin_arg = sin(arg);
		return n*dot(x,n)*(1 - cos_arg) + cos_arg*x + sin_arg*cross(n,x);
	}

	template<math::vector vector3, math::number scalar>
	vector3 rotate3(const vector3& x, const vector3& n, scalar cos_arg, scalar sin_arg) {
		return n*dot(x,n)*(1 - cos_arg) + cos_arg*x + sin_arg*cross(n,x);
	}

	template<math::vector vector3, typename quaternion>
	vector3 rotate3(const vector3& x, const quaternion& n_arg) {
		return rotate3(x, n_arg);///requires ADL interface.
	}

	template<math::vector vector3>
	vector3 mirror(const vector3& x, const vector3& n) {
///@diagram
///          + n
///      +   |   +
///     x \  |  / mirror(x,n)
///        \ | /
///  --------o-------+ t
/// 
///@theory
/// The horizontal component is identity and the vertical component is opposite.
///		x           = proj(x,n) + (x - proj(x,n))
///		mirror(x,n) = proj(x,n) - (x - proj(x,n))
/// 
///@optimization
///		mirror(x,n) = proj(x,n)*2 - x
/// 
		return n*dot(x,n)*2 - x;
///@note dot(mirror(x,n),n) = dot(x,n).
///@note mirror(x,x) = x.
///@note mirror(x,n) = mirror(x,-n), because proj(x,n).
	}

	template<math::vector vector3>
	vector3 reflect(const vector3& vIn, const vector3& n) {
///@figure
///          + n
///      \   |   +
///   vIn \  |  / vOut
///        + | /
///  --------*-------+ t
///           \
///            \
///             + this is reality 'vIn' in the coordinate, 
///               so the reality mirror is relative to 't' not is to 'n'.
///@formula
///  vIn = cos(vIn,n)*n + sin(vIn,n)*t     :"t" is tangent "n", sin(vIn,n)*t = vIn-cos(vIn,n)*n.
///
///  vOut = -cos(vIn,n)*n + sin(vIn,n)*t
///       = -dot(vIn,n)*n + vIn-dot(vIn,n)*n
///       = vIn - dot(vIn,n)*n*2
///
		return vIn - n*dot(vIn,n)*2;
///@note dot(reflect(x,n),n) = -dot(x,n).
///@note reflect(x,x) = -x.
///@note reflect(x,n) = reflect(x,-n), because proj(x,n).
#if 0
for (size_t i = 0; i != 1000; ++i) {
	auto vIn = normalize(vector3{ distribution(engine),distribution(engine),distribution(engine) });
	auto n    = normalize(vector3{ distribution(engine),distribution(engine),distribution(engine) });
	auto vOut = math::geometry::reflect(vIn, n);
	std::cout << "vIn:" << vIn << "," << length(vIn) 
		<< ",\tn:" << n 
		<< ",\tvOut:" << vOut << "," << length(vOut) << std::endl;

	bool vOutLengthCorrect = equals(length(vOut), 1, std::numeric_limits<scalar>::epsilon()*10);
	bool vOutCosCorrect    = equals(dot(vIn,n), -dot(vOut,n), std::numeric_limits<scalar>::epsilon()*10);
	if (!vOutLengthCorrect || !vOutCosCorrect) {
		std::cout << "Error" << std::endl;
		std::cout << "Error" << std::endl;
		std::cout << "Error" << std::endl;
		std::cout << "Error" << std::endl;
	}
}
#endif
	}

	template<math::vector vector3, math::number scalar>
	vector3 refract(const vector3& vIn, const vector3& n, scalar eta/* = etaIn/etaOut */) {
///@figure
///          + n           --------------*------------+ t
///      \   |                           |\\
///   vIn \  |                           | \  \
///        + |                           |  \    \
///  --------*-------+ t                 |   \      \ vIn
///           \                          |    \vOut    +
///           \ vOut                     |     + 
///            +                         + -n
///@formula
///	vIn = cos(vIn,n)*n + sin(vIn,n)*t          :"t" is tangent "n", sin(vIn,n)*t = vIn-cos(vIn,n)*n.
///
///	vOut = cos(vOut,n)*n + sin(vOut,n)*t
///	     = cos(vOut,n)*n + sin(vIn,n)*eta*t    :Transform "sin(vOut,n)" by Snell'Law "sin(vOut,n)*etaOut = sin(vIn,n)*etaIn".
///	     = cos(vOut,n)*n + (vIn - cos(vIn,n)*n)*eta
///	     = sqrt(1 - sin(vOut,n)^2)*n + (vIn - cos(vIn,n)*n)*eta  :node sign of "sqrt(1 - sin(vOut,n)^2)".
/// 
		assert(eta != 0);
		scalar cosIn   = dot(vIn,n);
		scalar sinIn2  = 1 - cosIn*cosIn;
		scalar sinOut2 = sinIn2*(eta*eta); assert(sinOut2 <= 1);///@total internal reflection. 
		scalar cosOut  = sqrt(1 - sinOut2);
		/*return (n*copysign(cosOut,cosIn) + (vIn - n*cosIn)*eta);*/
		return (n*(copysign(cosOut,cosIn) - cosIn*eta) + vIn*eta);
///@note sin(refract(vIn,n,...),n)*etaOut == sin(vIn,n)*etaIn (is Snell's Law) &&
///      faceforward(refract(vIn,n,...),n) == faceforward(vIn,n).
///@note refract(x,x,...) = x.
#if 0
scalar etaIn = 2;
scalar etaOut = 1;
for (size_t i = 0; i != 1000; ++i) {
	auto vIn = normalize(vector3{ distribution(engine),distribution(engine),distribution(engine) });
	auto n    = normalize(vector3{ distribution(engine),distribution(engine),distribution(engine) });
	if( (1 - dot(vIn,n)*dot(vIn,n))*pow(etaIn/etaOut,scalar(2)) > 1) {
		continue;
	}
	auto vOut = math::geometry::refract(vIn, n, etaIn/etaOut);
	std::cout << "vIn:" << vIn << "," << length(vIn) 
		<< ",\tn:" << n 
		<< ",\tvOut:" << vOut << "," << length(vOut) << std::endl;

	bool vOutLengthCorrect = equals(length(vOut), 1, std::numeric_limits<scalar>::epsilon()*10);
	if (!vOutLengthCorrect) {
		std::cout << "Error, Length" << std::endl;
		std::cout << "Error, Length" << std::endl;
		std::cout << "Error, Length" << std::endl;
		std::cout << "Error, Length" << std::endl;
	}
	bool vOutCosCorrect = equals(sqrt(1 - dot(vIn,n)*dot(vIn,n))*etaIn, sqrt(1 - dot(vOut,n)*dot(vOut,n))*etaOut, std::numeric_limits<scalar>::epsilon()*10);
	if (!vOutCosCorrect) {
		std::cout << "sin(vIn,n)*etaIn:"<<sqrt(1 - dot(vIn,n)*dot(vIn,n))*etaIn << ", sin(vOut,n)*etaOut:" << sqrt(1 - dot(vOut,n)*dot(vOut,n))*etaOut << std::endl;
		std::cout << "Error, Angle" << std::endl;
		std::cout << "Error, Angle" << std::endl;
		std::cout << "Error, Angle" << std::endl;
		std::cout << "Error, Angle" << std::endl;
	}
	bool vOutSideCorrect = (dot(vIn,n) <= 0  && dot(vOut,n) <= 0) || (dot(vIn,n) > 0  && dot(vOut,n) > 0);
	if (!vOutSideCorrect) {
		std::cout << "dot(vIn,n):" << dot(vIn,n) << ", dot(vOut,n):"<<dot(vOut,n) << std::endl;
		std::cout << "Error, Side" << std::endl;
		std::cout << "Error, Side" << std::endl;
		std::cout << "Error, Side" << std::endl;
		std::cout << "Error, Side" << std::endl;
	}
}
#endif
	}


///<h1>Homogeneous Coordinates</h1>
///<p>
///We will see in the next section that an affine transformation is a linear 
///transformation combined with a translation. However, translation does not 
///make sense for vectors because a vector only describes direction and magnitude, 
///independent of location; in other words, vectors should be unchanged under 
///translations. Translations should only be applied to points (i.e., position vectors). 
///Homogeneous coordinates provide a convenient notational mechanism that enables 
///us to handle points and vectors uniformly. With homogeneous coordinates, we 
///augment to 4-tuples and what we place in the fourth w-coordinate depends on 
///whether we are describing a point or vector. Specifically, we write:
///</p><ol><li>
///		<code>(x, y, z, 0)</code> for vectors </li><li>
///		<code>(x, y, z, 1)</code> for points </li>
///</ol>

	///<h1>rigidity transformation <br /> <code style="font-weight:lighter; font-size:smaller">
	/// translate( rotate( INPUT ) )</code></h1>
	template<typename vector3, typename quaternion = void>
	struct rigidity {
		using scalar = std::remove_cvref_t< decltype(std::declval<vector3>()[0]) >;
		quaternion q;
		vector3 t;

		vector3 transform(const vector3& v) const {
			/// v1 = t.translate( q.rotate( v ) ) 
			return rotate3(v, q) + t;
		}
		
		vector3 invtransform(const vector3& v1) const {
			/// v = q.invrotate( (-t).translate( v1 ) )
			return invrotate3(v1 - t, q);
		}

		vector3 transform_without_translation(const vector3& v) const {
			/// v1 = q.rotate( v ) 
			return rotate3(v, q);
		}

		vector3 invtransform_without_translation(const vector3& v1) const {
			/// v = q.invrotate( v1 )
			return invrotate3(v1, q);
		}
		
		vector3 transform_for_normal(const vector3& n) const {
			return transform_without_translation(n);
		}

		vector3 invtransform_for_normal(const vector3& n1) const {
			return invtransform_without_translation(n1);
		}

		rigidity transform(const rigidity& T) const {
			/// T1 = t.translate( q.rotate( T ) )
			return {q*T.q, transform(T.t)};
		}

		rigidity invtransform(const rigidity& T1) const {
			/// T = q.invrotate( (-t).translate( T1 ) ) 
			return {inv(q)*T1.q, invtransform(T1.t)};
		}

		const vector3 size() const {
			return {1, 1, 1};
		}

		const vector3 direction(const size_t i) const {
			///<p>Reference <q>math/number/quaternion.hpp, quaternion<..>::to_matrix4()</q></p>
			auto b = q[0]*q[0] - (q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
			if (i == 0) {
				return {q[1]*q[1]*2 + b, (q[2]*q[1] + q[3]*q[0])*2, (q[3]*q[1] - q[2]*q[0])*2};
			} else if (i == 1) {
				return {(q[1]*q[2] - q[3]*q[0])*2, q[2]*q[2]*2 + b, (q[3]*q[2] + q[1]*q[0])*2};
			} else {
				return {(q[1]*q[3] + q[2]*q[0])*2, (q[2]*q[3] - q[1]*q[0])*2, q[3]*q[3]*2 + b};
			}
		}

		const vector3& position() const { 
			return t; 
		}

		void scale(const vector3& ds) { abort(); }

		void stretch(const vector3& ds) { abort(); }
		
		void resize(const vector3& s1) { abort(); }

		void rotate(const vector3& n, const scalar arg) {
			//t.translate(                 q.rotate( ... )   )
			//t.translate( (n,arg).rotate( q.rotate( ... ) ) )
			q = rotation3(quaternion{0,n[0],n[1],n[2]}, arg) * q;
		}

		void revolve(const vector3& n, const scalar arg) {
			//                t.translate( q.rotate( ... ) )
			//(n,arg).rotate( t.translate( q.rotate( ... ) ) )
			auto qq = rotation3(quaternion{0,n[0],n[1],n[2]}, arg);
			q = qq * q;
			t = rotate3(t, qq);
		}
		
		void redirect(const size_t i, const vector3& V1_i) {
			//t.translate(  q.rotate( ... ) )
			//t.translate( q1.rotate( ... ) )
			const vector3 V_i = this->direction(i);
			vector3 n = normalize(cross(V_i, V1_i));
			scalar arg = acos(dot(V_i, V1_i));
			this->rotate(n, arg);
		}

		void translate(const vector3& dt) {
			//                t.translate( q.rotate( ... ) )
			//(dt).translate( t.translate( q.rotate( ... ) ) )
			t += dt;
		}
	
		void reposit(const vector3& t1) {
			// t.translate( q.rotate( ... ) )
			//t1.translate( q.rotate( ... ) )
			t = t1;
		}

		void inverse() {
			// t.translate( q.rotate( ... ) )
			// q.invrotate( -t.translate( ... ) )
			q = inv(q);
			t = -t;
			t = rotate3(t, q);
		}
		
		void fix_orthogonality() {
			q = normalize(q);
		}

		rigidity inversion() const {
			rigidity inv;
			inv.q = inv(q);
			inv.t = -t;
			inv.t = rotate3(inv.t, inv.q);
			return std::move(inv);
		}
	};

	///<h1>flexibility transformation <br /> <code style="font-weight:lighter; font-size:smaller">
	/// translate( rotate( scale( INPUT ) ) )</code></h1>
	template<typename vector3, typename quaternion = void>
	struct flexibility : rigidity<vector3, quaternion> {
		using scalar = std::remove_cvref_t< decltype(std::declval<vector3>()[0]) >;
		vector3 s;

#if 0
		const vector3& size()const {
			return s;
		}

		const vector3 direction(const size_t i) const {
			abort();
		}

		const vector3& position() const { 
			return t; 
		}

		vector3 transform_without_translation(const vector3& v) const {
			vector3 v1 = v * s;
			return ( dot(imag(q),v1)*imag(q) + cross(imag(q),v1)*real(q) )*2 + (real(q)*real(q) - dot(imag(q),imag(q)))*v1;
		}

		vector3 transform(const vector3& v1) const {
			return transform_without_translation(v1) + t;
		}
#endif

		///...
	};

	template<typename vector3>
	struct rigidity<vector3, void> { 
		using scalar = std::remove_cvref_t< decltype(std::declval<vector3>()[0]) >;
		vector3 V[3];///is (orthogonal)vector-space of (scaled)rotation.
		vector3 t;///is vector of translation.
		
		vector3 transform(const vector3& v) const {
			///<pre>
			///	v1 = t.translate( V.rotate( v ) )
			///</pre>
			return (V[2]*v[2] + (V[1]*v[1] + (V[0]*v[0] + t)));//optimization forward order for mul_add.
		}
		
		vector3 invtransform(const vector3& v1) const {
			///<pre>
			///	v = inv(V).rotate( (-t).translate( v1 ) )
			///</pre>
			///<p>Optimization for <em>V</em> is orthogonal vector space</p>
			///<pre>
			///	v = tp(V).rotate( (-t).translate( v1 ) )
			///</pre>
			vector3 v1nt = v1 - t;
			return {dot(v1nt,V[0]), dot(v1nt,V[1]), dot(v1nt,V[2])};
		}

		vector3 transform_without_translation(const vector3& v) const {
			///<pre>
			///	v1 = V.rotate( v )
			///</pre>
			return ((V[0]*v[0] + V[1]*v[1]) + V[2]*v[2]);
		}

		vector3 invtransform_without_translation(const vector3& v1) const {
			///<pre>
			///	v = inv(V).rotate( v1 )
			///</pre>
			///<p>Optimization for <em>V</em> is orthogonal vector space</p>
			///<pre>
			///	v = tp(V).rotate( v1 )
			///</pre>
			return {dot(v1,V[0]), dot(v1,V[1]), dot(v1,V[2])};
		}
		
		vector3 transform_for_normal(const vector3& n) const {
			///<p>From Normal-Vector-Trnasform <em>dot(n,v) = 0 => dot(n*inv(M),M*v) = 0</em>, we have</p>
			///<pre>
			///	n1 = tp(inv(V)) * n
			///</pre>
			///<p>Optimization for 'V' is orthogonal vector space</p>
			///<pre>
			///	n1 = tp(tp(V)) * n = V * n
			///</pre>
			return transform_without_translation(n);
		}

		vector3 invtransform_for_normal(const vector3& n1) const {
			return invtransform_without_translation(n1);
		}

		rigidity transform(const rigidity& T) const {
			///<pre>
			///	T1 = t.translate( V.rotate( T ) )
			///</pre>
			return {transform_without_translation(T.V[0]), transform_without_translation(T.V[1]),
				transform_without_translation(T.V[2]), transform(T.t)};
			///<p>We want to verify by Linear-Algebra.<br/>
			///   For directions, from last term of <em>*::to_(inv)matrix_without_translation(...)</em>, and set '?' is 't'</p>
			///<pre>
			///	v = {V,t} * {v1}
			///	    {0,1}   { 0}
			/// 
			///	{T1.V,0} = {V,t} * {T.V,0} (By some argumentation.)
			///	           {0,1}   { 0 ,0}
			///</pre>
			///<p>For position, from last term of <em>*::to_(inv)matrix(...)</em></p>
			///<pre>
			///	v1 = {V,t} * {v}
			///	     {0,1}   {1}
			/// 
			///	{0,T1.t} = {V,t} * {0,T.t}
			///	           {0,1}   {0, 1 }
			///</pre>
			///<p>Combination directions and position(directly place not use multiplication),</p>
			///<pre>
			///	{T1.V,T1.t} = {V,t} * {T.V,T.t}
			///	              {0,1}   { 0 , 1 }
			///</pre>
			///<p>Proof of completion.</p>
		}

		rigidity invtransform(const rigidity& T1) const {
			///<pre>
			///	T = inv(V).rotate( (-t).translate( T1 ) )
			///</pre>
			///<p>Optimization for <em>V</em> is orthogonal vector space</p>
			///<pre>
			///	T = tp(V).rotate( (-t).translate( T1 ) )
			///</pre>
			return {invtransform_without_translation(T1.V[0]), invtransform_without_translation(T1.V[1]),
				invtransform_without_translation(T1.V[2]), invtransform(T1.t)};
		}

		template<typename matrix4> matrix4 to_matrix() const {
			///<p>We want to get a Linear-Algebra</p>
			///<pre>
			///	v1 = M * v
			///</pre>
			///<p>From <em>*::transform(...)</em></p>
			///<pre>
			///	v1 = t.translate( V.rotate( v ) )
			///
			///	v1 = {1,t} * ( {V,0} * {v} )
			///	     {0,1}     {0,1}   {1}
			///
			///	v1 = {V,t} * {v}
			///	     {0,1}   {1}
			///</pre>
			return matrix4
			{ V[0][0], V[1][0], V[2][0], t[0],
			  V[0][1], V[1][1], V[2][1], t[1],
			  V[0][2], V[1][2], V[2][2], t[2],
			     0,       0,       0  ,    1  };
		}

		template<typename matrix4> matrix4 to_invmatrix() const {
			///<p>We want to get a Linear-Algebra</p>
			///<pre>
			///	v = M * v1
			///</pre>
			///<p>From <em>*::invtransform(...)</em>
			/// and optimization for <em>V</em> is orthogonal vector space</p>
			///<pre>
			///	v = tp(V).rotate( (-t).translate( v1 ) )
			///
			///	v = {tp(V),0} * ( {1,-t} * {v1} )
			///	    {  0  ,1}     {0, 1}   { 1}
			///
			///	v = {tp(V),tp(V)*-t} * {v1}
			///	    {  0  ,   1    }   { 1}
			///</pre>
			return matrix4
			{ V[0][0], V[0][1], V[0][2], dot(V[0],-t),
			  V[1][0], V[1][1], V[1][2], dot(V[1],-t),
			  V[2][0], V[2][1], V[2][2], dot(V[2],-t),
			     0,       0,       0,          1      };
		}

		template<typename matrix4> matrix4 to_matrix_without_translation() const {
			///<p>We want to get a Linear-Algebra <em>v1 = M * v</em> from <em>*::transform_without_translation(...)</em></p>
			///<pre>
			///	v1 = V.rotate( v )
			///
			///	v1 = {V,0} * {v}
			///	     {0,1}   {1}
			///</pre>
			return matrix4
			{ V[0][0], V[1][0], V[2][0], 0,
			  V[0][1], V[1][1], V[2][1], 0,
			  V[0][2], V[1][2], V[2][2], 0,
			     0,       0,       0  ,  1  };
			///<p>We can get a more useful form when 'v' is argumented by zero, </p>
			///<pre>
			///	v1 = {V,?} * {v}
			///	     {0,1}   {0}
			///</pre>
		}

		template<typename matrix4> matrix4 to_invmatrix_without_translation() const {
			///<p>We want to get a Linear-Algebra <em>v1 = M * v</em> from <em>*::invtransform_without_translation(...)</em>
			/// and optimization for <em>V</em> is orthogonal vector space</p>
			///<pre>
			///	v = tp(V).rotate( v1 )
			/// 
			/// v = {tp(V),0} * {v1}
			///	    {  0,  1}   { 1}
			///</pre>
			return matrix4
			{ V[0][0], V[0][1], V[0][2], 0,
				V[1][0], V[1][1], V[1][2], 0,
				V[2][0], V[2][1], V[2][2], 0,
					 0,       0,       0  ,  1 };
			///<p>We can get a more useful form when 'v1' is argumented by zero, </p>
			///<pre>
			///	v = {tp(V),?} * {v1}
			///	    {  0,  1}   { 0}
			///</pre>
		}

		template<typename matrix4> matrix4 to_matrix_for_normal() const {
			return to_matrix_without_translation<matrix4>();
		}

		template<typename matrix4> matrix4 to_invmatrix_for_normal() const {
			return to_invmatrix_without_translation<matrix4>();
		}

		template<typename matrix4> static rigidity from_matrix(const matrix4& m) {
			return 
			{ vector3{m[{0,0}], m[{0,1}], m[{0,2}]}, 
			  vector3{m[{1,0}], m[{1,1}], m[{1,2}]},
			  vector3{m[{2,0}], m[{2,1}], m[{2,2}]},
			  vector3{m[{3,0}], m[{3,1}], m[{3,2}]} };
		}

		const vector3 size() const {
			return {1, 1, 1};
		}

		const vector3& direction(const size_t i) const {
			return V[i];
		}

		const vector3& position() const { 
			return t; 
		}

		void scale(const vector3& ds) { abort(); }

		void stretch(const vector3& ds) { abort(); }
		
		void resize(const vector3& s1) { abort(); }

		void rotate(const vector3& n, const scalar arg) {
			//t.translate(                 V.rotate( ... )   )
			//t.translate( (n,arg).rotate( V.rotate( ... ) ) )
			scalar cos_arg = cos(arg), sin_arg = sin(arg);
			V[0] = ::math::geometry::rotate3(V[0], n, cos_arg, sin_arg);
			V[1] = ::math::geometry::rotate3(V[1], n, cos_arg, sin_arg);
			V[2] = ::math::geometry::rotate3(V[2], n, cos_arg, sin_arg);
		}

		void revolve(const vector3& n, const scalar arg) {
			//                t.translate( V.rotate( ... ) )
			//(n,arg).rotate( t.translate( V.rotate( ... ) ) )
			scalar cos_arg = cos(arg), sin_arg = sin(arg);
			V[0] = ::math::geometry::rotate3(V[0], n, cos_arg, sin_arg);
			V[1] = ::math::geometry::rotate3(V[1], n, cos_arg, sin_arg);
			V[2] = ::math::geometry::rotate3(V[2], n, cos_arg, sin_arg);
			t    = ::math::geometry::rotate3(t, n, cos_arg, sin_arg);
		}
		
		void redirect(const size_t i, const vector3& V1_i) {
			//t.translate(  V.rotate( ... ) )
			//t.translate( V1.rotate( ... ) )
			vector3 n = normalize(cross(V[i], V1_i));
			scalar arg = acos(dot(V[i], V1_i));
			this->rotate(n, arg);
		}

		void translate(const vector3& dt) {
			//                t.translate( V.rotate( ... ) )
			//(dt).translate( t.translate( V.rotate( ... ) ) )
			t += dt;
		}
	
		void reposit(const vector3& t1) {
			// t.translate( V.rotate( ... ) )
			//t1.translate( V.rotate( ... ) )
			t = t1;
		}

		void inverse() {
			std::swap(V[0][1], V[1][0]);
			std::swap(V[0][2], V[2][0]);
			std::swap(V[1][2], V[2][1]);
			t = -t;
			t = V[0]*t[0] + V[1]*t[1] + V[2]*t[2];
		}
		
		void fix_orthogonality() {
			///<p>From <em>Gram-Schmidt</em>.</p>
			V[1] -= ::math::geometry::proj(V[1], V[0]);
			V[2] -= ::math::geometry::proj(V[2], V[0]);
			V[2] -= ::math::geometry::proj(V[2], V[1]);///note: because dot(V[0],V[1]) = 0 so proj(x,V[1]) == proj(x - proj(x,V[0]),V[1]).
			V[0] = normalize(V[0]);
			V[1] = normalize(V[1]);
			V[2] = normalize(V[2]);
		}

		rigidity inversion() const {
			///<p>Linear-Algebra from *::invtransform(...)</p>
			///<pre>
			///	v = {inv(V),inv(V)*-t} * {v1}
			///	                         { 1}
			///	V = inv(V)
			///	t = inv(V)*-t
			///</pre>
			rigidity inv;
			inv.V[0] = {V[0][0], V[1][0], V[2][0]};
			inv.V[1] = {V[0][1], V[1][1], V[2][1]};
			inv.V[2] = {V[0][2], V[1][2], V[2][2]};
			inv.t = -t;
			inv.t = inv.V[0]*inv.t[0] + inv.V[1]*inv.t[1] + inv.V[2]*inv.t[2];
			return std::move(inv);
		}

		///<h2>Example</h2>
		///<dl>
		///<dt>Walk</dt> <dd >*.translate( path )</dd>
		///<dt>Look Around</dt> <dd>*.rotate( angular_path )</dd> 
		///<dt>Look At</dt> <dd>*.redirect( ni, center - *.position() )</dd>
		///<dt>Zoom</dt> <dd>*.translate( *.direction(ni) * z )</dd> 
		///<dt>Drag</dt> <dd>*.translate( *.direction(ti[0]) * x + *.direction(ti[1]) * y )</dd> 
		///<dt>Orbit</dt> <dd>
		///	*.translate( (pivot - origin) - (*.position() - origin) )<br />
		///	*.revolve( *.direction(ti[0]), xarg )<br />
		///	*.revolve( *.direction(ti[1]), yarg )<br />
		///	*.translate( (*.position() - origin) - (pivot - origin) ) </dd>
		/// <dt>...</dt>
		///</dl>
	};

	template<typename vector3>
	struct flexibility<vector3, void> : rigidity<vector3, void> {
	public:
		using scalar = std::remove_cvref_t< decltype(std::declval<vector3>()[0]) >;
		using rigidity<vector3, void>::V;
		using rigidity<vector3, void>::t;

		using rigidity<vector3, void>::transform_without_translation;

		vector3 invtransform_without_translation(const vector3& v1) const {
			///<pre>
			///	v = inv(V).scale_and_rotate( v1 )
			/// 
			///	v = inv(V) * v1.
			/// 
			/// v = inv({V[0],V[1],V[2]}) * v1 = inv({V[0],V[1],V[2],?}) * {v1}
			///	                                     { 0  , 0  , 0  ,1}    { 0}
			/// 
			///	v = adj({V[0],V[1],V[2]|)/det({V[0],V[1],V[2]}) * v1
			/// 
			///	    {cross(V[1],V[2])}
			/// v = {cross(V[2],V[0])}/dot(V[0],cross(V[1],V[2])) * v1
			///	    {cross(V[0],V[1])}
			///</pre>
			return vector3{dot(cross(V[1],V[2]),v1), dot(cross(V[2],V[0]),v1), dot(cross(V[0],V[1]),v1)} / dot(V[0],cross(V[1],V[2]));
		}

		using rigidity<vector3, void>::transform;

		vector3 invtransform(const vector3& v1) const {
			///<pre>
			///	v = inv(V).scale_and_rotate( (-t).translate( v1 ) )
			///</pre>
			return invtransform_without_translation(v1 - t);
		}

		vector3 transform_for_normal(const vector3& n) const {
			///<p>From Normal-Vector-Trnasform <em>dot(n,v) = 0 => dot(n*inv(M),M*v) = 0</em>, we have</p>
			///<pre>
			///	n1 = tp(inv(V)) * n
			///</pre>
			///<p>From <em>*::invtransform_without_translation(...)</em> and transpose it, we have</p>
			vector3 n1 = (cross(V[1],V[2])*n[0] + cross(V[2],V[0])*n[1] + cross(V[0],V[1])*n[2]) / dot(V[0],cross(V[1],V[2]));

			///<p>Because <em>V</em> may not be orthogonal, the length of <em>n</em> will changed. So should</p>
			return normalize(n1);
		}

		vector3 invtransform_for_normal(const vector3& n1) const {
			///<pre>
			///	n = inv(tp(inv(V))) * n
			///	n = tp(V) * n1
			///</pre>
			vector3 n = {dot(V[0],n1), dot(V[1],n1), dot(V[2],n1)};

			///<p>Because <em>V</em> may not be orthogonal, the length of <em>n</em> will changed. So should</p>
			return normalize(n);
		}

		flexibility transform(const rigidity<vector3,void>& T1) const {
			///<pre>
			///	T = t.translate( V.scale_and_rotate( T1 ) )
			///</pre>
			return {transform_without_translation(T1.V[0]), transform_without_translation(T1.V[1]),
				transform_without_translation(T1.V[2]), transform(T1.t)};//Avoid one copy and one optimization of the copy. .
		}

#if 0
//Test transform.
#include <iostream>
#include <random>

#include <math/geometry/transform.hpp>
#include <math/mdarray_vector.hpp>
using scalar = float;
using vector3 = math::smdarray<float, 4>;

int main() {
	std::default_random_engine rng{ std::random_device{}() };
	std::uniform_real_distribution<scalar> R{-20, 5};

	vector3 x0 = {R(rng), R(rng), R(rng)};
	vector3 x1 = {R(rng), R(rng), R(rng)};
	vector3 n = {R(rng), R(rng), R(rng)};
		n = normalize(n);
		n = n - math::geometry::proj(n, (x1 - x0));
		n = normalize(n);

	math::geometry::flexibility<vector3> A{ 
		vector3{R(rng),R(rng),R(rng)},{R(rng),R(rng),R(rng)},{R(rng),R(rng),R(rng)},
		{R(rng),R(rng),R(rng)} 
	};

	/// http://www.songho.ca/opengl/gl_normaltransform.html
	std::cout << "is orthogonal" << std::endl;
	std::cout << "dot(x,y) = " << dot(n, x1 - x0) << std::endl;
	std::cout << "dot(x,y) = " << dot(A.transform_for_normal(n), A.transform(x0) - A.transform(x1)) << std::endl;

	std::cout << "x0\tx1\tn" << std::endl;
	std::cout << "origin:   " << x0 << x1 << n << std::endl;
	std::cout << "transform:" << A.transform(x0) << A.transform(x1) << A.transform_for_normal(n) << std::endl;
	std::cout << "origin:   " << A.invtransform(A.transform(x0)) << A.invtransform(A.transform(x1)) << A.invtransform_for_normal(A.transform_for_normal(n)) << std::endl;

	return 0;
}
#endif

		flexibility invtransform(const flexibility& T1) const {
			///<pre>
			///	T = inv(V).scale_and_rotate( (-t).translate( T1 ) )
			///</pre>
			return {invtransform_without_translation(T1.V[0]), invtransform_without_translation(T1.V[1]),
				invtransform_without_translation(T1.V[2]), invtransform(T1.t)};
		}

		template<typename matrix4> matrix4 to_invmatrix_without_translation() const {
			///<p>We want to get a Linear-Algebra <em>v = M * v1</em> from <em>*::invtransform_without_translation(...)</em></p>
			///<pre>
			///	v = inv(V).scale_and_rotate( v1 )
			///
			///	v = {inv(V),0} * {v1}
			///	    {  0   ,1}   { 1}
			///</pre>
			///<p>The <em>inv(V)</em> from <em>*::invtransform_without_translation(...)</em> </p>
			vector3 tpinvV0 = cross(V[1],V[2]);
			scalar det = dot(tpinvV0, V[0]);
				tpinvV0 /= det;
			vector3 tpinvV1 = cross(V[2],V[0])/det;
			vector3 tpinvV2 = cross(V[0],V[1])/det;
			return matrix4
			{ tpinvV0[0], tpinvV0[1], tpinvV0[2], 0,
			  tpinvV1[0], tpinvV1[1], tpinvV1[2], 0,
			  tpinvV2[0], tpinvV2[1], tpinvV2[2], 0,
			     0,          0,          0,       1 };
		}

		template<typename matrix4> matrix4 to_invmatrix() const {
			///<p>We want to get a Linear-Algebra <em>v = M * v1</em> from <em>*::invtransform(...)</em></p>
			///<pre>
			///	v = inv(V).scale_and_rotate( (-t).translate( v1 ) )
			///
			///	v = {inv(V),0} * ( {1,-t} * {v1} )
			///	    {  0   ,1}     {0, 1}   { 1}
			///
			///	v = {inv(V),inv(V)*-t} * {v1}
			///	    {  0   ,    1    }   { 1}
			///</pre>
			///<p>The <em>inv(V)</em> from <em>*::invtransform_without_translation(...)</em> </p>
			vector3 tpinvV0 = cross(V[1],V[2]);
			scalar det = dot(tpinvV0, V[0]);
				tpinvV0 /= det;
			vector3 tpinvV1 = cross(V[2],V[0])/det;
			vector3 tpinvV2 = cross(V[0],V[1])/det;
			return matrix4
			{ tpinvV0[0], tpinvV0[1], tpinvV0[2], dot(tpinvV0,-t),
			  tpinvV1[0], tpinvV1[1], tpinvV1[2], dot(tpinvV1,-t),
			  tpinvV2[0], tpinvV2[1], tpinvV2[2], dot(tpinvV2,-t),
			     0,          0,          0,              1        };
		}

		template<typename matrix4> matrix4 to_matrix_for_normal() const {
			vector3 tpinvV0 = cross(V[1],V[2]);
			scalar det = dot(tpinvV0, V[0]);
				tpinvV0 /= det;
			vector3 tpinvV1 = cross(V[2],V[0])/det;
			vector3 tpinvV2 = cross(V[0],V[1])/det;
			return matrix4
			{ tpinvV0[0], tpinvV1[0], tpinvV2[0], 0,
			  tpinvV0[1], tpinvV1[1], tpinvV2[1], 0,
			  tpinvV0[2], tpinvV1[2], tpinvV2[2], 0,
			     0,          0,          0,       1  };
		}

		template<typename matrix4> matrix4 to_invmatrix_for_normal() const {
			return matrix4
			{ V[0][0], V[0][1], V[0][2], 0,
				V[1][0], V[1][1], V[1][2], 0,
				V[2][0], V[2][1], V[2][2], 0,
					 0,       0,       0  ,  1 };
		}

		const vector3 size() const {
			return {length(V[0]), length(V[1]), length(V[2])};
		}

		const vector3 direction(const size_t i) const {
			return normalize(V[i]);
		}

		const vector3& scaled_direction(const size_t i) const {
			return V[i];
		}

		void scale(const vector3& ds) {
			//t.translate(           V.scale_and_rotate( ... )   )
			//t.translate( ds.scale( V.scale_and_rotate( ... ) ) )
			V[0] *= ds[0];
			V[1] *= ds[1];
			V[2] *= ds[2];
		}

		void stretch(const vector3& ds) {
			//          t.translate( V.scale_and_rotate( ... ) )
			//ds.scale( t.translate( V.scale_and_rotate( ... ) ) )
			///<pre>
			///	     {s[0],    0,    0}
			///	T1 = {   0, s[1],    0} * {V,t}
			///	     {   0,    0, s[2]}
			/// 
			///	     {s[0]*V[0][0], s[0]*V[1][0], s[0]*V[2][0], s[0]*t[0]}
			///	T1 = {s[1]*V[0][1], s[1]*V[1][1], s[1]*V[2][1], s[1]*t[1]}
			///	     {s[2]*V[0][2], s[2]*V[1][2], s[2]*V[2][2], s[2]*t[2]}
			///</pre>
			V[0] *= ds[0];
			V[1] *= ds[1];
			V[2] *= ds[2];
			t *= ds;
		}
		
		void resize(const vector3& s1) {
			V[0] = normalize(V[0]) * s1[0];
			V[1] = normalize(V[1]) * s1[1];
			V[2] = normalize(V[2]) * s1[2];
		}

		void redirect(const size_t i, const vector3& V1_i) {
			//t.translate(  V.scale_and_rotate( ... )  )
			//t.translate( V1.rotate( V.scale( ... ) ) )
			const vector3 V_i = this->direction(i);
			vector3 n = normalize(cross(V_i, V1_i));
			scalar arg = acos(dot(V_i, V1_i));
			this->rotate(n, arg);
		}

#if 0
//Test scale_and_rotate
#include <iostream>
#include <random>

#include <math/geometry/transform.hpp>
#include <math/mdarray_vector.hpp>
using scalar = float;
using vector3 = math::smdarray<float, 4>;

int main() {
	std::default_random_engine rng{ std::random_device{}() };
	std::uniform_real_distribution<scalar> R{-20, 5};

	vector3 x0 = {R(rng), R(rng), R(rng)};
	std::cout << "x0 = " << x0 << std::endl;

	math::geometry::flexibility<vector3> A{ 
		vector3{R(rng),R(rng),R(rng)},{R(rng),R(rng),R(rng)},{R(rng),R(rng),R(rng)},
		{R(rng),R(rng),R(rng)}
	};

	{
		vector3 n = {R(rng), R(rng), R(rng)};
		scalar arg = R(rng);
		///'n' can be no normalized.

		std::cout << "referece rotate x0 by n and arg = " 
			<< math::geometry::rotate(A.transform(x0), n, arg) << std::endl;

		auto B = A; B.revolve(n, arg);
		std::cout << "    test rotate x0 by n and arg = " 
			<< B.transform(x0) << std::endl;
	}
	
	{
		vector3 n = {R(rng), R(rng), R(rng)};
		scalar arg = R(rng);
		///'n' can be no normalized.

		std::cout << "referece revolve x0 by n and arg = " 
			<< A.t + math::geometry::rotate(A.transform_without_translation(x0), n, arg) << std::endl;

		auto B = A; B.rotate(n, arg);
		std::cout << "    test revolve x0 by n and arg = " 
			<< B.transform(x0) << std::endl;
	}

	{
		vector3 n = {R(rng), R(rng), R(rng)};
		n = normalize(n);
		std::cout << "n = " << n << std::endl;

		std::cout << "A before                              = " 
			<< normalize(A.V[0])<<length(A.V[0]) << "," << A.V[1] << A.V[2] << std::endl;

		auto B = A; B.redirect(0, n);
		std::cout << "A after redirect 0st direction by 'n' = " 
			<< normalize(B.V[0])<<length(B.V[0]) << "," << B.V[1] << B.V[2] << std::endl;

		B.redirect(1, {0,1,0});
		std::cout << "recovery to ID 1st step = " 
			<< B.V[0] << B.V[1] << B.V[2] << std::endl;
		B.redirect(2, {0,0,1});
		std::cout << "recovery to ID 2nd step = " 
			<< B.V[0] << B.V[1] << B.V[2] << "\t has some problem." << std::endl;
	}

	return 0;
}
#endif
	};


	/// orthography transformation = remap(INPUT,...) = translate( scale( INPUT ) ).
	template<math::vector vector3>
	struct orthography {
		typedef std::remove_cvref_t<decltype(vector3{}[size_t()])> scalar;
		typedef vector3 vector_type;
		typedef scalar scalar_type;
#if _DEBUG
		vector3 c, r, c1, r1;
		vector3 get_source_center() const { return c; }
		vector3 get_source_halfextents() const { return r; }
		vector3 get_target_center() const { return c1; }
		vector3 get_target_halfextents() const { return r1; }

		explicit orthography(const vector3& c = {0,0,0}, const vector3& r = {1,1,1}, const vector3& c1 = {0,0,0}, const vector3& r1 = {1,1,1})
			: c(c), r(r), c1(c1), r1(r1) {}

		vector3 operator()(const vector3& p) const { return this->transform(p); }

		vector3 transform(const vector3& p) const {
			/// p1 = remap(p, c - r, c + r, c1 - r1, c1 + r1)
			///    = (p - (c - r))/(r*2)*(r1*2) + (c1 - r1)
			///    = (p - (c - r))/r*r1 + (c1 - r1)           (1)
			return (p - c + r)/r*r1 + c1 - r1;
		}

		vector3 invtransform(const vector3& p1) const {
			return (p1 - c1 + r1)/r1*r + c - r;
		}

		template<typename matrix4> explicit operator matrix4() const {
			/// p1 = (p - c + r)/r*r1 + (c1 - r1)
			///    = p/r*r1 - (c - r)/r*r1 + (c1 - r1)
			///    = p*S    +          C                      (2)
			vector3 S = r1/r;
			vector3 C = -(c - r)*S + c1 - r1;
			return matrix4
			{ S[0],  0,    0,   C[0],
				 0,   S[1],  0,   C[1],
				 0,    0,   S[2], C[2],
				 0,    0,    0,    1   };
		}

		template<typename matrix4> explicit orthography(const matrix4& m, const vector3& c1 = {0,0,0}, const vector3& r1 = {1,1,1}) : c(), r(), c1(c1), r1(r1) {
			vector3 S = { m.at(0,0), m.at(1,1), m.at(2,2) };
			vector3 C = { m.at(0,3), m.at(1,3), m.at(2,3) };

			/// S = r1/r
			/**/r = r1/S;                                  ///(3)

			/// C = - (c - r)/r*r1 + (c1 - r1)
			/// (c - r)/r*r1 = (c1 - r1) - C
			/// c = ((c1 - r1) - C)*r/r1 + r                  (4)
			/// 
			///@optimization S = r1/r, r = r1/S
			/// c = ((c1 - r1) - C)/S + r1/S
			/// c = ((c1 - r1) - C + r1)/S
			/**/c = (c1 - C)/S;                            ///(5)
		}

		orthography& inverse() {
			std::swap(c, c1);
			std::swap(r, r1);
			return *this;
		}

		orthography& zoom(const scalar& hscale, const scalar& vscale) {
			r *= vector3{ hscale, vscale, 1 };
			return *this;
		}
#else
		vector3 S, C, c1, r1;
		vector3 get_source_center() const { return (c1 - C)/S; }
		vector3 get_source_halfextents() const { return r1/S; }
		vector3 get_target_center() const { return c1; }
		vector3 get_target_halfextents() const { return r1; }

		explicit orthography(const vector3& c = {0,0,0}, const vector3& r = {1,1,1}, const vector3& c1 = {0,0,0}, const vector3& r1 = {1,1,1})
			: S(r1/r), C(-(c - r)*S + c1 - r1), c1(c1), r1(r1) {}

		vector3 operator()(const vector3& p) const { return this->transform(p); }
		
		vector3 transform(const vector3& p) const { return p*S + C; }
		
		vector3 invtransform(const vector3& p1) const { return (p1 - C)/S; }
		
		template<typename matrix4> explicit operator matrix4() const {
			return matrix4
			{ S[0],  0,    0,   C[0],
				 0,   S[1],  0,   C[1],
				 0,    0,   S[2], C[2],
				 0,    0,    0,    1   };
		}

		template<typename matrix4> explicit orthography(const matrix4& m, const vector3& c1 = {0,0,0}, const vector3& r1 = {1,1,1})
			: S({m.at(0,0), m.at(1,1), m.at(2,2)}), C({m.at(0,3), m.at(1,3), m.at(2,3)}), c1(c1), r1(r1) {}

		orthography& zoom(const scalar& hscale, const scalar& vscale) {
			/// S = r1/r
			/// S1 = r1/(r*scale)
			/// S1 = S/scale
			S /= vector3{ hscale, vscale, 1 };
			return *this;
		}
		
		orthography& inverse() {
			c1 = get_source_center();
			r1 = get_source_halfextents();
			auto S0 = S;
			S = 1/S0; C = -C/S0;
			return *this;
		}
#endif
	};

	/// perspective transformation = remap(scale(INPUT,INPUT[z]),...) = translate( scale( INPUT, INPUT[z] ) ).
	template<math::vector vector3>
	struct perspective {
		typedef std::remove_cvref_t<decltype(vector3{}[size_t()])> scalar;
		typedef vector3 vector_type;
		typedef scalar scalar_type;

		struct affine_transformation {
			vector3 SS, C;

			template<typename matrix4> explicit operator matrix4() const { 
				return matrix4{
					SS[0],0,    C[0], 0, 
					0,    SS[1],C[1], 0, 
					0,    0,    SS[2],C[2], 
					0,    0,    1,    0 };
			}

			vector3 operator()(const vector3& p) const { return this->transform(p); }

			vector3 transform(const vector3& p) const { 
				return (p*SS + vector3{p[2],p[2],1}*C)/p[2]; // formula (6)
			}

			vector3 invtransform(const vector3& p1) const {
				/// p1[2] = (p[2]*SS[2] + C[2])/p[2]
				/// p1[2] = SS[2] + C[2]/p[2]
				scalar pz = C[2]/(p1[2] - SS[2]);
				return (p1*pz - vector3{pz,pz,1}*C)/SS;
			}
		};

#if 0
		vector3 nc, nr, c1, r1;
		vector3 get_source_near_center() const { return nc;  }
		vector3 get_source_near_halfextents() const { return nr; }
		scalar get_source_back() const { return nc[2]; }
		scalar get_source_front() const { return nc[2]+nr[2]; }
		vector3 get_target_center() const { return c1; }
		vector3 get_target_halfextents() const { return r1; }
	
		explicit perspective(const vector3& nc = {0,0,0}, const vector3& nr = {1,1,1}, const vector3& c1 = {0,0,0}, const vector3& r1 = {1,1,1})
			: nc(nc), nr(nr), c1(c1), r1(r1) {}
	
		vector3 transform(const vector3& p) const {
#if 0
			///		                     far
			///		                      .
			///		                      . 
			///		                      *p
			///		                    / |
			///		                  /   |
			///		        near    /     |
			///		          .   /       |
			///		          . /         |
			///		          *np         |
			///		        / |           |
			///		      /   |           |
			///		    /eye  |nc         |
			///		   *------*-----------+ ...........  * nc+{0,0,nr[2]}
			///		          |<-- -- -- --nr[2]-- -- -->|
			/// 
			/// Transform (p[x],p[y]) onto "near plane" by p[z], use Similar-Triangles.
			///		         np[x|y]     np[z]
			///		solve { --------- = ------- }, np[z] = nc[z].                    (1)
			///		         p[x|y]      p[z]
			vector3 np = nc[2]/p[2] * p;

			/// Remap np from [nc - nr*{1,1,0}, nc + nr] to [c1 - r1, c1 + r1].
			///		p1 = remap(np, nc - nr*{1,1,0}, nc + nr, c1 - r1, c1 + r1)
			///		   = (np - (nc - nr*{1,1,0}))/(nr*{2,2,1})*(r1*2) + (c1 - r1)
			///		   = (np - (nc - nr*{1,1,0}))/nr*r1*{1,1,2} + (c1 - r1).         (2)
			return (vector3{np[0],np[1],p[2]} - nc + nr*vector3{1,1,0})/nr*r1*vector3{1,1,2} + c1 - r1;
#else
			///@optimization 
			///		p1 = formula (2)
			///		   = (p*{s,s,1} - (nc - nr*{1,1,0}))/nr*r1*{1,1,2} + (c1 - r1), s = nc[2]/p[2].      (3)
			scalar s = nc[2]/p[2];
			return (p*vector3{s,s,1} - nc + nr*vector3{1,1,0})/nr*r1*vector3{1,1,2} + c1 - r1;
#endif
		}

		vector3 invtransform(const vector3& p1) const {
			vector3 np = (p1 - c1 + r1)*nr/r1/vector3{1,1,2} + nc - nr*vector3{1,1,0};
			scalar s = nc[2]/np[2];
			return np/vector3{s,s,1};
		}

		void zoom(scalar hscale, scalar vscale) {
			nr *= vector3{hscale,vscale,1};
		}
		
		template<typename matrix4>
		explicit operator matrix4() const {
			/// From formula (3).
			/// 
			///		p1 = (p*{s,s,1} - (nc - nr*{1,1,0}))/nr*r1*{1,1,2} + (c1 - r1), s = nc[2]/p[2]
			///		   = p*{s,s,1}/nr*r1*{1,1,2} - (nc - nr*{1,1,0})/nr*r1*{1,1,2} + (c1 - r1)
			///		   = p/p[2]  *  SS           +                   C                                   (4)
			/// 
			/// Not linear, but we can get form of affine transform, "p1 = m*{p,1}/(m*{p,1}[3])".
			/// We set 'm*{p,1}[3]' is 'p[z]' to remove nolinear,
			///  
			///		    { ?, ?, ?, 0 }
			///		m = { ?, ?, ?, 0 }, assert( (m*{p,1})[3] == p[2] ).
			///		    { ?, ?, ?, 0 }
			///		    { 0, 0, 1, 0 }
			/// 
			/// Then we derivate "f(p)" in "p1 = f(p)/p[2]",
			/// 
			///		p1 = ( p*SS + p[2]*C )/p[2].
			/// 
			///		f(p) = p*{nc[2],nc[2],p[2]}/nr*r1*{1,1,2} + p[2]*( - (nc - nr*{1,1,0})/nr*r1*{1,1,2} + (c1 - r1) )
			///		     = p*SS                               + p[2]*C                                   (5)
			/// 
			/// Still component'z' in 'SS' is nonlinear, we need a new linear-formula for 'SS[z]', and
			/// component'z' in p[z]*C is invalid, we need a new multiplier for 'C[z]'.
			///	
			///		f(p) = p*{SS[0],SS[1],?} + {p[2],p[2],1}*{C[0],C[1],?}
			/// 
			///		       {SS[0],     , C[0],    }
			///		     = {     ,SS[1], C[1],    } * {p,1}.                                             (6)
			///		       {           ,SS[2],C[2]}
			///		       {           , 1   , 0  }
			vector3 S = r1/nr;
			vector3 SS = nc[2]*S;
			vector3 C = -(nc - nr)*S + c1 - r1;

			/// From formula (6), we need this form "p1[z] = (p[z]*SS[2] + C[2])/p[z] = SS[2] + C[2]/p[2]",
			/// subsititute p[z] below [nc[z], nc[z]+nr[z]], p1[z] below [c1[z] - r1[z], c1[z] + r1[z]],
			/// 
			///		{ { SS + C/nc[z]         = c1[z] - r1[z] },
			///		  { SS + C/(nc[z]+nr[z]) = c1[z] + r1[z] } }.
			/// 
			/// 'C' in equations,
			///		                                        C
			///		                             S + --------------- = c1[z] + r1[z] ;
			///		                                  nc[z] + nr[z]
			///		                          C             C
			///		       c1[z] - r1[z] - ------- + --------------- = c1[z] + r1[z]     :S + C/nc[z] = c1[z] - r1[z] => S = c1[z] - r1[z] -  C/nc[z] ;
			///		                        nc[z]     nc[z] + nr[z]
			///		                          C             C
			///		                      - ------- + --------------- = r1[z]*2 ;
			///		                        nc[z]     nc[z] + nr[z]
			///		    -C*(nc[z] + nr[z])     nc[z]*C
			///		----------------------- + ----------------------- = r1[z]*2 ;
			///		 nc[z]*(nc[z] + nr[z])     nc[z]*(nc[z] + nr[z])
			///		                                -C*nr[z]
			///		                          ----------------------- = r1[z]*2 ;
			///		                           nc[z]*(nc[z] + nr[z])
			///		                                                       r1[z]*2 * nc[z]*(nc[z] + nr[z])
			///		                                                C = - -------------------------------- .
			///		                                                                  nr[z]
			///@example r1[z] = 0.5, C = - 1 * nc[z]*(nc[z] + nr[z])/nr[z] = - near*far/Zrange.
			/// 
			/// 'S' in equations,
			///		            C
			///		S + ----------------- = c1[z] + r1[z] ;
			///		      nc[z] + nr[z]
			///		     r1[z]*2 * nc[z]
			///		S - ----------------- = c1[z] + r1[z] ;
			///		          nr[z]
			///		                                         r1[z]*2 * nc[z]
			///		                    S = c1[z] + r1[z] + ----------------- .
			///		                                              nr[z]
			///@example c1[z] = 0.5 and r1[z] = 0.5, S = 1 + nc[z]/nr[z] = (nr[z] + nc[z])/nr[z] = far/Zrange.
			scalar rr = r1[2]*2/nr[2];
			C[2] = -rr * nc[2]*(nc[2] + nr[2]);
			SS[2] = c1[2] + r1[2] + rr*nc[2];

			return matrix4
			{SS[0],  0,   C[0], 0,
				 0,  SS[1], C[1], 0,
				 0,    0,  SS[2],C[2],
				 0,    0,    1,   0  }; 
		}

		template<typename matrix4>
		explicit perspective(const matrix4& m, const vector3& c1 = {0,0,0}, const vector3& r1 = {1,1,1}) : nc(), nr(), c1(c1), r1(r1) {
			vector3 SS = {m.at(0,0), m.at(1,1), m.at(2,2)};
			vector3 C = {m.at(0,2), m.at(1,2), m.at(2,3)};

			/// nc[x|y],nr[x|y] dependent nc[2],nr[2], their order can be arbitraty.
			///		SS + C/nc[z] = c1[z] - r1[z]
			///		       nc[z] = C/(c1[z] - r1[z] - SS).
			/// 
			///		SS + C/(nc[z] + nr[z]) = c1[z] + r1[z]
			///		        nc[z] + nr[z]  = C/(c1[z] + r1[z] - SS)
			///		                nr[z]  = C/(c1[z] + r1[z] - SS) - nc[z].
			scalar ncZ = C[2]/(c1[2] - r1[2] - SS[2]);
			scalar nrZ = C[2]/(c1[2] + r1[2] - SS[2]) - ncZ;
		
			/// Solve "SS = nc[z]/nr*r1",
			///		nr = nc[z]*r1/SS
			nr = ncZ*r1/SS; nr[2] = nrZ;
			/// Solve "C = - (nc - nr)/nr*r1 + (c1 - r1)"
			///		(nc - nr)/nr*r1 = (c1 - r1) - C
			///		nc = ((c1 - r1) - C)/r1*nr + nr
			nc = (c1 - r1 - C)/r1*nr + nr; nc[2] = ncZ;
		}

		affine_transformation get_affine() const {
			vector3 S = r1/nr;
			vector3 SS = nc[2]*S;
			vector3 C = (-(nc - nr)*S + c1 - r1);
			scalar rr = r1[2]*2/nr[2];
			C[2] = -rr * nc[2]*(nc[2] + nr[2]);
			SS[2] = c1[2] + r1[2] + rr*nc[2];
			return affine_transformation{ SS, C, this };
		}
#else
		vector3 Ss, C, c1, r1;
		vector3 get_target_center() const { return c1; }
		vector3 get_target_halfextents() const { return r1; }

		explicit perspective(const vector3& nc = {0,0,0}, const vector3& nr = {1,1,1}, const vector3& c1 = {0,0,0}, const vector3& r1 = {1,1,1}) : Ss(), C(), c1(c1), r1(r1) {
			/// from formula (4)
			///		p1 = p*{s,s,1}/nr*r1*{1,1,2} - (nc - nr*{1,1,0})/nr*r1*{1,1,2} + (c1 - r1)
			///		   = p/p[2]  *  SS           +                   C
			///		   = p/{p[2],p[2],1}*nc[2]/nr*r1*{1,1,2} - (nc - nr*{1,1,0})/nr*r1*{1,1,2} + (c1 - r1)
			///		   = p/{p[2],p[2],1}  *     Ss           +                   C                             (7)
			vector3 S = r1/nr*vector3{1,1,2};
			Ss = S*vector3{nc[2],nc[2],1};
			C = -(nc - nr*vector3{1,1,0})*S + c1 - r1;
		}

		vector3 transform(const vector3& p) const {
			return p/vector3{p[2],p[2],1}*Ss + C;// formula (7)
		}

		vector3 invtransform(const vector3& p1) const {
			vector3 p = (p1 - C)/Ss;
			return p*vector3{p[2],p[2],1};
		}

		void zoom(scalar hscale, scalar vscale) {
			/// S = r1/nr*{1,1,2}
			/// S1 = r1/(nr*scale)*{1,1,2}
			/// S1 = S/scale;
			Ss /= vector3{hscale,vscale,1};
		}

		scalar get_source_back() const {
			/// S[z] = r1/nr*2
			/// C[z] = -nc[z]*S + c1 - r1
			/// (C[z] - c1 + r1)/S[z] = -nc[z]
			/// (C[z] - c1 + r1)/Ss[z] = -nc[z]
			return -(C[2] - c1[2] + r1[2])/Ss[2];
		}

		scalar get_source_front() const { 
			/// far = -(C[2] - c1[2] + r1[2])/S[2] + r1[2]/S[2]*2
			///     = -(C[2] - c1[2] + r1[2] - r1[2]*2)/S[2]
			///     = -(C[2] - c1[2] - r1[2])/Ss[2]
			return -(C[2] - c1[2] - r1[2])/Ss[2];
		}

		vector3 get_source_near_halfextents() const { 
			///        S           = r1/nr*{1,1,2}
			/// Ss/{nc[z],nc[z],1} = r1/nr*{1,1,2}
			/// nr = r1/Ss*{nc[z],nc[z],1}*{1,1,2}
			scalar ncz = this->get_source_back();
			return r1/Ss*vector3{ncz,ncz,2}; 
		}
		
		vector3 get_source_near_center() const { 
			/// C = -(nc - nr*{1,1,0})*S + c1 - r1
			/// (C - c1 + r1)/S - nr*{1,1,0} = -nc
			/// 
			///@optimization S = r1/nr*{1,1,2}, nr = r1/S*{1,1,2}
			/// (C - c1 + r1)/S - r1/S*{1,1,2}*{1,1,0} = -nc
			/// (C - c1 + r1 - r1*{1,1,0})/S = -nc
			/// -(C - c1 + r1*{0,0,1})/S = nc
			scalar ncz = this->get_source_back();
			return  -(C - c1 + r1*vector3{0,0,1})/Ss*vector3{ncz,ncz,1};
		}

		template<typename matrix4>
		explicit operator matrix4() const {
			scalar ncZ = this->get_source_near_center()[2];
			scalar nrZ = this->get_source_near_halfextents()[2];
			
			vector3 SS = this->Ss;
			vector3 C = this->C;
			scalar rr = r1[2]*2/nrZ;
			C[2] = -rr * ncZ*(ncZ + nrZ);
			SS[2] = c1[2] + r1[2] + rr*ncZ;
			return matrix4
			{SS[0],  0,   C[0], 0,
				 0,  SS[1], C[1], 0,
				 0,    0,  SS[2],C[2],
				 0,    0,    1,   0  }; 
		}

		template<typename matrix4>
		explicit perspective(const matrix4& m, const vector3& c1 = {0,0,0}, const vector3& r1 = {1,1,1}) : Ss(), C(), c1(c1), r1(r1) {
			vector3 SS = {m.at(0,0), m.at(1,1), m.at(2,2)};
			vector3 C = {m.at(0,2), m.at(1,2), m.at(2,3)};

			scalar ncZ = C[2]/(c1[2] - r1[2] - SS[2]);
			scalar nrZ = C[2]/(c1[2] + r1[2] - SS[2]) - ncZ;
		
			scalar Sz = r1[2]/nrZ*2;
			this->Ss = SS;
			this->Ss[2] = Sz;
			this->C = C;
			this->C[2] = -ncZ*Sz + c1[2] - r1[2];
		}

		affine_transformation get_affine() const {
			scalar ncZ = this->get_source_near_center()[2];
			scalar nrZ = this->get_source_near_halfextents()[2];
			
			vector3 SS = this->Ss;
			vector3 C = this->C;
			scalar rr = r1[2]*2/nrZ;
			C[2] = -rr * ncZ*(ncZ + nrZ);
			SS[2] = c1[2] + r1[2] + rr*ncZ;
			return affine_transformation{ SS, C };
		}
#endif

		explicit perspective(scalar fov_h, scalar fov_v, scalar n = 0.1f, scalar f = 1000.0f, const vector3& c1 = {0,0,0}, const vector3& r1 = {1,1,1})
			: perspective({0,0,n}, {tan(fov_h/2)*n, tan(fov_v/2)*n, f-n}, c1, r1) {}

		/// tan(fov_h/2)*2 = h
		/// w = h*aspect
		/// fov_v = atan(w/2)*2
		/*explicit perspective(scalar fov_h, scalar aspect, scalar n = 0.1f, scalar f = 1000.0f, const vector3& c1 = {0,0,0}, const vector3& r1 = {1,1,1})
			: perspective({0,0,n}, {tan(fov_h/2)*n, tan(fov_v/2)*n, f-n}, c1, r1) {}*/

		vector3 operator()(const vector3& p) const { return this->transform(p); }
	};

#if 0
	template<typename Vector3, typename Array3>
	struct ViewfieldSpan {
		using scalar_type = std::remove_cvref_t<decltype(Vector3()[0])>;
		using vector_type = Vector3;
		scalar_type viewfield[2];
		scalar_type near;
		scalar_type far;
		Vector3 center;
		Array3 halfextents;

		Vector3 operator()(Vector3 p) const {
			return PerspectiveSpan<Vector3,Array3>(viewfield, near, far, center, halfextents)(p);
		}

		template<typename Matrix4x4>
		explicit operator Matrix4x4() const {
			return (Matrix4x4)(PerspectiveSpan<Vector3,Array3>(viewfield, near, far, center, halfextents));
		}

		template<typename Matrix4x4>
		explicit ViewfieldSpan(Matrix4x4 m, Vector3 c1 = {0,0,0}, Array3 r1 = {1,1,1}) {
			auto perspective = PerspectiveSpan<Vector3,Array3>(m, c1, r1);
			this->viewfield[0] = atan(perspective.boundary_extents[0]/2 / perspective.boundary_nearcenter[2]) * 2;
			this->viewfield[1] = atan(perspective.boundary_extents[1]/2 / perspective.boundary_nearcenter[2]) * 2;
			this->near = perspective.boundary_nearcenter[2];
			this->far = perspective.boundary_nearcenter[2] + perspective.boundary_extents[2];
			this->center = perspective.center;
			this->halfextents = perspective.halfextents;
		}

		void zoom(scalar_type hscale, scalar_type vscale) {
			scalar_type x0 = tan(viewfield[0]/2)*near * 2;
			scalar_type y0 = tan(viewfield[1]/2)*near * 2;
			viewfield[0] = atan(x0*hscale/2 / near)*2;
			viewfield[1] = atan(y0*vscale/2 / near)*2;
		}

		scalar_type& operator[](size_t i) {
			return viewfield[i];
		}

		const scalar_type& operator[](size_t i) const {
			return viewfield[i];
		}

		ViewfieldSpan() = default;

		ViewfieldSpan(scalar_type hfov, scalar_type vfov, scalar_type n, scalar_type f, Vector3 c1 = {0,0,0}, Array3 r1 = {1,1,1})
			: viewfield{hfov,vfov}, near(n), far(f), center(c1), halfextents(r1) {}
	};
#endif

	/* rotation_matrix:{ 
		double pitch = asin(A.at(2,1));
		double yaw   = atan2(-A.at(2,0), A.at(2,2)); if(yaw < 0) yaw += pi * 2;
		double roll  = atan2(-A.at(0,1), A.at(1,1)); if(roll < 0) roll += pi * 2;
		double angle = acos( (A.at(0,0)+A.at(1,1)+A.at(2,2) - 1) / 2 );
		Vector3<double> toque = { A.at(2,1)-A.at(1,2), A.at(0,2)-A.at(2,0), A.at(1,0)-A.at(0,1) } / (2 * sin(angle));
	} */

	/* translation_matrix:{
		double x = A.at(0,3);
		double y = A.at(1,3);
		double z = A.at(2,3);
	} */

	/* scaling_matrix:{
		double sx = A.at(0,0);
		double sy = A.at(1,1);
		double sz = A.at(2,2);
	} */
	// Note:((value-lower)/domain)*2-1 = (value*2-center*2)/domain


	/// A view frustum is a pyramid that is truncated by a near and a far plane
	/// (which are parallel), making the volume finite. In fact, it becomes a polyhedron.
	///                                                       <<Real Time Rendering>>
	template<math::vector Vector4>
	struct viewfrustum {
		Vector4 equations[6];

		viewfrustum() = default;

		template<typename Matrix4x4, typename Vector3>
		viewfrustum(Matrix4x4 m, Vector3 c1 = {0,0,0}, Vector3 r1 = {1,1,1}, bool inner = true) {
			/// We known transform: 
			///		             {dot(MVP[0],v),
			///		v1 = MVP*v =  dot(MVP[1],v),
			///		              dot(MVP[2],v),
			///		              dot(MVP[3],v)} ,
			/// boundary condition:
			///		c1-r1 <= v1/v1.w <= c1+r1 .
			/// We target is extract boundaries, and
			/// We want boundary is form of plane. it's should similar as 'dot(v,N)=D' | 'dot(v,N)+D=0' | 'dot(v,N)>=D, positive to the plane'.
			typedef std::remove_cvref_t<decltype(m.row(0))> Matrix1x4;
			static_assert(sizeof(Matrix1x4) == sizeof(Vector4));
			if (inner) {
				/// We derive from boundary condition, lower boundary:
				///		      c1 - r1  <= v1/v1.w
				///		v1.w*(c1 - r1) <= v1
				///		             0 <= v1 - v1.w*(c1-r1)
				///		{{ 0 <= dot(MVP[0] - MVP[3]*(c1 - r1), v) },
				///		 { 0 <= dot(MVP[1] - MVP[3]*(c1 - r1), v) }, :0 <= v1.x - v1.w*(c1 - r1); 0 <= dot(MVP[0],v) - dot(MVP[3]*(c1 - r1),v); 0 <= dot(MVP[0] - MVP[3]*(c1-r1), v)
				///		 { 0 <= dot(MVP[2] - MVP[3]*(c1 - r1), v) }} .
				/// 
				/// upper boundary:
				///		v1/v1.w <= c1 + r1
				///		     v1 <= (c1 + r1)*v1.w
				///		     0  <= (c1 + r1)*v1.w - v1
				///		{{ 0 <= dot(MVP[3]*(c1 + r1) - MVP[0], v) },
				///		 { 0 <= dot(MVP[3]*(c1 + r1) - MVP[1], v) },
				///		 { 0 <= dot(MVP[3]*(c1 + r1) - MVP[2], v) }} .
				reinterpret_cast<Matrix1x4&>(equations[0]) = m.row(0) - m.row(3)*(c1[0] - r1[0]);
				reinterpret_cast<Matrix1x4&>(equations[1]) = m.row(1) - m.row(3)*(c1[1] - r1[1]);
				reinterpret_cast<Matrix1x4&>(equations[2]) = m.row(2) - m.row(3)*(c1[2] - r1[2]);
				reinterpret_cast<Matrix1x4&>(equations[3]) = m.row(3)*(c1[0] + r1[0]) - m.row(0);
				reinterpret_cast<Matrix1x4&>(equations[4]) = m.row(3)*(c1[1] + r1[1]) - m.row(1);
				reinterpret_cast<Matrix1x4&>(equations[5]) = m.row(3)*(c1[2] + r1[2]) - m.row(2);
			} else {
				/// lower boundary:
				///		      c1 - r1  > v1/v1.w
				///		v1.w*(c1 - r1) > v1
				///		v1.w*(c1 - r1) - v1 > 0
				///		...
				/// 
				/// upper boundary:
				///		v1/v1.w > c1 + r1
				///		v1 > v1.w*(c1 + r1)
				///		v1 - v1.w*(c1 + r1) > 0
				reinterpret_cast<Matrix1x4&>(equations[0]) = m.row(3)*(c1[0] - r1[0]) - m.row(0);
				reinterpret_cast<Matrix1x4&>(equations[1]) = m.row(3)*(c1[1] - r1[1]) - m.row(1);
				reinterpret_cast<Matrix1x4&>(equations[2]) = m.row(3)*(c1[2] - r1[2]) - m.row(2);
				reinterpret_cast<Matrix1x4&>(equations[3]) = m.row(0) - m.row(3)*(c1[0] + r1[0]);
				reinterpret_cast<Matrix1x4&>(equations[4]) = m.row(1) - m.row(3)*(c1[1] + r1[1]);
				reinterpret_cast<Matrix1x4&>(equations[5]) = m.row(2) - m.row(3)*(c1[2] + r1[2]);
			}
		}

		void normalize() {
			for (size_t i = 0; i != 6; ++i) {
				auto& eq = equations[i];
				auto len = sqrt(eq[0]*eq[0] + eq[1]*eq[1] + eq[2]*eq[2]);
				eq /= len;
			}
		}

		template<typename Plane>
		Plane plane(size_t i) const {
			return Plane{{equations[i][0],equations[i][1],equations[i][2]}, -equations[i][3]};
		}
	};

	template<typename Vector4, typename Vector3, typename Scalar>
	float viewfrustum7sphere(const Vector4* eq, const Vector3& c, const Scalar& r) {
		Vector4 c4 = {c[0],c[1],c[2],1};
		Scalar d0 = dot(eq[0], c4);
		if (-r <= d0) {
			Scalar d1 = dot(eq[1], c4);
			if (-r <= d1) {
				Scalar d2 = dot(eq[2], c4);
				if (-r <= d2) {
					Scalar d3 = dot(eq[3], c4);
					if (-r <= d3) {
						Scalar d4 = dot(eq[4], c4);
						if (-r <= d4) {
							Scalar d5 = dot(eq[5], c4);
							if (-r <= d5) {
								if (r <= d0 && r <= d1 && r <= d2 && r <= d3 && r <= d4 && r <= d5) {
									return 1.0f;
								} else {
									return 0.5f;
								}
							}
						}
					}
				}
			}
		}

		return -1.0f;
	}

	template<typename Vector4, typename Vector3, typename Scalar = std::remove_cvref_t<decltype(Vector3()[0])>>
	float viewfrustum7aabox(const Vector4* eq, const Vector3& c, const Vector3& r) {
		Scalar e0 = dot( abs(reinterpret_cast<const Vector3&>(eq[0])), r );
		Scalar s0 = dot( reinterpret_cast<const Vector3&>(eq[0]), c ) + eq[0][3];
		if (-e0 <= s0) {
			Scalar e1 = dot( abs(reinterpret_cast<const Vector3&>(eq[1])), r );
			Scalar s1 = dot( reinterpret_cast<const Vector3&>(eq[1]), c ) + eq[1][3];
			if (-e1 <= s1) {
				Scalar e2 = dot( abs(reinterpret_cast<const Vector3&>(eq[2])), r );
				Scalar s2 = dot( reinterpret_cast<const Vector3&>(eq[2]), c ) + eq[2][3];
				if (-e2 <= s2) {
					Scalar e3 = dot( abs(reinterpret_cast<const Vector3&>(eq[3])), r );
					Scalar s3 = dot( reinterpret_cast<const Vector3&>(eq[3]), c ) + eq[3][3];
					if (-e3 <= s3) {
						Scalar e4 = dot( abs(reinterpret_cast<const Vector3&>(eq[4])), r );
						Scalar s4 = dot( reinterpret_cast<const Vector3&>(eq[4]), c ) + eq[4][3];
						if (-e4 <= s4) {
							Scalar e5 = dot( abs(reinterpret_cast<const Vector3&>(eq[5])), r );
							Scalar s5 = dot( reinterpret_cast<const Vector3&>(eq[5]), c ) + eq[5][3];
							if (-e5 <= s5) {
								if (e0 <= s0 && e1 <= s1 && e2 <= s2 && e3 <= s3 && e4 <= s4 && e5 <= s5) {
									return 1.0f;
								} else {
									return 0.5f;
								}
							}
						}
					}
				}
			}
		}
	
		return -1.0f;
	}
} }// end of namespace math::geometry


//using namespace::calculation;
//
//matrix4_t X = eye<float,4,4>();
//auto c = mscale(X, vector3_t{ 2,2,2 });
//std::cout << mscale(X,
//	vector3_t{ 2,2,2 } ) << std::endl;
//std::cout << mrotate(mscale(X,
//	vector3_t{ 2,2,2 } ),
//	vector3_t{ 1,0,0 }, 0.5f ) << std::endl;
//std::cout << mtranslate( mrotate( mscale( X, 
//	vector3_t{2,2,2} ), 
//	vector3_t{1,0,0},0.5f ), 
//	vector3_t{0,0,5} ) << std::endl;
//
//matrix4_t A = eye<float,4,4>();
//vector3_t x = {1,0,5};
//A = mtranslate(A, x);
//std::cout <<"A:{\n"<<A<<"\n}"<< std::endl;
//std::cout <<"mirror(A,[1,0,0]):{\n"<<mreflect(A,vector3_t{1,0,0})<<"\n}"<< std::endl;
//std::cout <<"x:{"<<x<<"}"<< std::endl;
//std::cout <<"mirror(x,[1,0,0]):{"<<reflect(x,vector3_t{1,0,0})<<"}"<< std::endl;
//std::cout <<"mirror(A,[1,0,0])*x:{"<<mreflect(A,vector3_t{1,0,0})*vector4_t{0,0,0,1}<<"}"<< std::endl;


//vector3_t axis = { (float)rand(), (float)rand(), (float)rand() };
//float angle = (float)rand();
//axis = normalize(axis);
//
//vector3_t axis2 = { (float)rand(), (float)rand(), (float)rand() };
//float angle2 = (float)rand();
//axis2 = normalize(axis2);
//
//std::cout << "quaternion:{\n" << quaternion_cast<matrix4_t>( 
//	qrotate( polar(axis,angle), axis2,angle2 )
//	) << "\n}" << std::endl;
//std::cout << "correct:{\n" << 
//	mrotate( mrotate(eye<float,4,4>(),axis,angle), axis2,angle2 )
//	<< "\n}" << std::endl;


#ifdef _MATH_DOUBLE_ARRAY_HPP_
	template<typename T> inline
	math::affinet<T,3> scale(const math::affinet<T,3>& X, const math::vector3<T>& ds) {
		auto [X00,X01,X02,X03,
			X10,X11,X12,X13,
			X20,X21,X22,X23] = X._My_data;
		return math::affinet<T,3>
		/* {ds[0],  0  ,  0  ,0,   {X00,X01,X02,X03,    */ { ds[0]*X00,ds[0]*X01,ds[0]*X02,ds[0]*X03,
		/*    0  ,ds[1],  0  ,0, *  X10,X11,X12,X13,  = */   ds[1]*X10,ds[1]*X11,ds[1]*X12,ds[1]*X13,
		/*    0  ,  0  ,ds[2],0,    X20,X21,X22,X23,    */   ds[2]*X20,ds[2]*X21,ds[2]*X22,ds[2]*X23
		/*    0  ,  0  ,  0  ,1 }    0 , 0 , 0 , 1  }   */                                           };
	}

	template<typename T> inline
	math::matrix<T,4,4> scale(const math::matrix<T,4,4>& X, const math::vector3<T>& ds) {
		auto [X00,X01,X02,X03,
			X10,X11,X12,X13,
			X20,X21,X22,X23,
			X30,X31,X32,X33] = X._My_data;
		return math::matrix<T,4,4>
		/* {ds[0],  0  ,  0  ,0,   {X00,X01,X02,X03,    */ { ds[0]*X00,ds[0]*X01,ds[0]*X02,ds[0]*X03,
		/*    0  ,ds[1],  0  ,0, *  X10,X11,X12,X13,  = */   ds[1]*X10,ds[1]*X11,ds[1]*X12,ds[1]*X13,
		/*    0  ,  0  ,ds[2],0,    X20,X21,X22,X23,    */   ds[2]*X20,ds[2]*X21,ds[2]*X22,ds[2]*X23,
		/*    0  ,  0  ,  0  ,1 }   X30,X31,X32,X33 }   */         X30,      X31,      X32,      X33 };
	}

	template<typename T> inline
	math::affinet<T,3> pre_scale(const math::affinet<T,3>& X, const math::vector3<T>& ds) {
		auto [X00,X01,X02,X03,
			X10,X11,X12,X13,
			X20,X21,X22,X23] = X._My_data;
		return math::affinet<T,3>
		/*{X00,X01,X02,X03,   {ds[0], 0  ,  0  ,0,    */ { X00*ds[0],X01*ds[1],X02*ds[2],X03,
		/* X10,X11,X12,X13, *   0  ,ds[1],  0  ,0,  = */   X10*ds[0],X11*ds[1],X12*ds[2],X13,
		/* X20,X21,X22,X23,     0  ,  0  ,ds[2],0,    */   X20*ds[0],X21*ds[1],X22*ds[2],X23
		/*  0 , 0 , 0 , 1 }     0  ,  0  ,  0  ,1 }   */                                     };
	}

	template<typename T> inline
	math::matrix<T,4,4> pre_scale(const math::matrix<T,4,4>& X, const math::vector3<T>& ds) {
		auto [X00,X01,X02,X03,
			X10,X11,X12,X13,
			X20,X21,X22,X23,
			X30,X31,X32,X33] = X._My_data;
		return math::matrix<T,4,4>
		/*{X00,X01,X02,X03,   {ds[0], 0  ,  0  ,0,    */ { X00*ds[0],X01*ds[1],X02*ds[2],X03,
		/* X10,X11,X12,X13, *   0  ,ds[1],  0  ,0,  = */   X10*ds[0],X11*ds[1],X12*ds[2],X13,
		/* X20,X21,X22,X23,     0  ,  0  ,ds[2],0,    */   X20*ds[0],X21*ds[1],X22*ds[2],X23,
		/* X30,X31,X32,X33}     0  ,  0  ,  0  ,1 }   */   X30*ds[0],X31*ds[1],X32*ds[2],X33 };
	}

	/// [ 1 ,    ,   ]   [X00,X01,X02]   [X00,         X01,         X02         ]
	/// [   ,  1 ,   ] * [X10,X11,X12] = [X10,         X11,         X12         ]
	/// [   ,ds21, 1 ]   [X20,X21,X22]   [X20+ds21*X10,X21+ds21*X11,X22+ds21*X12]
	template<typename Matrix4, typename Scalar>
	Matrix4 mshear(const Matrix4& X, size_t i, size_t j, Scalar ds) {
		Matrix4 Y = X;
		Y.row(i) += X.row(j)*ds;
		return Y;
	}

	template<typename matrix4, typename vector3>
	matrix4 mtranslate(const matrix4& X, const vector3& dx) {
		/// [1, , ,dx[0]]   [X00,X01,X02,X03]   [X00,X01,X02,X03+X33*dx[0]]
		/// [ ,1, ,dx[1]] * [X10,X11,X12,X13] = [X10,X11,X12,X13+X33*dx[1]]
		/// [ , ,1,dx[2]]   [X20,X21,X22,X23]   [X20,X21,X22,X23+X33*dx[2]]
		/// [ , , ,  1  ]   [X30,X31,X32,X33]   [X30,X31,X32,    X33      ]
		matrix4 Y   = X;
		vector3 dxa = dx*X.at(3,3);
		Y.at(0,3) += dxa[0];
		Y.at(1,3) += dxa[1];
		Y.at(2,3) += dxa[2];
		return Y;
	}

	template<typename T> inline
	math::matrix<T,4,4> pre_translate(const math::matrix<T,4,4>& X, const math::vector3<T>& dx) {
		auto [X00,X01,X02,X03,
			X10,X11,X12,X13,
			X20,X21,X22,X23,
			X30,X31,X32,X33] = X._My_data;
		return math::matrix<T,4,4>
		/*{X00,X01,X02,X03,    {1,0,0,dx[0],    */ { X00, X01, X02, X00*dx[0] + X01*dx[1] + X02*dx[2] + X03,
		/* X10,X11,X12,X13,  *  0,1,0,dx[1],  = */   X10, X11, X12, X10*dx[0] + X11*dx[1] + X12*dx[2] + X13,
		/* X20,X21,X22,X23,     0,0,1,dx[2],    */   X20, X21, X22, X20*dx[0] + X21*dx[1] + X22*dx[2] + X23,
		/* X30,X31,X32,X33 }    0,0,0,  1   }   */   X30, X31, X32, X30*dx[0] + X31*dx[1] + X32*dx[2] + X33 };
	}

	/// Y = ( dot(transpose(n),n)*(1 - c) + c*eye() + s*cross(transpose(n),n) ) * X.
	template<typename Matrix4, typename UnitVector3, typename Scalar>
	Matrix4 mrotate(const Matrix4& X, const UnitVector3& n, Scalar arg) {
		/// [ x*x*(1-c) + c,   x*y*(1-c) - s*z, x*z*(1-c) + s*y ]
		/// [ y*x*(1-c) + s*z, y*y*(1-c) + c,   y*z*(1-c) - s*x ] * X
		/// [ z*x*(1-c) - s*y, z*y*(1-c) + s*x, z*z*(1-c) + c   ]
		Matrix4 Y;
		Scalar c = cos(arg),
					 s = sin(arg);
		auto n1c = n*(1 - c);
		Y.row(0) = (n[0]*n1c[0] + c     )*X.row(0) + (n[0]*n1c[1] - s*n[2])*X.row(1) + (n[0]*n1c[2] + s*n[1])*X.row(2);
		Y.row(1) = (n[1]*n1c[0] + s*n[2])*X.row(0) + (n[1]*n1c[1] + c     )*X.row(1) + (n[1]*n1c[2] - s*n[0])*X.row(2);
		Y.row(2) = (n[2]*n1c[0] - s*n[1])*X.row(0) + (n[2]*n1c[1] + s*n[0])*X.row(1) + (n[2]*n1c[2] + c     )*X.row(2);
		if (X.rows() == 4) {
			Y.row(3) = X.row(3);
		}

		return Y;
	}

	/// Y = ( dot(transpose(n),n)*2 - eye ) * X.
	template<typename Matrix4, typename UnitVector3>
	Matrix4 mmirror(const Matrix4& X, const UnitVector3& n) {
		Matrix4 Y;
		auto n2  = n*2;
		Y.row(0) = (n[0]*n2[0] - 1)*X.row(0) + (n[0]*n2[1])    *X.row(1) + (n[0]*n2[2])    *X.row(2);
		Y.row(1) = (n[1]*n2[0])    *X.row(0) + (n[1]*n2[1] - 1)*X.row(1) + (n[1]*n2[2])    *X.row(2);
		Y.row(2) = (n[2]*n2[0])    *X.row(0) + (n[2]*n2[1])    *X.row(1) + (n[2]*n2[2] - 1)*X.row(2);
		if (X.rows() == 4) {
			Y.row(3) = X.row(3);
		}

		return Y;
	}

	/// Y = ( eye - dot(transpose(n),n)*2 ) * X.
	template<typename Matrix4, typename UnitVector3>
	Matrix4 mreflect(const Matrix4& X, const UnitVector3& n) {
		Matrix4 Y;
		auto n2  = n*2;
		Y.row(0) = (1 - n[0]*n2[0])*X.row(0) +     (n[0]*n2[1])*X.row(1) +     (n[0]*n2[2])*X.row(2);
		Y.row(1) =     (n[1]*n2[0])*X.row(0) + (1 - n[1]*n2[1])*X.row(1) +     (n[1]*n2[2])*X.row(2);
		Y.row(2) =     (n[2]*n2[0])*X.row(0) +     (n[2]*n2[1])*X.row(1) + (1 - n[2]*n2[2])*X.row(2);
		if (X.rows() == 4) {
			Y.row(3) = X.row(3);
		}

		return Y;
	}
#endif

	#if 0
	template<typename Matrix4, typename Quaternion>
	Matrix4 quaternion_cast(const Quaternion& q) {
		auto qw = q.R_component_1();
		auto qx = q.R_component_2();
		auto qy = q.R_component_3();
		auto qz = q.R_component_4();
		auto s  = 2/(qx*qx + qy*qy + qz*qz + qw*qw);
		return Matrix4
		{ 1-s*(qy*qy+qz*qz),   s*(qx*qy-qw*qz),   s*(qx*qz+qw*qy), 0,
				s*(qx*qy+qw*qz), 1-s*(qx*qx+qz*qz),   s*(qy*qz-qw*qx), 0,
				s*(qx*qz-qw*qy),   s*(qy*qz+qw*qx), 1-s*(qx*qx+qy*qy), 0,
				0,                 0,                 0,               1 };
	}


	template<typename Matrix4x4, typename Scalar> inline
	Matrix4x4 rotation(Scalar xy/*[rad]*/, Scalar xz/*[rad]*/, Scalar yz/*[rad]*/) {
		/// wait improve...
		/// 
		///@article { title={Euler's rotation theorem}, author={Leonhard-Euler}, year={?} }
		/// @solve xy_rot * |r * cos(a)| = |r * cos(a + theta)|
		///                 |r * sin(a)|   |r * sin(a + theta)|
		///
		///        xy_rot  *  |cos(a)|   = |cos(a)*cos(theta)-sin(a)*sin(theta)|  : earse 'r', and apply 'trigonometric-law'
		///                   |sin(a)|     |sin(a)*cos(theta)+cos(a)*sin(theta)|
		///
		/// |xy_rot[0,0]*cos(a)+xy_rot[0,1]*sin(a)| = |cos(a)*cos(theta)-sin(a)*sin(theta)|
		/// |xy_rot[1,0]*cos(a)+xy_rot[1,1]*sin(a)|   |sin(a)*cos(theta)+cos(a)*sin(theta)|
		///
		///        xy_rot = |cos(theta), -sin(theta)|
		///                 |sin(theta),  cos(theta)|
		///
		///@then
		///        xz_rot = |cos(theta), 0, -sin(theta)|
		///                 |     0,     1,      0     |
		///                 |sin(theta), 0,  cos(theta)|
		///
		///        yz_rot = |1,      0,           0    |
		///                 |0, cos(theta), -sin(theta)|
		///                 |0, sin(theta),  cos(theta)|
		///
		///@example yz_rot * xz_rot * xy_rot, dependent of multiply order, independent of coordinate system
		///    |1,    0,        0   |   |cos(xz), 0, -sin(xz)|   |cos(xy), -sin(xy), 0|
		///  = |0, cos(yz), -sin(yz)| * |   0,    1,     0   | * |sin(xy),  cos(xy), 0|
		///    |0, sin(yz),  cos(yz)|   |sin(xz), 0,  cos(xz)|   |   0,        0,    1|
		///
		///    |         cos(xz),       0,         -sin(xz)|   |cos(xy), -sin(xy), 0|
		///  = |-sin(yz)*sin(xz), cos(yz), -sin(yz)*cos(xz)| * |sin(xy),  cos(xy), 0|
		///    | cos(yz)*sin(xz), sin(yz),  cos(yz)*cos(xz)|   |   0,        0,    1|
		///
		///    |         cos(xz)*cos(xy),                         cos(xz)*-sin(xy),                        -sin(xz)|
		///  = |-sin(yz)*sin(xz)*cos(xy)+cos(yz)*sin(xy), sin(yz)*sin(xz)*sin(xy)+cos(yz)*cos(xy), -sin(yz)*cos(xz)|
		///    | cos(yz)*sin(xz)*cos(xy)+sin(yz)*sin(xy), cos(yz)*sin(xz)*-sin(xy)+sin(yz)*cos(xy), cos(yz)*cos(xz)|
		///
		Scalar cxy = cos(xy);
		Scalar sxy = sin(xy);
		Scalar cxz = cos(xz);
		Scalar sxz = sin(xz);
		Scalar cyz = cos(yz);
		Scalar syz = sin(yz);
		return Matrix4x4
		{      cxz*cxy,             -cxz*sxy,             -sxz, 0,
			-syz*sxz*cxy+cyz*sxy,  syz*sxz*sxy+cyz*cxy, -syz*cxz, 0,
			 cyz*sxz*cxy+syz*sxy, -cyz*sxz*sxy+syz*cxy,  cyz*cxz, 0,
							0,                    0,                0,    1 };
	}

	template<typename Matrix3x3, typename Scalar> inline
	Matrix3x3 rotation33(Scalar xy/*[rad]*/, Scalar xz/*[rad]*/, Scalar yz/*[rad]*/) {
		Scalar cxy = cos(xy);
		Scalar sxy = sin(xy);
		Scalar cxz = cos(xz);
		Scalar sxz = sin(xz);
		Scalar cyz = cos(yz);
		Scalar syz = sin(yz);
		return Matrix3x3
		{      cxz*cxy,             -cxz*sxy,             -sxz,
			-syz*sxz*cxy+cyz*sxy,  syz*sxz*sxy+cyz*cxy, -syz*cxz,
			 cyz*sxz*cxy+syz*sxy, -cyz*sxz*sxy+syz*cxy,  cyz*cxz };
	}

	template<typename matrix4, typename scalar> inline
	matrix4 objectm(scalar sx, scalar sy, scalar sz, scalar qw, scalar qx, scalar qy, scalar qz, scalar x, scalar y, scalar z) {
		scalar s  = 2/(qx*qx + qy*qy + qz*qz + qw*qw);
		return matrix4
		{ (1-s*(qy*qy+qz*qz))*sx,   (s*(qx*qy-qw*qz))*sy,   (s*(qx*qz+qw*qy))*sz, x,
				(s*(qx*qy+qw*qz))*sx, (1-s*(qx*qx+qz*qz))*sy,   (s*(qy*qz-qw*qx))*sz, y,
				(s*(qx*qz-qw*qy))*sx,   (s*(qy*qz+qw*qx))*sy, (1-s*(qx*qx+qy*qy))*sz, z,
									0         ,            0          ,            0          , 1 };
	}
#endif