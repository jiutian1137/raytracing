#pragma once

#include <math/concepts.hpp>

#include "../../fraction.hpp"
#include "../../integral.hpp"
#include "../../geometry/shape.hpp"

#include <algorithm>
#include <execution>

namespace math { namespace physics {
	template<typename _Ty, typename spectrum, typename scalar>
	concept heightlayer = requires(const _Ty layer, scalar height, scalar scattering_angle) {
		{ layer.get_scattering(height) } -> std::convertible_to<spectrum>;
		{ layer.get_extinction(height) } -> std::convertible_to<spectrum>;
		{ layer.get_density(height) } -> std::convertible_to<scalar>;
		{ layer.phase(scattering_angle) } -> std::convertible_to<scalar>;
	};

	template<math::vector spectrum, math::vector vector3,
		heightlayer<spectrum, math::value_t<vector3>>... _Heightlayers>
	struct atmosphere {
		using scalar = math::value_t<vector3>;
		scalar radius;
		scalar max_height;
		std::tuple<_Heightlayers...> layers;
		math::quadrature<scalar> integrator[2];
		math::spherical_quadrature<vector3> angular_integrator;

		int intersect(const vector3& O, const vector3& D, const scalar max_t, scalar* t4) const {
			if (math::geometry::ray_sph_intersect4(O, D, false, this->radius, this->radius + this->max_height, t4) >= 2 && t4[0] < max_t) {
				t4[1] = std::min(t4[1],max_t);
				return 2;
			}
			return 0;
		}

		scalar/*[%]*/ get_density(const scalar& height) const {
			return std::apply([&height](auto&&... layer){ return (layer.get_density(height) + ...); }, layers);
		}

		spectrum/*[%]*/ get_scattering(const scalar& height, const scalar scattering_angle) const {
			return std::apply([&height,&scattering_angle](auto&&... layer){ return 
				((layer.get_scattering(height) * layer.phase(scattering_angle)) + ...);
			}, layers);
		}

		template<size_t _Iidx>
		spectrum/*[%]*/ get_transmittance(const vector3& ray_start, const vector3& ray_direction, const scalar& distance) const {
			assert( distance >= 0 );
			if (distance == 0) {
				return /* vexp(-{0) = */math::ones<spectrum>(1);
			} else {
#if 1
				spectrum sum = this->integrator[_Iidx](scalar(0), distance, [&,this](const scalar distance_i) {
					scalar height_i = length(ray_start + ray_direction * distance_i) - this->radius;
					return std::apply([&height_i](auto&&... layer) {
						return ( layer.get_extinction(height_i) + ... );
					}, this->layers);
				});
#else
				///@optimization integral of vector to integral of number.
				spectrum sum = std::apply([&,this](auto&&... layer) {
					scalar  r      = length(ray_start);
					scalar  mu     = dot(ray_start, ray_direction)/r;
					auto integrand = [&r,&mu,this](const scalar& x, const auto& layer) {
						return layer.density(sqrt(x*x + 2*r*mu*x + r*r) - this->radius)/*[optical length]*/;
					};
					return ( (layer.extinction_cross_section() * this->integrator[_Iidx](scalar(0), distance, integrand, layer)) + ... );
				}, layers);
#endif
				return vexp(-sum);
			}
		}

		spectrum/*[%]*/ get_radiance(const vector3& ray_start, const vector3& ray_direction, const scalar& distance, const vector3& light_vector, spectrum& first_transmitance) const {
			assert( distance >= 0 );
			if (distance == 0) { 
				first_transmitance = math::ones<spectrum>(1);
				return math::ones<spectrum>(0);
			} else {
				first_transmitance = this->get_transmittance<0>(ray_start, ray_direction, distance);
				return this->integrator[0](scalar(0), distance, [&,this](const scalar& distance_i) {
					vector3 point_i = ray_start + ray_direction * distance_i;
					scalar light_distance;
					spectrum radiance_i;
					if (!math::geometry::ray_sph_outside_test(point_i, light_vector, this->radius) && (light_distance = 
						math::geometry::ray_sph_inside_intersect(point_i, light_vector, this->radius + this->max_height)) > 0) {
						radiance_i = this->get_transmittance<1>(point_i, light_vector, light_distance) *
							this->get_scattering(length(point_i) - this->radius, dot(ray_direction, light_vector));
					} else {/* skip zero radiance */
						return math::ones<spectrum>(0);
					}

					if (0 < distance_i) {
						radiance_i *= this->get_transmittance<1>(ray_start, ray_direction, distance_i);
					}

					return radiance_i;
				});
			}
		}
	};

	template<math::vector spectrum, math::vector vector3,
		math::dynamic_container lookupt2d, math::vector vector2,
		heightlayer<spectrum, math::value_t<vector3>>... _Heightlayers>
	struct atmosphere2 : public atmosphere<spectrum, vector3, _Heightlayers...> {
		using scalar = math::value_t<vector3>;

		static vector2 lut2param(const scalar r_min, const scalar r_max, const scalar r, const scalar mu) {
			///@param r in [medium.inner_radius,medium.outer_radius]
			///@param mu in [-1,+1], requires rayisphere(r,mu, medium.inner_radius) <= 0
			///                 _ __
			///             - *Oray     -
			///         -      \  \  h      -
			///     .        - \     \ \       -
			///   .      .-   r \        /\.     -.
			///  .     .        \    ir/     \ H   .
			///       /          \   /        \ \  \
			/// .     .          \ /           .  \ .
			/// |    |            C-- -- -- or -- --*
			/// .     .                        .    .Oray+D*d_max
			///  \    \                       /    /
			///   .     .                   .     .
			///     .     .               .     .
			///       -      - _ ... _ -      -
			///          -                 -
			///               -  ...  -
			assert( r_min <= r && r <= r_max );
			assert( -1 <= mu && mu <= 1 );
			scalar mur = mu*r, rr = r*r, rr_min = r_min * r_min, rr_max = r_max * r_max;//assert( mu < 0 && mur*mur + rr_min >= rr );
			// Distance to top atmosphere boundary for a horizontal ray at ground level.
			scalar H = sqrt(rr_max - rr_min);
			// Distance to the horizon.
			scalar h = sqrt(rr - rr_min);
			// Distance to the top atmosphere boundary for the ray (r,mu) and its
			//  minimum and maximum values over all mu - obtained for (r,1) and (r,mu_horizon).
			scalar d = -mur + sqrt(max(mur*mur - (rr - rr_max), scalar(0)));
			scalar d_min = r_max - r;
			scalar d_max = h + H;
			return vector2{(d - d_min)/(d_max - d_min),  h/H};//"clamp" is built in the sampling function, so "no clamp is needed".
#if 0
			scalar d = mu < 0 ? -mur + sqrt(max(scalar(0), mur*mur - (rr - oror)))
				: (rr - oror)/(-mur - sqrt(max(scalar(0), mur*mur - (rr - oror))));
			/// But this form cannot get a simple inverse when mu >= 0.
			///		d = -mur - sqrt(mur*mur - (rr - oror))
			///		d*d + mur*mur + 2*d*mur = -mur*mur + (rr - oror)
			///		d*d + 2*mur*mur + 2*d*mur = (rr - oror)
#endif
		}

		static void param2lut(const scalar r_min, const scalar r_max, const vector2& param, scalar& r, scalar& mu, scalar& d) {
			assert( 0 <= param[0] && param[0] <= 1 );
			assert( 0 <= param[1] && param[1] <= 1 );
			// Distance to top atmosphere boundary for a horizontal ray at ground level.
			scalar H = sqrt(r_max*r_max - r_min*r_min);
			// Distance to the horizon.
			scalar h = H*param[1];
			r = sqrt(h*h + r_min*r_min);
			r = clamp(r, r_min, r_max);
			// Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
			// and maximum values over all mu - obtained for (r,1) and (r,mu_horizon).
			scalar d_min = r_max - r;
			scalar d_max = H + h;
				d = d_min + (d_max - d_min)*param[0];
			mu = d == 0 ? scalar(1) : (H*H - h*h - d*d)/(2*d*r);
			mu = clamp(mu, scalar(-1), scalar(+1));
		}
		
		lookupt2d transmittance_lookupt;
		using _Mybase = atmosphere<spectrum, vector3, _Heightlayers...>;

		void precompute_transmittance(const typename lookupt2d::size_type& the_size) {
			if (/*this->transmittance_lookupt.size() != the_size*/true) 
				this->transmittance_lookupt.resize(the_size);

			std::for_each(std::execution::par, this->transmittance_lookupt.begin(), this->transmittance_lookupt.end(),
			[this](spectrum& transmittance) {
				auto y = static_cast<typename lookupt2d::length_type>(std::distance(&this->transmittance_lookupt[0], &transmittance));
				auto x = y % transmittance_lookupt.stride(1);
				     y /= transmittance_lookupt.stride(1);
				scalar r, mu, distance;
				param2lut(this->radius, this->radius+this->max_height, 
					vector2{scalar(x),scalar(y)}/this->transmittance_lookupt.edge(), r, mu, distance);
				vector3 ray_start     = {0, r, 0};
				vector3 ray_direction = {sqrt(1 - mu*mu), mu, 0};

				transmittance = _Mybase::get_transmittance<0>(ray_start, ray_direction, distance);
			});
		}

		spectrum get_transmittance(const vector3& ray_start, const vector3& ray_direction) const {
			scalar  r  = length(ray_start);
			scalar  mu = dot(ray_start, ray_direction)/r;
			assert( !math::geometry::ray_sph_outside_test(ray_start, ray_direction, this->radius) );

			const scalar
				r_min = this->radius, 
				r_max = this->radius + this->max_height;
				r     = clamp(r, r_min, r_max);
				mu    = clamp(mu, scalar(-1), scalar(1));
			return math::tex_sample(this->transmittance_lookupt, lut2param(r_min, r_max, r, mu));
		}
		
		spectrum get_transmittance(const vector3& ray_start, const vector3& ray_direction, const scalar& /*distance*/d, bool intersected_bottom) const {
			assert( d >= 0 );
			if (d <= 0) {
				return /* exp( -0 ) = */math::ones<spectrum>(1);
			} else {
				scalar  r  = length(ray_start);
				scalar  mu = dot(ray_start, ray_direction)/r;
				/// cosine-law[ a*a + b*b - 2*a*b*cos(C) = c*c ], 
				/// cosine is even func and medium is symmetry since [ a*a + b*b + 2*a*b*cos(C) = c*c ].
				scalar  r1  = sqrt(r*r + d*d + 2*r*d*mu);
				scalar  mu1 = (r*mu + d)/r1;

				const scalar
					r_min = this->radius, 
					r_max = this->radius + this->max_height;
					r     = clamp(r, r_min, r_max);
					mu    = clamp(mu, scalar(-1), scalar(1));
					r1    = clamp(r1, r_min, r_max);
					mu1   = clamp(mu1, scalar(-1), scalar(1));
				/// transmittance( first to last )
				///		= integral<first, last>( exp(- density(x)*extinction)*dx )
				///		= "integral<first, mid>( exp(-density(x)*extinction)*dx )" * integral<mid, last>( exp(-density(x)*extinction)*dx )
				spectrum transmittance_first_to_last, transmittance_mid_to_last;
				if (!intersected_bottom/*math::geometry::ray_sph_outside_test(ray_start, ray_direction, this->radius)*/) {
					transmittance_first_to_last = math::tex_sample(this->transmittance_lookupt, lut2param(r_min, r_max, r, mu));
					transmittance_mid_to_last = math::tex_sample(this->transmittance_lookupt, lut2param(r_min, r_max, r1, mu1));
				}	else {
					transmittance_first_to_last = math::tex_sample(this->transmittance_lookupt, lut2param(r_min, r_max, r1, -mu1));
					transmittance_mid_to_last = math::tex_sample(this->transmittance_lookupt, lut2param(r_min, r_max, r, -mu));
				}
				return vmin(transmittance_first_to_last/transmittance_mid_to_last, 1);
			}
		}
	
		spectrum get_transmittance(const vector3& ray_start, const vector3& ray_direction, const scalar& distance) const {
			return transmittance(ray_start, ray_direction, distance,
				math::geometry::ray_sph_outside_test(ray_start, ray_direction, this->radius));
		}

		spectrum get_radiance(const vector3& ray_start, const vector3& ray_direction, const scalar& distance, const vector3& light_vector, spectrum& first_transmitance) const {
			assert( distance >= 0 );
			if (distance == 0) {
				first_transmitance = math::ones<spectrum>(1);
				return math::ones<spectrum>(0);
			} else {
				bool intersected_ground = math::geometry::ray_sph_outside_test(ray_start, ray_direction, this->radius);
				first_transmitance = this->get_transmittance(ray_start, ray_direction, distance, intersected_ground);
				return this->integrator[0](scalar(0), distance, [&,this](const scalar& distance_i) {
					vector3 point_i = ray_start + ray_direction * distance_i;
					spectrum radiance_i;
					if (!math::geometry::ray_sph_outside_test(point_i, light_vector, this->radius)) {
						radiance_i = this->get_transmittance(point_i, light_vector) *
							this->get_scattering(length(point_i) - this->radius, dot(ray_direction, light_vector));
					} else {/* skip zero radiance */
						return math::ones<spectrum>(0);
					}

					return radiance_i * this->get_transmittance(ray_start, ray_direction, distance_i, intersected_ground);
				});
			}
		}
	};

	template<math::vector spectrum, math::vector vector3,
		math::dynamic_container lookupt2d, math::vector vector2,
		math::dynamic_container lookupt4d, math::vector vector4, size_t _Rayleigh, size_t _Mie,
		heightlayer<spectrum, math::value_t<vector3>>... _Heightlayers>
	struct atmosphere3 : public atmosphere2<spectrum, vector3, lookupt2d, vector2, _Heightlayers...> {
		using scalar = math::value_t<vector3>;

		static vector4 lut2param(scalar r_min, scalar r_max, typename lookupt4d::length_type mu_size, scalar mu_s_min, 
			scalar r, scalar mu, scalar mu_s, scalar nu, bool ray_r_mu_intersects_ground) {
			assert( r_min <= r && r <= r_max );
			assert( -1 <= mu   && mu   <= 1 );
			assert( -1 <= mu_s && mu_s <= 1 );
			assert( -1 <= nu   && nu   <= 1 );
			const scalar mur = mu*r, rr = r*r, rr_min = r_min*r_min, rr_max = r_max*r_max;

			// Distance to top atmosphere boundary for a horizontal ray at ground level.
			scalar H = sqrt(rr_max - rr_min);
			// Distance to the horizon.
			scalar h = sqrt(r*r - rr_min);
			scalar u_r = h / H;

			// Discriminant of the quadratic equation for the intersections of the ray
			// (r,mu) with the ground (see RayIntersectsGround).
			scalar discriminant = mur*mur - (r*r - rr_min);
			scalar u_mu;
			if (ray_r_mu_intersects_ground) {
				// Distance to the ground for the ray (r,mu), and its minimum and maximum
				// values over all mu - obtained for (r,-1) and (r,mu_horizon).
				scalar d = -mur - sqrt(max(discriminant, scalar(0)));
				scalar d_min = r - r_min;
				scalar d_max = h;
				scalar mu_dur = scalar(mu_size/2 - 1)/scalar(mu_size - 1);
				u_mu = mu_dur*(d_max == d_min ? scalar(0) : (d - d_min)/(d_max - d_min));
			} else {
				// Distance to the top atmosphere boundary for the ray (r,mu), and its
				// minimum and maximum values over all mu - obtained for (r,1) and
				// (r,mu_horizon).
				scalar d = -mur + sqrt(max(discriminant + H*H, scalar(0)));
				scalar d_min = r_max - r;
				scalar d_max = h + H;
				scalar mu_dur = scalar(mu_size/2 - 1)/scalar(mu_size - 1);
				scalar mu_min = scalar(mu_size/2)/scalar(mu_size - 1);
				u_mu = mu_min + mu_dur*((d - d_min)/(d_max - d_min));
			}

			scalar d_min = r_max - r_min;
			scalar d_max = H;
			// Distance to the top atmosphere boundary for the ray (inner_radius, mu_s)
			scalar d = -r_min*mu_s + sqrt(max(rr_min*mu_s*mu_s - (rr_min - rr_max), scalar(0)));
			scalar a = (d - d_min)/(d_max - d_min);
			// Distance to the top atmosphere boundary for the ray (inner_radius, mu_s_min)
			scalar D = -r_min*mu_s_min + sqrt(max(rr_min*mu_s_min*mu_s_min - (rr_min - rr_max), scalar(0)));
			scalar A = (D - d_min)/(d_max - d_min);
			// An ad-hoc function equal to 0 for mu_s = mu_s_min (because then d = D and
			// thus a = A), equal to 1 for mu_s = 1 (because then d = d_min and thus
			// a = 0), and with a large slope around mu_s = 0, to get more texture 
			// samples near the horizon.
			scalar u_mu_s = (1 - a/A)/(1 + a);

			return vector4{u_r, u_mu, u_mu_s, (nu + 1)/2};//"clamp" is built in the sampling function, so "no clamp is needed".
		}

		static void param2lut(scalar r_min, scalar r_max, typename lookupt4d::length_type mu_size, scalar mu_s_min, const vector4& param,
			scalar& r, scalar& mu, scalar& mu_s, scalar& nu, scalar& distance, bool& ray_r_mu_intersects_ground) {
			assert( 0 <= param[0] && param[0] <= 1 );
			assert( 0 <= param[1] && param[1] <= 1 );
			assert( 0 <= param[2] && param[2] <= 1 );
			assert( 0 <= param[3] && param[3] <= 1 );
			const scalar mur = mu*r, rr = r*r, rr_min = r_min*r_min, rr_max = r_max*r_max;

			// Distance to top atmosphere boundary for a horizontal ray at ground level.
			scalar H = sqrt(rr_max - rr_min);
			// Distance to the horizon.
			scalar h = H * param[0];
			r = sqrt(h*h + rr_min);
			r = clamp(r, r_min, r_max);

			scalar mu_dur = scalar(mu_size/2 - 1)/scalar(mu_size - 1);
			scalar mu_min = scalar(mu_size/2)/scalar(mu_size - 1);
			if (param[1] <= mu_dur) {
				// Distance to the ground for the ray (r,mu), and its minimum and maximum
				// values over all mu - obtained for (r,-1) and (r,mu_horizon) - from which
				// we can recover mu:
				scalar d_min = r - r_min;
				scalar d_max = h;
				scalar d     = d_min + (param[1]/mu_dur)*(d_max - d_min);
				mu = (d == 0 ? scalar(-1.0) : -(h*h + d*d)/(2*r*d));
				distance = d;
				ray_r_mu_intersects_ground = true;
			} else if (mu_min <= param[1]) {
				// Distance to the top atmosphere boundary for the ray (r,mu), and its
				// minimum and maximum values over all mu - obtained for (r,1) and
				// (r,mu_horizon) - from which we can recover mu:
				scalar d_min = r_max - r;
				scalar d_max = h + H;
				scalar d     = d_min + ((param[1] - mu_min)/mu_dur)*(d_max - d_min);
				mu = (d == 0 ? scalar(1.0) : (H*H - h*h - d*d)/(2*r*d));
				distance = d;
				ray_r_mu_intersects_ground = false;
			} else {
				abort();
			}
			mu = clamp(mu, scalar(-1), scalar(+1));

			scalar x_mu_s = param[2];
			scalar d_min = r_max - r_min;
			scalar d_max = H;
			scalar D = -r_min*mu_s_min + sqrt(max(rr_min*mu_s_min*mu_s_min - (rr_min - rr_max), scalar(0)));
			scalar A = (D - d_min)/(d_max - d_min);
			scalar a = (A - x_mu_s * A) / (1 + x_mu_s * A);
			scalar d = d_min + min(a, A) * (d_max - d_min);
			mu_s = (d == 0 ? scalar(1.0) : ((H*H - d*d) / (2*r_min*d)));
			mu_s = clamp(mu_s, scalar(-1), scalar(+1));

			nu = param[3]*2 - 1;
			nu = clamp(nu, scalar(-1), scalar(+1));
		}

		static void param2lut(scalar r_min, scalar r_max, typename lookupt4d::length_type mu_size, scalar mu_s_min, const vector4& param,
			vector3& ray_start, vector3& ray_direction, vector3& light_vector, scalar& distance, bool& ray_r_mu_intersects_ground) {
			scalar r, mu, mu_s, nu;
			param2lut(r_min, r_max, mu_size, mu_s_min, param,
				r, mu, mu_s, nu, distance, ray_r_mu_intersects_ground);
#if 0
			ray_start     = {0, r, 0};
			ray_direction = {0/*sqrt(1 - mu*mu)*cos(a)*/, mu, 0/*sqrt(1 - mu*mu)*sin(a)*/};
			light_vector  = {0/*sqrt(1 - mu_s*mu_s)*cos(b)*/, mu_s, 0/*sqrt(1 - mu_s*mu_s)*sin(b)*/};
			/// dot(ray_direction, light_vector) = nu
			/// cos(a)*cos(b)*sqrt((1 - mu*mu)*(1 - mu_s*mu_s)) + mu*mu_s + sin(a)*sin(b)*sqrt((1 - mu*mu)*(1 - mu_s*mu_s)) = nu
			/// ( cos(a)*cos(b) + sin(a)*sin(b) )*sqrt((1 - mu*mu)*(1 - mu_s*mu_s)) = nu - mu*mu_s
			///   cos(a - b)*sqrt((1 - mu*mu)*(1 - mu_s*mu_s)) = nu - mu*mu_s
			///   cos(a - b) = (nu - mu*mu_s)/sqrt((1 - mu*mu)*(1 - mu_s*mu_s))
			///   b = a - acos( (nu - mu*mu_s)/sqrt((1 - mu*mu)*(1 - mu_s*mu_s)) )
			nu = clamp(nu, mu*mu_s - sqrt((1 - mu*mu)*(1 - mu_s*mu_s)), mu*mu_s + sqrt((1 - mu*mu)*(1 - mu_s*mu_s)));
			/// +-1 = (nu - mu*mu_s)/sqrt((1 - mu*mu)*(1 - mu_s*mu_s))
			/// +-sqrt((1 - mu*mu)*(1 - mu_s*mu_s)) + mu*mu_s = nu
			scalar a = 0;
			scalar b = a - acos(clamp( (nu - mu*mu_s)/sqrt((1 - mu*mu)*(1 - mu_s*mu_s)) , scalar(-1), scalar(+1)));
			if (isnan(b)) { b = 0; }
			ray_direction[0] = sqrt(1 - mu*mu)*cos(a);
			ray_direction[2] = sqrt(1 - mu*mu)*sin(a);
			light_vector[0] = sqrt(1 - mu_s*mu_s)*cos(b);
			light_vector[2] = sqrt(1 - mu_s*mu_s)*sin(b);
#else
			nu = clamp(nu, mu*mu_s - sqrt((1 - mu*mu)*(1 - mu_s*mu_s)), mu*mu_s + sqrt((1 - mu*mu)*(1 - mu_s*mu_s)));
			scalar a = 0;
			scalar b = a - acos(clamp( (nu - mu*mu_s)/sqrt((1 - mu*mu)*(1 - mu_s*mu_s)) , scalar(-1), scalar(+1)));
			if (isnan(b)) { b = 0; }
			ray_start     = {0, r, 0};
			ray_direction = {sqrt(1 - mu*mu)*cos(a), mu, sqrt(1 - mu*mu)*sin(a)};
			light_vector  = {sqrt(1 - mu_s*mu_s)*cos(b), mu_s, sqrt(1 - mu_s*mu_s)*sin(b)};
#endif
			/*scalar e0 = abs(dot(D,normalize(O - center)) - mu);
			scalar e1 = abs(dot(Ds,normalize(O - center)) - mu_s);
			scalar e2 = abs(dot(Ds,D) - nu);
			if (e0 > 0.0001 || e1 > 0.0001 || e2 > 0.0001) {
				std::cout << mu << "," << mu_s << ',' << nu << ',' << D << ',' << Ds << std::endl;
			}*/
		}

		lookupt4d rayleigh_radiance_lookupt, mie_radiance_lookupt;
		scalar mu_s_min = cos(120*0.01745329f);
		using _Mybase = atmosphere2<spectrum, vector3, lookupt2d, vector2, _Heightlayers...>;///only used in precompute_multiscattering.

		void precompute_scattering(const typename lookupt4d::size_type& the_size) {
			assert( the_size[1] % 2 == 0 );
			if (/*this->rayleigh_radiance_lookupt.size() != the_size*/true) {
				this->rayleigh_radiance_lookupt.resize(the_size);
				this->mie_radiance_lookupt.resize(the_size);
			}

			std::for_each(std::execution::par, this->rayleigh_radiance_lookupt.begin(), this->rayleigh_radiance_lookupt.end(), 
			[this](spectrum& rayleigh_unphased_radiance) {
				auto w = static_cast<typename lookupt2d::length_type>( std::distance(&this->rayleigh_radiance_lookupt[0], &rayleigh_unphased_radiance) );
				spectrum& mie_unphased_radiance = this->mie_radiance_lookupt[w];
				auto z = w % this->mie_radiance_lookupt.stride(3);
				     w /= this->mie_radiance_lookupt.stride(3);
				auto y = z % this->mie_radiance_lookupt.stride(2);
				     z /= this->mie_radiance_lookupt.stride(2);
				auto x = y % this->mie_radiance_lookupt.stride(1);
				     y /= this->mie_radiance_lookupt.stride(1);
				vector3 ray_start, ray_direction, light_vector;
				scalar  distance;
				bool    intersected_ground;
				param2lut(this->radius, this->radius + this->max_height, this->rayleigh_radiance_lookupt.size(1), this->mu_s_min, 
					vector4{scalar(x),scalar(y),scalar(z),scalar(w)}/this->rayleigh_radiance_lookupt.edge(), ray_start, ray_direction, light_vector, distance, intersected_ground);

				/// because 
				///		phase is complexity, so cannot saved in the lookuptable
				///		phase is constant
				/// so
				///		precompute unphased scattering 
				rayleigh_unphased_radiance = math::ones<spectrum>(0);
				mie_unphased_radiance      = math::ones<spectrum>(0);
				this->integrator[0](int{}, scalar(0), distance, [&,this](const scalar& distance_i, const scalar& weight_i) {
					vector3  point_i = ray_start + ray_direction * distance_i;
					spectrum radiance_i;
					if (!math::geometry::ray_sph_outside_test(point_i, light_vector, this->radius)) {
						radiance_i = this->get_transmittance(point_i, light_vector);
					} else {/* skip zero radiance */
						return;
					}

					if (0 < distance_i) {
						radiance_i *= this->get_transmittance(ray_start, ray_direction, distance_i, intersected_ground);
					}

					scalar height_i = length(point_i) - this->radius;
					rayleigh_unphased_radiance += radiance_i *
						std::get<_Rayleigh>(this->layers).get_scattering(height_i) *
						weight_i;
					mie_unphased_radiance += radiance_i *
						std::get<_Mie>(this->layers).get_scattering(height_i) *
						weight_i;
				});
			});
		}

#if 0
		void precompute_multiscattering(const typename lookupt4d::size_type& the_size, const typename lookupt2d::size_type& irradiance_size, std::ostream* log = nullptr) {
			///S(L0)
			this->precompute_scattering(the_size);
			const scalar
				r_min = this->radius,
				r_max = this->radius + this->max_height;

			///S(L0)
			lookupt4d delta_rayleigh_radiance_lookupt = this->rayleigh_radiance_lookupt;
			lookupt4d delta_mie_radiance_lookupt = this->mie_radiance_lookupt;

			///R(L0)
			lookupt2d delta_ground_irradiance_lookupt(irradiance_size);
			vector2 irradiance_lookupt_edge = static_vcast<vector2>(irradiance_size - 1);
			std::for_each(std::execution::par, delta_ground_irradiance_lookupt.begin(), delta_ground_irradiance_lookupt.end(), 
				[&,this](spectrum& ground_direct_irradiance) {
					size_t index = std::distance(&delta_ground_irradiance_lookupt[0], &ground_direct_irradiance);
					vector2 param = static_vcast<vector2>(fridx(index, delta_ground_irradiance_lookupt.stride()))/irradiance_lookupt_edge;
					scalar  r     = param[0]*(r_max - r_min) + r_min;
					scalar  mu_s  = param[1]*2 - 1;
					vector3 ray_start     = {0, r, 0};
					vector3 ray_direction = {sqrt(1 - mu_s*mu_s), mu_s, 0};

					if (!math::geometry::ray_sph_outside_test(ray_start, ray_direction, this->radius)) {
						scalar alpha_s = scalar(0.00935/2);
						// Approximate average of the cosine factor mu_s over the visible fraction of
						// the Sun disc.
						scalar average_cosine_factor =
							mu_s < -alpha_s ? scalar(0) : (mu_s > alpha_s ? mu_s :
									(mu_s + alpha_s) * (mu_s + alpha_s) / (4 * alpha_s));

						ground_direct_irradiance = this->get_transmittance(ray_start, ray_direction) * average_cosine_factor;
					} else {
						ground_direct_irradiance = math::ones<spectrum>(0);
					}
				}
			);

			for (size_t order = 2; order != 3; ++order) {
				///[S+R](L0), before update delta_radiance.
				lookupt4d delta_rayleigh_scattering_density_lookupt(the_size);
				lookupt4d delta_mie_scattering_density_lookupt(the_size);
				std::for_each(std::execution::par, delta_rayleigh_scattering_density_lookupt.begin(), delta_rayleigh_scattering_density_lookupt.end(),
					[&,this](spectrum& delta_rayleigh_unphased_scattering_density) {
						size_t index = std::distance(&delta_rayleigh_scattering_density_lookupt[0], &delta_rayleigh_unphased_scattering_density);
						spectrum& delta_mie_unphased_scattering_density = delta_mie_scattering_density_lookupt[index];

						vector3 ray_start, ray_direction, light_vector;
						scalar distance;
						bool intersected_ground;
						param2lut(this->radius, this->radius + this->max_height, this->rayleigh_radiance_lookupt.size(1), this->mu_s_min, 
							static_vcast<vector4>(fridx(index,delta_rayleigh_scattering_density_lookupt.stride()))/this->radiance_lookupt_edge,
							ray_start, ray_direction, light_vector, distance, intersected_ground);

						delta_rayleigh_unphased_scattering_density = math::ones<spectrum>(0);
						delta_mie_unphased_scattering_density = math::ones<spectrum>(0);
						this->angular_integrator(int{},
							[&,this](const vector3& incident_direction/*as in|out both correct*/, const scalar& weight_i) {
								scalar r    = length(ray_start);
								scalar mu   = dot(ray_start, incident_direction)/r;
								scalar mu_s = dot(ray_start, light_vector)/r;
								scalar nu   = dot(incident_direction, light_vector);
									r    = clamp(r, r_min, r_max);
									mu   = clamp(mu, scalar(-1), scalar(1));
									mu_s = clamp(mu_s, scalar(-1), scalar(1));
									nu   = clamp(nu, scalar(-1), scalar(1));
								vector4 param = lut2param(r_min, r_max, this->rayleigh_radiance_lookupt.size(1), mu_s_min, r, mu, mu_s, nu, intersected_ground);
									param *= this->radiance_lookupt_edge;
								spectrum incident_radiance = 
									math::sample_lattice4(delta_rayleigh_radiance_lookupt, param, math::samplmodes::clamp_on_positive(this->radiance_lookupt_edge)) * 
										(order == 2 ? std::get<_Rayleigh>(this->layers).phase(nu) : scalar(1)) + 
									math::sample_lattice4(delta_mie_radiance_lookupt, param, math::samplmodes::clamp_on_positive(this->radiance_lookupt_edge)) * 
										(order == 2 ? std::get<_Mie>(this->layers).phase(nu) : scalar(1));
								scalar t2[2];
								if (math::geometry::ray_sph_intersect2(ray_start, incident_direction, false, this->radius, t2) > 0) {
									vector3 point = ray_start + incident_direction * t2[0];
									scalar r    = length(point);
									scalar mu_s = dot(point, light_vector)/r;
									vector2 param = {(r - r_min)/(r_max - r_min), mu_s * scalar(0.5) + scalar(0.5)};//sampling function buildin clamp.
									spectrum albedo = math::ones<spectrum>(0.1);
									incident_radiance += math::sample_lattice2(delta_ground_irradiance_lookupt, param * irradiance_lookupt_edge, math::samplmodes::clamp_on_positive{ irradiance_lookupt_edge }) * 
										this->get_transmittance(ray_start, incident_direction, t2[0], true) *
										albedo/scalar(3.141592653589793);
								}
								delta_rayleigh_unphased_scattering_density +=
									incident_radiance *
									std::get<_Rayleigh>(this->layers).get_scattering(r - r_min) *
									std::get<_Rayleigh>(this->layers).phase(dot(-incident_direction, ray_direction)) *
									weight_i;
								delta_mie_unphased_scattering_density += 
									incident_radiance *
									std::get<_Mie>(this->layers).get_scattering(r - r_min) * 
									std::get<_Mie>(this->layers).phase(dot(-incident_direction, ray_direction)) *
									weight_i;
							}
						);
					}
				);
				//if (log) { (*log) << "order:"<< order << "\tdelta:{"<< delta[0]<<','<< delta[1]<< ','<<delta[2]<<"}" << std::endl; }

				///R([S+R](L0)) = R(S(L0)) :eliminated ground bounce, update delta_reflect_radiance before update delta_scattering_radiance.
				std::for_each(std::execution::par, delta_ground_irradiance_lookupt.begin(), delta_ground_irradiance_lookupt.end(), 
					[&,this](spectrum& ground_indirect_irradiance) {
						size_t y = std::distance(&delta_ground_irradiance_lookupt[0], &ground_indirect_irradiance);
						size_t x = y % delta_ground_irradiance_lookupt.size(0);
							y /= delta_ground_irradiance_lookupt.size(0);
						vector2 param = vector2{scalar(x),scalar(y)}/irradiance_lookupt_edge;
						scalar  r     = param[0]*(r_max - r_min) + r_min;
						scalar  mu_s  = param[1]*2 - 1;
						vector3 ray_start    = {0, r, 0};
						vector3 light_vector = {sqrt(1 - mu_s*mu_s), mu_s, 0};

						ground_indirect_irradiance = this->angular_integrator([&,this](const vector3& incident_direction) {
							scalar r    = length(ray_start);
							scalar mu   = dot(ray_start, incident_direction)/r;
							scalar mu_s = dot(ray_start, light_vector)/r;
							scalar nu   = dot(incident_direction, light_vector);
								r    = clamp(r, r_min, r_max);
								mu   = clamp(mu, scalar(-1), scalar(1));
								mu_s = clamp(mu_s, scalar(-1), scalar(1));
								nu   = clamp(nu, scalar(-1), scalar(1));
							vector4 param = lut2param(r_min, r_max, this->rayleigh_radiance_lookupt.size(1), mu_s_min, r, mu, mu_s, nu, math::geometry::ray_sph_outside_test(ray_start, incident_direction, this->radius));
								param *= this->radiance_lookupt_edge;
							spectrum incident_radiance = 
								math::sample_lattice4(delta_rayleigh_radiance_lookupt, param, math::samplmodes::clamp_on_positive(this->radiance_lookupt_edge)) * 
									(order == 2 ? std::get<_Rayleigh>(this->layers).phase(nu) : scalar(1)) + 
								math::sample_lattice4(delta_mie_radiance_lookupt, param, math::samplmodes::clamp_on_positive(this->radiance_lookupt_edge)) * 
									(order == 2 ? std::get<_Mie>(this->layers).phase(nu) : scalar(1));
							return incident_radiance/* reflect at accumulate */;
						});
					}
				);

				///S([S+R](L0)), update delta_scattering_radiance and accumulate radiance.
				std::for_each(std::execution::par, delta_rayleigh_radiance_lookupt.begin(), delta_rayleigh_radiance_lookupt.end(),
					[&,this](spectrum& delta_rayleigh_unphased_radiance) {
						size_t w = std::distance(&delta_rayleigh_radiance_lookupt[0], &delta_rayleigh_unphased_radiance);
						spectrum& delta_mie_unphased_radiance = delta_mie_radiance_lookupt[w];
						spectrum& rayleigh_unphased_radiance = this->rayleigh_radiance_lookupt[w];
						spectrum& mie_unphased_radiance      = this->mie_radiance_lookupt[w];
						size_t z = w % this->rayleigh_radiance_lookupt.stride(3);
							w /= this->rayleigh_radiance_lookupt.stride(3);
						size_t y = z % this->rayleigh_radiance_lookupt.stride(2);
							z /= this->rayleigh_radiance_lookupt.stride(2);
						size_t x = y % this->rayleigh_radiance_lookupt.stride(1);
							y /= this->rayleigh_radiance_lookupt.stride(1);
						vector3 ray_start, ray_direction, light_vector;
						scalar distance;
						bool intersected_ground;
						param2lut(this->radius, this->radius + this->max_height, this->rayleigh_radiance_lookupt.size(1), this->mu_s_min, 
							vector4{scalar(x),scalar(y),scalar(z),scalar(w)}/this->radiance_lookupt_edge,
							ray_start, ray_direction, light_vector, distance, intersected_ground);
				
						delta_rayleigh_unphased_radiance = math::ones<spectrum>(0);
						delta_mie_unphased_radiance = math::ones<spectrum>(0);
						this->integrator[0](int{}, scalar(0), distance, [&,this](const scalar& distance_i, const scalar& weight_i) {
							spectrum transmittance = this->get_transmittance(ray_start, ray_direction, distance_i, intersected_ground);
							vector3 point_i = ray_start + ray_direction * distance_i;
							scalar r    = length(point_i);
							scalar mu   = dot(point_i, ray_direction)/r;
							scalar mu_s = dot(point_i, light_vector)/r;
							scalar nu   = dot(ray_direction, light_vector)/r;
								r    = clamp(r, r_min, r_max);
								mu   = clamp(mu, scalar(-1), scalar(1));
								mu_s = clamp(mu_s, scalar(-1), scalar(1));
								nu   = clamp(nu, scalar(-1), scalar(1));
							vector4 param = lut2param(r_min, r_max, this->rayleigh_radiance_lookupt.size(1), mu_s_min, r, mu, mu_s, nu, intersected_ground);
								param *= this->radiance_lookupt_edge;
							delta_rayleigh_unphased_radiance +=
								math::sample_lattice4(delta_rayleigh_scattering_density_lookupt, param, math::samplmodes::clamp_on_positive(this->radiance_lookupt_edge)) *
								transmittance *
								weight_i;
							delta_mie_unphased_radiance += 
								math::sample_lattice4(delta_mie_scattering_density_lookupt, param, math::samplmodes::clamp_on_positive(this->radiance_lookupt_edge)) *
								transmittance *
								weight_i;
						});
						//delta_rayleigh_unphased_radiance /= std::get<_Rayleigh>(this->layers).phase(dot(ray_direction, light_vector));
						//delta_mie_unphased_radiance /= std::get<_Mie>(this->layers).phase(dot(ray_direction, light_vector));
						rayleigh_unphased_radiance += delta_rayleigh_unphased_radiance / std::get<_Rayleigh>(this->layers).phase(dot(ray_direction, light_vector));
						mie_unphased_radiance += delta_mie_unphased_radiance / std::get<_Mie>(this->layers).phase(dot(ray_direction, light_vector));
					}
				);
			}
		}
#endif

		spectrum get_radiance(const vector3& ray_start, const vector3& ray_direction, const vector3& light_vector, bool ray_r_mu_intersects_ground) const {
			scalar  r    = length(ray_start);
			scalar  mu   = dot(ray_start, ray_direction)/r;
			scalar  mu_s = dot(ray_start, light_vector)/r;
			scalar  nu   = dot(ray_direction, light_vector);
			//std::cout << std::format("r={0}, mu={1}, mu_s={2}, nu={3}\n", r, mu, mu_s, nu);

			const scalar
				r_min = this->radius,
				r_max = this->radius + this->max_height;
				r     = clamp(r, r_min, r_max);
				mu    = clamp(mu, scalar(-1), scalar(1));
				mu_s  = clamp(mu_s, scalar(-1), scalar(1));
				nu    = clamp(nu, scalar(-1), scalar(1));
			vector4 param = lut2param(r_min, r_max, this->rayleigh_radiance_lookupt.size(1), mu_s_min,
					r, mu, max(mu_s, mu_s_min), nu, ray_r_mu_intersects_ground);
			return (
				math::tex_sample(this->rayleigh_radiance_lookupt, param) * std::get<_Rayleigh>(this->layers).phase(nu) +
				math::tex_sample(this->mie_radiance_lookupt, param) * std::get<_Mie>(this->layers).phase(nu)
				);
		}

		spectrum get_radiance(const vector3& ray_start, const vector3& ray_direction, const vector3& light_vector) const {
			return get_radiance(ray_start, ray_direction, light_vector,
				math::geometry::ray_sph_outside_test(ray_start,ray_direction, this->radius));
		}

		spectrum get_radiance(const vector3& ray_start, const vector3& ray_direction, const scalar distance, const vector3& light_vector, spectrum& first_transmitance) const {
			/// scattering(O,D,x0,x1)
			///		= integral<x0,x1>( transmittance(O+D*x,Ds) * scattering(O+D*x) * transmittance(O,D,x0,x) * dx )
			///		= integral<x0,xmid>( transmittance(O+D*x,Ds) * scattering(O+D*x) * transmittance(O,D,0,x) * dx ) + transmittance(O,D,x0,xmid) * integral<xmid,x1>( transmittance(O+D*x,Ds) * scattering(O+D*x) * transmittance(O,D,0,x) * dx )
			///		= scattering(O,D,x0,xmid) + transmittance(O,D,x0,xmid) * scattering(O,D,xmid,x1)
			/// 
			/// scattering(O,D,x0,xmid) = scattering(O,D,x0,x1) - transmittance(O,D,x0,xmid) * scattering(O,D,xmid,x1)
			bool intersected_ground = math::geometry::ray_sph_outside_test(ray_start,ray_direction, this->radius);
			first_transmitance = this->get_transmittance(ray_start, ray_direction, distance, intersected_ground);
			return this->get_radiance(ray_start, ray_direction, light_vector, intersected_ground)
				- first_transmitance * this->get_radiance(ray_start + ray_direction * distance, ray_direction, light_vector, intersected_ground);
		}
	};
} }// end of namespace math::physics

/// 
///@thory volumetric cloud 
/// A volume is a density function of spatial f(pos) or f(x,y,z).
/// 
/// First we assume height variation not dependent spatial, then we can describ entire sky
/// use a two dimension map, we have
/// 
///		height_map(y) = density.
///		weather_map(x,z) = density.
///		f(x,y,z) = height_map(y) * weather_map(x,z).
/// 
/// This density function already considered variation of all dimensions, but not like cloud(like a flat cake).
/// We recall f(x,y,z) can be simulate cloud use procedural texture, such as perlin(x,y,z,?seed), it is like cloud.
/// direct relate them is
/// 
///		f(x,y,z) = height_map(y) * weather_map(x,z) * shape, shape maybe perlin(x,y,z,?seed).
/// 
/// Secend we simulate cloud cannot use complete super large volume, volume always downsampled.
/// Therefore we need to represent details of volume, we can be define a erosion function, that satisfy:
/// 1. in domain, erosion(f(x,y,z)) == 0 when f(x,y,z) == 0 .
/// 2. in range, erosion(f(x,y,z)) <= f(x,y,z) .
/// 3. erosion should continues .
/// 
///		erosion(f(x,y,z)) = f(x,y,z) - f(x,y,z) * s.
///		(similar as erosion(f(x,y,z)) = f(x,y,z) * s, but meaning of "s" is not same. f(x,y,z) * s = f(x,y,z) - f(x,y,z) * (1 - s) )
/// 
/// and we get a similar function is 
/// 
///		expansion(f(x,y,z)) = f(x,y,z) + (1 - f(x,y,z)) * s 
///		(represent expansion. but have discontinues at edge when no tiled case.) 
/// 
/// and have another expansion is
/// 
///		expansion(f(x,y,z)) = f(x,y,z) - f(x,y,z) * s, s <= 0.
///		expansion(f(x,y,z)) = f(x,y,z) + f(x,y,z) * s, s >= 0.
/// 
/// Final we add a erosion represent details is
/// 
///		erosion(f(x,y,z)) = erosion(height_map(y) * weather_map(x,z) * shape), shape maybe perlin(x,y,z,?seed).
/// 
///@example
/// 
/// A cloud can be seem as many spheres and some light tails around sphere.
/// 
///		many spheres = 1 - worley(x,y,z,?seed).
///		tails = perlin(x,y,z,?seed).
///		
/// Sch2017 use expansion for the shape.
/// 
///		f(x,y,z) = height_map(y) * weather_map(x,z) * shape(x,y,z.?seed),
///		shape(x,y,z.?seed) = (1 - worley) + (1 - (1 - worley)) * perlin.
///		(and may be erosion, shape(x,y,z.?seed) = erosion many spheres use tails.)
/// 