#pragma once

/// Blue-noise, low-freq and high-energy meaning blue.
///@license Free 
///@review 2022-9-27 
///@contact Jiang1998Nan@outlook.com 
#define _MATH_RANDOM_BLUE_NOISE_

#include "../mdarray.hpp"
#include "../function.hpp"// sequence_scramble_kernel.
#include <random>// std::uniform_int_distribution
#include <algorithm>// std::copy, std::swap, std::move, std::for_each
#include <execution>// std::execution::par
#include <atomic>// std::atomic_size_t

#ifdef _ENABLE_VISUALIZATION
	#define _VISUALIZATION_TYPE , typename visualization_func
	#define _VISUALIZATION_ARG , visualization_func visualize
	#define _VISUALIZATION_ARG_PASS , visualize
	#define _VISUALIZATION_PROCESS(...) visualize(__VA_ARGS__)
#else
	#define _VISUALIZATION_TYPE
	#define _VISUALIZATION_ARG
	#define _VISUALIZATION_ARG_PASS
	#define _VISUALIZATION_PROCESS(...)
#endif

namespace math {
	template<typename scalar>
	using matrix_view = math::mdarray_view<scalar, math::smdsize_t<2>>;

	template<typename scalar>
	using matrix = math::mdarray<scalar, math::smdsize_t<2>>;

	/// requires(...){ std::convertible_to<pred(reduce_energy(...),reduce_energy(...)), bool>; }
	template<typename execution_policy, typename scalar, typename random_engine, typename predicate, typename convolution _VISUALIZATION_TYPE>
	void shuffle_noise(execution_policy &&expo, math::matrix<scalar> &noise, random_engine &rng, const predicate &pred, const convolution &reduce_energy
		_VISUALIZATION_ARG, const size_t max_count = 4096, const double epsilon = 0.001) {
		std::uniform_int_distribution<size_t> which_swap(0, noise.length() - 1);

		math::mdarray<size_t,math::smdsize_t<1>> permutations(noise.length() / 2 * 2);//The 'permutations' is a series of swap-pair, whose size is aligned 2.
		math::mdarray<size_t,math::smdsize_t<1>> in_indices(noise.length());//The 'in_indices' is cache.
		math::mdarray<size_t,math::smdsize_t<1>> out_indices(noise.length());//The 'out_indices' is shuffle result.
		for (size_t i = 0; i != out_indices.length(); ++i) {
			out_indices[i] = i;
		}

		// Forloop do shuffle, result into 'out_indices'.
		std::atomic_size_t swap_counter = 0;
		for (size_t k = 0; k != max_count; ++k) {
			/// Cannot support inplace 'reduce_energy', so we record to in_indices first.
			std::copy(out_indices.begin(), out_indices.end(), 
				in_indices.begin());

			/// Here can be parallel, requires all swap-pair[s] not overlap.
			/// So swap all elements once, count of pairs is half of elements.
			for (size_t i = 0; i != permutations.length(); ++i) {
				permutations[i] = i;
			}
			for (size_t i = 0; i != permutations.length()/2; ++i) {
				std::swap(permutations[i], permutations[which_swap(rng)]);
			}
			/// However, swap all pairs will 'not converge',
			/// so we only swap half pairs.
			std::for_each(expo, (std::pair<size_t, size_t>*)(permutations.begin()), (std::pair<size_t, size_t>*)((permutations.begin() + permutations.length()/2)),
			[&pred, &noise, &reduce_energy, &in_indices, &out_indices, &swap_counter](const std::pair<size_t, size_t>& swap_pair) {
				// spatial index.
				size_t p = swap_pair.first;
				size_t q = swap_pair.second;

				// sample index, may be swap.
				size_t pvalue_i = in_indices[p];
				size_t qvalue_i = in_indices[q];

				auto old_energy = reduce_energy(noise, in_indices, p, pvalue_i) + reduce_energy(noise, in_indices, q, qvalue_i);
				auto new_energy = reduce_energy(noise, in_indices, p, qvalue_i) + reduce_energy(noise, in_indices, q, pvalue_i);
				if (pred(new_energy, old_energy)) {
					// do swap.
					out_indices[p] = qvalue_i;
					out_indices[q] = pvalue_i;
					++swap_counter;
				}
			});
			_VISUALIZATION_PROCESS(k, noise, out_indices, swap_counter.load());

			/// ------+-----------------+-----------------+-----------------+-----------------+-----------------+------------------+
			///  area |      16*16      |      32*32      |      64*64      |     128*128     |     256*256     |     512*512      |
			/// ------+-----------------+-----------------+-----------------+-----------------+-----------------+------------------+
			///       | [   0, 128):2   | [   0, 128):13  | [   0, 128):56  | [   0, 128):233 | [   0, 128):935 | [   0, 128):3753 |
			///  step | [ 128, 256):0   | [ 128, 256):3   | [ 128, 256):15  | [ 128, 256):62  | [ 128, 256):247 | [ 128, 256):996  |
			///  per  | [ 256, 512):0   | [ 256, 512):2   | [ 256, 512):9   | [ 256, 512):40  | [ 256, 512):164 | [ 256, 512):658  |
			///  swap | [ 512,1024):0   | [ 512,1024):1   | [ 512,1024):6   | [ 512,1024):26  | [ 512,1024):105 | [ 512,1024):415  |
			///  count| [1024,2048):0   | [1024,2048):0   | [1024,2048):3   | [1024,2048):15  | [1024,2048):63  | [1024,2048):250  |
			///       | [2048,4096):0   | [2048,4096):0   | [2048,4096):1   | [2048,4096):8   | [2048,4096):36  | [2048,4096):144  |
			/// ------+-----------------+-----------------+-----------------+-----------------+-----------------+------------------+
			/// epsilon dependent on the 'area'.(this table for interval == 1 && reduce_energy == [sig2016, "Blue-noise dithered sampling"])
			constexpr size_t interval = 16;
			if ((k + 1) % interval == 0) {
				if (swap_counter.load() <= noise.length()*epsilon*interval) {
					break;
				}
				swap_counter.store(0);
			}
			std::cout << k << std::endl;
		}

		// Apply shuffle result 'out_indices'.
		math::matrix<scalar> new_noise(noise.size());
		for (size_t i = 0; i != new_noise.length(); ++i) {
			new_noise[i] = noise[out_indices[i]];
		}
		noise = std::move(new_noise);
	}
	
	/// requires(...){ std::convertible_to<pred(reduce_energy(...),reduce_energy(...)), bool>; }
	template<typename scalar, typename random_engine, typename predicate, typename convolution _VISUALIZATION_TYPE>
	void shuffle_noise(math::matrix<scalar> &noise, random_engine &rng, const predicate &pred, const convolution &reduce_energy
		_VISUALIZATION_ARG, const size_t max_count = 4096, const double epsilon = 0.001) {
		math::shuffle_noise(std::execution::seq, noise, rng, pred, reduce_energy
			_VISUALIZATION_ARG_PASS, max_count, epsilon);
	}

	/// published by [Iliyan Georgiev and Marcos Fajardo. 2016, "Blue-noise Dithered Sampling"].
	///@theory
	///		                                      norm(p - q, 2)     norm(S[p] - S[q], d/2)
	///		sum<q=range,p(x,y)!=q(kx,ky)>( exp(- ---------------- - ------------------------) ), S = sampling function.
	///		                                        variance             variance_S              
	template<typename scalar, size_t radius = 6>
	struct blue_noise_kernel {
		scalar spatial_variance = static_cast<scalar>(2.1 * 2.1);
		scalar sample_variance = static_cast<scalar>(1 * 1);

		scalar operator()(const math::matrix_view<scalar>& noise, const math::mdarray<size_t,math::smdsize_t<1>>& sample_indices, size_t spatial_i, size_t sample_i) const {
			const size_t x = spatial_i % noise.size(0);
			const size_t y = spatial_i / noise.size(0);
			scalar energy_sum = 0;
			for (size_t _ky = y - radius; _ky != y + radius; ++_ky) {
				for (size_t _kx = x - radius; _kx != x + radius; ++_kx) {
					size_t kx = (_kx + noise.size(0)) % noise.size(0);
					size_t ky = (_ky + noise.size(1)) % noise.size(1);
					if (kx == x && ky == y) {
						continue;
					}
					
					scalar dx = abs(scalar(kx) - scalar(x));
					scalar dy = abs(scalar(ky) - scalar(y));
					dx = std::min(dx, scalar(noise.size(0)) - dx);
					dy = std::min(dy, scalar(noise.size(1)) - dy);
					scalar sqr_spatial_distance = (dx*dx + dy*dy);
					
					scalar     sample_distance  = sqrt(abs( noise[sample_i] - noise[sample_indices[ky*noise.stride(1) + kx]] ));
					
					energy_sum += exp(-sqr_spatial_distance/spatial_variance - sample_distance/sample_variance);
				}
			}

			return energy_sum;
		}
	};

#if 0
	/// published by [Eric Heitz, Laurent Belcour, Victor Ostromoukhov, David Coeurjolly, and Jean-Claude Iehl. 2019, "A low-discrepancy sampler that distributes Monte Carlo errors as a blue noise in screen space"].
	///@theory
	///                                         norm(p - q, 2)
	///		sum<q=range,p(x,y)!=q(kx,ky)>( exp(- ----------------) * norm(E[q] - E[p], 2) ), E = error_vector (can be heaviside_test<..>).
	///                                           variance
	template<typename integrand, size_t function_count = 1024, size_t radius = 6>
		//requires requires(integrand) { typename integrand::result_type; typename integrand::argument_type; /* integrad(random_engine&) */ }
	struct sequence_scramble_kernel {
		///@optimization decltype( math::sqr(math::vector<T,N>() - math::vector<T,N>()) ) == T.
		math::mdarray<typename integrand::result_type, 2> sample_distance_matrix;

		using scalar = typename integrand::result_type;

		/// requires(...){ sampling(sequence, scrambles, size_t, size_t); 
		///		std::convertible_to<sampling(sequence, scrambles, size_t, size_t), integrand::argument_type>; }
		template<typename execution_policy, typename random_engine, typename sequence_scalar, typename scrambles_scalar, typename sampling>
		sequence_scramble_kernel(execution_policy &&expo, random_engine &rng, const math::vector_view<sequence_scalar>& sequence, const math::matrix_view<scrambles_scalar>& scrambles, const sampling& sample) {
			// Random generate functions, these functions should be uniformly(considering all cases).
			math::svector<integrand,function_count> functions;
			for (size_t i = 0; i != functions.size(); ++i) {
				functions[i] = integrand(rng);
			}

			// Compute error_vectors by functions and sample(sequence,scrambles, ith-index, kth-sample).
			math::vector< math::svector<typename integrand::result_type, function_count> > error_vectors(scrambles.length());
			std::for_each(expo, error_vectors.begin(), error_vectors.end(), [&](auto& error_vector) {
				size_t i = std::distance(&error_vectors[0], &error_vector);
				for (size_t j = 0; j != function_count; ++j) {
					error_vector[j] = 0;
					for (size_t k = 0; k != sequence.size(); ++k) {
						error_vector[j] += functions[j]( sample(sequence,scrambles,i,k) );
					}
					error_vector[j] /= sequence.size();
				}
			});

			// Precompute sample_distances, which is a one-to-one matrix.
			this->sample_distance_matrix.resize(smdsize_t<2>{ scrambles.length(), scrambles.length() });
			std::for_each(expo, this->sample_distance_matrix.begin(), this->sample_distance_matrix.end(), [this,&error_vectors](scalar& sample_distance){
				size_t y = std::distance(&this->sample_distance_matrix[0], &sample_distance);
				size_t x = y % this->sample_distance_matrix.size(0);
				y /= this->sample_distance_matrix.size(1);

				sample_distance = dot(error_vectors[x] - error_vectors[y]);
			});
		}
		
		/// requires(...){ sampling(sequence, scrambles, size_t, size_t); 
		///		std::convertible_to<sampling(sequence, scrambles, size_t, size_t), integrand::argument_type>; }
		template<typename random_engine, typename sequence_scalar, typename scrambles_scalar, typename sampling>
		sequence_scramble_kernel(random_engine& rng, const math::vector_view<sequence_scalar> &sequence, const math::matrix_view<scrambles_scalar> &scrambles, const sampling &sample)
			: sequence_scramble_kernel(std::execution::seq, rng, sequence, scrambles, sample) {}

		/// requires(...){ typename(noise) == typename(function_args<constructor>::scrambles) }
		template<typename scrambles_scalar>
		scalar operator()(const math::matrix_view<scrambles_scalar>& noise, const math::vector_view<size_t>& sample_indices, size_t spatial_i, size_t sample_i) const {
			const size_t x = spatial_i % noise.size(0);
			const size_t y = spatial_i / noise.size(0);
			scalar energy_sum = 0;
			for (size_t _ky = y - radius; _ky != y + radius; ++_ky) {
				for (size_t _kx = x - radius; _kx != x + radius; ++_kx) {
					size_t kx = (_kx + noise.size(0)) % noise.size(0);
					size_t ky = (_ky + noise.size(1)) % noise.size(1);
					if (kx == x && ky == y) {
						continue;
					}
					
					scalar dx = abs(scalar(kx) - scalar(x));
					scalar dy = abs(scalar(ky) - scalar(y));
					dx = std::min(dx, scalar(noise.size(0)) - dx);
					dy = std::min(dy, scalar(noise.size(1)) - dy);
					scalar sqr_spatial_distance = (dx*dx + dy*dy);

					scalar     sample_distance  = this->sample_distance_matrix[{ sample_indices[ky*noise.stride(1) + kx], sample_i }];

					constexpr scalar variance = static_cast<scalar>(2.1 * 2.1);
					energy_sum += exp(-sqr_spatial_distance/variance) * sample_distance;
				}
			}

			return energy_sum;
		}
	};
	
	template<typename T, size_t N>
	struct heaviside_test {
		using result_type   = T;
		using argument_type = math::svector<T,N>;

		///@note We not have a generation of N-dimension uniform directions now,
		math::svector<T,N> pos, dir;
		result_type operator()(const argument_type& x) const {
			return static_cast<result_type>(dot(x - pos, dir) < 0 ? 0 : 1);
		}
		
		heaviside_test() = default;

		template<typename random_engine>
		heaviside_test(random_engine& rng) {
			std::uniform_real_distribution<T> U;
			if constexpr (N == 1) {
				pos = { U(rng) };
				dir = { U(rng) > 0.5 ? T(1) : T(-1) };
			} else if constexpr (N == 2) {
				T angle = U(rng) * 6.2831853f;
				pos = { U(rng), U(rng) };
				dir = { cos(angle), sin(angle) };
			} else {
				// ...
			}
		}
	};
#endif

#if 0
	///@example
	/// shuffle_noise_by_energy(true, noise64x64, 4096, 10, ..); // energy:from 31664 to 31093.6.
	/// shuffle_noise_by_energy(true, noise64x64, 4096, 5, ..); // energy:from 31658.1 to 31058.5.
	/// shuffle_noise_by_energy(true, noise64x64, 4096, 1, ..); // energy:from 31638.1 to 31050.5.
	template<typename predict, typename scalar, typename random_engine, typename binaryfunc>
	void shuffle_noise_by_energy(predict pred, math::multi_array<scalar>& noise, size_t count, size_t shuffle_size, random_engine& rng, binaryfunc reduce_energy /*, GLlibrary2& gl, wex::window_opengl& win, HGLRC opengl_context*/) {
		//std::shared_ptr<GLimage> gl_blue_noise;
		std::uniform_int<size_t> which_swap(0, noise.size() - 1);
		std::vector<std::pair<size_t, size_t>> swap_indices(shuffle_size);
		scalar the_energy = reduce_energy(noise);
		for (size_t i = 0; i != count; ++i) {
			swap_indices.resize(std::max<size_t>(1, rand() % shuffle_size));
			for (auto& swap_pair : swap_indices) {
				swap_pair.first = which_swap(rng);
				swap_pair.second = which_swap(rng);
			}

			for (auto fswap = swap_indices.begin(); fswap != swap_indices.end(); ++fswap) {
				std::swap( noise[fswap->first], noise[fswap->second] );
			}

			scalar new_energy = reduce_energy(noise);
			if (pred(new_energy, the_energy)) {
				the_energy = new_energy;
			} else {
				for (auto rswap = swap_indices.rbegin(); rswap != swap_indices.rend(); ++rswap) {
					std::swap( noise[rswap->first], noise[rswap->second] );
				}
			}

			/*gl.CreateImage({ GL_TEXTURE_2D, GL_R32F, (GLsizei)noise.columns(), (GLsizei)noise.rows() }, { {GL_RED,GL_FLOAT,noise.data()} }, gl_blue_noise);
			wex::imshow(win, *gl_blue_noise, opengl_context);
			wex::waitkey(std::chrono::milliseconds(33));*/
			std::cout << (i+1) << "/" << count << "\tenergy:" << the_energy << "\tswapi:"<<swap_indices.size() << std::endl;
		}
	}
#endif
}
