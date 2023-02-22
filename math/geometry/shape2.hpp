#pragma once

#include <cassert>
#include <typeinfo>

#include <intrin.h>
#include <bitset>
#include <vector>
#include <algorithm>
#include <span>

#include "shape.hpp"
#include <geometry/tree.hpp>

namespace math { namespace geometry {
///A vertex always contains more than one attributes. There are variety of arrangements, which
///(1) unsable to list all,
///(2) unification requires performance costs and increased complexity.
/// 
///This is similar to the situation of std::containers. It is
///(1) not possible to list all,
///(2) but always enough.
/// 
///		{ position:[], attributes:[{ normal:..., texcoord[0]:..., velocity:... }] }
/// 
///		[{ position:..., normal:..., texcoord[0]:..., velocity:... }]
/// 
///		...
#if 0
	template<typename position, typename... attributes>
	struct /*general_*/triangles {
		std::vector<position> pos;
		std::vector<std::tuple<attributes...>> attrs;
	};

	template<typename position, typename... attributes>
	struct /*structured_*/triangles {
		std::vector < std::tuple<position, attributes...> attrs;
	};

	template<typename position, typename... attributes>
	struct /*seperated_*/triangles : std::tuple<std::vector<attributes>...> {
		std::vector<position> pos;
		std::tuple<std::vector<attributes>...> attrs;
	};
#endif
///These static structures are very simple, inline implementation is better.
///So I only implement a dynamic structure.
/// 
///Another problem does not occur in the situation of std::container, is
///how to unify the arrangement of many attributes. (this unification is the interface level.) 
/// 
///		auto get_attribute(string semantic, function f) { 
///			for (attributes_i : attributes) 
///				if (attributes_i.semantic == semantic) 
///					return f( attributes_i );
///		}
/// 
///Group by semantics.


///
///		std::map<string, size_t> semantic_offsets;
///		size_t i = semantic_offsets[ "TEXCOORD" ];
/// 
/// Replace search-by-string by search-by-index.
/// 
///		std::vector<string> semantics;
///		std::flat_map<size_t, size_t> semantic_offsets;
///		size_t i = semantic_offsets[ *std::ranges::find(semantics, "TEXCOORD") ];
/// 
///And because (1) Semantics are not sparse for all geometry. (2) The scope of semantics is determined. so,
///		
///		constexpr std::vector<string> semantics;
///		std::vector<size_t> semantic_offsets;
///		size_t i = semantic_offsets[ *std::ranges::find(semantics, "TEXCOORD") ];
///
	constexpr std::string_view semantics[6] = { 
		"position",//unique
		"normal",  //unique
		"tangent", //unique
		"texcoord",//one is coord[0]. for reuse a image, another parameter coord[1] is need to sample the image.
		"other"
		"other_i"
	};

	using semantic_index_t = size_t;

	using semantic_t = std::pair<semantic_index_t, size_t>;
	
	constexpr semantic_index_t find_semantic(const std::string_view semantic) {
		for (size_t i = 0; i != std::size(semantics); ++i) {
			size_t j = 0;
			for ( ; j != semantics[i].size(); ++j)
				if (semantic[j] != semantics[i][j])
					break;

			if (semantic.size() > semantics[i].size()) {
				if (j == semantic.size())
					return i;
			} else {
				if (j == semantics[i].size())
					return i;
			}
		}

		return static_cast<semantic_index_t>(-1);
	}


	template<size_t num_semantics = 4>
	struct vertices {
		struct attribute_pointer {
			struct type_info {
				size_t hash_code;// = typeid(T).hash_code();
				bool normalized;
				size_t size;     // = std::size(T) for array;
				size_t alignment;// = alignof(T);
				size_t length;   // = sizeof(T);
			} type;
			struct buffer_type {
				unsigned char* data;
				size_t stride;
			} stream;

			template<typename T>
			const T& ref(size_t i) const {
				assert(sizeof(T) <= this->type.length);
				assert(alignof(T) <= this->type.alignment);
				return reinterpret_cast<const T&>(this->stream.data + this->stream.stride * i);
			}

			template<typename T>
			T& ref(size_t i) {
				return static_cast<T&>(*this)[i];
			}

			template<typename T>
			const T* ptr(size_t i) const {
				return reinterpret_cast<const T*>(this->stream.data + this->stream.stride * i);
			}

			template<typename T>
			T* ptr(size_t i) {
				return reinterpret_cast<T*>(this->stream.data + this->stream.stride * i);
			}
		};
		std::vector<std::shared_ptr<std::vector<unsigned char>>> buffers;
		std::vector<attribute_pointer> vertex_attributes; size_t num_vertices{0};//std::vector<std::vector<attribute_binding>> dynamic_vertex_attributes;
		size_t semantic_offsets[num_semantics]={size_t(-1),size_t(-1),size_t(-1),size_t(-1)};

		bool empty() const {
			return num_vertices == 0;
		}

		bool empty(semantic_index_t i) const {
			return !(semantic_offsets[i] < vertex_attributes.size());
		}

		size_t size() const {
			return vertex_attributes.size();
		}

		size_t size(semantic_index_t i) const {
			assert( semantic_offsets[i] < vertex_attributes.size() );
			size_t first = semantic_offsets[i++];
			for ( ; i < num_semantics; ++i)
				if (semantic_offsets[i] < vertex_attributes.size())
					return semantic_offsets[i] - first;
			return vertex_attributes.size() - first;
		}

		const attribute_pointer& operator[](const semantic_index_t i) const {
			assert( i < vertex_attributes.size() );
			return vertex_attributes[i];
		}

		const attribute_pointer& operator[](const semantic_t x) const {
			assert( semantic_offsets[x.first] != vertex_attributes.size() );
			return vertex_attributes[semantic_offsets[x.first] + x.second];
		}

		void push_back(semantic_index_t i, const typename attribute_pointer::type_info type) {
			const size_t semantic0 = i++;
			const bool semantic0_empty = !(semantic_offsets[semantic0] < vertex_attributes.size());
			for ( ; i < num_semantics; ++i) 
				if (semantic_offsets[i] < vertex_attributes.size()) {
					vertex_attributes.insert(std::next(vertex_attributes.begin(), semantic_offsets[i]), attribute_pointer{type, nullptr, 0});

					if (semantic0_empty) semantic_offsets[semantic0] = i;
					for ( ; i < num_semantics; ++i) 
						if (semantic_offsets[i] < vertex_attributes.size()) 
							++semantic_offsets[i];
					return;
				}
			if (semantic0_empty) semantic_offsets[semantic0] = vertex_attributes.size();
			vertex_attributes.push_back(attribute_pointer{type, nullptr, 0});
		}

		template<typename attribute_type>
		void push_back(semantic_index_t i) {
			using scalar_type = std::remove_cvref_t<decltype(std::declval<attribute_type>()[size_t()])>;
			push_back(i, {typeid(scalar_type).hash_code(), false,
				sizeof(attribute_type)/sizeof(scalar_type), alignof(attribute_type), sizeof(attribute_type)});
		}
	
		void erase(semantic_t) {
			abort();
		}

		void __tidy_buffers() {
			/// ...
		}

		void bind_stream(semantic_t index, std::shared_ptr<std::vector<unsigned char>> buffer, size_t offset, size_t length, size_t stride = 0) {
			auto& attrib = vertex_attributes[semantic_offsets[index.first] + index.second];
			attrib.stream.data = std::next(buffer->data(), offset);
			attrib.stream.stride = stride != 0 ? stride : std::max(attrib.type.alignment, attrib.type.length);
			if (num_vertices == 0) {
				num_vertices = length / attrib.stream.stride;
			} else {
				if (num_vertices != length / attrib.stream.stride)
					throw;
			}
		
			if (std::find(buffers.begin(), buffers.end(), buffer) == buffers.end()) 
				buffers.push_back(buffer);
			__tidy_buffers();
		}

		void bind_stream(semantic_index_t index, std::shared_ptr<std::vector<unsigned char>> buffer, size_t offset, size_t length, size_t stride = 0) {
			bind_stream({ index, 0 }, buffer, offset, length, stride);
		}
	};

	template<typename element_index_type, size_t num_semantics = 4>
	struct triangles : vertices<num_semantics> {
		using base = vertices<num_semantics>;
		using base::buffers;
		using base::vertex_attributes;
		using base::semantic_offsets;
		using base::__tidy_buffers;
		std::span<element_index_type> element_indices;

		void bind_index_stream(std::shared_ptr<std::vector<unsigned char>> buffer, size_t offset, size_t length) {
			element_indices = std::span(
				reinterpret_cast<element_index_type*>( std::next(buffer->data(), offset) ),
				length / sizeof(element_index_type)
			);

			if (std::find(buffers.begin(), buffers.end(), buffer) == buffers.end())
				buffers.push_back(buffer);
			__tidy_buffers();
		}
	
		template<typename attribute_type>
		void fetch(const semantic_t x, const element_index_type& v, attribute_type* three_atttr) const {
			assert(semantic_offsets[x.first] < vertex_attributes.size());
			assert(semantic_offsets[x.first] + x.second < vertex_attributes.size());
			const auto& __attribute = base::operator[](x);

			if constexpr (std::is_scalar_v<attribute_type>) {
				three_atttr[0] = __attribute.ref<attribute_type>( v[0] );
				three_atttr[1] = __attribute.ref<attribute_type>( v[1] );
				three_atttr[2] = __attribute.ref<attribute_type>( v[2] );
			} else {
				using scalar_type = value_t<attribute_type>;
				load(__attribute.ptr<scalar_type>( v[0] ), three_atttr[0]);
				load(__attribute.ptr<scalar_type>( v[1] ), three_atttr[1]);
				load(__attribute.ptr<scalar_type>( v[2] ), three_atttr[2]);
			}
		}

		template<typename attribute_type>
		void fetch(const semantic_index_t s, const element_index_type& v, attribute_type* three_atttr) const {
			fetch(semantic_t{s,0}, v, three_atttr);
		}

#if 0
#include <iostream>
#include <memory>
#include <vector>

#include <math/geometry/triangles.hpp>
#include <math/mdarray_vector.hpp>

int main() {
	using vector3 = math::smdarray<float, 3>;
	using index3 = math::smdarray<unsigned int, 3>;

	math::geometry::triangles<index3> X;

	auto positions = std::make_shared< std::vector<unsigned char> >(sizeof(vector3) * 4);
	reinterpret_cast<vector3*>(positions->data())[0] = { 0,0,0 };
	reinterpret_cast<vector3*>(positions->data())[1] = { 1,0,0 };
	reinterpret_cast<vector3*>(positions->data())[2] = { 1,1,0 };
	reinterpret_cast<vector3*>(positions->data())[3] = { 0,1,0 };
	X.push_back<vector3>(math::geometry::find_semantic("position"));
	X.bind_stream(math::geometry::find_semantic("position"), positions, 0, positions->size());

	auto normals = std::make_shared< std::vector<unsigned char> >(sizeof(vector3) * 4, '\0');
	reinterpret_cast<vector3*>(normals->data())[0] = { 0,0,1 };
	reinterpret_cast<vector3*>(normals->data())[1] = { 0,0,1 };
	reinterpret_cast<vector3*>(normals->data())[2] = { 0,0,1 };
	reinterpret_cast<vector3*>(normals->data())[3] = { 0,0,1 };
	X.push_back<vector3>(math::geometry::find_semantic("normal"));
	X.bind_stream(math::geometry::find_semantic("normal"), normals, 0, normals->size());

	auto indices = std::make_shared< std::vector<unsigned char> >(sizeof(index3) * 2, '\0');
	reinterpret_cast<index3*>(indices->data())[0] = { 0,1,2 };
	reinterpret_cast<index3*>(indices->data())[1] = { 2,3,0 };
	X.bind_index_stream(indices, 0, indices->size());

	using aligned_vector3 = math::smdarray<float, 3, __m128>;
	aligned_vector3 y[3];
	X.fetch(math::geometry::find_semantic("position"), X.element_indices[0], y);
	std::cout << y[0] << "," << y[1] << "," << y[2] << std::endl;
	X.fetch(math::geometry::find_semantic("position"), X.element_indices[1], y);
	std::cout << y[0] << "," << y[1] << "," << y[2] << std::endl;

	return 0;
}
#endif

		template<typename graph, typename attribute_type>
		graph build_bvh(const semantic_t index) const {
			assert(semantic_offsets[index.first] < vertex_attributes.size());
			assert(semantic_offsets[index.first] + index.second < vertex_attributes.size());
			const auto& __attribute = vertex_attributes[semantic_offsets[index.first] + index.second];
			assert( !__attribute.type.normalized );

			using bound_volume_type = typename graph::vertex_property;
			using bound_type = typename bound_volume_type::first_type;
			using scalar_type = value_t<attribute_type>;

			return ::geometry::make_boundary_volume_hierarchy<graph>(this->element_indices.begin(), this->element_indices.end(),
				[&__attribute](auto first, auto last) {
					attribute_type p0, p1, p2;
					load(__attribute.ptr<scalar_type>( (*first)[0] ), p0);
					load(__attribute.ptr<scalar_type>( (*first)[1] ), p1);
					load(__attribute.ptr<scalar_type>( (*first)[2] ), p2);
					auto boundary = bound_type::from(p0, p1, p2);
					for (auto seek = std::next(first); seek != last; ++seek) {
						load(__attribute.ptr<scalar_type>( (*seek)[0] ), p0);
						load(__attribute.ptr<scalar_type>( (*seek)[1] ), p1);
						load(__attribute.ptr<scalar_type>( (*seek)[2] ), p2);
						boundary = expand(boundary, bound_type::from(p0, p1, p2));
					}

					for (size_t i = 0; i != 3; ++i) {
						if (boundary.l[i] == boundary.u[i]) {
							boundary.l[i] -= 1e-5f;
							boundary.u[i] += 1e-5f;
						}
					}

					return boundary;
				},
				[&__attribute](auto first, auto last, const bound_type& boundary_) {
					auto minNum = std::numeric_limits<scalar_type>::lowest();
					auto maxNum = std::numeric_limits<scalar_type>::max();
					//	x.first = {attribute_type{maxNum,maxNum,maxNum}, attribute_type{minNum,minNum,minNum}};
					bound_type boundary = {attribute_type{maxNum,maxNum,maxNum}, attribute_type{minNum,minNum,minNum}};
					for (auto seek = first; seek != last; ++seek) {
						attribute_type p0, p1, p2;
						load(__attribute.ptr<scalar_type>( (*seek)[0] ), p0);
						load(__attribute.ptr<scalar_type>( (*seek)[1] ), p1);
						load(__attribute.ptr<scalar_type>( (*seek)[2] ), p2);
						boundary = expand(boundary, center(bound_type::from(p0, p1, p2)));
					}
					for (size_t i = 0; i != 3; ++i) {
						if (boundary.l[i] == boundary.u[i]) {
							boundary.l[i] -= 1e-5f;
							boundary.u[i] += 1e-5f;
						}
					}

					auto   sides    = boundary.size();
					size_t max_side = (sides[0] > sides[1] && sides[0] > sides[2]) ? 0 : (sides[1] > sides[0] && sides[1] > sides[2]) ? 1 : 2;
					/*std::nth_element(first, std::next(first,std::distance(first,last)/2), last, [&](auto& a, auto& b){
						attribute_type p0, p1, p2, q0, q1, q2;
						load(__attribute.ptr<T>( a[0] ), p0);
						load(__attribute.ptr<T>( a[1] ), p1);
						load(__attribute.ptr<T>( a[2] ), p2);
						load(__attribute.ptr<T>( b[0] ), q0);
						load(__attribute.ptr<T>( b[1] ), q1);
						load(__attribute.ptr<T>( b[2] ), q2);
						return center(bound_type::from(p0, p1, p2))[max_side] < center(bound_type::from(q0, q1, q2))[max_side];
						});
					return std::next(first, std::distance(first,last)/2);*/
					/*auto mid = std::partition(first, last, [&](const auto& i) {
							attribute_type p0, p1, p2;
							load(__attribute.ptr<T>( i[0] ), p0);
							load(__attribute.ptr<T>( i[1] ), p1);
							load(__attribute.ptr<T>( i[2] ), p2);
							return center(bound_type::from(p0, p1, p2))[max_side] < center(boundary)[max_side];
						});
					return (mid == first || mid == last) ? std::next(first, std::distance(first,last)/2) : mid;*/
#if 1
					std::pair<bound_type, size_t> buckets[12];
					for (auto& x : buckets) {
						auto minNum = std::numeric_limits<scalar_type>::lowest();
						auto maxNum = std::numeric_limits<scalar_type>::max();
						x.first = {attribute_type{maxNum,maxNum,maxNum}, attribute_type{minNum,minNum,minNum}};
						x.second = 0;
					}

					std::for_each(first, last, [&](auto& element_index) {
						attribute_type p0, p1, p2;
						load(__attribute.ptr<scalar_type>( element_index[0] ), p0);
						load(__attribute.ptr<scalar_type>( element_index[1] ), p1);
						load(__attribute.ptr<scalar_type>( element_index[2] ), p2);
						auto f = (center(bound_type::from(p0, p1, p2)) - boundary.begin()) / boundary.size();
						size_t b = min(int(f[max_side] * 12), int(12-1));
						buckets[b].first = expand(buckets[b].first, bound_type::from(p0, p1, p2));
						buckets[b].second += 1;
					});

					scalar_type cost[12 - 1];
					for (size_t i = 0; i != 12 - 1; ++i) {
						auto minNum = std::numeric_limits<scalar_type>::lowest();
						auto maxNum = std::numeric_limits<scalar_type>::max();
						bound_type b0 = {attribute_type{maxNum,maxNum,maxNum}, attribute_type{minNum,minNum,minNum}},
							b1 = {attribute_type{maxNum,maxNum,maxNum}, attribute_type{minNum,minNum,minNum}};
						size_t count0 = 0, count1 = 0;
						for (size_t j = 0; j <= i; ++j) {
								b0 = expand(b0, buckets[j].first);
								count0 += buckets[j].second;
						}
						for (size_t j = i + 1; j < 12; ++j) {
								b1 = expand(b1, buckets[j].first);
								count1 += buckets[j].second;
						}
						auto x = b0.size();
						auto A0 = 2*(x[0]*x[1] + x[0]*x[2] + x[1]*x[2]);
							x = b1.size();
						auto A1 = 2*(x[0]*x[1] + x[0]*x[2] + x[1]*x[2]);
							x = boundary.size();
						auto A = 2*(x[0]*x[1] + x[0]*x[2] + x[1]*x[2]);

						cost[i] = 1 + (count0 * A0 + count1 * A1)/A;
					}

					// Find bucket to split at that minimizes SAH metric
					auto minCost = cost[0];
					size_t minCostSplitBucket = 0;
					for (size_t i = 1; i < 12 - 1; ++i) {
							if (cost[i] < minCost) {
									minCost = cost[i];
									minCostSplitBucket = i;
							}
					}

					// Either create leaf or split primitives at selected SAH
					// bucket
					scalar_type leafCost = (scalar_type)std::distance(first,last);
					auto mid = std::partition(first, last, [&](const auto &element_index) {
							attribute_type p0, p1, p2;
							load(__attribute.ptr<scalar_type>( element_index[0] ), p0);
							load(__attribute.ptr<scalar_type>( element_index[1] ), p1);
							load(__attribute.ptr<scalar_type>( element_index[2] ), p2);
							auto f = (center(bound_type::from(p0, p1, p2)) - boundary.begin()) / boundary.size();
							size_t b = min(int(f[max_side] * 12), int(12-1));
							return b <= minCostSplitBucket;
						});
					return (mid == first || mid == last) ? std::next(first, std::distance(first,last)/2) : mid;
#endif
				}
			);
		}

		template<typename graph, typename attribute_type>
		graph build_bvh(const semantic_index_t s) const {
			return build_bvh<graph, attribute_type>(semantic_t{ s, 0 });
		}

#if 0
	auto tree = X.build_bvh<geometry::adjacency_list<geometry::tree_constraits, std::pair<geometry::range<aligned_vector3>, std::span<index3>>>, 
		aligned_vector3>(math::geometry::find_semantic("position"));

	math::geometry::ray<aligned_vector3> ray{ aligned_vector3{0.001f,0.001f,-5}, aligned_vector3{0,0,1} };
	math::geometry::intersection_result<aligned_vector3> result{.t = 10000};
	if (intersection(X, tree, math::geometry::find_semantic("position"), ray, 0.0f, result)) {
		std::cout << "hit " << result.t << std::endl;
		aligned_vector3 h;
		intersection(math::geometry::triangles2<aligned_vector3,index3>{X, tree}, result.id, ray(result.t), 
			math::geometry::semantic_t{math::geometry::find_semantic("normal"),0}, &h);
		std::cout << h << std::endl;
		
	} else {
		std::cout << "missing" << std::endl;
	}
#endif
	};

	template<typename vectorN, typename index3, semantic_index_t s = 0, size_t M = 4>
	struct triangles2 : triangles<index3, M> {
		using base = triangles<index3, M>;
		using bvh_type = ::geometry::adjacency_list<::geometry::tree_constraits, std::pair<range<vectorN>, std::span<index3>>>;

		bvh_type bvh;

		using base::build_bvh;

		void build_bvh() {
			bvh = static_cast<base&>(*this).build_bvh<bvh_type>(s);
		}
	};

	template<size_t num_semantics = 4>
	struct n_gons : vertices<num_semantics> {
		using element_index_type = std::span<size_t>;
		std::vector<element_index_type> element_index;

		//...
	};
	

	template<typename vectorN>
	struct intersection_result {
		size_t id;
		value_t<vectorN> t;

		value_t<vectorN> distance() const { return t; }
	};

	template<typename index3, size_t M, typename graph, typename vectorN>
	bool intersection(const triangles<index3,M>& tri, const graph& bvh, const semantic_t x, const ray<vectorN>& ray, const value_t<vectorN> lower, intersection_result<vectorN>& result) {
		using scalar = value_t<vectorN>;
		const scalar t0 = result.t;
		typename graph::vertex_descriptor node = 0;
		while (node != bvh.null_vertex()) {
			bool can_skip = disjoint(range{lower, result.t}, intersection(bvh[node].first, ray));
			if (!can_skip && bvh.vertex(node).is_leaf()) {
				for (const index3& indices : bvh[node].second) {
					vectorN p[3]; tri.fetch(x, indices, p);
					scalar t = intersection(triangle<vectorN>::from_points(p[0], p[1], p[2]), ray);
					if (lower < t && t < result.t) {
						result.id = (&indices) - (&tri.element_indices[0]);
						result.t = t;
					}
				}
			}
			node = can_skip ? bvh.vertex(node).skip : bvh.vertex(node).next;
#if 0
			if (bvh.vertex(node).is_leaf()) {

				if (bvh[node].second.size() == 1) {
					const auto& indices = bvh[node].second[0];
					attribute_type p0 = get_position(indices[0]), p1 = get_position(indices[1]), p2 = get_position(indices[2]);
					auto element_i = triangle<vector3>::from_points(as<vector3>(p0), as<vector3>(p1), as<vector3>(p2));
					if (dot(ray.direction(), cross(element_i.ori.e[0], element_i.ori.e[1])) < 0) {
						scalar t = intersection(element_i, ray);
						if (0 <= t && t < result.distance) {
							result.primitive = this;
							result.element   = /*std::distance*/(&indices) - (&elements[0]);
							result.distance  = t;
						}
					}
				} else if (!before(result.distance, intersection(bvh[node].first, ray))) {
					for (const auto& indices : bvh[node].second) {
						attribute_type p0 = get_position(indices[0]), p1 = get_position(indices[1]), p2 = get_position(indices[2]);
						auto element_i = triangle<vector3>::from_points(as<vector3>(p0), as<vector3>(p1), as<vector3>(p2));
						if (dot(ray.direction(), cross(element_i.ori.e[0], element_i.ori.e[1])) < 0) {
							scalar t = intersection(element_i, ray);
							if (0 <= t && t < result.distance) {
								result.primitive = this;
								result.element   = /*std::distance*/(&indices) - (&elements[0]);
								result.distance  = t;
							}
						}
					}
				}

				node = bvh.vertex(node).next;
			} else {
				auto section = intersection(bvh[node].first, ray);
				bool can_skip = (empty(section) || result.distance < section.begin());
				node = (can_skip ? bvh.vertex(node).skip : bvh.vertex(node).next);
			}
#endif
		}

		return result.t != t0;
	}

	template<typename index3, size_t M, typename graph, typename vectorN>
	bool intersection(const triangles<index3,M>& tri, const graph& bvh, const semantic_index_t s, const ray<vectorN>& ray, const value_t<vectorN> lower, intersection_result<vectorN>& result) {
		return intersection(tri, bvh, semantic_t{s,0}, ray, lower, result);
	}

	template<typename vectorN, typename index3, semantic_index_t s, size_t M>
	bool intersection(const triangles2<vectorN,index3,s,M>& tri, const ray<vectorN>& ray, const value_t<vectorN> lower, intersection_result<vectorN>& result) {
#if 0
		return intersection(tri, tri.bvh, s, ray, lower, result);
#else //inline optimization.
		using scalar = value_t<vectorN>;
		const scalar t0 = result.t;
		typename triangles2<vectorN,index3,s,M>::bvh_type::vertex_descriptor node = 0;
		while (node != tri.bvh.null_vertex()) {
			bool can_skip = disjoint(range{lower, result.t}, intersection(tri.bvh[node].first, ray));
			if (!can_skip && tri.bvh.vertex(node).is_leaf()) {
				for (const index3& indices : tri.bvh[node].second) {
					vectorN p[3]; tri.fetch(s, indices, p);
					scalar t = intersection(triangle<vectorN>::from_points(p[0], p[1], p[2]), ray);
					if (lower < t && t < result.t) {
						result.id = (&indices) - (&tri.element_indices[0]);
						result.t = t;
					}
				}
			}
			node = can_skip ? tri.bvh.vertex(node).skip : tri.bvh.vertex(node).next;
		}

		return result.t != t0;
#endif
	}
	
	template<typename vectorN, typename index3, semantic_index_t s, size_t M>
	auto intersection(const triangles2<vectorN,index3,s,M>& tri, size_t id, const ray<vectorN>& ray) {
		vectorN p[3]; tri.fetch(s, tri.element_indices[id], p);
		return intersection(triangle<vectorN>::from_points(p[0], p[1], p[2]), ray);
	}

	template<typename vectorN, typename index3, semantic_index_t s, size_t M, typename attribute_type>
	auto intersection(const triangles2<vectorN,index3,s,M>& tri, size_t id, const vectorN X, const semantic_t x, attribute_type* attrib) {
		vectorN p[3]; tri.fetch(s, tri.element_indices[id], p);
		auto u = intersection(triangle<vectorN>::from_points(p[0], p[1], p[2]), X);
		attribute_type a[3]; tri.fetch(x, tri.element_indices[id], a);
		(*attrib) = triangle<vectorN>::lerp_from_points(a[0], a[1], a[2], u);
		return u;
	}

	template<typename vectorN, typename index3, semantic_index_t s, size_t M, typename attribute_type>
	auto intersection(const triangles2<vectorN,index3,s,M>& tri, size_t id, const vectorN X, const std::vector<semantic_t>& x, attribute_type attrib[]) {
		vectorN p[3]; tri.fetch(s, tri.element_indices[id], p);
		auto u = intersection(triangle<vectorN>::from_points(p[0], p[1], p[2]), X);
		for (const auto& x_i : x) {
			attribute_type a[3]; tri.fetch(x_i, tri.element_indices[id], a);
			*attrib++ = triangle<vectorN>::lerp_from_points(a[0], a[1], a[2], u);
		}
		return u;
	}
}}//end of math::geometry