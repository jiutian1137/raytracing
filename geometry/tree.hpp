#pragma once

/// Tree Optimization of Graph.
///@license Free 
///@review 2023-01-06 
///@contact Jiang1998Nan@outlook.com 
#define _GEOMETRY_TREE_

#include "graph.hpp"
#include <vector>
#include <stack>

namespace geometry {
	template<int traits, typename __vertex_property, typename __graph_property> 
		requires ((traits & tree_constraits) != 0 && (traits & relation_mask) != undirected)
	class adjacency_list<traits, __vertex_property, no_property, __graph_property, void> {
	public:
		static constexpr int relation = (traits & relation_mask);
		using vertex_id       = void;
		using vertex_property = __vertex_property;
		using edge_property   = no_property;
		using graph_property  = __graph_property;

		using vertex_descriptor = size_t;
		struct edge_descriptor { vertex_descriptor source, target; };
		static constexpr vertex_descriptor null_vertex() { 
			return static_cast<vertex_descriptor>(-1);
		}

		struct __directed_vertex_type {
///(1)
/// vertex_descriptor child, skip;
/// 
/// void skippable_traversal(can_skip, do_something) {
///		while (is_end(node)) {
///			if (is_leaf(node)) {
///				do_something(node);
///				node = skip;
///			} else {
///				node = (can_skip(node) ? skip : child);
///			}
///		}
/// }
/// 
/// void traversal(do_something) {
///		for ( ; is_end(node); node = is_leaf(node) ? skip : child) {///Cannot direct traversal.
///			do_something(node);
///		}
/// }
/// 
/// void for_each_children(do_something) {
///		for (node = child; node.skip != skip; node = node.skip) {
///			do_something(node);
///		}
/// }
/// 
///(2)
/// vertex_descriptor next, skip;
/// 
/// void skippable_traversal(can_skip, do_something) {
///		while (is_end(node)) {
/// 		///if (can_skip(node)) {///But leaf not need skip, this coat expensive.
///			///	node = skip;
///			///} else {
///			///	do_something(node);
///			///	node = next;
///			///}
///			if (is_leaf(node)) {
///				do_something(node);
///				node = next;/*assert(skip == next);*/
///			} else {
///				node = (can_skip(node) ? skip : next);
///			}
///		}
/// }
/// 
/// void traversal(do_something) {
///		for ( ; is_end(node); node = next) {///Can direct traversal.
///			do_something(node);
///		}
/// }
/// 
/// void for_each_children(do_something) {
///		for (node = next; node.skip != skip; node = node.skip) {
///			do_something(node);
///		}
/// }
/// 
/// Because: 1. skippable_traversal() both are same, but noted that must check is_leaf first.
/// 2. traversal() not same, (2) better than (1). 
/// 3. for_each_children() both are same.
/// So we use (2).
			vertex_property property_;
			vertex_descriptor next{null_vertex()};
			vertex_descriptor skip{null_vertex()};
			constexpr bool is_leaf() const { return next == skip; }
		};

		struct __bidirected_vertex_type : public __directed_vertex_type {
			vertex_descriptor parent{null_vertex()};
		};

		using vertex_type = std::conditional_t<relation == directed, __directed_vertex_type, __bidirected_vertex_type>;

	public:
		vertex_descriptor add_vertex(const vertex_property& prop) {
			vertices.push_back({prop});
			return vertices.size() - 1;
		}

		vertex_descriptor add_vertex(vertex_property&& prop) {
			vertices.push_back({std::move(prop)});
			return vertices.size() - 1;
		}

		void __set_skip(vertex_descriptor v, vertex_descriptor skip) {
			if (vertices[v].next == vertices[v].skip) { 
				vertices[v].next = skip;
				vertices[v].skip = skip;
			} else {
				auto n = vertices[v].next;
				for ( ; ; n = vertices[n].next) {
					if (vertices[n].skip == vertices[v].skip) {
						vertices[n].skip = skip;
					}

					if (vertices[n].next == vertices[v].skip) {
						vertices[n].next = skip;
						break;
					}
				}
				vertices[v].skip = skip;
			}
		}

		edge_descriptor add_edge(vertex_descriptor s, vertex_descriptor t, const edge_property& = {}) {
			//assert(vertices[t].is_root());
			if (vertices[s].next != null_vertex())
				__set_skip(t, vertices[s].next);///target.next_sibling = source.child.
			vertices[s].next = t;             ///source.child = target.
			
			if constexpr (relation == bidirected) {
				vertices[t].parent = s;
			}
			return {s, t};
		}
		
		void remove_edge(vertex_descriptor s, vertex_descriptor t) {
			if constexpr (relation == bidirected) {
				//assert(vertices[t].parent == s);
				vertices[t].parent = null_vertex();
			}

			if (vertices[s].next == t) { 
				vertices[s].next = vertices[t].skip;///source.child = target.next_sibling.
				__set_skip(t, null_vertex());       ///target.next_sibling = null.
			} else {
				auto p = vertices[s].next;
				for( ; vertices[p].skip != t; ) 
					p = vertices[p].skip;
				__set_skip(p, vertices[t].skip);    ///target.prev_sibling.next_sibling = target.next_sibling.
				__set_skip(t, null_vertex());       ///target.next_sibling = null.
			}
		}

		void remove_vertex(vertex_descriptor v) {
			if constexpr (relation == bidirected) {
				if (vertices[v].parent != null_vertex()) {
					remove_edge(vertices[v].parent, v);
				}
				for (auto n = vertices[v].next; n != vertices[v].skip; n = vertices[n].skip) {
					vertices[n].parent = null_vertex();
				}
			} else {
				for (vertex_type& source : vertices) {
					if (source.next == v) {
						source.next = vertices[v].skip;
						__set_skip(v, null_vertex());
						break;
					} else if (source.skip == v) {
						__set_skip(std::distance(&vertices[0], &source), vertices[v].skip);
						__set_skip(v, null_vertex());
						break;
					}
				}
			}

			vertices.erase(std::next(vertices.begin(), v));
			for (vertex_type& vi : vertices) {
				if (vi.next != null_vertex() && vi.next > v) {
					--vi.next;
				}
				if (vi.skip != null_vertex() && vi.skip > v) {
					--vi.skip;
				}
				if constexpr (relation == bidirected) {
					if (vi.parent != null_vertex() && vi.parent > v) {
						--vi.parent;
					}
				}
			}
		}

		void clear() noexcept {
			vertices.clear();
		}

		bool empty() const noexcept {
			return vertices.empty();
		}

		const vertex_type& vertex(vertex_descriptor v) const {
			return vertices[v];
		}

		vertex_type& vertex(vertex_descriptor v) {
			return vertices[v];
		}

		const vertex_property& operator[](vertex_descriptor v) const {
			return vertices[v].property_;
		}

		vertex_property& operator[](vertex_descriptor v) {
			return vertices[v].property_;
		}

		std::vector<vertex_type> vertices;
		graph_property property_;
	};

	/* requires std::is_same<graph::vertex_property, std::pair<boundary, span<value_type>>> */
	template<typename graph, typename value_iterator, 
		typename full_bound_fn, typename partition_by_boundary_fn>
	graph make_boundary_volume_hierarchy(value_iterator values_first, value_iterator values_last, 
		const full_bound_fn& full_bound, const partition_by_boundary_fn& partition, const size_t max_values = 2) 
	{
		graph bvh;

		struct stack_value {
			size_t depth;
			typename graph::vertex_descriptor node;
			value_iterator first, last;
		};

		const size_t max_depth = size_t(log2( std::max(std::distance(values_first, values_last)/max_values, size_t(1)) ));
		std::stack<stack_value> stack;
		stack.push(stack_value{.depth=0, .node=bvh.add_vertex({}), .first=values_first, .last=values_last});
		while (!stack.empty()) {
			size_t depth = stack.top().depth;
			auto   node  = stack.top().node;
			auto   first = stack.top().first;
			auto   last  = stack.top().last;
			stack.pop();

			bvh[node].first = full_bound(first, last);
			bvh[node].second = {first, last};
			if (depth == max_depth || std::next(first) == last) {
				// do nothing.
			} else {
				auto child0 = bvh.add_vertex({});
				auto child1 = bvh.add_vertex({});
				bvh.add_edge(node, child0);
				bvh.add_edge(node, child1);
				auto mid = partition(first, last, bvh[node].first);
				stack.push(stack_value{.depth=depth+1, .node=child0, .first=first, .last=mid});
				stack.push(stack_value{.depth=depth+1, .node=child1, .first=mid, .last=last});
			}
		}

		return std::move(bvh);
	}
}

#if 0
#include <geometry/tree.hpp>
#include <iostream>

int main() {
	geometry::adjacency_list<geometry::tree_constraits/*|geometry::directed*/, int> Tree;

	auto v0 = Tree.add_vertex(1);
	auto v1 = Tree.add_vertex(2);
	auto v2 = Tree.add_vertex(3);
	auto v3 = Tree.add_vertex(4);
	auto v4 = Tree.add_vertex(5);
	Tree.add_edge(v0, v1);
	Tree.add_edge(v0, v3);
	Tree.add_edge(v1, v2);
	Tree.add_edge(v2, v4);
	
	Tree.remove_vertex(v1);
	//Tree.remove_edge(v0, v1);
	/*Tree.add_edge(v0, v1);

	Tree.remove_vertex(v3);*/
	//Tree.remove_vertex(v2-1);

	size_t node = 0;
	for ( ; node != Tree.null_vertex(); node = Tree.vertex(node).next) {
		std::cout << Tree[node] << std::endl;
	}
	

	return 0;
}
#endif