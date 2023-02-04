#pragma once

/// Graph Strcuture.
///@license Free 
///@review 2022-6-7 
///@author LongJiangnan, Jiang1998Nan@outlook.com 
#define _GEOMETRY_GRAPH_

#include <cassert>
#include <utility>/* std::pair<,> */
#include <memory>/* std::shared_ptr<> */

#include <stack>
#include <queue>

#include <vector>
#include <list>
#include <map>
#if !(defined _HAS_CXX23 && _HAS_CXX23 != 0)
namespace std {
	template<class Key, class T, class Compare = less<Key>,
		class KeyContainer = vector<Key>, class MappedContainer = vector<T>>
	class flat_map {
	public:
		// types
		using key_type               = Key;
		using mapped_type            = T;
		using value_type             = pair<key_type, mapped_type>;
		using key_compare            = Compare;
		using reference              = pair<const key_type&, mapped_type&>;
		using const_reference        = pair<const key_type&, const mapped_type&>;
		using size_type              = size_t;
		using difference_type        = ptrdiff_t;
		using key_container_type     = KeyContainer;
		using mapped_container_type  = MappedContainer;

		class value_compare {
		private:
			key_compare comp;                                 // exposition only
			value_compare(key_compare c) : comp(c) { }        // exposition only
		public:
			bool operator()(const_reference x, const_reference y) const {
				return comp(x.first, y.first);
			}
		};
 
		struct containers {
			key_container_type keys;
			mapped_container_type values;
		};

		containers c;               // exposition only
		key_compare compare;        // exposition only

		struct const_iterator {
			using iterator_concept  = std::random_access_iterator_tag;
			using iterator_category = std::random_access_iterator_tag;
			using value_type        = flat_map::value_type;
			using difference_type   = flat_map::difference_type;
			using pointer           = void;
			using reference         = flat_map::const_reference;

			reference operator*() const {
				return reference(*_Ptr, 
					*std::next(_Cont->c.values.begin(), 
						std::distance(_Cont->c.keys.begin(), _Ptr)) );
			}

			reference operator[](size_type pos) const {
				return reference(_Ptr[pos],
					*std::next(_Cont->c.values.begin(), 
						std::distance(_Cont->c.keys.begin(), _Ptr) + pos) );
			}
			
			const_iterator& operator++() {
				++_Ptr;
				return *this;
			}

			const_iterator operator++(int) {
				auto _Copy = *this;
				++(*this);
				return _Copy;
			}

			const_iterator& operator--() {
				--_Ptr;
				return *this;
			}

			const_iterator operator--(int) {
				auto _Copy = *this;
				--(*this);
				return _Copy;
			}

			const_iterator& operator+=(difference_type diff) {
				_Ptr += diff;
				return *this;
			}

			const_iterator& operator-=(difference_type diff) {
				_Ptr -= diff;
				return *this;
			}

			const_iterator operator+(difference_type diff) const {
				return const_iterator(*this) += diff;
			}

			const_iterator operator-(difference_type diff) const {
				return const_iterator(*this) -= diff;
			}

			bool operator==(const const_iterator& right) const {
				assert(_Cont == right._Cont);
				return _Ptr == right._Ptr;
			}

			bool operator!=(const const_iterator& right) const {
				return !((*this) == right);
			}

			difference_type operator-(const const_iterator& right) const {
				assert(_Cont == right._Cont);
				return _Ptr - right._Ptr;
			}

			typename key_container_type::const_iterator _Ptr;
			const flat_map* _Cont;
		};
		struct iterator : public const_iterator {
			using _Base = const_iterator;
			using reference = flat_map::reference;

			reference operator*() const {
				return reference(*_Base::_Ptr,
					const_cast<mapped_type&>(*std::next(_Base::_Cont->c.values.begin(),
						std::distance(_Base::_Cont->c.keys.begin(), _Base::_Ptr))) );
			}

			reference operator[](size_type pos) const {
				return reference(_Base::_Ptr[pos],
					const_cast<mapped_type&>(*std::next(_Base::_Cont->c.values.begin(),
						std::distance(_Base::_Cont->c.keys.begin(), _Base::_Ptr) + pos)) );
			}

			iterator& operator++() {
				_Base::operator++();
				return *this;
			}

			iterator operator++(int) {
				auto _Copy = *this;
				++(*this);
				return _Copy;
			}

			iterator& operator--() {
				_Base::operator--();
				return *this;
			}

			iterator operator--(int) {
				auto _Copy = *this;
				--(*this);
				return _Copy;
			}

			iterator& operator+=(difference_type diff) {
				_Base::operator+=(diff);
				return *this;
			}

			iterator& operator-=(difference_type diff) {
				_Base::operator-=(diff);
				return *this;
			}

			iterator operator+(difference_type diff) const {
				return iterator(*this) += diff;
			}

			iterator operator-(difference_type diff) const {
				return iterator(*this) -= diff;
			}

			_Base::difference_type operator-(const const_iterator& right) const {
				return _Base::operator-(right);
			}
		};
		using reverse_iterator       = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;
 
 
		// construct/copy/destroy
		flat_map() : c(), compare(key_compare()) { }
 
#if 0
		flat_map(key_container_type key_cont, mapped_container_type mapped_cont);
		template<class Allocator>
			flat_map(const key_container_type& key_cont,
							 const mapped_container_type& mapped_cont,
							 const Allocator& a);
 
		flat_map(sorted_unique_t, key_container_type key_cont,
						 mapped_container_type mapped_cont);
		template<class Allocator>
			flat_map(sorted_unique_t, const key_container_type& key_cont,
							 const mapped_container_type& mapped_cont, const Allocator& a);
 
		explicit flat_map(const key_compare& comp)
			: c(), compare(comp) { }
		template<class Allocator>
			flat_map(const key_compare& comp, const Allocator& a);
		template<class Allocator>
			explicit flat_map(const Allocator& a);
 
		template<class InputIterator>
			flat_map(InputIterator first, InputIterator last,
							 const key_compare& comp = key_compare())
				: c(), compare(comp) { insert(first, last); }
		template<class InputIterator, class Allocator>
			flat_map(InputIterator first, InputIterator last,
							 const key_compare& comp, const Allocator& a);
		template<class InputIterator, class Allocator>
			flat_map(InputIterator first, InputIterator last, const Allocator& a);
 
		template</*container-compatible-range*/<value_type> R>
			flat_map(from_range_t fr, R&& rg)
				: flat_map(fr, std::forward<R>(rg), key_compare()) { }
		template</*container-compatible-range*/<value_type> R, class Allocator>
			flat_map(from_range_t, R&& rg, const Allocator& a);
		template</*container-compatible-range*/<value_type> R>
			flat_map(from_range_t, R&& rg, const key_compare& comp)
				: flat_map(comp) { insert_range(std::forward<R>(rg)); }
		template</*container-compatible-range*/<value_type> R, class Allocator>
			flat_map(from_range_t, R&& rg, const key_compare& comp, const Allocator& a);
 
		template<class InputIterator>
			flat_map(sorted_unique_t s, InputIterator first, InputIterator last,
							 const key_compare& comp = key_compare())
				: c(), compare(comp) { insert(s, first, last); }
		template<class InputIterator, class Allocator>
			flat_map(sorted_unique_t, InputIterator first, InputIterator last,
							 const key_compare& comp, const Allocator& a);
		template<class InputIterator, class Allocator>
			flat_map(sorted_unique_t, InputIterator first, InputIterator last,
							 const Allocator& a);
 
		flat_map(initializer_list<value_type> il, const key_compare& comp = key_compare())
				: flat_map(il.begin(), il.end(), comp) { }
		template<class Allocator>
			flat_map(initializer_list<value_type> il, const key_compare& comp,
							 const Allocator& a);
		template<class Allocator>
			flat_map(initializer_list<value_type> il, const Allocator& a);
 
		flat_map(sorted_unique_t s, initializer_list<value_type> il,
						 const key_compare& comp = key_compare())
				: flat_map(s, il.begin(), il.end(), comp) { }
		template<class Allocator>
			flat_map(sorted_unique_t, initializer_list<value_type> il,
							 const key_compare& comp, const Allocator& a);
		template<class Allocator>
			flat_map(sorted_unique_t, initializer_list<value_type> il, const Allocator& a);
 
		flat_map& operator=(initializer_list<value_type> il);
#endif

		// iterators
		iterator               begin() noexcept       { return iterator{ c.keys.begin(), this}; }
		const_iterator         begin() const noexcept { return const_iterator{ c.keys.begin(), this }; }
		iterator               end() noexcept       { return iterator{ c.keys.end(), this }; }
		const_iterator         end() const noexcept { return const_iterator{ c.keys.end(), this }; }
 
		reverse_iterator       rbegin() noexcept { return reverse_iterator(end()); }
		const_reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }
		reverse_iterator       rend() noexcept { return reverse_iterator(begin()); }
		const_reverse_iterator rend() const noexcept { return reverse_iterator(begin()); }
#if 0
 
		const_iterator         cbegin() const noexcept;
		const_iterator         cend() const noexcept;
		const_reverse_iterator crbegin() const noexcept;
		const_reverse_iterator crend() const noexcept;
#endif
 
		// capacity
		[[nodiscard]] bool empty() const noexcept { return c.keys.empty(); }
		size_type size() const noexcept { return c.keys.size(); }
		size_type max_size() const noexcept { return std::min(c.keys.max_size(), c.values.max_size()); }
 
		// element access

		mapped_type& operator[](const key_type& x) {
			auto found = std::lower_bound(c.keys.begin(), c.keys.end(), x, compare);
			if (found != c.keys.end() && !compare(x, *found)) {
				return *std::next(c.values.begin(), std::distance(c.keys.begin(), found));
			} else {
				auto mapped_iter = 
					c.values.insert( std::next(c.values.begin(), std::distance(c.keys.begin(), found)), mapped_type{} );
				c.keys.insert( found, x );
				return *mapped_iter;
			}
		}

		mapped_type& operator[](key_type&& x) {
			auto found = std::lower_bound(c.keys.begin(), c.keys.end(), x, compare);
			if (found != c.keys.end() && !compare(x, *found)) {
				return *std::next(c.values.begin(), std::distance(c.keys.begin(), found));
			} else {
				auto mapped_iter = 
					c.values.insert( std::next(c.values.begin(), std::distance(c.keys.begin(), found)), mapped_type{} );
				c.keys.insert( found, std::move(x) );
				return *mapped_iter;
			}
		}

#if 0
		template<class K> mapped_type& operator[](K&& x);
		mapped_type& at(const key_type& x);
		const mapped_type& at(const key_type& x) const;
		template<class K> mapped_type& at(const K& x);
		template<class K> const mapped_type& at(const K& x) const;
#endif

		// modifiers
#if 0
		template<class... Args> pair<iterator, bool> emplace(Args&&... args);
		template<class... Args>
			iterator emplace_hint(const_iterator position, Args&&... args);
 
		pair<iterator, bool> insert(const value_type& x)
			{ return emplace(x); }
		pair<iterator, bool> insert(value_type&& x)
			{ return emplace(std::move(x)); }
		iterator insert(const_iterator position, const value_type& x)
			{ return emplace_hint(position, x); }
		iterator insert(const_iterator position, value_type&& x)
			{ return emplace_hint(position, std::move(x)); }
 
		template<class P> pair<iterator, bool> insert(P&& x);
		template<class P>
			iterator insert(const_iterator position, P&&);
		template<class InputIterator>
			void insert(InputIterator first, InputIterator last);
		template<class InputIterator>
			void insert(sorted_unique_t, InputIterator first, InputIterator last);
		template</*container-compatible-range*/<value_type> R>
			void insert_range(R&& rg);
 
		void insert(initializer_list<value_type> il)
			{ insert(il.begin(), il.end()); }
		void insert(sorted_unique_t s, initializer_list<value_type> il)
			{ insert(s, il.begin(), il.end()); }
 
		containers extract() &&;
		void replace(key_container_type&& key_cont, mapped_container_type&& mapped_cont);
#endif
 
		template<class... Args>
			pair<iterator, bool> try_emplace(const key_type& k, Args&&... args) {
			auto found = std::lower_bound(c.keys.begin(), c.keys.end(), k, compare);
			if (found != c.keys.end() && !compare(k, *found)) {
				return pair( iterator{ found, this }, false );
			} else {
				auto mapped_iter = 
					c.values.insert( std::next(c.values.begin(), std::distance(c.keys.begin(), found)), mapped_type( std::forward<Args&&>(args)... ) );
				c.keys.insert( found, k );
				return pair( mapped_iter, true );
			}
		}

		template<class... Args>
			pair<iterator, bool> try_emplace(key_type&& k, Args&&... args) {
			auto found = std::lower_bound(c.keys.begin(), c.keys.end(), k, compare);
			if (found != c.keys.end() && !compare(k, *found)) {
				return pair( iterator{ found, this }, false );
			} else {
				auto mapped_iter = 
					c.values.insert( std::next(c.values.begin(), std::distance(c.keys.begin(), found)), mapped_type( std::forward<Args&&>(args)... ) );
				c.keys.insert( found, std::move(k) );
				return pair( mapped_iter, true );
			}
		}

#if 0
		template<class K, class... Args>
			pair<iterator, bool> try_emplace(K&& k, Args&&... args);
		template<class... Args>
			iterator try_emplace(const_iterator hint, const key_type& k, Args&&... args);
		template<class... Args>
			iterator try_emplace(const_iterator hint, key_type&& k, Args&&... args);
		template<class K, class... Args>
			iterator try_emplace(const_iterator hint, K&& k, Args&&... args);
#endif

		template<class M>
			pair<iterator, bool> insert_or_assign(const key_type& k, M&& obj) {
			auto found = std::lower_bound(c.keys.begin(), c.keys.end(), k, compare);
			if (found != c.keys.end() && !compare(k, *found)) {
				*std::next(c.values.begin(), std::distance(c.keys.begin(), found)) = std::move(obj);
				return pair( iterator{ found, this }, false );
			} else {
				auto mapped_iter = 
					c.values.insert( std::next(c.values.begin(), std::distance(c.keys.begin(), found)), std::move(obj) );
				found = c.keys.insert( found, k );
				return pair( iterator{ found, this }, true );
			}
		}

		template<class M>
			pair<iterator, bool> insert_or_assign(key_type&& k, M&& obj) {
			auto found = std::lower_bound(c.keys.begin(), c.keys.end(), k, compare);
			if (found != c.keys.end() && !compare(k, *found)) {
				*std::next(c.values.begin(), std::distance(c.keys.begin(), found)) = std::move(obj);
				return pair( iterator{ found, this }, false );
			} else {
				c.values.insert( std::next(c.values.begin(), std::distance(c.keys.begin(), found)), std::move(obj) );
				found = c.keys.insert( found, std::move(k) );
				return pair( iterator{ found, this }, true );
			}
		}

#if 0
		template<class K, class M>
			pair<iterator, bool> insert_or_assign(K&& k, M&& obj);
		template<class M>
			iterator insert_or_assign(const_iterator hint, const key_type& k, M&& obj);
		template<class M>
			iterator insert_or_assign(const_iterator hint, key_type&& k, M&& obj);
		template<class K, class M>
			iterator insert_or_assign(const_iterator hint, K&& k, M&& obj);
#endif

		iterator erase(iterator position) {
			if (position != end()) {
				c.values.erase(std::next(c.values.begin(), std::distance(begin(), position)));
				position._Ptr = c.keys.erase(position._Ptr);
			}
			return position;
		}

		iterator erase(const_iterator position) {
			if (position != end()) {
				c.values.erase(std::next(c.values.begin(), std::distance(begin(), position)));
				position._Ptr = c.keys.erase(position._Ptr);
			}
			return position;
		}

		size_type erase(const key_type& x) {
			auto found = std::lower_bound(c.keys.begin(), c.keys.end(), x, compare);
			size_type index = static_cast<size_type>( std::distance(c.keys.begin(), found) );
			if (found != c.keys.end() && !compare(x, *found)) {
				c.keys.erase(found);
				c.values.erase(std::next(c.values.begin(), index));
			}
			return index;
		}

#if 0
		template<class K> size_type erase(K&& x);
		iterator erase(const_iterator first, const_iterator last);
#endif

		void swap(flat_map& y) noexcept {
			c.keys.swap(y.c.keys);
			c.values.swap(y.c.values);
		}
		
		void clear() noexcept {
			c.keys.clear();
			c.values.clear();
		}
 
		// observers
#if 0
		key_compare key_comp() const;
		value_compare value_comp() const;
#endif

		const key_container_type& keys() const noexcept      { return c.keys; }
		const mapped_container_type& values() const noexcept { return c.values; }
 
		// map operations

		iterator find(const key_type& x) {
			auto found = std::lower_bound(c.keys.begin(), c.keys.end(), x, compare);
			if (found != c.keys.end() && !compare(x, *found)) {
				return iterator{ found, this };
			} else {
				return end();
			}
		}

		const_iterator find(const key_type& x) const {
			auto found = std::lower_bound(c.keys.begin(), c.keys.end(), x, compare);
			if (found != c.keys.end() && !compare(x, *found)) {
				return iterator{ found, this };
			} else {
				return end();
			}
		}

#if 0
		template<class K> iterator find(const K& x);
		template<class K> const_iterator find(const K& x) const;

		size_type count(const key_type& x) const;
		template<class K> size_type count(const K& x) const;
#endif
 
		bool contains(const key_type& x) const {
			return std::binary_search(c.keys.begin(), c.keys.end(), x, compare);
		}

#if 0
		template<class K> bool contains(const K& x) const;
 
		iterator lower_bound(const key_type& x);
		const_iterator lower_bound(const key_type& x) const;
		template<class K> iterator lower_bound(const K& x);
		template<class K> const_iterator lower_bound(const K& x) const;
 
		iterator upper_bound(const key_type& x);
		const_iterator upper_bound(const key_type& x) const;
		template<class K> iterator upper_bound(const K& x);
		template<class K> const_iterator upper_bound(const K& x) const;
 
		pair<iterator, iterator> equal_range(const key_type& x);
		pair<const_iterator, const_iterator> equal_range(const key_type& x) const;
		template<class K> pair<iterator, iterator> equal_range(const K& x);
		template<class K> pair<const_iterator, const_iterator> equal_range(const K& x) const;
#endif
 
		friend bool operator==(const flat_map& x, const flat_map& y);
 
		//friend /*synth-three-way-result*/<value_type>
		//	operator<=>(const flat_map& x, const flat_map& y);
 
		friend void swap(flat_map& x, flat_map& y) noexcept
			{ x.swap(y); }
	};
}
#endif

namespace geometry {
	template<typename Graph>
	concept unlabeled_graph = requires(Graph g) {
		typename Graph::vertex_descriptor;
		typename Graph::vertex_property;
		typename Graph::vertex_iterator;
		typename Graph::edge_descriptor;
		typename Graph::edge_property;
		typename Graph::edge_iterator;
		typename Graph::graph_property;
		std::is_integral_v<typename Graph::vertex_descriptor>;

		g.add_vertex(typename Graph::vertex_property{});
		g.add_edge(typename Graph::vertex_descriptor{}, typename Graph::vertex_descriptor{}, typename Graph::edge_property{});
		g.add_edge(typename Graph::edge_descriptor{}, typename Graph::edge_property{});
	
		{ g.num_vertices() } -> std::convertible_to<size_t>;
		{ g.num_edges() } -> std::convertible_to<size_t>;
		{ g.vertices() } -> std::same_as< std::pair<typename Graph::vertex_iterator, typename Graph::vertex_iterator> >;
		{ g.edges() } -> std::same_as< std::pair<typename Graph::edge_iterator, typename Graph::edge_iterator> >;
	};

	template<typename Graph>
	concept labeled_graph = requires(Graph g) {
		typename Graph::vertex_descriptor;
		typename Graph::vertex_property;
		typename Graph::vertex_iterator;
		typename Graph::edge_descriptor;
		typename Graph::edge_property;
		typename Graph::edge_iterator;
		typename Graph::graph_property;

		g.add_vertex(typename Graph::vertex_descriptor{});
		g.add_vertex(typename Graph::vertex_descriptor{}, typename Graph::vertex_property{});
		g.add_edge(typename Graph::edge_descriptor{});
		g.add_edge(typename Graph::edge_descriptor{}, typename Graph::edge_property{});
		g.add_edge(typename Graph::vertex_descriptor{}, typename Graph::vertex_descriptor{});
		g.add_edge(typename Graph::vertex_descriptor{}, typename Graph::vertex_descriptor{}, typename Graph::edge_property{});

		{ g.num_vertices() } -> std::convertible_to<size_t>;
		{ g.num_edges() } -> std::convertible_to<size_t>;
		{ g.vertices() } -> std::same_as< std::pair<typename Graph::vertex_iterator, typename Graph::vertex_iterator> >;
		{ g.edges() } -> std::same_as< std::pair<typename Graph::edge_iterator, typename Graph::edge_iterator> >;
	};

	enum graph_traits {
		directed   = 0b00,
		undirected = 0b01,
		bidirected = 0b10,
		relation_mask = 0b11,

/// boost::graph use type of container (vec, list, map, set, ...). but this method may limit 
/// optimization. Example: edge_container of tree<no random_acess> is not any type of container,
/// only one vertex_descriptor or two vertex_descriptor for condition skipping traversal.
/// We use purpose of container.
		__random_acess  = 0b0001,
		__binary_search = 0b0010,
		__direct_insert = 0b0100,
/// direct_insert not implies direct_erase. Example: a list with vector container can direct_insert
/// by push_back value then set its link, but cannot direct_erase.
		__direct_erase  = 0b1000,

		__vertex_shift       = 2,
		random_access_vertex = __random_acess << __vertex_shift,
		binary_search_vertex = __binary_search << __vertex_shift,
		direct_insert_vertex = __direct_insert << __vertex_shift,
		direct_erase_vertex  = __direct_erase  << __vertex_shift,
		
		__edge_shift         = 2 + 4,
		random_access_edge   = __random_acess << __edge_shift,
		binary_search_edge   = __binary_search << __edge_shift,
		direct_insert_edge   = __direct_insert << __edge_shift,
		direct_erase_edge    = __direct_erase  << __edge_shift,

		tree_constraits = 1<<23,
		path_constraits = 1<<24
	};

	struct no_property {};

	///@container
	/// A graph is a set vertices and a set edges.
	///		
	///		std::pair<container<vertex>,container<edge>> graph;  (1)
	/// 
	/// The difference between directed_graph and undirected_graph is whether common edge.
	/// The directed_graph is simpler because it have obvious one to many relationship(one vertex corresponds to many edges).
	/// 
	///		container<pair<vertex,container<edge>>> digraph;     (2)
	/// 
	///		1. edges of 'digraph' can be ignore one side, therefore (2)structure requires less space then (1)strcture, 
	///			but the space of edges are discontinuous unless a special container allocator is used.
	///		2. (1)structure can O(1) direct access out_edges from vertex, and (1)structure cannot do that unless more space is used.
	/// 
	/// The undirected_graph also have a common edge relationship.
	/// 
	///		std::pair< container<pair<vertex,container<edge_iterator>>>, list<edge> > graph;     (3)
	/// 
	///		1. for remove edge requires directly, we need edge_iterator and requires iterator be permanently(map|set|list).
	///		2. others are same as digraph.
	/// 
	/// The (1)structure takes up more space and is more complex (for fast operation need more space and complexity),
	///		so we use (2)structure and (3)structure.
	/// The containers usually std::vector<..>, but labeled_graph may need to fast search std::map<label,..>|std::set<pair<label,..>>,
	///		and may need to fast search vertices but not edges, so there should be many combinations.
	///@note label is unique in graph, so it is 'id'.
	template<int traits, typename __vertex_property, typename __edge_property = no_property, typename __graph_property = no_property, typename __vertex_label = void>
	class adjacency_list {
	public:
		static constexpr int relation = traits&relation_mask;
		static constexpr bool labeled = std::is_same_v<__vertex_label, void>;
		///@note
		///                    binary_search    no_binary_search
		/// random_access      std::vector<..>  std::vector<...>
		/// no_random_access   std::map<..,..>  std::list<...>
		static constexpr bool random_access_vertices = (traits&random_access_vertex) != 0;
		static constexpr bool binary_search_vertices = (traits&binary_search_vertex) != 0;
		static constexpr bool random_access_out_edges = (traits&random_access_edge) != 0;
		static constexpr bool binary_search_out_edges = true;
		///@note 
		/// value_type& pointer::operator*(),              pointer may not permanent.
		/// value_type& container::operator[](descriptor), descriptor must be permanent, but not readable.
		/// value_type& container::find(id),               id must be permanent, but cannot get value_type in O(1).
		using vertex_id       = __vertex_label;
		using vertex_property = __vertex_property;
		using edge_property   = __edge_property;
		using graph_property  = __graph_property;
		

		template<int _Which_graph_type>
		struct _Which_container {//default is 'directed'_graph.
			/// 1. random access vertices, we can use index to do everything normally.
			///		but can't random access vertices, we must pay more for doing something with the index.
			/// 2. can't random access vertices, we can get permanent address, and use it to do everything normally.
			///		but random access vertices, address is not permanent, do something will error.
			/// so we use index for random_access_vertices, use address for no_random_access_vertices.
			///@note we not use iterator for address, because cannot known 'vertex_container' before known 'out_edge_container'.
			using vertex_descriptor = std::conditional_t<random_access_vertices, size_t, void*>;
			struct edge_descriptor { 
				vertex_descriptor source; std::conditional_t<random_access_out_edges, size_t, void*> 
					impl/* = decltype(source.out_edges)::iterator{ target, edge_container::iterator }*/; };
			
			using edge_container = void;

			using out_edge_container =
				std::conditional_t<random_access_out_edges, std::flat_map<vertex_descriptor, edge_property>,
					std::map<vertex_descriptor, edge_property>  >;
			using in_edge_container = void;

			struct vertex { vertex_property prop; out_edge_container out_edges; };
			using vertex_container =
				std::conditional_t<labeled, 
					std::conditional_t<random_access_vertices, std::flat_map<vertex_id, vertex>,
						std::map<vertex_id, vertex> >,
					std::conditional_t<random_access_vertices, std::vector<vertex>,
						std::list<vertex> > >;

			vertex_container vertices;
		};

		template<>
		struct _Which_container<undirected> {
			using vertex_descriptor = std::conditional_t<random_access_vertices, size_t, void*>;
			struct edge_descriptor { 
				vertex_descriptor source; std::conditional_t<random_access_out_edges, size_t, void*>
					impl/* = decltype(source.out_edges)::iterator{ target, edge_container::iterator }*/,
					impl2/* = decltype(target.out_edges)::iterator{ source, edge_container::iterator }*/; };
			
			using edge_container = std::list<edge_property>;

			using out_edge_container =
				std::conditional_t<random_access_out_edges, std::flat_map<vertex_descriptor, typename edge_container::iterator>,
					std::map<vertex_descriptor, typename edge_container::iterator>  >;
			using in_edge_container = void;

			struct vertex { vertex_property prop; out_edge_container out_edges; };
			using vertex_container =
				std::conditional_t<labeled, 
					std::conditional_t<random_access_vertices, std::flat_map<vertex_id, vertex>,
						std::map<vertex_id, vertex> >,
					std::conditional_t<random_access_vertices, std::vector<vertex>,
						std::list<vertex> > >;

			vertex_container vertices;
			edge_container edges;
		};

		template<>
		struct _Which_container<bidirected> {
			using vertex_descriptor = std::conditional_t<random_access_vertices, size_t, void*>;
			struct edge_descriptor { 
				vertex_descriptor source; std::conditional_t<random_access_out_edges, size_t, void*>
					impl/* = decltype(source.out_edges)::iterator{ target, edge_container::iterator }*/,
					impl2/* = decltype(target.in_edges)::iterator{ source, edge_container::iterator } */; };
			
			using edge_container = std::list<edge_property>;

			using out_edge_container =
				std::conditional_t<random_access_out_edges, std::flat_map<vertex_descriptor, typename edge_container::iterator>,
					std::map<vertex_descriptor, typename edge_container::iterator>  >;
			using in_edge_container = out_edge_container;

			struct vertex { vertex_property prop; out_edge_container out_edges, in_edges; };
			using vertex_container =
				std::conditional_t<labeled, 
					std::conditional_t<random_access_vertices, std::flat_map<vertex_id, vertex>,
						std::map<vertex_id, vertex> >,
					std::conditional_t<random_access_vertices, std::vector<vertex>,
						std::list<vertex> > >;

			vertex_container vertices;
			edge_container edges;
		};

		using container = _Which_container<relation>;
		using vertex_descriptor   = typename container::vertex_descriptor;
		using edge_descriptor     = typename container::edge_descriptor;
		using edge_container      = typename container::edge_container;
		using out_edge_container  = typename container::out_edge_container;
		using in_edge_container   = typename container::in_edge_container;
		using vertex              = typename container::vertex;
		using vertex_container    = typename container::vertex_container;

		using vertex_reference       = vertex &;
		using vertex_const_reference = const vertex &;
		struct edge_reference       { const vertex &source, &target; edge_property &prop; };
		struct edge_const_reference { const vertex &source, &target; const edge_property &prop; };

		container c;

		template<typename _Container>
		static auto _Get_descriptor(_Container& cont, typename _Container::iterator iter) {
			if constexpr (std::is_same_v<typename std::iterator_traits<typename _Container::iterator>::iterator_category, std::random_access_iterator_tag>) {
				return static_cast<size_t>(std::distance(cont.begin(), iter));
			} else {
				return static_cast<void*>(iter._Ptr);
			}
		}

		template<typename _Container, typename _Desc>
		static auto _Get_iter(_Container& cont, _Desc descriptor) {
			if constexpr (std::is_same_v<typename std::iterator_traits<typename _Container::iterator>::iterator_category, std::random_access_iterator_tag>) {
				static_assert( std::is_integral_v<_Desc> );
				return std::next(cont.begin(), descriptor);
			} else {
				static_assert( std::is_pointer_v<_Desc> );
				auto iter = cont.begin();
				iter._Ptr = reinterpret_cast<decltype(iter._Ptr)>(descriptor);
				return iter;
			}
		}


		static vertex_descriptor null_vertex() {
			if constexpr (random_access_vertices) {
				return static_cast<vertex_descriptor>(-1);
			} else {
				return static_cast<vertex_descriptor>(nullptr);
			}
		}

		template<typename... vertex_property_any>
		vertex_descriptor _Add_vertex(vertex_property_any&&... value) {
			if constexpr (!labeled) {
				c.vertices.push_back({ std::forward<vertex_property_any&&>(value)... });
				if constexpr (random_access_vertices)
					return static_cast<vertex_descriptor>(c.vertices.size() - 1);
				return _Get_descriptor(c.vertices, std::prev(c.vertices.end()));
			} else {
				abort();
			}
		}

		template<typename... vertex_property_any>
		std::pair<vertex_descriptor, bool> 
			_Insert_or_assign_vertex(const vertex_id& key, vertex_property_any&&... value) {
			if constexpr (!labeled) {
				abort();
			} else {
				auto result = c.vertices.insert_or_assign(key, vertex{ std::forward<vertex_property_any&&>(value)... });
				if constexpr (random_access_vertices) {
					size_t pos = _Get_descriptor(c.vertices, result.first);
					for (auto vertexX : c.vertices) {
						for (auto out_edgeX : vertexX.second.out_edges) 
							if (out_edgeX.first >= pos) 
								++const_cast<size_t&>(out_edgeX.first);
						if constexpr (bidirected) {
							for (auto in_edgeX : vertexX.second.in_edges)
								if (in_edgeX.first >= pos)
									++const_cast<size_t&>(in_edgeX.first);
						}
					}
				}
				return std::pair(_Get_descriptor(c.vertices, result.first), result.second);
			}
		}

		vertex_descriptor add_vertex(const vertex_property& prop) {
			return _Add_vertex(prop);
		}
		
		vertex_descriptor add_vertex(vertex_property&& prop) {
			return _Add_vertex(std::move(prop));
		}
		
		vertex_descriptor add_vertex() {
			return _Add_vertex();
		}

		vertex_descriptor add_vertex(const vertex_id& key, const vertex_property& value) {
			return _Insert_or_assign_vertex(key, value).first;
		}

		vertex_descriptor add_vertex(const vertex_id& key, vertex_property&& value) {
			return _Insert_or_assign_vertex(key, std::move(value)).first;
		}

		vertex_descriptor find(const vertex_id& id) {
			if constexpr (!labeled) {
				if constexpr (random_access_vertices)
					return static_cast<size_t>(id) < c.vertices.size() ? static_cast<vertex_descriptor>(id) : null_vertex();
				return static_cast<size_t>(id) < c.vertices.size() ? _Get_descriptor(c.vertices, std::next(c.vertices.begin(), id)) : null_vertex();
			} else {
				auto found_id = c.vertices.find(id);
				return (found_id != c.vertices.end()) ? _Get_descriptor(c.vertices, found_id) : null_vertex();
			}
		}

		bool contains(const vertex_id& id) const {
			if constexpr (!labeled) {
				return static_cast<size_t>(id) < c.vertices.size();
			} else {
				return c.vertices.contains(id);
			}
		}
		
		vertex_const_reference operator[](vertex_descriptor v) const {
			assert( v != null_vertex() );
			if constexpr (!labeled) {
				return *_Get_iter(c.vertices, v);
			} else {
				return ( *_Get_iter(c.vertices, v) ).second;
			}
		}

		vertex_reference operator[](vertex_descriptor v) {
			assert( v != null_vertex() );
			if constexpr (!labeled) {
				return *_Get_iter(c.vertices, v);
			} else {
				return ( *_Get_iter(c.vertices, v) ).second;
			}
		}
		
		std::pair<typename vertex_container::iterator,typename vertex_container::iterator> vertices() {
			return std::pair(c.vertices.begin(), c.vertices.end());
		}

		std::pair<typename vertex_container::const_iterator,typename vertex_container::const_iterator> vertices() const {
			return std::pair(c.vertices.begin(), c.vertices.end());
		}


		template<typename... edge_property_any>
		std::pair<edge_descriptor, bool> _Insert_or_assign_edge(vertex_descriptor source, vertex_descriptor target, edge_property_any&&... prop) {
			assert( source != null_vertex() );
			assert( target != null_vertex() );
			if constexpr (relation == directed) {
				auto result = (*this)[source].out_edges.insert_or_assign(target, edge_property{ std::forward<edge_property_any&&>(prop)... });
				return std::pair(edge_descriptor{ source, _Get_descriptor((*this)[source].out_edges, result.first) }, result.second);
			} else {
				auto result = (*this)[source].out_edges.find(target);
				if (result == (*this)[source].out_edges.end()) {
					c.edges.push_back(edge_property{ std::forward<edge_property_any&&>(prop)... });
					result = (*this)[source].out_edges.insert_or_assign(target, std::prev(c.edges.end())).first;
					if constexpr (relation == undirected) {
						auto result2 = (*this)[target].out_edges.insert_or_assign(source, std::prev(c.edges.end())).first;
						return std::pair(edge_descriptor{ source, _Get_descriptor((*this)[source].out_edges, result), _Get_descriptor((*this)[target].out_edges, result2) }, true);
					} else /* type == geometry::bidirected */ {
						auto result2 = (*this)[target].in_edges.insert_or_assign(source, std::prev(c.edges.end())).first;
						return std::pair(edge_descriptor{ source, _Get_descriptor((*this)[source].out_edges, result), _Get_descriptor((*this)[target].in_edges, result2) }, true);
					}
				} else {
					(* (*result).second) = edge_property{ std::forward<edge_property_any&&>(prop)... };
					if constexpr (relation == undirected) {
						return std::pair(edge_descriptor{ source, _Get_descriptor((*this)[source].out_edges, result), _Get_descriptor((*this)[target].out_edges, (*this)[target].out_edges.find(source)) }, false);
					} else {
						return std::pair(edge_descriptor{ source, _Get_descriptor((*this)[source].out_edges, result), _Get_descriptor((*this)[target].in_edges, (*this)[target].in_edges.find(source)) }, false);
					}
				}
			}
		}

		edge_descriptor add_edge(vertex_descriptor source, vertex_descriptor target, const edge_property& prop) {
			return _Insert_or_assign_edge(source, target, prop).first;
		}

		edge_descriptor add_edge(vertex_descriptor source, vertex_descriptor target, edge_property&& prop) {
			return _Insert_or_assign_edge(source, target, std::move(prop)).first;
		}

		edge_descriptor add_edge(vertex_descriptor source, vertex_descriptor target) {
			return _Insert_or_assign_edge(source, target).first;
		}

		edge_descriptor find(vertex_descriptor source, vertex_descriptor target) {
			assert( source != null_vertex() );
			assert( target != null_vertex() );
			if constexpr (relation == directed) {
				return edge_descriptor{ source, _Get_descriptor((*this)[source].out_edges,  (*this)[source].out_edges.find(target)) };
			}	else {
				if constexpr (relation == undirected) {
					return edge_descriptor{ source, _Get_descriptor((*this)[source].out_edges,  (*this)[source].out_edges.find(target)), 
						_Get_descriptor((*this)[target].out_edges,  (*this)[target].out_edges.find(source)) };
				} else /* relation == bidirected */ {
					return edge_descriptor{ source, _Get_descriptor((*this)[source].out_edges,  (*this)[source].out_edges.find(target)),
						_Get_descriptor((*this)[target].in_edges,  (*this)[target].in_edges.find(source)) };
				}
			}
		}

		edge_descriptor find(const vertex_id& s_id, const vertex_id& t_id) {
			vertex_descriptor source = find(s_id);
			if (source == null_vertex()) {
				abort();
			}

			vertex_descriptor target = find(t_id);
			if (target == null_vertex()) {
				return edge_descriptor{ source, _Get_descriptor((*this)[source].out_edges,  (*this)[source].out_edges.end()) };
			}

			return find(source, target);
		}

		edge_reference operator[](edge_descriptor e) {
			if constexpr (directed) {
				return edge_reference{ (*this)[e.source], (*this)[(*_Get_iter((*this)[e.source].out_edges, e.impl)).first], (*_Get_iter((*this)[e.source].out_edges, e.impl)).second };
			} else {
				return edge_reference{ (*this)[e.source], (*this)[(*_Get_iter((*this)[e.source].out_edges, e.impl)).first], *(*_Get_iter((*this)[e.source].out_edges, e.impl)).second};
			}
		}

		/*edge_reference operator[](std::pair<vertex&, typename out_edge_container::iterator> epair) {
			return edge_reference{ epair.first, *_Get_iter(vertices, epair.second->first), epair.second->second };
		}*/

		/*edge_reference operator[](std::pair<vertex&, out_edge&> epair) {
			return edge_reference{ epair.first, *_Get_iter(vertices, epair.second.first), epair.second.second };
		}*/

		void remove_edge(vertex_descriptor source, vertex_descriptor target) {// O(log2(n))
			assert( source != null_vertex() );
			assert( target != null_vertex() );
			if constexpr (relation == directed) {
				(*this)[source].out_edges.erase(target);
			} else {
				auto the_out_edge = (*this)[source].out_edges.find(target);
				if (the_out_edge != (*this)[source].out_edges.end()) {
					auto the_edge = (*the_out_edge).second;
					(*this)[source].out_edges.erase(the_out_edge);
					if constexpr (relation == undirected) {
						(*this)[target].out_edges.erase(source);
					} else /* relation == bidirected */ {
						(*this)[target].in_edges.erase(source);
					}
					c.edges.erase(the_edge);
				}
			}
		}

		void remove_edge(edge_descriptor e) {// O(1)
			assert( e.source != null_vertex() );
			if constexpr (relation == directed) {
				(*this)[e.source].out_edges.erase( _Get_iter((*this)[e.source].out_edges,e.impl) );
			} else {
				auto the_out_edge = _Get_iter((*this)[e.source].out_edges, e.impl);
				if (the_out_edge != (*this)[e.source].out_edges.end()) {
					auto target = (*_Get_iter((*this)[e.source].out_edges, e.impl)).first;
					auto the_edge = (*_Get_iter((*this)[e.source].out_edges, e.impl)).second;
					(*this)[e.source].out_edges.erase( _Get_iter((*this)[e.source].out_edges, e.impl) );
					if constexpr (relation == undirected) {
						(*this)[target].out_edges.erase( _Get_iter((*this)[target].out_edges, e.impl2) );
					} else /* relation == bidirected */ {
						(*this)[target].in_edges.erase( _Get_iter((*this)[target].out_edges, e.impl2) );
					}
					c.edges.erase(the_edge);
				}
			}
		}

		void remove_vertex(vertex_descriptor v) {
			for (auto vX = c.vertices.begin(); vX != c.vertices.end(); ++vX) {
				remove_edge(_Get_descriptor(c.vertices, vX), v);
				if constexpr (relation == bidirected)
					remove_edge(v, _Get_descriptor(c.vertices, vX));
			}
			c.vertices.erase(_Get_iter(c.vertices, v));

		/*	if constexpr (random_access_vertices) {
				for (auto& vertexX : c.vertices) {
					for (auto& out_edgeX : vertexX.out_edges)
						if (out_edgeX.first > v)
							--const_cast<vertex_descriptor&>(out_edgeX.first);
					if constexpr (type == bidirected) 
						for (auto& in_edgeX : vertexX.in_edges)
							if (in_edgeX.first > v)
								--const_cast<vertex_descriptor&>(in_edgeX.first);
				}
			}*/
		}

		void clear() noexcept {
			c.vertices.clear();
			if constexpr (relation == undirected || relation == bidirected) {
				c.edges.clear();
			}
		}

#if 0
		void remove_vertex(vertex_descriptor v) {
			for (auto vX = c.vertices.begin(); vX != c.vertices.end(); ++vX)
				remove_edge(_Get_descriptor(vertices, vX), v);
			vertices.erase( _Get_iter(vertices, v) );
			if constexpr (random_access_vertices) {
				for (auto& vertexX : vertices)
					for (auto& out_edgeX : vertexX.out_edges)
						if (out_edgeX.first > v)
							if constexpr (!binary_search_out_edges) {
								--out_edgeX.first;
							} else {
								--const_cast<vertex_descriptor&>(out_edgeX.first);
							}
			}
		}
#endif

#if 0
		out_edge_descriptor add_edge(vertex_descriptor source, vertex_descriptor target, const edge_property& eproperty = {}) {
			auto& out_edges = (*this)[source].out_edges;
			if constexpr (!binary_search_out_edges) {
				out_edges.push_back({ target, eproperty });
				if constexpr (random_access_out_edges) {
					return edge_descriptor{ source, target, out_edges.size() - 1 };
				} else {
					return edge_descriptor{ source, target, _Get_descriptor(out_edges, std::prev(out_edges.end())) };
				}
			} else if constexpr (binary_search_out_edges && random_access_out_edges) {
				auto lower_bound = std::lower_bound(out_edges.begin(), out_edges.end(), target, [](out_edge& e, vertex_descriptor target){
					return e.first < target; }
				);
				auto eitr = out_edges.insert(lower_bound, { target, eproperty });
				return edge_descriptor{ source, target, _Get_descriptor(out_edges, eitr) };
			} else /*if constexpr (binary_search_out_edges && !random_access_out_edges)*/ {
				auto eitr = out_edges.insert_or_assign(target, eproperty).first;
				return edge_descriptor{ source, target, _Get_descriptor(out_edges, eitr) };
			}
		}

		out_edge_descriptor get(vertex_descriptor source, vertex_descriptor target) {
			auto& out_edges = (*this)[source].out_edges;
			if constexpr (!binary_search_out_edges) {
				auto found = std::find_if(out_edges.begin(), out_edges.end(), [target](out_edge& e){ 
					return e.first == target; }
				);
				return edge_descriptor{ source, target, _Get_descriptor(out_edges, found) };
			} else if constexpr (binary_search_out_edges && random_access_out_edges) {
				auto lower_bound = std::lower_bound(out_edges.begin(), out_edges.end(), target, [](out_edge& e, vertex_descriptor target){ 
					return e.first < target; }
				);
				return edge_descriptor{ source, target, _Get_descriptor(out_edges, lower_bound == out_edges.end() || lower_bound->first != target ? out_edges.end() : lower_bound) };
			} else /*if constexpr (binary_search_out_edges && !random_access_out_edges)*/ {
				return edge_descriptor{ source, target, _Get_descriptor(out_edges, out_edges.find(target)) };
			}
		}
		
		out_edge_descriptor find(vertex_id s_id, vertex_id t_id) {
			vertex_descriptor source = find(s_id);
			if (_Get_iter(vertices, source) == vertices.end()) {
				abort();
			}

			vertex_descriptor target = find(t_id);
			if (_Get_iter(vertices, target) == vertices.end()) {
				auto& out_edges = (*this)[source].out_edges;
				return edge_descriptor{ source, target, _Get_descriptor(out_edges, out_edges.end()) };
			}

			return get(source, target);
		}

		edge_reference operator[](edge_descriptor e) {
			auto& source = *_Get_iter(vertices, e.source);
			auto& target = *_Get_iter(vertices, e.target);
			return edge_reference{ source, target, _Get_iter(source.out_edges, e.impl)->second };
		}

		edge_reference operator[](std::pair<vertex&, typename out_edge_container::iterator> epair) {
			return edge_reference{ epair.first, *_Get_iter(vertices, epair.second->first), epair.second->second };
		}

		edge_reference operator[](std::pair<vertex&, out_edge&> epair) {
			return edge_reference{ epair.first, *_Get_iter(vertices, epair.second.first), epair.second.second };
		}

		bool contains(vertex_id s_id, vertex_id t_id) const {
			if constexpr (!labeled) {
				if (static_cast<size_t>(s_id) >= vertices.size()
					|| static_cast<size_t>(t_id) >= vertices.size()) {
					return false;
				}
			}
			edge_descriptor found = find(s_id, t_id);
			auto& out_edges = (*this)[found.source].out_edges;
			return (out_edges.end() != _Get_iter(out_edges, found.impl));
		}

		void remove_edge(vertex_descriptor source, vertex_descriptor target) {
			edge_descriptor found = get(source, target);
			auto& out_edges = (*this)[source].out_edges;
			if (out_edges.end() != _Get_iter(out_edges, found.impl)) {
				out_edges.erase(_Get_iter(out_edges, found.impl));
			}
		}

		void remove_vertex(vertex_descriptor v) {
			for (auto vX = vertices.begin(); vX != vertices.end(); ++vX)
				remove_edge(_Get_descriptor(vertices, vX), v);
			vertices.erase( _Get_iter(vertices, v) );
			if constexpr (random_access_vertices) {
				for (auto& vertexX : vertices)
					for (auto& out_edgeX : vertexX.out_edges)
						if (out_edgeX.first > v)
							if constexpr (!binary_search_out_edges) {
								--out_edgeX.first;
							} else {
								--const_cast<vertex_descriptor&>(out_edgeX.first);
							}
			}
		}
		
		/*bool valid(vertex_descriptor source, vertex_descriptor target) const {
			
		}*/

		void clear() noexcept {
			vertices.clear();
		}

		/// do vertices[i] = std::move(other.vertices[i]).
		void merge(adjacency_list& other) {
			if constexpr (random_access_vertices) {
				const size_t offset = vertices.size();
				
				vertices.reserve( vertices.size() + other.vertices.size() );
				for (size_t i = 0; i != other.vertices.size(); ++i) {
					vertices.push_back( std::move( other.vertices[i] ) );
				}
				other.clear();

				for (size_t i = offset; i != vertices.size(); ++i) {
					for (auto& out_edgeX : vertices[i].out_edges) {
						if constexpr (!binary_search_out_edges) {
							out_edgeX.first += offset;
						} else {
							const_cast<vertex_descriptor&>(out_edgeX.first) += offset;
						}
					}
				}
			} else {
				// unlabeled graph, vertices container must be std::list<>.
				vertices.merge(other.vertices, [](vertex&, vertex&){ return false; });
			}
		}
#endif
	};

	///@note when labeled && random_access_vertices, we cannot use result of g.add_vertex(...), should be real-time find as g.find(...).
	template<int traits, typename vertex_label, typename vertex_property, typename edge_property = no_property, typename graph_property = no_property>
	using labeled_adjacency_list = adjacency_list<traits, vertex_property, edge_property, graph_property, vertex_label>;
		

	template<typename graph, typename insert_iterator>
	size_t find_roots(graph& g, insert_iterator roots) {
		/*std::vector<bool> visited(g.vertices().size(), false);
		for (auto& vertexX : g.vertices()) {
			for (auto& out_edgeX : g.out_edges(vertexX)) {
				visited[ std::distance(g.vertices().begin(), g._Get_iter(g.vertices(), out_edgeX.target)) ] = true;
			}
		}

		size_t i = 0, ct = 0;
		for (auto viter = g.vertices.begin(); viter != g.vertices.end(); ++viter, ++i) {
			if (!visited[i]) {
				(*roots)++ = g._Get_descriptor(g.vertices, viter);
				++ct;
			}
		}
		return ct;*/
		return 0;
	}

#if 0
	/**
		* @param s "https://ericrowland.github.io/investigations/polyhedra.html"
		* @param a "https://www.inchcalculator.com/isosceles-triangle-calculator/"
		*   = pi/2 - baseangle
		*   (s/2 / r) = sin(ag/2)
		*   asin(s/2 / r)*2 = ag
		*   baseangle = (pi - ag) / 2 = (pi - asin(s/2 / r)*2) / 2 = pi/2 - asin(s/2 / r)
		*   = pi/2 - pi/2 + asin(s/2 / r)
		*   = asin(s/2 / r)
		* @cube
		*   double r = 1.0;
		*   double s = r * 2/sqrt(3.0);
		*   auto g = calculation::polygraph<Graph>(4, 3, s, asin(s/2 / r), rotate);
		* @tetrahedron
		*   double h = 1.0;
		*   double s = h / sqrt(2.0 / 3);
		*   auto g = calculation::polygraph<Graph>(3, 3, s, asin(h / s), rotate);
		* @octahedron
		*   double h = 1.0;
		*   double s = h * sqrt(2.0);
		*   auto g = calculation::polygraph<Graph>(3, 4, s, asin(h / s), rotate);
		* @icosahedron
		*   double a = 1.0;
		*   double s = a / ((3 + sqrt(5.0))/(4*sqrt(3.0)));
		*   double h = sqrt(3.0) * s / 2;
		*   double r = sqrt(a*a + (2*h/3)*(2*h/3));
		*   auto g = calculation::polygraph<Graph>(3, 5, s, asin(s/2 / r), rotate);
	*/
	template<typename Graph, typename Real, typename Func>
	Graph polygraph(int p, int q, Real s, Real a, Func rotate, const Real epsilon = 1e-12) {
		const Real pi = 3.1415926535897932384626433832795;
		const Real phi = pi*2 / q;

		typedef typename Graph::vertex_iterator VertexIterator;
		typedef typename Graph::vertex_descriptor VertexDescriptor;
		typedef typename Graph::vertex_property_type VertexProperty;
		typedef decltype(VertexProperty::normal) Vector; 
		Graph g;

		/** modified from the breadth first search
			* @idea is all the vertices processed are in the 'graph'
		*/
		std::queue<VertexDescriptor> Q;
		Q.push(g.add_vertex(VertexProperty{Vector{0,0,0}, Vector{1,0,0}, Vector{0,1,0}}).id());
		while (!Q.empty()) {
			VertexDescriptor isour = Q.front();
			VertexIterator psour = g[isour];
			Q.pop();
			for (size_t i = 0; i != (size_t)(q); ++i) {
				VertexProperty targ;
				targ.normal = psour->normal;
				targ.tangent = rotate(psour->tangent, psour->normal,phi*i);
				targ.position = psour->position;
		
				Vector targ_bitangent = cross(targ.tangent, targ.normal);
				targ.position += rotate(targ_bitangent, targ.tangent, a) * s;
				targ.normal = rotate(targ.normal, targ.tangent, a*2);
				targ.tangent = rotate(targ.tangent, targ.normal, pi-phi);

				VertexDescriptor itarg = -1;
				for (auto [vi,viend] = g.vertices(); vi != viend; ++vi) {
					if (abs(vi->position[0] - targ.position[0]) < epsilon &&
							abs(vi->position[1] - targ.position[1]) < epsilon &&
							abs(vi->position[2] - targ.position[2]) < epsilon)
					{
						itarg = vi.id();
						break;
					}
				}

				if (itarg == -1) {
					itarg = g.add_vertex(targ).id();
					Q.push(itarg);
				}

				// record the path
				if constexpr (Graph::undirected) {
					g.add_edge(isour, itarg);
				} else {
					g.add_edge(isour, itarg);
					g.add_edge(itarg, isour);
				}
			}
		}

		return g;
	}

	template<typename Graph>
	auto polygraph_circles(int p, int q, Graph& g) {
		typedef std::vector<typename Graph::vertex_descriptor> CircleKey;
		typedef std::vector<typename Graph::vertex_iterator> Circle;
		const auto less = [](const CircleKey& a, const CircleKey& b) {
			if (a.size() == b.size()) {
				for (size_t i = 0; i != a.size(); ++i)
					if (a[i] != b[i])
						return a[i] < b[i];
				return false;
			} else {
				return a.size() < b.size();
			}
		};
		std::map<CircleKey, Circle, decltype(less)> circles = 
			std::map<CircleKey, Circle,decltype(less)>(less);

		/** modified from the breadth first search
			* @idea is each vertex is not enqueued at most once, limit distance is 'p'
		*/
		std::queue<Circle> Q;
		for (auto [vi, viend] = g.vertices(); vi != viend; ++vi)
			Q.push(Circle{vi});
		while (!Q.empty()) {
			Circle path = std::move(Q.front()); 
			Q.pop();
			if (path.size() != p) {
				for (auto [ei, eiend] = path.back().out_edges(); ei != eiend; ++ei) {
					/* if (ei.target != path.parent) */
					if (path.size() == 1 || ei.target() != path[path.size() - 2]) {
						Circle path2 = path;
						path2.push_back(ei.target());
						Q.push(std::move(path2));
					}
				}
			}
			else {// process p-polygon
				for (auto [ei, eiend] = path.back().out_edges(); ei != eiend; ++ei){
					if (ei.target() == path.front()) {
						CircleKey k = CircleKey(path.size());
						for (size_t i = 0; i != path.size(); ++i)
							k[i] = path[i].id();
						std::sort(k.begin(), k.end());
						circles.insert_or_assign(k, path);
						break;
					}
				}
			}
		}

		return circles;
	}


	/**
		* @brief TriangleMesh + Graph
		*   1. Indirect(not direct, for cache speed) input to TriangleMesh
		*   2. Graph traits
	*/
	/*using VertexProperty = int;
	using VertexDescriptor = size_t;
	using FaceProperty = int;*/
	template<typename VertexProperty, typename VertexDescriptor = size_t, typename FaceProperty = no_property>
	class AdjacencyTriangleMesh {
		static_assert(std::is_integral_v<VertexDescriptor>);
	public:
		struct FaceS;

		struct VertS {
			VertexProperty property;
			FaceS* adjface;
			bool regular;
			bool boundary;
			size_t valence() {
				FaceS* f = this->adjface;
				if (!this->boundary) {
					// Compute valence of interior vertex
					size_t nf = 1;
					while ((f = f->nextface(this)) != this->adjface) ++nf;
					return nf;
				} else {
					// Compute valence of boundary vertex
					size_t nf = 1;
					while ((f = f->nextface(this)) != nullptr) ++nf;
					f = this->adjface;
					while ((f = f->prevface(this)) != nullptr) ++nf;
					return nf + 1;
				}
			}

			VertS() : property(VertexProperty()), adjface(nullptr), regular(false), boundary(false) {}
			VertS(const VertexProperty& p) : property(p), adjface(nullptr), regular(false), boundary(false) {}
		};
		
		struct FaceS {
			FaceProperty property;
			VertS* vertices[3];
			FaceS* adjfaces[3];
			size_t vnum(VertS* vert) const {
				for (int i = 0; i < 3; ++i)
					if (vertices[i] == vert)
						return i;
				return -1;
			}
			FaceS* nextface(VertS* vert) { return adjfaces[vnum(vert)]; }
			FaceS* prevface(VertS* vert) { return adjfaces[(vnum(vert)+2)%3]; }
			VertS* nextvert(VertS* vert) { return vertices[(vnum(vert)+1)%3]; }
			VertS* prevvert(VertS* vert) { return vertices[(vnum(vert)+2)%3]; }
			VertS* othervert(VertS* v0, VertS* v1) {
				for (size_t i = 0; i != 3; ++i)
					if (vertices[i] != v0 && vertices[i] != v1)
						return vertices[i];
				return nullptr;
			}

			FaceS() : property(FaceProperty{}), vertices{ nullptr,nullptr,nullptr }, adjfaces{ nullptr,nullptr,nullptr } {}
			FaceS(VertS* v0, VertS* v1, VertS* v2) : property(FaceProperty()), vertices{ v0,v1,v2 }, adjfaces{ nullptr,nullptr,nullptr } {}
		};

		struct EdgeS {
			VertS* vertices[2];
			FaceS* faces[2];
			size_t face0_i;
			bool operator<(const EdgeS& e2) const {
				return (vertices[0] == e2.vertices[0]) ? vertices[1] < e2.vertices[1]
					: vertices[0] < e2.vertices[0];
			}

			EdgeS() : vertices{ nullptr,nullptr }, faces{ nullptr,nullptr }, face0_i(-1) {}
			EdgeS(VertS* v0, VertS* v1) : vertices{ v0, v1 }, faces{ nullptr,nullptr }, face0_i(-1) {}
		};

		std::vector<VertS> vertex_array;
		std::vector<FaceS> face_array;

		template<typename VertexPropertyIterator>
		void add_vertices(VertexPropertyIterator first, VertexPropertyIterator last) {
			if (first != last) {
				const size_t old_size = vertex_array.size();
				const VertS* old_data = vertex_array.data();
				for ( ; first != last; ++first) {
					vertex_array.push_back( VertS(*first) );
				}

				VertS* new_data = vertex_array.data();
				if (new_data != old_data && old_size != 0) {
					for (FaceS& f : face_array) {
						f.vertices[0] = new_data + (f.vertices[0] - old_data);
						f.vertices[1] = new_data + (f.vertices[1] - old_data);
						f.vertices[2] = new_data + (f.vertices[2] - old_data);
					}
				}
			}
		}

		template<typename VertexDescriptorIterator>
		void add_triangles(VertexDescriptorIterator ifirst, VertexDescriptorIterator ilast) {
			if (ifirst != ilast) {
				if (std::distance(ifirst, ilast) % 3 != 0) {
					throw std::exception("logic error");
				}
				size_t face_count = std::distance(ifirst, ilast) / 3;

				// Reallocate memory if new_size > capacity
				const size_t old_size = face_array.size();
				const FaceS* old_data = face_array.data();
				face_array.resize(old_size + face_count, FaceS());

				// Refresh old adjacency faces if memory reallocated
				FaceS* new_data = face_array.data();
				if (new_data != old_data && old_size != 0) {
					for (VertS& vertex : vertex_array) {
						if (vertex.adjface != nullptr) {
							vertex.adjface = new_data + (vertex.adjface - old_data);
						}
					}
					for (auto pface = face_array.begin(), face_last = std::next(pface, old_size); pface != face_last; ++pface) {
						for (size_t i = 0; i != 3; ++i) {
							if (pface->adjfaces[i] != nullptr) {
								pface->adjfaces[i] = new_data + (pface->adjfaces[i] - old_data);
							}
						}
					}
				}

				// Add new faces, Update vertex.adjacency_property
				for (auto pface = std::next(face_array.begin(), old_size), face_last = face_array.end(); pface != face_last; ++pface) {
					for (size_t i = 0; i != 3; ++i) {
						assert( *ifirst < vertex_array.size() );
						VertS* pvertex_i = &vertex_array[*ifirst++];
						pface->vertices[i] = pvertex_i;
						pvertex_i->adjface = pface._Ptr;
					}
				}

				// Update face.adjacency_faces, O(log2(E/2)*F)
				std::set<EdgeS> edgeset;
				for (FaceS& face : face_array) {
					for (size_t i = 0; i != 3; ++i) {
						VertS* sour = face.vertices[i];
						VertS* targ = face.vertices[(i+1)%3];
						EdgeS  edge = EdgeS(std::min(sour,targ), std::max(sour,targ));
						if (edgeset.find(edge) == edgeset.end()) {
							// Record one side face
							edge.faces[0] = &face;
							edge.face0_i  = i;
							edgeset.insert(edge);
						} else {
							edge = *edgeset.find(edge);
							// Link the face to another
							FaceS& face0 = *edge.faces[0];
							size_t i0    = edge.face0_i;
							face0.adjfaces[i0] = &face;
							face.adjfaces[i] = &face0;
							edgeset.erase(edge);
						}
					}
				}

				// Update vertex.adjacency_state
				for (VertS& vertex : vertex_array) {
					size_t rings = 0;
					FaceS* pface = vertex.adjface;
					do {
						pface = pface->nextface(&vertex);
					} while (pface != nullptr && pface != vertex.adjface && ++rings <= 16777216);
					vertex.boundary = (pface == nullptr); /* assert(pface != vertex.adjface); */
					if (rings == 16777216) {
						/* Note: when correct face_order, acos() domain is [-1,+1], 
							so should acos(clamp(...,-1.0,+1.0)) */
						throw std::exception();
					}
					
					size_t vertex_valence = vertex.valence();
					if (!vertex.boundary && vertex_valence == 6) {
						vertex.regular = true;
					} else if (vertex.boundary && vertex_valence == 4) {
						vertex.regular = true;
					} else {
						vertex.regular = false;
					}
				}
			}
		}

	public:

		AdjacencyTriangleMesh() : vertex_array(), face_array() {}

		AdjacencyTriangleMesh(AdjacencyTriangleMesh&& other) noexcept : vertex_array(std::move(other.vertex_array)), face_array(std::move(other.face_array)) {}

		AdjacencyTriangleMesh(const AdjacencyTriangleMesh& other) {
			this->vertex_array = other.vertex_array;
			this->face_array = other.face_array;
			
			if (other.num_vertices() != 0) {
				const VertS* olddata = &other.vertex_array[0];
				VertS* newdata = &this->vertex_array[0];
				for (size_t i = 0; i != other.num_faces(); ++i) {
					for (size_t k = 0; k != 3; ++k) {
						assert(this->face_array[i].vertices[k] != nullptr);
						this->face_array[i].vertices[k] = newdata + (other.face_array[i].vertices[k] - olddata);
					}
				}
			}

			if (other.num_faces() != 0) {
				const FaceS* olddata = &other.face_array[0];
				FaceS* newdata = &this->face_array[0];
				for (size_t i = 0; i != other.num_vertices(); ++i) {
					if (this->vertex_array[i].adjface != nullptr) {
						this->vertex_array[i].adjface = newdata + (other.vertex_array[i].adjface - olddata);
					}
				}
				for (size_t i = 0; i != other.num_faces(); ++i) {
					for (size_t k = 0; k != 3; ++k) {
						if (this->face_array[i].adjfaces[k] != nullptr) {
							this->face_array[i].adjfaces[k] = newdata + (other.face_array[i].adjfaces[k] - olddata);
						}
					}
				}
			}
		}

		AdjacencyTriangleMesh& operator=(AdjacencyTriangleMesh&& other) noexcept {
			vertex_array = std::move(other.vertex_array);
			face_array = std::move(other.face_array);
			return *this;
		}

		size_t num_vertices() const {
			return vertex_array.size();
		}

		size_t num_edges() const {
			return face_array.size() * 3 / 2;
		}

		size_t num_faces() const {
			return face_array.size();
		}
	 
		using VertContainer = std::vector<VertS>;
		using VertContainerIterator = typename VertContainer::iterator;
		using FaceContainer = std::vector<FaceS>;
		using FaceContainerIterator = typename FaceContainer::iterator;

		struct FaceIterator;

		struct VertIterator : public std::iterator_traits<VertexProperty*> {
			VertS* pvertex;
			VertContainer* pvertices;
			FaceContainer* pfaces;

			VertIterator() : pvertex(nullptr), pvertices(nullptr), pfaces(nullptr) {}
			VertIterator(VertS* v, VertContainer& V, FaceContainer& F) : pvertex(v), pvertices(&V), pfaces(&F) {}
			
			VertIterator& operator++() { ++pvertex; return *this; }
			VertIterator& operator--() { --pvertex; return *this; }

			VertIterator operator++(int) { VertIterator tmp = *this; ++(*this); return tmp; }
			VertIterator operator--(int) { VertIterator tmp = *this; --(*this); return tmp; }

			VertIterator& operator+=(ptrdiff_t diff) { pvertex += diff; return *this; }
			VertIterator& operator-=(ptrdiff_t diff) { pvertex -= diff; return *this; }

			VertIterator operator+(ptrdiff_t diff) const { return VertIterator(pvertex + diff, *pvertices, *pfaces); }
			VertIterator operator-(ptrdiff_t diff) const { return VertIterator(pvertex - diff, *pvertices, *pfaces); }

			bool operator==(const VertIterator& other) const { return pvertex == other.pvertex; }
			bool operator!=(const VertIterator& other) const { return pvertex != other.pvertex; }

			bool operator==(nullptr_t) const { return pvertex == nullptr; }
			bool operator!=(nullptr_t) const { return pvertex != nullptr; }

			VertexProperty& operator*() const {
				return pvertex->property;
			}

			VertexProperty* operator->() const {
				return &(pvertex->property);
			}

			const VertexDescriptor id() const {
				return pvertex - pvertices->data();
			}

			VertexProperty& property() const {
				return pvertex->property;
			}

			FaceIterator ring_begin() const {
				return FaceIterator(pvertex->adjface, *pvertices, *pfaces);
			}

			FaceIterator ring_end() const {
				return FaceIterator(nullptr, *pvertices, *pfaces);
			}

			size_t valence() const {
				return pvertex->valence();
			}

			bool regular() const {
				return pvertex->regular;
			}

			bool boundary() const {
				return pvertex->boundary;
			}
		};

		struct FaceIterator : public std::iterator_traits<FaceProperty*> {
			FaceS* pface;
			VertContainer* pvertices;
			FaceContainer* pfaces;

			FaceIterator() : pface(nullptr), pvertices(nullptr), pfaces(nullptr) {}
			FaceIterator(FaceS* f, VertContainer& V, FaceContainer& F) : pface(f), pvertices(&V), pfaces(&F) {}

			FaceIterator& operator++() { ++pface; return *this; }
			FaceIterator& operator--() { --pface; return *this; }

			FaceIterator operator++(int) { FaceIterator tmp = *this; ++(*this); return tmp; }
			FaceIterator operator--(int) { FaceIterator tmp = *this; --(*this); return tmp; }

			FaceIterator& operator+=(ptrdiff_t diff) { pface += diff; return *this; }
			FaceIterator& operator-=(ptrdiff_t diff) { pface -= diff; return *this; }

			FaceIterator operator+(ptrdiff_t diff) const { return FaceIterator(pface + diff, *pvertices, *pfaces); }
			FaceIterator operator-(ptrdiff_t diff) const { return FaceIterator(pface - diff, *pvertices, *pfaces); }

			bool operator==(const FaceIterator& other) const { return pface == other.pface; }
			bool operator!=(const FaceIterator& other) const { return pface != other.pface; }

			bool operator==(nullptr_t) const { return pface == nullptr; }
			bool operator!=(nullptr_t) const { return pface != nullptr; }

			FaceProperty& operator*() const { 
				return pface->property; 
			}

			FaceProperty* operator->() const { 
				return &(pface->property);  
			}

			const size_t id() const {
				return pface - pfaces->data();
			}

			FaceProperty& property() const {
				return pface->property;
			}

			size_t vertex_number(VertIterator pvertex) const {
				return pface->vnum(pvertex.pvertex);
			}
			
			VertIterator vertex(size_t i) const {
				assert(i < 3);
				return VertIterator(pface->vertices[i], *pvertices, *pfaces);
			}

			VertIterator vertex_next(VertIterator pvertex) const {
				return VertIterator(pface->nextvert(pvertex.pvertex), *pvertices, *pfaces);
			}

			VertIterator vertex_prev(VertIterator pvertex) const {
				return VertIterator(pface->prevvert(pvertex.pvertex), *pvertices, *pfaces);
			}

			VertIterator vertex_other(VertIterator pv0, VertIterator pv1) const {
				return VertIterator(pface->othervert(pv0.pvertex, pv1.pvertex), *pvertices, *pfaces);
			}

			FaceIterator adjacency_face(size_t i) const {
				return FaceIterator(pface->adjfaces[i], *pvertices, *pfaces);
			}

			FaceIterator ring_next(VertIterator center) const {
				return FaceIterator(pface->nextface(center.pvertex), *pvertices, *pfaces);
			}

			FaceIterator ring_prev(VertIterator center) const {
				return FaceIterator(pface->prevface(center.pvertex), *pvertices, *pfaces);
			}
		};
		

		VertIterator add_vertex(const VertexProperty& property) {
			this->add_vertices(&property, (&property) + 1);
			return VertIterator(&this->vertex_array.back(), vertex_array, face_array);
		}

		/*struct OutEdgeIterator {

		};*/

		std::pair<VertIterator, VertIterator> vertices() {
			return { VertIterator(vertex_array.data(), vertex_array, face_array),
				VertIterator(vertex_array.data()+vertex_array.size(), vertex_array, face_array) };
		}

		std::pair<FaceIterator, FaceIterator> faces() {
			return { FaceIterator(face_array.data(), vertex_array, face_array),
				FaceIterator(face_array.data()+face_array.size(), vertex_array, face_array) };
		}
	};

	template<typename VertexProperty>
	AdjacencyTriangleMesh<VertexProperty> subdiv(const AdjacencyTriangleMesh<VertexProperty>& _mesh, size_t count) {
		typedef decltype(VertexProperty::position) Position;
		AdjacencyTriangleMesh<VertexProperty> mesh = _mesh;

		for(size_t i = 0; i != count; ++i) {
			AdjacencyTriangleMesh<VertexProperty> sdmesh;

			// Allocate memory (geometric growth)
			sdmesh.vertex_array.reserve(mesh.vertex_array.size()*2 * 2);
			sdmesh.face_array.reserve(mesh.face_array.size()*4 * 2);
			
			// Compute source vertex (even vertex)
			for (auto [vertex, vertexlast] = mesh.vertices(); vertex != vertexlast; ++vertex) {
				if ( !vertex.boundary() ) {
					/**
						* @regular
						*         1/16                   1/16
						*            * -- -- -- -- -- --*
						*          / \                 /  \
						*        /     \            /       \
						*       /        \       /           \
						* 1/16/            \ source            \
						*    * ------------- * --------------- * 1/16
						*     \            /  \              /
						*       \       /       \        /
						*         \  /            \   /
						*     1/16 * -- -- -- -- --* 1/16
						* 
						* @irregular
						*                3/16
						*                 *
						*               /  \\
						*             /    \ \
						*           /       \ \
						*         /         \   \
						* 3/16  /         source \
						*     * -- -- -- -- *\    \
						*          \           \    \
						*                 \       \  \
						*                        \  \ \
						*                             * 3/16
					*/
					size_t valence = vertex.valence();
					double beta = vertex.regular() ? 1.0/16.0
						: valence == 3 ? 3.0/16.0
						: 3.0/(valence*8);

					Position position = vertex->position * (1.0 - beta*valence);
					auto ri = vertex.ring_begin();
					do {
						position += ri.vertex_next(vertex)->position * beta;
						ri = ri.ring_next(vertex);
					} while (ri != vertex.ring_begin() && ri != vertex.ring_end());
					
					sdmesh.add_vertex( VertexProperty{position} );
				} else {
					abort();
				}
			}

			// Compute edge vertex (odd vertex), and faces
			std::vector<std::array<size_t, 3>> faces;
			std::map<typename decltype(sdmesh)::EdgeS, typename decltype(sdmesh)::VertIterator> verts;
			for (auto [fi, fiend] = mesh.faces(); fi != fiend; ++fi) {
				size_t o[3];
				size_t s[3];
				for (size_t i = 0; i != 3; ++i) {
					auto sour = fi.vertex(i);
					auto targ = fi.vertex((i+1)%3);
					auto edge = decltype(sdmesh)::EdgeS(std::min(sour.pvertex,targ.pvertex), std::max(sour.pvertex,targ.pvertex));
					if (verts.find(edge) == verts.end()) {
						if (fi.adjacency_face(i) != nullptr) {/* edge != boundary */
							/**
								* @interior
								*            1/8             3/8
								*              * -- -- *- -- *
								*             / \    /  \   /  \
								*           /    \ /     \ /     \
								*         /       * -- -- P        \
								*       /          \    /           \
								*     /             \  /              \
								*    * ------------- *3/8 ------------ * 1/8
								*     \            /  \              /
								*       \       /       \        /
								*         \  /            \   /
								*          * -- -- -- -- --*
								* 
								*                        * 1/8
								*                      /  \
								*                   /      \
								*           3/8  /          \ 3/8
								*              * -- -- P --- *
								*             / \    /  \   /  \
								*           /    \ /     \ /     \
								*         /       * -- -- *        \
								*       /          \    /           \
								*     /             \  /              \
								*    * ------------- *1/8 ------------ *
								*     \            /  \              /
								*       \       /       \        /
								*         \  /            \   /
								*          * -- -- -- -- --*
								* 
								*            3/8             1/8
								*              * -- -- *- -- *
								*             / \    /  \   /  \
								*           /    \ /     \ /     \
								*         /       P -- -- *        \
								*       /          \    /           \
								* 1/8 /             \  /              \
								*    * ------------- *3/8 ------------ *
								*     \            /  \              /
								*       \       /       \        /
								*         \  /            \   /
								*          * -- -- -- -- --*
							*/
							Position position = sour->position * 3/8 + targ->position * 3/8
								+ fi.vertex_other(sour, targ)->position / 8
								+ fi.adjacency_face(i).vertex_other(sour, targ)->position / 8;

							verts.insert_or_assign(edge, sdmesh.add_vertex( VertexProperty{position} ));
						} else {
							abort();
						}
					}

					o[i] = sour.id();
					s[i] = verts[edge].id();
				}

				/**
					*       0
					*      /\
					*  s2 /__\ s0
					*    /\  /\
					* 2 /__\/__\ 1
					*      s1
				*/
				faces.push_back({o[0],s[0],s[2]});
				faces.push_back({s[0],o[1],s[1]});
				faces.push_back({s[1],o[2],s[2]});
				faces.push_back({s[2],s[0],s[1]});
			}

			sdmesh.add_triangles(&(faces[0][0]), (&(faces[0][0])) + faces.size()*3);
			mesh = std::move(sdmesh);
		}

		return mesh;
	}


	template<typename VertexProperty>
	concept bvh_aabboundary_vertex = requires(VertexProperty v) {
		v.boundary;
		v.boundary.center();
		v.boundary.halfextents();
		VertexProperty{ v.boundary };
	};

	template<typename VertexProperty>
	concept bvh_spherebounndary_vertex = requires(VertexProperty v) {
		v.boundary.center();
		v.boundary.radius();
		VertexProperty{ v.boundary };
	};

	template<typename Graph, typename VertexProperties>
	void bvh(std::vector<VertexProperties> vertices, Graph& graph, size_t maxdepth = -1) {
		if (maxdepth <= 1 || vertices.size() <= 1) {
			for (auto v = vertices.begin(); v != vertices.end(); ++v) {
				graph.add_vertex(*v);
			}
			return;
		}

		std::queue< std::tuple<typename Graph::vertex_descriptor, size_t, 
			decltype(vertices.begin()), decltype(vertices.end())> > Q;
		Q.push({-1, 0, vertices.begin(), vertices.end()});
		while (!Q.empty()) {
			auto [parent,depth,vstart,vend] = Q.front();
			Q.pop();

			if (std::next(vstart) == vend || depth + 1 == maxdepth) {
				// Insert leaf nodes.
				for (auto vi = vstart; vi != vend; ++vi) {
					graph.add_edge(graph.find(parent), graph.add_vertex(*vi));
				}
			} else {
				// Get boundary of all primitives.
				auto boundary = vstart->boundary;
				for (auto vi = std::next(vstart); vi != vend; ++vi) {
					boundary = Union(boundary, vi->boundary);
				}
				// Get max-extend by boundary.
				size_t dim = 0;
				if constexpr (bvh_aabboundary_vertex<VertexProperties>) {
					dim = (boundary.halfextents()[0] > boundary.halfextents()[1] && boundary.halfextents()[0] > boundary.halfextents()[2])
						? 0 : (boundary.halfextents()[1] > boundary.halfextents()[2])
						? 1 :
						2;
				}

				// Partition by max-extend.
				auto vmid = std::partition(vstart, vend,
					[&boundary,&dim](const VertexProperties& v) { return v.boundary.center()[dim] < boundary.center()[dim]; }
				);
				// Recompute mid by partition failed.
				if (vmid == vstart || vmid == vend) {
					vmid = std::next(vstart, std::distance(vstart, vend) / 2);
				}

				// Insert interior node, and push two branch by mid.
				auto interior = graph.add_vertex(VertexProperties{ boundary });
				if (parent != -1) {
					graph.add_edge(graph.find(parent), interior);
					}
				Q.push({interior.id(), depth+1, vstart, vmid});
				Q.push({interior.id(), depth+1, vmid, vend});
			}
		}
	}

	template<typename Graph, typename VertexProperties>
	Graph bvh(std::vector<VertexProperties> vertices, size_t maxdepth = -1) {
	Graph graph;
	bvh(vertices, graph, maxdepth);
	return graph;
}
#endif
}// end of namespace calculation