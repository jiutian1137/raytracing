///Jiang1998Nan@outlook.com

#include <string>
#include <iostream>
#include <algorithm>
#include <execution>
#include <vector>
#include <map>
#include <stack>
#include <fstream>

#include <math/concepts.hpp>
#include <math/mdarray_vector.hpp>
using scalar = float;
using vector2 = math::smdarray<float, 2, __m128>;
using vector3 = math::smdarray<float, 3, __m128>;
using vector4 = math::smdarray<float, 4, __m128>;
using spectrum = math::smdarray<float, 3, __m128>;

#include <random>
#include <math/random/sobol_seq.hpp>

#include <math/geometry/transform.hpp>
#include <math/number/quaternion.hpp>
#include <math/integral.hpp>

#include <math/physics/microfacet.hpp>
#include <math/physics/light_transport.hpp>

/// cv::imread.
#include <opencv2/opencv.hpp>
#pragma comment(lib, "opencv_world440.lib")

/// gltf.
#include <property_tree/json.hpp>
#include <base64.hpp>

struct raytracing_traits {
	using scalar_type = scalar;
	using vector3_type = vector3;
	using spectrum_type = spectrum;

	using ray_type = math::geometry::ray<vector3>;
	
	struct arguments_type { vector2 random; };
	
	using attribute_type = vector4;

	struct intersect_result_type {
		math::geometry::ray<vector3> ray;
		scalar distance; 
		size_t element; 
		const math::physics::raytracing::primitive<raytracing_traits>* primitive;
		const void* object;
		math::geometry::flexibility<vector3> transformation;
	};

	struct interact_result_type { 
		spectrum radiance;
		spectrum transmittance;

		/// importance sample lightsource, dsolidsngle as hit rate.
		math::geometry::ray<vector3> incident_ray;
		scalar incident_distance;
		spectrum incident_radiance;

		struct stack_value {
			const math::physics::raytracing::primitive<raytracing_traits>* primitive;
			const void* object;
			scalar ior;
		};
		std::stack<stack_value> primitives;
		void push(const math::physics::raytracing::primitive<raytracing_traits>* primitive, const void* object, scalar ior) { primitives.push({primitive, object, ior}); }
		void pop() { primitives.pop(); }
		auto top() const { return primitives.top().primitive; }
		auto top_obj() const { return primitives.top().object; }
		bool empty() const { return primitives.empty(); }
		scalar enter_ior() const { return primitives.empty() ? scalar(1) : primitives.top().ior; }
		scalar exit_ior() const { return primitives.size() < 2 ? scalar(1) : primitives._Get_container()[primitives.size() - 2].ior; }
	};
};

#include <graphics/opengl/library2.hpp>
#pragma comment(lib, "opengl32.lib")
#include <graphics/wex.hpp>

struct APPLICATION : wex::mainwindow<wex::window_opengl> {
	using base = wex::mainwindow<wex::window_opengl>;

	explicit 
	APPLICATION(wex::window& the_window) : base(the_window) {
		opengl_context =
			wex::createGLRC(base::context);
		wex::bindGLRC(base::context, *opengl_context);
		gl = GLlibrary2(L"opengl32.dll");

		size_t width, height;
		std::tie(width,height) = base::window().get_clientsize();
		result = decltype(result)( mdsize_t<2>{width,height} );
		finresult = decltype(finresult)( mdsize_t<2>{width,height} );
		view   = decltype(view){ {1,0,0,0}, {0,5,-15} /* vector3{1,0,0},vector3{0,1,0},vector3{0,0,1}, vector3{0,15,-15}*/ };
		view.rotate({1,0,0},0.2f);
		/*view.reposit({ 11.4307f,4.93676f,12.5719f });
		view.redirect(2, { -0.622702f,-0.263185f,-0.736869f });*/
		proj   = decltype(proj)( scalar(0.8f), scalar(0.8f) );

		it.subsample_count = 16;
		it.sample_count = 16384;

		sequences.resize(it.sample_count);
		math::sobol_seq<unsigned> seq(12222);
		for (auto& sample_i : sequences) {
			seq.generate(2, sample_i.data());
		}

		scrembles.resize(128, 128);
		{
			std::cout << "shuffle bluenoise" << std::endl;
			std::default_random_engine rng;
			std::uniform_int_distribution<unsigned> dist;
			for (size_t i = 0; i != scrembles.length(); ++i) {
				scrembles[i] = { dist(rng), dist(rng) };
			}
#if 0
			math::shuffle_noise(std::execution::par, scrembles, rng, std::less<>(), [](const math::matrix_view<math::smdarray<unsigned,2>>& scrembles, const math::mdarray<size_t,math::smdsize_t<1>>& sample_indices, size_t spatial_i, size_t sample_i) {
				const size_t x = spatial_i % scrembles.size(1);
				const size_t y = spatial_i / scrembles.size(1);
				constexpr auto invd = math::smdarray<float, 2>{ 1.0f/float(std::numeric_limits<unsigned>::max()), 1.0f/float(std::numeric_limits<unsigned>::max()) };
				scalar energy_sum = 0;
				for (size_t _ky = y - 6; _ky != y + 6; ++_ky) {
					for (size_t _kx = x - 6; _kx != x + 6; ++_kx) {
						size_t kx = (_kx + scrembles.size(1)) % scrembles.size(1);
						size_t ky = (_ky + scrembles.size(1)) % scrembles.size(1);
						if (kx == x && ky == y) {
							continue;
						}
					
						scalar dx = abs(scalar(kx) - scalar(x));
						scalar dy = abs(scalar(ky) - scalar(y));
						dx = std::min(dx, scalar(scrembles.size(1)) - dx);
						dy = std::min(dy, scalar(scrembles.size(1)) - dy);
						scalar sqr_spatial_distance = (dx*dx + dy*dy);
					
						scalar     sample_distance  = (length(static_vcast<math::smdarray<float,2>>(scrembles[sample_i])*invd - static_vcast<math::smdarray<float,2>>(scrembles[sample_indices[ky*scrembles.size(1) + kx]])*invd));
					
						energy_sum += exp(-sqr_spatial_distance/static_cast<scalar>(2.1 * 2.1) - sample_distance/static_cast<scalar>(1 * 1));
					}
				}

				return energy_sum;
			});
#endif
			std::cout << "OK" << std::endl;
		}

		property_tree::json_string jsgltf;
		std::string filefolder = "New Folder\\";
		{
			//std::ifstream fin("models/Lamborghini Centenario LP-770 Baby Blue SDC/source/LAMBORGHINI CENTENARIO INTERIOR SDCBB.gltf", std::ios::in|std::ios::binary);
			std::ifstream fin(filefolder+"scene.gltf", std::ios::in|std::ios::binary);
			fin.seekg(0, std::ios::beg);
			fin.seekg(0, std::ios::end);
			std::string source(size_t(fin.tellg()), '\0');
			fin.seekg(0, std::ios::beg);
			fin.read(source.data(), source.size());
			fin.close();
			jsgltf = std::move(property_tree::json_string(source.begin(), source.end()));
		}

		auto Diamond = std::make_shared< math::physics::raytracing::dielectric<raytracing_traits> >();
		Diamond->eta  = 2.4065f;
		Diamond->multiplier = {1,1,1};
		Diamond->roughness = 0.3f;
		materials.insert_or_assign("Diamond", Diamond);

		auto Ice = std::make_shared< math::physics::raytracing::dielectric<raytracing_traits> >();
		Ice->eta  = 1.3069f;
		Ice->multiplier = {1,1,1};
		Ice->roughness = 0.3f;
		materials.insert_or_assign("Ice", Ice);
		
		auto Gold = std::make_shared< math::physics::raytracing::conductor<raytracing_traits> >();
		Gold->eta  = {0.13100f,0.42415f,1.4684f};
		Gold->etak = {4.0624f,2.4721f,1.9530f};
		Gold->roughness = {0.5f,0.5f,1.0f};
		materials.insert_or_assign("Gold", Gold);

		auto Silver = std::make_shared< math::physics::raytracing::conductor<raytracing_traits> >();
		Silver->eta  = {0.041000f,0.059582f,0.050000f};
		Silver->etak = {4.8025f,3.5974f,2.1035f};
		Silver->roughness = {1.0f,1.0f,1.0f};
		materials.insert_or_assign("Silver", Silver);
		/*auto Silver = std::make_shared< math::raytracing::principle_brdf<raytracing_traits> >();
		Silver->color_factor  = {0.7f,0.7f,0.7f};
		Silver->metallic_factor = 0.0f;
		Silver->roughness_factor = {1.0f,1.0f,1.0f};
		materials.insert_or_assign("Silver", Silver);*/
		
		auto Copper = std::make_shared< math::physics::raytracing::conductor<raytracing_traits> >();
		Copper->eta  = {0.21258f,0.55135f,1.2878f};
		Copper->etak = {4.1003f,2.8023f,2.2596f};
		Copper->roughness = {0.5f,0.5f,1.0f};
		materials.insert_or_assign("Copper", Copper);

		/*auto Volume = std::make_shared< math::raytracing::empty_volume<raytracing_traits> >();
		Volume->sigma_t = {0.01f,0.01f,0.01f};
		Volume->g = 0.35f;
		materials.insert_or_assign("Volume", Volume);*/

		std::cout << jsgltf["materials"].size() << std::endl;
		for (size_t i = 0, iend = jsgltf["materials"].size(); i != iend; ++i) {
			const auto jsmaterial = jsgltf["materials"][i];
			auto material = std::make_shared< math::physics::raytracing::principle_brdf<raytracing_traits> >();

			if (jsmaterial.contains("pbrMetallicRoughness")) {
				if (jsmaterial["pbrMetallicRoughness"].contains("roughnessFactor")) {
					float r = jsmaterial["pbrMetallicRoughness"]["roughnessFactor"];
					material->roughness_factor = { r,r, 1.0f };
				} else {
					material->roughness_factor = {0.5f,0.5f,1.0f};
				}
			
				if (jsmaterial["pbrMetallicRoughness"].contains("baseColorFactor")) {
					auto baseColorFactor = jsmaterial["pbrMetallicRoughness"]["baseColorFactor"];
					material->color_factor = {baseColorFactor[0], baseColorFactor[1], baseColorFactor[2],1};
				} else {
					material->color_factor = {1,1,1,1};
				}

				if (jsmaterial["pbrMetallicRoughness"].contains("metallicFactor")) {
					material->metallic_factor = jsmaterial["pbrMetallicRoughness"]["metallicFactor"];
				} else {
					material->metallic_factor = 1.0f;
				}

				if (jsmaterial["pbrMetallicRoughness"].contains("baseColorTexture")) {
					auto jstexture = jsgltf["textures"][ jsmaterial["pbrMetallicRoughness"]["baseColorTexture"]["index"] ];
					if (jstexture.contains("texCoord")) {
						material->color_texture_index = jstexture["texCoord"];
					}

					auto jsimage = jsgltf["images"][ jstexture["source"] ];
					if (jsimage.contains("bufferView")) {
						
					} else if (jsimage.contains("uri")) {
						auto image = cv::imread(filefolder + std::string(jsimage["uri"]), cv::IMREAD_UNCHANGED);
						assert(image.channels() == 3 || image.channels() == 4);
						material->color.resize(mdsize_t<2>{ size_t(image.cols), size_t(image.rows) });
						if (image.channels() == 4) {
							for (size_t k = 0, kend = material->color.length(); k != kend; ++k) {
								auto temp = image.at<cv::Vec4b>(int(k));
								material->color[k] = { float(temp[2])/255.0f,float(temp[1])/255.0f,float(temp[0])/255.0f,float(temp[3])/255.0f };
							}
						} else if (image.channels() == 3) {
							for (size_t k = 0, kend = material->color.length(); k != kend; ++k) {
								auto temp = image.at<cv::Vec3b>(int(k));
								material->color[k] = { float(temp[2])/255.0f,float(temp[1])/255.0f,float(temp[0])/255.0f,1 };
							}
						} else {
							abort();
						}
					}
				}

				if (jsmaterial["pbrMetallicRoughness"].contains("metallicRoughnessTexture")) {
					auto jstexture = jsgltf["textures"][ jsmaterial["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"] ];
					if (jstexture.contains("texCoord")) {
						material->metallic_and_roughness_texture_index = jstexture["texCoord"];
					}

					auto jsimage = jsgltf["images"][ jstexture["source"] ];
					if (jsimage.contains("bufferView")) {
						
					} else if (jsimage.contains("uri")) {
						auto image = cv::imread(filefolder + std::string(jsimage["uri"]), cv::IMREAD_UNCHANGED);
						assert(image.channels() == 3 || image.channels() == 4);
						material->metallic_and_roughness.resize(mdsize_t<2>{ size_t(image.cols), size_t(image.rows) });
						if (image.channels() == 4) {
							for (size_t k = 0, kend = material->color.length(); k != kend; ++k) {
								auto temp = image.at<cv::Vec4b>(int(k));
								material->metallic_and_roughness[k] = {temp[0]/255.0f,temp[1]/255.0f};
							}
						} else if (image.channels() == 3) {
							for (size_t k = 0, kend = material->color.length(); k != kend; ++k) {
								auto temp = image.at<cv::Vec3b>(int(k));
								material->metallic_and_roughness[k] = {temp[0]/255.0f,temp[1]/255.0f};
							}
						} else {
							abort();
						}
					}
				}

				material->attributes.insert(material->attributes.end(),
					max(material->color_texture_index, material->metallic_and_roughness_texture_index)+1, {});

				//if (jsmaterial["pbrMetallicRoughness"].contains("baseColorFactor")) {
				//	auto baseColorFactor = jsmaterial["pbrMetallicRoughness"]["baseColorFactor"];
				//	material->etak = {4.8025f,3.5974f,2.1035f};
				//	material->eta = {max(1 - scalar(baseColorFactor[0]),0.05f)/* * material->etak[0]*/ * 3, max(1 - scalar(baseColorFactor[1]),0.05f)/* * material->etak[1]*/ * 1.5f, max(1 - scalar(baseColorFactor[2]),0.05f)/* * material->etak[2]*/};
				//} else {
				//	material->eta  = {0.041000f,0.059582f,0.050000f};
				//	material->etak = {4.8025f,3.5974f,2.1035f};
				//}
			} else {
				material->color_factor = {1,0,0,1};
				material->roughness_factor = {0.01f,0.01f,1.0f};
				material->metallic_factor = 1.0f;
			}
		
			if (jsmaterial.contains("doubleSided") && bool(jsmaterial["doubleSided"])) {
				material->doublesided = true;
			}

			if (jsmaterial.contains("alphaMode") && std::string_view(jsmaterial["alphaMode"]) != "OPAQUE") {
				material->opaque = false;
			}

			materials[jsmaterial["name"]] = material;
		}

		std::vector< std::shared_ptr<std::vector<unsigned char>> > buffers(jsgltf["buffers"].size());
		for (size_t i = 0; i != buffers.size(); ++i) {
			buffers[i] = std::make_shared<std::vector<unsigned char>>( jsgltf["buffers"][i]["byteLength"] );
			std::string_view buffer_i = jsgltf["buffers"][i]["uri"];
			if (buffer_i.starts_with("data:application/octet-stream;base64,")) {
				buffer_i = buffer_i.substr(strlen("data:application/octet-stream;base64,"));
				base64::decode(buffers[i]->data(), buffer_i.data(), buffer_i.size());
			}
			else if (buffer_i.ends_with(".bin")) {
				std::ifstream fin(filefolder + std::string(buffer_i), std::ios::in|std::ios::binary);
				fin.read((char*)buffers[i]->data(), buffers[i]->size());
				fin.close();
			}
		}

		for (size_t i = 0, iend = jsgltf["meshes"].size(); i != iend; ++i) {
			auto jsmesh = jsgltf["meshes"][i];
			scene.geometries.push_back({});
			for (size_t j = 0, jend = jsmesh["primitives"].size(); j != jend; ++j) {
				auto jsprimitive = jsmesh["primitives"][j];
				auto jsindices = jsgltf["accessors"][jsprimitive["indices"]];
				auto jsposition = jsgltf["accessors"][jsprimitive["attributes"]["POSITION"]];
				
				if (int(jsindices["componentType"]) == GL_UNSIGNED_SHORT) {
					auto primitive = std::make_shared<math::physics::raytracing::triangle_mesh<raytracing_traits, math::smdarray<unsigned short,3>>>();
					primitive->material = materials[jsgltf["materials"][jsprimitive["material"]]["name"]].get();
					
					auto bufferView = jsgltf["bufferViews"][jsindices["bufferView"]]; 
					primitive->bind_element_stream(buffers[ bufferView["buffer"] ], (bufferView.contains("byteOffset") ? size_t(bufferView["byteOffset"]) : 0) + (jsindices.contains("byteOffset") ? size_t(jsindices["byteOffset"]) : 0),
						min(size_t(jsindices["count"])*sizeof(unsigned short), size_t(bufferView["byteLength"])));

					bufferView = jsgltf["bufferViews"][jsposition["bufferView"]];
					primitive->num_vertices = jsposition["count"];
					primitive->set_position_attribute<math::smdarray<float,3>>();
					primitive->bind_position_stream(buffers[ bufferView["buffer"] ], (bufferView.contains("byteOffset") ? size_t(bufferView["byteOffset"]) : 0) + (jsposition.contains("byteOffset") ? size_t(jsposition["byteOffset"]) : 0), 
						bufferView.contains("byteStride") ? bufferView["byteStride"] : sizeof(math::smdarray<float,3>) );

					if (jsprimitive["attributes"].contains("NORMAL")) {
						auto jsnormal = jsgltf["accessors"][jsprimitive["attributes"]["NORMAL"]];
						bufferView = jsgltf["bufferViews"][jsnormal["bufferView"]];
						primitive->set_normal_attribute<math::smdarray<float,3>>(0);
						primitive->bind_normal_stream(0, buffers[ bufferView["buffer"] ], (bufferView.contains("byteOffset") ? size_t(bufferView["byteOffset"]) : 0) + (jsnormal.contains("byteOffset") ? size_t(jsnormal["byteOffset"]) : 0),
							bufferView.contains("byteStride") ? bufferView["byteStride"] : sizeof(math::smdarray<float,3>) );
					}

					for (size_t i = 0; true; ++i) {
						std::string TEXCOORD_i = "TEXCOORD_" + std::to_string(i);
						if (!jsprimitive["attributes"].contains(TEXCOORD_i.c_str())) {
							break;
						}
						auto jstexcoord = jsgltf["accessors"][jsprimitive["attributes"][TEXCOORD_i]];
						bufferView = jsgltf["bufferViews"][jstexcoord["bufferView"]];
						primitive->set_texcoord_attribute<math::smdarray<float,2>>(i);
						primitive->bind_texcoord_stream(i, buffers[ bufferView["buffer"] ], (bufferView.contains("byteOffset") ? size_t(bufferView["byteOffset"]) : 0) + (jstexcoord.contains("byteOffset") ? size_t(jstexcoord["byteOffset"]) : 0),
							bufferView.contains("byteStride") ? bufferView["byteStride"] : sizeof(math::smdarray<float,2>) );
					}

					primitive->build_bvh();

					scene.geometries.back().push_back(primitive);
				} else if (int(jsindices["componentType"]) == GL_UNSIGNED_INT) {
					auto primitive = std::make_shared<math::physics::raytracing::triangle_mesh<raytracing_traits, math::smdarray<unsigned int,3>>>();
					primitive->material = materials[jsgltf["materials"][jsprimitive["material"]]["name"]].get();
					
					auto bufferView = jsgltf["bufferViews"][jsindices["bufferView"]]; 
					primitive->bind_element_stream(buffers[ bufferView["buffer"] ], (bufferView.contains("byteOffset") ? size_t(bufferView["byteOffset"]) : 0) + (jsindices.contains("byteOffset") ? size_t(jsindices["byteOffset"]) : 0),
						min(size_t(jsindices["count"])*sizeof(unsigned int), size_t(bufferView["byteLength"])));

					bufferView = jsgltf["bufferViews"][jsposition["bufferView"]];
					primitive->num_vertices = jsposition["count"];
					primitive->set_position_attribute<math::smdarray<float,3>>();
					primitive->bind_position_stream(buffers[ bufferView["buffer"] ], (bufferView.contains("byteOffset") ? size_t(bufferView["byteOffset"]) : 0) + (jsposition.contains("byteOffset") ? size_t(jsposition["byteOffset"]) : 0),
						bufferView.contains("byteStride") ? bufferView["byteStride"] : sizeof(math::smdarray<float,3>) );

					if (jsprimitive["attributes"].contains("NORMAL")) {
						auto jsnormal = jsgltf["accessors"][jsprimitive["attributes"]["NORMAL"]];
						bufferView = jsgltf["bufferViews"][jsnormal["bufferView"]];
						primitive->set_normal_attribute<math::smdarray<float,3>>(0);
						primitive->bind_normal_stream(0, buffers[ bufferView["buffer"] ], (bufferView.contains("byteOffset") ? size_t(bufferView["byteOffset"]) : 0) + (jsnormal.contains("byteOffset") ? size_t(jsnormal["byteOffset"]) : 0),
							bufferView.contains("byteStride") ? bufferView["byteStride"] : sizeof(math::smdarray<float,3>) );
					}

					for (size_t i = 0; true; ++i) {
						std::string TEXCOORD_i = "TEXCOORD_" + std::to_string(i);
						if (!jsprimitive["attributes"].contains(TEXCOORD_i.c_str())) {
							break;
						}
						auto jstexcoord = jsgltf["accessors"][jsprimitive["attributes"][TEXCOORD_i]];
						bufferView = jsgltf["bufferViews"][jstexcoord["bufferView"]];
						primitive->set_texcoord_attribute<math::smdarray<float,2>>(i);
						primitive->bind_texcoord_stream(i, buffers[ bufferView["buffer"] ], (bufferView.contains("byteOffset") ? size_t(bufferView["byteOffset"]) : 0) + (jstexcoord.contains("byteOffset") ? size_t(jstexcoord["byteOffset"]) : 0),
							bufferView.contains("byteStride") ? bufferView["byteStride"] : sizeof(math::smdarray<float,2>) );
					}

					primitive->build_bvh();

					scene.geometries.back().push_back(primitive);
				} else {
					throw;
				}
			}
		}

		for (size_t i = 0, iend = jsgltf["nodes"].size(); i != iend; ++i) {
			auto jsnode = jsgltf["nodes"][i];
			if (jsnode.contains("children")) {
				scene.nodes.push_back({decltype(scene)::types::node, scene.node_instances.size()});
				scene.node_instances.push_back(std::vector<size_t>(jsnode["children"].size()));
				for (size_t j = 0; j != scene.node_instances.back().size(); ++j) {
					scene.node_instances.back()[j] = jsnode["children"][j];
				}
			} else if (jsnode.contains("mesh")){
				scene.nodes.push_back({decltype(scene)::types::geometry, scene.geometry_instances.size()});
				scene.geometry_instances.push_back(jsnode["mesh"]);
			} else {
				scene.nodes.push_back({decltype(scene)::types::node, scene.node_instances.size()});
				scene.node_instances.push_back(std::vector<size_t>(0));
			}

			if (/*jsnode.contains("children") || jsnode.contains("mesh")*/true) {
				scene.nodes.back().geom2world = {vector3{1,0,0},vector3{0,1,0},vector3{0,0,1},vector3{0,0,0}};

				if (jsnode.contains("rotation")) {
					auto q = jsnode["rotation"];
					float qx = q[0], qy = q[1], qz = q[2], qw = q[3];
					float s = 2/(qx*qx + qy*qy + qz*qz + qw*qw);
					scene.nodes.back().geom2world.V[0] = { (1-s*(qy*qy+qz*qz)), (s*(qx*qy+qw*qz)), (s*(qx*qz-qw*qy)) };
					scene.nodes.back().geom2world.V[1] = { (s*(qx*qy-qw*qz)), (1-s*(qx*qx+qz*qz)), (s*(qy*qz+qw*qx)) };
					scene.nodes.back().geom2world.V[2] = { (s*(qx*qz+qw*qy)), (s*(qy*qz-qw*qx)), (1-s*(qx*qx+qy*qy)) };
				}

				if (jsnode.contains("scale")) {
					auto jsscale = jsnode["scale"];
					scene.nodes.back().geom2world.scale({jsscale[0],jsscale[1],jsscale[2]});
				}

				if (jsnode.contains("translation")) {
					auto jstranslation = jsnode["translation"];
					scene.nodes.back().geom2world.translate({jstranslation[0],jstranslation[1],jstranslation[2]});
				}

				scene.nodes.back().geom2local = scene.nodes.back().geom2world;
				scene.nodes.back().name = jsnode["name"];
			}
		}
		if (!jsgltf["scenes"].empty()) {
			std::queue<size_t> S;
			std::vector<bool> visited(scene.nodes.size(), false);
			for (size_t i = 0; i != scene.nodes.size(); ++i) {
				if (scene.nodes[i].type == decltype(scene)::types::node) {
					for (size_t child : scene.node_instances[scene.nodes[i].index]) {
						visited[child] = true;
					}
				}
			}
			for (size_t i = 0; i != visited.size(); ++i) {
				if (!visited[i]) {
					S.push(i);
				}
			}
			/*for (size_t i = 0, iend = jsgltf["scenes"][0]["nodes"].size(); i != iend; ++i) {
				S.push(jsgltf["scenes"][0]["nodes"][i]);
			}*/

			while (!S.empty()) {
				size_t front = S.front();
				S.pop();

				if (scene.nodes[front].type == decltype(scene)::types::node) {
					/*if (scene.nodes[front].name == "Empty.002") {
						std::cout << "debug";
					}*/
					for (size_t child : scene.node_instances[scene.nodes[front].index]) {
						scene.nodes[child].geom2world = scene.nodes[front].geom2world.transform(scene.nodes[child].geom2local);
						S.push(child);
					}
				}
			}
		}

		srand(unsigned(time(nullptr)));

		//geometry_t many_balls;
		//for (size_t i = 0; i != 3; ++i) {
		//	auto metalball = std::make_shared< math::raytracing::shape_primitive<raytracing_traits, math::geometry::sphere<vector3>> >();
		//	metalball->material = rand() % 2 ? materials["Silver"].get() : materials["Gold"].get(); 
		//	metalball->center() = {rand() / float(RAND_MAX) * 2 - 1, rand() / float(RAND_MAX) * 2 - 1, rand() / float(RAND_MAX) * 2 - 1};
		//	metalball->center() = normalize(metalball->center());
		//	metalball->center() *= (rand()/float(RAND_MAX)*5 + 10);
		//	metalball->radius() = rand()/float(RAND_MAX)*0.5f + 1.5f;
		//	//metalball->center() = { 5, 12 - 1 - metalball->radius(), 0 };
		//	metalball->center()[1] = max(metalball->center()[1], metalball->radius());
		//	many_balls.push_back(metalball);
		//}
		//geometries.push_back(many_balls);

		scene.geometries.push_back({});
		for (size_t i = 0; i != 15; ++i) {
			auto metalbox = std::make_shared< math::physics::raytracing::shape_primitive<raytracing_traits, math::geometry::box<vector3>> >();
			metalbox->material = rand() % 2 ? materials["Silver"].get() : materials["Gold"].get(); 
			metalbox->halfextents() = { rand()/float(RAND_MAX)*1 + 0.5f, rand()/float(RAND_MAX)*1 + 0.5f, rand()/float(RAND_MAX)*1 + 0.5f };
			metalbox->center() = { rand()/float(RAND_MAX)*2 - 1, rand()/float(RAND_MAX), rand()/float(RAND_MAX)*2 - 1 };
			metalbox->center() = normalize(metalbox->center());
			metalbox->center() *= (rand()/float(RAND_MAX)*10 + 5);
			metalbox->center()[1] = metalbox->halfextents()[1]-0.1f;
			scene.geometries.back().push_back(metalbox);
		}
		
		auto ground = std::make_shared< math::physics::raytracing::shape_primitive<raytracing_traits, math::geometry::box<vector3>> >();
		ground->material = materials["Silver"].get();
		ground->center() = {0,0,0}; 
		ground->halfextents() = {1000,0.01f,1000};
		//ground->invert = true;
	/*	ground->normal() = {0,1,0};
		ground->distance() = 0;*/
		scene.geometries.push_back({});
		scene.geometries.back().push_back(ground);
		scene.nodes.push_back(decltype(scene)::node_type{.type=decltype(scene)::types::geometry, .index = scene.geometry_instances.size(), .geom2world = {vector3{1,0,0},vector3{0,1,0},vector3{0,0,1},{0,0,0}}});
		scene.geometry_instances.push_back(scene.geometries.size() - 1);

		scene.build_bvh();
#if 0
#if 1
#endif
		/*, "bunny.ply"*/
		std::vector<std::string> filenames = {"dragon.ply"/*"happy.ply", "dragon.ply"*/};
		for (const auto& filename : filenames) {
			auto ply = std::make_shared<math::raytracing::triangle_mesh<raytracing_traits, math::smdarray<unsigned int,3>>>();

			std::fstream fin(filename, std::ios::in); assert(fin.is_open());
			std::string str;
			size_t i = 0;
			std::vector<std::pair<std::string, std::string>> properties;
			std::string format;
			std::string version;
			size_t element_face;
			fin >> str; assert(str == "ply");
			fin >> str; assert(str == "format"); fin >> format >> version;
			fin >> str; while (str == "comment") { std::getline(fin, str);
				fin >> str; } 
			assert(str == "element"); fin >> str; assert(str == "vertex"); fin >> ply->num_vertices;
			fin >> str; while (str == "property") { properties.push_back({}); fin >> properties.back().first >> properties.back().second;
				fin >> str; }
			assert(str == "element"); fin >> str; assert(str == "face"); fin >> element_face;
			if (format == "ascii") {
				while (str != "end_header") { fin >> str; }
				size_t sizeof_vertex = sizeof(float) * properties.size();
				auto buffer = std::make_shared<std::vector<unsigned char>>(ply->num_vertices * sizeof_vertex);
				auto* fptr = reinterpret_cast<float*>(buffer->data());
				for (size_t i = 0; i != ply->num_vertices; ++i) {
					for (size_t j = 0; j != properties.size(); ++j) {
						fin >> (*fptr++);
					}
				}
				auto ebuffer = std::make_shared<std::vector<unsigned char>>(element_face * sizeof(unsigned int)*3);
				auto* eptr = reinterpret_cast<unsigned int*>(ebuffer->data());
				for (size_t i = 0; i != element_face; ++i) {
					size_t j; fin >> j; assert(j == 3);
					fin >> (*eptr++) >> (*eptr++) >> (*eptr++);
				}

				size_t x = -1, y = -1, z = -1,
					nx = -1, ny = -1, nz = -1;
				for (size_t i = 0; i != properties.size(); ++i) {
					if (properties[i].second == "x") {
						x = i;
					} else if (properties[i].second == "y") {
						y = i;
					} else if (properties[i].second == "z") {
						z = i;
					} else if (properties[i].second == "nx") {
						nx = i;
					} else if (properties[i].second == "ny") {
						ny = i;
					} else if (properties[i].second == "nz") {
						nz = i;
					}
				}
				assert(x != -1 && y != -1 && z != -1 && ((x+y+z)/3 == x+1 || (x+y+z)/3 == x || (x+y+z)/3 == x-1));
				ply->set_position_attribute<math::smdarray<float,3>>();
				ply->bind_position_stream(buffer, sizeof(float)*min(min(x,y),z), sizeof_vertex);
				if (nx != -1) {
					assert(nx != -1 && ny != -1 && nz != -1 && ((nx+ny+nz)/3 == nx+1 || (nx+ny+nz)/3 == nx || (nx+ny+nz)/3 == nx-1));
					ply->set_normal_attribute<math::smdarray<float,3>>(0);
					ply->bind_normal_stream(0, buffer, sizeof(float)*min(min(nx,ny),nz), sizeof_vertex);
				}
				ply->bind_element_stream(ebuffer, 0, ebuffer->size());
			} else {
				/// ...
			}

			ply->build_bvh();
			//ply->material = rand()%2 ? materials["Gold"].get() : materials["Silver"].get();
			ply->material = materials["TIRE32"].get();
			/*std::cout << "format " << format << " " << version << std::endl;
			std::cout << "element vertex " << ply->num_vertices << std::endl;
			for (const auto& prop : properties) {
				std::cout << "property " << prop.first << " " << prop.second << std::endl;
			}
			std::cout << "element face " << ply->num_elements << std::endl;*/

			scene.geometries.push_back({});
			scene.geometries.back().push_back(ply);
		}

		/*scene.nodes.push_back(decltype(scene)::node_type{.index=scene.geometry_instances.size(), .geom2world={{vector3{1,0,0},vector3{0,1,0},vector3{0,0,1}},{0,0,0}}});
		scene.geometry_instances.push_back(0);*/
		scene.nodes.push_back(decltype(scene)::node_type{.index=scene.geometry_instances.size(), .geom2world={{vector3{1,0,0},vector3{0,1,0},vector3{0,0,1}},{0,0,0}}});
		scene.geometry_instances.push_back(1);
		for (int i = 0; i <= 0; ++i) {
			for (int j = 0; j <= 0; ++j) {
				float angle = (rand()/float(RAND_MAX) * 2 - 1) * 6.28f; 
				float x = (rand()/float(RAND_MAX) * 2 - 1);
				float z = (rand()/float(RAND_MAX) * 2 - 1);
				scene.nodes.push_back(decltype(scene)::node_type{.index=scene.geometry_instances.size(), .geom2world = {{vector3{10,0,0},vector3{0,10,0},vector3{0,0,10}},{i * 2.0f/*+x*/,-0.65f,j * 2.0f/*+z*/}}});
				scene.geometry_instances.push_back(2/* + (rand() % 2)*/);
				//scene.nodes.back().geom2world.rotate({0,1,0}, angle);
			}
		}
		scene.build_bvh();
		selected = &scene.nodes.back();
#endif

		/*auto light0 = std::make_shared< math::raytracing::shape_lightsource<raytracing_config, math::geometry::sphere<vector3>> >();
		light0->center() = {0,5,0};
		light0->radius() = 0.5f*2;
		light0->power = { 100,100,100 };
		objects.push_back(light0);*/

		/*auto boundary = std::make_shared< math::raytracing::shape_primitive<raytracing_traits, math::geometry::box<vector3>> >();
		boundary->material = materials["Volume"].get();
		boundary->center() = {0,49,0};
		boundary->halfextents() = {200,50,200};*/
		//boundary->radius() = 20;
		//objects.push_back(boundary);

		for (int i = 0; i <= 0; ++i) {
			for (int j = 0; j <= 0; ++j) {
				auto lightij = std::make_shared< math::physics::raytracing::point_light<raytracing_traits> >();
				lightij->position  = /*{i*2.0f,5,j*2.0f}*/{5.0f,8.0f,0};
				lightij->color     = {1,1,1};
				lightij->intensity = 500;
				lightsources.push_back(lightij);
			}
		}
	}

	virtual 
	~APPLICATION() noexcept override {
		wex::unbindGLRC();
		wex::deleteGLRC(opengl_context);
	}

	virtual 
	void wait_message(wex::message_queue& sender) override {
		if (!it.complete()) {
			std::for_each(std::execution::par, result.begin(), result.end(), [this](spectrum& result_i) {
				size_t yi = std::distance(&result[0], &result_i);
				size_t xi = yi % result.size(0);
				yi /= result.size(0);

					if ( xi%4+(yi%4)*4 != it.subsample_index ) {
						return;
					}
					if (it.sample_index == 0) {
						result_i = {0,0,0};
					}
					vector2 Xi; {
						auto e = sequences[it.sample_index];
						auto s = scrembles[mdsize_t<2>{xi%128,yi%128}];
						Xi = {scalar(e[0]^s[0])/std::numeric_limits<unsigned>::max(),
							scalar(e[1]^s[1])/std::numeric_limits<unsigned>::max()};
					}
					scalar x = (scalar(xi) + Xi[0] - 0.5f) / scalar(result.size(0) - 1);
					scalar y = (scalar(yi) + Xi[1] - 0.5f) / scalar(result.size(1) - 1);

					auto ray = math::geometry::ray<vector3>::from_segment(
						view.transform( proj.invtransform( vector3{x,y,0}*2-1 ) ),
						view.transform( proj.invtransform( vector3{x,y,1}*2-1 ) ) );


					raytracing_traits::interact_result_type the_interact_result{ .radiance = {0,0,0}, .transmittance = {1,1,1} };
					raytracing_traits::intersect_result_type the_intersect_result;
					raytracing_traits::arguments_type the_arguments{ .random = Xi };
#if 1
					try
					{
						/*if (sampler.sample_index == 0 && xi == 250 && yi == 200) {
							std::cout << "debug";
						}*/
					//the_interact_result.push(objects.back().get(),1.0f);
					for (size_t i = 0; i != 8; ++i) {
						///intersect_surface_and_emissive
						the_intersect_result.distance  = std::numeric_limits<scalar>::max();
						the_intersect_result.primitive = nullptr;
						the_intersect_result.object    = nullptr;
						//if (!the_interact_result.empty()) {
						//	//math::geometry::ray<vector3> rayobj = math::geometry::ray<vector3>::from_ray(
						//	//	object_i.transform.invtransform(ray.start_point()),
						//	//	object_i.transform.invtransform(ray.direction(), 1));
						//	//the_interact_result.primitives.top().first->exit(ray, the_arguments, the_intersect_result);
						//	//for (const auto& object_i : this->objects) {
						//	//	//if (object_i.get() != the_interact_result.primitives.top().first) {
						//	//		//object_i->enter(ray, the_arguments, the_intersect_result);
						//	//	//}
						//	//	math::geometry::ray<vector3> rayobj = math::geometry::ray<vector3>::from_ray(
						//	//		object_i.transform.invtransform(ray.start_point()),
						//	//		object_i.transform.invtransform(ray.direction(), 1));
						//	//	for (const auto& primitive_i : *object_i.geometry) {
						//	//		if (primitive_i->enter(rayobj, the_arguments, the_intersect_result)) {
						//	//			the_intersect_result.object = &object_i;
						//	//		}
						//	//	}
						//	//}
						//	abort();
						//} else {
							/*for (const auto& object_i : this->objects) {
								math::geometry::ray<vector3> rayobj = math::geometry::ray<vector3>::from_ray(
									object_i.transform.invtransform(ray.start_point()),
									object_i.transform.without_translation.invtransform(ray.direction()));
								for (const auto& primitive_i : *object_i.geometry) {
									if (primitive_i->enter(rayobj, the_arguments, the_intersect_result)) {
										the_intersect_result.object = &object_i;
									}
								}
							}*/
							if (!the_interact_result.empty()) {
								const auto& transformation = scene.nodes[(size_t)the_interact_result.top_obj()].geom2world;
								const auto ray_in_obj = math::geometry::ray<vector3>::from_ray(
									transformation.invtransform(ray.start_point()),
									transformation.invtransform_without_translation(ray.direction()));
								the_interact_result.top()->exit(ray_in_obj, the_arguments, the_intersect_result);
								the_intersect_result.object = the_interact_result.top_obj();
							}

							size_t node = 0;
							while (node != static_cast<size_t>(-1)) {
								bool can_skip = before(the_intersect_result.distance, intersection(scene.bvh[node].first, ray));
								if (!can_skip && scene.bvh.vertex(node).is_leaf()) {
									for (const auto& primitive_i : scene.bvh[node].second) {
										const auto& transformation = scene.nodes[primitive_i.node].geom2world;
										const auto& primitive = scene.geometries[ scene.geometry_instances[ scene.nodes[primitive_i.node].index ] ][primitive_i.primitive];
										const auto ray_in_obj = math::geometry::ray<vector3>::from_ray(
											transformation.invtransform(ray.start_point()), 
											transformation.invtransform_without_translation(ray.direction()));
										if (primitive->enter(ray_in_obj, the_arguments, the_intersect_result)) {
											the_intersect_result.object = reinterpret_cast<const void*>(primitive_i.node);
										}
									}
								}
								node = can_skip ? scene.bvh.vertex(node).skip : scene.bvh.vertex(node).next;
							}
						//}
						if (the_intersect_result.primitive == nullptr) {
							/*const vector3 direction = normalize(vector3{ 0.707f, 0.707f, 0 });
							the_interact_result.radiance += the_interact_result.transmittance * max(pow(dot(direction, ray.direction()),5.0f),scalar(0));*/
							/*scalar t = 1 - exp(- max(ray.d[1], 0.0f));
							spectrum sky = (spectrum{0.2f,0.57f,1.0f} - spectrum{0.9f,0.9f,0.9f})*t + spectrum{0.9f,0.9f,0.9f};
							t = clamp(dot(vector3{0.7071f,0.7071f,0}, ray.d), 0.0f, 1.0f);
							spectrum sun = vector3{1,1,1}*pow(t, 3000.0f) + 0.8f*vector3{1.0f,0.6f,0.3f}*exp2( t*3000.0f-3000.0f ) + 0.3f*vector3{1.0f,0.5f,0.2f}*exp2( t*80.0f-80.0f );
							the_interact_result.radiance += the_interact_result.transmittance * (sky + sun);*/
							break;
						}

#if 0
						///scattering
						if (!the_interact_result.empty()) {
							scalar distance = the_interact_result.top()->march_interior(ray, the_arguments, the_intersect_result, the_interact_result);
							if (distance < the_intersect_result.distance) {
								///accumulate_lightsource for scattering
								if (lightsources.empty()) {
									the_interact_result.incident_visibility = 0;
								} else {
									the_interact_result.incident_ray.set_start_point(ray.start_point() + ray.direction() * distance);
									the_interact_result.incident_ray.set_direction(-ray.direction());
									lightsources[0]->interact_boundary(the_interact_result.incident_ray, the_arguments, the_intersect_result, the_interact_result);
									the_interact_result.incident_visibility = 1;
									for (const auto& object_i : this->objects) {
										the_interact_result.incident_visibility *= object_i->get_transmittance(the_interact_result.incident_ray, {0,the_interact_result.incident_distance - 1/128.0f})[0];
										if (the_interact_result.incident_visibility == 0) {
											break;
										}
									}
								}
								the_interact_result.top()->interact_interior(ray, the_arguments, the_intersect_result, distance, the_interact_result);
								continue;
							}
						}
#endif

						///accumulate_lightsource
						if (lightsources.empty()) {
							the_interact_result.incident_radiance = {0,0,0};
						} else {
							the_interact_result.incident_ray.set_start_point(ray.start_point() + ray.direction() * the_intersect_result.distance);
							the_interact_result.incident_ray.set_direction(-ray.direction());
							lightsources[min((size_t)(the_arguments.random[0]*lightsources.size()), lightsources.size() - 1)]
								->interact_boundary(the_interact_result.incident_ray, the_arguments, the_intersect_result, the_interact_result);
							the_interact_result.incident_radiance *= lightsources.size();
							geometry::range<scalar> incident_range = {0,the_interact_result.incident_distance - 1/128.0f};
							/*for (const auto& object_i : this->objects) {
								math::geometry::ray<vector3> incident_rayobj = math::geometry::ray<vector3>::from_ray(
									object_i.transform.invtransform(the_interact_result.incident_ray.start_point()),
									object_i.transform.without_translation.invtransform(the_interact_result.incident_ray.direction()));
								for (const auto& primitive_i : *object_i.geometry) {
									the_interact_result.incident_radiance *= primitive_i->get_transmittance(incident_rayobj, {0,the_interact_result.incident_distance - 1/128.0f})[0];
									if (the_interact_result.incident_radiance == 0) {
										break;
									}
								}
							}*/
							size_t node = 0;
							while (node != static_cast<size_t>(-1)) {
								bool can_skip = disjoint(incident_range, intersection(scene.bvh[node].first, the_interact_result.incident_ray));
								if (!can_skip && scene.bvh.vertex(node).is_leaf()) {
									for (const auto& primitive_i : scene.bvh[node].second) {
										const auto& transformation = scene.nodes[primitive_i.node].geom2world;
										const auto& primitive = scene.geometries[ scene.geometry_instances[ scene.nodes[primitive_i.node].index ] ][primitive_i.primitive];
										const auto ray_in_obj = math::geometry::ray<vector3>::from_ray(
											transformation.invtransform(the_interact_result.incident_ray.start_point()), 
											transformation.invtransform_without_translation(the_interact_result.incident_ray.direction()));
										the_interact_result.incident_radiance *= primitive->get_transmittance(ray_in_obj, incident_range)[0];
										if (the_interact_result.incident_radiance == 0) {
											break;
										}
									}
								}
								if (the_interact_result.incident_radiance == 0) {
									break;
								}
								node = can_skip ? scene.bvh.vertex(node).skip : scene.bvh.vertex(node).next;
							}
						}
						///transmission_with_emissive
						the_intersect_result.transformation = scene.nodes[reinterpret_cast<size_t>(the_intersect_result.object)].geom2world;
						const auto& transformation = scene.nodes[reinterpret_cast<size_t>(the_intersect_result.object)].geom2world;
						the_intersect_result.ray = math::geometry::ray<vector3>::from_ray(
							transformation.invtransform(ray.start_point()), 
							transformation.invtransform_without_translation(ray.direction()));
						the_intersect_result.primitive->interact_boundary(ray, the_arguments, the_intersect_result, the_interact_result);
						
						if (the_interact_result.transmittance == 0) {
							break;
						}
					}
					}
					catch (const std::exception& e)
					{
						std::cout << it.sample_index << ", ["<<xi<<","<<yi<<"], " << e.what() << std::endl;
					}
#endif
					
					result[mdsize_t<2>{xi,yi}] += the_interact_result.radiance;
				}
			);

			std::for_each(finresult.begin(), finresult.end(), [this](spectrum& finresult_i) {
				size_t yi = std::distance(&finresult[0], &finresult_i);
				const auto& result_i = result[yi];
				size_t xi = yi % result.size(0);
					yi /= result.size(0);

				size_t reference = (it.subsample_index + it.subsample_count - it.start_subsample_index)%it.subsample_count;
				size_t measure = ((xi%4+(yi%4)*4) + it.subsample_count - it.start_subsample_index)%it.subsample_count;
				if (measure <= reference) {
					finresult_i = result_i/(it.sample_index+1);
				} else {
					finresult_i = result_i/max(it.sample_index,size_t(1));
				}
				if (xi < this->prev_mouse_pos.x) {
					finresult_i /= finresult_i + 1;
				}
			});

			it.advance();

			std::wstring message = L"Raytracing ";
			message += L"i: ";
			message += std::to_wstring(it.sample_index);
			message += L", s:";
			message += std::to_wstring(it.subsample_index);
			message += L", t:";
			message += std::to_wstring( std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - sender.current_time).count() );
			message += L"[ms]";
			base::window().set_name(message.c_str());

			wex::imshow(*this, *opengl_context, *gl.CreateImage( GLimage_info{GL_TEXTURE_2D,GL_RGB32F,GLsizei(finresult.size(0)),GLsizei(finresult.size(1))},
				{ GLimage_level_data{ std::is_same_v<spectrum::package_type,float> ? GL_RGB : GL_RGBA, GL_FLOAT, finresult.data() } })
			);
		}
	}

	virtual
	void process_mouse_wheel_message(const wex::message& msg, wex::message_queue& sender) override {
		base::process_mouse_wheel_message(msg, sender);

		//if (viewspeed == 0) {
			view.translate(normalize(view.position() - pivot) * length(view.position() - pivot) * (GET_WHEEL_DELTA_WPARAM(msg.wParam) > 0 ? -1.0f : +1.0f) * 0.1f);
			//view.cen_scale((GET_WHEEL_DELTA_WPARAM(msg.wParam) > 0 ? -1.0f : +1.0f) * 0.1f, math::geometry::coordinate_traits<>{}, (float*)nullptr).correct();
			std::cout<<"at {"<<view.position()<<"}" <<" look to {"<<view.direction(2)<<"}"<<std::endl;
			it.restart();
		/*} else {
			viewspeed += viewspeed*(GET_WHEEL_DELTA_WPARAM(msg.wParam) > 0 ? -1.0f : +1.0f) * 0.1f;
			std::cout << "speed:" << viewspeed << std::endl;
		}*/
	}

	virtual
	void process_mouse_move_message(const wex::message& msg, LONG dx, LONG dy, wex::message_queue& sender) override {
		base::process_mouse_wheel_message(msg, sender);

		if (msg.wParam & MK_MBUTTON) {
			//if (viewspeed == 0) {
				if (GetKeyState(VK_SHIFT) >> 7) 
					view.translate((view.direction(0) * dx + view.direction(1) * -dy) * length(view.position() - pivot) * -0.002f);
				else {
					/*vector3 relative_position = view.position() - pivot;
					vector3 difference = relative_position - view.position();*/
					vector3 difference = -pivot;
					view.translate(difference);
					view.revolve({0,1,0}/*view.direction(1)*/, dx * 0.0174532925f * -0.2f);
					view.revolve(view.direction(0), -dy * 0.0174532925f * -0.2f);
					view.translate(-difference);
					view.fix_orthogonality();
				}
			/*} else {
				if (GetKeyState(VK_SHIFT) >> 7) {
					view.translate(-dy * viewspeed * view.get_front());
					view.translate(dx * viewspeed * view.get_right());
				} else {
					view.rotate(vector3{ 0,1,0 }, dx * 0.25f * 0.0174532925f).rotate(light.get_right(), dy * 0.25f * 0.0174532925f);
					view.r[1] = 0;
					view.correct();
				}
			}*/
			std::cout<<"at {"<<view.position()<<"}" <<" look to {"<<view.direction(2)<<"}"<<std::endl;
			it.restart();
		} else if (msg.wParam & MK_RBUTTON) {
			/*light.rotate(vector3{0,1,0}, dx * 0.25f * 0.0174532925f)
				.rotate(light.get_right(), dy * 0.25f * 0.0174532925f)
				.correct();*/
			//std::cout<<"sun look to {"<<light.f[0]<<","<<light.f[1]<<","<<light.f[2]<<"}"<<std::endl;
			//sampler.restart();
		}
	}

	virtual void process_keyboard_message(const wex::message& msg, wex::message_queue& sender) override {
		base::process_keyboard_message(msg, sender);

		if (msg.message == WM_KEYDOWN) {
			switch (msg.wParam) {
			case '1': 
				std::dynamic_pointer_cast< math::physics::raytracing::dielectric<raytracing_traits> >(materials["Ice"])->roughness += 0.1f;
				std::cout << std::dynamic_pointer_cast< math::physics::raytracing::dielectric<raytracing_traits> >(materials["Ice"])->roughness << std::endl;
				it.restart();
				break;
			case '2':
				std::dynamic_pointer_cast< math::physics::raytracing::dielectric<raytracing_traits> >(materials["Ice"])->roughness -= 0.1f;
				std::cout << std::dynamic_pointer_cast< math::physics::raytracing::dielectric<raytracing_traits> >(materials["Ice"])->roughness << std::endl;
				it.restart();
				break;
			case 'W': case 'w':
				selected->geom2world.translate({0,0,1});
				scene.build_bvh();
				it.restart();
				break;
			case 'S': case 's':
				selected->geom2world.translate({0,0,-1});
				scene.build_bvh();
				it.restart();
				break;
			case 'A': case 'a':
				selected->geom2world.translate({-1,0,0});
				scene.build_bvh();
				it.restart();
				break;
			case 'D': case 'd':
				selected->geom2world.translate({1,0,0});
				scene.build_bvh();
				it.restart();
				break;
			case 'E': case 'e':
				selected->geom2world.translate({0,1,0});
				scene.build_bvh();
				it.restart();
				break;
			case 'F': case 'f': 
				selected->geom2world.translate({0,-1,0});
				scene.build_bvh();
				it.restart();
				break;
			case 'R': case 'r':
				selected->geom2world.rotate((vector3{1,0,0}), 0.05f);
				scene.build_bvh();
				it.restart();
				break;
			case 'T': case 't':
				selected->geom2world.rotate((vector3{0,1,0}), 0.05f);
				scene.build_bvh();
				it.restart();
				break;
			case 'Y': case 'y':
				selected->geom2world.rotate((vector3{0,0,1}), 0.05f);
				scene.build_bvh();
				it.restart();
				break;
			case 'G': case 'g':
				selected->geom2world.scale({1.2f,1.2f,1.2f});
				scene.build_bvh();
				it.restart();
				break;
			case 'H': case 'h':
				selected->geom2world.scale({1/1.2f,1/1.2f,1/1.2f});
				scene.build_bvh();
				it.restart();
				break;
			default:
				break;
			}
		}
	}

	HGLRC opengl_context;
	GLlibrary2 gl;

	math::mdarray<spectrum, mdsize_t<2>> result;
	math::mdarray<spectrum, mdsize_t<2>> finresult;
	math::geometry::rigidity<vector3, math::quaternion<float,__m128>> view;
	math::geometry::perspective<vector3> proj;
	math::sampling2d_iterator it;
	std::vector<math::smdarray<unsigned,2>> sequences;
	math::mdarray<math::smdarray<unsigned,2>, mdsize_t<2>> scrembles;

	std::map<std::string, std::shared_ptr<math::physics::raytracing::material<raytracing_traits>>> materials;

	std::vector<std::shared_ptr<math::physics::raytracing::point_light<raytracing_traits>>> lightsources;

	math::physics::raytracing::scene< math::physics::raytracing::primitive<raytracing_traits> > scene;
	decltype(scene)::node_type* selected{nullptr};
	vector3 pivot = {0,0,0};
};

#include <math/integral.hpp>


int main() {
	/*auto sdasd = math::geometry::oriented_orthogonal_basis<vector3>::from_normal(normalize(vector3{-0.000128502,-0.999999,-0.00112299}));
	std::cout << length(sdasd.N) << std::endl;
	std::cout << length(sdasd.T[0]) << std::endl;
	std::cout << length(sdasd.T[1]) << std::endl;
	std::cout << sdasd.check() << std::endl;*/

//	math::geometry::oriented_orthogonal_basis<vector3> TBN = 
//	{.N={-0.00371878,-0.990213,-0.00838369},.T={vector3{1,0.00291486,-0.550033},[-0.000112102,0.979279,0.010356]}
//azimuth = 4.07386
//cos_elevation = 0.44946
//sin_elevation = 0.893301
//
//	return 0;

	//vector3 N = normalize(vector3{0.1f,1,0});
	//vector3 V = normalize(vector3{1,1,0});
	//scalar etaI = 1.0f;
	//scalar etaT = 1.3f;
	//vector3 L = math::geometry::refract(-V,N, etaI/etaT);

	//vector3 Nm = normalize(-etaI*V - etaT*L);

	//scalar Lterm = abs(dot(V,Nm)*dot(L,Nm) / (dot(V,N)/**dot(L,N)*/));
	//scalar F = reflectance(max(dot(V,Nm), scalar(0)), etaI, etaT);
	//scalar G = math::physics::ggx_g(dot(N,V),dot(Nm,V),dot(N,L),dot(Nm,L),0.2f);
	//scalar D = math::physics::ggx_d(dot(N,Nm),0.2f);
	//
	//scalar Rterm = pow(etaT,2.0f) / pow(etaI*dot(V,Nm) + etaT*dot(L,Nm),2.0f);
	//auto sss = Lterm * (1 - F)*G*D*Rterm;
	//std::cout << sss << std::endl;

	/*const auto get_tangent2 = [](vector3 n, vector3& b1, vector3& b2) {
		float sign = copysignf(1.0f, n[2]);
		const float a = -1 / (sign + n[2]);
		const float b = n[0] * n[1] * a;
		b1 = vector3{1 + sign * n[0] * n[0] * a, sign * b, -sign * n[0]};
		b2 = vector3{b, sign + n[1] * n[1] * a, -n[1]};
	};*/

	/*math::sobol_seq<unsigned> SOBEL;
	math::physics::ggx_visible_normal_distribution<vector3> distribution{ 0.004f, 0.004f };
	vector3 N = { 0,1,0 };
	vector3 T[2] = { vector3{0,0,1}, vector3{1,0,0} };
	vector3 V = normalize(vector3{-0.00328652f,0.395357f,0.918522f});
	for (size_t i = 0; i != 100; ++i) {
		vector3 Nm = distribution(SOBEL.generate<vector2>(), N, T, V);
		std::cout << N << Nm << std::endl;
	}
	return 0;*/

	//for (scalar arg = 0; arg < 6.28f; arg += 0.0174f) {
	//	vector3 N = {cos(arg),sin(arg),0};
	//	vector3 T, B;
	//	/*math::geometry::*/get_tangent2(N, T, B/*, math::geometry::coordinate_traits<>{}*/);
	//	scalar error = (pow(length(N) - 1,2) + pow(length(T) - 1,2) + pow(length(B) - 1,2) + pow(dot(N,T),2) + pow(dot(N,B),2) + pow(dot(T,B),2))/6;
	//	std::cout << N << T << B << "\t"<<error<<std::endl;
	//}

	/*for (scalar x = 1; x >= 0; x -= 0.05f) {
		std::cout << x << "\t" << math::physics::ggx_d(x, 0.2f) << std::endl;
	}


	return 0;*/
#if 0
	std::default_random_engine engine;
	std::uniform_real_distribution<scalar> distribution;

	auto vIn = normalize(vector3{/*distribution(engine),distribution(engine),distribution(engine)*/0,1,0});

	static constexpr scalar pi = static_cast<scalar>(3.1415926535897932384626433832795);
	static constexpr auto CS = math::geometry::sphconfig<>();

	///
	///                       1 - g*g
	/// PG(angle) = --------------------------------
	///              (1 + g*g - 2*g*cos(angle))^3/2
	/// 
	///                (1 - g*g)
	/// cos(angle) =  -----------^2/3 - 1 - g*g
	///                PG(angle)
	///              -----------------------------
	///                          2*g
	/// 

	const auto invf = [](scalar U, scalar g) {
		scalar gg = g*g;
		scalar temp = (1 - gg)/(1 - g + 2*g*U);
		return (1 + gg - temp*temp)/(2*g);
	};

	std::cout << math::p7quad<scalar>(0, pi, [=](scalar theta) { return
		math::p7quad<scalar>(0, pi*2, [=](scalar phi) {
			vector3 v = math::geometry::sph2cart(vector3{phi,theta,1}, CS);
			return sin(theta) * math::physics::henyey_greenstein_phase(dot(vIn,v), 0.76f) * dot(v,vector3{1,1,1});
		});
	})/12.566370614359f << std::endl;

	math::sobol_seq<unsigned> sobol;

	scalar sum = 0;
	for (size_t i = 0; i != 16000; ++i) {
		auto U = sobol.generate<vector2>();
		scalar u = distribution(engine);
		scalar cos_elevation = invf(U[0], 0.76f);
		scalar sin_elevation = std::sqrt(std::max((scalar)0, 1 - cos_elevation * cos_elevation));
		scalar azimuth = 2 * 3.1415926535897932f * U[1];

		decltype(cos(azimuth)) normal, tangent, bitangent;
			normal = cos_elevation;
			tangent = cos(azimuth) * sin_elevation;
			bitangent = sin(azimuth) * sin_elevation;

		vector3 v{bitangent,normal,tangent};
		if (abs(cos_elevation - dot(vIn, v)) >= 0.0000001f) {
			std::cout << "ERROR" << std::endl;
			std::cout << "ERROR" << std::endl;
			std::cout << "ERROR" << std::endl;
			std::cout << "ERROR" << std::endl;
		}
		sum += /*math::physics::henyey_greenstein_phase(dot(vIn, v), 0.76f) * */dot(v,vector3{1,1,1}) /*/ math::physics::henyey_greenstein_phase(cos_elevation, 0.76f)*/;
	}
	std::cout << sum/16000 << std::endl;

	return 0;
#endif

	/*std::default_random_engine engine;
	std::uniform_real_distribution<scalar> distribution;

	henyeygreenstein_distribution<vector3> smp(0.76f);

	vector3 sum = { 0,0,0 };
	for (size_t i = 0; i != 100; ++i) {
		auto dir = smp(vector2{distribution(engine),distribution(engine)}, math::geometry::sphconfig<>{});
		sum += dir;
		std::cout << dir << "," << length(dir) << std::endl;
	}
	std::cout << "AVG:" << sum / 100 << std::endl;

	return 0;*/

#if 0
	math::geometry::sphere<vector3> sph1 = { {4.3f,4.f,3.0f}, 1.0f/3 };
	math::geometry::sphere<vector3> sph2 = { {4.3f,4.f+sph1.ori.r+1.0f/7,3.0f}, 1.0f/7 };

	math::geometry::ray<vector3> ray = {
		//sph1.c,
		{4.3f,4.f+1.0f/3,3.0f},
		{0,1,0}
	};
	auto t = math::geometry::intersection(sph1, ray);
	//ray.s += ray.d * t.end();
	for (size_t i = 0; i != 10000; ++i) {
		ray.d = normalize(vector3{float(rand()),float(rand()-10000),float(rand()-10000)});
		auto t1 = math::geometry::intersection(sph1, ray);
		auto t2 = math::geometry::intersection(sph2, ray);

		///
		///		     _
		///		     /|
		///		---*-----------+
		///		                \
		///		              S  \
		///		                  \
		/// 
		/// To ensure that the media must be ejected after entering the media (even if it is wrong).
		/// We must accept negative results, but not all.
		/// We select two nearest results in two sides (positive|negative), if not then throw.
		/// We prioritize positive value, and then the nearest negative value. (because next section.)
		/// (note: may not be correct result, but the result will be acceptable.)
		/// 
		/// 
		///		-----------X--------+
		///		-----------------+   \
		///		                  \ S \
		///		     _             \   \
		///		     /|
		///		---*-----------+
		///		                \
		///		              S  \
		///		                  \
		/// 
		/// When the ray in medium"S", next intersection must eject from the medium.
		/// There are two result, one at self is "-epsilon" the other at X is "positive value".
		/// If we still prioritize positive value, then get a error result. But cannot other way better
		/// than this error result. (because next section.)
		/// 
		/// 
		///		-----------X--------+
		///		                     \
		///		                S     \
		///        _                 \
		///		     /|                 \
		///		---*-----------+         \
		///		                \         \
		///		Y----------+     \    
		///		---------+  \     \    
		///		          \S \     \
		///              \  \
		/// 
		/// If we select Y (it is nearest me), it's not just the error result, but the same situation
		/// may occur again on same boundary. So we prioritize positive value.
		/// 
		///		    /|
		///		   / |
		///		  / *
		///		 /  | \
		///		/  |   _\|
		/// 
		/// When the ray ready enter another medium at boundary.
		/// 
		/// 
		std::cout << "[" << t1.begin() << "," << t1.end() << ")\t[" << t2.begin() << "," << t2.end() << ")\n";
		if (t1.empty()) {//round-down, out of medium now.
			std::cout << "eject, error" << std::endl;
		} else if (t1.end() >= 0) {
			std::cout << "eject, t = " << t1.end() << std::endl;
		} else if (t1.end() < 0) {
			std::cout << "eject, t = " << 0 << std::endl;
		}
		/*if (t1.end() > t2.begin()) {
		} else {
			auto error = abs(t1.end() - t2.begin()/t1.end());
			if (error > std::numeric_limits<scalar>::epsilon()*4) {
			}
		}*/
	}
	return 0;
#endif

#if 0
	constexpr scalar pi = static_cast<scalar>(3.1415926535897932384626433832795);

	auto uniform_normal_probability = []<typename T>(vector3 normal) { 
		//return static_cast<T>(1.0/12.566370614359172953850573533118); 
		//return static_cast<T>(1.0/ 3.1415926535897932384626433832795);
		return 1;
	};

	constexpr auto CS = math::geometry::sphconfig<>();
	auto uniform_normal_distribution = [CS]<typename T, typename uengine>(T unused, uengine& rng) {
		T x = std::generate_canonical<T,sizeof(T)*8>(rng);
		T y = std::generate_canonical<T,sizeof(T)*8>(rng);
		//std::cout << x << "," << y << std::endl;
		return math::geometry::sph2cart(vector3{
			x*static_cast<T>(6.283185307179586476925286766559), 
			y*static_cast<T>(1.5707963267948966192313216916398),
			static_cast<T>(1)
		}, CS);
	};

	auto rng = std::default_random_engine();

	vector3 V = normalize(uniform_normal_distribution(scalar(0), rng));
	vector3 N = normalize(uniform_normal_distribution(scalar(0), rng));
	scalar roughness = 0.0f;

	std::cout << math::p7quad<scalar>(0, pi, [=](scalar theta) { return
			math::p7quad<scalar>(0, pi*2, [=](scalar phi) {
				vector3 Nm = math::geometry::sph2cart(vector3{phi,theta,1}, CS);
				vector3 L = math::geometry::reflect(-V,Nm);
				return sin(theta) * math::physics::reflectance1<scalar>(0.5f, dot(V,L))
					* math::physics::ggx_d(dot(N,Nm), roughness)
					* math::physics::ggx_g(dot(N,V), dot(V,Nm), roughness) * math::physics::ggx_g(dot(N,L), dot(L,Nm), roughness)
					//* math::physics::ggx_g(dot(N,V), dot(V,Nm), dot(N,L), dot(L,Nm), roughness)
					/ abs(4 * dot(N,V) * dot(N,L))
					;
			});
		}) << std::endl;

	math::sobol_seq< unsigned int > X;
	scalar x[2];
	scalar sum = 0;
	for (size_t i = 0; i != 1000; ++i) {
		X.generate(2, x);
		///
		/// pdf = D * abs(dot(N,Nm))
		/// 
		///           F*G*D
		/// integral ----------------------------
		///           4 * abs(dot(N,V)*dot(N,L))
		///      F*G*D
		/// sum ---------------------------- / pdf
		///      4 * abs(dot(N,V)*dot(N,L))
		/// 
		/// D*G Nm*V         F*G*D
		/// --------- * X = ----------------------------
		///  N*V             4 * abs(dot(N,V)*dot(N,L))
		///                  F
		///             X = -----------------
		///                  Nm*V * 4 * N*L

		vector3 Nm = math::physics::ggx_ndf_sample(N/*, V*/, roughness, x[0], x[1], CS);
		vector3 L = math::geometry::reflect(-V,Nm);
		sum += math::physics::reflectance1<scalar>(0.5f, dot(V,L))
			* math::physics::ggx_g(dot(N,V), dot(V,Nm), roughness) * math::physics::ggx_g(dot(N,L), dot(L,Nm), roughness)
			/ abs(4 * dot(N,V) * dot(N,L))
			/ dot(N,Nm)

		//vector3 Nm = math::physics::ggx_vndf_sample(N, V, roughness, x[0], x[1], CS);
		//vector3 L = math::geometry::reflect(-V,Nm);
		//sum += math::physics::reflectance1<scalar>(0.5f, dot(V,L))
		//	* math::physics::ggx_g(dot(N,L), dot(L,Nm), roughness)
		//	//* math::physics::ggx_g(dot(N,V), dot(V,Nm), dot(N,L), dot(L,Nm), roughness)
		//	/ abs(4 * dot(N,L) * dot(Nm,V))
			;
	}
	std::cout << sum/1000 << std::endl;
#endif

#if 0
	for (float theta = 0; theta < 1.5707963267948966; theta += 0.01f) {
		{
			/// Test surface as solid.
			/// 
			///         +- -- -- -- -- -+
			///         |               |
			///         |               +- -- -+
			/// --------|---------------|-->   |
			///         |               +- -- -+
			///         |               |
			///         +- -- -- -- -- -+
			/// 
			float T;
			float R = reflectance(cos(theta), 1.5f, 1.5f, &T);
			std::cout << std::format("{0} + {1} = {2}", R, T, R+T);
		}

		{
			///         +- -- -- -- -- -+
			///         |               |
			///         |             +-|-- -+
			/// --------|-------------|-|--> |
			///         |             +-|-- -+
			///         |               |
			///         +- -- -- -- -- -+
			/// 
			float T1;
			float R1 = reflectance(cos(theta), 1.5f, 1.0f, &T1);
			float T2;
			float R2 = reflectance(cos(theta), 1.0f, 1.5f, &T2);
			std::cout << std::format("\t{0} + {1}*{2} + {1}*{3} = {4}", R1, T1, R2, T2, R1+T1*R2+T1*T2);
		}
		std::cout << std::endl;
	}
	return 0;
#endif
	wex::message_queue app(GetModuleHandle(nullptr));
	app.create_window_and_userdata<APPLICATION>(L"Raytracing", WS_OVERLAPPEDWINDOW, 0, 0, 1024, 1024)->show();
	return app.run();
}