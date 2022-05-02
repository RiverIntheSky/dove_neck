#include "feather.h"
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/timer.h>
#include <fstream>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <unordered_map>
#include <unordered_set>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _shape-feather:

Feather (:monosp:`feather`)
----------------------------------------------------

.. pluginparameters::


 * - filename
   - |string|
   - The .ply file that stores the feather geometry
 * - radius
   - |float|
   - Radius of the barbule in object-space units (Default: 0.04)
 * - scale
   - |float|
   - the portion of the underlying cylinder
 * - span
   - |float|
   - the angle between the barbule y axis and the barb axis
 * - flip_normals
   - |bool|
   -  Is the barbule inverted, i.e. should the normal vectors
      be flipped? (Default: |false|, i.e. the normals point outside)
 * - to_world
   - |transform|
   - Specifies an optional linear object-to-world transformation. Note that non-uniform scales are
   not permitted! (Default: none, i.e. object space = world space)

   This shape plugin describes a feather shape with barbule primitives.
   Note that the feather does not have endcaps -- also,
   its normals point outward, which means that the inside will be treated
   as fully absorbing by most material models. If this is not
   desirable, consider using the :ref:`twosided <bsdf-twosided>` plugin.

   A simple example for instantiating a barbule, whose interior is visible:

   .. code-block:: xml

   <shape type="feather">
   <string name="filename" value="feathers/barbule_straight.ply"/>
   <float name="radius" value="0.02"/>
   <bsdf type="irid">
   <string name="int_ior" value="air"/>
   <string name="film_ior" value="keratin"/>
   <string name="ext_ior" value="air"/>
   <float name="height" value="595.0"/>
   <string name="distribution" value="ggx"/>
   <float name="alpha_u" value="0.01"/>
   <float name="alpha_v" value="0.01"/>
   </bsdf>
   <ref id="melanin" name="interior"/>
   <transform name="to_world">
   <rotate y="1" angle="0"/>
   </transform>
   </shape>
*/


MTS_VARIANT Feather<Float, Spectrum>::Feather(const Properties &props):
    Shape<Float, Spectrum>(props) {
    m_kdtree = new ShapeKDTree(props);
    float radius = props.float_("radius", 0.5f);
    m_cover = props.float_("cover", 0.78f);
    m_center = props.bool_("center", false);
    m_span = props.float_("span", 0.45f);
    m_roughness = props.vector3f("roughness", 0.f);

    // TODO: sample in houdini and solve NaN
    auto pmgr = PluginManager::instance();
    Properties props_sampler("independent");
    props_sampler.set_int("sample_count", 4);
    m_sampler = static_cast<Sampler *>(pmgr->create_object<Sampler>(props_sampler));

    // update radius
    auto [S, Q, T] = transform_decompose(m_to_world.matrix, 25);
    radius *= S[0][0];

    auto fs = Thread::thread()->file_resolver();
    fs::path file_path = fs->resolve(props.string("filename"));

    std::string file_name = file_path.filename().string();

    auto fail = [&](const char *descr) {
	Throw("Error while loading PLY file \"%s\": %s!", file_name, descr);
    };

    Log(Info, "Loading feather from \"%s\" ..", file_name);
    if (!fs::exists(file_path))
	fail("file not found");

    ref<Stream> stream = new FileStream(file_path);
    Timer timer;

    std::vector<PLYElement> elements;
    parse_ply_header(stream, elements);
    stream = parse_ascii((FileStream *) stream.get(), elements);

    bool has_vertex_texcoords = false;

    ref<Struct> vertex_struct = new Struct();
    ref<Struct> line_struct = new Struct();
    constexpr size_t elements_per_packet = 1024;

    for (auto &el : elements) {
	if (el.name == "vertex") {
	    for (auto name : { "x", "y", "z" })
		vertex_struct->append(name, struct_type_v<InputFloat>);
	    if (el.struct_->has_field("H") && el.struct_->has_field("mu") && el.struct_->has_field("height")) {
		for (auto name : { "H", "mu", "height"})
		    vertex_struct->append(name, struct_type_v<InputFloat>);
		has_vertex_texcoords = true;
	    }
	    for (auto name : { "out1", "out2", "out3" })
		vertex_struct->append(name, struct_type_v<InputFloat>);

	    size_t i_struct_size = el.struct_->size();
	    size_t o_struct_size = vertex_struct->size();

	    ref<StructConverter> conv;
	    try {
		conv = new StructConverter(el.struct_, vertex_struct);
	    } catch (const std::exception &e) {
		fail(e.what());
	    }

	    m_vertex_count = (ScalarSize) el.count;
	    m_vertex_positions_buf = empty<FloatStorage>(m_vertex_count * 3);
	    if (has_vertex_texcoords)
		m_vertex_texcoords_buf = empty<FloatStorage>(m_vertex_count * 3);
	    m_vertex_normals_buf = empty<FloatStorage>(m_vertex_count * 3);

	    m_vertex_positions_buf.managed();
	    m_vertex_texcoords_buf.managed();
	    m_vertex_normals_buf.managed();

	    if constexpr (is_cuda_array_v<Float>)
			     cuda_sync();

	    size_t packet_count     = el.count / elements_per_packet;
	    size_t remainder_count  = el.count % elements_per_packet;
	    size_t i_packet_size    = i_struct_size * elements_per_packet;
	    size_t i_remainder_size = i_struct_size * remainder_count;
	    size_t o_packet_size    = o_struct_size * elements_per_packet;

	    std::unique_ptr<uint8_t[]> buf(new uint8_t[i_packet_size]);
	    std::unique_ptr<uint8_t[]> buf_o(new uint8_t[o_packet_size]);

	    InputFloat* position_ptr = m_vertex_positions_buf.data();
	    InputFloat* texcoord_ptr = m_vertex_texcoords_buf.data();
	    InputFloat* normal_ptr   = m_vertex_normals_buf.data();

	    for (size_t i = 0; i <= packet_count; ++i) {
		uint8_t *target = (uint8_t *) buf_o.get();
		size_t psize = (i != packet_count) ? i_packet_size : i_remainder_size;
		size_t count = (i != packet_count) ? elements_per_packet : remainder_count;
		stream->read(buf.get(), psize);
		if (unlikely(!conv->convert(count, buf.get(), buf_o.get())))
		    fail("incompatible contents -- is line segment number set correctly?");

		for (size_t j = 0; j < count; ++j) {
		    InputPoint3f p = enoki::load<InputPoint3f>(target);
		    p = m_to_world * p;
		    if (unlikely(!all(enoki::isfinite(p))))
			fail("mesh contains invalid vertex positions/normal data");
		    // m_bbox.expand(p);
		    store_unaligned(position_ptr, p);
		    position_ptr += 3;

		    if (has_vertex_texcoords) {
			InputVector3f uvw = enoki::load<InputVector3f>(target + sizeof(InputFloat) * 3);
			store_unaligned(texcoord_ptr, uvw);
			texcoord_ptr += 3;
		    }

		    InputNormal3f n = enoki::load<InputNormal3f>(target + sizeof(InputFloat) * (has_vertex_texcoords? 6 : 3));
		    n = normalize(m_to_world.transform_affine(n));
		    store_unaligned(normal_ptr, n);
		    normal_ptr += 3;

		    target += o_struct_size;
		}
	    }
	} else if (el.name == "face") {
	    std::string field_name;
	    if (el.struct_->has_field("vertex_index.count"))
		field_name = "vertex_index";
	    else if (el.struct_->has_field("vertex_indices.count"))
		field_name = "vertex_indices";
	    else
		fail("vertex_index/vertex_indices property not found");

	    for (size_t i = 0; i < SEGNUM; ++i)
		line_struct->append(tfm::format("i%i", i), struct_type_v<ScalarIndex>);

	    size_t i_struct_size = el.struct_->size();
	    size_t o_struct_size = line_struct->size();

	    ref<StructConverter> conv;
	    try {
		conv = new StructConverter(el.struct_, line_struct);
	    } catch (const std::exception &e) {
		fail(e.what());
	    }

	    m_line_count = (ScalarSize) el.count;
	    m_lines_buf = empty<DynamicBuffer<UInt32>>(m_line_count * SEGNUM);
	    m_lines_buf.managed();

	    ScalarIndex* line_ptr = m_lines_buf.data();

	    size_t packet_count     = el.count / elements_per_packet;
	    size_t remainder_count  = el.count % elements_per_packet;
	    size_t i_packet_size    = i_struct_size * elements_per_packet;
	    size_t i_remainder_size = i_struct_size * remainder_count;
	    size_t o_packet_size    = o_struct_size * elements_per_packet;

	    std::unique_ptr<uint8_t[]> buf(new uint8_t[i_packet_size]);
	    std::unique_ptr<uint8_t[]> buf_o(static_cast<uint8_t*>(std::aligned_alloc(alignof(ScalarIndexN), o_packet_size)));

	    for (size_t i = 0; i <= packet_count; ++i) {
		uint8_t *target = (uint8_t *) buf_o.get();
		size_t psize = (i != packet_count) ? i_packet_size : i_remainder_size;
		size_t count = (i != packet_count) ? elements_per_packet : remainder_count;

		stream->read(buf.get(), psize);
		if (unlikely(!conv->convert(count, buf.get(), buf_o.get())))
		    fail("incompatible contents -- is line segment number set correctly?");
		for (size_t j = 0; j < count; ++j) {
		    ScalarIndexN line = enoki::load<ScalarIndexN>(target);
		    store_unaligned(line_ptr, line);
		    line_ptr += SEGNUM;
		    target += o_struct_size;
		}
	    }
	} else {
	    Log(Warn, "\"%s\": Skipping unknown element \"%s\"", file_name, el.name);
	    stream->seek(stream->tell() + el.struct_->size() * el.count);
	}
    }

    Log(Debug, "\"%s\": read %i lines, %i vertices (%s in %s)",
	file_name, m_line_count, m_vertex_count,
	util::mem_string(m_line_count * line_struct->size() +
			 m_vertex_count * vertex_struct->size()),
	util::time_string(timer.value())
	);

    for (ScalarSize i = 0; i < m_line_count; ++i) {
	auto line = line_indices(i);
	for (ScalarSize j = 0; j < SEGNUM; ++j)
	    assert(line[j] < m_vertex_count);
	for (ScalarSize j = 0; j < SEGNUM - 1; ++j) {
	    ScalarNormal3f n = vertex_normal(line[j]);
	    ScalarVector3f t1, t2;
	    ScalarVector3f uvw(0.f);
	    if (has_vertex_texcoords) {
		uvw = vertex_texcoord(line[j]);
	    }
	    Mask active = true;
	    Vector3f pertubation((*m_sampler).next_1d(active),
				 (*m_sampler).next_1d(active),
				 (*m_sampler).next_1d(active));
	    pertubation = pertubation * 2.f - 1.f;
	    pertubation *= m_roughness;
	    if (j == 0) {
		t1 = normalize(vertex_position(line[j + 1]) - vertex_position(line[j]));
	    } else {
		t1 = normalize(vertex_position(line[j + 1]) - vertex_position(line[j - 1]));
	    }
	    if (j == SEGNUM - 2) {
		t2 = normalize(vertex_position(line[j + 1]) - vertex_position(line[j]));
	    } else {
		t2 = normalize(vertex_position(line[j + 2]) - vertex_position(line[j]));
	    }
	    ref<Barb<Float, Spectrum>> barb;
	    // barb is thinner at the end
	    float curve_x = (j + 0.5) / SEGNUM;
	    float scale = 1.f;
	    if (curve_x > 0.6f)
		scale -= 5.f * sqr(curve_x - 0.6f);

	    if (m_center) {
		barb = new Barb<Float, Spectrum>(vertex_position(line[j + 1]), vertex_position(line[j]),
						 n, t2, t1, uvw, pertubation, radius, m_cover * scale, m_span, props);

	    } else {
		barb = new Barb<Float, Spectrum>(vertex_position(line[j + 1]) - n * radius, vertex_position(line[j]) - n * radius,
						 n, t2, t1, uvw, pertubation, radius, m_cover * scale, m_span, props);
 	    }
	    m_kdtree->add_shape(barb);
	}
    }

    if (!m_kdtree->ready())
	m_kdtree->build();

    m_bbox = m_kdtree->bbox();
}

MTS_VARIANT void Feather<Float, Spectrum>::parse_ply_header(Stream *stream,
							    std::vector<PLYElement> &elements) {
    std::unordered_map<std::string, Struct::Type> fmt_map;
    fmt_map["int"]    = Struct::Type::Int32;
    fmt_map["float"]  = Struct::Type::Float32;
    fmt_map["uchar"]  = Struct::Type::UInt8;

    Struct::ByteOrder byte_order = Struct::host_byte_order();
    ref<Struct> struct_;

    while (true) {
	std::string line = stream->read_line();
	std::istringstream iss(line);
	std::string token;
	if (!(iss >> token))
	    continue;

	if (token == "element") {
	    iss >> token;
	    elements.emplace_back();
	    auto &element = elements.back();
	    element.name = token;
	    iss >> token;
	    element.count = (size_t) stoull(token);
	    struct_ = element.struct_ = new Struct(true, byte_order);
	} else if (token == "property") {
	    iss >> token;
	    if (token == "list") {
		iss >> token;
		auto it1 = fmt_map.find(token);
		if (it1 == fmt_map.end())
		    Throw("invalid PLY header: unknown format type \"%s\"", token);
		iss >> token;
		auto it2 = fmt_map.find(token);
		if (it2 == fmt_map.end())
		    Throw("invalid PLY header: unknown format type \"%s\"", token);
		iss >> token;
		struct_->append(token + ".count", it1->second, +Struct::Flags::Assert, SEGNUM);
		for (int i = 0; i < SEGNUM; ++i)
		    struct_->append(tfm::format("i%i", i), it2->second);
	    } else {
		auto it = fmt_map.find(token);
		if (it == fmt_map.end())
		    Throw("invalid PLY header: unknown format type \"%s\"", token);
		iss >> token;
		uint32_t flags = +Struct::Flags::None;
		if (it->second >= Struct::Type::Int8 &&
		    it->second <= Struct::Type::UInt64)
		    flags = Struct::Flags::Normalized | Struct::Flags::Gamma;
		struct_->append(token, it->second, flags);
	    }

	} else if (token == "end_header") {
	    break;
	}
    }
}

MTS_VARIANT ref<Stream> Feather<Float, Spectrum>::parse_ascii(FileStream *in,
							      const std::vector<PLYElement> &elements) {
    ref<Stream> out = new MemoryStream();
    std::fstream &is = *in->native();

    for (auto const &el : elements) {
	for (size_t i = 0; i < el.count; ++i) {
	    for (auto const &field : *(el.struct_)) {
		switch (field.type) {
		case Struct::Type::UInt8: {
		    int value;
		    if (!(is >> value))
			Throw("Could not parse \"uchar\" value for field %s (may be due to non-triangular faces)", field.name);
		    if (value != SEGNUM)
			Throw("Expected segment number: %s, received segment number: %s = %s", SEGNUM, field.name, value);
		    out->write((uint8_t) value);
		}
		    break;

		case Struct::Type::Int32: {
		    int32_t value;
		    if (!(is >> value)) Throw("Could not parse \"int\" value for field %s", field.name);
		    out->write(value);
		}
		    break;

		case Struct::Type::Float32: {
		    float value;
		    if (!(is >> value)) Throw("Could not parse \"float\" value for field %s", field.name);
		    out->write(value);
		}
		    break;
		default:
		    Throw("internal error");
		}
	    }
	}
    }
    out->seek(0);
    return out;
}

MTS_VARIANT Feather<Float, Spectrum>::~Feather() {
    // #if defined(MTS_ENABLE_EMBREE)
    //     if constexpr (!is_cuda_array_v<Float>)
    //         rtcReleaseScene(m_embree_scene);
    // #endif
}

MTS_VARIANT typename Feather<Float, Spectrum>::PreliminaryIntersection3f
Feather<Float, Spectrum>::ray_intersect_preliminary(const Ray3f &ray_,
						    Mask active) const {
    return m_kdtree->template ray_intersect_preliminary<false>(ray_, active);
}

MTS_VARIANT typename Feather<Float, Spectrum>::Mask
Feather<Float, Spectrum>::ray_test(const Ray3f &ray_,
				   Mask active) const {
    return m_kdtree->template ray_intersect_preliminary<true>(ray_, active).is_valid();
}

MTS_VARIANT typename Feather<Float, Spectrum>::SurfaceInteraction3f
Feather<Float, Spectrum>::compute_surface_interaction(const Ray3f &ray,
						      PreliminaryIntersection3f pi,
						      HitComputeFlags flags,
						      Mask active) const {
    MTS_MASK_ARGUMENT(active);
    // #if defined(MTS_ENABLE_EMBREE)
    //     if constexpr (!is_cuda_array_v<Float>) {
    //         if constexpr (!is_array_v<Float>) {
    //             Assert(pi.shape_index < m_shapes.size());
    //             pi.shape = m_shapes[pi.shape_index];
    //         } else {
    //             using ShapePtr = replace_scalar_t<Float, const Base *>;
    //             Assert(all(pi.shape_index < m_shapes.size()));
    //             pi.shape = gather<ShapePtr>(m_shapes.data(), pi.shape_index, active);
    //         }

    //         SurfaceInteraction3f si = pi.shape->compute_surface_interaction(ray, pi, flags, active);
    //         si.shape = pi.shape;

    //         return si;
    //     }
    // #endif

    return pi.shape->compute_surface_interaction(ray, pi, flags, active);
}

MTS_VARIANT typename Feather<Float, Spectrum>::ScalarFloat
Feather<Float, Spectrum>::surface_area() const {
    NotImplementedError("surface_area");
}

MTS_VARIANT typename Feather<Float, Spectrum>::ScalarSize
Feather<Float, Spectrum>::effective_primitive_count() const {
#if !defined(MTS_ENABLE_EMBREE)
    if constexpr (!is_cuda_array_v<Float>)
		     return m_kdtree->primitive_count();
#endif

// #if defined(MTS_ENABLE_EMBREE) || defined(MTS_ENABLE_OPTIX)
//     ScalarSize count = 0;
//     for (auto shape : m_shapes)
//         count += shape->primitive_count();

//     return count;
// #endif
}

MTS_VARIANT std::string Feather<Float, Spectrum>::to_string() const {
    std::ostringstream oss;
    oss << "Feather[" << std::endl
        // << "   numVertices = " << m_kdtree->get_vertexCount() << ","
        // << "   numSegments = " << m_kdtree->getSegmentCount() << ","
        // << "   numBarbs = " << m_kdtree->getBarbCount() << ","
        // << "   radius = " << m_kdtree->getRadius()
        << "]";
    return oss.str();
}

MTS_IMPLEMENT_CLASS_VARIANT(Feather, Shape)
MTS_EXPORT_PLUGIN(Feather, "Feather shape");
NAMESPACE_END(mitsuba)
