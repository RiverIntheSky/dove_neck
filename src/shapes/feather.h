#pragma once

#include "barb.h"
#include <mitsuba/render/shape.h>
#include <mitsuba/render/kdtree.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/mstream.h>
#include <mitsuba/core/distr_1d.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/sampler.h>

#if defined(MTS_ENABLE_EMBREE)
    #include <embree3/rtcore.h>
#endif
#if defined(MTS_ENABLE_OPTIX)
    #include <mitsuba/render/optix/shapes.h>
#endif

#define SEGNUM 64

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class Feather final: public Shape<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Shape, m_to_world, m_id)
    MTS_IMPORT_TYPES(ShapeKDTree, Sampler)

    using typename Base::ScalarIndex;
    using ScalarIndexN = Array<ScalarIndex, SEGNUM>;
    using typename Base::ScalarSize;
    // Mesh is always stored in single precision
    using InputFloat = float;
    using InputPoint3f  = Point<InputFloat, 3>;
    using InputVector3f  = Vector<InputFloat, 3>;
    using InputNormal3f = Normal<InputFloat, 3>;
    using FloatStorage = DynamicBuffer<replace_scalar_t<Float, InputFloat>>;

    Feather(const Properties &props);
    ~Feather();

    // =============================================================
    //! @{ \name Access the internal vertex data
    // =============================================================

    /// Return the list of vertices underlying the barb shape
    // const std::vector<Point> &getVertices() const;

    // /**
    //  * Return a boolean list specifying whether a vertex
    //  * marks the beginning of a new barbule
    //  */
    // const std::vector<bool> &getStartBarbule() const;

    //! @}
    // =============================================================


    // =========================================================================
    //! @{ \name Accessors (vertices, faces, normals, etc)
    // =========================================================================
    /// Return vertex positions buffer
    FloatStorage& vertex_positions_buffer() { return m_vertex_positions_buf; }
    /// Const variant of \ref vertex_positions_buffer.
    const FloatStorage& vertex_positions_buffer() const { return m_vertex_positions_buf; }

    /// Return vertex normals buffer
    FloatStorage& vertex_normals_buffer() { return m_vertex_normals_buf; }
    /// Const variant of \ref vertex_normals_buffer.
    const FloatStorage& vertex_normals_buffer() const { return m_vertex_normals_buf; }

    /// Return vertex texcoords buffer
    FloatStorage& vertex_texcoords_buffer() { return m_vertex_texcoords_buf; }
    /// Const variant of \ref vertex_texcoords_buffer.
    const FloatStorage& vertex_texcoords_buffer() const { return m_vertex_texcoords_buf; }

    /// Return line indices buffer
    DynamicBuffer<UInt32>& lines_buffer() { return m_lines_buf; }
    /// Const variant of \ref lines_buffer.
    const DynamicBuffer<UInt32>& lines_buffer() const { return m_lines_buf; }

    /// Returns the line indices associated with triangle \c index
    template <typename Index>
    MTS_INLINE auto line_indices(Index index, mask_t<Index> active = true) const {
        using Result = Array<replace_scalar_t<Index, uint32_t>, SEGNUM>;
        return gather<Result>(m_lines_buf, index, active);
    }

    /// Returns the world-space position of the vertex with index \c index
    template <typename Index>
    MTS_INLINE auto vertex_position(Index index, mask_t<Index> active = true) const {
        using Result = Point<replace_scalar_t<Index, InputFloat>, 3>;
        return gather<Result>(m_vertex_positions_buf, index, active);
    }

    /// Returns the normal direction of the vertex with index \c index
    template <typename Index>
    MTS_INLINE auto vertex_normal(Index index, mask_t<Index> active = true) const {
        using Result = Normal<replace_scalar_t<Index, InputFloat>, 3>;
        return gather<Result>(m_vertex_normals_buf, index, active);
    }

    /// Returns the UV texture coordinates of the vertex with index \c index
    template <typename Index>
    MTS_INLINE auto vertex_texcoord(Index index, mask_t<Index> active = true) const {
        using Result = Point<replace_scalar_t<Index, InputFloat>, 3>;
        return gather<Result>(m_vertex_texcoords_buf, index, active);
    }

    // =============================================================
    //! @{ \name Implementation of the \ref Shape interface
    // =============================================================
    PreliminaryIntersection3f ray_intersect_preliminary(const Ray3f &ray_,
                                                        Mask active) const override;

    Mask ray_test(const Ray3f &ray_, Mask active) const override;
    SurfaceInteraction3f compute_surface_interaction(const Ray3f &ray,
                                                     PreliminaryIntersection3f pi,
                                                     HitComputeFlags flags,
                                                     Mask active) const override;

    // const KDTreeBase<AABB> *getKDTree() const;

    ScalarBoundingBox3f bbox() const override { return m_bbox; }

    ScalarFloat surface_area() const override;

    ScalarSize primitive_count() const override {return 1; }

    MTS_INLINE ScalarSize effective_primitive_count() const override;

    //! @}
    // =============================================================

    /// Return a human-readable representation
    std::string to_string() const override;

    MTS_DECLARE_CLASS()
private:
    ref<Sampler> m_sampler;
    bool m_center;
    float m_cover, m_span;
    Vector3f m_roughness;
    ScalarSize m_vertex_count = 0;
    ScalarSize m_line_count = 0;
    FloatStorage m_vertex_positions_buf;
    FloatStorage m_vertex_normals_buf;
    FloatStorage m_vertex_texcoords_buf;
    DynamicBuffer<UInt32> m_lines_buf;

    ScalarBoundingBox3f m_bbox;

    ref<ShapeKDTree> m_kdtree;
    struct PLYElement {
        std::string name;
        size_t count;
        ref<Struct> struct_;
    };
    void parse_ply_header(Stream *stream, std::vector<PLYElement> &elements);
    ref<Stream> parse_ascii(FileStream *in, const std::vector<PLYElement> &elements);
};

NAMESPACE_END(mitsuba)
