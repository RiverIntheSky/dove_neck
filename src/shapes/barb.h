#pragma once

#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>

// static float PHI_MID = 0.756045006640150f;
// static float PHI_HALF_SPAN = 0.550857537253203f;
static float PHI_MID = 0.f;
static float PHI_HALF_SPAN = 0.3f;

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class Barb final : public Shape<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Shape, m_to_world, m_to_object, set_children,
                    get_children_string, parameters_grad_enabled)
    MTS_IMPORT_TYPES()

    using typename Base::ScalarIndex;
    using typename Base::ScalarSize;
    using Double = std::conditional_t<is_cuda_array_v<Float>, Float, Float64>;
    static constexpr auto Pi        = math::Pi<Float>;
    static constexpr auto HalfPi    = math::HalfPi<Float>;

    Barb(const Properties &props) : Base(props) {
	/// Are the sphere normals pointing inwards? default: no
	m_flip_normals = props.bool_("flip_normals", false);
	m_cover = props.float_("cover", 0.78f);
	m_span = props.float_("span", 0.45f);

	// Update the to_world transform if face points and radius are also provided
	// default unit micrometer
	float radius = props.float_("radius", 0.2f);
	ScalarPoint3f e0 = props.point3f("e0", ScalarPoint3f(0.f, 0.f, -0.5f)),
	    e1 = props.point3f("e1", ScalarPoint3f(0.f, 0.f, 0.5f));

	m_t1 = normalize(e0 - e1);
	m_t2 = normalize(e0 - e1);

	ScalarFrame3f rot(e1 - e0);
	rot.t = ScalarNormal3f(0.f, 1.f, 0.f);
	rot.s = normalize(cross(rot.t, rot.n));

	m_to_world = m_to_world *
	    ScalarTransform4f::translate(e0) *
	    ScalarTransform4f::to_frame(rot) *
	    ScalarTransform4f::scale(ScalarVector3f(radius, radius, 1.f));

	update();
	set_children();
    }

    Barb(ScalarPoint3f e0, ScalarPoint3f e1, ScalarNormal3f n, ScalarVector3f t1, ScalarVector3f t2,
	 ScalarVector3f uvw, float radius, float cover, float span, const Properties &props):
	Base(props), m_t1(t1), m_t2(t2), m_uvw(uvw), m_cover(cover), m_span(span) {
	/// Are the sphere normals pointing inwards? default: no
	m_flip_normals = props.bool_("flip_normals", false);

	ScalarFrame3f rot(e1 - e0);
	rot.s = normalize(cross(n, rot.n));
	rot.t = normalize(cross(rot.n, rot.s));

	m_to_world = ScalarTransform4f::translate(e0) *
	    ScalarTransform4f::to_frame(rot) *
	    ScalarTransform4f::scale(ScalarVector3f(radius, radius, 1.f));

	update();
	set_children();
    }

    void update() {
	// Extract center and radius from to_world matrix (25 iterations for numerical accuracy)
	auto [S, Q, T] = transform_decompose(m_to_world.matrix, 25);

	if (abs(S[0][1]) > 1e-6f || abs(S[0][2]) > 1e-6f || abs(S[1][0]) > 1e-6f ||
	    abs(S[1][2]) > 1e-6f || abs(S[2][0]) > 1e-6f || abs(S[2][1]) > 1e-6f)
	    Log(Warn, "'to_world' transform shouldn't contain any shearing!");

	if (!(abs(S[0][0] - S[1][1]) < 1e-6f))
	    Log(Warn, "'to_world' transform shouldn't contain non-uniform scaling along the X and Y axes!");
	m_length = S[2][2];

	// circle
	m_radius = S[0][0];
	m_theta_a_c = PHI_MID - PHI_HALF_SPAN;
	m_theta_e_c = PHI_MID + PHI_HALF_SPAN;

	// Reconstruct the to_world transform with uniform scaling and no shear
	m_to_world = transform_compose(ScalarMatrix3f(1.f), Q, T);
	m_to_object = m_to_world.inverse();

	m_t1 = m_to_object * m_t1;
	m_t2 = m_to_object * m_t2;

	m_span_y = m_radius * cot(m_span);
	m_span_x = m_radius * tan(m_span);

	m_y_min = cos(m_cover);
	m_y_max = cos(0.1);
    }

    ScalarBoundingBox3f bbox() const override {
	ScalarBoundingBox3f bbox;
	float x_m = sin(m_cover);
	ScalarVector3f x1 = m_to_world * ScalarVector3f(x_m, 1.f, 0.f) * m_radius,
	    x2 = m_to_world * ScalarVector3f(-x_m, m_y_min, 0.f) * m_radius;

	ScalarPoint3f e0 = m_to_world * ScalarPoint3f(0.f, 0.f, 0.f),
	    e1 = m_to_world * ScalarPoint3f(0.f, 0.f, m_length);

	ScalarVector3f t1 = m_to_world * m_t1;
	ScalarVector3f t2 = m_to_world * m_t2;

	assert(dot(e0 - e1, t1) > 0.f);
	ScalarVector3f proj1 = (e1 - e0) / dot(e0 - e1, t1);
	bbox.expand(ScalarPoint3f(e0 + x1 + dot(x1, t1) * proj1));
	bbox.expand(ScalarPoint3f(e0 + x2 + dot(x2, t1) * proj1));

	assert(dot(e0 - e1, t2) > 0.f);
	ScalarVector3f proj2 = (e1 - e0) / dot(e0 - e1, t2);
	bbox.expand(ScalarPoint3f(e1 + x1 + dot(x1, t2) * proj2));
	bbox.expand(ScalarPoint3f(e1 + x2 + dot(x2, t2) * proj2));
	return bbox;
    }

    // =============================================================
    //! @{ \name Ray tracing routines
    // =============================================================

    PreliminaryIntersection3f ray_intersect_preliminary(const Ray3f &ray_,
                                                        Mask active) const override {
	MTS_MASK_ARGUMENT(active);

	PreliminaryIntersection3f pi = zero<PreliminaryIntersection3f>();
	Ray3f ray = m_to_object.transform_affine(ray_);

	Double mint = Double(ray.mint),
	    maxt = Double(ray.maxt);

	Double ox = Double(ray.o.x()),
	    oy = Double(ray.o.y()),
	    oz = Double(ray.o.z()),
	    dx = Double(ray.d.x()),
	    dy = Double(ray.d.y()),
	    dz = Double(ray.d.z());

	scalar_t<Double> radius = scalar_t<Double>(m_radius),
	    length = scalar_t<Double>(m_length);

	ScalarVector3d t1 = ScalarVector3d(m_t1);
	ScalarVector3d t2 = ScalarVector3d(m_t2);

	Float tc = math::Infinity<Float>;

	Double A = sqr(dx) + sqr(dy),
	    B = scalar_t<Double>(2.f) * (dx * ox + dy * oy),
	    C = sqr(ox) + sqr(oy) - sqr(radius);

	auto [solution_found, near_t, far_t] =
	    math::solve_quadratic(A, B, C);

	// Barb doesn't intersect with the segment on the ray
	Mask out_bounds = !(near_t <= maxt && far_t >= mint); // NaN-aware conditionals

	Double z_pos_near = oz + dz * near_t,
	    z_pos_far  = oz + dz * far_t,
	    y_pos_near = oy + dy * near_t,
	    y_pos_far  = oy + dy * far_t,
	    x_pos_near = ox + dx * near_t,
	    x_pos_far  = ox + dx * far_t;

	// The cylinder containing the barb fully contains the segment of the ray
	Mask in_bounds = near_t < mint && far_t > maxt;

	Mask near_intersect = y_pos_near >= radius * Double(m_y_min) && y_pos_near <= radius * Double(m_y_max) && near_t >= mint
	    && t1.x() * x_pos_near + t1.y() * y_pos_near + t1.z() * z_pos_near <= 0
	    && t2.x() * x_pos_near + t2.y() * y_pos_near + t2.z() * (z_pos_near - length) >= 0;

	Mask far_intersect = y_pos_far >= radius * Double(m_y_min) && y_pos_far <= radius * Double(m_y_max)&& far_t <= maxt
	    && t1.x() * x_pos_far + t1.y() * y_pos_far + t1.z() * z_pos_far <= 0
	    && t2.x() * x_pos_far + t2.y() * y_pos_far + t2.z() * (z_pos_far - length) >= 0;

	Mask active_c = solution_found && !out_bounds && !in_bounds && (near_intersect || far_intersect);

	tc = select(active & active_c, select(near_intersect, Float(near_t), Float(far_t)),
		    math::Infinity<Float>);

	pi.t = tc;

	pi.shape = this;

	return pi;
    }

    Mask ray_test(const Ray3f &ray_, Mask active) const override {
	MTS_MASK_ARGUMENT(active);

	Ray3f ray = m_to_object.transform_affine(ray_);
	Double mint = Double(ray.mint),
	    maxt = Double(ray.maxt);

	Double ox = Double(ray.o.x()),
	    oy = Double(ray.o.y()),
	    oz = Double(ray.o.z()),
	    dx = Double(ray.d.x()),
	    dy = Double(ray.d.y()),
	    dz = Double(ray.d.z());

	scalar_t<Double> length = scalar_t<Double>(m_length);

	ScalarVector3d t1 = ScalarVector3d(m_t1);
	ScalarVector3d t2 = ScalarVector3d(m_t2);

	scalar_t<Double> radius = scalar_t<Double>(m_radius);

	Double A = sqr(dx) + sqr(dy),
	    B = scalar_t<Double>(2.f) * (dx * ox + dy * oy),
	    C = sqr(ox) + sqr(oy) - sqr(radius);

	auto [solution_found, near_t, far_t] =
	    math::solve_quadratic(A, B, C);

	// Barb doesn't intersect with the segment on the ray
	Mask out_bounds = !(near_t <= maxt && far_t >= mint); // NaN-aware conditionals

	Double z_pos_near = oz + dz * near_t,
	    z_pos_far  = oz + dz * far_t,
	    y_pos_near = oy + dy * near_t,
	    y_pos_far  = oy + dy * far_t,
	    x_pos_near = ox + dx * near_t,
	    x_pos_far  = ox + dx * far_t;

	// The cylinder containing the barb fully contains the segment of the ray
	Mask in_bounds = near_t < mint && far_t > maxt;

	Mask active_c = active;
	active_c &= solution_found && !out_bounds && !in_bounds &&
	    ((y_pos_near >= radius * Double(m_y_min) && y_pos_near <= radius * Double(m_y_max) && near_t >= mint
	      && t1.x() * x_pos_near + t1.y() * y_pos_near + t1.z() * z_pos_near <= 0
	      && t2.x() * x_pos_near + t2.y() * y_pos_near + t2.z() * (z_pos_near - length) >= 0) ||
	     (y_pos_far >= radius * Double(m_y_min) &&  y_pos_far <= radius * Double(m_y_max) && far_t <= maxt
	      && t1.x() * x_pos_far + t1.y() * y_pos_far + t1.z() * z_pos_far <= 0
	      && t2.x() * x_pos_far + t2.y() * y_pos_far + t2.z() * (z_pos_far - length) >= 0));
	return active_c;
    }

    SurfaceInteraction3f compute_surface_interaction(const Ray3f &ray,
                                                     PreliminaryIntersection3f pi,
                                                     HitComputeFlags flags,
                                                     Mask active) const override {
	MTS_MASK_ARGUMENT(active);

	bool differentiable = false;
	if constexpr (is_diff_array_v<Float>)
			 differentiable = requires_gradient(ray.o) ||
			 requires_gradient(ray.d) ||
			 parameters_grad_enabled();

	// Recompute ray intersection to get differentiable prim_uv and t
	if (differentiable && !has_flag(flags, HitComputeFlags::NonDifferentiable))
	    pi = ray_intersect_preliminary(ray, active);

	active &= pi.is_valid();

	SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
	si.t = select(active, pi.t, math::Infinity<Float>);

	si.p = ray(pi.t);

	Vector3f local = m_to_object.transform_affine(si.p);

	Float theta = atan2(local.y(), local.x());

	// TODO: scale
	if ((theta - HalfPi) < -HalfPi * 0.08f) {
	    si.dp_du = Vector3f(-local.y(), local.x(), m_span_x);
	    si.dp_dv = Vector3f(local.y(), -local.x(), m_span_y);
	} else if ((theta - HalfPi) > HalfPi * 0.08f) {
	    si.dp_du = Vector3f(local.y(), -local.x(), m_span_x);
	    si.dp_dv = Vector3f(local.y(), -local.x(), -m_span_y);
	} else { /* ramus */
	    si.t = math::Infinity<Float>;
	}

	si.dp_du = m_to_world.transform_affine(si.dp_du);
	si.dp_dv = m_to_world.transform_affine(si.dp_dv);

	si.n = Normal3f(normalize(cross(si.dp_du, si.dp_dv)));

	/* Mitigate roundoff error issues by a normal shift of the computed
	   intersection point */
	si.p += si.n * (m_radius - norm(head<2>(local)));

	if (m_flip_normals)
	    si.n *= -1.f;

	si.sh_frame.n = si.n;
	si.time = ray.time;

	// assume dn_du is not occupied, use it to store feather uv
	si.dn_du = Vector3f(m_uvw.x(), -m_uvw.y(), m_uvw.z());

	return si;
    }

    //! @}
    // =============================================================

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
    }

    void parameters_changed(const std::vector<std::string> &/*keys*/) override {
        update();
        Base::parameters_changed();
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "Barb[" << std::endl
            << "  to_world = " << string::indent(m_to_world, 13) << "," << std::endl
            << "  radius = "  << m_radius << "," << std::endl
            << "  length = "  << m_length << "," << std::endl
	    //            << "  surface_area = " << surface_area() << "," << std::endl
            << "  " << string::indent(get_children_string()) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
    private:
    Float m_span_x, m_span_y;
    ScalarVector3f m_t1, m_t2;
    bool m_flip_normals;
    ScalarVector3f m_uvw;
    ScalarFloat m_cover, m_span, m_y_min, m_y_max, m_radius, m_length, m_theta_a_c, m_theta_e_c;   // circle
};

MTS_IMPLEMENT_CLASS_VARIANT(Barb, Shape)
NAMESPACE_END(mitsuba)
