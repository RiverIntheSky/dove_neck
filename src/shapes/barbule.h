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

#if defined(MTS_ENABLE_OPTIX)
    #include "optix/barbule.cuh"
#endif

// #define FANCY_BARB

// static float PHI_MID = 0.756045006640150f;
// static float PHI_HALF_SPAN = 0.550857537253203f;
static float PHI_MID = 0.f;
static float PHI_HALF_SPAN = 0.3f;

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class Barbule final : public Shape<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Shape, m_to_world, m_to_object, set_children,
                    get_children_string, parameters_grad_enabled)
    MTS_IMPORT_TYPES()

    using typename Base::ScalarIndex;
    using typename Base::ScalarSize;

    Barbule(const Properties &props) : Base(props) {
	/// Are the sphere normals pointing inwards? default: no
	m_flip_normals = props.bool_("flip_normals", false);

	// Update the to_world transform if face points and radius are also provided
	// default unit micrometer
	float radius = props.float_("radius", 0.04f);
	ScalarPoint3f e0 = props.point3f("e0", ScalarPoint3f(0.f, 0.f, -0.012f)),
	    e1 = props.point3f("e1", ScalarPoint3f(0.f, 0.f, 0.012f));

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

    Barbule(ScalarPoint3f e0, ScalarPoint3f e1, ScalarNormal3f n, ScalarVector3f t1, ScalarVector3f t2,
	    ScalarVector3f uvw, float radius, const Properties &props):
	Base(props), m_t1(t1), m_t2(t2), m_uvw(uvw) {
	/// Are the sphere normals pointing inwards? default: no
	m_flip_normals = props.bool_("flip_normals", false);

	ScalarFrame3f rot(e1 - e0);
	rot.t = n;
	rot.s = normalize(cross(rot.t, rot.n));

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
	m_scale = S[0][0] / 40.f;

	// circle
	m_radius = S[0][0];
	m_theta_a_c = PHI_MID - PHI_HALF_SPAN;
	m_theta_e_c = PHI_MID + PHI_HALF_SPAN;
	m_k_c = -1.060499542699545f;
	m_b_c = m_scale * 49.680157478830810f;

# ifdef FANCY_BARB
	m_ratio_c = 0.458059047907465f;

	// ellipse
	m_a = m_scale * 3.536009652119812f;
	m_b = m_scale * 2.315960074728735f;
	m_xc = m_scale * 9.085174678692397f;
	m_yc = m_scale * 36.474328531413940f;
	m_theta_a_e = 1.179531456009883f;
	m_theta_e_e = math::Pi<ScalarFloat> * 1.5f;
	m_k_e = 3.305112925345886f;
	m_b_e = m_scale * 4.130840197113802f;
	m_ratio_e = 0.110762203419122f;

	// parabola
	m_p2 = -0.028752795120653f / m_scale;
	m_p1 = 0.522448332343573f;
	m_p0 = m_scale * 31.785101276718756f;
	m_x_a = m_scale * 9.085174678692397f;
	m_x_e = m_scale * 39.160912188184874f;
	m_k_p = -0.864761518712974f;
	m_b_p = m_scale * 42.014877909603900f;
	m_ratio_p = 0.431178748673414f;
# endif

	// Reconstruct the to_world transform with uniform scaling and no shear
	m_to_world = transform_compose(ScalarMatrix3f(1.f), Q, T);
	m_to_object = m_to_world.inverse();

	m_t1 = m_to_object * m_t1;
	m_t2 = m_to_object * m_t2;
	// m_inv_surface_area = rcp(surface_area() * m_scale);
    }
    
    ScalarBoundingBox3f bbox() const override {
	ScalarBoundingBox3f bbox;
	ScalarVector3f x1 = m_to_world * ScalarVector3f(39.160912188184874, 8.150028011563727, 0.f) * m_scale;
# ifdef FANCY_BARB
	ScalarVector3f x2 = m_to_world * ScalarVector3f(4.865824385326900, 36.221112359934950, 0.f) * m_scale,
	    x3 = m_to_world * ScalarVector3f(10.496066274644999, 43.099708054678630, 0.f) * m_scale,
	    x4 = m_to_world * ScalarVector3f(44.791154077502970, 15.028623706307407, 0.f) * m_scale;
# else
	ScalarVector3f x2 = m_to_world * ScalarVector3f(10.433660251595880, 38.615265553330960, 0.f) * m_scale,
	    x3 = m_to_world * ScalarVector3f(14.738574609844620, 42.674592488343500, 0.f) * m_scale,
	    x4 = m_to_world * ScalarVector3f(43.465826546433610, 12.209354946576270, 0.f) * m_scale;
# endif

	ScalarPoint3f e0 = m_to_world * ScalarPoint3f(0.f, 0.f, 0.f),
	    e1 = m_to_world * ScalarPoint3f(0.f, 0.f, m_length);
	ScalarVector3f t1 = m_to_world * m_t1;
	ScalarVector3f t2 = m_to_world * m_t2;

	assert(dot(e0 - e1, t1) > 0.f);
	ScalarVector3f proj1 = (e1 - e0) / dot(e0 - e1, t1);
	bbox.expand(ScalarPoint3f(e0 + x1 + dot(x1, t1) * proj1));
	bbox.expand(ScalarPoint3f(e0 + x2 + dot(x2, t1) * proj1));
	bbox.expand(ScalarPoint3f(e0 + x3 + dot(x3, t1) * proj1));
	bbox.expand(ScalarPoint3f(e0 + x4 + dot(x4, t1) * proj1));

	assert(dot(e0 - e1, t2) > 0.f);
	ScalarVector3f proj2 = (e1 - e0) / dot(e0 - e1, t2);
	bbox.expand(ScalarPoint3f(e1 + x1 + dot(x1, t2) * proj2));
	bbox.expand(ScalarPoint3f(e1 + x2 + dot(x2, t2) * proj2));
	bbox.expand(ScalarPoint3f(e1 + x3 + dot(x3, t2) * proj2));
	bbox.expand(ScalarPoint3f(e1 + x4 + dot(x4, t2) * proj2));
	return bbox;
    }

    // =============================================================
    //! @{ \name Ray tracing routines
    // =============================================================

    PreliminaryIntersection3f ray_intersect_preliminary(const Ray3f &ray_,
                                                        Mask active) const override {
	MTS_MASK_ARGUMENT(active);

	PreliminaryIntersection3f pi = zero<PreliminaryIntersection3f>();
	
	using Double = std::conditional_t<is_cuda_array_v<Float>, Float, Float64>;

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
	    length = scalar_t<Double>(m_length),
	    kc = scalar_t<Double>(m_k_c),
	    bc = scalar_t<Double>(m_b_c);

	ScalarVector3d t1 = ScalarVector3d(m_t1);
	ScalarVector3d t2 = ScalarVector3d(m_t2);
	
	Float tc = math::Infinity<Float>;
# ifdef FANCY_BARB
	scalar_t<Double> a = scalar_t<Double>(m_a),
	    b = scalar_t<Double>(m_b),
	    p2 = scalar_t<Double>(m_p2),
	    p1 = scalar_t<Double>(m_p1),
	    p0 = scalar_t<Double>(m_p0),
	    xc = scalar_t<Double>(m_xc),
	    yc = scalar_t<Double>(m_yc),
	    ke = scalar_t<Double>(m_k_e),
	    be = scalar_t<Double>(m_b_e),
	    kp = scalar_t<Double>(m_k_p),
	    bp = scalar_t<Double>(m_b_p);
	Float te = math::Infinity<Float>,
	    tp = math::Infinity<Float>;
# endif	
	{   // circle
	    Double A = sqr(dx) + sqr(dy),
		B = scalar_t<Double>(2.f) * (dx * ox + dy * oy),
		C = sqr(ox) + sqr(oy) - sqr(radius);

	    auto [solution_found, near_t, far_t] =
		math::solve_quadratic(A, B, C);

	    // Barbule doesn't intersect with the segment on the ray
	    Mask out_bounds = !(near_t <= maxt && far_t >= mint); // NaN-aware conditionals

	    Double z_pos_near = oz + dz * near_t,
		z_pos_far  = oz + dz * far_t,
		y_pos_near = oy + dy * near_t,
		y_pos_far  = oy + dy * far_t,
		x_pos_near = ox + dx * near_t,
		x_pos_far  = ox + dx * far_t;

	    // The cylinder containing the barbule fully contains the segment of the ray
	    Mask in_bounds = near_t < mint && far_t > maxt;

	    Mask near_intersect = y_pos_near >= kc * x_pos_near + bc && near_t >= mint
		&& t1.x() * x_pos_near + t1.y() * y_pos_near + t1.z() * z_pos_near <= 0
		&& t2.x() * x_pos_near + t2.y() * y_pos_near + t2.z() * (z_pos_near - length) >= 0;

	    Mask far_intersect = y_pos_far >= kc * x_pos_far + bc  && far_t <= maxt
		&& t1.x() * x_pos_far + t1.y() * y_pos_far + t1.z() * z_pos_far <= 0
		&& t2.x() * x_pos_far + t2.y() * y_pos_far + t2.z() * (z_pos_far - length) >= 0;

	    Mask active_c = solution_found && !out_bounds && !in_bounds && (near_intersect || far_intersect);

	    tc = select(active & active_c, select(near_intersect, Float(near_t), Float(far_t)),
			math::Infinity<Float>);
	}
# ifdef FANCY_BARB
	{    // ellipse
	    Double A = sqr(dx * b) + sqr(dy * a),
		B = scalar_t<Double>(2.f) * (sqr(b) * dx * (ox - xc) + sqr(a) * dy * (oy - yc)),
		C = sqr(b * (ox - xc)) + sqr(a * (oy - yc)) - sqr(a * b);

	    auto [solution_found, near_t, far_t] =
		math::solve_quadratic(A, B, C);
	    
	    // Arc doesn't intersect with the segment on the ray
	    Mask out_bounds = !(near_t <= maxt && far_t >= mint); // NaN-aware conditionals

	    Double z_pos_near = oz + dz * near_t,
		z_pos_far  = oz + dz * far_t,
		y_pos_near = oy + dy * near_t,
		y_pos_far  = oy + dy * far_t,
		x_pos_near = ox + dx * near_t,
		x_pos_far  = ox + dx * far_t;

	    // The cylinder containing the barbule fully contains the segment of the ray
	    Mask in_bounds = near_t < mint && far_t > maxt;

	    Mask near_intersect = y_pos_near >= ke * x_pos_near + be && near_t >= mint
		&& t1.x() * x_pos_near + t1.y() * y_pos_near + t1.z() * z_pos_near <= 0
		&& t2.x() * x_pos_near + t2.y() * y_pos_near + t2.z() * (z_pos_near - length) >= 0;

	    Mask far_intersect = y_pos_far >= ke * x_pos_far + be  && far_t <= maxt
		&& t1.x() * x_pos_far + t1.y() * y_pos_far + t1.z() * z_pos_far <= 0
		&& t2.x() * x_pos_far + t2.y() * y_pos_far + t2.z() * (z_pos_far - length) >= 0;

	    Mask active_e = solution_found && !out_bounds && !in_bounds && (near_intersect || far_intersect);

	    te = select(active & active_e, select(near_intersect, Float(near_t), Float(far_t)),
			math::Infinity<Float>);
	}

	{   // parabola
	    Double A = p2 * sqr(dx),
		B = scalar_t<Double>(2.f) * p2 * dx * ox + p1 * dx - dy,
		C = p2 * sqr(ox) + p1 * ox + p0 - oy;

	    auto [solution_found, near_t, far_t] =
		math::solve_quadratic(A, B, C);
		     
	    // Arc doesn't intersect with the segment on the ray
	    Mask out_bounds = !(near_t <= maxt && far_t >= mint); // NaN-aware conditionals

	    Double z_pos_near = oz + dz * near_t,
		z_pos_far  = oz + dz * far_t,
		y_pos_near = oy + dy * near_t,
		y_pos_far  = oy + dy * far_t,
		x_pos_near = ox + dx * near_t,
		x_pos_far  = ox + dx * far_t;

	    // The cylinder containing the barbule fully contains the segment of the ray
	    Mask in_bounds = near_t < mint && far_t > maxt;

	    Mask near_intersect = y_pos_near >= kp * x_pos_near + bp && near_t >= mint
		&& t1.x() * x_pos_near + t1.y() * y_pos_near + t1.z() * z_pos_near <= 0
		&& t2.x() * x_pos_near + t2.y() * y_pos_near + t2.z() * (z_pos_near - length) >= 0;

	    Mask far_intersect = y_pos_far >= kp * x_pos_far + bp && far_t <= maxt
		&& t1.x() * x_pos_far + t1.y() * y_pos_far + t1.z() * z_pos_far <= 0
		&& t2.x() * x_pos_far + t2.y() * y_pos_far + t2.z() * (z_pos_far - length) >= 0;

	    Mask active_p = solution_found && !out_bounds && !in_bounds && (near_intersect || far_intersect);

	    tp = select(active & active_p, select(near_intersect, Float(near_t), Float(far_t)),
			math::Infinity<Float>);
	}
	
	pi.t = select(tc < te, tc, te);
	pi.t = select(tp < pi.t, tp, pi.t);
# else
	pi.t = tc;
# endif
	
	pi.shape = this;

	return pi;
    }

    Mask ray_test(const Ray3f &ray_, Mask active) const override {
	MTS_MASK_ARGUMENT(active);
	using Double = std::conditional_t<is_cuda_array_v<Float>, Float, Float64>;

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
	{   // circle
	    scalar_t<Double> radius = scalar_t<Double>(m_radius),
		kc = scalar_t<Double>(m_k_c),
		bc = scalar_t<Double>(m_b_c);
	    
	    Double A = sqr(dx) + sqr(dy),
		B = scalar_t<Double>(2.f) * (dx * ox + dy * oy),
		C = sqr(ox) + sqr(oy) - sqr(radius);

	    auto [solution_found, near_t, far_t] =
		math::solve_quadratic(A, B, C);

	    // Barbule doesn't intersect with the segment on the ray
	    Mask out_bounds = !(near_t <= maxt && far_t >= mint); // NaN-aware conditionals

	    Double z_pos_near = oz + dz * near_t,
		z_pos_far  = oz + dz * far_t,
		y_pos_near = oy + dy * near_t,
		y_pos_far  = oy + dy * far_t,
		x_pos_near = ox + dx * near_t,
		x_pos_far  = ox + dx * far_t;

	    // The cylinder containing the barbule fully contains the segment of the ray
	    Mask in_bounds = near_t < mint && far_t > maxt;

	    Mask active_c = active;
	    active_c &= solution_found && !out_bounds && !in_bounds &&
		((y_pos_near >= kc * x_pos_near + bc && near_t >= mint
		  && t1.x() * x_pos_near + t1.y() * y_pos_near + t1.z() * z_pos_near <= 0
		  && t2.x() * x_pos_near + t2.y() * y_pos_near + t2.z() * (z_pos_near - length) >= 0) ||
		 (y_pos_far >= kc * x_pos_far + bc  && far_t <= maxt
		  && t1.x() * x_pos_far + t1.y() * y_pos_far + t1.z() * z_pos_far <= 0
		  && t2.x() * x_pos_far + t2.y() * y_pos_far + t2.z() * (z_pos_far - length) >= 0));
# ifdef FANCY_BARB
	    if (any_or<true>(active_c))
# endif
		return active_c;
	}
# ifdef FANCY_BARB
	{    // ellipse
	    scalar_t<Double> a = scalar_t<Double>(m_a),
		b = scalar_t<Double>(m_b),
		xc = scalar_t<Double>(m_xc),
		yc = scalar_t<Double>(m_yc),
		ke = scalar_t<Double>(m_k_e),
		be = scalar_t<Double>(m_b_e);
		
	    Double A = sqr(dx * b) + sqr(dy * a),
		B = scalar_t<Double>(2.f) * (sqr(b) * dx * (ox - xc) + sqr(a) * dy * (oy - yc)),
		C = sqr(b * (ox - xc)) + sqr(a * (oy - yc)) - sqr(a * b);

	    auto [solution_found, near_t, far_t] =
		math::solve_quadratic(A, B, C);
	    
	    // Arc doesn't intersect with the segment on the ray
	    Mask out_bounds = !(near_t <= maxt && far_t >= mint); // NaN-aware conditionals

	    Double z_pos_near = oz + dz * near_t,
		z_pos_far  = oz + dz * far_t,
		y_pos_near = oy + dy * near_t,
		y_pos_far  = oy + dy * far_t,
		x_pos_near = ox + dx * near_t,
		x_pos_far  = ox + dx * far_t;

	    // The cylinder containing the barbule fully contains the segment of the ray
	    Mask in_bounds = near_t < mint && far_t > maxt;

	    Mask active_e = active;
	    active_e &= solution_found && !out_bounds && !in_bounds &&
		((y_pos_near >= ke * x_pos_near + be && near_t >= mint
		  && t1.x() * x_pos_near + t1.y() * y_pos_near + t1.z() * z_pos_near <= 0
		  && t2.x() * x_pos_near + t2.y() * y_pos_near + t2.z() * (z_pos_near - length) >= 0) ||
		 (y_pos_far >= ke * x_pos_far + be  && far_t <= maxt
		  && t1.x() * x_pos_far + t1.y() * y_pos_far + t1.z() * z_pos_far <= 0
		  && t2.x() * x_pos_far + t2.y() * y_pos_far + t2.z() * (z_pos_far - length) >= 0));
	    if (any_or<true>(active_e))
		return active_e;
	}

	{   // parabola
	    scalar_t<Double> p2 = scalar_t<Double>(m_p2),
		p1 = scalar_t<Double>(m_p1),
		p0 = scalar_t<Double>(m_p0),
		kp = scalar_t<Double>(m_k_p),
		bp = scalar_t<Double>(m_b_p);
	    Double A = p2 * sqr(dx),
		B = scalar_t<Double>(2.f) * p2 * dx * ox + p1 * dx - dy,
		C = p2 * sqr(ox) + p1 * ox + p0 - oy;

	    auto [solution_found, near_t, far_t] =
		math::solve_quadratic(A, B, C);
		     
	    // Arc doesn't intersect with the segment on the ray
	    Mask out_bounds = !(near_t <= maxt && far_t >= mint); // NaN-aware conditionals

	    Double z_pos_near = oz + dz * near_t,
		z_pos_far  = oz + dz * far_t,
		y_pos_near = oy + dy * near_t,
		y_pos_far  = oy + dy * far_t,
		x_pos_near = ox + dx * near_t,
		x_pos_far  = ox + dx * far_t;

	    // The cylinder containing the barbule fully contains the segment of the ray
	    Mask in_bounds = near_t < mint && far_t > maxt;

	    Mask active_p = active;
	    active_p &= solution_found && !out_bounds && !in_bounds &&
		((y_pos_near >= kp * x_pos_near + bp && near_t >= mint
		  && t1.x() * x_pos_near + t1.y() * y_pos_near + t1.z() * z_pos_near <= 0
		  && t2.x() * x_pos_near + t2.y() * y_pos_near + t2.z() * (z_pos_near - length) >= 0) ||
		 (y_pos_far >= kp * x_pos_far + bp && far_t <= maxt
		  && t1.x() * x_pos_far + t1.y() * y_pos_far + t1.z() * z_pos_far <= 0
		  && t2.x() * x_pos_far + t2.y() * y_pos_far + t2.z() * (z_pos_far - length) >= 0));
	    if (any_or<true>(active_p))
		return active_p;
	}
	return Mask(false);
# endif
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
	Vector3f dp_du;
      	
# ifdef FANCY_BARB
	(si.uv).y() = local.z() / m_length;
	if (any_or<true>(abs(norm(head<2>(local)) - m_radius) < math::Epsilon<Float> * 10.f)) {
	    // circle
	    Float theta = atan2(local.y(), local.x());
	    (si.uv).x() = (theta - m_theta_a_c) / (m_theta_e_c - m_theta_a_c) * m_ratio_c;
	    dp_du = (m_theta_e_c - m_theta_a_c) / m_ratio_c * Vector3f(-local.y(), local.x(), 0.f);
	} else if (any_or<true>(local.y() >= m_k_e * local.x() + m_b_e)) {
	    // ellipse
	    Float theta = atan2((local.y() - m_yc) / m_b, (local.x() - m_xc) / m_a);
	    (si.uv).x() = (theta - m_theta_a_e) / (m_theta_e_e - m_theta_a_e) * m_ratio_e + m_ratio_c;
	    dp_du = (m_theta_e_e - m_theta_a_e) / m_ratio_e * Vector3f(-m_a * sin(theta), m_b * cos(theta), 0.f);
	} else {
	    // parabola
	    (si.uv).x() = (local.x() - m_x_a) / (m_x_e - m_x_a) * m_ratio_p + 1.f - m_ratio_p;
	    dp_du = (m_x_e - m_x_a) / m_ratio_p * Vector3f(1.f, fmadd(2.f*m_p2, local.x(), m_p1), 0.f);
	}
	si.dp_du = m_to_world.transform_affine(dp_du);
# else
	Float theta = atan2(local.y(), local.x());
	si.uv = Point2f((theta - m_theta_a_c) / (m_theta_e_c - m_theta_a_c), local.z() / m_length);
	dp_du = (m_theta_e_c - m_theta_a_c) * Vector3f(-local.y(), local.x(), 0.f);
	si.dp_du = m_to_world.transform_affine(dp_du);
# endif
	Vector3f dp_dv = Vector3f(0.f, 0.f, m_length);

	si.dp_dv = m_to_world.transform_affine(dp_dv);
	si.n = Normal3f(normalize(cross(si.dp_du, si.dp_dv)));

# ifndef FANCY_BARB
	/* Mitigate roundoff error issues by a normal shift of the computed
	   intersection point */
	si.p += si.n * (m_radius - norm(head<2>(local)));
# endif

	if (m_flip_normals)
	    si.n *= -1.f;

	si.sh_frame.n = si.n;
	si.time = ray.time;

	// assume dn_du is not occupied, use it to store feather uv
	si.dn_du = Vector3f(m_uv.x(), m_uv.y(), 0.f);
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
#if defined(MTS_ENABLE_OPTIX)
        optix_prepare_geometry();
#endif
    }

#if defined(MTS_ENABLE_OPTIX)
    using Base::m_optix_data_ptr;

    void optix_prepare_geometry() override {
        if constexpr (is_cuda_array_v<Float>) {
	    if (!m_optix_data_ptr)
		m_optix_data_ptr = cuda_malloc(sizeof(OptixBarbuleData));

	    OptixBarbuleData data = {bbox(), m_to_world, m_to_object, m_length,
		m_radius, m_flip_normals, m_k_c, m_b_c,
		m_a, m_b, m_p2, m_p1, m_p0, m_xc, m_yc,
		m_k_e, m_b_e, m_k_p, m_b_p, m_t1, m_t2,
		m_theta_a_c, m_theta_e_c, m_ratio_c,
		m_theta_a_e, m_theta_e_e, m_ratio_e,
		m_ratio_p, m_x_a, m_x_e};

	    cuda_memcpy_to_device(m_optix_data_ptr, &data, sizeof(OptixBarbuleData));
	}
    }
#endif  

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "Barbule[" << std::endl
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
    ScalarFloat m_scale;
    ScalarVector3f m_t1, m_t2;
    bool m_flip_normals;
    ScalarVector2f m_uv;
    ScalarFloat m_radius, m_length, m_theta_a_c, m_theta_e_c, m_k_c, m_b_c, m_ratio_c;   // circle
    ScalarFloat m_a, m_b, m_xc, m_yc, m_theta_a_e, m_theta_e_e, m_k_e, m_b_e, m_ratio_e; // ellipse
    ScalarFloat m_p2, m_p1, m_p0, m_x_a, m_x_e, m_k_p, m_b_p, m_ratio_p;     // parabola
};

MTS_IMPLEMENT_CLASS_VARIANT(Barbule, Shape)
NAMESPACE_END(mitsuba)
