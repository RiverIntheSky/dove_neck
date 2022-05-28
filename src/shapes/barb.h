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
    static constexpr auto HalfPi    = math::HalfPi<Float>;

    Barb(const Properties &props) : Base(props) {
        // Update the to_world transform if face points and radius are also provided
        float radius = props.float_("radius", .01f);
	m_ramus_arc = props.float_("ramus_arc", .001f);
	m_cover = props.float_("cover", 0.78f);
	m_span = props.float_("span", 0.45f);
	m_uvw = ScalarVector3f(props.float_("H", 0.5f),
			       props.float_("mu", 0.35f),
			       props.float_("height", 0.65f)); // green feather

        ScalarPoint3f e0 = props.point3f("e0", ScalarPoint3f(0.f, 0.f, -0.5f)),
                      e1 = props.point3f("e1", ScalarPoint3f(0.f, 0.f, 0.5f));

	ScalarVector3f n = normalize(props.vector3f("n", ScalarVector3f(0.f)));

	m_t1 = normalize(props.vector3f("t1", e0 - e1));
	m_t2 = normalize(props.vector3f("t2", e0 - e1));

	ScalarFrame3f rot(e1 - e0);
	rot.s = normalize(cross(n, rot.n));
	rot.t = normalize(cross(rot.n, rot.s));

	m_to_world = m_to_world *
	    ScalarTransform4f::translate(e0) *
	    ScalarTransform4f::to_frame(rot) *
	    ScalarTransform4f::scale(ScalarVector3f(radius, radius, 1.f));

	update();
	set_children();
    }

    Barb(ScalarPoint3f e0, ScalarPoint3f e1, ScalarNormal3f n, ScalarVector3f t1, ScalarVector3f t2,
	 ScalarVector3f uvw, float radius, float cover, float span, float arc, const Properties &props):
	Base(props), m_t1(t1), m_t2(t2), m_uvw(uvw), m_cover(cover), m_span(span), m_ramus_arc(arc) {

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

        m_radius = S[0][0];
        m_length = S[2][2];

	// Reconstruct the to_world transform with uniform scaling and no shear
	m_to_world = transform_compose(ScalarMatrix3f(1.f), Q, T);
	m_to_object = m_to_world.inverse();

	m_t1 = m_to_object * m_t1;
	m_t2 = m_to_object * m_t2;

	m_span_y = m_radius * cot(m_span);
	m_span_x = m_radius * tan(m_span);

	// local coordinate
	auto [s, c] = sincos(m_cover);
	m_y_min = c - 1.f;
	m_x_m = s;
	m_y_max = cos(m_ramus_arc) - 1.f;
    }

    ScalarBoundingBox3f bbox() const override {
 	ScalarVector3f x1 = m_to_world * ScalarVector3f(m_x_m, m_y_max, 0.f) * m_radius,
	               x2 = m_to_world * ScalarVector3f(-m_x_m, m_y_min, 0.f) * m_radius,
	    	       x3 = m_to_world * ScalarVector3f(m_x_m, m_y_min, 0.f) * m_radius,
	               x4 = m_to_world * ScalarVector3f(-m_x_m, m_y_max, 0.f) * m_radius;

        ScalarPoint3f e0 = m_to_world * ScalarPoint3f(0.f, 0.f, 0.f),
                      e1 = m_to_world * ScalarPoint3f(0.f, 0.f, m_length);

	ScalarVector3f t1 = m_to_world * m_t1;
	ScalarVector3f t2 = m_to_world * m_t2;

	ScalarBoundingBox3f bbox;
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

    //! @}
    // =============================================================

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
	                 length = scalar_t<Double>(m_length),
	    	         y_max = scalar_t<Double>(m_y_max) * radius,
	                 y_min = scalar_t<Double>(m_y_min) * radius;

	ScalarVector3d t1 = ScalarVector3d(m_t1);
	ScalarVector3d t2 = ScalarVector3d(m_t2);

        Double A = sqr(dx) + sqr(dy),
               B = scalar_t<Double>(2.f) * (dx * ox + dy * (oy + radius)),
               C = sqr(ox) + sqr(oy) + 2.0 * oy * radius;


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

	Mask near_intersect = y_pos_near >= y_min && y_pos_near <= y_max && near_t >= mint
	    && t1.x() * x_pos_near + t1.y() * y_pos_near + t1.z() * z_pos_near <= 0
	    && t2.x() * x_pos_near + t2.y() * y_pos_near + t2.z() * (z_pos_near - length) >= 0;

	Mask far_intersect = y_pos_far >= y_min && y_pos_far <= y_max && far_t <= maxt
	    && t1.x() * x_pos_far + t1.y() * y_pos_far + t1.z() * z_pos_far <= 0
	    && t2.x() * x_pos_far + t2.y() * y_pos_far + t2.z() * (z_pos_far - length) >= 0;

        active &= solution_found && !out_bounds && !in_bounds && (near_intersect || far_intersect);

        pi.t = select(active, select(near_intersect, Float(near_t), Float(far_t)),
                      math::Infinity<Float>);


	pi.shape = this;

	return pi;
    }

    Mask ray_test(const Ray3f &ray_, Mask active) const override {
	MTS_MASK_ARGUMENT(active);

        Ray3f ray = m_to_object.transform_affine(ray_);
        Double mint = Double(ray.mint);
        Double maxt = Double(ray.maxt);

        Double ox = Double(ray.o.x()),
               oy = Double(ray.o.y()),
               oz = Double(ray.o.z()),
               dx = Double(ray.d.x()),
               dy = Double(ray.d.y()),
               dz = Double(ray.d.z());

        scalar_t<Double> radius = scalar_t<Double>(m_radius),
             	         length = scalar_t<Double>(m_length),
	                 y_max = scalar_t<Double>(m_y_max) * radius,
	                 y_min = scalar_t<Double>(m_y_min) * radius;

	ScalarVector3d t1 = ScalarVector3d(m_t1);
	ScalarVector3d t2 = ScalarVector3d(m_t2);

        Double A = sqr(dx) + sqr(dy),
               B = scalar_t<Double>(2.f) * (dx * ox + dy * (oy + radius)),
               C = sqr(ox) + sqr(oy) + 2.0 * oy * radius;

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

        // Barb fully contains the segment of the ray
        Mask in_bounds = near_t < mint && far_t > maxt;

        Mask valid_intersection =
            active && solution_found && !out_bounds && !in_bounds &&
	    ((t1.x() * x_pos_near + t1.y() * y_pos_near + t1.z() * z_pos_near <= 0 &&
	      t2.x() * x_pos_near + t2.y() * y_pos_near + t2.z() * (z_pos_near - length) >= 0 &&
	      y_pos_near <= y_max && y_pos_near >= y_min && near_t >= mint) ||
	     (t1.x() * x_pos_far + t1.y() * y_pos_far + t1.z() * z_pos_far <= 0 &&
	      t2.x() * x_pos_far + t2.y() * y_pos_far + t2.z() * (z_pos_far - length) >= 0 &&
	      y_pos_far <= y_max && y_pos_far >= y_min && far_t <= maxt));

        return valid_intersection;
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

        Float phi = atan2(local.y() + m_radius, local.x());
	if ((phi - HalfPi) < -m_ramus_arc) {
	    si.dp_du = Vector3f(-local.y() - m_radius, local.x(), m_span_x);
	    si.dp_dv = Vector3f(local.y() + m_radius, -local.x(), m_span_y);
	} else if ((phi - HalfPi) > m_ramus_arc) {
	    si.dp_du = Vector3f(local.y() + m_radius, -local.x(), m_span_x);
	    si.dp_dv = Vector3f(local.y() + m_radius, -local.x(), -m_span_y);
	} else { /* ramus */
	    si.t = math::Infinity<Float>;
	}

	si.dp_du = m_to_world.transform_affine(si.dp_du);
	si.dp_dv = m_to_world.transform_affine(si.dp_dv);

	si.n = Normal3f(normalize(cross(si.dp_du, si.dp_dv)));

	si.sh_frame.n = si.n;
	si.time = ray.time;

	// assume dn_du is not occupied, use it to store feather uv
	si.dn_du = Vector3f(m_uvw.x(), -m_uvw.y(), m_uvw.z());

	// tilted more at the root
	Float taper_root = 0.1f * m_cover;
	Float dist_to_center = abs(phi - HalfPi);
	if (dist_to_center < taper_root)
	    si.dn_du.y() += 0.5f * (dist_to_center / taper_root - 1.f);

	// thinner at the tip
	Float taper_tip = 0.94f * m_cover;
	Float dist_to_tip = abs(phi - HalfPi);
	if (dist_to_tip > taper_tip) {
	    if (fmod(3.f * local.z(), m_length) > m_length * (dist_to_tip - m_cover) / (taper_tip - m_cover))
		si.t = math::Infinity<Float>;
	}

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
            << "  " << string::indent(get_children_string()) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ScalarVector3f m_t1, m_t2;
    ScalarVector3f m_uvw;
    ScalarFloat m_cover, m_span, m_span_x, m_span_y, m_y_min, m_y_max, m_x_m, m_radius, m_length,
	m_ramus_arc;
};

MTS_IMPLEMENT_CLASS_VARIANT(Barb, Shape)
NAMESPACE_END(mitsuba)
