#pragma once

#include <mitsuba/core/properties.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/ior.h>
#include <mitsuba/render/barbule.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/sampler.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class BarbuleBsdfEval : public BSDF<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(BSDF, m_flags, m_components)
    MTS_IMPORT_TYPES(Texture, Sampler)
    static constexpr auto Pi        = math::Pi<Float>;
    static constexpr auto HalfPi    = math::HalfPi<Float>;
    static constexpr auto TwoPi     = math::TwoPi<Float>;
    static constexpr auto Inf       = math::Infinity<Float>;

    /* convert between gamma and phi */
    MTS_INLINE Float to_phi(const Float gamma) const {
	 auto [sin_gamma, cos_gamma] = sincos(gamma);
	 return atan2(m_b * sin_gamma, cos_gamma);
    }

    MTS_INLINE Float to_gamma(const Float phi) const {
	 auto [sin_phi, cos_phi] = sincos(phi);
	 return atan2(sin_phi, m_b * cos_phi);
    }

    /* convert from gamma to the parametrized point */
    MTS_INLINE Point2f to_p(const Float gamma) const {
	 auto [sin_gamma, cos_gamma] = sincos(gamma);
	 return Point2f(m_b * cos_gamma, sin_gamma);
    }

    /* compute the nearst intersection of a ray defined by point p and direction d
     * with the ellipse. The intersection is point (b*cos(gamma), a*sin(gamma)), returns
     * gamma. If no intersection is found, returns NaN
     */
    MTS_INLINE Float intersect(Point2f p, Vector2f d) const {
	 Float x = p.x(),
	      y = p.y(),
	      dx = d.x(),
	      dy = d.y(),
	      A = sqr(dx) + m_b2 * sqr(dy),
	      B = 2.f * (x * dx + y * dy * m_b2),
	      C = sqr(x) + m_b2 * sqr(y) - m_b2;
	 auto [solution_found, near_t, far_t] = math::solve_quadratic(A, B, C);

	 Float near_gamma = select(near_t > 0.f,
				   atan2(m_b * (y + near_t * dy),
					 x + near_t * dx), 0.f / 0.f);
	 near_gamma = select(in_range(m_gamma_0, near_gamma, m_gamma_1), near_gamma, 0.f / 0.f);

	 Float far_gamma = select(far_t > 0.f,
				  atan2(m_b * (y + far_t * dy),
					x + far_t * dx), 0.f / 0.f);
	 far_gamma = select(in_range(m_gamma_0, far_gamma, m_gamma_1), far_gamma, 0.f / 0.f);

	 return select(solution_found,
		       select(enoki::isfinite(near_gamma), near_gamma,
			      select(enoki::isfinite(far_gamma), far_gamma, 0.f / 0.f)),
		       0.f / 0.f);
    }

    /* compute the nearst intersection of a ray defined by point p and direction d
     * with the ellipse centered at O, orientation r. The intersection is point
     * R(r) * (b*cos(gamma), a*sin(gamma))' + O, returns gamma.
     * If no intersection is found, returns Inf.
     */
    MTS_INLINE Float intersect(Point2f p, Float d, Point2f O, Float r) const {
	auto [s, c] = sincos(d);
	Point2f p_ = rotate(p - O, -r);
	Vector2f d_ = rotate(Vector2f(c, s), -r);
	return intersect(p_, d_);
    }

    /* rotate point p by angle r */
    MTS_INLINE Point2f rotate(Point2f p, Float r) const {
	auto [s, c] = sincos(r);
	return Point2f(c * p.x() - s * p.y(), s * p.x() + c * p.y());
    }

    /* compute the tangent of the ellipse. The tangent line passes through the
     * point p, the tangent point is (b*cos(gamma_t0), a*sin(gamma_t0))
     * and (b*cos(gamma_t1), a*sin(gamma_t1))
     * returns [gamma_t0, gamma_t1] and gamma_t0 <= gamma_t1,
     * returns nan if point inside ellipse.
     */
    MTS_INLINE std::tuple<Float, Float> tangent(Point2f p) const {
	Float x = p.x(), y = p.y();
	Float tmp1 = atan2(m_b * y, x);
	Float tmp2 = acos(m_b * rsqrt(sqr(x) + sqr(m_b * y)));
	return {tmp1 - tmp2, tmp1 + tmp2};
    }

    /* compute the tangent of the ellipse centered at O, orientation r.
     * The tangent line passes through the point p,
     * the tangent point is R(r) * (b*cos(gamma_t0), a*sin(gamma_t0))' + O
     * and R(r) * (b*cos(gamma_t1), a*sin(gamma_t1))' + O, with the former one being the point
     * with the smaller y coordinate.
     */
    MTS_INLINE std::tuple<Float, Float> tangent(Point2f p, Point2f O, Float r) const {
	auto [s, c] = sincos(r);
	Point2f p_ = rotate(p - O, -r);
	return tangent(p_);
    }

    /* find the valid range of gamma on the ellipse when light incidents
     * in gamma direction. The range is defined by [gamma_h0, gamma_h1]
     */
    MTS_INLINE std::tuple<Float, Float, Float, Float>
    find_valid_gamma(const Float gamma, const Float H, const Float mu) const {
	Float gamma_ = gamma - mu; // gamma_ in local ellipse coordinate
	Float gamma_c = to_gamma(min(gamma_ + HalfPi, m_phi_1));

	Point2f c = rotate(to_p(gamma_c), mu); // point of contact
	Vector2f d(-cos(gamma), -sin(gamma)); // incident ray direction
	Point2f p0 = rotate(to_p(m_gamma_0), mu);
	Point2f p1 = rotate(to_p(m_gamma_1), mu);

	auto [gamma_t0_, gamma_t1_] = tangent(p1, Point2f(0.f, H), mu);

	Float gamma_t0 = atan2(cos(gamma_t0_), m_b * sin(-gamma_t0_));
	Float gamma_t1 = atan2(cos(gamma_t1_), m_b * sin(-gamma_t1_));
	Float gamma_t2 = atan2(p0.y() - p1.y(), p0.x() - p1.x());

	// lower bound of valid region
	Float gamma_h0
	    = select(gamma > HalfPi, m_gamma_1,
		     fmax(intersect(c, gamma + Pi, Point2f(0.f, H), mu),
			  select(gamma_ < gamma_t0, m_gamma_0,
				 to_gamma(gamma_ - HalfPi))));

	gamma_h0 = fmin(fmax(gamma_h0, m_gamma_0), m_gamma_1);

	Float phi_h0 = to_phi(gamma_h0);
	Point2f pa = rotate(to_p(gamma_h0), mu);

	// upper bound of valid region
	Float gamma_h1
	    = select(gamma < -HalfPi,
		     fmin(select(gamma < gamma_t2,
				 intersect(p1, gamma + Pi, Point2f(0.f, H), mu),
				 intersect(p0, gamma + Pi, Point2f(0.f, H), mu)),
			  gamma_c),
		     select(gamma > HalfPi, m_gamma_1,
			    select(gamma_ > gamma_t1,
				   intersect(pa, gamma + Pi, Point2f(0.f, -H), mu), gamma_c)));

	gamma_h1 = fmin(fmax(gamma_h1, gamma_h0), m_gamma_1);

	Float phi_h1 = to_phi(gamma_h1);

	return {gamma_h0, gamma_h1, phi_h0, phi_h1};
    }

    /* compute NDF given theta_h and gamma_h */
    MTS_INLINE Float NDF(Float theta_h, Float phi_h, Float H) const {
	auto [s, c] = sincos(phi_h);
	Float D_phi = m_b2 / norm(m_H) * pow(sqr(s) + m_b2 * sqr(c), -1.5f);
	return select(phi_h > m_phi_0 && phi_h < m_phi_1 && abs(theta_h) < m_theta_d,
		      D_phi * m_D_theta / cos(theta_h) / H, 0.f);
    }

    /* check if a < t < b */
    MTS_INLINE Mask in_range(Float a, Float t, Float b) const {
	return (a < t) && (t < b);
    }

    MTS_INLINE std::tuple<Float, Float, Float> get_uvw(const SurfaceInteraction3f & si) const {
	// feather uv overwrites bsdf parameters
	Float H = (si.dn_du).x();
	H *= 2.5f; // legacy
	H = select(H == 0, m_H, H) + (si.dn_dv).x();
	Float mu = (si.dn_du).y();
	mu = select(mu == 0, m_mu, mu);
	Float height = (si.dn_du).z();
        height = select(height == 0, m_height, height);
	return {H, mu, height};
    }

    BarbuleBsdfEval(const Properties &props) : Base(props) {
	m_reflectance_melanin = ior_from_file<Spectrum, Texture>("data/ior/melanin.r.spd");

	// Specifies the external index of refraction at the interface
	m_ext_ior = lookup_ior(props, "ext_ior", "air");

	std::string film_material = props.string("film_ior", "keratin");
	try {
	    std::tie(m_film_ior, std::ignore) =
		complex_ior_from_file<Spectrum, Texture>(props.string("material", film_material));
	} catch (const std::exception& e) {
	    m_film_ior = props.texture<Texture>("none", lookup_ior(props, "film_ior",  film_material));
	}

	// Height of the layer in micrometer
	m_height = props.float_("height", 0.65f); // green feather

        BSDFFlags extra = BSDFFlags::Anisotropic;
	// Reflection into the upper hemisphere
        m_components.push_back(BSDFFlags::GlossyReflection | BSDFFlags::FrontSide | extra);
	m_components.push_back(BSDFFlags::Null | BSDFFlags::FrontSide |
			       BSDFFlags::BackSide | extra);

        m_flags = m_components[0] | m_components[1];

	// shape parameters
	m_b = props.float_("b", 0.0479f); // short axis
	m_phi_0 = props.float_("phi_0", -1.0708f);
	m_phi_1 = props.float_("phi_1", 2.0246f);

	m_gamma_0 = to_gamma(m_phi_0);
	m_gamma_1 = to_gamma(m_phi_1);
	m_theta_d = Pi * 0.025f;
	m_D_theta = rcp(2.f * sin(m_theta_d));
	m_mu = props.float_("mu", -0.35f);
	m_H = props.float_("H", 0.5f);

	// derived parameters
	m_inv_b = rcp(m_b);
	m_b2 = sqr(m_b);

	// Cauchy function fits of ior
	if constexpr (is_spectral_v<Spectrum>) {
	    m_keratin_ior_cauchy = Spectrum(1.532f, 5.89e3f, 0.f, 0.f);
	    m_melanin_ior_cauchy = Spectrum(1.648f, 2.37e4f, 0.56f, 270.f);
	} else
	    Throw("Only spectral variants supported");

	auto pmgr = PluginManager::instance();
	Properties props_sampler("independent");
	props_sampler.set_int("sample_count", 4);
        m_sampler = static_cast<Sampler *>(pmgr->create_object<Sampler>(props_sampler));
    }

    /* helper function extracting bits */
    MTS_INLINE UInt32 helper_odd_even(UInt32 x) const {
	x = (x | x >> 1) & 0x33333333;
	x = (x | x >> 2) & 0x0f0f0f0f;
	x = (x | x >> 4) & 0x00ff00ff;
	x = (x | x >> 8) & 0x0000ffff;
	return x;
    }

    /* extract odd bits */
    MTS_INLINE UInt32 odd(UInt32 x) const {
	return helper_odd_even(x & 0x55555555);
    }

    /* extract even bits */
    MTS_INLINE UInt32 even(UInt32 x) const {
	return helper_odd_even((x & 0xaaaaaaaa) >> 1);
    }

    // /* compute x / 2^n */
    // MTS_INLINE Float rpow2(Float x, UInt32 n) const {
    // 	UInt32 bx;
    // 	memcpy(&bx, &x, sizeof(Float));
    // 	UInt32 exponent = bx & 0x7f800000;   // extract exponent bits 30..23
    // 	exponent -= (n << 23);               // subtract n from exponent
    // 	bx = bx & ~0x7f800000 | exponent;    // insert modified exponent back into bits 30..23
    // 	Float result;
    // 	memcpy(&result, &bx, sizeof(Float));
    // 	return result;
    // }

    /* convert one random sample to two random samples */
    MTS_INLINE std::pair<Point2f, Point2f> expand_sample(Point2f n) const {
	UInt32 x = n.x() * (1 << 23);
	UInt32 x1 = odd(x) >> 1;
	UInt32 x2 = even(x);
	Point2f n1 = 0.00048828125 * Point2f(x1, x2); /* 2^-11 */

	UInt32 y = n.y() * (1 << 23);
	UInt32 y1 = odd(y) >> 1;
	UInt32 y2 = even(y);
	Point2f n2 = 0.00048828125 * Point2f(y1, y2);

	return {n1, n2};
    }

    MTS_INLINE Float dir_theta(const Vector3f& w) const {
	return atan2(w.y(), sqrt(sqr(w.x()) + sqr(w.z())));
    }

    MTS_INLINE Float dir_gamma(const Vector3f& w) const {
	return atan2(w.x(), w.z());
    }

    MTS_INLINE std::pair<Float, Float> dir_sph(const Vector3f& w) const {
	return std::make_pair(dir_theta(w), dir_gamma(w));
    }

    MTS_INLINE Vector3f sph_dir(Float theta, Float gamma) const {
	auto [sin_theta, cos_theta] = sincos(theta);
	auto [sin_gamma,   cos_gamma]   = sincos(gamma);
	return Vector3f(sin_gamma * cos_theta, sin_theta, cos_gamma * cos_theta);
    }

    /* Sample a point on a 2D normal distribution. Internally uses the Box-Muller transformation */
    MTS_INLINE Point2f square_to_normal(const Point2f &sample, Float sigma, Point2f mu) const {
	Float r   = sqrt(-2.f * log(1.f - sample.x())) * sigma,
	    gamma = 2.f * Pi * sample.y();

	auto [s, c] = sincos(gamma);
	return { c * r + mu.x(), s * r + mu.y()};
    }

    MTS_INLINE Spectrum get_spectrum(const SurfaceInteraction3f &si) const {
	Spectrum wavelengths;
	if constexpr (is_spectral_v<Spectrum>) {
	    wavelengths[0] = si.wavelengths[0]; wavelengths[1] = si.wavelengths[1];
	    wavelengths[2] = si.wavelengths[2]; wavelengths[3] = si.wavelengths[3];
	} else
	    Throw("Only spectral variants supported");
	return wavelengths;
    }

    /* evaluate ior at current wavelengths given cauchy variables */
    MTS_INLINE Complex<Spectrum> eval_ior(Spectrum wavelengths,
					  const Spectrum& cauchy_vars) const {
	return Complex<Spectrum>(fmadd(cauchy_vars[1], rcp(sqr(wavelengths)), cauchy_vars[0]),
				 cauchy_vars[2] * exp(-wavelengths * rcp(cauchy_vars[3])));
    }

    /* Compute the reflectance and transmittance off a thin film */
    MTS_INLINE std::tuple<Spectrum, Spectrum> airy(Spectrum wavelengths, Spectrum ct_i,
						   Spectrum& height) const {
	Spectrum film_ior = eval_ior(wavelengths, m_keratin_ior_cauchy).x();
        Spectrum ct_t = safe_sqrt(1.f - (1.f - sqr(ct_i)) * sqr(Spectrum(m_ext_ior) / film_ior));

	/* Amplitudes of reflected and refracted waves */
	auto [r12s, r12p, t12s, t12p] = fresnel_dielectric(Spectrum(m_ext_ior), film_ior, ct_i);
	auto [r23s, r23p, t23s, t23p] = fresnel_dielectric(film_ior, Spectrum(m_ext_ior), ct_t);

	/* optical path difference and phase difference */
	Spectrum OPD = 2e3f * height * film_ior * ct_t;
	Spectrum d_gamma = TwoPi * OPD * rcp(wavelengths);

	Complex<Spectrum> exp_d_gamma = exp(Complex<Spectrum>(0.f, -d_gamma));

	Complex<Spectrum> rs = r12s + t12s * r23s * t23s *
	    rcp(exp_d_gamma - sqr(r23s));
	Complex<Spectrum> rp = r12p + t12p * r23p * t23p *
	    rcp(exp_d_gamma - sqr(r23p));

	Spectrum R = 0.5f * (squared_norm(rs) + squared_norm(rp));
	R = max(min(R, Spectrum(1.f)), Spectrum(0.f));

	Spectrum T = 1.f - R;
	return {R, T};
    }

   /* Compute the trt component of a thin film */
    MTS_INLINE Spectrum airy_trt(Spectrum wavelengths, Spectrum ct_i, Spectrum ct_o,
				 Spectrum R23, Spectrum& height) const {
	Spectrum film_ior = eval_ior(wavelengths, m_keratin_ior_cauchy).x();
        Spectrum ct_it = safe_sqrt(1.f - (1.f - sqr(ct_i)) * sqr(Spectrum(m_ext_ior) / film_ior));
	Spectrum ct_ot = safe_sqrt(1.f - (1.f - sqr(ct_o)) * sqr(Spectrum(m_ext_ior) / film_ior));

	/* Amplitudes of reflected and refracted waves */
	auto [r01s, r01p, t01s, t01p] = fresnel_dielectric(Spectrum(m_ext_ior), film_ior, ct_i);
	auto [r10s, r10p, t10s, t10p] = fresnel_dielectric(film_ior, Spectrum(m_ext_ior), ct_it);
	auto [r01s_, r01p_, t01s_, t01p_] = fresnel_dielectric(Spectrum(m_ext_ior), film_ior, ct_o);
	auto [r10s_, r10p_, t10s_, t10p_] = fresnel_dielectric(film_ior, Spectrum(m_ext_ior), ct_ot);

	/* optical path difference and phase difference */
        Spectrum OPD = 2e3f * height * film_ior * ct_it;
	Spectrum d_gamma = TwoPi * OPD * rcp(wavelengths);
	Complex<Spectrum> exp_d_gamma = exp(Complex<Spectrum>(0.f, d_gamma));

	Spectrum OPD_ = 2e3f * height * film_ior * ct_ot;
	Spectrum d_gamma_ = TwoPi * OPD_ * rcp(wavelengths);
	Complex<Spectrum> exp_d_gamma_ = exp(Complex<Spectrum>(0.f, d_gamma_));

	Complex<Spectrum> ts = t01s * t10s * rcp(1.f - r10s * r10s * exp_d_gamma);
	Complex<Spectrum> tp = t01p * t10p * rcp(1.f - r10p * r10p * exp_d_gamma);

	Complex<Spectrum>  trts = ts * t01s_ * t10s_ * rcp(1.f - r10s_ * r10s_ * exp_d_gamma_);
	Complex<Spectrum>  trtp = tp * t01p_ * t10p_ * rcp(1.f - r10p_ * r10p_ * exp_d_gamma_);

	Spectrum TRT = 0.5f * (squared_norm(trts) + squared_norm(trtp));
	TRT = R23 * max(min(TRT, Spectrum(1.f)), Spectrum(0.f));

	return TRT;
    }

    /* sample normal direction given incident direction
     * returns the sampled normal, the probability sampling the normal
     * and the transmission, the probability sampling r or trt component
     */
    MTS_INLINE std::tuple<Normal3f, Float, Float, Float>
    distr_barbule(Vector3f wi, Point2f &sample1, const Float H, const Float mu) const {
	auto [theta_i, gamma_i] = dir_sph(wi);
	// compute valid azimuthal sample range
	auto [gamma_h0, gamma_h1, phi_h0, phi_h1] = find_valid_gamma(gamma_i, H, mu);
	Point2f pa = rotate(to_p(gamma_h0), mu),
	        pe = rotate(to_p(gamma_h1), mu);
	/* a point that is guaranteed outside of the ellipse */
	Point2f h0 = 1.1f * Point2f(cos(gamma_i), sin(gamma_i));
	/* vector perpendicular to gamma_i */
	Vector2f dp = Vector2f(cos(gamma_i - HalfPi), sin(gamma_i - HalfPi));
	Float h_min = dot(dp, pe),
	      h_max = dot(dp, pa);
	Float dh_r = h_max - h_min;

	Float r_trt = max(min(dh_r / (H * abs(cos(gamma_i))), 1.f), 0.f);

	/* Sample azimuthal component */
	Float h = sample1.x() * dh_r + h_min;
	Point2f ph = h0 + dp * h;
	Float phi_h = to_phi(intersect(ph, gamma_i + Pi, Point2f(0.f), mu));

	/* Sample longitudinal component */
	Float theta_max = min(m_theta_d, -m_theta_d + 2 * theta_i + Pi);
	Float theta_min = max(-m_theta_d, m_theta_d + 2 * theta_i - Pi);
	Float l_max = sin(theta_i - theta_min);
	Float l_min = sin(theta_i - theta_max);
	Float l = sample1.y() * (l_max - l_min) + l_min;
	Float theta_h = theta_i - asin(l);

	Mask active = dh_r > 0;

	/* sampled normal */
	Normal3f m = select(active, sph_dir(theta_h, phi_h + mu), 0.f/0.f);

	/* Evaluate NDF */
	Float D = select(active, NDF(theta_h, phi_h, H), 0.f/0.f);

	if (abs(gamma_i) > HalfPi) { /* below surface */
	    gamma_i -= sign(gamma_i) * Pi;
	    assert(abs(gamma_i) < HalfPi);
	    auto [gamma_h0, gamma_h1, phi_h0, phi_h1] = find_valid_gamma(gamma_i, H, mu);
	    pa = rotate(to_p(gamma_h0), mu);
	    pe = rotate(to_p(gamma_h1), mu);
	    dp = Vector2f(cos(gamma_i - HalfPi), sin(gamma_i - HalfPi));
	    h_min = dot(dp, pe);
	    h_max = dot(dp, pa);
	    dh_r = h_max - h_min;
	}

	Float t = max(min(1.f - dh_r / (H * abs(cos(gamma_i))), 1.f), 0.f);

	return {m, D, t, r_trt};
    }

    /* sample normal direction given incident and outgoing direction
     * returns the sampled normal, the probability sampling the normal,
     * the transmission, the probability sampling r or trt component and sample weight
     * this sampling procedure has lower rejection rate
     */
    MTS_INLINE std::tuple<Normal3f, Float, Float, Float, Float>
    distr_barbule(Vector3f wi, Vector3f wo, Point2f &sample1, const Float H, const Float mu) const {
	auto [theta_i, gamma_i] = dir_sph(wi);
	auto [theta_o, gamma_o] = dir_sph(wo);
	// compute valid azimuthal sample range
	auto [gamma_h0_i, gamma_h1_i, phi_h0_i, phi_h1_i] = find_valid_gamma(gamma_i, H, mu);
	auto [gamma_h0_o, gamma_h1_o, phi_h0_o, phi_h1_o] = find_valid_gamma(gamma_o, H, mu);

	/* a point that is guaranteed outside of the ellipse */
	Point2f h0 = 1.1f * Point2f(cos(gamma_i), sin(gamma_i));
	/* vector perpendicular to gamma_i */
	Vector2f dp = Vector2f(cos(gamma_i - HalfPi), sin(gamma_i - HalfPi));

	/* visible in incoming direction */
	Point2f pa = rotate(to_p(gamma_h0_i), mu),
	    pe = rotate(to_p(gamma_h1_i), mu);
	Float h_min = dot(dp, pe),
	    h_max = dot(dp, pa);
	Float dh_r = h_max - h_min;

	Float r_trt = max(min(dh_r / (H * abs(cos(gamma_i))), 1.f), 0.f);

	/* visibility in outgoing direction */
	Float gamma_h0 = fmax(gamma_h0_i, gamma_h0_o);
	Float gamma_h1 = fmin(gamma_h1_i, gamma_h1_o);

	Point2f pa_ = rotate(to_p(gamma_h0), mu),
	    pe_ = rotate(to_p(gamma_h1), mu);
	Float h_min_ = dot(dp, pe_),
	      h_max_ = dot(dp, pa_);
	Float dh_r_ = h_max_ - h_min_;

	/* Sample azimuthal component */
	Float h = sample1.x() * dh_r_ + h_min_;
	Point2f ph = h0 + dp * h;
	Float phi_h = to_phi(intersect(ph, gamma_i + Pi, Point2f(0.f), mu));

	/* Sample longitudinal component */
	Float theta_max = min(m_theta_d, -m_theta_d + 2 * theta_i + Pi);
	Float theta_min = max(-m_theta_d, m_theta_d + 2 * theta_i - Pi);
	Float l_max = sin(theta_i - theta_min);
	Float l_min = sin(theta_i - theta_max);
	Float l = sample1.y() * (l_max - l_min) + l_min;
	Float theta_h = theta_i - asin(l);

	Mask active = dh_r > 0 && dh_r_ > 0 && gamma_h0 < gamma_h1;

	/* sampled normal */
	Normal3f m = select(active, sph_dir(theta_h, phi_h + mu), 0.f/0.f);

	/* Evaluate NDF */
	Float D = select(active, NDF(theta_h, phi_h, H), 0.f/0.f);

	if (abs(gamma_i) > HalfPi) { /* below surface */
	    gamma_i -= sign(gamma_i) * Pi;
	    assert(abs(gamma_i) < HalfPi);
	    auto [gamma_h0, gamma_h1, phi_h0, phi_h1] = find_valid_gamma(gamma_i, H, mu);
	    pa = rotate(to_p(gamma_h0), mu);
	    pe = rotate(to_p(gamma_h1), mu);
	    dp = Vector2f(cos(gamma_i - HalfPi), sin(gamma_i - HalfPi));
	    h_min = dot(dp, pe);
	    h_max = dot(dp, pa);
	    dh_r = h_max - h_min;
	}

	/* transmittance */
	Float t = max(min(1.f - dh_r / (H * abs(cos(gamma_i))), 1.f), 0.f);

	Float weight = select(active, dh_r_ / dh_r, 0.f);

	return {m, D, t, r_trt, weight};
    }

    /* check visibility in incident and outgoing directions */
    MTS_INLINE Mask G(Normal3f m, Vector3f wi, Vector3f wo, const Float H, const Float mu) const {
	auto [theta_i, gamma_i] = dir_sph(wi);
	auto [theta_h, gamma_h] = dir_sph(m);
	auto [theta_o, gamma_o] = dir_sph(wo);

	Float phi_h = gamma_h - mu;
	Point2f c = rotate(to_p(to_gamma(phi_h)), mu);

	Point2f neighbor(0.f, H);

	Mask Gi = (!enoki::isfinite(intersect(c, gamma_i, neighbor, mu)) &&
		   !enoki::isfinite(intersect(c, gamma_i, -neighbor, mu)));
	Mask Go = (!enoki::isfinite(intersect(c, gamma_o, neighbor, mu)) &&
		   !enoki::isfinite(intersect(c, gamma_o, -neighbor, mu)));

	return (Gi && Go && (dot(m, wi) > 0.f) && (dot(m, wo) > 0.f));
    }

    // si.wi camera
    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float sample1,
                                             const Point2f &sample2,
                                             Mask active) const override {
	Throw("sample() not implemented for BSDF visualizer.");
    }

    // internal called _eval
    Spectrum _eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
		   const Vector3f &wi, const Vector3f& wo, Mask active) const {
	MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

	auto [theta_i, gamma_i] = dir_sph(wi);
	auto [theta_o, gamma_o] = dir_sph(wo);

        // TODO: below surface
	Float cos_theta_i = abs(Frame3f::cos_theta(wi)),
 	      cos_theta_o = abs(Frame3f::cos_theta(wo));

	Normal3f m = normalize(wi + wo);

	auto [theta_h, gamma_h] = dir_sph(m);

	auto [H, mu, height] = get_uvw(si);

	Float phi_h = gamma_h - mu;

	// R component
	/* Evaluate the Fresnel factor */
	Spectrum hi = height;
	Spectrum wavelengths = get_spectrum(si);
	auto [I, _] = airy(wavelengths, abs(dot(wi, m)), hi);
	/* Evaluate NDF */
	Float D = NDF(theta_h, phi_h, H);
	UnpolarizedSpectrum R = I * D * 0.25f / cos_theta_i;
	/* check visibility */
	Mask active_r = active && G(m, wi, wo, H, mu) && (D > 0.f);

        // TRT component
	// sample normal direction
	Point2f sample1 = const_cast<Sampler&>(*m_sampler).next_2d(active);
	Normal3f m_trt;
	Float t, r_trt, weight;
	std::tie(m_trt, std::ignore, t, r_trt, weight) = distr_barbule(wi, wo, sample1, H, mu);

	Mask active_trt = active && G(m_trt, wi, wo, H, mu);

	/* Melanin layer */
	UnpolarizedSpectrum R_ = m_reflectance_melanin->eval(si, active);
	UnpolarizedSpectrum TRT = airy_trt(wavelengths, abs(dot(m_trt, wi)), abs(dot(m_trt, wo)),
					   R_, hi) * math::InvPi<Float> * cos_theta_o * weight;

	return select(active_r, R * r_trt, 0.f) + select(active_trt, TRT * r_trt, 0.f);
    }

    // evaluate bsdf
    // wo illumination, si.wi camera
    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
	return _eval(ctx, si, wo, si.wi, active);
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
	return 0.f;
    }

    Spectrum eval_null_transmission(const SurfaceInteraction3f & si,
				    Mask active) const override {
        return unpolarized<Spectrum>(1.f);
    }

    void traverse(TraversalCallback *callback) override {
	Base::traverse(callback);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "BarbuleBsdfEval[" << std::endl
            << "  height = " << string::indent(m_height) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
protected:
    Float m_mu, m_H;
    Float m_height;
    ScalarFloat m_ext_ior;
    ref<Texture> m_film_ior;
    Float m_theta_d;
    Float m_sigma, m_b, m_b2, m_inv_a, m_inv_b,
	m_gamma_0, m_gamma_1, m_phi_0, m_phi_1,
	m_D_theta;
    Spectrum m_wavelengths;
    ref<Texture> m_reflectance_melanin;
    ref<Sampler> m_sampler;
    Spectrum m_melanin_ior_cauchy, m_keratin_ior_cauchy;
};

MTS_IMPLEMENT_CLASS_VARIANT(BarbuleBsdfEval, BSDF)
NAMESPACE_END(mitsuba)
