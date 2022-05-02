#include "barbuleBsdfEval.h"

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class Barbule : public BarbuleBsdfEval<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(BarbuleBsdfEval, m_height, m_mu, m_H, m_reflectance_melanin,
		    rotate, to_p, m_sampler, distr_barbule, dir_sph, find_valid_gamma,
		    G, NDF, airy, airy_trt, expand_sample, get_spectrum, get_uvw)
    MTS_IMPORT_TYPES(Texture, Sampler)
    static constexpr auto Pi        = math::Pi<Float>;
    static constexpr auto HalfPi    = math::HalfPi<Float>;
    Barbule(const Properties &props) : Base(props) {
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float sample1,
                                             const Point2f &sample2,
                                             Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

	auto [theta_i, gamma_i] = dir_sph(si.wi);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        BSDFSample3f bs = zero<BSDFSample3f>();

	auto [H, mu, height] = get_uvw(si);

	auto [s1, s2] = expand_sample(sample2);
	auto [m, D, t, r_trt] = distr_barbule(si.wi, s1, H, mu);

	// R
	/* Evaluate NDF */
	Spectrum wavelengths = get_spectrum(si);
	Spectrum hi = height;
	Float pdf_r = 0.25f * D  / cos_theta_i;
	Vector3f wo_r = reflect(si.wi, m);
	auto [I, _] = airy(wavelengths, abs(dot(si.wi, m)), hi);
	Mask active_r = pdf_r > 0.f && G(m, si.wi, wo_r, H, mu);

	// TRT
	Vector3f wo_trt = warp::square_to_cosine_hemisphere(s2);
	Float pdf_trt = warp::square_to_cosine_hemisphere_pdf(wo_trt);
	wo_trt = select(dot(m, wo_trt) > 0.f, wo_trt, -wo_trt);
	Spectrum R_ = m_reflectance_melanin->eval(si, active);
        UnpolarizedSpectrum TRT = airy_trt(wavelengths, abs(dot(m, si.wi)), abs(dot(m, wo_trt)),
	 				   R_, hi);
	Mask active_trt = G(m, si.wi, wo_trt, H, mu);

	// select lobe based on energy
	Float r = select(active_r, hmean(I) * r_trt, 0.f);
	Float trt = select(active_trt, hmean(TRT) * r_trt, 0.f);
	/* if D == 0, lights are transmitted or absorbed */
	Float total_energy = select(D > 0, r + trt + t, 1.f);

	sample1 *= total_energy;
	Mask selected_r = sample1 < r && active_r;
	Mask selected_trt = sample1 >= r && sample1 < (r + trt) && active_trt;
	Mask selected_t = active && !selected_r && ! selected_trt;

	bs.wo = select(selected_r, wo_r, select(selected_trt, wo_trt, -si.wi));
	bs.pdf               = select(selected_r, pdf_r * r,
				      select(selected_trt, pdf_trt * trt, t)) / total_energy;
	bs.eta = 1.f;
        bs.sampled_type = select(selected_t, UInt32(+BSDFFlags::Null),
				 UInt32(+BSDFFlags::GlossyReflection));
        bs.sampled_component = select(selected_t, UInt32(1), UInt32(0));

	UnpolarizedSpectrum value = select(selected_r, I / hmean(I), select(selected_trt, TRT / hmean(TRT), 1.f)) * total_energy;
	// TODO: the lower hemisphere??
        return {bs, select(active, value, 0.f)};
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        return this->_eval(ctx, si, si.wi, wo, active);
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);
        Float cos_theta_i = Frame3f::cos_theta(si.wi),
	    cos_theta_o = Frame3f::cos_theta(wo);

	active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

	auto [H, mu, height] = get_uvw(si);

	// R case
	Spectrum wavelengths = get_spectrum(si);
	Normal3f m =  normalize(si.wi + wo);
	/* Evaluate NDF */
	auto [theta_h, gamma_h] = dir_sph(m);
	Float D = NDF(theta_h, gamma_h - mu, H);
	Float pdf_r = 0.25f * D / abs(cos_theta_i);
	Mask active_r = active && G(m, si.wi, wo, H, mu) && D > 0.f;
	/* Evaluate the Fresnel factor */
	Spectrum hi = height;
	auto [I, _] = airy(wavelengths, abs(dot(si.wi, m)), hi);

	// TRT case
	/* sample normal direction */
	Point2f sample1 = const_cast<Sampler&>(*m_sampler).next_2d(active);
	Normal3f m_trt;
	Float t, r_trt, weight;
	std::tie(m_trt, std::ignore, t, r_trt, weight) = distr_barbule(si.wi, wo, sample1, H, mu);
	Mask active_trt = active && G(m_trt, si.wi, wo, H, mu);
        /* Melanin layer */
	UnpolarizedSpectrum R_ = m_reflectance_melanin->eval(si, active);
	UnpolarizedSpectrum TRT = airy_trt(wavelengths, abs(dot(m_trt, si.wi)), abs(dot(m_trt, wo)),
					   R_, hi);

	Float r = select(active_r, hmean(I) * r_trt, 0.f);
	Float trt = select(active_trt, hmean(TRT) * r_trt, 0.f);
	Float total_energy = r + trt + t;
	Float pdf_trt = warp::square_to_cosine_hemisphere_pdf(wo) * weight;

	return select(active_r, pdf_r * r / total_energy, 0.f) +
	    select(active_trt, pdf_trt * trt / total_energy, 0.f);
    }

    Spectrum eval_null_transmission(const SurfaceInteraction3f & si,
                                    Mask active) const override {
	auto [theta_i, gamma_i] = dir_sph(si.wi);
	auto [H, mu, height] = get_uvw(si);

	// compute valid azimuthal sample range
	auto [gamma_h0, gamma_h1, phi_h0, phi_h1] = find_valid_gamma(gamma_i, H, mu);
	Point2f pa = rotate(to_p(gamma_h0), mu),
  	        pe = rotate(to_p(gamma_h1), mu);

	/* vector perpendicular to gamma_i */
	Vector2f dp = Vector2f(cos(gamma_i - HalfPi), sin(gamma_i - HalfPi));
	Float h_min = dot(dp, pe),
	      h_max = dot(dp, pa);
	Float dh_r = h_max - h_min;

	if (dh_r == 0) { /* only transmission */
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

	return unpolarized<Spectrum>(select(active, t, 0.f));
    }

    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_VARIANT(Barbule, BSDF)
MTS_EXPORT_PLUGIN(Barbule, "Barbule BSDF")
NAMESPACE_END(mitsuba)
