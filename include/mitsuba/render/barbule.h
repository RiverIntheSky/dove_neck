#pragma once

#include <mitsuba/core/vector.h>
#include <enoki/complex.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * \brief Calculates the polarized Fresnel reflection and transmission amplitude
 * at a planar interface between two dielectrics
 *
 * \param eta_1
 *      Refractive index of the first medium
 *
 * \param eta_1
 *      Refractive index of the second medium
 *
 * \param cos_theta_i
 *     Cosine of the angle between the surface normal and the incident ray
 *
 * \return A tuple r_s, r_p, t_s, t_p) consisting of
 *
 *     r_s         reflection coefficient with s-polarization
 *
 *     r_p         reflection coefficient with p-polarization
 *
 *     t_s         transmission coefficient with s-polarization
 *
 *     t_p         transmission coefficient with p-polarization
 */
template <typename Spectrum>
std::tuple<Spectrum, Spectrum, Spectrum, Spectrum> fresnel_dielectric
    (Spectrum eta_1, Spectrum eta_2, Spectrum cos_theta_i) {
    auto outside_mask = cos_theta_i >= 0.f;

    Spectrum eta = eta_2 / eta_1,
	rcp_eta = eta_1 / eta_2,
	eta_it = select(outside_mask, eta, rcp_eta),
	eta_ti = select(outside_mask, rcp_eta, eta);

    /* Using Snell's law, calculate the squared sine of the
       angle between the surface normal and the transmitted ray */
    Spectrum cos_theta_t_sqr =
	fnmadd(fnmadd(cos_theta_i, cos_theta_i, 1.f), sqr(eta_ti), 1.f);

    /* Find the absolute cosines of the incident/transmitted rays */
    Spectrum cos_theta_i_abs = abs(cos_theta_i);
    Spectrum cos_theta_t_abs = safe_sqrt(cos_theta_t_sqr);

    /* Amplitudes of reflected and refracted waves */
    Spectrum r_s = fnmadd(eta_it, cos_theta_t_abs, cos_theta_i_abs) /
	fmadd(eta_it, cos_theta_t_abs, cos_theta_i_abs);

    Spectrum r_p = fmsub(eta_it, cos_theta_i_abs, cos_theta_t_abs) /
	fmadd(eta_it, cos_theta_i_abs, cos_theta_t_abs);

    Spectrum t_s = r_s + 1.f;

    Spectrum t_p = fmadd(r_p, eta_ti, eta_ti);

    return {r_s, r_p, t_s, t_p};
}

NAMESPACE_END(mitsuba)
