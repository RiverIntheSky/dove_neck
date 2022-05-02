#include <mitsuba/core/fwd.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/sensor.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sensor-bcsdfvisualizer:

BSDF visualizer (:monosp:`bsdfvisualizer`)
--------------------------------------------

.. pluginparameters::

 * - none

This sensor plugin implements BSDF visualizer, looking at the world center
from all spherical directions

To create a bsdf visualizer:

.. code-block:: xml
    :name: sphere-meter

    <sensor type="bsdfvisualizer">
        <film type="hdrfilm">
            <integer name="width" value="90"/>
	    <integer name="height" value="90"/>
	    <string name="pixel_format" value="rgb"/>
	    <string name="component_format" value="float32"/>
	    <rfilter type="gaussian"/>
        </film>
        <sampler type="ldsampler">
            <integer name="sample_count" value="64"/>
        </sampler>
    </sensor>

*/

MTS_VARIANT class BSDFvisualizer final : public Sensor<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Sensor, m_film)
    MTS_IMPORT_TYPES(Shape)

    BSDFvisualizer(const Properties &props) : Base(props) {
        if (props.has_property("to_world"))
            Throw("Found a 'to_world' transformation -- this is not allowed. ");
    }

    std::pair<RayDifferential3f, Spectrum>
    sample_ray_differential(Float time, Float wavelength_sample,
                            const Point2f & position_sample,
                            const Point2f & /*aperture_sample*/,
                            Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

	// 1. Spatial component
# if 0  // sperical coordinates
	Float theta = (position_sample.y() - 0.5f) * math::Pi<Float>;
	Float phi = (position_sample.x() - 0.5f) * math::TwoPi<Float>;

	auto [st, ct] = sincos(theta);
	auto [sp, cp] = sincos(phi);
	Vector3f local(sp * ct, st, cp * ct);
	Float r = 0.f; // irrelevant

	Point3f ori = local;
# else  // scatterometer coordinates
	/* map x, y to [-1, 1] */
        Float x = position_sample.x() * 2.f - 1.f;
	Float y = position_sample.y() * 2.f - 1.f;
	Float r = sqrt(sqr(x) + sqr(y));
	Float theta = r * math::HalfPi<Float>;
	Float phi = atan2(y, x);

	Point3f ori = math::sphdir(theta, phi);
# endif
        // 2. Sample directional component
	// r > 1 => invalid ray
	Vector3f dir = select(r > 1.f, math::Infinity<Point3f>, -ori);

        // 3. Sample spectrum
        auto [wavelengths, wav_weight] = sample_wavelength<Float, Spectrum>(wavelength_sample);
	wav_weight = select(r > 1.f, 0.f, wav_weight);
	/* sample_rgb_spectrum() in <mitsuba/core/spectrum.h>
	 * could return value larger than MTS_WAVELENGTH_MAX,
	 * causing ior = 0 because of eval() in src/spectra/uniform.cpp,
	 * causing NaN in fresnel_conductor() */
	wavelengths = min(max(wavelengths, MTS_WAVELENGTH_MIN), MTS_WAVELENGTH_MAX);

	RayDifferential3f ray(ori, dir, time, wavelengths);

	return std::make_pair(ray, wav_weight);
    }

    ScalarBoundingBox3f bbox() const override {
	// Return an invalid bounding box
	return ScalarBoundingBox3f(); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "BSDFvisualizer[" << std::endl
            << "  film = " << m_film << "," << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_VARIANT(BSDFvisualizer, Sensor)
MTS_EXPORT_PLUGIN(BSDFvisualizer, "BSDFvisualizer");
NAMESPACE_END(mitsuba)
