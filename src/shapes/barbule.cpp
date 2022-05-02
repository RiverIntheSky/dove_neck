#include "barbule.h"

NAMESPACE_BEGIN(mitsuba)

/**!

.. _shape-barbule:

Barbule (:monosp:`barbule`)
----------------------------------------------------

.. pluginparameters::


 * - p0
   - |point|
   - Object-space starting point of the barbule's centerline.
     (Default: (0, 0, 0))
 * - p1
   - |point|
   - Object-space endpoint of the barbule's centerline (Default: (0, 0, 1))
 * - radius
   - |float|
   - Radius of the barbule in object-space units (Default: 1)
 * - flip_normals
   - |bool|
   -  Is the barbule inverted, i.e. should the normal vectors
      be flipped? (Default: |false|, i.e. the normals point outside)
 * - to_world
   - |transform|
   - Specifies an optional linear object-to-world transformation. Note that non-uniform scales are
     not permitted! (Default: none, i.e. object space = world space)

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/shape_barbule_onesided.jpg
   :caption: Barbule with the default one-sided shading
.. subfigure:: ../../resources/data/docs/images/render/shape_barbule_twosided.jpg
   :caption: Barbule with two-sided shading
.. subfigend::
   :label: fig-barbule

This shape plugin describes a simple barbule intersection primitive.
It should always be preferred over approximations modeled using
triangles. Note that the barbule does not have endcaps -- also,
its normals point outward, which means that the inside will be treated
as fully absorbing by most material models. If this is not
desirable, consider using the :ref:`twosided <bsdf-twosided>` plugin.

A simple example for instantiating a barbule, whose interior is visible:

.. code-block:: xml

    <shape type="barbule">
        <float name="radius" value="0.3"/>
        <bsdf type="twosided">
            <bsdf type="diffuse"/>
        </bsdf>
    </shape>
 */

MTS_EXPORT_PLUGIN(Barbule, "Barbule shape");
NAMESPACE_END(mitsuba)
