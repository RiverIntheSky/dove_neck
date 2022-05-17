# Overview
This repository includes the implementation of paper [**Rendering Iridescent Rock Dove Neck Feathers**](https://doi.org/10.1145/3528233.3530749) in [Mitsuba 2](https://mitsuba2.readthedocs.io/en/latest/index.html).
# Instruction
Copy the [Mitsuba 2](https://mitsuba2.readthedocs.io/en/latest/index.html) repository, then compile as instructed in the docs. The default variant is `scalar_spectral`.
Add these two lines to `src/bsdfs/CMakeLists.txt`:
```cmake
add_plugin(barbulebsdfev        barbuleBsdfEval.cpp)
add_plugin(barbulebsdf          barbule.cpp)
```
Add to `src/sensors/CMakeLists.txt`:
```cmake
add_plugin(bsdfvisualizer   bsdfvisualizer.cpp)
```
Add to `src/shapes/CMakeLists.txt`:
```cmake
add_plugin(feather     feather.cpp)
```
Then copy the current folder to the mitsuba folder, keeping the structure, then compile again.
# Usage
in the `build` folder, do `dist/mitsuba ../scenes/xxx.xml`
for the sequence with linear light source, do
```bash
for o in -1 00 01 02 03 04 05 06
do
    dist/mitsuba -Do=${o} -o ../scenes/la${o}0.exr ../scenes/la.xml
done
```
# Note
In Mitsuba 2, \omega_i denotes the camera ray and \omega_o the light ray. Therefore, when visualizing the BSDF, we use the `barbulebsdfev` plug-in, swapping \omega_i and \omega_o.
