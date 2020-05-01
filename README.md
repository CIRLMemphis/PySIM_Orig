# Structured illumination microscopy in python using physics-guided NN

Images collected by 3D-SIM can be modeled as

`g = Hf`

where `H` is the forward model, `g` is the data of size `[X, Y, Z]` and `f` is the object of size `[2X, 2Y, 2Z]`.

**Inverse prolem:** given the data `g`, i.e. M sets of raw images of size [X, Y, Z], restore the object `f` of size [2X, 2Y, 2Z].

**Model-based optimization method:** one can solve the above inverse problem by solving the optimization problem:

f<sub>r</sub> = Ag := argmin<sub>f</sub> (||Hf - Ug|| + R(f))

where f<sub>r</sub> is the restoration, `U` is the upsampling operator/function, `R` is the regularization, and H<sup>-1</sup> is the inverse operator/function.

**Neural network method:** one can solve the above inverse problem by first training a neural network

(w<sub>t</sub>, b<sub>t</sub>) = min<sub>w,b</sub>(N<sub>w,b</sub>(g<sub>train</sub>) - f<sub>train</sub>)

where N<sub>w,b</sub> is some neural network taking `g`, i.e. M sets of raw images of size [X, Y, Z], as input and outputing object `f` of size [2X, 2Y, 2Z].

**Physics-guided Neural network method:** basically combining the two methods above.
