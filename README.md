# Solar Neutrino Physics

## Introduction

Since 2017 we have been studying physics related to solar neutrinos. With a background in engineering and a natural inclination to try to connect equations with real objects of experience, in our research of the subject we wrote various computer code segments in a `Python` `SciPy` `NumPy` `Matplotlib` environment to check correspondence with published experimental results (as well as to test our grasp of the material). Note: Download any pdf files and read them on your computer or other device; GitHub no longer displays pdf documents in the browser. 

Though we provide stand-alone `Python` scripts here, we developed in the `JupyterLab` context to facilitate writing math and text notes in `Markdown` and LaTeX in parallel with the `Python` code and code output or graphs (which displays within the notebook). We intend to make available that code, along with enough explanation to make it potentially useful, to any interested parties. We have benefited from the physics and computing articles of many and hope that our own work will also clear up particular questions or otherwise assist students of this subject matter.
## General requirements and computing environment
We have most recently tested the scripts using some or all of the following (particular scripts may use different subsets):
1. `Python 3.7.4`
2. `Numpy 1.17.2`
3. `Matplotlib 3.1.1`
4. `Scipy 1.3.1`
5. `Sympy 1.4`

We developed the code on a `Microsoft Windows 8` 2014 Dell laptop (64-bit dual-core, 6 GB RAM), but have been working in a `Linux-4.15.0` environment, i.e., a Dell Precision 3541 laptop (64-bit four-core, 4 GB RAM) running `Ubuntu 18.04.4` since April 2020. The `Jupyter Lab` host environment is version `1.1.4`. We acquired the software listed as an `Anaconda` distribution (on both machines/OS environments).

## electron survival probability vs production energy in solar plasma (graph)

### Python file 20190620MSWscanAllenergyNoSympy.py

See the **article** [Solar Neutrino Survival Probability Scan](https://github.com/AncientZygote/neutrinos/blob/master/20200508MSWallScan.pdf) for a discussion of some of the relevant physics (including why there is no appreciable level-jump probability and hyperlink citations where available to physics papers and texts used, as well as some of our related work) for the Python script: [*20190620MSWscanAllenergyNoSympy.py*](https://github.com/AncientZygote/neutrinos/blob/master/20190620MSWscanAllenergyNoSympy.py). 

#### description and sample output
This script (execute from a `Python` session or add a custom shebang line) calculates and graphs the electron neutrino survival probability at Earth detection for neutrinos produced in solar
plasma with matter potential (the Mikheyev, Smirnov, Wolfenstein or MSW effect). It should be noted that some of the tutorials available on the web do not use the correct equations and therefore do not reproduce the observations at Borexino or SNO.

The resulting graph output of the script as a `jpg` is:

![Scan](/20190620SolPsurvTo100Mev.jpeg)

We do offer a `Jupyter Lab` version of the code at [Jupyter Notebook MSW Scan](https://sites.google.com/view/bentleyphysics/MSWG1), which can be copied cell-by-cell into your `Jupyter notebook` environment. Because the notebook is displayed in `html` format at the website link given, it would not download as a notebook file `.ipynb` which could be directly opened in `Jupyter`. You could also copy code into a notebook using the `Python` script code. The `Python` script was in fact developed in a `Jupyter notebook` and extracted and tested as a `Python` stand-alone subsequently.
#### coding discussion
The script is well-commented explicitly and through variable and function naming (we necessarily try to make it possible to determine what we did at a later date). The **problem at hand** is to *scan an array of neutrino production energies and produce an array of electron neutrino survival probabilities*, governed by the relevant physics equations and electron number density in the solar plasma at the production point. 

The computational **problem is decomposed** into a set of `Numpy` `vectorize` **functions** operating on the **array data** in sequence, i.e., a sequence of functions invoked on progressively processed data. There is no attempt to minimize memory usage since the problem does not require it: Our goal is rather to make each processing step functionally distinct and explicit in purpose (we did check the impact on performance of `vectorize` below). The result of **vectorizing** a `Python` function is
* a function which accepts one or more `Numpy` arrays
* executes essentially an implicit `for`-loop over the operations specified in the function declaration, and
* returns an array. 

This approach makes for very clean code, if not necessarily high performance (but `Numpy` array functions are already higher performance than `Python` alone typically).
For example, the following function from the script (implementing equation 14.58 in the 2016 PDG Review of Particle Physics) accepts a sequence of production energies and returns the required electron number density in solar plasma to achieve MSW resonance at each energy (i.e., returns an array of electron number densities):
<pre>
def NRESF (energy_sequence):
    "scan neutrino energy array GeV->MeV, gen array of N_res MSW e/cm^3"
    return ( 6.56e6 * delta_m_sqr * cos2theta * N_0 / (energy_sequence * 1e3) )
NRESFvec = np.vectorize(NRESF)
</pre>

All of the iteration over the arrays is nicely hidden in the `Numpy` vectorizing mechanics wrapping the `Python` and implicit `Numpy` code. Because the energy sequence variable is an array, a `Numpy` so-called `ufunc` *could* be substituted for the infix notation operation, e.g., `(energy_sequence * 1e3)` could result in a direct `Numpy` vector multiply of a scalar `1e3` times each member of the vector (array) `energy_sequence`, i.e., `np.multiply(energy_sequence, 1e3)`. We `import numpy as np` so replace prefix module name `numpy` with `np` in the function invocation. We do not know for certain if the `vectorize` function actually recognizes the opportunity for that `ufunc` operation, as it primarily evaluates the `Python` function defined on successive elements of the input array(s), like the `Python` `map` function, which is essentially a `for` loop.
We checked the execution time of the vectorized `NRESFvec` function and observed 158 microsec. We wrote a non-vector version to compare:
<pre>
# pull all one-time constant manipulation out
the_constant_part = 6.56e6 * delta_m_sqr * cos2theta * N_0
# provide a scratchpad array to prevent time spent on numpy temporary array creating and deletion
scratchpad_nresf = np.zeros(n_E_steps, dtype='float64')
def NRESF_nv (energy_sequence):
    "scan neutrino energy array GeV->MeV, gen array of N_res MSW e/cm^3"
    np.multiply(energy_sequence, 1e3, out=scratchpad_nresf)
    np.reciprocal(scratchpad_nresf, out=scratchpad_nresf)
    return ( np.multiply(scratchpad_nresf, the_constant_part)  )

</pre>
The `NRESF_nv` non-vectorized function executed in 4.36 microseconds, 36 times faster using the available `ufunc` optimizations (probably could have used one more scratchpad array, but do not want to make a career of it). We did cheat a bit by pulling the arithmetic operations on constants out of the function, so we did that for the vectorized version and shaved off 37 microseconds, but still 27 times slower. Well, we do like the convenience of the vectorized form and performance is not an issue in this application. 
The resulting arrays of survival probability (dependent variable) and neutrino production energy (independent variable) are handed off to `Matplotlib` for graphing. The arrays can, of course, be inspected by various interactive methods and portions of the code retasked for other related purposes, there being a fairly fine granularity in the functional decomposition. The neutrino parameters and production conditions can be easily modified in the code as desired. The `Matplotlib` code is well-commented and should be useful as a point of departure for those unfamiliar with that graphing software. The image display from a `Python` run typically offers the viewer the opportunity to save the resulting graph in several file formats, e.g., `jpg`, `png`, etc.
## approach of neutrino mass eigenstates at MSW resonance (graph)

### Python file 20190716ZhouNoSympySymHamiltCODEonlyZexactOurSolar.py

See the **article** [Approach of eigenvalues in medium](https://github.com/AncientZygote/neutrinos/blob/master/20200926LZeigPath.pdf) for discussion of the relevant physics and the computer code provided.

See the **article** [Solar Neutrino Survival Probability Scan](https://github.com/AncientZygote/neutrinos/blob/master/20200508MSWallScan.pdf) for an incidental discussion of **level-jump probability** and hyperlink citations where available to physics papers and texts used, as well as some of our related work.

#### description and sample output
The Python script: [20190716ZhouNoSympySymHamiltCODEonlyZexactOurSolar.py](https://github.com/AncientZygote/neutrinos/blob/master/20190716ZhouNoSympySymHamiltCODEonlyZexactOurSolar.py) graphs the *static relationship* between possible solar electron production energies and the electron number density (*held constant* at production density and possible production energy varied) in terms of the corresponding values of the **mass eigenstates**. This is normally discussed in the context of the possibility of a mass eigenstate level jump at their **closest approach at MSW resonance**. See our article above (present here at `github`) discussion of the possibility of a level jump, where that is shown to be similar to the ratio of the off-diagonal elements and on-diagonal elements of a general two-state system matrix. Perhaps a look at the **graph the script produces** will help in understanding its purpose and the physics context:\

![eigenstates vs E](/20190716ZhouNoSymimg.png)

#### discussion
In the graph, the production electron density was held constant and the values that the neutrino mass eigenstates would have a different production energies was varied in order to show the fact that these states are at closest approach at MSW resonance. This does not correspond to the dynamic transit of the neutrino out of the Sun radially, where we instead would hold the production energy constant and vary the density that the neutrino transits on the way out to vacuum (perhaps we will our post code for that also eventually, the methods being identical). The case graphed is more applicable to understanding the possibility of a level jump, since the probability is greatest where the mass eigenstates are nearest one another. As we discuss in the article cited above, that probability also depends on the smoothness of the function describing the decrease of electron number density from solar core to surface, i.e., a discontuity increases the possibility of a jump (but in the case of the Sun this density decreases smoothly, more or less exponentially, so the probability of a jump is very low).
## code to calculate nuclear reaction rate for 8B solar neutrinos
### Python file 201908085NACREcalc8BprotColRateCODEonly.py
#### description and sample output
The Python script: [201908085NACREcalc8BprotColRateCODEonly.py](https://github.com/AncientZygote/neutrinos/blob/master/201908085NACREcalc8BprotColRateCODEonly.py) calculates the nuclear reaction rate of boron neutrinos in the Sun using equations and methods published in NACRE II: an update of the NACRE compilation . . .,” (2013), update for charged-particle-induced thermonuclear reaction rates for nuclei A < 16, [NACRE II](http://arxiv.org/abs/1310.7099v1)

Related quantities required for the calculation, e.g., estimated 8B projectile density, free proton projectile density and collision frequency, are also printed, along with the estimated reaction rate given the approximate abundances at the radial location in the Sun and temperature. That result is used to estimate a corresponding neutrino packet length were the collision interval setting that emission time boundary (we show elsewhere that this usual assumption in the literature cannot be correct, see [Bounds on Solar Neutrino Packet Lengths](https://sites.google.com/view/bentleyphysics/solar-neutrino-packet-length)). This code would be useful for a student of nuclear physics at undergraduate level, as well as the student interested in neutrinos (albeit neutrino packet length is rather a specialized area).

Again, we really cannot discuss the details of the physics underlying the code without using mathematical notation, so will prepare a pdf document with that discussion and post that here soon.
**Code output from the script** with the current parameter settings in the code from a Linux terminal session:

<pre>
dalton@dalton-Precision-3541:~$ python 201908085NACREcalc8BprotColRateCODEonly.py
Temperature 0.0149 GK
reduced mass (mu hat): 0.89490630 u
8B atomic mass (Krane): 8.024606 u
8B atom electron mass: 0.002743 u
8B nuclear mass minus electron mass: 8.021863 u
electron mass: 0.000549 u
proton mass: 1.0072765 u
simulated S factor for 8B p collisions: 5.2861 keV b
NACRE II E0 (Gamow peak) = 20.8616 keV
calculated N_A <sigma v> 8B->p: 9.0261e-13 cm^3 mol^-1 s^-1
integrated over E range 5.000 to 30.000 keV
calculated sigma v: 1.4988e-36 cm^3 s^-1
3He density: 3.4914e+20 cm^-3
7Be density: 5.8306e+19 cm^-3
8B resulting density: 6.9968e+16 cm^-3
nucleon projectile density: 5.4670e+25 cm^-3
estimated collision frequency p on 8B: 5.7332e+06 s^-1
estimated collision interval p on 8B 1.7442e-07 s
corresponding neutrino packet len c*t then 52.2908 m
cross check above integrated result:
tau: 48.6063
NA sigma v: 9.4129e-13 cm^3 mol^-1 s^-1
reduced mass: 0.8949 u
target charge: 5.0
energy: 20.8616 keV
eta at Gamow peak E0: 5.1578
eta * 2 * pi = 32.4075
tunneling probability: 8.4256e-15
</pre>
