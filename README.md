# Model reduction using manifold boundaries

This repository contains scripts for ion channel model reduction using the manifold boundary approximation method (MBAM) and parameter inference using [Myokit](http://myokit.org) and [PINTS](https://github.com/pints-team/pints) modules in Python. This code is associated with the paper:

***"Ion channel model reduction using manifold boundaries".*** (In preparation). Whittaker, D. G., Wang, J., Shuttleworth, J., Kemp, J. M., Venkateshappa, R., Claydon, T. W., Mirams, G. R.

## Prerequisites
It is recommended to install libraries and run this repository's scripts in a virtual environment to avoid version conflicts between different projects.
To create such a virtual environment open a terminal and do the following:
- Clone the repository (`git clone https://github.com/CardiacModelling/model-reduction-manifold-boundaries.git`)
- Type `cd model-reduction-manifold-boundaries` to navigate inside the repository
- Set up a virtual environment using `virtualenv --python=python3 venv`
- Activate the virtual environment using `source venv/bin/activate`
- Install the required packages by typing `pip install -r requirements.txt`.

When you are finished working with the repository, type `deactivate` to exit the virtual environment. The virtual environment can be used again with the `activate` command as shown above.

## Manifold boundary approximation method (MBAM)

All code relevant to the MBAM can be found in the [MBAM](https://github.com/CardiacModelling/model-reduction-manifold-boundaries/tree/main/MBAM) folder.

### Generating data

For each iteration of the MBAM, it is required to compute the geodesic path using the MBAM, and then to calibrate the reduced model. This can be performed by running the corresponding `compute_geodesics_i*.py` and `calibrate_model_i*.py` scripts. For example, to generate the data for the 5th iteration of the MBAM, type

- `python compute_geodesics_i5.py --ssv_threshold 1e-5` to compute the geodesic path
- `python calibrate_model_i5.py` to calibrate the reduced model which uses parameter values from the end of the geodesic path found in the previous script as the starting point

Exact input settings used to generate the data in the manuscript can be found in `input_settings_i*.csv` files in [MBAM/txt_files/](https://github.com/CardiacModelling/model-reduction-manifold-boundaries/tree/main/MBAM/txt_files) folder.

### Visualising data

To visualise completed iterations, the `--done` and `--plot` input arguments can be used. For example, to show diagnostic plots of the 5th iteration of the MBAM:

- `python compute_geodesics_i5.py --done --plot` will show plots of (1) the eigenvalue spectrum at the start of the geodesic path (before the model has been reduced), (2) dynamics of the state variables at the end of the geodesic path (in this case we can see that the occupancy of C3 is practically zero), (3) initial and final eigenvector components along the geodesic path, (3) parameter values and velocities along the geodesic path, and (4) how the current at the end of the geodesic path compares to reference system measurements from the full model
- `python calibrate_model_i5.py --done --plot` will show plots of the reduced and full model output (1) before and (2) after calibration to the full model output, and (3) dynamics of the state variables for the reduced model (in this case we can see that we have one fewer variable as the C3 state was removed)

### Toy model

The Michaelis Menten reaction kinetics "toy model" can be found in [MBAM/toy_model](https://github.com/CardiacModelling/model-reduction-manifold-boundaries/tree/main/MBAM/toy_model). To save related figures, simply type `python MMR_Plots.py`.

## Parameter inference using real data

All code for parameter inference using real experimental data can be found in the [Parameter_inference_real_data](https://github.com/CardiacModelling/model-reduction-manifold-boundaries/tree/main/Parameter_inference_real_data) folder.

### Generating data

To fit a model to experimental data simply type `python cmaesfit_iid_noise.py --model [MODEL NAME]` where `[MODEL NAME]` can be one of `wang`, `wang-r1`, `wang-r2`, `wang-r3`, `wang-r4`, `wang-r5`, `wang-r6`, `wang-r7`, `wang-r8`. For example, `python cmaesfit_iid_noise.py --model wang --repeats 50` to fit parameters of the original Wang model using 50 "repeats" or initial guesses, or `python cmaesfit_iid_noise.py --model wang-r5` to fit parameters of the reduced, Wang-r5 model.

### Visualising data

To visualise inferred parameters, in the [Parameter_inference_real_data/cmaesfits/](https://github.com/CardiacModelling/model-reduction-manifold-boundaries/tree/main/Parameter_inference_real_data/cmaesfits) folder type `python plot_params.py --model [MODEL NAME]`, e.g. `python plot_params.py --model wang` for the full Wang model. Adding the `--show` input argument will print figures to screen rather than saving them to file.

To visualise fits of the model to a voltage clamp protocol, in the [Parameter_inference_real_data/figures/](https://github.com/CardiacModelling/model-reduction-manifold-boundaries/tree/main/Parameter_inference_real_data/figures) folder simply type `python plot-[PROTOCOL NAME].py --model [MODEL NAME]` where `[PROTOCOL NAME]` can be one of `activation`, `inactivation`, `complex-ap`, `staircase`. For example, to visualise the fit of the Wang model to the staircase protocol data, simply type `python plot_staircase.py --model wang`. Again, adding the `--show` input argument will print figures to screen rather than saving them to file.

## Acknowledging this work

If you publish any work based on the contents of this repository please cite (PLACEHOLDER):

### Related publications

The experimental data in this work are taken from:

Kemp, J. M., Whittaker, D. G., Venkateshappa, R., Pang, Z., Johal, R., Sergeev, V., Tibbits, G. F., Mirams, G. R., Claydon, T. W.
(2021).
[Electrophysiological characterization of the hERG R56Q LQTS variant and targeted rescue by the activator RPR260243](https://doi.org/10.1085/jgp.202112923).
_Journal of General Physiology_ 153 (10): e202112923.
