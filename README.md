# Model reduction using manifold boundaries

This repository contains scripts for ion channel model reduction using the manifold boundary approximation method and parameter inference using [Myokit](http://myokit.org) and [PINTS](https://github.com/pints-team/pints) modules in Python. This code is associated with the paper:

***"Ion channel model reduction using manifold boundaries".*** (In preparation). Whittaker, D. G., Wang, J., Shuttleworth J., Venkateshappa, R., Claydon, T. W., Mirams, G. R.

## Prerequisites
It is recommended to install libraries and run this repository's scripts in a virtual environment to avoid version conflicts between different projects.
To create such a virtual environment open a terminal and do the following:
- Clone the repository (`git clone https://github.com/CardiacModelling/model-reduction-manifold-boundaries.git`)
- Type `cd model-reduction-manifold-boundaries` to navigate inside the repository
- Set up a virtual environment using `virtualenv --python=python3.7 venv`
- Activate the virtual environment using `source venv/bin/activate`
- Install the required packages by typing `pip install -r requirements.txt`.

When you are finished working with the repository, type `deactivate` to exit the virtual environment. The virtual environment can be used again with the `activate` command as shown above.

## Manifold boundary approximation method (MBAM) results

## Parameter inference using real data

## Acknowledging this work

If you publish any work based on the contents of this repository please cite (PLACEHOLDER):

Kemp, J. M., Whittaker, D. G., Venkateshappa, R., Pang, Z., Johal, R., Sergeev, V., Tibbits, G. F., Mirams, G. R., Claydon, T. W.
(2021).
[Electrophysiological characterization of the hERG R56Q LQTS variant and targeted rescue by the activator RPR260243](https://doi.org/10.1085/jgp.202112923).
_Journal of General Physiology_ 153 (10): e202112923.

### Related publications

The experimental data in this work are taken from:

Kemp, J. M., Whittaker, D. G., Venkateshappa, R., Pang, Z., Johal, R., Sergeev, V., Tibbits, G. F., Mirams, G. R., Claydon, T. W.
(2021).
[Electrophysiological characterization of the hERG R56Q LQTS variant and targeted rescue by the activator RPR260243](https://doi.org/10.1085/jgp.202112923).
_Journal of General Physiology_ 153 (10): e202112923.
