## Helio-Learning: Data Learned prediction of space weather

### Idea
The project will utilize Machine Learning to predict upper-atmospheric (ionospheric) electrical conductance in the aurora/polar regions. The auroral conductance is severely affected by space weather phenomena like solar flares and coronal mass ejections (CMEs).  This project will enable significant improvements in present predictive investigations into the physics and estimation of the auroral conductance.

### Problem
Global magnetohydrodynamic (MHD) models are used predominantly in space weather prediction. In almost all such global models, the estimation of the ionospheric conductance is conducted using outdated empirical models. This poses a massive challenge, when predicting quantities like the conductance in the aurora/polar region, which are driven mostly by electromagnetic interactions with charged particles, that MHD cannot predict accurately.

### Aim
We intend to use machine learning to predict this quantity using inputs from in-situ satellite data and ground-based instruments. This ML-based predictive model will be designed to take multiple inputs and return the conductance as an output. Once operational, the model will be installed into the University of Michigan’s Space Weather Modeling Framework (SWMF) to study improvements in space weather predictive skill.

### Data
| Quantity               | Dataset                                                    | Coverage    |
| ---------------------- |:----------------------------------------------------------:| -----------:|
| Field Aligned Currents | Assimilative Mapping of Ionospheric Electrodynamics (AMIE) | 2000 - 2010 |


### Labels and Features
1. Labels: **Hall Conductance**, **Pedersen Conductance**
2. Features: **Field Aligned Currents**, **Latitude**, **MLT**
