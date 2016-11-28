# Calibration with AWRA Modelling System

The `awrams.calibration` package provides supporting functionality for calibrating the model through automatic optimisers. This support includes reuseable objective functions, and functionality to aggregate model results to catchment scale for comparison against streamflow gauge data.

With this support functionality, it is possible to run calibration job using either an external optimisation package (such as PEST or Hydromad), or an implementation of Shuffled Complex Evolution (SCE) provided with the `awrams.calibration` package.

This document provides hints for setting up a calibration task using either PEST or the supplied implementation of SCE. This is not a complete tutorial, and it is targeted at advanced users who have a familiarity with both Python and with calibration using optimisation tools.

## General remarks

Regardless of which optimisation software used, you will need to configure a way for the the optimiser to run the model, with a given set of parameters, and to assess the results against some goodness of fit measure. This can be done by creating a bespoke Python script or function that sets the model parameters, based on input from the optimiser, runs the model over a relevant spatial and temporal domain, and then computes the goodness of fit measures for communication back to the optimiser.

## Calibration with built in SCE

The notebook `notebooks/Calibration_SCE.ipynb` demonstrates setting up a simple, one-catchment calibration problem using
the supplied SCE optimiser and the Nash Sutcliffe Coefficient.

This example uses the default settings for the AWRA-L model, aside from scalar parameters, which are calibrated.

## Calibration using PEST

THe notebook `notebooks/Calibration_PEST.ipynb` suggests an approach to calibrating the model using the optimisers available in PEST.

Broadly speaking, this will involve setting up

* A PEST case/control file describing the PEST configuration for the problem. This file will need to include:
  * a list of parameters to calibrate including details of their ranges, etc
  * a list of 'observations' to match. The observations can just be objective function ideal values
  * reference to a template file, through which PEST can modify model parameters
  * reference to an instruction file, which teaches PEST how to extract the observations from the model outputs.
  * a command line for running the model
* A template file, which includes tags for each model parameter to be modified by PEST. PEST will fill this file in with the values for a given simulation.
* A script to run the model. This can be the template file
* An instruction file, which describes a model output file and how PEST can extract observations from the outputs
