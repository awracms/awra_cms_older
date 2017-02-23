﻿


# Australian Water Resources Assessment (AWRA) Community Modelling System #

Welcome to the AWRA Community Modelling System GitHub repository. This is a Modelling System supported by the Australian [Bureau of Meteorology](http://www.bom.gov.au) (the Bureau) and is used for Water Resource Assessment purposes.  

## Background##

The AWRA modelling system (AWRAMS) has been developed by the [Commonwealth Scientific and Industrial Research Organisation](http://www.csiro.au/) (CSIRO) and the Bureau towards fulfilling the Bureau's water reporting requirements specified in the [Water Act (2007)](http://www.bom.gov.au/water/regulations/waterAct2007AuxNav.shtml).

The operational AWRAMS has been used towards supplying retrospective water balance estimates published by the Bureau within:

 - [Water in Australia](http://www.bom.gov.au/water/waterinaustralia): an annual national picture of water availability and use in a particular financial year
 - [Water Resource Assessments](http://www.bom.gov.au/water/awra) produced prior to Water In Australia
 - [Regional water information](http://www.bom.gov.au/water/rwi) water resource assessments
 - [National Water Account](http://www.bom.gov.au/water/nwa): that provides an annual set of water accounting reports for ten nationally significant water resource management regions. Adelaide, Burdekin, Canberra, Daly, Melbourne, Murray–Darling Basin, Ord, Perth, South East Queensland and Sydney. 

The Bureau's operational implementation of AWRA-L (the landscape component of AWRA) outputs daily 0.05 degree gridded soil moisture, runoff, evapotranspiration, and deep drainage values. These are available from yesterday back to 1911 through the Australian Landscape Water Balance website [www.bom.gov.au/water/landscape](http://www.bom.gov.au/water/landscape).

![AWRA-L conceptual diagram](http://www.bom.gov.au/water/landscape/images/allFluxes.73826d3d.png "allFluxes.73826d3d.png")

AWRA-L is a one dimensional, 0.05° grid based water balance model over the Australian continent that has semi-distributed representation of the soil, groundwater and surface water stores. AWRA-L is a three soil layer (top: 0-10cm, shallow: 10cm-100cm, deep: 100cm-600cm), two hydrological response unit (shallow rooted versus deep rooted) model.


##	The AWRA Community Modelling System (AWRA CMS)##

**AWRA CMS Components**: The package includes the following components: 

 - **Simulation**: generating model outputs over a given spatial extent and time period according to specified input data
 - **Calibration**: automated alteration of model parameters to closely match observed
 - **Visualisation**: eg. viewing of gridded outputs and timeseries,    
 - **Benchmarking**: testing the model against observed data functionality
 - **Utilities:** common functionality required by the following four components

**Languages**: AWRA CMS is a [Python](https://www.python.org/) programming language based modelling system, with the core model algorithms implemented in high performing native languages (C) and generic functionality provided by robust, open source libraries (e.g. [NetCDF](http://www.unidata.ucar.edu/software/netcdf/), [HDF5](https://support.hdfgroup.org/HDF5/), [MPI](https://www.open-mpi.org/)). 

**Operating System:** The AWRA CMS system has been developed, tested and used on Linux OS. Installation guidance is provided for several flavours of Linux (latest versions of Ubuntu, Fedora, Centos, Scientific Linux, Debian, Mint and “Bash on Ubuntu on Windows”). 

**Interface**: The [Jupyter](http://jupyter.org/) interactive environment, run through a web browser, is used for running components of the modelling system. 

## Potential Users of AWRA CMS ##
   
The Community Modelling System has been released to the public to enable different applications (for example local/regional scenario assessment) and further development by external users.

Potential users are anticipated to predominately be interested in the ability to run the system with local data (including scenario modelling) and to modify the system with new capabilities. The potential collaborators have expressed a range of potential goals for their use of the Community Model, including performing comparisons with existing models, tailoring the hydrological performance to specific land uses and cropping types, and applying the model overseas.
	
Broadly speaking, there are four potential user categories of the AWRA Operational Model outputs and the AWRA Community Modelling System:

 - **Data user**: who accessing the model outputs through the Bureau's website
 - **Case study user**: who work with the Bureau to evaluate his/her case using 100 years of data
 - **Applying users**: who would primarily be interested in applying the current model to a region of interest using localised and/or scenario data where available, and
 - **Contributor users**: who will extend the capabilities of the model with new research and coding (modify the system with new capabilities)

It is expected that the majority of early adopters of the AWRA Community Modelling System will be **Applying users** looking to apply the system with local data/scenarios, with more **Contributor users** adopting the system as it becomes well known and established.

Importantly, the community modelling system is not intended to address the needs of users and organisations that simply need to consume the model outputs from the operational system (**Data user** and **Case study users** listed above). For these situations, and in general, the existing Bureau’s web site and registered data services will continue to be the point of truth for AWRA-L operational model outputs (email: [awrams@bom.gov.au](mailto:%20awrams@bom.gov.au)). 

## Licensing ##
By accessing or using the AWRAMS software, code, data or documentation, you agree to be bound by the AWRAMS licence (see [LICENCE.txt](https://github.com/awracms/awra_cms/blob/master/LICENSE.txt)).

**Registered Users** are required to sign and return to the Bureau the AWRAMS licence. 

Note that while source code is available and contributions from the public are welcome, the AWRA CMS is **not** [Open Source software](https://opensource.org/osd-annotated) or [Free Software](https://www.gnu.org/philosophy/free-sw.en.html) as its license is incompatible with these definitions.  You can check those links for more information.


User Registration
-----------------
There are two types of **Users** with the AWRA CMS: Users and **Registered Users**. 

Any **User** can access the AWRA CMS package, the User Manual and limited data via [GitHub](https://github.com/awracms/awra_cms) (this site), and are bound by the AWRAMS licence.  However, to gain access to the evaluation datasets a user must become an **AWRA CMS Registered User**. Reasons to become an AWRA CMS Registered User include:

 - **Access to Data:** complete calibration and benchmarking datasets (catchment-average time-series of streamflow, soil moisture and evapotranspiration, point estimates of soil moisture and evapotranspiration) and solar radiation "climatology" gridded dataset for input prior to 1990.
 - **Documentation**: 
	 - User manual
	 - Technical Model Description Report
	 - Benchmarking Report
 - **Ability to contribute to AWRA:** allowing the user to create a [fork](https://help.github.com/articles/fork-a-repo/) of github repository further allowing the opportunity to contribute to the ongoing development of the AWRA CMS code. Users who wish to submit enhancements of the modelling system to the Bureau are also required to agree to the AWRAMS Contributor License Agreement ([CONTRIBUTOR_LICENCE_AGREEMENT.txt]( https://github.com/awracms/awra_cms/blob/master/CONTRIBUTOR_LICENCE_AGREEMENT.txt)).
 - **Ongoing support**

To become a Registered User send an email to [awracms@bom.gov.au](mailto:%20awracms@bom.gov.au) and the Bureau will send you the licence agreement and instructions on how to access the datasets. 

**Climate Data**: Limited input climate data is supplied with the AWRA CMS.

## Contributing to AWRA CMS##

One of the main motivations of the community model is to gain the benefits of the AWRA CMS community contributing improvements to the system. It is anticipated that research scientists and agencies will be interested in:

 - **altering model algorithms** towards better performance
 - **increasing functionality** and
 - **adding new datasets** to the system. 

The suggested process for users who wish to contribute to the CMS is as follows:

1. The user [creates a fork](https://help.github.com/articles/fork-a-repo/)
   of the modelling system on their own GitHub account, and clones this
   repository to work from.
2. The user makes their desired changes, and pushes the changes to their 
   GitHub repository.  These changes should include documentation.
3. The user [makes a pull request](https://help.github.com/articles/creating-a-pull-request/)
   against the upstream (Bureau) repository.  This should include a rationale,
   document the testing that has been done, and describe any relevant tradeoffs.
4. The Bureau assesses that code scientifically and against various performance criteria including performance against benchmark data, system performance, code complexity and maintainability. If successful according to that testing the Bureau releases the new version of the CMS.
