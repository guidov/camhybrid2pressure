camhybrid2pressure
====
| Convert Community Earth System Model (CESM) Community Atmospheric Model (CAM) data from hybrid to pressure coordinates in Python
| Initial release

Compiling the cython module
---------------------------

module load gcc # if you need this to load gcc compiler

cd src
python setup.py build_ext --inplace

Then move the .so file to where the convert.py command is

Testing on Model data
---------------------
Besides time series from simulations can be extracted with the following command.
The conversion needs T, PS, PHIS, and coordinates

e.g. using NCO (or PyReshaper)
ncrcat -v Z3,T,PS,PHIS cesmrun.cam2.h0.19??-01.nc cesmrun.cam2.h0.Z3_T_PS_PHIS_hyb.nc
modify the filename in convert_camvarhyb2pres.py
python convert_camvarhyb2pres.py

TODO
----
-  parallelize, improve speed, use dask distributed, command line interface to accept arguments


