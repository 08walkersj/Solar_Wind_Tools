[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13749003.svg)](https://doi.org/10.5281/zenodo.13749003)

# Solar Wind Tools

![alt text](https://github.com/08walkersj/Solar_Wind_Tools/blob/master/Development_Code/solar_wind.gif "Image Credit: NASA's Goddard Space Flight Center")

This is a repostiory dedicate to downloading and processing of solar wind data for space physics applications and studies.\
We base this code around the OMNI 1-min data but tools should have extended application.

OMNI: King, J. H. and Papitashvili, N. E. (2005). Solar wind spatial scales in and comparisons of hourly Wind and ACE plasma and magnetic field data. Journal of Geophysical Research: Space Physics, 110(A2), A02104. https://doi.org/10.1029/2004JA010649


## Please implement extra changes and suggestions in a fork and let us know
### Current Limitations
- We have only implemented the newell coupling function. Please add new ones such as Milan so others can use it.

## To do (list of possible new developments to do)
- Speed up omni download with parallel downloads. Rather than appending to HDF one file at a time batch download multiple files in parallel and append to the HDF file.
