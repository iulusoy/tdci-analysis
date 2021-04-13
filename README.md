# Data analysis package - I. S. Ulusoy, January 2021
## Installation instructions
You can install the package using   
`pip install -i https://test.pypi.org/simple/ tdcianalyis`

## Requirements
The package requires 
```
numpy
pandas
seaborn
```
to run. Developers will also need the 
```
pytest
sphinx
recommonmark
```
packages for testing and building the documentation.

## Documentation
The documentation can be found [here](https://tdci-analysis.readthedocs.io/en/latest/).

## Features of the package

The data analysis package allows you to process tdci output data and analyse the results from your run. The main features are:
- Processing of `expect.t` data: This functionality allows you to read in `expect.t` data and plot relevant entries.
- Processing of `efield.t` data: You may read in and plot a laser pulse and generate it's Fourier transform to identify the width in frequency domain.
- Processing of `nstate_i.t` data: This will construct the autocorrelation function with respect to time `$t=0$` and plot the autocorrelation function. The Fourier transform of the autocorrelation function is generated and plotted, resulting in a spectrum. The values of the autocorrelation function will be printed to a file.
- Processing of `npop.t` data: Using the MO populations, the correlations between MO's are investigated and both printed out as well as plotted. High correlation means that two MO's exchange population.
- Procesing of `table.dat` data: If copied into a file `table.dat`, the L2 norm of the transition dipole moment in length and velocity gauge can be computed, in order to highlight the convergence of the initial wave function in Hilbert space (completeness). A complete wavefunction will result in identical transition dipole moments no matter which gauge, and the resulting L2 norm will be zero.

**Important:: Make sure you provide the correct input and output directories when using the modules!**
