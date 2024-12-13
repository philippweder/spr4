# SPR4
This repository contains the code used for the simulations in the paper
_Optimal strokes for the 4-sphere swimmer at low Reynolds number in the regime of small deformations_. In particular,
the code reproducing all the figures is provided.

## File structure
The files are split into two subdirectories:
- The directory `src` contains the immediate source code, i.e.
  - a python module ``spr4`` with the main class ``Spr4`` in ``base.py``
  - geometric utility functions in ```geom_utils.py``` for bivector decompositions etc.
  - plotting utility functions in ```plot_utils.py```

- The directory ``case studies`` contains the code that reproduces the figures in the article:
  - ``convergence.py`` performs the convergence experiment
  - ```non_uniqueness.py``` plots the trajectories of two non-unique optimal control curves
  - ```trajectories.py``` plots three example trajectories

## Requirements
All requirements can be found in ``requirements.txt``.

To install them using pip, run the following command:

    pip3 install -r requirements.txt

## Contact information
- François Alouges, francois.alouges@polytechnique.edu
- Aline Lefebvre-Lepot, aline.lefebvre@polytechnique.edu
- Philipp Weder, philipp.weder@epfl.ch

## Licence
Licensed under the [MIT License](LICENSE)



