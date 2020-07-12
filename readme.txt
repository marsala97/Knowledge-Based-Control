-------------------------------------------------
    SC42050 Knowledge-based control systems
-------------------------------------------------
         REPRODUCIBILITY ASSIGNMENT

    - A Reinforcement Learning-Based Architecture for Fuzzy Logic Control


The code related to the reproducibility assignment is collected in the Python
files in this repository.

System requirements: Python 3.x  (might be compatible with Python 2.7, but it's not tested)
Dependencies:
    - gym
    - numpy
    - scipy
    - matplotlib


The following executables are related to the controllers
and tests performed on them:
    - run_ARIC.py       : Code related to the training of the ARIC controller
        NOTE: ARIC learning is unstable. Sometimes works and sometimes the random initial weights lead
              to dead-end solutions where the learning gets stuck, as described in the report. If the
              controller does not learn in the first 10 attempts, just re-run the file.
    - run_LQR.py        : Code related to the testing of the LQR controller
    - run_Q_agent.py    : Code related to the training of the Q-learning controller
    - run_sensitivity   : Code related to the sensitivity test performed on the ARIC

The following files contain functionality imported by the executables above:
    - ARIC_model.py            : The actual implementation of the ARIC controller as described by Berenji
    - membership_functions.py  : Definitions of the membership functions for ARIC inputs and outputs
    - fuzzy_set.py             : Handy class for representing and working with fuzzy functions
    - pole_cart.py             : Own implementation of the pole-cart model
    - utils.py                 : Small handy functions that reduce clutter code in the implementations
