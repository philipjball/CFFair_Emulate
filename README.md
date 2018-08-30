# Counterfactual Fairness (Kusner et al. 2017) Python Replication

Allows for replication of the Counterfactual Fairness paper results in Python using PyStan.

Options are:
* -do_l2: Performs the replication of the L2 (Fair K) model, which can take a while depending on computing power
* -save_l2: Saves the resultant models (or not) for the L2 (Fair K) model, which produces large-ish files (100s MBs)

Dependencies include:
* Python 3.5.5
* NumPy 1.14.3
* Pandas 0.23.0
* Scikit-learn 0.19.1
* PyStan 2.17.1.0
* StatsModels 0.9.0

To run with default settings (Perform L2 tests, but don't save the Posterior Samples), type:
> python CounterFair_Emulate.py