import numpy as np

# Identified friction coefficients from Table 6 (the paper's own optimized
# solution mu_hat, not the [37] comparison column) in:
#   Xu, Fan, Chen, Ng, Ang, Fang, Zhu, Zhao, "Dynamic Identification of the
#   KUKA LBR iiwa Robot With Retrieval of Physical Parameters Using Global
#   Optimization," IEEE Access, vol. 8, pp. 108018-108031, 2020.
#   DOI: 10.1109/ACCESS.2020.3000997
# Identified on the iiwa 14 R820 rather than our 7 R800. Same 7-joint
# architecture, but not numerically exact for this specific arm.
# Paper reports both fv and fc in units of N*m*s/rad; treated here as
# fv [N*m*s/rad] (viscous) and fc [N*m] (Coulomb), matching the standard
# tau_fric = Fv*v + Fc*sign(v) model.
FV = np.array([0.2968, 0.1638, 0.2853, 0.1000, 0.1000, 0.1000, 0.1000])
FC = np.array([0.2377, 0.5253, 0.0334, 0.1255, 0.0734, 0.1267, 0.1372])


def tau_fric(v, Fv=FV, Fc=FC):
    """Resistive joint friction torque: viscous + Coulomb."""
    return Fv * v + Fc * np.sign(v)
