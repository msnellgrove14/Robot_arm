"""Plot joint-space tracking error for FF+PD against FF+TVLQR on the same
reference trajectory and the same torque disturbance.

    python render_tracking_plot.py
"""

import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin

from simulators import simulate_lqr, simulate_pd
from trajectories import circular_trajectory

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, 'lbr_iiwa7_r800.urdf')
OUT = os.path.join(HERE, 'arm_tracking_pd_vs_lqr.png')

T, DT = 5.0, 0.002
DISTURBANCE = 2.0


def main():
    model = pin.buildModelFromUrdf(URDF)
    poses = circular_trajectory(0.25, T=T, dt=DT, center=np.array([0.45, 0.0, 0.5]))
    if isinstance(poses, tuple):
        poses = poses[-1] if not isinstance(poses[0], (list, np.ndarray)) else poses[0]

    print('running PD...')
    pd = simulate_pd(model, poses, T, DT, disturbance_magnitude=DISTURBANCE)
    print('running TVLQR...')
    lqr = simulate_lqr(model, poses, T, DT, disturbance_magnitude=DISTURBANCE)

    def err(r):
        e = np.asarray(r['errors'])
        return e if e.ndim == 1 else np.linalg.norm(e, axis=1)

    e_pd, e_lqr = err(pd), err(lqr)
    t = np.asarray(pd['t'])[: len(e_pd)]

    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=130)
    ax.plot(t, e_pd, lw=1.4, color='#d1495b', label='feedforward + PD')
    ax.plot(np.asarray(lqr['t'])[: len(e_lqr)], e_lqr, lw=1.4, color='#0f6d5f',
            label='feedforward + time-varying LQR')
    ax.set_yscale('log')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('joint-space tracking error [rad]')
    ax.set_title(
        f'Tracking the same circular path under identical torque disturbance '
        f'({DISTURBANCE} N·m)'
    )
    ax.grid(alpha=0.3, which='both')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT, facecolor='white')
    print(f'PD   final {e_pd[-1]:.4e}, max {e_pd.max():.4e}')
    print(f'LQR  final {e_lqr[-1]:.4e}, max {e_lqr.max():.4e}')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
