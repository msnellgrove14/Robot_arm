import numpy as np
import pinocchio as pin


def pd_control(q, q_des, v, Kp, Kd, v_des=None):
    """PD joint-space controller: tau = Kp*(q_des - q) - Kd*(v - v_des).
    v_des defaults to 0 (pure damping) if not given. Kp/Kd may be scalars
    or per-joint arrays (size model.nv).
    """
    if v_des is None:
        v_des = np.zeros_like(v)
    return Kp * (q_des - q) - Kd * (v - v_des)


def critically_damped_gains(model, data, q, wn=20.0, zeta=1.0):
    """Per-joint Kp and Kd, scaled by each joint's own effective inertia, which is
    the diagonal of the joint-space mass matrix. This gives every joint the same
    natural frequency wn and damping ratio zeta instead of applying one scalar
    gain everywhere.

    A single scalar Kd does not work across this arm's inertia range. The wrist
    sits around 0.003 and the shoulder around 24, so a Kd stiff enough to hold
    the shoulder is numerically unstable on the wrist, where Kd*dt/I passes the
    explicit integration stability limit, while a Kd small enough for the wrist
    barely damps the shoulder at all. Dividing through by I makes Kd*dt/I equal
    to 2*zeta*wn*dt for every joint regardless of its inertia.
    """
    M = pin.crba(model, data, q)
    I = np.diag(M).copy()
    Kp = I * wn**2
    Kd = 2 * zeta * I * wn
    return Kp, Kd


def tvlqr_backward_pass(A_list, B_list, Q, R, Qf):
    """Finite-horizon discrete-time TVLQR backward (Riccati) pass.

    A_list, B_list: length-N sequences of per-timestep discrete Jacobians
    (e.g. from linearize.linearize_trajectory), A_k/B_k mapping x_k -> x_{k+1}.
    Q, R: state/control cost weights. Qf: terminal state cost (P_N).

    Returns K_list: length-N list of feedback gains, one per timestep, for
    u_k = u_ff_k - K_list[k] @ (x_k - x_ref_k).
    """
    N = len(A_list)
    n = A_list[0].shape[0]
    P = Qf
    K_list = [None]*N
    for k in reversed(range(N)):
        A_k, B_k = A_list[k], B_list[k]
        S = R + B_k.T @ P @ B_k
        K_k = np.linalg.solve(S, B_k.T @ P @ A_k)     # solve, not explicit inverse — better conditioned
        P = Q + A_k.T @ P @ A_k - A_k.T @ P @ B_k @ K_k
        K_list[k] = K_k
    return K_list