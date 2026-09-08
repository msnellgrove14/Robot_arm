import time

import numpy as np
import pinocchio as pin

from integrate import rk4_step
from inverse_kinematics import solve_ik_trajectory
from spline import cubic_spline_interpolation
from control import pd_control, critically_damped_gains, tvlqr_backward_pass
from linearize import linearize_trajectory
from disturbances import random_torque
from prm_query import shortcut_path


def _reference_trajectory(model, data, joint_id, poses, T, dt, q_init):
    """IK the pose trajectory into joint space, then spline-fit it.
    Returns (t, q_ref, v_ref, a_ref, successes).
    """
    q_traj, successes = solve_ik_trajectory(model, data, joint_id, poses, q_init, print_every=0)
    t = np.linspace(0, T, round(T / dt) + 1)
    q_ref, v_ref, a_ref = cubic_spline_interpolation(t, q_traj)
    return t, q_ref, v_ref, a_ref, successes


def simulate_pd(
    model,
    poses,
    T,
    dt,
    joint_id=7,
    q_init=None,
    wn=20.0,
    zeta=1.0,
    disturbance_magnitude=0.0,
    viz=None,
    viz_every=10,
    real_time=False,
):
    """Feedforward (pin.rnea) plus PD tracking: reference pose trajectory to IK to
    spline to simulation with rk4_step, using per-joint critically damped gains
    and an optional random torque disturbance each step.

    With viz given and real_time=True, playback is paced to the trajectory's own
    timescale. At each displayed frame it sleeps only the gap between how much
    wall clock time should have passed and how much actually has, so it corrects
    for computation and viz.display network overhead instead of drifting.

    Returns a dict with t, q_ref, v_ref, q_sim (one row per step), errors
    (|q_sim - q_ref| per step) and successes (IK convergence per pose).
    """
    if q_init is None:
        q_init = pin.neutral(model)

    data = model.createData()
    t, q_ref, v_ref, a_ref, successes = _reference_trajectory(model, data, joint_id, poses, T, dt, q_init)

    data_sim = model.createData()
    q_sim = q_ref[0].copy()
    v_sim = v_ref[0].copy()
    Kp, Kd = critically_damped_gains(model, data_sim, q_ref[0], wn=wn, zeta=zeta)

    q_sim_traj = np.zeros_like(q_ref)
    errors = np.zeros(len(t))
    wall_start = time.perf_counter() if (viz is not None and real_time) else None

    for k in range(len(t)):
        tau_ff = pin.rnea(model, data_sim, q_ref[k], v_ref[k], a_ref[k])
        tau_fb = pd_control(q_sim, q_ref[k], v_sim, Kp, Kd)
        tau_dist = random_torque(model, magnitude=disturbance_magnitude)
        q_sim, v_sim = rk4_step(q_sim, v_sim, tau_ff + tau_fb + tau_dist, model, data_sim, dt)

        q_sim_traj[k] = q_sim
        errors[k] = np.linalg.norm(q_sim - q_ref[k])
        if viz is not None and k % viz_every == 0:
            viz.display(q_sim)
            if real_time:
                sleep_time = (wall_start + t[k]) - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

    return {
        "t": t, "q_ref": q_ref, "v_ref": v_ref,
        "q_sim": q_sim_traj, "errors": errors, "successes": successes,
    }


def simulate_lqr(
    model,
    poses,
    T,
    dt,
    joint_id=7,
    q_init=None,
    Q_pos_weight=100.0,
    R_weight=0.01,
    disturbance_magnitude=0.0,
    viz=None,
    viz_every=10,
    real_time=False,
):
    """Feedforward (pin.rnea) plus TVLQR tracking: reference pose trajectory to IK
    to spline, then linearize the whole trajectory, run the TVLQR backward pass,
    and simulate with rk4_step, with an optional random torque disturbance each
    step.

    With viz given and real_time=True, playback is paced to the trajectory's own
    timescale, correcting for computation and network overhead rather than
    drifting.

    Returns a dict with t, q_ref, v_ref, q_sim (one row per step), errors
    (|q_sim - q_ref| per step), successes (IK convergence per pose) and K_list,
    the TVLQR gains.
    """
    if q_init is None:
        q_init = pin.neutral(model)

    data = model.createData()
    t, q_ref, v_ref, a_ref, successes = _reference_trajectory(model, data, joint_id, poses, T, dt, q_init)

    data_sim = model.createData()
    nv = model.nv
    tau_traj = np.array([pin.rnea(model, data_sim, q_ref[k], v_ref[k], a_ref[k]) for k in range(len(t))])
    A_traj, B_traj = linearize_trajectory(model, data_sim, q_ref, v_ref, tau_traj, dt)

    Q = np.eye(2 * nv)
    Q[:nv, :nv] *= Q_pos_weight
    R = np.eye(nv) * R_weight
    K_list = tvlqr_backward_pass(A_traj, B_traj, Q, R, Q)

    q_sim = q_ref[0].copy()
    v_sim = v_ref[0].copy()
    q_sim_traj = np.zeros_like(q_ref)
    errors = np.zeros(len(t))
    wall_start = time.perf_counter() if (viz is not None and real_time) else None

    for k in range(len(t)):
        tau_ff = pin.rnea(model, data_sim, q_ref[k], v_ref[k], a_ref[k])
        x_sim = np.concatenate([q_sim, v_sim])
        x_ref = np.concatenate([q_ref[k], v_ref[k]])
        tau_fb = -K_list[k] @ (x_sim - x_ref)
        tau_dist = random_torque(model, magnitude=disturbance_magnitude)
        q_sim, v_sim = rk4_step(q_sim, v_sim, tau_ff + tau_fb + tau_dist, model, data_sim, dt)

        q_sim_traj[k] = q_sim
        errors[k] = np.linalg.norm(q_sim - q_ref[k])
        if viz is not None and k % viz_every == 0:
            viz.display(q_sim)
            if real_time:
                sleep_time = (wall_start + t[k]) - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

    return {
        "t": t, "q_ref": q_ref, "v_ref": v_ref,
        "q_sim": q_sim_traj, "errors": errors, "successes": successes,
        "K_list": K_list,
    }


def simulate_lqr_from_path(
    model,
    q_path,
    T,
    dt,
    Q_pos_weight=100.0,
    R_weight=0.01,
    disturbance_magnitude=0.0,
    viz=None,
    viz_every=10,
    real_time=False,
    use_shortcutting=False,
    capsules=None,
    excluded_pairs=None,
    spheres=None,
    max_step=0.1,
):
    """Same feedforward plus TVLQR pipeline as simulate_lqr, but starting from a
    joint-space path that is already known, such as a PRM query result, so no IK
    is needed. Knot times are assigned proportional to cumulative joint-space
    distance along the path, scaled to fit T, then spline fit and evaluated on a
    dense dt grid.

    use_shortcutting=True first prunes q_path with prm_query.shortcut_path, since
    a graph shortest path minimizes joint-space distance rather than task-space
    directness and can come out jagged. That needs capsules, from
    self_collision.build_link_capsules. Pass the same sphere list the roadmap was
    built with if the path also has to avoid external obstacles, otherwise
    shortcutting only checks self collision and could cut through one.

    Returns the same dict as simulate_lqr without `successes`, since no IK
    happens here.
    """
    q_path = np.asarray(q_path)
    data_sim = model.createData()

    if use_shortcutting:
        if capsules is None:
            raise ValueError("use_shortcutting=True requires capsules (see self_collision.build_link_capsules)")
        q_path = shortcut_path(q_path, model, data_sim, capsules, excluded_pairs, spheres, max_step=max_step)

    # drop consecutive near-duplicate waypoints, such as a chained path
    # can contain a node that happens to exactly coincide with an existing one
    # (a zero-length hop), which breaks the strictly-increasing knot-time
    # requirement CubicSpline needs below
    dedup_lengths = np.linalg.norm(np.diff(q_path, axis=0), axis=1)
    keep_mask = np.concatenate([[True], dedup_lengths > 1e-9])
    q_path = q_path[keep_mask]

    seg_lengths = np.linalg.norm(np.diff(q_path, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    t_knots = np.linspace(0, T, len(q_path)) if cumulative[-1] < 1e-9 else cumulative / cumulative[-1] * T

    t = np.linspace(0, T, round(T / dt) + 1)
    q_ref, v_ref, a_ref = cubic_spline_interpolation(t_knots, q_path, t_eval=t)

    nv = model.nv
    tau_traj = np.array([pin.rnea(model, data_sim, q_ref[k], v_ref[k], a_ref[k]) for k in range(len(t))])
    A_traj, B_traj = linearize_trajectory(model, data_sim, q_ref, v_ref, tau_traj, dt)

    Q = np.eye(2 * nv)
    Q[:nv, :nv] *= Q_pos_weight
    R = np.eye(nv) * R_weight
    K_list = tvlqr_backward_pass(A_traj, B_traj, Q, R, Q)

    q_sim = q_ref[0].copy()
    v_sim = v_ref[0].copy()
    q_sim_traj = np.zeros_like(q_ref)
    errors = np.zeros(len(t))
    wall_start = time.perf_counter() if (viz is not None and real_time) else None

    for k in range(len(t)):
        tau_ff = pin.rnea(model, data_sim, q_ref[k], v_ref[k], a_ref[k])
        x_sim = np.concatenate([q_sim, v_sim])
        x_ref = np.concatenate([q_ref[k], v_ref[k]])
        tau_fb = -K_list[k] @ (x_sim - x_ref)
        tau_dist = random_torque(model, magnitude=disturbance_magnitude)
        q_sim, v_sim = rk4_step(q_sim, v_sim, tau_ff + tau_fb + tau_dist, model, data_sim, dt)

        q_sim_traj[k] = q_sim
        errors[k] = np.linalg.norm(q_sim - q_ref[k])
        if viz is not None and k % viz_every == 0:
            viz.display(q_sim)
            if real_time:
                sleep_time = (wall_start + t[k]) - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

    return {
        "t": t, "q_ref": q_ref, "v_ref": v_ref,
        "q_sim": q_sim_traj, "errors": errors,
        "K_list": K_list,
    }
