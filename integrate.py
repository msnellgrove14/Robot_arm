import pinocchio as pin ##type: ignore
import numpy as np

def rk4_step(q, v, tau, model, data, dt, friction_fn=None):
    def accel(qi, vi):
        tau_net = tau if friction_fn is None else tau - friction_fn(vi)
        return pin.aba(model, data, qi, vi, tau_net)

    # k1: slope at the start of the step
    k1_v = accel(q, v)
    k1_q = v

    # k2: slope at the midpoint, using k1 to get there
    q2 = pin.integrate(model, q, 0.5 * dt * k1_q)
    v2 = v + 0.5 * dt * k1_v
    k2_v = accel(q2, v2)
    k2_q = v2

    # k3: slope at the midpoint again, using k2 to get there
    q3 = pin.integrate(model, q, 0.5 * dt * k2_q)
    v3 = v + 0.5 * dt * k2_v
    k3_v = accel(q3, v3)
    k3_q = v3

    # k4: slope at the end of the step, using k3 to get there
    q4 = pin.integrate(model, q, dt * k3_q)
    v4 = v + dt * k3_v
    k4_v = accel(q4, v4)
    k4_q = v4

    # weighted average of the 4 slopes (classic RK4 weights)
    dq = (dt / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
    dv = (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    q_new = pin.integrate(model, q, dq)
    v_new = v + dv

    return q_new, v_new


def wrap_to_pi(q):
    return (q + np.pi) % (2 * np.pi) - np.pi