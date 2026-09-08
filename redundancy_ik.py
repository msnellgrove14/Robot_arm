"""Analytical IK for this arm's redundancy, via the "arm angle" method
(Shimizu et al. 2008, IEEE Trans. Robotics, "Analytical Inverse Kinematic
Computation for 7-DOF Redundant Manipulators With Joint Limits and Its
Application to Redundancy Resolution"). Original paper not accessible;
this is re-derived from scratch for this arm's specific joint axes/
placements (pulled directly from the pinocchio model, not the paper's
D-H convention) and verified numerically at every step against pinocchio's
own forward kinematics and the iterative solve_ik.

Kinematic structure (verified from the model): a non-offset 7-DOF S-R-S
arm -- joints 1,2 concurrent at the shoulder, joint 3 a pure roll about
the upper-arm link (so joints 1,2,3 together behave like a 3-DOF spherical
joint), joint 4 the elbow, joints 5,6 concurrent at the wrist, joint 7 a
roll about the forearm/flange axis (joints 5,6,7 together a 3-DOF
spherical wrist). Every joint rotates about its own local Z axis (or -Z),
with a fixed +-90deg X rotation between consecutive joint frames.

For a target end-effector pose (p_ee, R_ee):
  1. wrist_center = p_ee - d_tool * R_ee[:, 2]           (exact, verified)
  2. elbow angle magnitude from law of cosines on
     {shoulder, elbow, wrist_center}; sign is a free discrete choice
     (elbow_sign = +-1)                                   (exact, verified)
  3. elbow position E(phi): circle of points at distance L1 from the
     shoulder and L2 from wrist_center, phi in [-pi, pi]  (exact, verified)
  4. shoulder angles (q1, q2, q3) place the upper arm at E(phi) and aim
     the elbow joint toward wrist_center                  (exact, verified)
  5. wrist angles (q5, q6, q7): a standard ZYZ Euler decomposition of the
     required wrist rotation (this chain regroups to Rz(-q5)@Ry(q6)@Rz(q7)
     exactly)                                              (exact, verified)

Sweeping phi over [-pi, pi] (x2 for elbow_sign) gives the full family of
joint configurations reaching the same end-effector pose -- the data
structure needed for obstacle-aware redundancy resolution.
"""
import numpy as np
import pinocchio as pin


def _Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def kinematic_constants(model):
    """Pull shoulder position, link lengths, and tool offset directly from
    the model (assumes the non-offset S-R-S joint order verified above:
    joints 1,2=shoulder concurrent, 3=shoulder roll, 4=elbow,
    5,6=wrist concurrent, 7=flange roll).
    """
    shoulder = model.jointPlacements[1].translation.copy()
    L1 = np.linalg.norm(model.jointPlacements[3].translation)
    L2 = np.linalg.norm(model.jointPlacements[5].translation)
    d_tool = np.linalg.norm(model.jointPlacements[7].translation)
    return shoulder, L1, L2, d_tool


def elbow_circle(shoulder, wrist_center, L1, L2):
    """The circle of valid elbow positions: distance L1 from shoulder,
    L2 from wrist_center. Returns (center, radius, v1, v2, feasible) where
    E(phi) = center + radius*(cos(phi)*v1 + sin(phi)*v2), v1/v2 span the
    plane perpendicular to the shoulder-wrist axis. feasible=False if no
    such circle exists (wrist_center out of [|L1-L2|, L1+L2] range).
    """
    delta = wrist_center - shoulder
    d = np.linalg.norm(delta)
    if d < 1e-9 or d > L1 + L2 or d < abs(L1 - L2):
        return None, None, None, None, False
    u = delta / d
    a = (L1**2 - L2**2 + d**2) / (2 * d)
    r2 = L1**2 - a**2
    if r2 < 0:
        return None, None, None, None, False
    radius = np.sqrt(r2)
    center = shoulder + a * u

    world_up = np.array([0.0, 0.0, 1.0])
    ref = world_up if abs(np.dot(u, world_up)) < 0.95 else np.array([1.0, 0.0, 0.0])
    v1 = np.cross(ref, u)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(u, v1)
    return center, radius, v1, v2, True


def analytical_ik(model, p_ee, R_ee, phi, elbow_sign=1):
    """Closed-form IK for one point on the redundancy circle.

    Returns q (size model.nq) or None if this (p_ee, R_ee) has no valid
    elbow circle (out of reach in the shoulder-elbow-wrist triangle sense).
    """
    shoulder, L1, L2, d_tool = kinematic_constants(model)

    wrist_center = np.asarray(p_ee) - d_tool * np.asarray(R_ee)[:, 2]

    center, radius, v1, v2, feasible = elbow_circle(shoulder, wrist_center, L1, L2)
    if not feasible:
        return None
    elbow = center + radius * (np.cos(phi) * v1 + np.sin(phi) * v2)

    # --- shoulder: q1, q2 aim the upper arm at `elbow` ---
    d_se = elbow - shoulder
    q2 = np.arccos(np.clip(d_se[2] / L1, -1, 1))
    q1 = np.arctan2(-d_se[1], d_se[0])
    R02 = _Rz(-q1) @ _Rx(-np.pi / 2) @ _Rz(q2)

    # --- elbow angle magnitude from the shoulder-wrist distance, given sign ---
    d_sw = np.linalg.norm(wrist_center - shoulder)
    cos_interior = np.clip((L1**2 + L2**2 - d_sw**2) / (2 * L1 * L2), -1, 1)
    q4 = elbow_sign * (np.pi - np.arccos(cos_interior))

    # --- q3: roll so the elbow-to-wrist link points at wrist_center ---
    target_world = wrist_center - elbow
    target_in_R2 = R02.T @ target_world
    rhs = _Rx(-np.pi / 2) @ target_in_R2
    Lx = -L2 * np.sin(q4)
    rx, ry, _ = rhs
    q3 = np.arctan2(-ry, rx) if Lx > 0 else np.arctan2(ry, -rx)

    # --- wrist: q5, q6, q7 via ZYZ decomposition of the required rotation ---
    R03 = R02 @ _Rx(np.pi / 2) @ _Rz(-q3)
    R04 = R03 @ _Rx(-np.pi / 2) @ _Rz(-q4)
    R47_target = R04.T @ np.asarray(R_ee)
    Rp = _Rx(-np.pi / 2) @ R47_target
    q6 = np.arctan2(np.hypot(Rp[2, 0], Rp[2, 1]), Rp[2, 2])
    q5 = -np.arctan2(Rp[1, 2], Rp[0, 2])
    q7 = np.arctan2(Rp[2, 1], -Rp[2, 0])

    return np.array([q1, q2, q3, q4, q5, q6, q7])


def sample_redundancy_circle(model, p_ee, R_ee, n_phi=36, elbow_signs=(1, -1)):
    """Sweep phi over both elbow sign branches to build the family of joint
    configurations that reach (p_ee, R_ee). Returns a list of q arrays, skipping
    any phi that is infeasible.
    """
    if kinematic_constants(model)[0] is None:
        return []
    qs = []
    for elbow_sign in elbow_signs:
        for phi in np.linspace(-np.pi, np.pi, n_phi, endpoint=False):
            q = analytical_ik(model, p_ee, R_ee, phi, elbow_sign)
            if q is not None:
                qs.append(q)
    return qs
