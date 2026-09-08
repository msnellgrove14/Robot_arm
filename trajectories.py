import numpy as np
import pinocchio as pin


def circular_trajectory(radius, T=5.0, dt=0.01, center=None, orientation=None):
    """Circular task-space trajectory in the XY plane, one full revolution
    over [0, T].

    Returns a list of pin.SE3 poses, one per timestep (length int(T/dt)+1).
    """
    if center is None:
        center = np.array([0.5, 0.0, 0.3])
    if orientation is None:
        orientation = np.eye(3)

    N = int(T / dt)
    poses = []
    for k in range(N + 1):
        t = k * dt
        theta = 2 * np.pi * t / T
        pos = center + radius * np.array([np.cos(theta), np.sin(theta), 0.0])
        poses.append(pin.SE3(orientation, pos))
    return poses


def _time_array(T, dt):
    N = int(T / dt)
    return np.array([k * dt for k in range(N + 1)])


def _smoothstep(s):
    """Minimum-jerk-like time scaling: 0 velocity/acceleration at s=0,1."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


# --- Level 1: straight line, fixed orientation -----------------------------
def linear_trajectory(start=None, end=None, T=5.0, dt=0.01, orientation=None):
    """Straight-line position trajectory from start to end, eased with a
    minimum-jerk time profile (zero velocity/acceleration at both ends)
    instead of moving at constant speed. Fixed orientation throughout.
    """
    if start is None:
        start = np.array([0.5, -0.1, 0.3])
    if end is None:
        end = np.array([0.5, 0.1, 0.3])
    if orientation is None:
        orientation = np.eye(3)

    poses = []
    for t in _time_array(T, dt):
        s = _smoothstep(t / T)
        pos = start + s * (end - start)
        poses.append(pin.SE3(orientation, pos))
    return poses


# --- Level 3: figure-eight (Lissajous), fixed orientation -------------------
def figure_eight_trajectory(A, B, T=5.0, dt=0.01, center=None, orientation=None):
    """Figure-eight trajectory in the XY plane, a Lissajous curve with an x:y
    frequency ratio of 1:2, one full pattern over [0, T]. The path crosses itself
    and its curvature changes along the way, so it is a harder track than a
    circle. Fixed orientation.
    """
    if center is None:
        center = np.array([0.5, 0.0, 0.3])
    if orientation is None:
        orientation = np.eye(3)

    poses = []
    for t in _time_array(T, dt):
        theta = 2 * np.pi * t / T
        pos = center + np.array([A * np.sin(theta), B * np.sin(2 * theta), 0.0])
        poses.append(pin.SE3(orientation, pos))
    return poses


# --- Level 4: helix, fixed orientation --------------------------------------
def helical_trajectory(radius, pitch, n_turns=2.0, T=5.0, dt=0.01, center=None, orientation=None):
    """Helical trajectory: circular XY motion with a linear Z climb of `pitch` per
    revolution, n_turns revolutions over [0, T]. Fixed orientation.
    """
    if center is None:
        center = np.array([0.5, 0.0, 0.3])
    if orientation is None:
        orientation = np.eye(3)

    poses = []
    for t in _time_array(T, dt):
        theta = 2 * np.pi * n_turns * t / T
        z = pitch * n_turns * (t / T)
        pos = center + np.array([radius * np.cos(theta), radius * np.sin(theta), z])
        poses.append(pin.SE3(orientation, pos))
    return poses


# --- Level 5: fixed position, oscillating orientation -----------------------
def oscillating_orientation_trajectory(position=None, axis="x", amplitude=0.3, T=5.0, dt=0.01):
    """Fixed end-effector position, with the orientation oscillating sinusoidally
    about a fixed axis by +-amplitude, one full cycle over [0, T]. This is the
    first trajectory that exercises orientation tracking instead of holding it
    constant.
    """
    if position is None:
        position = np.array([0.5, 0.0, 0.3])

    poses = []
    for t in _time_array(T, dt):
        angle = amplitude * np.sin(2 * np.pi * t / T)
        R = pin.utils.rotate(axis, angle)
        poses.append(pin.SE3(R, position))
    return poses


# --- Level 6: helix + oscillating orientation, combined ---------------------
def helical_with_wobble_trajectory(
    radius, pitch, n_turns=2.0, T=5.0, dt=0.01, center=None,
    wobble_axis="x", wobble_amplitude=0.15,
):
    """Helical position path with an oscillating orientation wobble layered on top,
    so position and orientation both vary at the same time. Closer to a real tool
    path that has to keep reorienting while it moves.
    """
    if center is None:
        center = np.array([0.5, 0.0, 0.3])

    poses = []
    for t in _time_array(T, dt):
        theta = 2 * np.pi * n_turns * t / T
        z = pitch * n_turns * (t / T)
        pos = center + np.array([radius * np.cos(theta), radius * np.sin(theta), z])

        wobble_angle = wobble_amplitude * np.sin(2 * np.pi * t / T)
        R = pin.utils.rotate(wobble_axis, wobble_angle)

        poses.append(pin.SE3(R, pos))
    return poses


# --- Level 7: full 3D Lissajous position, fixed orientation -----------------
def lissajous_3d_trajectory(
    Ax, Ay, Az, fx=1.0, fy=2.0, fz=3.0, T=5.0, dt=0.01, center=None,
    phase=None, orientation=None,
):
    """3D Lissajous position trajectory, with x, y and z each driven by their own
    frequency so the path is not confined to a plane. Fixed orientation.
    """
    if center is None:
        center = np.array([0.5, 0.0, 0.3])
    if phase is None:
        phase = np.array([0.0, 0.0, 0.0])
    if orientation is None:
        orientation = np.eye(3)

    poses = []
    for t in _time_array(T, dt):
        theta = 2 * np.pi * t / T
        pos = center + np.array([
            Ax * np.sin(fx * theta + phase[0]),
            Ay * np.sin(fy * theta + phase[1]),
            Az * np.sin(fz * theta + phase[2]),
        ])
        poses.append(pin.SE3(orientation, pos))
    return poses


# --- Level 8: 3D Lissajous position + full independent tumbling orientation -
def tumbling_lissajous_trajectory(
    Ax, Ay, Az, fx=1.0, fy=2.0, fz=3.0, T=5.0, dt=0.01, center=None, phase=None,
    wx=1.0, wy=1.5, wz=2.5, orientation_amplitude=0.5,
):
    """3D Lissajous position path combined with an independent tumbling orientation,
    rotating about x, y and z at their own frequencies and composed as
    R = Rz @ Ry @ Rx. Position and orientation both vary continuously and
    independently of each other.
    """
    if center is None:
        center = np.array([0.5, 0.0, 0.3])
    if phase is None:
        phase = np.array([0.0, 0.0, 0.0])

    poses = []
    for t in _time_array(T, dt):
        theta = 2 * np.pi * t / T
        pos = center + np.array([
            Ax * np.sin(fx * theta + phase[0]),
            Ay * np.sin(fy * theta + phase[1]),
            Az * np.sin(fz * theta + phase[2]),
        ])

        rx = orientation_amplitude * np.sin(wx * theta)
        ry = orientation_amplitude * np.sin(wy * theta)
        rz = orientation_amplitude * np.sin(wz * theta)
        R = pin.utils.rotate("z", rz) @ pin.utils.rotate("y", ry) @ pin.utils.rotate("x", rx)

        poses.append(pin.SE3(R, pos))
    return poses


# --- Level 9: random-Fourier position + orientation -------------------------
def chaotic_fourier_trajectory(
    n_harmonics=5, pos_amplitude=0.15, orientation_amplitude=0.4,
    T=5.0, dt=0.01, center=None, seed=None,
):
    """Position (x, y, z) and orientation (about x, y, z) are each a sum of
    n_harmonics random sinusoids, with a random integer frequency, phase and
    weight per harmonic. No two axes move in a clean relationship to each other,
    so the path sweeps a wide unstructured region of the task space and is the
    hardest trajectory here to track.
    """
    if center is None:
        center = np.array([0.5, 0.0, 0.3])

    rng = np.random.default_rng(seed)

    def random_signal(n_axes, amplitude):
        freqs = rng.integers(1, n_harmonics + 1, size=(n_axes, n_harmonics))
        phases = rng.uniform(0, 2 * np.pi, size=(n_axes, n_harmonics))
        weights = rng.uniform(0.3, 1.0, size=(n_axes, n_harmonics))
        weights /= weights.sum(axis=1, keepdims=True)  # normalize so amplitude bound holds
        return freqs, phases, weights

    pos_freqs, pos_phases, pos_weights = random_signal(3, pos_amplitude)
    rot_freqs, rot_phases, rot_weights = random_signal(3, orientation_amplitude)

    def eval_signal(theta, freqs, phases, weights, amplitude):
        out = np.zeros(3)
        for a in range(3):
            out[a] = amplitude * np.sum(weights[a] * np.sin(freqs[a] * theta + phases[a]))
        return out

    poses = []
    for t in _time_array(T, dt):
        theta = 2 * np.pi * t / T
        pos = center + eval_signal(theta, pos_freqs, pos_phases, pos_weights, pos_amplitude)
        rot = eval_signal(theta, rot_freqs, rot_phases, rot_weights, orientation_amplitude)
        R = pin.utils.rotate("z", rot[2]) @ pin.utils.rotate("y", rot[1]) @ pin.utils.rotate("x", rot[0])
        poses.append(pin.SE3(R, pos))
    return poses


# --- Level 10: orbit about the robot's own base axis, tangent orientation --
def orbital_trajectory(
    r0, r_amplitude, r_frequency, z0, z_amplitude, z_frequency,
    T=5.0, dt=0.01, n_orbits=1.0,
):
    """The end effector orbits the robot's own vertical axis. The circle is centered
    on (0, 0), so it passes through the base and joint1 sweeps a full 360 degrees
    n_orbits times over [0, T], unlike the earlier trajectories which orbit an
    offset point. On top of that sweep the radius breathes in and out
    sinusoidally and the height oscillates at a higher frequency than the radius.

    Orientation is tangent to the path, so the tool's local Z axis points along
    the direction of motion. It is computed from the path's own velocity rather
    than being parameterized separately like the earlier trajectories.

    Note that joint1's real mechanical limit is +-2.97 rad, which is less than a
    full revolution, so an n_orbits=1.0 sweep is not achievable on hardware.
    About 92% of the IK solutions land outside the limit. Nothing in this IK and
    sim pipeline enforces joint limits, so this is kept as a kinematic-only
    demo.
    """
    t_arr = _time_array(T, dt)
    theta = 2 * np.pi * n_orbits * t_arr / T
    r = r0 + r_amplitude * np.sin(2 * np.pi * r_frequency * t_arr / T)
    z = z0 + z_amplitude * np.sin(2 * np.pi * z_frequency * t_arr / T)

    positions = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
    velocities = np.gradient(positions, dt, axis=0)

    # Build the tangent frame by propagating it continuously from the
    # previous sample (Gram-Schmidt the old "right" against the new
    # "forward") instead of independently re-deriving each frame from a
    # fixed world reference with a threshold fallback. The independent
    # version has a discontinuity right at the fallback's switch point,
    # because "forward" changes smoothly but the reference axis flips there,
    # so "right" and "up" jump even though the path barely moved.
    # Propagating is continuous by construction, with no threshold to trip.
    world_up = np.array([0.0, 0.0, 1.0])
    poses = []
    right_prev = None
    for pos, vel in zip(positions, velocities):
        forward = vel / (np.linalg.norm(vel) + 1e-9)

        if right_prev is None:
            # only the very first frame needs an arbitrary reference
            ref = world_up if abs(np.dot(forward, world_up)) < 0.95 else np.array([1.0, 0.0, 0.0])
            right = np.cross(ref, forward)
        else:
            right = right_prev - np.dot(right_prev, forward) * forward
        right /= np.linalg.norm(right) + 1e-9
        up = np.cross(forward, right)

        R = np.column_stack([right, up, forward])
        poses.append(pin.SE3(R, pos))
        right_prev = right
    return poses
