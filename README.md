# Robot arm planning and control

A KUKA LBR iiwa7 in Pinocchio: analytical inverse kinematics, feedforward
torque from recursive Newton-Euler, time-varying LQR tracking, and a
probabilistic roadmap that plans around obstacles.

I started this project mainly to learn a robotics library properly, and I
picked Pinocchio because it is what a lot of modern robotics code is built on.
The arm has seven joints, so it is redundant and there is a whole family of
joint configurations that put the end effector in the same place, which ends up
mattering for the roadmap.

![Tracking error, PD against time-varying LQR](arm_tracking_pd_vs_lqr.png)

Both controllers running the same circular path under the same 2 N·m torque
disturbance. The PD controller wanders up to 2.62 rad off the reference while
the LQR stays under 0.027 rad.

## What is in here

**Inverse kinematics.** There is a damped least squares Newton solver in
`inverse_kinematics.py`, but this arm also has a closed-form analytical
solution, which is what most of the project uses. Because it is redundant, that
solution is a circle of valid elbow positions rather than a single answer, and
`redundancy_ik.py` sweeps around it.

**Trajectory to torques.** A task-space trajectory maps into joint space one
waypoint at a time, then `spline.py` fits a cubic spline so there are
velocities and accelerations to work with, and Pinocchio's recursive
Newton-Euler gives the feedforward torque.

**Control.** `control.py` has both the PD controller and the time-varying LQR
backward pass. The PD gains have to be scaled by each joint's own inertia,
because the wrist and the shoulder are orders of magnitude apart and a single
gain stiff enough for one is unstable on the other. The LQR removes that
problem entirely at the cost of storing a gain matrix per timestep.

**Planning.** `prm_sampling.py` samples the task space, `self_collision.py` and
`external_collision.py` filter the resulting configurations, `prm_roadmap.py`
builds a sparse weighted graph over what survives, and `prm_query.py` runs
Dijkstra between two poses.

## Running it

```bash
python -m venv arm_env
source arm_env/bin/activate
pip install pin numpy scipy meshcat

jupyter notebook main.ipynb          # IK, splines, PD and LQR tracking
jupyter notebook PRM_obstacles.ipynb # roadmap with obstacles
```

The roadmap data in `data/` is not committed because it is large. Both PRM
notebooks rebuild it.

Rendering the figures needs a few extra packages:

```bash
pip install matplotlib trimesh pycollada
python render_animation.py --tour 5   # arm_prm_tour.mp4
python render_tracking_plot.py        # arm_tracking_pd_vs_lqr.png
```
