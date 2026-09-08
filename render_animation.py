"""Render an MP4 of the arm following a PRM path around the sphere obstacles.

Same approach as the UR5 project's renderer: draw the real URDF meshes with
matplotlib rather than screen-capturing a live Meshcat session, so this runs
headless. Meshes are loaded once and only transformed per frame.

    python render_animation.py            # obstacle-avoiding PRM path
    python render_animation.py --seed 3   # a different start/goal pair
"""

import argparse
import os
import shutil
import subprocess

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from external_collision import Sphere
from prm_query import dijkstra_path, embed_X_in_graph, shortcut_path
from prm_sampling import sample_batch
from self_collision import build_link_capsules, default_excluded_pairs
from spline import cubic_spline_interpolation
from workspace import is_pose_reachable

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, 'lbr_iiwa7_r800.urdf')
FRAMES = os.path.join(HERE, '_anim_frames')
OUT = os.path.join(HERE, 'arm_prm_animation.mp4')
FPS = 30

# same obstacle field as PRM_obstacles.ipynb
SPHERES = [
    Sphere([0.5, 0.0, 0.5], 0.08),
    Sphere([-0.5, 0.3, 0.5], 0.08),
    Sphere([0.4, 0.4, 0.4], 0.08),
    Sphere([-0.4, -0.4, 0.6], 0.08),
    Sphere([0.0, 0.5, 0.3], 0.08),
    Sphere([0.3, -0.5, 0.5], 0.08),
    Sphere([-0.3, 0.2, 0.8], 0.08),
    Sphere([0.6, -0.2, 0.6], 0.07),
    Sphere([-0.2, -0.5, 0.4], 0.08),
    Sphere([0.2, 0.3, 0.7], 0.07),
]


def load_meshes(visual_model):
    """(vertices, faces, parentJoint, placement) per geometry, loaded once."""
    out = []
    for g in visual_model.geometryObjects:
        path = g.meshPath
        if not os.path.exists(path):
            continue
        m = trimesh.load(path, force='mesh')
        v = np.asarray(m.vertices) * np.asarray(g.meshScale).reshape(3)
        out.append((v, np.asarray(m.faces), g.parentJoint, g.placement))
    return out


def shade(tris, light=np.array([0.5, 0.7, 0.9])):
    """Lambert shading from face normals, so the arm reads as a solid object
    instead of a flat silhouette."""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    lit = np.clip(n / norm @ (light / np.linalg.norm(light)), 0, 1)
    c = 0.32 + 0.62 * lit
    return np.stack([c * 0.84, c * 0.88, c * 0.93], axis=1)


def sphere_wire(center, radius, n=14):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--frames', type=int, default=150)
    ap.add_argument('--attempts', type=int, default=40)
    ap.add_argument('--tour', type=int, default=2,
                    help='number of waypoints to chain (2 = a single A->B path)')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        URDF, package_dirs=HERE
    )
    data = model.createData()
    meshes = load_meshes(visual_model)
    print(f'{model.name}: nq={model.nq}, {len(meshes)} visual meshes')

    from scipy.sparse import load_npz

    q_pool = np.load(os.path.join(HERE, 'data/q_pool_external_collision_free.npz'))
    q_pool = q_pool[q_pool.files[0]]
    graph = load_npz(os.path.join(HERE, 'data/prm_graph_obstacles.npz'))
    capsules = build_link_capsules(collision_model)
    excluded = default_excluded_pairs(len(capsules))
    print(f'roadmap: {q_pool.shape[0]} nodes')

    rng = np.random.default_rng(args.seed)

    # The roadmap is not fully connected, so pick both endpoints out of its
    # largest connected component. That guarantees Dijkstra finds a path and
    # avoids re-embedding task-space poses into a 1.6M-node k-d tree.
    from scipy.sparse.csgraph import connected_components

    n_comp, labels = connected_components(graph, directed=False)
    big = np.bincount(labels).argmax()
    members = np.flatnonzero(labels == big)
    print(f'{n_comp} components; largest has {members.size} nodes')

    def leg(a, b):
        idx_path, cost = dijkstra_path(graph, int(a), int(b))
        if idx_path is None or not np.isfinite(cost) or len(idx_path) < 4:
            return None
        return q_pool[idx_path]

    q_path = None
    for attempt in range(args.attempts):
        nodes = rng.choice(members, size=args.tour, replace=False)
        legs = [leg(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
        if any(l is None for l in legs):
            continue
        # drop each leg's first node, which repeats the previous leg's last
        q_path = np.vstack([legs[0]] + [l[1:] for l in legs[1:]])
        print(f'attempt {attempt}: {len(legs)} legs, {len(q_path)} nodes total')
        break

    if q_path is None:
        raise SystemExit('no usable path found; try another --seed')

    q_path = shortcut_path(q_path, model, data, capsules, excluded, spheres=SPHERES)
    t_knots = np.linspace(0, 1, len(q_path))
    t_eval = np.linspace(0, 1, args.frames)
    q_traj = cubic_spline_interpolation(t_knots, q_path, t_eval)[0]

    shutil.rmtree(FRAMES, ignore_errors=True)
    os.makedirs(FRAMES)

    fig = plt.figure(figsize=(6.4, 6.4), dpi=110)
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    for k, q in enumerate(q_traj):
        ax.clear()
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        # one collection for the whole arm so matplotlib depth-sorts the
        # links against each other rather than per-mesh
        tris = []
        for v, f, joint, placement in meshes:
            M = data.oMi[joint] * placement
            vt = (M.rotation @ v.T).T + M.translation
            tris.append(vt[f])
        tris = np.concatenate(tris)
        ax.add_collection3d(
            Poly3DCollection(tris, facecolors=shade(tris), edgecolor='none')
        )
        for s in SPHERES:
            x, y, z = sphere_wire(np.asarray(s.position), s.radius)
            ax.plot_surface(x, y, z, color='#d1495b', alpha=0.55, linewidth=0)

        ax.set_xlim(-0.62, 0.62)
        ax.set_ylim(-0.62, 0.62)
        ax.set_zlim(0.0, 1.05)
        ax.set_box_aspect((1, 1, 0.9))
        ax.set_axis_off()
        ax.view_init(elev=22, azim=-60 + 30 * k / len(q_traj))
        fig.savefig(os.path.join(FRAMES, f'{k:04d}.png'), facecolor='white')
        if k % 25 == 0:
            print(f'  frame {k}/{len(q_traj)}')

    plt.close(fig)
    subprocess.run(
        ['ffmpeg', '-y', '-framerate', str(FPS), '-i',
         os.path.join(FRAMES, '%04d.png'),
         '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p', args.out],
        check=True, capture_output=True
    )
    shutil.rmtree(FRAMES, ignore_errors=True)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
