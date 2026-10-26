"""2D LiDAR scan matching (point-to-line ICP) + local-map front end + occupancy grid.

좌표 규약
- 로봇 좌표계: x = 앞, y = 왼쪽 (m). LiDAR 각도는 앞이 0°이고 시계 방향으로 증가(앞 0°, 오른쪽 90°).
- 월드 좌표계: 첫 스캔 위치가 원점, 첫 방향이 +x.
- pose = (x, y, theta[rad]) : 월드에서 본 로봇의 위치와 방향 (반시계 +).
"""
import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------- SE(2) helpers
def compose(a, b):
    """a ∘ b : b 를 a 의 좌표계에서 본 변환으로 보고 월드로 옮김."""
    c, s = np.cos(a[2]), np.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))


def inverse(a):
    c, s = np.cos(a[2]), np.sin(a[2])
    return (-(c * a[0] + s * a[1]), -(-s * a[0] + c * a[1]), wrap(-a[2]))


def wrap(th):
    return (th + np.pi) % (2 * np.pi) - np.pi


def transform(pts, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return pts @ np.array([[c, s], [-s, c]]) + np.array([pose[0], pose[1]])


# ---------------------------------------------------------------- preprocessing
def polar_to_xy(angle_deg, dist_mm, min_r=0.15, max_r=6.0, body_mask=None):
    r = np.asarray(dist_mm, float) / 1000.0
    a = np.asarray(angle_deg, float)
    ok = (r >= min_r) & (r <= max_r)
    if body_mask is not None:
        ok &= ~body_mask[a.astype(int) % 360]
    th = -np.radians(a[ok])
    return np.column_stack([r[ok] * np.cos(th), r[ok] * np.sin(th)])


def voxel_downsample(pts, voxel):
    key = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[np.sort(idx)]


def estimate_normals(pts, tree, k=8, radius=0.20, max_thickness=0.03):
    """각 점의 법선과, 직선(벽) 위에 있다고 볼 수 있는지(good)를 반환."""
    kk = min(k, len(pts))
    d, idx = tree.query(pts, k=kk)
    nbr = pts[idx]
    valid = d < radius
    cnt = valid.sum(1)
    w = valid[..., None]
    mean = (nbr * w).sum(1) / np.maximum(cnt, 1)[:, None]
    diff = (nbr - mean[:, None, :]) * w
    cov = np.einsum('nki,nkj->nij', diff, diff) / np.maximum(cnt, 1)[:, None, None]
    evals, evecs = np.linalg.eigh(cov)
    normal = evecs[:, :, 0]
    good = (cnt >= 4) & (np.sqrt(np.maximum(evals[:, 0], 0)) < max_thickness)
    return normal, good


class Target:
    def __init__(self, pts):
        self.pts = pts
        self.tree = cKDTree(pts)
        self.nrm, self.good = estimate_normals(pts, self.tree)


# ---------------------------------------------------------------- ICP (point-to-line)
def icp_p2l(src, tgt, init, iters=30, gate0=0.5, gate_min=0.12, sigma=0.04):
    """src(로봇 좌표계 점) 를 tgt(월드 좌표계 점) 에 맞추는 pose 를 구함.
    반환: (pose, info) 또는 None.  info 에는 품질 지표가 들어 있음."""
    x, y, th = init
    H = np.zeros((3, 3))
    for it in range(iters):
        c, s = np.cos(th), np.sin(th)
        p = src @ np.array([[c, s], [-s, c]]) + np.array([x, y])
        gate = max(gate_min, gate0 * (0.8 ** it))
        d, idx = tgt.tree.query(p)
        m = (d < gate) & tgt.good[idx]
        if m.sum() < 30:
            return None
        pp, q, n = p[m], tgt.pts[idx[m]], tgt.nrm[idx[m]]
        e = np.einsum('ij,ij->i', n, pp - q)
        w = 1.0 / (1.0 + (e / sigma) ** 2)
        J = np.column_stack([n[:, 0], n[:, 1], -n[:, 0] * pp[:, 1] + n[:, 1] * pp[:, 0]])
        H = (J.T * w) @ J
        g = (J.T * w) @ e
        try:
            delta = -np.linalg.solve(H + 1e-9 * np.eye(3), g)
        except np.linalg.LinAlgError:
            return None
        dx, dy, dth = delta
        cd, sd = np.cos(dth), np.sin(dth)
        x, y = cd * x - sd * y + dx, sd * x + cd * y + dy
        th += dth
        if abs(dx) < 1e-4 and abs(dy) < 1e-4 and abs(dth) < 2e-4:
            break
    # 최종 위치에서 품질 지표 계산
    c, s = np.cos(th), np.sin(th)
    p = src @ np.array([[c, s], [-s, c]]) + np.array([x, y])
    d, idx = tgt.tree.query(p)
    m = (d < gate_min) & tgt.good[idx]
    if m.sum() < 30:
        return None
    pp, q, n = p[m], tgt.pts[idx[m]], tgt.nrm[idx[m]]
    e = np.einsum('ij,ij->i', n, pp - q)
    inl = np.abs(e) < 0.05
    Jt = np.column_stack([n[:, 0], n[:, 1]])[inl]
    eig_t = np.linalg.eigvalsh(Jt.T @ Jt / max(inl.sum(), 1))   # 이동 관측 가능성 (0~1)
    info = dict(inlier_frac=float(inl.sum() / len(src)),
                rms=float(np.sqrt(np.mean(e[inl] ** 2))) if inl.any() else np.inf,
                trans_min_eig=float(eig_t[0]))
    return (x, y, wrap(th)), info


# ---------------------------------------------------------------- SLAM front end
def run_scan_matching(scans, K=10, kf_dist=0.08, kf_ang=np.radians(6),
                      max_step_m=0.3, max_step_rad=np.radians(20), voxel=0.02):
    """scans: 로봇 좌표계 점 배열의 리스트. 반환: poses(N,3), infos(list), failed(list[bool])"""
    poses, infos, failed = [], [], []
    keyframes = []                       # 월드 좌표계 점들
    last_kf_pose = None
    rel_prev = (0.0, 0.0, 0.0)
    tgt = None

    def rebuild():
        allpts = voxel_downsample(np.vstack(keyframes[-K:]), voxel)
        return Target(allpts)

    for k, pts in enumerate(scans):
        if k == 0:
            pose = (0.0, 0.0, 0.0)
            keyframes.append(transform(pts, pose))
            tgt = rebuild()
            last_kf_pose = pose
            poses.append(pose); infos.append(dict(inlier_frac=1.0, rms=0.0, trans_min_eig=0.5)); failed.append(False)
            continue
        prev = poses[-1]
        inits = [compose(prev, rel_prev), prev]
        best = None
        for init in inits:
            res = icp_p2l(pts, tgt, init)
            if res is None:
                continue
            if best is None or res[1]['inlier_frac'] > best[1]['inlier_frac']:
                best = res
        ok = best is not None
        if ok:
            step = compose(inverse(prev), best[0])
            if np.hypot(step[0], step[1]) > max_step_m or abs(step[2]) > max_step_rad:
                ok = False
        if ok:
            pose, info = best
            rel_prev = compose(inverse(prev), pose)
        else:
            pose, info = prev, dict(inlier_frac=0.0, rms=np.inf, trans_min_eig=0.0)
            rel_prev = (0.0, 0.0, 0.0)
        poses.append(pose); infos.append(info); failed.append(not ok)
        dk = compose(inverse(last_kf_pose), pose)
        if ok and (np.hypot(dk[0], dk[1]) > kf_dist or abs(dk[2]) > kf_ang):
            keyframes.append(transform(pts, pose))
            tgt = rebuild()
            last_kf_pose = pose
    return np.array(poses), infos, failed


# ---------------------------------------------------------------- occupancy grid
def build_occupancy(scans, poses, res=0.05, max_r=6.0, every=2,
                    l_occ=0.9, l_free=-0.4, l_max=4.0, pad=0.5):
    allp = np.vstack([transform(s, p) for s, p in zip(scans[::every], poses[::every])])
    xmin, ymin = allp.min(0) - pad
    xmax, ymax = allp.max(0) + pad
    xmin, ymin = min(xmin, poses[:, 0].min() - pad), min(ymin, poses[:, 1].min() - pad)
    xmax, ymax = max(xmax, poses[:, 0].max() + pad), max(ymax, poses[:, 1].max() + pad)
    W = int(np.ceil((xmax - xmin) / res)); Hh = int(np.ceil((ymax - ymin) / res))
    L = np.zeros((Hh, W), np.float32)
    for pts, pose in zip(scans[::every], poses[::every]):
        w = transform(pts, pose)
        o = np.array([pose[0], pose[1]])
        vec = w - o
        rng = np.linalg.norm(vec, axis=1)
        n_steps = np.maximum((rng / (res * 0.5)).astype(int), 1)
        free_cells = set()
        for v, r, ns in zip(vec, rng, n_steps):
            ts = (np.arange(ns) / ns)[:, None] * v[None, :]            # 끝점 직전까지
            cc = np.floor((o + ts - [xmin, ymin]) / res).astype(int)
            free_cells.update(map(tuple, cc))
        occ_cells = set(map(tuple, np.floor((w - [xmin, ymin]) / res).astype(int)))
        free_cells -= occ_cells
        if free_cells:
            fc = np.array(list(free_cells))
            L[fc[:, 1], fc[:, 0]] += l_free
        if occ_cells:
            oc = np.array(list(occ_cells))
            L[oc[:, 1], oc[:, 0]] += l_occ
        np.clip(L, -l_max, l_max, out=L)
    return L, (xmin, ymin, res)
