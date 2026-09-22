# planner.py -- solves directly for the best arm pose with a Jacobian + least-squares fit;
# falls back to trying lots of poses (grid search) only if that direct solve doesn't pan out
import numpy as np
from . import kinematics as kin
from . import layout

# limits
SAFE_PAN = np.radians(38.0)
SAFE_TILT = np.radians(28.0)
SAFE_INC = np.radians(48.0)
SAFE_RANGE = (0.09, 0.31)
MIN_PLANE_DIST = 0.08 # keep head/elbow this far in front of the panel


class Panel:
    #the panel's position and rotation, in world coordinates
    def __init__(self, R, t):
        self.R = np.asarray(R, float)
        self.t = np.asarray(t, float)

    @property
    def z_board(self):                      # the panel's facing direction, in world coordinates
        return self.R[:, 2]

    def to_world(self, pts_board):
        #turns board points into world points
        return np.asarray(pts_board, float) @ self.R.T + self.t


def grid(centers, half_deg, step_deg, lo, hi):
    #builds a batch of candidate joint angles to try
    axes = []
    for c, h, l, u in zip(centers, half_deg, lo, hi):
        a = np.arange(max(c - np.radians(h), l), min(c + np.radians(h), u) + 1e-9, np.radians(step_deg))
        axes.append(a)
    g = np.meshgrid(*axes, indexing='ij')
    flat_axes = []
    for x in g:
        flat_axes.append(x.ravel())
    return np.stack(flat_axes, -1)


def full_q(q3):
    #adds pan and tilt (set to 0) to make a full 5-joint pose
    return np.concatenate([q3, np.zeros((len(q3), 2))], 1)


def arm_side_ok(Q, panel):
    #the head and elbow must stay in front of the panel, not through it
    p, _, elbow = kin.head_frame(Q)
    n = -panel.z_board
    ok = ((p - panel.t) @ n >= MIN_PLANE_DIST) & ((elbow - panel.t) @ n >= MIN_PLANE_DIST)
    return ok & (p[:, 2] > 0.05)


def search(score_fn, q_now, panel, yaw0):
    #a rough search first, then a finer search around the best rough answer
    lo = kin.Q_MIN[:3]
    hi = kin.Q_MAX[:3]
    # rough pass: 3 degree steps, base yaw only within +-60 deg of the panel's direction
    Q = grid([yaw0, np.radians(35), np.radians(-70)], [60, 200, 200], 3.0, lo, hi)
    s = score_fn(full_q(Q))
    s = np.where(arm_side_ok(full_q(Q), panel), s, -np.inf)   # throw out poses with the arm on the wrong side
    best = Q[np.argmax(s)]
    if not np.isfinite(s.max()):
        return None, -np.inf
    # fine pass: 0.5 degree steps around the rough winner
    Q = grid(list(best), [3, 3, 3], 0.5, lo, hi)
    s = score_fn(full_q(Q))
    s = np.where(arm_side_ok(full_q(Q), panel), s, -np.inf)
    i = int(np.argmax(s))
    return Q[i], float(s[i])


def key_targets_world(chars, panel, kb_x0, kb_y0):
    #world position of the centre of each key we need to press
    pts = []
    for name in chars:
        gx, gy = layout.key_center_units(name)
        pts.append([kb_x0 + gx * layout.U, kb_y0 + gy * layout.U, 0.0])   # the key's offset from the keyboard's corner, in metres
    return panel.to_world(np.array(pts))


def margins(Q, targets_w, panel):
    #works out how much room to spare each pose has on every limit
    p, R, _ = kin.head_frame(Q)
    d_w = targets_w[None] - p[:, None]
    d_h = np.einsum('nji,nmj->nmi', R, d_w)                           # rotate every target into each pose's own frame
    rng = np.linalg.norm(d_h, axis=-1)
    pan = np.arctan2(d_h[..., 1], d_h[..., 0])
    tilt = np.arctan2(d_h[..., 2], np.hypot(d_h[..., 0], d_h[..., 1]))
    inc = np.arccos(np.clip((d_w @ panel.z_board) / rng, -1, 1))
    lo, hi = SAFE_RANGE
    # the tightest limit is what actually matters
    m = np.minimum.reduce([(SAFE_PAN - np.abs(pan)) / SAFE_PAN,
                           (SAFE_TILT - np.abs(tilt)) / SAFE_TILT,
                           (SAFE_INC - inc) / SAFE_INC,
                           (rng - lo) / ((hi - lo) / 2), (hi - rng) / ((hi - lo) / 2)])
    return m


def keyboard_center_world(panel, kb_x0, kb_y0):
    #world position of the middle of the whole keyboard, not just the keys this word happens to
    #need -- so the camera's position doesn't depend on which letters we're typing
    cx = kb_x0 + (layout.GRID_W / 2.0) * layout.U
    cy = kb_y0 + (layout.GRID_H / 2.0) * layout.U
    return panel.to_world([cx, cy, 0.0])


def head_point(q3, d):
    #where the camera sits, and which way it's facing, for one (yaw, shoulder, elbow) guess --
    #the same maths as kin.head_frame, just for a single pose instead of a batch, since the
    #least-squares solve below works one guess at a time
    q0, q1, q2 = q3
    c0, s0 = np.cos(q0), np.sin(q0)
    c1, s1 = np.cos(q1), np.sin(q1)
    phi = q1 + q2
    cp, sp = np.cos(phi), np.sin(phi)
    a = np.array([cp * c0, cp * s0, sp])          # direction the camera is facing
    p = (np.array([0.0, 0.0, kin.BASE_H])
         + kin.UPPER * np.array([c1 * c0, c1 * s0, s1]) + kin.FOREARM * a)
    return p, a


def head_jacobian(q3, d):
    #the Jacobian of "where the camera would end up standing, d metres back from whatever it's
    #looking at" with respect to (yaw, shoulder, elbow) -- lets us solve directly for the joint
    #angles with least squares, instead of scoring a huge grid of candidate poses
    q0, q1, q2 = q3
    c0, s0 = np.cos(q0), np.sin(q0)
    c1, s1 = np.cos(q1), np.sin(q1)
    phi = q1 + q2
    cp, sp = np.cos(phi), np.sin(phi)
    er = np.array([-s0, c0, 0.0])                  # how moving yaw shifts a point
    g1 = np.array([-s1 * c0, -s1 * s0, c1])        # how moving the shoulder shifts the forearm direction
    gp = np.array([-sp * c0, -sp * s0, cp])        # how shoulder+elbow together shift the forearm direction
    d_yaw = (kin.UPPER * c1 + (kin.FOREARM + d) * cp) * er
    d_shoulder = kin.UPPER * g1 + (kin.FOREARM + d) * gp
    d_elbow = (kin.FOREARM + d) * gp
    return np.column_stack([d_yaw, d_shoulder, d_elbow])   # 3x3


def solve_camera_pose(target_w, d, q_now, iters=50, eps=1e-4, lam=1e-2):
    #Gauss-Newton least-squares solve for the (yaw, shoulder, elbow) that puts the camera d
    #metres from target_w, facing it square-on. each step fits a straight-line (linear) model of
    #the geometry using the Jacobian, solves that with least squares, and repeats until the error
    #is tiny -- this replaces trying thousands of poses with solving the geometry directly
    lo, hi = kin.Q_MIN[:3], kin.Q_MAX[:3]
    q = np.clip(np.asarray(q_now[:3], float), lo, hi)
    I3 = np.eye(3)
    for _ in range(iters):
        p, a = head_point(q, d)
        e = (p + d * a) - target_w                 # how far off from "d metres in front of target_w" we are
        if np.linalg.norm(e) < eps:
            break
        J = head_jacobian(q, d)
        step = -np.linalg.solve(J.T @ J + (lam * lam) * I3, J.T @ e)   # damped least-squares step
        q_try = np.clip(q + step, lo, hi)
        p2, a2 = head_point(q_try, d)
        if np.linalg.norm((p2 + d * a2) - target_w) < np.linalg.norm(e):
            q, lam = q_try, max(lam * 0.5, 1e-6)    # that step helped -- take it, trust the linear model a bit more
        else:
            lam = min(lam * 2.0, 1e3)               # that step made things worse -- distrust it, take a smaller one next time
    p, a = head_point(q, d)
    residual = float(np.linalg.norm((p + d * a) - target_w))
    return q, residual


def plan_typing(panel, targets_w, q_now, kb_x0, kb_y0):
    #finds the pose with the most room to spare across every key we need to press.
    #first solves directly for the best position relative to the keyboard with the Jacobian +
    #least-squares fit above; only falls back to trying lots of poses if that doesn't pan out
    #(unreachable, wrong side of the panel, or a key that's still out of range)
    yaw0 = np.arctan2(targets_w[:, 1].mean(), targets_w[:, 0].mean())
    center = keyboard_center_world(panel, kb_x0, kb_y0)
    d = 0.5 * (SAFE_RANGE[0] + SAFE_RANGE[1])       # stand in the middle of the comfortable range

    q3, residual = solve_camera_pose(center, d, q_now)
    if residual < 1e-3:
        q = np.concatenate([q3, [0.0, 0.0]])
        if arm_side_ok(q[None], panel)[0]:
            m = margins(q[None], targets_w, panel)
            if m.min() >= 0.05:   # same safety bar the caller uses -- a barely-positive margin isn't good enough
                return q, float(m.min())

    def score(Q):
        #a pose is only as good as its worst key, and travelling further costs a little
        m = margins(Q, targets_w, panel)
        return m.min(1) - 0.02 * np.linalg.norm(Q[:, :3] - q_now[:3], axis=1)

    q3, s = search(score, q_now, panel, yaw0)
    if q3 is None:
        return None, -np.inf
    q = np.concatenate([q3, [0.0, 0.0]])
    m = margins(q[None], targets_w, panel)
    return q, float(m.min())