import cv2
import numpy as np
from . import layout

PANEL_W, PANEL_H = 0.400, 0.175       # metres
MARKER_SIDE = 0.020
MARKER_CENTERS = {0: (0.016, 0.016), 1: (0.384, 0.016), 2: (0.384, 0.159), 3: (0.016, 0.159)}


def marker_corners_board(mid):
    #the 4 corners of a marker, in board coordinates
    cx, cy = MARKER_CENTERS[mid]
    h = MARKER_SIDE / 2
    return np.array([[cx - h, cy - h, 0], [cx + h, cy - h, 0], [cx + h, cy + h, 0], [cx - h, cy + h, 0]], float)


def pixel_to_ray(K, u, v):
    #STEP 2: turns a pixel into a ray coming out of the camera
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    r = np.array([x, y, 1.0])
    return r / np.linalg.norm(r)


def solve_marker_depths(rays, dists, iters=50):
    #STEPS 3-5: works out how far along each ray a point actually is, using the known distances between them
    n = len(rays)
    lam = np.full(n, 0.25)                         # initial guess: a typical operating depth
    pairs = list(dists.keys())
    for _ in range(iters):
        P = lam[:, None] * rays                    # guess where each point is right now
        res = np.zeros(len(pairs))
        J = np.zeros((len(pairs), n))
        for k, (i, j) in enumerate(pairs):
            diff = P[i] - P[j]
            d = np.linalg.norm(diff)
            res[k] = d - dists[(i, j)]             # how far off this guess is
            J[k, i] = diff @ rays[i] / d
            J[k, j] = -diff @ rays[j] / d
        step = np.linalg.solve(J.T @ J + 1e-9 * np.eye(n), J.T @ res)   # one step closer to the answer
        lam = lam - step
    return lam                                     # the solved depths


def panel_axes(P1, P2, P3):
    #STEPS 7-9: works out the panel's left/right, up/down, and facing directions
    u = P2 - P1
    v = P3 - P1
    u_hat = u / np.linalg.norm(u)                    # normalize
    v_hat = v - (v @ u_hat) * u_hat
    v_hat = v_hat / np.linalg.norm(v_hat)
    n = np.cross(u_hat, v_hat)                       # the panel's normal direction
    return np.column_stack([u_hat, v_hat, n])        # these 3 directions become the rotation matrix


class PanelEstimate:
    #the panel's position and rotation, relative to the camera
    def __init__(self, R, t, reproj_px):
        self.R = R
        self.t = t
        self.reproj_px = reproj_px


class Perception:
    def __init__(self, K, keyboard_png):
        self.K = np.asarray(K, float).reshape(3, 3)               # the camera has no distortion, so we don't need distortion numbers
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX   # sub-pixel corners
        params.minMarkerPerimeterRate = 0.01     # markers are tiny (24 px) from the home pose
        params.polygonalApproxAccuracyRate = 0.05
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)
        self.kb_gray = cv2.cvtColor(cv2.imread(keyboard_png), cv2.COLOR_BGR2GRAY)   # the keyboard template photo, loaded once

    def detect_markers(self, bgr):
        #finds the 4 markers we care about and returns their pixel corners
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        found = {}
        if ids is not None:
            for c, i in zip(corners, ids.ravel()):
                if int(i) in MARKER_CENTERS:
                    found[int(i)] = c.reshape(4, 2)
        return found

    def estimate_panel_pose(self, bgr, min_markers=2):
        #works out where the panel actually is, using only the markers, no PnP solver
        found = self.detect_markers(bgr)              # STEP 1: detect markers
        n = len(found)
        if n < max(min_markers, 3):                       # need at least 3 points for a unique rotation
            return None, found
        if 0 not in found or 1 not in found or 3 not in found:   # panel_axes needs markers 0, 1 and 3
            return None, found
        ids = sorted(found)
        # uses every corner, not an averaged centre, since averaging pixels adds a small bias
        obj_list = []                                       # known board-frame corners, all 4 per marker
        for i in ids:
            obj_list.append(marker_corners_board(i))
        obj = np.concatenate(obj_list)
        img_list = []                                        # matching detected pixel corners, same order
        for i in ids:
            img_list.append(found[i])
        img = np.concatenate(img_list)
        ray_list = []
        for u, v in img:
            ray_list.append(pixel_to_ray(self.K, u, v))
        rays = np.array(ray_list)
        m = len(rays)
        dists = {}                                          # distance between every pair of corners
        for i in range(m):
            for j in range(i + 1, m):
                dists[(i, j)] = np.linalg.norm(obj[i] - obj[j])
        lam = solve_marker_depths(rays, dists)
        if np.any(lam <= 0):                               # a corner ended up behind the camera
            return None, found
        cam_pts = lam[:, None] * rays                        # where each corner actually is
        corner0 = {}    # where each marker's first corner sits in the list
        for k, mid in enumerate(ids):
            corner0[mid] = k * 4
        R = panel_axes(cam_pts[corner0[0]], cam_pts[corner0[1]], cam_pts[corner0[3]])
        t = cam_pts[corner0[0]] - R @ obj[corner0[0]]        # where the panel actually is
        if t[2] <= 0:                                       # panel behind the camera makes no sense
            return None, found
        pc = (obj @ R.T + t) @ self.K.T                      # turns board points into pixels, to check our answer
        proj = pc[:, :2] / pc[:, 2:3]
        err = float(np.sqrt(np.mean(np.sum((proj - img) ** 2, 1))))   # how far off our answer is, in pixels
        return PanelEstimate(R, t, err), found

    def rectify(self, bgr, est, px_per_m=2000.0):
        #warps the image into a flat, top-down view of the panel
        H = self.K @ np.column_stack([est.R[:, 0], est.R[:, 1], est.t])   # turns a board point into a pixel
        S = np.diag([px_per_m, px_per_m, 1.0])
        size = (int(PANEL_W * px_per_m), int(PANEL_H * px_per_m))
        rect = cv2.warpPerspective(bgr, H @ np.linalg.inv(S), size,
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        for mid, (cx, cy) in MARKER_CENTERS.items():   # black out the markers so they don't confuse the template match
            r = 0.0145
            x0 = int((cx - r) * px_per_m)
            x1 = int((cx + r) * px_per_m)
            y0 = int((cy - r) * px_per_m)
            y1 = int((cy + r) * px_per_m)
            rect[max(y0, 0):y1, max(x0, 0):x1] = 0
        return rect

    def locate_keyboard(self, bgr, est, px_per_m=2000.0):
        #finds where the keyboard sits on the panel, by matching a template photo
        rect = cv2.cvtColor(self.rectify(bgr, est, px_per_m), cv2.COLOR_BGR2GRAY)
        tw = int(round(layout.GRID_W * layout.U * px_per_m))
        th = int(round(layout.GRID_H * layout.U * px_per_m))
        tmpl = cv2.resize(self.kb_gray, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(rect, tmpl, cv2.TM_CCOEFF_NORMED)
        min_val, best, min_loc, max_loc = cv2.minMaxLoc(res)
        bx, by = max_loc
        sx = 0.0                                             # a small correction to make the match more precise
        sy = 0.0
        if bx > 0 and bx < res.shape[1] - 1:
            l = res[by, bx - 1]
            c = res[by, bx]
            r = res[by, bx + 1]
            denom_x = l - 2 * c + r
            if denom_x != 0:
                sx = 0.5 * (l - r) / denom_x
            else:
                sx = 0.0
        if by > 0 and by < res.shape[0] - 1:
            u = res[by - 1, bx]
            c = res[by, bx]
            d = res[by + 1, bx]
            denom_y = u - 2 * c + d
            if denom_y != 0:
                sy = 0.5 * (u - d) / denom_y
            else:
                sy = 0.0
        return (bx + sx) / px_per_m, (by + sy) / px_per_m, float(best)