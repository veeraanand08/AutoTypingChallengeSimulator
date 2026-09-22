import numpy as np

JOINT_NAMES = ['base_yaw', 'shoulder_pitch', 'elbow_pitch', 'head_pan', 'head_tilt']
# how far each joint can travel, in radians
Q_MIN = np.radians([-120.0, -30.0, -140.0, -45.0, -35.0])
Q_MAX = np.radians([120.0, 100.0, 0.0, 45.0, 35.0])
V_MAX = np.array([0.6, 0.6, 0.8, 1.0, 1.0])       # top speed for each joint
A_MAX = np.array([1.5, 1.5, 2.0, 3.0, 3.0])       # top acceleration for each joint
BASE_H, UPPER, FOREARM = 0.30, 0.60, 0.40          # arm link lengths, in metres
STYLUS_MIN, STYLUS_MAX = 0.05, 0.35                # legal distance range for a press
MAX_INCIDENCE = np.radians(55.0)                   # steepest angle a press can still land at

# FORWARD KINEMATICS -> given the first 3 joint angles --> works out position and direction of camera/stylus
def head_frame(q):
    q = np.asarray(q, float)
    yaw = q[..., 0]
    th1 = q[..., 1]
    th2 = q[..., 2]
    c0 = np.cos(yaw)                # base rotation around the vertical axis
    s0 = np.sin(yaw)
    phi = th1 + th2                                 # the forearm's real angle from horizontal
    z0 = np.zeros_like(yaw)                          # placeholder
    # elbow position
    p_elbow = np.stack([UPPER * np.cos(th1) * c0, UPPER * np.cos(th1) * s0, BASE_H + UPPER * np.sin(th1)], -1)
    # head position
    p_head = p_elbow + FOREARM * np.stack([np.cos(phi) * c0, np.cos(phi) * s0, np.sin(phi)], -1)
    # build the head's own X, Y, Z directions
    X = np.stack([np.cos(phi) * c0, np.cos(phi) * s0, np.sin(phi)], -1)   # X points straight out along the forearm
    Y = np.stack([-s0, c0, z0], -1)                                       # Y is sideways, at a right angle to X
    Z = np.stack([-np.sin(phi) * c0, -np.sin(phi) * s0, np.cos(phi)], -1)  # Z is at a right angle to both X and Y
    R = np.stack([X, Y, Z], -1)
    return p_head, R, p_elbow


def camera_pose(q):
    #turns the head frame into the camera's own frame (x=right, y=down, z=forward)
    p, R, _ = head_frame(q)
    Rc = np.stack([-R[..., :, 1], -R[..., :, 2], R[..., :, 0]], -1)   # re-order and flip axes to match the camera convention
    return Rc, p


def aim_to_point(q, target_world):
    #given a target point in the world, works out the pan/tilt/range needed to point at it
    p, R, _ = head_frame(q)
    d_world = np.asarray(target_world, float) - p           # vector from the head to the target, in world coordinates
    d_head = np.einsum('...ji,...j->...i', R, d_world)     # rotate that vector into the head's own frame
    rng = np.linalg.norm(d_head, axis=-1)                    # straight-line distance to the target
    pan = np.arctan2(d_head[..., 1], d_head[..., 0])        # left/right angle needed
    tilt = np.arctan2(d_head[..., 2], np.hypot(d_head[..., 0], d_head[..., 1]))  # up/down angle needed
    return {'pan': pan, 'tilt': tilt, 'range': rng}


def aim_world(q):
    #works out which direction the stylus is currently pointing, in world coordinates
    q = np.asarray(q, float)
    _, R, _ = head_frame(q)
    pan = q[..., 3]
    tilt = q[..., 4]
    u = np.stack([np.cos(tilt) * np.cos(pan), np.cos(tilt) * np.sin(pan), np.sin(tilt)], -1)   # pan/tilt turned into a direction
    return np.einsum('...ij,...j->...i', R, u)      # turn that direction into world coordinates