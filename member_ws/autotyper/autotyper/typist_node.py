import os
import numpy as np
import rclpy
import tf2_ros
from concurrent.futures import ThreadPoolExecutor
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data)
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger

from autotype_msgs.msg import EpisodeResult, JointVelocityCommand

#file imports
from . import kinematics as kin
from . import layout, planner
from .perception import PANEL_H, PANEL_W, Perception, marker_corners_board

# QoS profiles
QOS_LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
QOS_CMD = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)


def quat_to_R(x, y, z, w):
    #turns a quaternion into a 3x3 rotation matrix
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Typist(Node):
    def __init__(self):
        super().__init__('typist')
        P = self.declare_parameter
        self.n_episodes = int(P('episodes', 1).value)          # how many episodes to run
        self.image_topic = P('image_topic', '/camera/image_raw').value
        self.kp_arm = float(P('kp_arm', 4.0).value)             # gain for joints 0-2
        self.kp_head = float(P('kp_head', 8.0).value)           # gain for pan and tilt
        self.speed_frac = float(P('speed_frac', 0.9).value)     # fraction of max speed we use
        self.pos_tol = float(P('pos_tol', 0.002).value)         # how close counts as arrived
        self.still_vel = float(P('still_vel', 0.010).value)     # speed below this counts as not moving
        self.settle_time = float(P('settle_time', 0.30).value)  # seconds still before we trust data
        self.aim_tol = float(P('aim_tol', 0.0015).value)        # aim accuracy needed to press
        self.hit_tol = float(P('hit_tol', 0.005).value)         # allowed distance from key centre
        self.js_timeout = float(P('js_timeout', 0.25).value)    # joint_states older than this is stale
        self.cam_timeout = float(P('cam_timeout', 3.0).value)   # how long to wait for a fresh image
        self.abort_timeout = float(P('abort_timeout', 10.0).value)  # seconds of lost data before abort
        self.min_kb_score = float(P('min_kb_score', 0.6).value)
        self.verify_px = float(P('verify_px', 12.0).value)

        # ROS interfaces
        self.create_subscription(JointState, '/joint_states', self.on_js, 10)
        self.create_subscription(Image, self.image_topic, self.on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.on_info, QOS_LATCHED)
        self.create_subscription(String, '/sim/launch_key', self.on_key, QOS_LATCHED)
        self.create_subscription(EpisodeResult, '/sim/result', self.on_result, QOS_LATCHED)
        self.pub_cmd = self.create_publisher(JointVelocityCommand, '/arm/cmd_joint_velocity', QOS_CMD)
        self.pub_press = self.create_publisher(Empty, '/arm/press', 10)
        self.pub_done = self.create_publisher(Empty, '/sim/done', 10)
        self.cli_reset = self.create_client(Trigger, '/sim/reset')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # states
        self.q = None                             # latest joint positions
        self.qd = None                            # latest joint velocities
        self.js_rx = -1e9                        # last time joint_states arrived
        self.still_since = None                  # when the arm became still
        self.img = None
        self.img_rx = -1e9
        self.K = None
        self.key_str = None
        self.key_count = 0
        self.result = None
        self.per = None
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.job = None
        self.job_then = None
        self.cmd = np.zeros(5)
        self.lost_since = None
        self.episode = 0
        self.finished = False
        self.reset_future = None
        self.state = 'startup_reset'             # reset once before ep0 too, same as between every later episode
        self.state_since = self.now()
        self.chars = []
        self.look_t = None
        self.ok = True
        self.create_timer(0.02, self.tick)

    # small helpers
    def now(self):
        #current time in seconds
        return self.get_clock().now().nanoseconds * 1e-9

    def goto(self, state):
        #switches the state machine to a new state
        self.state = state
        self.state_since = self.now()
        self.look_t = None

    def fail(self, reason):
        #reports the failure reason and moves to the finish state
        self.get_logger().error(f'[ep{self.episode}] FAILED: {reason}')
        self.ok = False
        self.goto('finish')

    def still_for(self, now):
        #how long the arm has been still
        if self.still_since is None:
            return 0.0
        else:
            return now - self.still_since

    def start_job(self, fn, *args, then):
        #runs a slow function in the background thread
        self.job = self.pool.submit(fn, *args)
        self.job_then = then

    def on_js(self, m):
        #saves the newest joint positions and velocities
        idx = []                                                 # match by name, not position
        for n in kin.JOINT_NAMES:
            idx.append(m.name.index(n))
        positions = []
        for i in idx:
            positions.append(m.position[i])
        self.q = np.array(positions)
        if len(m.velocity) == 5:
            velocities = []
            for i in idx:
                velocities.append(m.velocity[i])
            self.qd = np.array(velocities)
        else:
            self.qd = np.zeros(5)
        now = self.now()
        self.js_rx = now
        if np.max(np.abs(self.qd)) <= self.still_vel:            # remember when we became still
            if self.still_since is None:
                self.still_since = now
        else:
            self.still_since = None

    def on_image(self, m):
        #saves the newest camera image
        if m.encoding != 'bgr8':
            return
        self.img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3).copy()
        self.img_rx = self.now()

    def on_info(self, m):
        #saves the camera intrinsics
        self.K = np.array(m.k, float)

    def on_key(self, m):
        #saves the launch key word
        self.key_str = m.data
        self.key_count += 1

    def on_result(self, m):
        #saves the episode result once the sim sends it
        self.result = m

    def tick(self):
        #runs 50 times a second, this is the main loop
        if self.finished:
            return
        now = self.now()
        self.cmd = np.zeros(5)
        # if joint_states are missing or old, stop and wait
        if self.q is None or now - self.js_rx > self.js_timeout:
            if not self.lost_since:
                self.lost_since = now
            self.publish_cmd(now)
            if now - self.lost_since > self.abort_timeout and self.state != 'aborted':
                self.get_logger().error('joint_states lost for too long: holding still (aborted)')
                self.state = 'aborted'
            return
        if self.lost_since is not None:
            self.lost_since = None
            if self.state == 'aborted':
                self.state = 'wait_data'
        # if a background job is running, handle its result when done
        if self.job is not None:
            if self.job.done():
                job = self.job
                cb = self.job_then
                self.job = None
                self.job_then = None
                try:
                    cb(job.result())
                except Exception as e:                        # any error means stop safely
                    self.fail(f'exception: {e!r}')
        else:
            getattr(self, 'st_' + self.state)(now)             # run the function for the current state
        if self.finished:                     # node was shut down inside a state function
            return
        self.publish_cmd(now)

    def publish_cmd(self, now):
        #sends the current velocity command, every tick, so the sim never thinks we went silent
        m = JointVelocityCommand()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = list(kin.JOINT_NAMES)
        velocity = []
        for v in self.cmd:
            velocity.append(float(v))
        m.velocity = velocity
        self.pub_cmd.publish(m)

    def vel_for(self, e, kp, idx):
        #turns a position error into a safe velocity command that can still stop in time
        v = kp * e
        cap = np.minimum(self.speed_frac * kin.V_MAX[idx], 0.7 * np.sqrt(2 * kin.A_MAX[idx] * np.abs(e)))
        return np.sign(v) * np.minimum(np.abs(v), cap)

    def drive_arm_to(self, q_goal):
        #commands all 5 joints toward a goal pose, returns the biggest remaining error
        e = q_goal - self.q
        self.cmd[:3] = self.vel_for(e[:3], self.kp_arm, slice(0, 3))
        self.cmd[3:] = self.vel_for(e[3:], self.kp_head, slice(3, 5))
        return np.max(np.abs(e))

    def camera_pose_world(self):
        #gets the camera's position and rotation in world space, using TF if it agrees with our own math
        Rf, tf_ = kin.camera_pose(self.q)
        try:
            tr = self.tf_buffer.lookup_transform('world', 'camera_optical_frame', rclpy.time.Time())
            q = tr.transform.rotation
            R = quat_to_R(q.x, q.y, q.z, q.w)
            t = np.array([tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z])
            err_mm = 1000 * float(np.linalg.norm(t - tf_))
            if err_mm < 5.0:                                  # close enough to our own math, trust it
                return R, t, 'tf'
        except Exception:                                   # TF isn't ready yet
            pass
        return Rf, tf_, 'fk'                                  # fall back to our own math

    # STATES

    def st_startup_reset(self, now):
        #resets the sim once before episode 0 too, the same way next_episode() resets it between
        #every later episode -- otherwise ep0 starts from whatever pose a previous run (or a
        #stale sim connection) left the arm at, which was never guaranteed to see the panel
        if self.reset_future is None:
            self.reset_future = self.cli_reset.call_async(Trigger.Request())
            self.key_count_before = self.key_count
            return
        f = self.reset_future
        if not f.done():
            if now - self.state_since > 5.0:
                return self.fail('/sim/reset service did not answer')
            return
        if self.key_count > self.key_count_before or now - self.state_since > 3.0:
            self.goto('wait_data')

    def st_wait_data(self, now):
        #waits for a launch key, camera info, and a recent image before starting the episode
        if self.K is None and now - self.state_since > 2.0:        # use a default camera if none arrives
            self.K = np.array([900, 0, 639.5, 0, 900, 359.5, 0, 0, 1.0])
        have = (self.key_str is not None and self.K is not None
                and now - self.img_rx < self.cam_timeout)
        if not have:
            return
        if self.per is None:                                      # only build this once we have camera info
            png = os.path.join(get_package_share_directory('autotyper'), 'assets', 'keyboard_grid.png')
            self.per = Perception(self.K, png)
        bad = []                                                   # characters we don't have a key for
        self.chars = []
        for c in self.key_str:
            if c in layout.CHAR_KEYS:
                self.chars.append(layout.CHAR_KEYS[c])
            else:
                bad.append(c)
        self.ok = not bad
        self.presses = []
        self.next_char = 0
        self.retries = 0
        self.panel = None
        self.kb = None
        self.result = None
        if bad:
            return self.fail(f'unsupported characters {bad}')
        self.goto('look')

    def wait_fresh_image(self, now):
        #true once the arm has been still for a bit and we have a new image after that
        if self.still_for(now) < self.settle_time:
            self.look_t = None
            return False
        if self.look_t is None:
            self.look_t = now
            return False
        if self.img_rx <= self.look_t + 0.05:
            if now - self.look_t > self.cam_timeout:
                self.fail(f'camera data lost: no image for {now - self.look_t:.1f}s')
            return False
        return True

    def st_look(self, now):
        #takes one look at the panel from wherever the arm currently is
        if not self.wait_fresh_image(now):
            return
        R, t, src = self.camera_pose_world()
        self.start_job(self.job_estimate, self.img.copy(), R, t, src, 'look', 4, then=self.after_look)

    def after_look(self, r):
        #if the look failed, try again; otherwise plan where to move to type
        if r is None or r['kb'][2] < self.min_kb_score:
            self.retries += 1
            if self.retries > 3:
                return self.fail('could not find the panel/keyboard from the home pose')
            return self.goto('look')
        self.panel = r['panel']
        self.kb = r['kb']
        self.retries = 0
        self.targets = planner.key_targets_world(self.chars, self.panel, self.kb[0], self.kb[1])   # world position of every key we need to press
        self.start_job(self.job_plan_type, self.panel, self.targets, self.q.copy(), self.kb[0], self.kb[1], then=self.after_plan_type)

    def job_plan_type(self, panel, targets, q, kb_x0, kb_y0):
        #runs the pose search in the background thread
        qt, slack = planner.plan_typing(panel, targets, q, kb_x0, kb_y0)
        return qt, slack

    def after_plan_type(self, r):
        #if no good pose was found, fail; otherwise move there
        qt, slack = r
        if qt is None or slack < 0.05:
            return self.fail(f'launch key not reachable from any pose (slack={slack:.2f})')
        self.goal_q = qt
        self.after_move = 'verify'
        self.goto('move')

    def st_move(self, now):
        #drives toward the goal pose, moves on once close enough and settled
        err = self.drive_arm_to(self.goal_q)
        if err < self.pos_tol and self.still_for(now) >= self.settle_time:
            self.goto(self.after_move)
        elif now - self.state_since > 30.0:
            self.fail('move timed out')

    def st_verify(self, now):
        #takes a fresh look at the typing pose to double check the panel estimate
        if not self.wait_fresh_image(now):
            return
        R, t, _ = self.camera_pose_world()
        self.start_job(self.job_verify, self.img.copy(), R, t, self.panel, then=self.after_verify)

    def job_verify(self, img, R, t, panel):
        #checks how far off our panel estimate is from what the camera actually sees now
        found = self.per.detect_markers(img)
        errs = {}
        for mid, px in found.items():
            pw = panel.to_world(marker_corners_board(mid))
            pc = (pw - t) @ R                                  # world to camera
            if (pc[:, 2] <= 0).any():
                continue
            uv = (self.K.reshape(3, 3) @ pc.T).T
            uv = uv[:, :2] / uv[:, 2:3]
            errs[mid] = float(np.sqrt(np.mean(np.sum((uv - px) ** 2, 1))))   # pixel error for this marker
        return errs

    def after_verify(self, errs):
        #fails the episode if the panel estimate disagrees too much with the live image
        if errs:
            worst = max(errs.values())
        else:
            worst = None
        if worst is not None and worst > self.verify_px:
            return self.fail(f'panel estimate disagrees with the live image by {worst:.1f}px')
        self.goto('aim')

    def predict_hit(self, q):
        #works out where the stylus would touch the panel right now
        p, _, _ = kin.head_frame(q)
        d = kin.aim_world(q)
        n = -self.panel.z_board                                  # panel normal toward the arm
        dn = d @ n
        if dn >= -1e-6:                                          # aim ray never reaches the panel
            return None
        t = ((self.panel.t - p) @ n) / dn                        # distance along the ray to the panel
        hit_b = self.panel.R.T @ (p + t * d - self.panel.t)      # hit point, in board coordinates
        return hit_b, float(t), float(np.arccos(np.clip(-dn, -1, 1)))

    def st_aim(self, now):
        #swivels pan/tilt to point at the current key, then tries to press once settled
        i = self.next_char
        target = self.targets[i]
        a = kin.aim_to_point(self.q, target)
        goal = self.q.copy()
        goal[3] = a['pan']
        goal[4] = a['tilt']
        # only the head joints move here, the arm joints stay at rest
        e = goal[3:] - self.q[3:]
        self.cmd[3:] = self.vel_for(e, self.kp_head, slice(3, 5))
        pan_ok = a['pan'] > kin.Q_MIN[3] + 0.02 and a['pan'] < kin.Q_MAX[3] - 0.02
        tilt_ok = a['tilt'] > kin.Q_MIN[4] + 0.02 and a['tilt'] < kin.Q_MAX[4] - 0.02
        lim_ok = pan_ok and tilt_ok
        if not lim_ok:
            return self.fail(f'key {self.chars[i]} outside pan/tilt limits')
        if np.max(np.abs(e)) < self.aim_tol and self.still_for(now) >= 0.15:
            self.decide_press(now, i)
        elif now - self.state_since > 8.0:
            self.fail(f'could not settle on key {self.chars[i]}')

    def decide_press(self, now, i):
        #checks every condition before actually pressing a key
        name = self.chars[i]
        hit = self.predict_hit(self.q)
        reasons = []
        if hit is None:
            reasons.append('no_intersect')
        else:
            hit_b, rng, inc = hit
            gx = (hit_b[0] - self.kb[0]) / layout.U   # predicted hit, in key-grid units
            gy = (hit_b[1] - self.kb[1]) / layout.U
            cx, cy = layout.key_center_units(name)
            dist = float(np.hypot((gx - cx) * layout.U, (gy - cy) * layout.U))                # distance from predicted hit to key centre, in metres
            under = layout.key_at_units(gx, gy, inset=0.12)
            if under != name:
                reasons.append(f'wrong_key({under})')
            if dist > self.hit_tol:
                reasons.append(f'off_centre_{dist * 1000:.1f}mm')
            rng_ok = kin.STYLUS_MIN + 0.03 <= rng and rng <= kin.STYLUS_MAX - 0.02
            if not rng_ok:
                reasons.append(f'range_{rng:.3f}')
            if inc > kin.MAX_INCIDENCE - np.radians(5):
                reasons.append(f'incidence_{np.degrees(inc):.0f}deg')
        if np.max(np.abs(self.qd)) > self.still_vel:              # make sure we're really still
            reasons.append('moving')
        if now - self.js_rx > self.js_timeout:
            reasons.append('stale_joint_states')
        if reasons:
            return                                                # decline, try again next tick
        self.pub_press.publish(Empty())
        self.presses.append(name)
        self.next_char += 1
        self.press_t = now
        self.goto('post_press')

    def st_post_press(self, now):
        #waits for the debounce time, then moves to the next character or finishes
        if now - self.press_t < 0.20:                            # sim debounce is 0.15s
            return
        if self.next_char >= len(self.chars):
            return self.goto('finish')
        self.goto('aim')

    def st_finish(self, now):
        #tells the sim we're done, then waits for the result before moving to the next episode
        if not hasattr(self, 'done_sent') or self.done_sent != self.episode:
            self.pub_done.publish(Empty())
            self.done_sent = self.episode
            self.goto('finish')                                  # come back here to wait for the result
            return
        if self.result is None:
            if now - self.state_since > 5.0:
                return self.next_episode(now)
            return
        r = self.result
        self.get_logger().info(f'[ep{self.episode}] seed {r.seed}: target={r.target} typed={r.typed} '
                               f'exact={r.exact_match} elapsed={r.elapsed:.1f}s')
        self.next_episode(now)

    def next_episode(self, now):
        #starts the next episode, or shuts down if that was the last one
        self.episode += 1
        if self.episode >= self.n_episodes:
            self.cmd = np.zeros(5)
            self.publish_cmd(now)             # leave the arm commanded to hold still
            self.finished = True
            rclpy.shutdown()
            return
        self.reset_future = self.cli_reset.call_async(Trigger.Request())
        self.key_count_before = self.key_count
        self.goto('resetting')

    def st_resetting(self, now):
        #waits for the sim to confirm the reset, then waits for a new launch key
        f = self.reset_future
        if not f.done():
            if now - self.state_since > 5.0:
                self.fail('/sim/reset service did not answer')
            return
        if self.key_count > self.key_count_before or now - self.state_since > 3.0:
            self.key_str = self.key_str
            self.goto('wait_data')

    def st_aborted(self, now):
        #does nothing, arm just stays still after a data-loss abort
        pass

    def job_estimate(self, img, R_wc, t_wc, src, stage, min_markers):
        #finds the panel pose and the keyboard position from one camera image
        est, found = self.per.estimate_panel_pose(img, min_markers=min_markers)
        if est is None:
            return None
        panel = planner.Panel(R_wc @ est.R, R_wc @ est.t + t_wc)     # move the panel pose from camera frame to world frame
        c = panel.to_world([PANEL_W / 2, PANEL_H / 2, 0])
        ang = np.degrees(np.arccos(np.clip(panel.z_board[0], -1, 1)))      # angle from facing straight on
        # reject a panel pose that makes no physical sense
        x_ok = c[0] > 0.5 and c[0] < 1.6
        y_ok = abs(c[1]) < 0.7
        z_ok = c[2] > 0.0 and c[2] < 1.0
        angle_ok = ang < 50
        sane = x_ok and y_ok and z_ok and angle_ok
        if not sane or est.reproj_px > 3.0:
            return None
        x0, y0, score = self.per.locate_keyboard(img, est)
        kb = (x0, y0, score)
        return {'panel': panel, 'est': est, 'kb': kb}


def main():
    #starts the node and keeps it running until shutdown
    rclpy.init()
    node = Typist()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.cmd = np.zeros(5)
            node.publish_cmd(node.now())          # leave the arm commanded to hold still
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()