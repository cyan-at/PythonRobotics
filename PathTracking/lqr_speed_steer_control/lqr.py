'''
https://atsushisakai.github.io/PythonRobotics/modules/6_path_tracking/lqr_speed_and_steering_control/lqr_speed_and_steering_control.html
'''

import math
import sys
import numpy as np
import scipy.linalg as la
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

class bcolors:
  # https://godoc.org/github.com/whitedevops/colors
  DEFAULT = "\033[39m"
  BLACK = "\033[30m"
  RED = "\033[31m"
  GREEN = "\033[32m"
  YELLOW = "\033[33m"
  BLUE = "\033[34m"
  MAGENTA = "\033[35m"
  CYAN = "\033[36m"
  LGRAY = "\033[37m"
  DARKGRAY = "\033[90m"
  FAIL = "\033[91m"
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  OKBLUE = '\033[94m'
  HEADER = '\033[95m'
  LIGHTCYAN = '\033[96m'
  WHITE = "\033[97m"

  ENDC = '\033[0m'
  BOLD = '\033[1m'
  DIM = "\033[2m"
  UNDERLINE = '\033[4m'
  BLINK = "\033[5m"
  REVERSE = "\033[7m"
  HIDDEN = "\033[8m"

  BG_DEFAULT = "\033[49m"
  BG_BLACK = "\033[40m"
  BG_RED = "\033[41m"
  BG_GREEN = "\033[42m"
  BG_YELLOW = "\033[43m"
  BG_BLUE = "\033[44m"
  BG_MAGENTA = "\033[45m"
  BG_CYAN = "\033[46m"
  BG_GRAY = "\033[47m"
  BG_DKGRAY = "\033[100m"
  BG_LRED = "\033[101m"
  BG_LGREEN = "\033[102m"
  BG_LYELLOW = "\033[103m"
  BG_LBLUE = "\033[104m"
  BG_LMAGENTA = "\033[105m"
  BG_LCYAN = "\033[106m"
  BG_WHITE = "\033[107m"

def print_color_str(s, clr):
  print(clr + s + bcolors.ENDC)

def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Angle modulo operation
    Default angle modulo range is [-pi, pi)

    Parameters
    ----------
    x : float or array_like
        A angle or an array of angles. This array is flattened for
        the calculation. When an angle is provided, a float angle is returned.
    zero_2_2pi : bool, optional
        Change angle modulo range to [0, 2pi)
        Default is False.
    degree : bool, optional
        If True, then the given angles are assumed to be in degrees.
        Default is False.

    Returns
    -------
    ret : float or ndarray
        an angle or an array of modulated angle.

    Examples
    --------
    >>> angle_mod(-4.0)
    2.28318531

    >>> angle_mod([-4.0])
    np.array(2.28318531)

    >>> angle_mod([-150.0, 190.0, 350], degree=True)
    array([-150., -170.,  -10.])

    >>> angle_mod(-60.0, zero_2_2pi=True, degree=True)
    array([300.])

    """
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle

def pi_2_pi(angle):
    return angle_mod(angle)

def modulo(angle):
    # // reduce the angle  
    angle =  angle % 360 

    # // force it to be the positive remainder, so that 0 <= angle < 360  
    angle = (angle + 360) % 360

    # // force into the minimum absolute value residue class, so that -180 < angle <= 180  
    if (angle > 180):
        angle -= 360

    return angle

def modulo_rad(rad):
    x = modulo(np.rad2deg(rad))
    return np.deg2rad(x)

class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

        self.last_yaw = None

        self.e = 0.0
        self.e_th = 0.0

def solve_dare(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    x = Q
    x_next = Q
    max_iter = 150
    eps = 0.01

    for i in range(max_iter):
        x_next = A.T @ x @ A - A.T @ x @ B @ \
                 la.inv(R + B.T @ x @ B) @ B.T @ x @ A + Q
        if (abs(x_next - x)).max() < eps:
            break
        x = x_next

    return x_next

def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    try:

        # first, try to solve the ricatti equation
        X = solve_dare(A, B, Q, R)

        # compute the LQR gain
        K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

        eig_result = la.eig(A - B @ K)
    except Exception as e:
        import ipdb; ipdb.set_trace()

    return K, X, eig_result[0]

def smallest_diff(a, b):
    # candidates = np.array([
    #     a - b,
    #     a - (b + 2*np.pi),
    #     a - (b - 2*np.pi)
    # ])
    # candidates2 = np.abs(candidates)
    # return candidates[np.argmin(candidates2)]

    # a = (a + np.pi) % 2*np.pi - np.pi
    # return a

    # return math.atan2(np.sin(a-b), np.cos(a-b))

    diff = ( np.rad2deg(a) - np.rad2deg(b) + 180 ) % 360 - 180
    a = diff + 360 if diff < -180 else diff
    return np.deg2rad(a)

def calc_nearest_index(state, cx, cy, cyaw, a, b, debug=False):
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind)

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    # print("IND", ind, len(cyaw))

    # angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    angle = smallest_diff(cyaw[ind], math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind

def calc_nearest_index2(state, cx, cy, cyaw, a, b, debug=False):
    ind = min(int(b // a), len(cx) - 1)

    mind = math.sqrt(
        (state.x - cx[ind]) ** 2 + (state.y - cy[ind]) ** 2)
    print("ind", ind)
    print("mind", mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind

def interpolate(arr, factor):
    res = []
    i = 0
    while i < len(arr) - 1:
        res.extend(
            np.linspace(arr[i], arr[i+1], factor, endpoint=False)
        )
        i += 1
    res.append(arr[-1])
    return res

def do_repeat(arr, factor):
    res = []
    i = 0
    while i < len(arr) - 1:
        res.extend(
            np.repeat(arr[i+1], factor)
        )
        i += 1
    res.append(arr[-1])
    return res

def best_rotation_candidate(a, b):
    offsets = [
        0.0,
        2*np.pi,
        -2*np.pi
    ]

    b_candidates = [
        b,
        b + 2*np.pi,
        b - 2*np.pi
    ]

    candidates = np.array([
        a - b,
        a - (b + 2*np.pi),
        a - (b - 2*np.pi)
    ])
    candidates2 = np.abs(candidates)

    return b_candidates[np.argmin(candidates2)], np.min(candidates2), offsets[np.argmin(candidates2)]

def rotation_smooth(rads):
    if len(rads) == 0:
        return rads

    modulated = [rads[0]]
    i = 1
    offset_stack = 0.0

    while i < len(rads):
        smoothed_rad, dist, offset = best_rotation_candidate(
            modulated[i-1], rads[i])
        dist2 = modulo_rad(rads[i] - modulated[i-1])

        # print(
        #     "HEY!!!! %.3f vs %.3f = %.3f, %.3f @ %.3f" % (
        #         modulated[i-1], rads[i],
        #         dist, dist2,
        #         smoothed_rad
        #         )
        #     )

        if np.abs(dist2 - np.pi / 2) < np.abs(dist2 - np.pi):
            # print("genuine turn")
            # modulated.append(rads[i])
            modulated.append(smoothed_rad)
        else:
            # print("wraparound")
            modulated.append(smoothed_rad)

        i += 1

    return modulated

def find_max_index_less_than(arr, num):
    """
    Finds the index of the maximum value in a NumPy array that is less than a specified number.

    Args:
    arr: The NumPy array to search.
    num: The upper bound for the values to consider.

    Returns:
    The index of the maximum value less than num, or None if no such value exists.
    """
    valid_indices = np.where(arr < num)[0]

    if valid_indices.size == 0:
        return None

    max_index_within_limit = valid_indices[np.argmax(arr[valid_indices])]
    return max_index_within_limit

def calc_nearest_index3(state, cx, cy, cyaw, a, b, debug=False):
    ind = find_max_index_less_than(a, b)
    if ind is None:
        return 0, 0.0

    if debug:
        print("ind", ind)

    mind = math.sqrt(
        (state.x - cx[ind]) ** 2 + (state.y - cy[ind]) ** 2)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))

    if debug:
        print("closest", cx[ind], cy[ind])
        print("current", state.x, state.y)
        print("angle to closest", math.atan2(dyl, dxl))
        print("goal yaw", cyaw[ind])
        print("angle", angle)

    # angle = smallest_diff(cyaw[ind], math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind

def calc_nearest_index_hybrid13(state, cx, cy, cyaw, a, b, debug=False):
    ind1, e1 = calc_nearest_index(
        state, cx, cy, cyaw, a, b, debug=debug)

    ind2, e2 = calc_nearest_index3(
        state, cx, cy, cyaw, a, b, debug=debug)

    # import ipdb; ipdb.set_trace()

    # ind1 might be AHEAD of ind2
    # because the closest item is farther from start
    # than the travel
    if (ind1 > ind2):
        if debug:
            print("ahead (%d, %.3f) vs (%d, %.3f), state=(%.3f, %.3f)" % (ind1, e1, ind2, e2, state.x, state.y))
        return ind1, e1
    else:
        if debug:
            print("behind (%d, %.3f) vs (%d, %.3f), state=(%.3f, %.3f)" % (ind1, e1, ind2, e2, state.x, state.y))

    return ind2, e2

def lqr_speed_steering_control(
    state,
    pe, pth_e, dt,
    cx, cy, cyaw,
    # ck,
    # sp,
    Q, R, L, tv, totalt,
    distance_traveled, cumsums, debug=True):
    if debug:
        print("t: %.3f" % (totalt))

    ind, e = calc_nearest_index_hybrid13(
        state, cx, cy, cyaw, cumsums, distance_traveled, debug=debug)

    best_dist_estimate = cumsums[ind]

    v = state.v
    # print("state.v", v)
    # th_e = pi_2_pi(state.yaw - cyaw[ind])

    expected_yaw = cyaw[ind]

    # th_e = smallest_diff(
    #     state.yaw,
    #     expected_yaw)

    # print("ind", ind, len(cyaw))
    # print("state.yaw", state.yaw)
    # print("expected_yaw", expected_yaw)
    th_e = modulo_rad(state.yaw - expected_yaw)

    # th_e = (state.yaw - cyaw[ind]) % 2*np.pi
    # th_e = (th_e + 2*np.pi) % 2*np.pi

    # th_e = modulo_rad(state.yaw - cyaw[ind])

    # A = [1.0, dt, 0.0, 0.0, 0.0
    #      0.0, 0.0, v, 0.0, 0.0]
    #      0.0, 0.0, 1.0, dt, 0.0]
    #      0.0, 0.0, 0.0, 0.0, 0.0]
    #      0.0, 0.0, 0.0, 0.0, 1.0]
    A = np.zeros((5, 5))
    A[0, 0] = 1.0
    A[0, 1] = dt
    A[1, 2] = v
    A[2, 2] = 1.0
    A[2, 3] = dt
    A[4, 4] = 1.0

    # B = [0.0, 0.0
    #     0.0, 0.0
    #     0.0, 0.0
    #     v/L, 0.0
    #     0.0, dt]
    B = np.zeros((5, 2))
    B[3, 0] = v / L
    B[4, 1] = dt

    # print("Q", Q)

    K, _, _ = dlqr(A, B, Q, R)

    # state vector
    # x = [e, dot_e, th_e, dot_th_e, delta_v]
    # e: lateral distance to the path
    # dot_e: derivative of e
    # th_e: angle difference to the path
    # dot_th_e: derivative of th_e
    # delta_v: difference between current speed and target speed
    x = np.zeros((5, 1))
    x[0, 0] = e

    # x[1, 0] = (e - pe) / dt
    '''
    when the calc_nearest_index3 sign changes, this can blow up
    it is reasonable we only care to control the MAGNITUDE of this
    the signage maybe does not matter (?)
    x[1, 0] = np.abs(e - pe) / dt

    i.e.
    ind 297
    closest 0.3305452202422401 -2.08874005450175
    current 0.7554259032823595 -1.945941151618235
    angle to closest -2.817361559602424
    goal yaw -2.8094245446288317
    angle 0.007937014973592227
    x [[ 0.448]
     [ 17.859]
     [ 0.454]
     [-0.171]
     [-0.000]]
    controller: 21.85: ((-2.355, -2.809)   (0.448, -0.445)      (0.454, 0.463 = -0.171)                  -2.809, -2.812 -> 0.050
    delta 0.018, -1.601 -> -1.583
    yaw update -2.355, (-1.583, 2.239) -> -0.116
    state.v 0.19977933240264226
    '''
    # x[1, 0] = np.abs(e - pe) / dt
    if debug:
        print("e: %.3f, pe %.3f => %.3f" % (e, pe, x[1, 0]))
    if (np.sign(e) != np.sign(pe)):
        if debug:
            print_color_str("SIGN CHANGE!", bcolors.WARNING)
        normalized_pe = np.sign(e) * np.abs(pe)
        x[1, 0] = np.abs(e - normalized_pe) / dt

    x[2, 0] = th_e
    dot_th_e = modulo_rad(th_e - pth_e) / dt
    x[3, 0] = dot_th_e

    # print("v!", v)
    # print("tv!")
    x[4, 0] = v - tv

    # print("K", K)
    # print("x", x)

    # input vector
    # u = [delta, accel]
    # delta: steering angle
    # accel: acceleration
    ustar = -K @ x

    # calc steering input
    # k = ck[ind]

    k = 0.0
    if state.last_yaw is not None:
        k = smallest_diff(cyaw[ind], state.last_yaw) / dt

        if debug:
            print("x {}".format(x))

            print("controller: %.2f: ((%.3f, %.3f)   (%.3f, %.3f)      (%.3f, %.3f = %.3f)                  %.3f, %.3f -> %.3f" % (
                totalt,
                state.yaw, expected_yaw,
                e, pe,
                th_e, pth_e, dot_th_e,
                cyaw[ind], state.last_yaw, k))
    state.last_yaw = cyaw[ind]

    ff = math.atan2(L * k, 1)  # feedforward steering angle
    # ff = 0
    fb = modulo_rad(ustar[0, 0])  # feedback steering angle
    delta = modulo_rad(ff + fb)

    if np.abs(fb) > np.pi / 4:
        if debug:
            print_color_str("delta %.3f, %.3f -> %.3f" % (
                ff,
                fb,
                delta), bcolors.BG_LCYAN)
    else:
        if debug:
            print("delta %.3f, %.3f -> %.3f" % (
                ff,
                fb,
                delta))

    # calc accel input
    accel = ustar[1, 0]

    # print("ustar", ustar)

    return delta, ind, e, th_e, accel, expected_yaw, fb, best_dist_estimate, ind

# LQR parameters

# state vector
# x = [e, dot_e, th_e, dot_th_e, delta_v]
# e: lateral distance to the path
# dot_e: derivative of e
# th_e: angle difference to the path
# dot_th_e: derivative of th_e
# delta_v: difference between current speed and target speed

lqr_Q = np.eye(5)
# lqr_Q[0, 0] = 0.5
# lqr_Q[0, 0] = 1.0
lqr_Q[1, 1] = 0.0
# lqr_Q[2, 2] = 0.5
lqr_Q[3, 3] = 0.1
lqr_Q[4, 4] = 0.1 # speed de-prioritized vs position

# lqr_Q[2, 2] = 5e-3

lqr_R = np.eye(2)
# lqr_R[0, 0] = 1.0
# lqr_R[1, 1] = 0.5
dt = 0.05  # time tick[s]
L = 0.36  # Wheel base of the vehicle [m]
max_steer = np.deg2rad(180.0)  # maximum steering angle[rad]
