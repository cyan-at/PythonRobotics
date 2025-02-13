import math
import sys
import numpy as np
import scipy.linalg as la
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

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

    # first, try to solve the ricatti equation
    X = solve_dare(A, B, Q, R)

    # compute the LQR gain
    K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    eig_result = la.eig(A - B @ K)

    return K, X, eig_result[0]

def calc_nearest_index(state, cx, cy, cyaw, a, b):
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind)

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind

def calc_nearest_index2(state, cx, cy, cyaw, a, b):
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

def calc_nearest_index3(state, cx, cy, cyaw, a, b):
    ind = find_max_index_less_than(a, b)
    if ind is None:
        return -1, 0.0

    mind = math.sqrt(
        (state.x - cx[ind]) ** 2 + (state.y - cy[ind]) ** 2)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind

def lqr_speed_steering_control(
    state,
    pe, pth_e, dt,
    cx, cy, cyaw,
    # ck,
    # sp,
    Q, R, L, tv, totalt,
    distance_traveled, cumsums):
    ind, e = calc_nearest_index3(
        state, cx, cy, cyaw, cumsums, distance_traveled)

    v = state.v
    th_e = pi_2_pi(state.yaw - cyaw[ind])

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
    x[1, 0] = (e - pe) / dt
    x[2, 0] = th_e
    x[3, 0] = (th_e - pth_e) / dt

    # print("v!", v)
    # print("tv!")
    x[4, 0] = v - tv

    # input vector
    # u = [delta, accel]
    # delta: steering angle
    # accel: acceleration
    ustar = -K @ x

    # calc steering input
    # k = ck[ind]

    k = 0.0
    if state.last_yaw is not None:
        k = (cyaw[ind] - state.last_yaw) / dt
    # print("k", k, state.last_yaw)
    state.last_yaw = cyaw[ind]

    ff = math.atan2(L * k, 1)  # feedforward steering angle
    # ff = 0
    fb = pi_2_pi(ustar[0, 0])  # feedback steering angle
    delta = ff + fb

    # calc accel input
    accel = ustar[1, 0]

    return delta, ind, e, th_e, accel

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

lqr_R = np.eye(2)
# lqr_R[0, 0] = 1.0
# lqr_R[1, 1] = 0.5
dt = 0.05  # time tick[s]
L = 0.36  # Wheel base of the vehicle [m]
max_steer = np.deg2rad(180.0)  # maximum steering angle[rad]
