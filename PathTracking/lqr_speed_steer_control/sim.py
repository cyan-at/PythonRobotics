#! /usr/bin/env python3

"""

Path tracking simulation with LQR speed and steering control

author Atsushi Sakai (@Atsushi_twi)

./sim.py /home/charlieyan1/Dev/jim/pymap2d/sdf_path_0000210.npy
./sim.py /home/charlieyan1/Dev/jim/pymap2d/sdf_path_0000280.npy

"""
from lqr import *

import matplotlib.pyplot as plt

# from PathPlanning.CubicSpline import cubic_spline_planner

import argparse

# from scipy.spatial.transform import Rotation as R
# from scipy.spatial.transform import Slerp

# from spliner import *

def two_d_make_x_y_theta_hom(x, y, theta):
    hom = np.eye(3)

    theta = theta % (2 * np.pi)
    # 2019-08-02 parentheses!!!

    hom[0, 0] = np.cos(theta)
    hom[0, 1] = -np.sin(theta)
    hom[1, 0] = np.sin(theta)
    hom[1, 1] = np.cos(theta)

    hom[0, 2] = x
    hom[1, 2] = y
    return hom

def plot_line(ax, a, b, mode, color, linewidth, alpha=0.2):
    if mode == 3:
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            [a[2], b[2]],
            color=color,
            linewidth=linewidth,
            alpha=alpha)
    elif mode == 2:
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha)

def plot_gnomon(ax, g, mode=3, length=0.1, linewidth=5, c=None, offset = 0.0):
    '''
    mode is dimension of 'canvas'
    '''
    if (mode == 3):
        o = g.dot(np.array([offset, offset, 0.0, 1.0]))
    elif (mode == 2):
        o = g.dot(np.array([offset, offset, 1.0]))

    if (mode == 3):
        x = g.dot(np.array([length*1 + offset, offset, 0.0, 1.0]))
        if c is not None:
            plot_line(ax, o, x, mode, c, linewidth)
        else:
            plot_line(ax, o, x, mode, 'r', linewidth)
    elif (mode == 2):
        x = g.dot(np.array([length*1 + offset, offset, 1.0]))
        if c is not None:
            plot_line(ax, o, x, mode, c, linewidth)
        else:
            plot_line(ax, o, x, mode, 'r', linewidth)

    if (mode == 3):
        y = g.dot(np.array([offset, length*2 + offset, 0.0, 1.0]))
        if c is not None:
            plot_line(ax, o, y, mode, c, linewidth)
        else:
            plot_line(ax, o, y, mode, 'g', linewidth)
    elif (mode == 2):
        y = g.dot(np.array([offset, length*2 + offset, 1.0]))
        if c is not None:
            plot_line(ax, o, y, mode, c, linewidth)
        else:
            plot_line(ax, o, y, mode, 'g', linewidth)

    if (mode == 3):
        z = g.dot(np.array([offset, 0.0, length*3 + offset, 1.0]))
        if c is not None:
            plot_line(ax, o, z, mode, c, linewidth)
        else:
            plot_line(ax, o, z, mode, 'b', linewidth)

def update(L, state, a, delta):
    if delta >= max_steer:
        delta = max_steer
    if delta <= - max_steer:
        delta = - max_steer

    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
    state.v = state.v + a * dt

    return state

def do_simulation(
    state,
    cx, cy, cyaw,
    # ck, speed_profile,
    goal,
    T,
    tv,
    args,
    full_dist):
    goal_dis = 0.3
    stop_speed = 0.05

    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    v_msg = [0.0]
    w_msg = [0.0]
    ais = [0.0]
    t = [0.0]
    travel = [0.0]

    debug1 = [0.0]
    debug2 = [0.0]
    debug3 = [0.0]
    debug4 = [0.0]
    debug5 = [0.0]
    debug6 = [0.0]

    distances = [] # len(cx) - 1
    cumsums = [0.0] # len(cx)
    i = 0
    pt = np.array([cx[i], cy[i]])
    while i < len(cx) - 1:
        new_pt = np.array([cx[i+1], cy[i+1]])
        dist = np.linalg.norm(new_pt - pt, ord=2)

        distances.append(dist)
        cumsums.append(cumsums[-1] + dist)

        i += 1
        pt = new_pt
    cumsums = np.array(cumsums)

    time = 0.0
    distance_traveled = 0.0
    best_dist_estimate = 0.0
    best_idx = 0

    if args.dist <= 0.0:
        terminate_dist = cumsums[-2]
        # not -1, because at -1, the nearest_idx might be the path start
    else:
        # T = min(T, full_T)
        terminate_dist = args.dist

    print("terminate_dist", terminate_dist)

    # import ipdb; ipdb.set_trace()

    total_e = 0.0
    ticks = 0

    while terminate_dist > best_dist_estimate:

    # while best_idx != len(cyaw) - 2:
        # tv = min(0.2, time * 1) # ramp up to tv

        dl, target_ind, state.e, state.e_th, ai, expected_yaw, fb, best_dist_estimate, best_idx =\
            lqr_speed_steering_control(
            state, state.e, state.e_th,
            dt,
            cx, cy, cyaw,
            # ck,
            # speed_profile,
            lqr_Q, lqr_R, L, tv, time,
            distance_traveled, cumsums, debug=args.debug)

        # print("best_dist_estimate",
        #     best_dist_estimate,
        #     best_idx,
        #     len(cyaw) - 1,
        #     terminate_dist,
        #     full_dist)

        distance_traveled += state.v * dt

        dyaw = state.v / L * math.tan(dl) * dt

        old_yaw = state.yaw

        # print("ai", ai)

        state = update(L, state, ai, dl)

        # print("after state.v", state.v)

        if args.debug:
            if np.abs(dyaw) > np.pi / 4:
                print_color_str("yaw update %.3f, (%.3f, %.3f) -> %.3f" % (
                    old_yaw, dl, dyaw, state.yaw), bcolors.BG_RED)
            else:
                print("yaw update %.3f, (%.3f, %.3f) -> %.3f" % (
                    old_yaw, dl, dyaw, state.yaw))

        if abs(state.v) <= stop_speed:
            target_ind += 1

        time = time + dt

        # check goal
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        # if math.hypot(dx, dy) <= goal_dis:
        #     print("Goal")
        #     break

        v_msg_i = v_msg[-1] + ai * dt
        w_msg_i = dyaw

        v_msg.append(v_msg_i)
        w_msg.append(w_msg_i)
        ais.append(ai)
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        travel.append(distance_traveled)

        debug1.append(state.e_th)
        debug2.append(expected_yaw)
        debug3.append(fb)
        debug4.append(dyaw)
        debug5.append(dl)
        debug6.append(state.e)

        if args.debug:
            print("state.v", state.v)

        # if target_ind % 1 == 0 and show_animation:
        #     plt.cla()
        #     # for stopping simulation with the esc key.
        #     plt.gcf().canvas.mpl_connect(
        #         'key_release_event',
        #         lambda event: [exit(0) if event.key == 'escape' else None])
        #     plt.plot(cx, cy, "-r", label="course")
        #     plt.plot(x, y, "ob", label="trajectory")
        #     plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
        #     plt.axis("equal")
        #     plt.grid(True)
        #     plt.title("speed[km/h]:" + str(round(state.v * 3.6, 2))
        #               + ",target index:" + str(target_ind))
        #     plt.pause(0.0001)

        total_e += np.abs(state.e)
        ticks += 1

        if args.debug:
            print("")

    # import ipdb; ipdb.set_trace()

    if args.debug:
        print("total_e", total_e)
        print("ticks", ticks)
    rmse = np.sqrt(total_e / ticks)

    return t, x, y, yaw, v, v_msg, w_msg, ais, travel, debug1, debug2, debug3, debug4, debug5, debug6, rmse, terminate_dist


def calc_speed_profile(cyaw, target_speed):
    speed_profile = [target_speed] * len(cyaw)

    direction = 1.0

    # Set stop point
    for i in range(len(cyaw) - 1):
        dyaw = abs(cyaw[i + 1] - cyaw[i])
        switch = math.pi / 4.0 <= dyaw < math.pi / 2.0

        if switch:
            direction *= -1

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

        if switch:
            speed_profile[i] = 0.0

    # speed down
    # for i in range(40):
    #     speed_profile[-i] = target_speed / (50 - i)
    #     if speed_profile[-i] <= 1.0 / 3.6:
    #         speed_profile[-i] = 1.0 / 3.6

    return speed_profile

def two_d_rvec_vec_from_matrix_2d(m):
    x = m[0, 2]
    y = m[1, 2]

    atan2 = np.arctan2(m[1, 0], m[0, 0])
    # SUPER #COOL #IMPORTANT #port
    # this is how we get the original theta 2019-08-02
    # https://stackoverflow.com/a/32549077
    # deal with quadrant logic

    '''
    theta_0 = math.acos(m[0, 0])
    theta_1 = math.asin(m[1, 0])
    theta_2 = math.asin(-1.0 * m[0, 1])
    theta_3 = math.acos(m[1, 1])
    # TODO(someone) finish this up,
    # do not always all match, sometimes negative
    thetas = [theta_0, theta_1, theta_2, theta_3]
    mode_thetas = Util.modes(thetas)
    if len(mode_thetas) > 0:
        # print("more than one theta found", mode_thetas)
        pass
    '''

    return [x, y, atan2], None

def slerp(p0, p1, t):
    omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

def slerp3(arr, xys, factor):
    res = []
    i = 0
    while i < len(arr) - 1:
        diff = arr[i+1] - arr[i]

        if (np.abs(diff) < 1e-8):
            res.extend(
                np.linspace(arr[i], arr[i+1], factor, endpoint=False)
            )
            i += 1
            continue

        # import ipdb; ipdb.set_trace()
        distance = np.linalg.norm(xys[i+1] - xys[i], ord=2)

        interp = []

        min_t = np.abs(diff) / 0.5
        min_d = min_t * 0.2 # target_v

        # print("min_d", min_d)

        # # sinusoidal interpolation
        # for j in range(factor):
        #     k = j / float(factor) # will never hit 1.0
        #     a = 1-np.cos(np.pi*k)
        #     v = arr[i] + diff * a
        #     interp.append(v)

        distance_traveled = 0.0
        delta = np.abs(distance) / factor
        # print("DELTA", delta)
        # print("min_d", min_d)
        for j in range(factor):
            distance_traveled += delta

            if (distance_traveled + min_d >= distance):
                alpha = (distance_traveled - (distance - min_d)) / min_d
                # print("ALPHA", alpha)
                interp.append(arr[i] + alpha * diff)
            else:
                interp.append(arr[i])


            # if j < factor * 0.9:
            #     interp.append(arr[i])

            # else:
            #     v = arr[i+1]
            #     interp.append(v)

        res.extend(
            interp
            # np.linspace(arr[i], arr[i+1], factor, endpoint=False)
        )
        i += 1

    res.append(arr[-1])
    return res

def main():
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('file',
        type=str, help='file', default="/home/charlieyan1/Dev/jim/pymap2d/sdf_path_0000220.npy")
    parser.add_argument('--dist',
        type=float, default=-1.0)
    parser.add_argument('--headless',
        help="headless, default=False",
        action='store_true')
    parser.add_argument('--debug',
        help="headless, default=False",
        action='store_true')
    args = parser.parse_args()

    # print("arg.headless", args.headless)

    payload = np.load(
        # '/home/charlieyan1/Dev/jim/pymap2d/sdf_path_0000001.npy',
        # '/home/charlieyan1/Dev/jim/pymap2d/sdf_path_0000055.npy',
        # '/home/charlieyan1/Dev/jim/pymap2d/sdf_path_0000240.npy',
        args.file,
        allow_pickle=True).tolist()
    all_homs = payload['path']
    xythetas = np.array(
        [two_d_rvec_vec_from_matrix_2d(x)[0] for x in all_homs])

    ax = xythetas[:, 0]
    ay = xythetas[:, 1]

    # this is #important, the key is to set the yaw
    # for a wpt to align with the next path segment
    # ayaw = xythetas[:, 2]
    # this, in combination with slerp3
    ayaw = list(xythetas[1:, 2])
    ayaw.append(xythetas[-1, 2]) # note: maintain alignment on last waypoint

    # import ipdb; ipdb.set_trace()

    r2 = 0.5
    full_dist = r2 * max(1, len(ax) - 1)

    ##########################

    # square = 2.0
    # full_dist = square * 4

    '''
    # CW
    ax = np.array([
        ax[0],
        ax[0],
        ax[0] + square,
        ax[0] + square,
        ax[0]
        ])
    ay = np.array([
        ay[0],
        ay[0] + square,
        ay[0] + square,
        ay[0],
        ay[0]
        ])

    ayaw = np.array([
        np.pi / 2,
        0.0,
        -np.pi / 2,
        np.pi,
        np.pi, # note: maintain alignment on last waypoint
        ])
    '''

    '''
    # CCW
    ax = np.array([
        ax[0],

        ax[0] + square,

        ax[0] + square,

        ax[0],

        ax[0]
        ])
    ay = np.array([
        ay[0],

        ay[0],

        ay[0] + square,

        ay[0] + square,

        ay[0]
        ])

    ayaw = np.array([
        0.0,
        np.pi / 2,
        np.pi,
        -np.pi / 2,
        -np.pi / 2, # note: maintain alignment on last waypoint
        ])
    '''

    ##########################

    # cx = ax
    # cy = ay
    # cyaw = ayaw

    ##########################

    # bx, by, byaw, bk, s = cubic_spline_planner.calc_spline_course(
    #     ax, ay, ds=0.1)
    # cx = bx
    # cy = by
    # cyaw = byaw

    ##########################

    # import ipdb; ipdb.set_trace()

    ##########################

    cx = interpolate(ax, 40)
    cy = interpolate(ay, 40)

    ##########################

    # # INTERPOATION ROTAIONS SUCKS!
    # cyaw = [(x + np.pi) % (2 * np.pi) - np.pi for x in ayaw]
    # cyaw = rotation_smooth(cyaw)
    # # this is key
    # # interpolating between -3.14 and 3.14 through 0 is spatially wrong
    # cyaw = interpolate(cyaw, 40)

    # INTERPOATION ROTAIONS SUCKS!
    cyaw = [(x + np.pi) % (2 * np.pi) - np.pi for x in ayaw]
    cyaw = rotation_smooth(cyaw, args.debug)

    # import ipdb; ipdb.set_trace()

    # this is key
    # interpolating between -3.14 and 3.14 through 0 is spatially wrong
    xys = [np.array([ax[i], ay[i]]) for i in range(len(ax))]

    cyaw = slerp3(cyaw, xys, 40)
    # cyaw = do_repeat(cyaw, 40)

    if args.debug:
        print("len(cx)", len(cx))
        print("len(cy)", len(cy))
        print("len(cyaw)", len(cyaw))
        print("################################################")

    ##########################

    # key_times = [x * 40* dt for x in range(len(ax))]
    # key_rots = R.from_euler('z', ayaw, degrees=False)
    # slerp = Slerp(key_times, key_rots) 
    # sample_times = np.linspace(0.0, max(key_times), len(cx))
    # interp_rots = slerp(sample_times)
    # cyaw = interp_rots.as_euler('xyz', degrees=False)[:, -1]
    # print(len(cyaw))

    ##########################

    # cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
    #     ax, ay, ds=1.0)

    ##########################

    # velocities = [[0.0, 0.0]]
    # i = 0
    # while i < len(ax) - 1:
    #     dx = ax[i+1] - ax[i]
    #     dy = ay[i+1] - ay[i]
    #     velocities.append([dx, dy])
    #     i += 1
    # # velocities.append([0.0, 0.0])

    # # import ipdb; ipdb.set_trace()

    # ctrl_pts = [CtrlPt({
    #     0: np.array([ax[i], ay[i], 0.0]),
    #     # 1: np.array([velocities[i][0], velocities[i][1], 0.0])
    #     }) for i in range(len(ax))]
    # spliner = Spliner() # create a new spliner on each call
    # s_array, s_data = spliner.process(0, 8, ctrl_pts)

    # order = s_data['order']
    # num_dof = s_data['num_dof']
    # time, state = reformat_deriv_major(
    #     order,
    #     num_dof,
    #     s_array,
    #     s_data)
    # pos_order = state[0]
    # cx = pos_order[:, 0]
    # cy = pos_order[:, 1]

    # plt.plot(ax, ay, "xb", label="waypoints")
    # plt.plot(cx, cy, "-r", label="target course")
    # # plt.plot(x, y, "-g", label="tracking")
    # plt.scatter(ax, ay)
    # plt.grid(True)
    # plt.axis("equal")
    # plt.xlabel("x[m]")
    # plt.ylabel("y[m]")
    # plt.legend()
    # plt.show()

    # import ipdb; ipdb.set_trace()

    ##########################

    # print("LQR steering control tracking start!!")
    # ax = [0.0, 6.0, 12.5, 10.0, 17.5, 20.0, 25.0]
    # ay = [0.0, -3.0, -5.0, 6.5, 3.0, 0.0, 0.0]
    goal = [cx[-1], cy[-1]]

    # target_speed = 10.0 / 3.6  # simulation parameter km/h -> m/s
    # sp = calc_speed_profile(cyaw, target_speed)

    state = State(
        x=cx[0],
        y=cy[0],
        yaw=cyaw[0],
        v=0.0)

    tv = 0.2

    t, x, y, yaw, v, v_msg, w_msg, ais, distance_traveled, debug1, debug2, debug3, debug4, debug5, debug6, rmse, terminate_dist = do_simulation(
        state,
        cx, cy, cyaw,
        # ck, sp,
        goal,
        args.dist,
        tv,
        args,
        full_dist)

    print("RMSE!!! %.3f" % (rmse))

    if not args.headless:  # pragma: no cover
        plt.close()

        fig, a = plt.subplots(1)
        plt.plot(ax, ay, "xb", label="waypoints")
        plt.plot(cx, cy, "-r", label="target course")
        plt.plot(x, y, "-g", label="tracking")
        plt.scatter(ax, ay)

        for i in range(len(ax)):
            hom = two_d_make_x_y_theta_hom(ax[i], ay[i], ayaw[i])
            plot_gnomon(a, hom, mode=2, length=0.1, linewidth=2, c='b')

        for i in range(len(cx)):
            # print("xyyaw", cx[i], cy[i], cyaw[i])
            hom = two_d_make_x_y_theta_hom(cx[i], cy[i], cyaw[i])
            plot_gnomon(a, hom, mode=2, length=0.05, linewidth=1)

        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.title("%s: RMSE: %.3f, dist: %.3f" % (
            args.file,
            rmse,
            terminate_dist))

        fig, axs = plt.subplots(2, sharex=True)

        axs[0].plot(t, v, label="speed")
        axs[0].plot(t, v_msg, c='r', label="v_msg")
        axs[0].plot(t, w_msg, c='g', label="w_msg")
        axs[0].plot(t, ais, c='b', label="ais")
        # axs[0].plot(t, debug1, c='m', label="debug1")
        # axs[0].plot(t, yaw, c='k', label="yaw")

        # plt.plot(t, distance_traveled, c='b', label="distance_traveled")
        axs[0].grid(True)
        # axs[0].xlabel("Time [sec]")
        # axs[0].ylabel("Speed [m/s]")
        axs[0].legend()

        axs[1].plot(
            # [dt * i for i in range(len(cyaw))],
            # # [np.rad2deg(iyaw) for iyaw in cyaw],
            # cyaw,
            t,
            debug2,
            "-r",
            label="expected yaw")
        axs[1].plot(
            # list(range(len(cyaw))),
            # yaw[:len(cyaw)],
            t,
            yaw,
            c='k',
            label="yaw")
        axs[1].plot(
            # list(range(len(cyaw))),
            # yaw[:len(cyaw)],
            t,
            debug1,
            c='b',
            label="th_e")
        axs[1].plot(
            # list(range(len(cyaw))),
            # yaw[:len(cyaw)],
            t,
            debug3,
            c='g',
            label="fb")
        axs[1].plot(
            # list(range(len(cyaw))),
            # yaw[:len(cyaw)],
            t,
            debug4,
            c='c',
            label="dyaw")
        axs[1].plot(
            # list(range(len(cyaw))),
            # yaw[:len(cyaw)],
            t,
            debug5,
            c='y',
            label="dl")
        axs[1].plot(
            # list(range(len(cyaw))),
            # yaw[:len(cyaw)],
            t,
            debug6,
            '*c',
            label="state.e")

        # plt.plot(list(range(len(cyaw))), debug1, "-r", label="debug1")
        axs[1].grid(True)
        axs[1].legend()
        # axs[1].xlabel("line length[m]")
        # axs[1].ylabel("yaw angle[deg]")

        plt.show()

if __name__ == '__main__':
    main()
