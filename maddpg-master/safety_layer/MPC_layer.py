import numpy as np
import ecos
import picos as pic
import cvxopt
import copy
from matplotlib import pyplot as plt
import matplotlib.patches as mpathes

class InitialTrajectory:
    def __init__(self):
        self.flag = False

    def _set(self, trajectory, v, n):
        self.x = trajectory[0, 0:n + 1]
        self.y = trajectory[1, 0:n + 1]
        self.V = v


class NoFlyZone:
    def __init__(self, r):
        self.M = 5
        self.agent_r = r

    def _set(self, obs):
        self.Flag_NFZ = np.ones(self.M)
        self.x_NFZ = np.array(np.zeros(self.M))
        self.y_NFZ = np.array(np.zeros(self.M))
        self.a_NFZ = np.array(np.zeros(self.M))
        self.b_NFZ = np.array(np.zeros(self.M))
        for i in range(self.M):
            self.x_NFZ[i] = obs[6 + i * 3 + 0] + obs[2]
            self.y_NFZ[i] = obs[6 + i * 3 + 1] + obs[3]
            self.a_NFZ[i] = obs[6 + i * 3 + 2] + self.agent_r
            self.b_NFZ[i] = obs[6 + i * 3 + 2] + self.agent_r
            if self.x_NFZ[i] == -1 and self.y_NFZ[i] == -1 and self.a_NFZ[i] == -1:
                self.M = i
                break
        self.x_NFZ = np.array(self.x_NFZ)
        self.y_NFZ = np.array(self.y_NFZ)
        self.a_NFZ = np.array(self.a_NFZ)
        self.b_NFZ = np.array(self.b_NFZ)


class InitialConfiguration:
    def __init__(self, v_max, v_min, at_max, ah_max, N):
        self.v_max = v_max  # 5
        self.v_min = v_min  # 2
        self.at_max = at_max  # 3
        self.ah_max = ah_max  # 7
        self.N = N  # 10
        # self.x0 = 0
        # self.y0 = 0
        # self.psi0 = 60 * np.pi / 180
        # self.v0 = 5
        # self.xf = 50
        # self.yf = 50
        # self.psif = -60 * np.pi / 180
        # self.vf = 5
        # self.sf = np.sqrt(np.square(self.xf - self.x0) + np.square(self.yf - self.y0)) - 20 - 30

    def _set(self, x0, y0, psi0, v0, xf, yf, vf):
        self.x0 = x0
        self.y0 = y0
        self.psi0 = psi0
        self.v0 = v0
        self.xf = xf
        self.yf = yf
        self.vf = vf
        self.sf = self.N * self.v0 * 1.2
        dist = np.sqrt(np.square(self.x0 - self.xf) + np.square(self.y0 - self.yf))
        if dist < self.sf:
            self.N = min(np.int(np.floor(dist/self.v0)), self.N)
            self.N = max(1, self.N)
            self.sf = self.N * self.v0 * 1.2


def get_angle_between_two_points(x0, y0, xf, yf):  # 已知两点，求两点连线与x轴的夹角（0，360）
    abs_thita = np.abs(np.arctan((yf - y0) / (xf - x0)))
    thita = 0
    if xf - x0 >= 0 and yf - y0 >= 0:  # 第一象限
        thita = abs_thita
    if xf - x0 >= 0 >= yf - y0:  # 第四象限
        thita = 2 * np.pi - abs_thita
    if xf - x0 <= 0 <= yf - y0:  # 第二象限
        thita = np.pi - abs_thita
    if xf - x0 <= 0 and yf - y0 <= 0:  # 第三象限
        thita =  np.pi + abs_thita
    return thita


def coordinate_transformation(x0, y0, psi0, xf, yf, x1, y1, M):
    x1_new = np.zeros(M)
    y1_new = np.zeros(M)
    x1_s = np.zeros(M)
    y1_s = np.zeros(M)
    # relative value
    x_initial = x0
    y_initial = y0
    xf = xf - x_initial
    yf = yf - y_initial
    x0 = x0 - x_initial
    y0 = y0 - y_initial
    thita = get_angle_between_two_points(x0, y0, xf, yf)
    xf_new = np.cos(-thita) * xf + -np.sin(-thita) * yf
    yf_new = np.sin(-thita) * xf + np.cos(-thita) * yf
    psi0 = psi0 - thita
    for m in range(M):
        x1_s[m] = x1[m] - x_initial
        y1_s[m] = y1[m] - y_initial
    for m in range(M):
        x1_new[m] = np.cos(-thita) * x1_s[m] + -np.sin(-thita) * y1_s[m]
        y1_new[m] = np.sin(-thita) * x1_s[m] + np.cos(-thita) * y1_s[m]
    xf = xf_new
    yf = yf_new
    x1_s = x1_new
    y1_s = y1_new
    return x0, y0, psi0, xf, yf, thita, x_initial, y_initial, x1_s, y1_s


def optimization(UAV_config, no_fly_zone, initial_trajectory, plot_procedure=False, plot_ending = False):
    nx = 4
    nu = 3
    Ts = UAV_config.sf / UAV_config.N
    ds_value = np.linspace(0, UAV_config.sf, num=UAV_config.N + 1)
    ele = nx * (UAV_config.N + 1) + nu * UAV_config.N
    No_Fly_zone_number = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '*']
    result = np.vstack((initial_trajectory.x, initial_trajectory.y))
    detal_s = np.zeros(UAV_config.N)
    for i in range(UAV_config.N):
        detal_s[i] = np.sqrt(UAV_config.ah_max / np.square(initial_trajectory.V))
    # coordinate transformation
    [x0, y0, psi0, xf, yf, thita, x_initial, y_initial, x1, y1] \
        = coordinate_transformation(UAV_config.x0, UAV_config.y0, UAV_config.psi0, UAV_config.xf,
                                    UAV_config.yf, no_fly_zone.x_NFZ, no_fly_zone.y_NFZ, no_fly_zone.M)
    np_ = 100
    avoid_circle1 = np.zeros([2, np_ + 1, no_fly_zone.M])  # no - fly - zone 1
    for j in range(no_fly_zone.M):
        for i in range(np_ + 1):
            new_x = np.cos(thita) * x1[j] + -np.sin(thita) * y1[j] + x_initial + \
                    no_fly_zone.a_NFZ[j] * np.cos(i / np_ * 2 * np.pi)  # no - fly - zone 1
            new_y = np.sin(thita) * x1[j] + np.cos(thita) * y1[j] + y_initial + no_fly_zone.a_NFZ[j] * np.sin(i / np_ * 2 * np.pi)
            avoid_circle1[0, i, j] = np.cos(-thita) * (new_x - x_initial) + -np.sin(-thita) * (new_y - y_initial)  # no - fly - zone 1
            avoid_circle1[1, i, j] = np.sin(-thita) * (new_x - x_initial) + np.cos(-thita) * (new_y - y_initial)
    for i in range(UAV_config.N + 1):
        new_x = np.cos(-thita) * (result[0, i] - x_initial) + -np.sin(-thita) * (result[1, i] - y_initial)
        new_y = np.sin(-thita) * (result[0, i] - x_initial) + np.cos(-thita) * (result[1, i] - y_initial)
        result[0, i] = new_x
        result[1, i] = new_y
    flag = True
    iter = 1
    deta_x = 0
    deta_y = 0
    deta1_deta1 = 0
    deta1_v = 0
    while flag:
        # print("================== iter={} ====================".format(iter))
        prob = pic.Problem()
        y = prob.add_variable('y', ele)
        deta = prob.add_variable('deta', UAV_config.N)
        deta1 = prob.add_variable('deta1', UAV_config.N)
        k = prob.add_variable('k', UAV_config.N)
        s = prob.add_variable('s', UAV_config.N)
        k_x = prob.add_variable('k_x', 1)
        k_y = prob.add_variable('k_y', 1)
        temp1 = prob.add_variable('temp1', (UAV_config.N, 2))
        temp2 = prob.add_variable('temp2', (UAV_config.N, 2))
        temp3 = prob.add_variable('temp3', (UAV_config.N, 2))
        temp4 = prob.add_variable('temp4', (UAV_config.N, 2))
        constraints = []
        for i in range(UAV_config.N):  # 动力学方程
            constraints.append(prob.add_constraint(xf * (y[(i + 1) * nx + 0] - y[i * nx + 0]) ==
                                                   Ts * y[(UAV_config.N + 1) * nx + i * nu + 0]))
            constraints.append(prob.add_constraint(xf * (y[(i + 1) * nx + 1] - y[i * nx + 1]) ==
                                                   Ts * y[i * nx + 2]))
            constraints.append(prob.add_constraint(y[(i + 1) * nx + 2] - y[i * nx + 2] ==
                                                   Ts * y[(UAV_config.N + 1) * nx + i * nu + 1]))
            constraints.append(prob.add_constraint(y[(i + 1) * nx + 3] - y[i * nx + 3] ==
                                                   Ts * 2 * y[(UAV_config.N + 1) * nx + i * nu + 2]))
        for i in range(UAV_config.N):
            constraints.append(prob.add_constraint(y[i * nx + 2] == temp1[i, 0]))
            constraints.append(prob.add_constraint(y[(UAV_config.N + 1) * nx + i * nu + 0] == temp1[i, 1]))
            constraints.append(prob.add_constraint(pic.norm(temp1[i, :], 2) <= 1))
            constraints.append(prob.add_constraint(y[(i + 1) * nx + 3] <= np.float(np.square(UAV_config.v_max))))
            constraints.append(prob.add_constraint(y[(i + 1) * nx + 3] >= np.float(np.square(UAV_config.v_min))))
            constraints.append(prob.add_constraint(y[(UAV_config.N + 1) * nx + i * nu + 2] <=
                                                   UAV_config.at_max))
            constraints.append(prob.add_constraint(y[(UAV_config.N + 1) * nx + i * nu + 2] >=
                                                   -UAV_config.at_max))
            constraints.append(prob.add_constraint(1 / np.sqrt(2) - y[i * nx + 3] / np.sqrt(2) == temp2[i, 0]))
            constraints.append(prob.add_constraint(np.sqrt(2) * deta[i] == temp2[i, 1]))
            constraints.append(prob.add_constraint(pic.norm(temp2[i, :], 2) <= 1 / np.sqrt(2) + y[i * nx + 3] / np.sqrt(2)))
            constraints.append(prob.add_constraint(k[i] / np.sqrt(2) - deta[i] / np.sqrt(2) == temp3[i, 0]))
            constraints.append(prob.add_constraint(np.sqrt(2) == temp3[i, 1]))
            constraints.append(prob.add_constraint(pic.norm(temp3[i, :], 2) <= k[i] / np.sqrt(2) + deta[i] / np.sqrt(2)))
            constraints.append(prob.add_constraint(y[(UAV_config.N + 1) * nx + i * nu + 0] /
                                                   np.sqrt(2) - s[i] / np.sqrt(2) == temp4[i, 0]))
            constraints.append(prob.add_constraint(np.sqrt(2 / UAV_config.ah_max) * deta1[i] == temp4[i, 1]))
            constraints.append(prob.add_constraint(pic.norm(temp4[i, :], 2) <= y[(UAV_config.N + 1) * nx + i * nu + 0] / np.sqrt(2) + s[i] / np.sqrt(2)))
            constraints.append(prob.add_constraint(y[(UAV_config.N + 1) * nx + i * nu + 1] <= np.square(detal_s[i]) +
                                                                                                         2 * detal_s[i] * (deta1[i] - detal_s[i])))
            constraints.append(prob.add_constraint(y[(UAV_config.N + 1) * nx + i * nu + 1] >= -np.square(detal_s[i]) -
                                                                                                         2 * detal_s[i] * (deta1[i] - detal_s[i])))
            constraints.append(prob.add_constraint(s[i] + 1 / np.square(np.square(initial_trajectory.V)) * (y[i * nx + 3] - np.float(np.square(initial_trajectory.V))) <= 1 / np.square(initial_trajectory.V)))
            constraints.append(prob.add_constraint(y[i * nx + 3] == np.float(np.square(initial_trajectory.V))))

            if iter >= 2:
                constraints.append(prob.add_constraint(xf * y[i * nx + 0] - result[0, i] <= deta_x))
                constraints.append(prob.add_constraint(xf * y[i * nx + 0] - result[0, i] >= -deta_x))  # 可信赖域约束x
                constraints.append(prob.add_constraint(xf * y[i * nx + 1] - result[1, i] <= deta_y))
                constraints.append(prob.add_constraint(xf * y[i * nx + 1] - result[1, i] >= -deta_y))  # 可信赖域约束x
                constraints.append(prob.add_constraint(deta1[i] - detal_s[i] <= deta1_deta1))
                constraints.append(prob.add_constraint(deta1[i] - detal_s[i] >= -deta1_deta1))  # 可信赖域约束deta1
                constraints.append(prob.add_constraint(y[i * nx + 3] - np.float(np.square(initial_trajectory.V)) <= deta1_v))
                constraints.append(prob.add_constraint(y[i * nx + 3] - np.float(np.square(initial_trajectory.V)) >= -deta1_v))  # 可信赖域约束v
                constraints.append(prob.add_constraint(y[i * nx + 3] == np.float(np.square(initial_trajectory.V))))

        for j in range(no_fly_zone.M):
            if no_fly_zone.Flag_NFZ[j] == 1:
                for i in range(1, UAV_config.N + 1):
                    x_i = result[0, i]  # 0 1
                    y_i = result[1, i]
                    A1 = np.cos(thita)
                    B1 = -np.sin(thita)
                    C1 = x_initial - (np.cos(thita) * x1[j] - np.sin(thita) * y1[j] + x_initial)
                    A2 = np.sin(thita)
                    B2 = np.cos(thita)
                    C2 = y_initial - (np.sin(thita) * x1[j] + np.cos(thita) * y1[j] + y_initial)
                    af_ax = 2 * A1 * (A1 * x_i + B1 * y_i + C1) / np.square(no_fly_zone.a_NFZ[j]) + 2 * A2 * (
                                A2 * x_i + B2 * y_i + C2) / np.square(no_fly_zone.b_NFZ[j])
                    af_ay = 2 * B1 * (A1 * x_i + B1 * y_i + C1) / np.square(no_fly_zone.a_NFZ[j]) + 2 * B2 * (
                                A2 * x_i + B2 * y_i + C2) / np.square(no_fly_zone.b_NFZ[j])
                    f_xy = np.square((A1 * x_i + B1 * y_i + C1)) / np.square(no_fly_zone.a_NFZ[j]) + (
                                np.square(A2 * x_i + B2 * y_i + C2)) / np.square(no_fly_zone.b_NFZ[j]) - 1
                    constraints.append(prob.add_constraint(f_xy + af_ax * (xf * y[i * nx + 0] - x_i) +
                                                           af_ay * (xf * y[i * nx + 1] - y_i) >= 0))

        constraints.append(prob.add_constraint(y[nx * UAV_config.N + 1] * xf - yf <= k_y))
        constraints.append(prob.add_constraint(y[nx * UAV_config.N + 1] * xf - yf >= -k_y))
        constraints.append(prob.add_constraint(y[nx * UAV_config.N + 0] * xf - xf <= k_x))
        constraints.append(prob.add_constraint(y[nx * UAV_config.N + 0] * xf - xf >= -k_x))
        constraints.append(prob.add_constraint(y[0] == x0 / xf))
        constraints.append(prob.add_constraint(y[1] == y0 / xf))
        constraints.append(prob.add_constraint(y[2] == np.sin(psi0)))
        constraints.append(prob.add_constraint(y[3] == np.float(np.square(UAV_config.v0))))
        constraints.append(prob.add_constraint(y[nx * UAV_config.N + 3] == np.float(np.square(UAV_config.vf))))
        coe1 = 10
        coe3 = 10
        coe4 = 10
        coe2 = 10
        temp = np.matrix(np.ones([1, UAV_config.N]))
        prob.set_objective("min", coe1 * temp * k +
                           coe3 * k_x + coe4 * k_y +
                           coe2 * (result[0, -1] - y[nx * UAV_config.N + 0] * xf) *
                           (result[0, -1] - y[nx * UAV_config.N + 0] * xf) +
                           coe2 * (result[1, -1] - y[nx * UAV_config.N + 1] * xf) *
                           (result[1, -1] - y[nx * UAV_config.N + 1] * xf))
        solution = prob.solve(verbose=0, solver='ecos')
        '''print(prob)
        print(solution["status"])
        print(prob.obj_value())
        print(k_y)'''
        y = y.value
        k = k.value
        s = s.value
        k_x = k_x.value
        k_y = k_y.value
        deta = deta.value
        deta1 = deta1.value
        x_pos = xf * y[0: (UAV_config.N + 1) * nx: nx]
        y_pos = xf * y[1: (UAV_config.N + 1) * nx: nx]
        fai_s = np.array(y[2: (UAV_config.N + 1) * nx: nx])
        v_ba = np.array(y[3: (UAV_config.N + 1) * nx: nx])
        fai_c = np.array(y[(UAV_config.N + 1) * nx + 0: (UAV_config.N + 1) * nx + UAV_config.N * nu: nu])
        fai_d = np.array(y[(UAV_config.N + 1) * nx + 1: (UAV_config.N + 1) * nx + UAV_config.N * nu: nu])
        at = np.array(y[(UAV_config.N + 1) * nx + 2: (UAV_config.N + 1) * nx + UAV_config.N * nu: nu])
        u12_square = np.array([(np.square(fai_c[i]) + np.square(fai_s[i])) for i in range(UAV_config.N)])
        v_value = np.array([np.sqrt(v_ba[i, 0]) for i in range(UAV_config.N)])
        ah = v_ba[0:UAV_config.N] * fai_d / fai_c
        thita_value = np.arcsin(fai_s)
        temp1 = coe1 * temp * np.matrix(k)
        if iter >= 1:
            x_differ = np.zeros([UAV_config.N + 1, 1])
            y_differ = np.zeros([UAV_config.N + 1, 1])
            deta1_differ = np.zeros([UAV_config.N + 1, 1])
            v_ba_differ = np.zeros([UAV_config.N + 1, 1])
            for i in range(UAV_config.N):
                x_differ[i] = xf * y[i * nx + 0] - result[0, i]
                y_differ[i] = xf * y[i * nx + 1] - result[1, i]
                if UAV_config.N == 1:
                    deta1 = np.array(deta1)
                deta1_differ[i] = deta1[i] - detal_s[i]
                v_ba_differ[i] = v_ba[i] - np.square(initial_trajectory.V)
            deta_x = np.max(np.abs(x_differ))
            deta_y = np.max(np.abs(y_differ))
            deta1_deta1 = np.max(np.abs(deta1_differ))
            deta1_v = np.max(np.abs(v_ba_differ))

            # print('Maximum difference between two successive solutions are:\n')
            # print("x: {}, y:{}  deta1:{}   v_ba:{}".format(np.max(np.abs(x_differ)), np.max(np.abs(y_differ)),
            #                                               np.max(np.abs(deta1_differ)), np.max(np.abs(v_ba_differ))))
            if np.max(np.abs(x_differ)) <= 1 and np.max(np.abs(y_differ)) <= 1 and \
                    np.max(np.abs(deta1_differ)) <= 0.1 and np.max(np.abs(v_ba_differ)) <= 1:
                flag = 0
            else:
                iter = iter + 1
                if iter > 5:
                    break
                for i in range(UAV_config.N):
                    detal_s[i] = deta1[i]
                result = np.squeeze(np.array([x_pos, y_pos]))
    x_pos_new = np.cos(thita) * x_pos + -np.sin(thita) * y_pos + x_initial
    y_pos_new = np.sin(thita) * x_pos + np.cos(thita) * y_pos + y_initial
    xf_new = np.cos(thita) * xf + -np.sin(thita) * yf + x_initial
    yf_new = np.sin(thita) * xf + np.cos(thita) * yf + y_initial
    x_pos = x_pos_new
    y_pos = y_pos_new
    thita_value = thita_value + thita
    x1_new = np.zeros([no_fly_zone.M, 1])
    y1_new = np.zeros([no_fly_zone.M, 1])
    for i in range(no_fly_zone.M):
        x1_new[i] = np.cos(thita) * x1[i] + -np.sin(thita) * y1[i] + x_initial
        y1_new[i] = np.sin(thita) * x1[i] + np.cos(thita) * y1[i] + y_initial
        x1[i] = x1_new[i]
        y1[i] = y1_new[i]
    np_ = 100
    avoid_circle1 = np.zeros([2, np_ + 1, no_fly_zone.M])  # no - fly - zone 1
    for j in range(no_fly_zone.M):
        for i in range(np_ + 1):
            avoid_circle1[0, i, j] = x1[j] + no_fly_zone.a_NFZ[j] * np.cos(i / np_ * 2 * np.pi)  # no - fly - zone 1
            avoid_circle1[1, i, j] = y1[j] + no_fly_zone.b_NFZ[j] * np.sin(i / np_ * 2 * np.pi)

    '''fig, ax = plt.subplots()
    ax.plot(u12_square)
    ax.set_title("u12_square")
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1.2))
    ax.plot([0, 1], [1, 1])
    plt.show()'''
    return x_pos, y_pos, thita_value, v_value, at, ah, avoid_circle1


class MpcLayer:
    def __init__(self, env=None):
        self.env = env
        v_max = self.env.agents[0].state.p_vel
        v_min = self.env.agents[0].state.p_vel
        at_max = 0.1
        ah_max = 0.7
        self.N = 2
        self.initial_trajectory = InitialTrajectory()
        self.UAV_config = InitialConfiguration(v_max, v_min, at_max, ah_max, self.N)
        self.NoFlyZone = NoFlyZone(self.env.world.agents[0].size)

    def get_safe_action(self, obs, action, trajectory):
        x0 = self.env.agents[0].state.p_pos[0]
        y0 = self.env.agents[0].state.p_pos[1]
        psi0 = self.env.agents[0].state.theta
        v0 = self.env.agents[0].state.p_vel
        xf = self.env.world.landmarks[-1].state.p_pos[0]
        yf = self.env.world.landmarks[-1].state.p_pos[1]
        vf = self.env.agents[0].state.p_vel
        self.UAV_config._set(x0, y0, psi0, v0, xf, yf, vf)
        self.NoFlyZone._set(obs)
        self.initial_trajectory._set(trajectory, v0, self.UAV_config.N)
        x_pos, y_pos, thita_value, v_value, at, ah, avoid_circle1 = optimization(self.UAV_config, self.NoFlyZone,
                                                                               self.initial_trajectory)
        '''fig, ax0 = plt.subplots()
        for i, landmark in enumerate(self.env.world.landmarks):
            p_pos = landmark.state.p_pos
            r = landmark.size
            circle = mpathes.Circle(p_pos, r)
            ax0.add_patch(circle)
        for i in range(self.UAV_config.N):
            p_pos = np.array([x_pos[i], y_pos[i]])
            r = self.env.world.agents[0].size
            circle = mpathes.Circle(p_pos, r)
            ax0.add_patch(circle)
        ax0.set_xlim((-1, 40))
        ax0.set_ylim((-10.3, 10.3))
        ax0.axis('equal')
        plt.show()'''
        '''data_save = np.concatenate((self.NoFlyZone.a_NFZ, self.NoFlyZone.x_NFZ,
                                    self.NoFlyZone.y_NFZ, self.initial_trajectory.x,
                                    self.initial_trajectory.y))
        data_save = np.array(data_save)
        np.savetxt("data_save.txt", data_save)'''
        theta_ = self.env.agents[0].state.theta
        omega_ = self.env.agents[0].state.omega
        theta_MPC = np.arctan((y_pos[1] - y_pos[0])/(x_pos[1] - x_pos[0]))
        omega_MPC = (theta_MPC - theta_)/self.env.world.dt
        d_omega_MPC = (omega_MPC - omega_)/self.env.world.dt
        d_omega = action[3] - action[4]
        delta_action = d_omega_MPC / 0.12 - d_omega
        action[3] = action[3] + delta_action / 2
        action[4] = action[4] - delta_action / 2
        self.UAV_config.N = self.N
        return action, False




