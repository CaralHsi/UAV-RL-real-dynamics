import numpy as np
import ecos
import picos as pic
import cvxopt
import copy


class InitialTrajectory:
    def __init__(self):
        self.flag = False
        self.x = np.array([0, 1.27732865391054,        2.55465730782108])#,        3.83198596173162,        5.10931461564215,
                  #6.37504767981705,        8.12842158341820,        9.93494857991142,        11.7247115826235,
                  #13.5144745853355,        15.7495023354394])
        self.y = np.array([0,        1.68655568392241,        3.37311136784482])#,        5.05966705176722,
                  #6.74622273568963,        8.44379451715695,        9.63650668397935,        10.7231554374058,
                  #11.8513340399307,        12.9795126424556,        13.5831805739428])
        self.V = 5


class NoFlyZone:
    def __init__(self):
        self.M = 0
        self.Flag_NFZ = np.ones(self.M)
        self.x_NFZ = np.array([10, 15, 20, 21, 27, 40, 30, 12, 38, 30, 10.8])
        self.y_NFZ = np.array([5,  40, 30, 11, 22, 28, 36, 18, 15, 6,  29])
        self.a_NFZ = np.array([5,  8, 4.5, 6, 6.5, 5,  6,  8,  5,  4,  4])
        self.b_NFZ = np.array([5,  5, 4.5, 4, 5.5, 5,  6,  5,  8,  4,  5.5])


class InitialConfiguration:
    def __init__(self):
        self.v_max = 5
        self.v_min = 2
        self.at_max = 3
        self.ah_max = 7
        self.N = 2
        self.x0 = 0
        self.y0 = 0
        self.psi0 = 60 * np.pi / 180
        self.v0 = 5
        self.xf = 50
        self.yf = 50
        self.psif = -60 * np.pi / 180
        self.vf = 5
        self.sf = np.sqrt(np.square(self.xf - self.x0) + np.square(self.yf - self.y0)) - 20 - 30



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
        thita = np.pi + abs_thita
    return thita


def coordinate_transformation(x0, y0, psi0, xf, yf, psif, x1, y1, M):
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
    psif = psif - thita
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
    return x0, y0, psi0, xf, yf, psif, thita, x_initial, y_initial, x1_s, y1_s


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
    [x0, y0, psi0, xf, yf, psif, thita, x_initial, y_initial, x1, y1] \
        = coordinate_transformation(UAV_config.x0, UAV_config.y0, UAV_config.psi0, UAV_config.xf,
                                    UAV_config.yf, UAV_config.psif, no_fly_zone.x_NFZ, no_fly_zone.y_NFZ, no_fly_zone.M)
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
        print("================== iter={} ====================".format(iter))
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
            constraints.append(prob.add_constraint(y[(i + 1) * nx + 3] <= np.int(np.square(UAV_config.v_max))))
            constraints.append(prob.add_constraint(y[(i + 1) * nx + 3] >= np.int(np.square(UAV_config.v_min))))
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
            constraints.append(prob.add_constraint(s[i] + 1 / np.square(np.square(initial_trajectory.V)) * (y[i * nx + 3] - np.int(np.square(initial_trajectory.V))) <= 1 / np.square(initial_trajectory.V)))
            constraints.append(prob.add_constraint(y[i * nx + 3] == np.int(np.square(initial_trajectory.V))))

            if iter >= 2:
                constraints.append(prob.add_constraint(xf * y[i * nx + 0] - result[0, i] <= deta_x))
                constraints.append(prob.add_constraint(xf * y[i * nx + 0] - result[0, i] >= -deta_x))  # 可信赖域约束x
                constraints.append(prob.add_constraint(xf * y[i * nx + 1] - result[1, i] <= deta_y))
                constraints.append(prob.add_constraint(xf * y[i * nx + 1] - result[1, i] >= -deta_y))  # 可信赖域约束x
                constraints.append(prob.add_constraint(deta1[i] - detal_s[i] <= deta1_deta1))
                constraints.append(prob.add_constraint(deta1[i] - detal_s[i] >= -deta1_deta1))  # 可信赖域约束deta1
                constraints.append(prob.add_constraint(y[i * nx + 3] - np.int(np.square(initial_trajectory.V)) <= deta1_v))
                constraints.append(prob.add_constraint(y[i * nx + 3] - np.int(np.square(initial_trajectory.V)) >= -deta1_v))  # 可信赖域约束v
                constraints.append(prob.add_constraint(y[i * nx + 3] == np.int(np.square(initial_trajectory.V))))

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

        constraints.append(prob.add_constraint(y[nx * UAV_config.N + 1] * xf - yf <= k_y + 1e-5))
        constraints.append(prob.add_constraint(y[nx * UAV_config.N + 1] * xf - yf >= -k_y-1e-5))
        constraints.append(prob.add_constraint(y[nx * UAV_config.N + 0] * xf - xf <= k_x))
        constraints.append(prob.add_constraint(y[nx * UAV_config.N + 0] * xf - xf >= -k_x))
        constraints.append(prob.add_constraint(y[0] == x0 / xf))
        constraints.append(prob.add_constraint(y[1] == y0 / xf))
        constraints.append(prob.add_constraint(y[2] == np.sin(psi0)))
        constraints.append(prob.add_constraint(y[3] == np.int(np.square(UAV_config.v0))))
        constraints.append(prob.add_constraint(y[nx * UAV_config.N + 3] == np.int(np.square(UAV_config.vf))))
        constraints.append(prob.add_constraint(y[nx * UAV_config.N + 2] == np.sin(0)))
        coe1 = 10
        coe3 = 10
        coe4 = 10
        temp = np.matrix(np.ones([1, UAV_config.N]))
        prob.set_objective("min", coe1 * temp * k + coe3 * k_x + coe4 * k_y)
        solution = prob.solve(verbose=1, solver='ecos')
        print(prob)
        print(solution["status"])
        print(prob.obj_value())
        print(k_y)
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
        if iter >= 1:
            x_differ = np.zeros([UAV_config.N + 1, 1])
            y_differ = np.zeros([UAV_config.N + 1, 1])
            deta1_differ = np.zeros([UAV_config.N + 1, 1])
            v_ba_differ = np.zeros([UAV_config.N + 1, 1])
            for i in range(UAV_config.N):
                x_differ[i] = xf * y[i * nx + 0] - result[0, i]
                y_differ[i] = xf * y[i * nx + 1] - result[1, i]
                deta1_differ[i] = deta1[i] - detal_s[i]
                v_ba_differ[i] = v_ba[i] - np.square(initial_trajectory.V)
            deta_x = np.max(np.abs(x_differ))
            deta_y = np.max(np.abs(y_differ))
            deta1_deta1 = np.max(np.abs(deta1_differ))
            deta1_v = np.max(np.abs(v_ba_differ))

            print('Maximum difference between two successive solutions are:\n')
            print("x: {}, y:{}  deta1:{}   v_ba:{}".format(np.max(np.abs(x_differ)), np.max(np.abs(y_differ)),
                                                           np.max(np.abs(deta1_differ)), np.max(np.abs(v_ba_differ))))
            if np.max(np.abs(x_differ)) <= 1 and np.max(np.abs(y_differ)) <= 1 and \
                    np.max(np.abs(deta1_differ)) <= 0.1 and np.max(np.abs(v_ba_differ)) <= 1:
                flag = 0
            else:
                iter = iter + 1
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

        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(u12_square)
        ax.set_title("u12_square")
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1.2))
        ax.plot([0, 1], [1, 1])
        plt.show()
    return x_pos, y_pos, thita_value, v_value, at, ah, avoid_circle1


class MpcLayer:
    def __init__(self, env=None):
        self.initial_trajectory = InitialTrajectory()
        self.UAV_config = InitialConfiguration()
        self.NoFlyZone = NoFlyZone()
        self.env = env
        x_pos, y_pos, thita_value, v_value, at, ah, avoid_circle1 = optimization(self.UAV_config, self.NoFlyZone,
                                                                                 self.initial_trajectory)

    def get_safe_action(self, obs, action):
        self.initial_trajectory._set()
        x_pos, y_pos, thita_value, v_value, at, ah, avoid_circle1 = optimization(self.UAV_config, self.NoFlyZone,
                                                                                 self.initial_trajectory)
        action = None
        return action, False

if __name__ == '__main__':
    MpcLayer = MpcLayer()



