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
        self.theta = trajectory[2, 0:n + 1]
        self.omega = trajectory[3, 0:n + 1]
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
            self.a_NFZ[i] = obs[6 + i * 3 + 2] + 1.3 * self.agent_r
            self.b_NFZ[i] = obs[6 + i * 3 + 2] + 1.3 * self.agent_r
            if self.x_NFZ[i] == -1 and self.y_NFZ[i] == -1 and self.a_NFZ[i] == -1:
                self.M = i
                break
        self.x_NFZ = np.array(self.x_NFZ)
        self.y_NFZ = np.array(self.y_NFZ)
        self.a_NFZ = np.array(self.a_NFZ)
        self.b_NFZ = np.array(self.b_NFZ)


class InitialConfiguration:
    def __init__(self, v, N):
        self.v = v
        self.N = N  # 10

    def _set(self, x0, y0, theta0, omega0, xf, yf):
        self.x0 = x0
        self.y0 = y0
        self.theta0 = theta0
        self.omega0 = omega0
        self.xf = xf
        self.yf = yf
        self.sf = self.N * self.v * 1.2
        dist = np.sqrt(np.square(self.x0 - self.xf) + np.square(self.y0 - self.yf))
        if dist < self.sf:
            self.N = min(np.int(np.floor(dist/self.v)), self.N)
            self.N = max(1, self.N)
            self.sf = self.N * self.v * 1.2


def optimization(UAV_config, no_fly_zone, initial_trajectory, plot_procedure=False, plot_ending = False):
    nx = 4
    nu = 1
    ele = nx * (UAV_config.N + 1) + nu * UAV_config.N
    result = np.vstack((initial_trajectory.x, initial_trajectory.y, initial_trajectory.theta,
                        initial_trajectory.omega))
    # coordinate transformation
    np_ = 100
    avoid_circle1 = np.zeros([2, np_ + 1, no_fly_zone.M])  # no - fly - zone 1
    for j in range(no_fly_zone.M):
        for i in range(np_ + 1):
            avoid_circle1[0, i, j] = no_fly_zone.x_NFZ[j] + no_fly_zone.a_NFZ[j] * np.cos(i / np_ * 2 * np.pi)
            avoid_circle1[1, i, j] = no_fly_zone.y_NFZ[j] + no_fly_zone.a_NFZ[j] * np.sin(i / np_ * 2 * np.pi)
    flag = True
    iter = 1
    deta_x = 0
    deta_y = 0
    deta_theta = 0
    deta_omega = 0
    x_pos = result[0, :]
    y_pos = result[1, :]
    theta_pos = result[2, :]
    omega_pos = result[3, :]
    while flag:
        # print("================== iter={} ====================".format(iter))
        prob = pic.Problem()
        y = prob.add_variable('y', ele)
        y1 = prob.add_variable('y1', UAV_config.N + 1)
        y2 = prob.add_variable('y2', UAV_config.N + 1)
        constraints = []
        for i in range(UAV_config.N):  # 动力学方程
            constraints.append(prob.add_constraint(y[(i + 1) * nx + 0] - y[i * nx + 0] ==
                                                   UAV_config.v * (-np.sin(result[2, i + 1])) *
                                                   (y[(i + 1) * nx + 2] - result[2, i + 1]) + UAV_config.v *
                                                   np.cos(result[2, i + 1])))
            constraints.append(prob.add_constraint(y[(i + 1) * nx + 1] - y[i * nx + 1] ==
                                                   UAV_config.v * np.cos(result[2, i + 1]) *
                                                   (y[(i + 1) * nx + 2] - result[2, i + 1]) + UAV_config.v *
                                                   np.sin(result[2, i + 1])))
            constraints.append(prob.add_constraint(y[(i + 1) * nx + 2] - y[i * nx + 2] ==
                                                   y[(i + 1) * nx + 3]))
            constraints.append(prob.add_constraint(y[(i + 1) * nx + 3] - y[i * nx + 3] ==
                                                   y[(UAV_config.N + 1) * nx + i * nu + 0]))
        for i in range(UAV_config.N):
            constraints.append(prob.add_constraint(y[(UAV_config.N + 1) * nx + i * nu + 0] >= -0.24))
            constraints.append(prob.add_constraint(y[(UAV_config.N + 1) * nx + i * nu + 0] <= 0.24))
            constraints.append(prob.add_constraint(y[i * nx + 0] - result[0, i] == y1[i]))
            constraints.append(prob.add_constraint(y[i * nx + 1] - result[1, i] == y2[i]))
            if iter >= 2:
                constraints.append(prob.add_constraint(y[i * nx + 0] - result[0, i] <= deta_x))
                constraints.append(prob.add_constraint(y[i * nx + 0] - result[0, i] >= -deta_x))
                constraints.append(prob.add_constraint(y[i * nx + 1] - result[1, i] <= deta_y))
                constraints.append(prob.add_constraint(y[i * nx + 1] - result[1, i] >= -deta_y))
                # constraints.append(prob.add_constraint(y[i * nx + 2] - result[2, i] <= deta_theta))
                # constraints.append(prob.add_constraint(y[i * nx + 2] - result[2, i] <= deta_theta))
                # constraints.append(prob.add_constraint(y[i * nx + 3] - result[3, i] <= deta_omega))
                # constraints.append(prob.add_constraint(y[i * nx + 3] - result[3, i] <= deta_omega))
        for j in range(no_fly_zone.M):
            if no_fly_zone.Flag_NFZ[j] == 1:
                for i in range(1, UAV_config.N + 1):
                    af_ax = 2 * (result[0, i] - no_fly_zone.x_NFZ[j]) / np.square(no_fly_zone.a_NFZ[j])
                    af_ay = 2 * (result[1, i] - no_fly_zone.y_NFZ[j]) / np.square(no_fly_zone.b_NFZ[j])
                    f_xy = np.square((result[0, i] - no_fly_zone.x_NFZ[j])) / np.square(no_fly_zone.a_NFZ[j]) + (
                                np.square(result[1, i] - no_fly_zone.y_NFZ[j])) / np.square(no_fly_zone.b_NFZ[j]) - 1
                    constraints.append(prob.add_constraint(f_xy + af_ax * (y[i * nx + 0] - result[0, i]) +
                                                           af_ay * (y[i * nx + 1] - result[1, i]) >= 0))
        constraints.append(prob.add_constraint(y[0] == UAV_config.x0))
        constraints.append(prob.add_constraint(y[1] == UAV_config.y0))
        constraints.append(prob.add_constraint(y[2] == UAV_config.theta0))
        constraints.append(prob.add_constraint(y[3] == UAV_config.omega0))
        constraints.append(prob.add_constraint(y[UAV_config.N * nx + 0] - result[0, -1] == y1[UAV_config.N]))
        constraints.append(prob.add_constraint(y[UAV_config.N * nx + 1] - result[1, -1] == y2[UAV_config.N]))

        coe1 = 10
        prob.set_objective("min", coe1 * sum([y1[i] * y1[i] for i in range(UAV_config.N + 1)]) +
                           coe1 * sum([y2[i] * y2[i] for i in range(UAV_config.N + 1)]))
        '''
                prob.set_objective("min", coe1 * temp * (y[0: (UAV_config.N + 1) * nx: nx] - result[0, :]) *
                           (y[0: (UAV_config.N + 1) * nx: nx] - result[0, :]).T +
                           coe1 * temp * (y[1: (UAV_config.N + 1) * nx: nx] - result[1, :]) *
                           (y[1: (UAV_config.N + 1) * nx: nx] - result[1, :]).T)
        '''
        solution = prob.solve(verbose=0, solver='ecos')
        '''print(prob)
        print(solution["status"])
        print(prob.obj_value())
        print(k_y)'''
        y = y.value
        x_pos = np.array(y[0: (UAV_config.N + 1) * nx: nx])
        y_pos = np.array(y[1: (UAV_config.N + 1) * nx: nx])
        theta_pos = np.array(y[2: (UAV_config.N + 1) * nx: nx])
        omega_pos = np.array(y[3: (UAV_config.N + 1) * nx: nx])
        d_omega_pos = np.array(y[(UAV_config.N + 1) * nx: (UAV_config.N + 1) * nx + UAV_config.N])
        if iter >= 1:
            x_differ = np.zeros([UAV_config.N + 1, 1])
            y_differ = np.zeros([UAV_config.N + 1, 1])
            theta_differ = np.zeros([UAV_config.N + 1, 1])
            omega_differ = np.zeros([UAV_config.N + 1, 1])
            for i in range(UAV_config.N):
                x_differ[i] =y[i * nx + 0] - result[0, i]
                y_differ[i] = y[i * nx + 1] - result[1, i]
                theta_differ[i] = y[i * nx + 2] - result[2, i]
                omega_differ[i] = y[i * nx + 3] - result[3, i]
            deta_x = np.max(np.abs(x_differ))
            deta_y = np.max(np.abs(y_differ))
            deta_theta = np.max(np.abs(theta_differ))
            deta_omega = np.max(np.abs(omega_differ))

            # print('Maximum difference between two successive solutions are:\n')
            # print("x: {}, y:{}  deta1:{}   v_ba:{}".format(np.max(np.abs(x_differ)), np.max(np.abs(y_differ)),
            #                                               np.max(np.abs(deta1_differ)), np.max(np.abs(v_ba_differ))))
            if np.max(np.abs(x_differ)) <= 0.005 and np.max(np.abs(y_differ)) <= 0.005 and \
                    np.max(np.abs(theta_differ)) <= 0.01 and np.max(np.abs(omega_differ)) <= 0.01:
                flag = 0
            else:
                iter = iter + 1
                if iter > 5:
                    break
                result = np.squeeze(np.array([x_pos, y_pos, theta_pos, omega_pos]))
    return x_pos, y_pos, theta_pos, omega_pos, d_omega_pos


class MpcLayer:
    def __init__(self, env=None):
        self.env = env
        v = self.env.agents[0].state.p_vel
        self.N = 6
        self.initial_trajectory = InitialTrajectory()
        self.UAV_config = InitialConfiguration(v, self.N)
        self.NoFlyZone = NoFlyZone(self.env.world.agents[0].size)

    def get_safe_action(self, obs, action, trajectory):
        x0 = self.env.agents[0].state.p_pos[0]
        y0 = self.env.agents[0].state.p_pos[1]
        theta0 = self.env.agents[0].state.theta
        omega0 = self.env.agents[0].state.omega
        v = self.env.agents[0].state.p_vel
        xf = self.env.world.landmarks[-1].state.p_pos[0]
        yf = self.env.world.landmarks[-1].state.p_pos[1]
        self.UAV_config._set(x0, y0, theta0, omega0, xf, yf)
        self.NoFlyZone._set(obs)
        self.initial_trajectory._set(trajectory, v, self.UAV_config.N)
        x_pos, y_pos, theta_pos, omega_pos, d_omega_pos = optimization(self.UAV_config, self.NoFlyZone,
                                                                               self.initial_trajectory)
        '''fig, ax0 = plt.subplots()
        for i, landmark in enumerate(self.env.world.landmarks):
            p_pos = landmark.state.p_pos
            r = landmark.size
            circle = mpathes.Circle(p_pos, r)
            ax0.add_patch(circle)
        for i in range(self.UAV_config.N + 1):
            p_pos = np.array([self.initial_trajectory.x[i], self.initial_trajectory.y[i]])
            r = self.env.world.agents[0].size
            circle = mpathes.Circle(p_pos, r, facecolor='lightsalmon', edgecolor='orangered')
            ax0.add_patch(circle)
        for i in range(self.UAV_config.N + 1):
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
        '''theta_MPC = np.arctan((y_pos[1] - y_pos[0])/(x_pos[1] - x_pos[0]))
        omega_MPC = (theta_MPC - theta_)/self.env.world.dt
        d_omega_MPC = (omega_MPC - omega_)/self.env.world.dt'''

        theta_MPC = theta_pos[1]
        omega_MPC = omega_pos[1]
        d_omega_MPC = d_omega_pos[0]
        omega__ = omega_ + d_omega_MPC * 1
        theta__ = theta_ + omega__ * 1
        x_pos_ = np.float(x_pos[0])
        y_pos_ = np.float(y_pos[0])
        x_pos_ += np.cos(theta__) * 0.2
        y_pos_ += np.sin(theta__) * 0.2

        d_omega = action[3] - action[4]
        delta_action = d_omega_MPC / 0.12 - d_omega
        action[3] = action[3] + delta_action / 2
        action[4] = action[4] - delta_action / 2
        self.UAV_config.N = self.N


        p_pos_0_real = np.float(x_pos[0]) + 0.2 * np.cos(theta_MPC)
        p_pos_1_real = np.float(y_pos[0]) + 0.2 * np.sin(theta_MPC)
        temp = np.concatenate((p_pos_0_real, p_pos_1_real))

        for i, landmark in enumerate(self.env.world.landmarks[0:-1]):
            dist = np.sqrt(np.sum(np.square(temp - landmark.state.p_pos))) -\
                   (self.env.agents[0].size + landmark.size)
            dist__ = np.sqrt(np.sum(np.square(np.concatenate((x_pos[1], y_pos[1]))
                                              - landmark.state.p_pos))) -\
                     (self.env.agents[0].size + landmark.size)
            if i == 0:
                dist_ = dist
            if dist <= 0:
                print(0, ' ', dist, dist__)
                print(np.sqrt(np.sum(np.square(temp - np.concatenate((x_pos[1], y_pos[1]))))))
        #print("dist, p_pos_0_real, p_pos_1_real, theta_MPC, omega_MPC, d_omega_MPC: \n",
              #dist_, p_pos_0_real, p_pos_1_real, theta_MPC, omega_MPC, d_omega_MPC)

        return action, True




