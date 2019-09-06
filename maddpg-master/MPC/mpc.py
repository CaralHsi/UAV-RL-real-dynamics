import numpy as np
import ecos
import picos as pic
import cvxopt
import copy
from matplotlib import pyplot as plt
import matplotlib.patches as mpathes
import scipy.sparse as sp


def equality_curve(n, sequence):
    """
    :param n: n + 1 points ready to be converted
    :param sequence: the input sequence to be converted
    :return: this function converts the sequence to another sequence with (n + 1) points in the
             equal interval.
    """
    result = np.zeros([n + 1, 2])  # initial result
    result[0, :] = sequence[0, :]  # the front row is the same
    result[n, :] = sequence[-1, :]  # the last tow is the same
    r = np.shape(sequence)[0]  # the number of rows

    # the total distance
    save = np.zeros(r - 1)
    distance = 0
    for i in range(r - 1):
        save[i] = np.sqrt((sequence[i + 1, 1] - sequence[i, 1]) ** 2 +
                          (sequence[i + 1, 0] - sequence[i, 0]) ** 2)
        distance += save[i]

    # the angle at each point
    angle = np.zeros(r - 1)
    for i in range(r - 1):
        if sequence[i, 0] == sequence[i + 1, 0]:
            angle[i] = np.pi / 2
        else:
            angle[i] = np.arctan((sequence[i + 1, 1] - sequence[i, 1]) /
                                 (sequence[i + 1, 0] - sequence[i, 0]))

    # separate to n points
    theta = distance/n
    k = 1
    s = 0

    while True:
        sum = 0
        j = 0
        for i in range(s, r-1):
            sum += save[i]
            if sum > theta:
                j = i
                break
        kk = save[j] - (sum - theta)
        x_new = None
        y_new = None
        if (sequence[j + 1, 0] <= sequence[j, 0] and sequence[j + 1, 1] >= sequence[j, 1]) or \
                (sequence[j + 1, 0] <= sequence[j, 0] and sequence[j + 1, 1] <= sequence[j, 1]):
            x_new = sequence[j, 0] - kk * np.cos(angle[j])
            y_new = sequence[j, 1] - kk * np.sin(angle[j])
        if (sequence[j + 1, 0] >= sequence[j, 0] and sequence[j + 1, 1] <= sequence[j, 1]) or \
                (sequence[j + 1, 0] >= sequence[j, 0] and sequence[j + 1, 1] >= sequence[j, 1]):
            x_new = sequence[j, 0] + kk * np.cos(angle[j])
            y_new = sequence[j, 1] + kk * np.sin(angle[j])
        save[j] = np.sqrt((sequence[j + 1, 1] - y_new) ** 2 + (sequence[j + 1, 0] - x_new) ** 2)
        if sequence[j, 0] == sequence[j + 1, 0]:
            angle[j] = np.pi / 2
        else:
            angle[j] = np.arctan((sequence[j + 1, 1] - y_new) / (sequence[j + 1, 0] - x_new))

        sequence[j, 0] = x_new
        sequence[j, 1] = y_new
        result[k, 0] = x_new
        result[k, 1] = y_new
        k += 1
        s = j
        if k > n - 1:
            break

    # return
    return result, distance


class InitialTrajectory:
    def __init__(self, Config, NoFlyZone):
        self.Config = Config
        self.NoFlyZone = NoFlyZone
        self.flag = False
        self.x = np.array([0, 1.27732865391054,        2.55465730782108,        3.83198596173162,        5.10931461564215,
                  6.37504767981705,        8.12842158341820,        9.93494857991142,        11.7247115826235,
                  13.5144745853355,        15.7495023354394])
        self.y = np.array([0,        1.68655568392241,        3.37311136784482,        5.05966705176722,
                  6.74622273568963,        8.44379451715695,        9.63650668397935,        10.7231554374058,
                  11.8513340399307,        12.9795126424556,        13.5831805739428])
        self.V = 5

    def set(self, Ts, thita_initial, Direction_NFZ, x0, y0, psi0, xf, yf, psif,
            thita, x_initial, y_initial, x1, y1):
        Initial_point = np.zeros([self.NoFlyZone.M + 2, 3])
        x0_old = np.cos(thita) * x0 - np.sin(thita) * y0 + x_initial
        y0_old = np.sin(thita) * x0 + np.cos(thita) * y0 + y_initial
        xf_old = np.cos(thita) * xf - np.sin(thita) * yf + x_initial
        yf_old = np.sin(thita) * xf + np.cos(thita) * yf + y_initial
        k_initial = (yf_old - y0_old) / (xf_old - x0_old)


        number_initial_point = 0
        Initial_point[number_initial_point, 0] = x0
        Initial_point[number_initial_point, 1] = y0
        coe_A = k_initial
        coe_B = -1
        coe_C = -k_initial * x0_old + y0_old

        for i in range(self.NoFlyZone.M):
            x1_old = np.cos(thita) * x1[i] - np.sin(thita) * y1[i] + x_initial
            y1_old = np.sin(thita) * x1[i] + np.cos(thita) * y1[i] + y_initial

            if abs(coe_A * x1_old + coe_B * y1_old + coe_C) <= np.sqrt(
                    (coe_A * self.NoFlyZone.a_NFZ[i]) ** 2 + (coe_B * self.NoFlyZone.b_NFZ[i]) ** 2):
                if Direction_NFZ[i] == 1 or Direction_NFZ[i] == 2:
                    if y1[i] - y0 >= 0:
                        if 0 <= thita_initial * 180 / np.pi < 90:
                            rand_x = x1_old + np.sqrt(2) / 2 * self.NoFlyZone.a_NFZ[i]
                            rand_y = y1_old - np.sqrt(2) / 2 * self.NoFlyZone.b_NFZ[i]
                        elif 180 <= thita_initial * 180 / np.pi < 270:
                            rand_x = x1_old - np.sqrt(2) / 2 * self.NoFlyZone.a_NFZ[i]
                            rand_y = y1_old + np.sqrt(2) / 2 * self.NoFlyZone.b_NFZ[i]
                        elif 270 <= thita_initial * 180 / np.pi < 360:
                            rand_x = x1_old - np.sqrt(2) / 2 * self.NoFlyZone.a_NFZ[i]
                            rand_y = y1_old - np.sqrt(2) / 2 * self.NoFlyZone.b_NFZ[i]
                        else:
                            rand_x = x1_old + np.sqrt(2) / 2 * self.NoFlyZone.a_NFZ[i]
                            rand_y = y1_old + np.sqrt(2) / 2 * self.NoFlyZone.b_NFZ[i]
                        number_initial_point += 1
                        Initial_point[number_initial_point, 0] = np.cos(-thita) * (rand_x - x_initial) - \
                                                                 np.sin(-thita) * (rand_y - y_initial)
                        Initial_point[number_initial_point, 1] = np.sin(-thita) * (rand_x - x_initial) + \
                                                                 np.cos(-thita) * (rand_y - y_initial)
                        Initial_point[number_initial_point, 2] = 1
                    else:
                        if 0 <= thita_initial * 180 / np.pi < 90:
                            rand_x = x1_old - np.sqrt(2) / 2 * self.NoFlyZone.a_NFZ[i]
                            rand_y = y1_old + np.sqrt(2) / 2 * self.NoFlyZone.b_NFZ[i]
                        elif 180 <= thita_initial * 180 / np.pi < 270:
                            rand_x = x1_old + np.sqrt(2) / 2 * self.NoFlyZone.a_NFZ[i]
                            rand_y = y1_old - np.sqrt(2) / 2 * self.NoFlyZone.b_NFZ[i]
                        elif 270 <= thita_initial * 180 / np.pi < 360:
                            rand_x = x1_old + np.sqrt(2) / 2 * self.NoFlyZone.a_NFZ[i]
                            rand_y = y1_old + np.sqrt(2) / 2 * self.NoFlyZone.b_NFZ[i]
                        else:
                            rand_x = x1_old - np.sqrt(2) / 2 * self.NoFlyZone.a_NFZ[i]
                            rand_y = y1_old - np.sqrt(2) / 2 * self.NoFlyZone.b_NFZ[i]
                        number_initial_point += 1
                        Initial_point[number_initial_point, 0] = np.cos(-thita) * (rand_x - x_initial) - \
                                                                 np.sin(-thita) * (rand_y - y_initial)
                        Initial_point[number_initial_point, 1] = np.sin(-thita) * (rand_x - x_initial) + \
                                                                 np.cos(-thita) * (rand_y - y_initial)
                        Initial_point[number_initial_point, 2] = 1

                else:
                    assert Direction_NFZ[i] == 3
                    if (y1[i] - y0) >= 0:
                        number_initial_point += 1
                        Initial_point[number_initial_point, 0] = x1[i]
                        Initial_point[number_initial_point, 1] = y1[i] - self.NoFlyZone.a_NFZ[i]
                        Initial_point[number_initial_point, 2] = 1
                    else:
                        number_initial_point += 1
                        Initial_point[number_initial_point, 0] = x1[i]
                        Initial_point[number_initial_point, 1] = y1[i] + self.NoFlyZone.a_NFZ[i]
                        Initial_point[number_initial_point, 2] = 1

        number_initial_point = number_initial_point + 1
        Initial_point[number_initial_point, 0] = xf
        Initial_point[number_initial_point, 1] = yf
        Initial_point[number_initial_point, 2] = 1
        Initial_point[0, 2] = 1
        for i in range(number_initial_point + 1):
            for j in range(i+1, number_initial_point + 1):
                if Initial_point[i, 0] >= Initial_point[j, 0] + 0.001:
                    randx = Initial_point[i, 0]
                    randy = Initial_point[i, 1]
                    randz = Initial_point[i, 2]
                    Initial_point[i, 0] = Initial_point[j, 0]
                    Initial_point[i, 1] = Initial_point[j, 1]
                    Initial_point[i, 2] = Initial_point[j, 2]
                    Initial_point[j, 0] = randx
                    Initial_point[j, 1] = randy
                    Initial_point[j, 2] = randz

        for i in range(number_initial_point):
            if Initial_point[i, 0] < x0:
                Initial_point[i, 2] = 0
        key_i = 0
        for i in range(number_initial_point):
            key_i = i
            if Initial_point[i, 2] == 1:
                break

        Initial_point1 = np.zeros([number_initial_point + 1 - (key_i - 0), 2])
        Initial_point1 = Initial_point[key_i:number_initial_point + 1, 0:2]
        number_initial_point = number_initial_point - (key_i - 0)

        # 将初始轨迹截取到指定sf长度并且离散化为N个离散点
        distance = 0
        for i in range(number_initial_point):
            distance = distance + np.sqrt((Initial_point1[i + 1, 0] - Initial_point1[i, 0]) ** 2 +
                                          (Initial_point1[i + 1, 1] - Initial_point1[i, 1]) ** 2)
            if distance >= self.Config.sf:
                distance = distance - np.sqrt((Initial_point1[i + 1, 0] - Initial_point1[i, 0]) ** 2 +
                                              (Initial_point1[i + 1, 1] - Initial_point1[i, 1]) ** 2)
                rand_thita = get_angle_between_two_points(Initial_point1[i, 0], Initial_point1[i, 1],
                                                          Initial_point1[i + 1, 0], Initial_point1[i + 1, 1])
                Initial_point1[i + 1, 0] = Initial_point1[i, 0] + (self.Config.sf - distance) * np.cos(rand_thita)
                Initial_point1[i + 1, 1] = Initial_point1[i, 1] + (self.Config.sf - distance) * np.sin(rand_thita)
                number_initial_point = i + 1
                break
        output = np.zeros([number_initial_point + 1, 2])
        output[:, 0] = Initial_point1[0: number_initial_point + 1, 0]
        output[:, 1] = Initial_point1[0: number_initial_point + 1, 1]

        [result, all_distance] = equality_curve(self.Config.N, output)
        for i in range(self.Config.N + 1):
            for j in range(self.NoFlyZone.M):
                x1_old = np.cos(thita) * x1[j] - np.sin(thita) * y1[j] + x_initial
                y1_old = np.sin(thita) * x1[j] + np.cos(thita) * y1[j] + y_initial
                result_old_x = np.cos(thita) * result[i, 0] - np.sin(thita) * result[i, 1] + x_initial
                result_old_y = np.sin(thita) * result[i, 0] + np.cos(thita) * result[i, 1] + y_initial

                if ((result_old_x - x1_old) ** 2) / (self.NoFlyZone.a_NFZ[j] ** 2) +\
                        ((result_old_y - y1_old) ** 2) / (self.NoFlyZone.b_NFZ[j] ** 2) <= 1:
                    new_x = self.NoFlyZone.a_NFZ[j] * self.NoFlyZone.b_NFZ[j] * \
                            (result_old_x - x1_old) \
                            / np.sqrt(self.NoFlyZone.b_NFZ[j] ** 2 * (result_old_x - x1_old) ** 2 +
                                      self.NoFlyZone.a_NFZ[j] ** 2 * (result_old_y - y1_old) ** 2) +\
                            x1_old
                    new_y = self.NoFlyZone.a_NFZ[j] * self.NoFlyZone.b_NFZ[j] * \
                            (result_old_y - y1_old) \
                            / np.sqrt(self.NoFlyZone.b_NFZ[j] ** 2 * (result_old_x - x1_old) ** 2 +
                                      self.NoFlyZone.a_NFZ[j] ** 2 * (result_old_y - y1_old) ** 2) +\
                            y1_old
                    result[i, 0] = np.cos(-thita) * (new_x - x_initial) - np.sin(-thita) * (new_y - y_initial)
                    result[i, 1] = np.sin(-thita) * (new_x - x_initial) + np.cos(-thita) * (new_y - y_initial)
        thita_s_s = np.zeros(self.Config.N + 1)
        for i in range(self.Config.N):
            thita_s_s[i] = (result[i + 1, 1] - result[i, 1]) / Ts
        thita_s_s[self.Config.N] = thita_s_s[self.Config.N - 1]
        deta1_s = np.zeros(self.Config.N)
        v_s = np.zeros(self.Config.N)
        for i in range(self.Config.N):
            v_s[i] = self.Config.v_max ** 2
            deta1_s[i] = (abs(thita_s_s[i + 1] - thita_s_s[i]) / Ts + 0.1) ** 0.5

        self.x = result[:, 0]
        self.y = result[:, 1]
        result = result.transpose()
        return result, deta1_s, v_s, thita_s_s


class EntityState:
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None
        # physical theta
        self.theta = None


class Agent:
    def __init__(self):
        # state
        self.state = EntityState()
        # size
        self.size = None


class Landmark:
    def __init__(self):
        # state
        self.state = EntityState()
        # size
        self.sizea = None
        self.sizeb = None


class NoFlyZone:
    def __init__(self):
        self.x_NFZ = np.array([10, 15, 20, 21, 27, 40, 30, 12, 38, 30, 10.8])
        self.y_NFZ = np.array([5,  40, 30, 11, 22, 28, 36, 18, 15, 6,  29])
        self.a_NFZ = np.array([5,  8, 4.5, 6, 6.5, 5,  6,  8,  5,  4,  4])
        self.b_NFZ = np.array([5,  5, 4.5, 4, 5.5, 5,  6,  5,  8,  4,  5.5])
        '''self.M = 11
        self.Flag_NFZ = np.ones(self.M)
        self.agent = Agent()
        self.agent.size = np.random.uniform(0.2, 0.3)'''
        self.num_district = np.int(36 / 6)
        self.num_landmarks_district = [np.int(np.random.uniform(10, 12)) for i in range(self.num_district)]
        self.num_landmarks = np.sum(self.num_landmarks_district) + 1
        self.M = self.num_landmarks
        self.Flag_NFZ = np.ones(self.M)
        self.observing_range = 5
        self.min_corridor = 0.15
        self.agent = Agent()
        self.landmarks = [Landmark() for _ in range(self.num_landmarks)]
        for i, landmark in enumerate(self.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            if i == (len(self.landmarks) - 1):
                landmark.sizea = 0.1
                landmark.sizeb = 0.1
        # make initial conditions
        self.set()
        self.x_NFZ = np.zeros(self.num_landmarks)
        self.y_NFZ = np.zeros(self.num_landmarks)
        self.a_NFZ = np.zeros(self.num_landmarks)
        self.b_NFZ = np.zeros(self.num_landmarks)
        for i, landmark in enumerate(self.landmarks):
            self.x_NFZ[i] = landmark.state.p_pos[0]
            self.y_NFZ[i] = landmark.state.p_pos[1]
            self.a_NFZ[i] = landmark.sizea + 1.3 * self.agent.size
            self.b_NFZ[i] = landmark.sizeb + 1.3 * self.agent.size

        '''self.agent.state.p_pos = [-0.91, -0.06785]
        self.agent.state.theta = [0.7548]
        self.x_NFZ = [ 5.82008468, 3.40707466,  1.75948333,  0.64869947,  1.87345485,  3.30804166,
  6.45242111,  6.42390779,  3.05618959,  3.49589372, 6.25335704, 11.14339408,
  7.58401626,  8.89521401, 11.82035728, 13.30974934, 10.3962112,  10.79290917,
  8.14941786,  8.81073955, 12.85122809, 13.38766996, 19.37748301, 16.98628528,
 16.31406301, 16.22873212, 18.8058493 , 19.36146465, 14.03208535, 18.78850686,
 17.28076194, 14.32558597, 18.85007307, 24.44187204, 24.86120162, 23.77046895,
 25.30462211, 22.30203911, 20.76700602, 22.2902599 , 25.21997666, 23.17369526,
 20.64145864, 22.24792151, 28.85402125, 29.5726398 , 24.79097328, 27.65919925,
 27.45397494, 30.26483407, 26.7767662 , 27.88325444, 27.68341789, 31.18818406,
 29.17474947, 31.92744235, 33.86498668 ,32.79060557, 36.39519969, 34.07817699,
 35.4632649,  31.65248852, 36.14312463 ,37.49779759, 36.21783971, 37.28002046,
 32.79162251]
        self.y_NFZ = [-0.98094004, -3.9981381,  -2.16727815,  5.36084874,  1.70010709,  7.10115796,
 -6.8509246,   1.73767354, -7.20955752, -0.3240589,   5.74709785 , 1.21205162,
 -4.14632713,  2.79105242, -5.01056279,  0.13725276, -6.89616059  ,5.08250724,
  7.27552687,  0.34506259,  7.02028479, -2.64458    , 0.05918305  ,6.36116796,
 -4.79143637,  0.43441723, -5.59890677,  5.50485082 ,-7.22091967 ,-2.91878661,
 -7.93346109,  4.37574773,  7.75817522 ,-3.82268144  ,7.18544546 , 4.0549706,
  1.99227011,  1.66239594, -6.73094541, -2.15227317 ,-7.93131225 ,-5.41303933,
 -4.41047381,  7.35539109, -0.72123802 ,-4.51269342 ,-0.23747402 ,-5.65599211,
  5.57986347,  1.48057704, -2.49787985  ,2.39465675 , 7.98564683 , 6.19204581,
 -7.65583027, -7.0407255 ,  6.112264    ,3.26586138 ,-7.96891172 , 0.56631951,
 -3.63963264, -3.49445738,  5.41904424  ,1.62166752 ,-0.28502658 , 7.45469063,
  0.03266683]
        self.a_NFZ = [1.01113537, 1.07733601, 1.23737069, 1.06789677, 1.28402622, 1.04529923,
 1.00997293, 1.23170172, 0.98591886, 1.1343677 , 1.23424978, 0.98344768,
 0.96171825, 1.30179908, 1.15101052, 1.14674494, 1.00347665, 1.25762493,
 1.03621874, 0.9882276,  1.19537995, 1.21920729, 1.07244248, 1.04507567,
 1.11607534, 1.20843482, 1.02937742, 1.14215576, 1.26237429, 1.15006056,
 1.18868559, 1.13438866, 1.04709702, 0.99006636, 0.96400915, 1.08825278,
 1.11821563, 1.06694792, 1.12133646, 1.1856723 , 1.23690285, 0.95734706,
 0.95704027, 1.12422569, 1.13422689, 0.98676713, 1.08958712, 1.12197172,
 1.01089683, 1.29075197, 1.20605624, 0.97920833, 1.23635113, 1.20874843,
 1.28697642, 1.05933807, 1.2494611 , 1.09975145, 1.23363225, 1.07511419,
 1.24686034, 1.11377849, 0.98356173, 1.18955924, 0.96150501, 1.01253169,
 0.15566639]
        self.b_NFZ = self.a_NFZ
        self.M = len(self.a_NFZ)
        self.landmarks = [Landmark() for _ in range(self.M)]
        for i, landmark in enumerate(self.landmarks):
            landmark.state.p_pos = [0, 0]
            landmark.state.p_pos[0] = self.x_NFZ[i]
            landmark.state.p_pos[1] = self.y_NFZ[i]
        self.Flag_NFZ = np.ones(self.M)'''

    def set(self):
        # set agent size
        self.agent.size = np.random.uniform(0.03, 0.07)
        # set agent velocity
        self.agent.state.p_vel = np.array([0.2])
        # set agent theta
        self.agent.state.theta = np.random.uniform(-np.pi / 3, np.pi / 3, 1)
        # set agent position
        self.agent.state.p_pos = np.squeeze(np.array([np.random.uniform(-1, -0.9, 1), np.random.uniform(-0.1, +0.1, 1)]))
        # if the landmark is the target, the size is unchanged
        for i, landmark in enumerate(self.landmarks):
            if i == (len(self.landmarks) - 1):
                landmark.size = 0.1
        # random properties for landmarks
        for i, landmark in enumerate(self.landmarks):
            if i == len(self.landmarks) - 1:
                landmark.color = np.array([0.21, 0.105, 0.30])
            else:
                landmark.color = np.array([0.25, 0.25, 0.25])
        landmark =self.landmarks[-1]
        landmark.state.p_pos = np.squeeze(np.array([np.random.uniform(15, 36, 1), np.random.uniform(-0.1, +0.1, 1)]))
        for num_d in range(self.num_district):
            done = 0
            while not done:
                fail = 0
                temp = np.int(np.sum(self.num_landmarks_district[0:num_d]))
                temp1 = np.int(np.sum(self.num_landmarks_district[0:num_d + 1]))
                for i, landmark in enumerate(self.landmarks[temp: temp1]):
                    if fail:
                        break
                    flag = 1  # to set landmarks separately
                    max_num = 0
                    while flag:
                        max_num = max_num + 1
                        if max_num > 10:
                            fail = 1
                            break
                        landmark.state.p_pos = np.squeeze(np.array([np.random.uniform(num_d * 6 + 0.5,
                                                                                      (num_d + 1) * 6 + 1.5, 1),
                                                                    np.random.uniform(-8, +8, 1)]))
                        landmark.sizea = np.random.uniform(0.90, 1.55)
                        landmark.sizeb = np.random.uniform(0.90, 1.55)
                        temp1 = []
                        temp2 = []
                        temp1.append(np.sqrt(np.sum(np.square(self.agent.state.p_pos - landmark.state.p_pos))))
                        temp2.append(self.agent.size + landmark.sizea * 0.5 + landmark.sizeb * 0.5 + self.min_corridor)
                        k = i + temp
                        temp3 = np.int(np.sum(self.num_landmarks_district[0:(num_d - 1 if num_d > 0 else 0)]))
                        for j in range(temp3, k + 1):
                            if j == k:
                                j_ = -1
                            else:
                                j_ = j
                            temp1.append(np.sqrt(np.sum(np.square(self.landmarks[j_].state.p_pos - landmark.state.p_pos))))
                            temp1_  = np.sqrt(np.sum(np.square(self.landmarks[j_].state.p_pos - landmark.state.p_pos)))
                            if j_ == -1:
                                temp2.append(self.landmarks[j_].sizea * 0.5 + self.landmarks[j_].sizeb * 0.5 +
                                             landmark.sizea * 0.5 + landmark.sizeb * 0.5 + self.min_corridor)
                                temp2_ = self.landmarks[j_].sizea * 0.5 + self.landmarks[j_].sizeb * 0.5 + \
                                         landmark.sizea * 0.5 + landmark.sizeb * 0.5 + self.min_corridor
                            else:
                                temp2.append(self.landmarks[j_].sizea * 0.5 + self.landmarks[j_].sizeb * 0.5 +
                                             landmark.sizea * 0.5 + landmark.sizeb * 0.5 + self.min_corridor)
                                temp2_ = self.landmarks[j_].sizea * 0.5 + self.landmarks[j_].sizeb * 0.5 + \
                                         landmark.sizea * 0.5 + landmark.sizeb * 0.5 + self.min_corridor
                            if temp1_ < temp2_:
                                break
                        if min(np.array(temp1) - np.array(temp2)) > 0:
                            flag = 0
                if fail:
                    continue
                done = 1


class InitialConfiguration:
    def __init__(self, noflyzone=None):
        self.v_max = 5 if noflyzone is None else 0.1
        self.v_min = 2 if noflyzone is None else 0.1
        self.at_max = 0.07
        self.ah_max = 0.07
        self.N = 10
        self.x0 = 0 if noflyzone is None else copy.deepcopy(noflyzone.agent.state.p_pos[0])
        self.y0 = 0 if noflyzone is None else copy.deepcopy(noflyzone.agent.state.p_pos[1])
        self.psi0 = 60 * np.pi / 180 if noflyzone is None else copy.deepcopy(noflyzone.agent.state.theta)
        self.v0 = 5 if noflyzone is None else 0.1
        self.xf = 50 if noflyzone is None else copy.deepcopy(noflyzone.landmarks[-1].state.p_pos[0])
        self.yf = 50 if noflyzone is None else copy.deepcopy(noflyzone.landmarks[-1].state.p_pos[1])
        self.psif = -60 * np.pi / 180 if noflyzone is None else 0
        self.vf = 5 if noflyzone is None else 0.1
        self.sf = np.sqrt(np.square(self.xf - self.x0) + np.square(self.yf - self.y0)) - 20 - 30 if \
            noflyzone is None else 8  # np.sqrt(np.square(self.xf - self.x0) + np.square(self.yf - self.y0)) * 0.3


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
    ele = nx * (UAV_config.N + 1) + nu * UAV_config.N + nx * UAV_config.N + 2
    no_fly_zone_number = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '*']

    Direction_NFZ = np.zeros(no_fly_zone.M)
    thita_initial = get_angle_between_two_points(UAV_config.x0, UAV_config.y0, UAV_config.xf, UAV_config.yf)
    for i in range(no_fly_zone.M):
        if no_fly_zone.a_NFZ[i] > no_fly_zone.b_NFZ[i]:
            Direction_NFZ[i] = 1
        elif no_fly_zone.a_NFZ[i] < no_fly_zone.b_NFZ[i]:
            Direction_NFZ[i] = 2
        else:
            Direction_NFZ[i] = 3
    # coordinate transformation
    x0, y0, psi0, xf, yf, psif, thita, x_initial, y_initial, x1, y1 \
        = coordinate_transformation(UAV_config.x0, UAV_config.y0,
                                    UAV_config.psi0, UAV_config.xf,
                                    UAV_config.yf, UAV_config.psif,
                                    no_fly_zone.x_NFZ, no_fly_zone.y_NFZ,
                                    no_fly_zone.M)
    np_ = 100
    avoid_circle1 = np.zeros([2, np_ + 1, no_fly_zone.M])  # no - fly - zone 1
    for j in range(no_fly_zone.M):
        for i in range(np_ + 1):
            new_x = np.cos(thita) * x1[j] + -np.sin(thita) * y1[j] + x_initial + \
                    no_fly_zone.a_NFZ[j] * np.cos(i / np_ * 2 * np.pi)  # no - fly - zone 1
            new_y = np.sin(thita) * x1[j] + np.cos(thita) * y1[j] + y_initial + no_fly_zone.a_NFZ[j] * np.sin(i / np_ * 2 * np.pi)
            avoid_circle1[0, i, j] = np.cos(-thita) * (new_x - x_initial) + -np.sin(-thita) * (new_y - y_initial)  # no - fly - zone 1
            avoid_circle1[1, i, j] = np.sin(-thita) * (new_x - x_initial) + np.cos(-thita) * (new_y - y_initial)

    if initial_trajectory.flag:
        result = np.vstack((initial_trajectory.x, initial_trajectory.y))
        deta1_s = np.zeros(UAV_config.N)
        v_s = np.zeros(UAV_config.N)
        theta_s_s = np.zeros(UAV_config.N + 1)

        for i in range(UAV_config.N):
            theta_s_s[i] = (result[1, i + 1] - result[1, i]) / Ts
            v_s[i] = initial_trajectory.V[i] ** 2
            if i == UAV_config.N - 1:
                theta_s_s[i + 1] = theta_s_s[i]
            # detal_s[i] = np.sqrt(UAV_config.ah_max / np.square(initial_trajectory.V[i]))
            deta1_s[i] = (abs(theta_s_s[i + 1] - theta_s_s[i]) / Ts + 0.1) ** 0.5
        for i in range(UAV_config.N + 1):
            new_x = np.cos(-thita) * (result[0, i] - x_initial) + -np.sin(-thita) * (result[1, i] - y_initial)
            new_y = np.sin(-thita) * (result[0, i] - x_initial) + np.cos(-thita) * (result[1, i] - y_initial)
            result[0, i] = new_x
            result[1, i] = new_y
    else:
        result, deta1_s, v_s, thita_s_s = initial_trajectory.set(Ts, thita_initial, Direction_NFZ, x0, y0,
                                                      psi0, xf, yf, psif,
                                                      thita, x_initial, y_initial, x1, y1)


    num_NFZ = no_fly_zone.M

    flag = True
    flag_noactive = False
    iter = 1
    deta_x = 0
    deta_y = 0
    deta1_deta1 = 0
    deta1_v = 0
    while flag:
        print("================== iter={} ====================".format(iter))
        if iter == 1:
            num_iter2 = 0
        else:
            num_iter2 = 8
        A = np.zeros([4 * UAV_config.N + 4, ele])
        b = np.zeros(4 * UAV_config.N + 4)

        for i in range(UAV_config.N):
            A[0 + 4 * i: 4 + 4 * i, 0 + 4 * i: 8 + 4 * i] = [[-xf, 0, 0, 0, xf, 0, 0, 0],
                                                             [0, -xf, -Ts, 0, 0, xf, 0, 0],
                                                             [0, 0, -1, 0, 0, 0, 1, 0],
                                                             [0, 0, 0, -1, 0, 0, 0, 1]]
            A[0 + 4 * i: 4 + 4 * i, 0 + 4 * (UAV_config.N + 1) + 3 * i: 3 + 4 * (UAV_config.N + 1) + 3 * i] = \
            [[-Ts, 0, 0], [0, 0, 0], [0, -Ts, 0], [0, 0, -2*Ts]]

        A[4 * UAV_config.N + 0: 4 * UAV_config.N + 3, 0: 4] = [[xf, 0, 0, 0], [0, xf, 0, 0], [0, 0, 0, 1]]
        b[4 * UAV_config.N + 0: 4 * UAV_config.N + 3] = [x0, y0, UAV_config.v0 ** 2]
        A[4 * UAV_config.N + 3, (UAV_config.N + 1) * 4 - 1] = 1
        b[4 * UAV_config.N + 3] = UAV_config.vf ** 2

        G = np.zeros([7 * UAV_config.N + 4 + num_iter2 * UAV_config.N +
                      num_NFZ * UAV_config.N + 3 * UAV_config.N * 4, ele])
        h = np.zeros(7 * UAV_config.N + 4 + num_iter2 * UAV_config.N +
                      num_NFZ * UAV_config.N + 3 * UAV_config.N * 4)

        for i in range(UAV_config.N):
            G[0 + 2 * i: 2 + 2 * i, 3 + 4 + i * 4] = [1, -1]
            h[0 + 2 * i: 2 + 2 * i] = [UAV_config.v_max ** 2, -UAV_config.v_min ** 2]

            G[2 * UAV_config.N + 0 + 2 * i: 2 * UAV_config.N + 2 + 2 * i, 4 * (UAV_config.N + 1) + 2 + 3 * i] = [1, -1]
            h[2 * UAV_config.N + 0 + 2 * i: 2 * UAV_config.N + 2 + 2 * i] = [UAV_config.at_max,
                                                                                UAV_config.at_max]

            G[4 * UAV_config.N + 0 + 2 * i: 4 * UAV_config.N + 2 + 2 * i, 4 * (UAV_config.N + 1) + 1 + 3 * i] = [1, -1]
            G[4 * UAV_config.N + 0 + 2 * i: 4 * UAV_config.N + 2 + 2 * i, 4 * (UAV_config.N + 1) + 3 * UAV_config.N + 1 + 4 * i] = [-2 * deta1_s[i], -2 * deta1_s[i]]
            h[4 * UAV_config.N + 0 + 2 * i: 4 * UAV_config.N + 2 + 2 * i] = [-deta1_s[i] ** 2, -deta1_s[i] ** 2]

            G[6 * UAV_config.N + 0 + 1 * i, 3 + i * 4] = 1 / v_s[i] ** 2
            G[6 * UAV_config.N + 0 + 1 * i, 4 * (UAV_config.N + 1) + 3 * UAV_config.N + 3 + 4 * i] = 1
            h[6 * UAV_config.N + 0 + 1 * i] = 2 / v_s[i]

        G[7 * UAV_config.N + 0, 4 * UAV_config.N + 1] = xf
        G[7 * UAV_config.N + 0, 4 * (UAV_config.N + 1) + 3 * UAV_config.N + 4 * UAV_config.N + 1] = -1
        h[7 * UAV_config.N + 0] = yf

        G[7 * UAV_config.N + 1, 4 * UAV_config.N + 1] = -xf
        G[7 * UAV_config.N + 1, 4 * (UAV_config.N + 1) + 3 * UAV_config.N + 4 * UAV_config.N + 1] = -1
        h[7 * UAV_config.N + 1] = -yf

        G[7 * UAV_config.N + 2, 4 * UAV_config.N + 0] = xf
        G[7 * UAV_config.N + 2, 4 * (UAV_config.N + 1) + 3 * UAV_config.N + 4 * UAV_config.N + 0] = -1
        h[7 * UAV_config.N + 2] = xf

        G[7 * UAV_config.N + 3, 4 * UAV_config.N + 0] = -xf
        G[7 * UAV_config.N + 3, 4 * (UAV_config.N + 1) + 3 * UAV_config.N + 4 * UAV_config.N + 0] = -1
        h[7 * UAV_config.N + 3] = -xf

        if iter >= 2:
            for i in range(UAV_config.N):
                G[7 * UAV_config.N + 4 + 0 + 2 * i: 7 * UAV_config.N + 4 + 2 + 2 * i, 0 + 4 * i] = [xf, -xf]
                h[7 * UAV_config.N + 4 + 0 + 2 * i: 7 * UAV_config.N + 4 + 2 + 2 * i] = [deta_x + result[0, i],
                deta_x - result[0, i]]

                G[7 * UAV_config.N + 3 + 2 * UAV_config.N + 1 + 2 * i: 7 * UAV_config.N + 4 + 2 * UAV_config.N + 2 + 2 * i, 1 + 4 * i] = [xf, -xf]
                h[7 * UAV_config.N + 3 + 2 * UAV_config.N + 1 + 2 * i: 7 * UAV_config.N + 4 + 2 * UAV_config.N + 2 + 2 * i] = [deta_y + result[1, i], deta_y - result[1, i]]

                G[7 * UAV_config.N + 3 + 4 * UAV_config.N + 1 + 2 * i: 7 * UAV_config.N + 4 + 4 * UAV_config.N + 2 + 2 * i, 4 * (UAV_config.N + 1) + 3 * UAV_config.N + 1 + 4 * i] = [1, -1]
                h[7 * UAV_config.N + 3 + 4 * UAV_config.N + 1 + 2 * i: 7 * UAV_config.N + 4 + 4 * UAV_config.N + 2 + 2 * i] = [deta1_deta1 + deta1_s[i], deta1_deta1 - deta1_s[i]]

                G[7 * UAV_config.N + 3 + 6 * UAV_config.N + 1 + 2 * i: 7 * UAV_config.N + 4 + 6 * UAV_config.N + 2 + 2 * i, 3 + 4 * i] = [1, -1]
                h[7 * UAV_config.N + 3 + 6 * UAV_config.N + 1 + 2 * i: 7 * UAV_config.N + 4 + 6 * UAV_config.N + 2 + 2 * i] = [deta1_v + v_s[i], deta1_v - v_s[i]]

        for j in range(num_NFZ):
            if no_fly_zone.Flag_NFZ[j]:
                for i in range(1, UAV_config.N + 1):
                    x_i = result[0, i]
                    y_i = result[1, i]
                    A1 = np.cos(thita)
                    B1 = -np.sin(thita)
                    C1 = x_initial - (np.cos(thita) * x1[j] - np.sin(thita) * y1[j] + x_initial)
                    A2 = np.sin(thita)
                    B2 = np.cos(thita)
                    C2 = y_initial - (np.sin(thita) * x1[j] + np.cos(thita) * y1[j] + y_initial)
                    af_ax = 2 * A1 * (A1 * x_i + B1 * y_i + C1) / no_fly_zone.a_NFZ[j] ** 2 +\
                            2 * A2 * (A2 * x_i + B2 * y_i + C2) / no_fly_zone.b_NFZ[j] ** 2
                    af_ay = 2 * B1 * (A1 * x_i + B1 * y_i + C1) / no_fly_zone.a_NFZ[j] ** 2 +\
                            2 * B2 * (A2 * x_i + B2 * y_i + C2) / no_fly_zone.b_NFZ[j] ** 2
                    f_xy = (A1 * x_i + B1 * y_i + C1) ** 2 / no_fly_zone.a_NFZ[j] ** 2 + (A2 * x_i + B2 * y_i + C2) ** 2 / no_fly_zone.b_NFZ[j] ** 2 - 1

                    G[7 * UAV_config.N + 3 + num_iter2 * UAV_config.N + j * UAV_config.N + 1 + 1 * (i - 1), i * nx + 0] = -af_ax * xf
                    G[7 * UAV_config.N + 3 + num_iter2 * UAV_config.N + j * UAV_config.N + 1 + 1 * (i - 1), i * nx + 1] = -af_ay * xf
                    h[7 * UAV_config.N + 3 + num_iter2 * UAV_config.N + j * UAV_config.N + 1 + 1 * (i - 1)] = f_xy - af_ax * x_i - af_ay * y_i

        for i in range(UAV_config.N):
            G[7 * UAV_config.N + 4 + num_iter2 * UAV_config.N + num_NFZ * UAV_config.N + 1 + 3 * i, 2 + 4 * i] = -1
            G[7 * UAV_config.N + 4 + num_iter2 * UAV_config.N + num_NFZ * UAV_config.N + 2 + 3 * i, 4 * (UAV_config.N + 1) + 0 + 3 * i] = -1
            h[7 * UAV_config.N + 4 + num_iter2 * UAV_config.N + num_NFZ * UAV_config.N + 0 + 3 * i] = 1

            G[7 * UAV_config.N + 4 + num_iter2 * UAV_config.N + num_NFZ * UAV_config.N + 3 * UAV_config.N + 0 + 3 * i, 3 + 4 * i] = -1 / np.sqrt(2)
            G[7 * UAV_config.N + 4 + num_iter2 * UAV_config.N + num_NFZ * UAV_config.N + 3 * UAV_config.N + 1 + 3 * i, 3 + 4 * i] = 1 / np.sqrt(2)
            G[7 * UAV_config.N + 4 + num_iter2 * UAV_config.N + num_NFZ * UAV_config.N + 3 * UAV_config.N + 2 + 3 * i, 4 * (UAV_config.N + 1) + 3 * UAV_config.N + 0 + 4 * i] = -1 / np.sqrt(2)
            h[7 * UAV_config.N + 4 + num_iter2 * UAV_config.N + num_NFZ * UAV_config.N + 3 * UAV_config.N + 0 + 3 * i] = 1 / np.sqrt(2)
            h[7 * UAV_config.N + 4 + num_iter2 * UAV_config.N + num_NFZ * UAV_config.N + 3 * UAV_config.N + 1 + 3 * i] = 1 / np.sqrt(2)

            G[7 * UAV_config.N + 4 + num_iter2 * UAV_config.N + num_NFZ * UAV_config.N + 3 * UAV_config.N * 2 + 0 + 3 * i, 4 * (UAV_config.N + 1) + 3 * UAV_config.N + 0 + 4 * i] = -1 / np.sqrt(2)
            G[7 * UAV_config.N + 4 + num_iter2 * UAV_config.N + num_NFZ * UAV_config.N + 3 * UAV_config.N * 2 + 0 + 3 * i, 4 * (UAV_config.N + 1) + 3 * UAV_config.N + 2 + 4 * i] = -1 / np.sqrt(2)
            G[7 * UAV_config.N + 4 + num_iter2 * UAV_config.N + num_NFZ * UAV_config.N + 3 * UAV_config.N * 2 + 1 + 3 * i, 4 * (UAV_config.N + 1) + 3 * UAV_config.N + 0 + 4 * i] = 1 / np.sqrt(2)
            G[7 * UAV_config.N + 4 + num_iter2 * UAV_config.N + num_NFZ * UAV_config.N + 3 * UAV_config.N * 2 + 1 + 3 * i, 4 * (UAV_config.N + 1) + 3 * UAV_config.N + 2 + 4 * i] = -1 / np.sqrt(2)
            h[7 * UAV_config.N + 4 + num_iter2 * UAV_config.N + num_NFZ * UAV_config.N + 3 * UAV_config.N * 2 + 2 + 3 * i] = np.sqrt(2)

            G[7 * UAV_config.N + 4 + num_iter2 * UAV_config.N + num_NFZ * UAV_config.N + 3 * UAV_config.N * 3 + 0 + 3 * i, 4 * (UAV_config.N + 1) + 0 + 3 * i] = -1 / np.sqrt(2)
            G[7 * UAV_config.N + 4 + num_iter2 * UAV_config.N + num_NFZ * UAV_config.N + 3 * UAV_config.N * 3 + 0 + 3 * i, 4 * (UAV_config.N + 1) + 3 * UAV_config.N + 3 + 4 * i] = -1 / np.sqrt(2)
            G[7 * UAV_config.N + 4 + num_iter2 * UAV_config.N + num_NFZ * UAV_config.N + 3 * UAV_config.N * 3 + 1 + 3 * i, 4 * (UAV_config.N + 1) + 0 + 3 * i] = -1 / np.sqrt(2)
            G[7 * UAV_config.N + 4 + num_iter2 * UAV_config.N + num_NFZ * UAV_config.N + 3 * UAV_config.N * 3 + 1 + 3 * i, 4 * (UAV_config.N + 1) + 3 * UAV_config.N + 3 + 4 * i] = 1 / np.sqrt(2)
            G[7 * UAV_config.N + 4 + num_iter2 * UAV_config.N + num_NFZ * UAV_config.N + 3 * UAV_config.N * 3 + 2 + 3 * i, 4 * (UAV_config.N + 1) + 3 * UAV_config.N + 1 + 4 * i] = -np.sqrt(2 / UAV_config.ah_max)

        coe1 = 10
        coe3 = 100
        coe4 = 100
        c = np.zeros(ele)
        for i in range(UAV_config.N):
            c[4 * (UAV_config.N + 1) + 3 * UAV_config.N + 2 + 4 * i] = coe1
        c[4 * (UAV_config.N + 1) + 3 * UAV_config.N + 4 * UAV_config.N + 0] = coe3
        c[4 * (UAV_config.N + 1) + 3 * UAV_config.N + 4 * UAV_config.N + 1] = coe4
        dims = {'l': np.int(7 * UAV_config.N + 4 + num_iter2 * UAV_config.N + num_NFZ * UAV_config.N),
                'q': [3 for _ in range(4 * UAV_config.N)]}
        solution = ecos.solve(c, sp.csr_matrix(G), h, dims, sp.csr_matrix(A), b, verbose=True)

        xe = solution['x']
        print(solution['info'])
        y = xe[0:4 * (UAV_config.N + 1) + 3 * UAV_config.N]
        k = xe[4 * (UAV_config.N + 1) + 3 * UAV_config.N + 2: 4 * (UAV_config.N + 1) +
               3 * UAV_config.N + 4 * UAV_config.N: 4]
        sk = xe[4 * (UAV_config.N + 1) + 3 * UAV_config.N + 3: 4 * (UAV_config.N + 1) +
                3 * UAV_config.N + 4 * UAV_config.N: 4]
        k_x = xe[-2]
        k_y = xe[-1]
        deta = xe[4 * (UAV_config.N + 1) + 3 * UAV_config.N + 0: 4 * (UAV_config.N + 1) +
                  3 * UAV_config.N + 4 * UAV_config.N: 4]
        deta1 = xe[4 * (UAV_config.N + 1) + 3 * UAV_config.N + 1: 4 * (UAV_config.N + 1) +
                   3 * UAV_config.N + 4 * UAV_config.N: 4]

        x_pos = xf * y[0: (UAV_config.N + 1) * nx: nx]
        y_pos = xf * y[1: (UAV_config.N + 1) * nx: nx]
        fai_s = np.array(y[2: (UAV_config.N + 1) * nx: nx])
        v_ba = np.array(y[3: (UAV_config.N + 1) * nx: nx])

        fai_c = np.array(y[(UAV_config.N + 1) * nx + 0: (UAV_config.N + 1) * nx + UAV_config.N * nu: nu])
        fai_d = np.array(y[(UAV_config.N + 1) * nx + 1: (UAV_config.N + 1) * nx + UAV_config.N * nu: nu])
        at = np.array(y[(UAV_config.N + 1) * nx + 2: (UAV_config.N + 1) * nx + UAV_config.N * nu: nu])
        u12_square = np.array([(np.square(fai_c[i]) + np.square(fai_s[i])) for i in range(UAV_config.N)])
        v_value = np.array([np.sqrt(v_ba[i]) for i in range(UAV_config.N + 1)])
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
                deta1_differ[i] = deta1[i] - deta1_s[i]
                v_ba_differ[i] = v_ba[i] - v_s[i]
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
                    deta1_s[i] = deta1[i]
                    v_s[i] = v_ba[i]
                result = np.squeeze(np.array([x_pos, y_pos]))
        if min(u12_square) <= 0.99 or min(u12_square) >= 1.01:
            flag = 0
            flag_noactive = True
        else:
            flag_noactive = False
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
    return x_pos, y_pos, thita_value, v_value, at, ah, avoid_circle1, flag_noactive


class Mpc:
    def __init__(self):
        self.NoFlyZone = NoFlyZone()
        self.UAV_config = InitialConfiguration(self.NoFlyZone)
        self.initial_trajectory = InitialTrajectory(self.UAV_config, self.NoFlyZone)
        self.x = None
        self.y = None
        self.r = None

    def plt(self, fig, ax0, num_mpc=0, flag_noactive=False):
        if flag_noactive:
            print('a')
        if self.x is not None:
            for i in range(self.n + 1):
                p_pos = np.array([self.x[i], self.y[i]])
                r = self.r
                circle = mpathes.Circle(p_pos, r, color='salmon' if i == 0 else 'mistyrose')
                ax0.add_patch(circle)
        if num_mpc == 1:
            for x, y, ra, rb in zip(self.NoFlyZone.x_NFZ[:-1], self.NoFlyZone.y_NFZ[:-1],
                                    self.NoFlyZone.a_NFZ[:-1], self.NoFlyZone.b_NFZ[:-1]):
                p_pos = [x, y]
                ellipse = mpathes.Ellipse(p_pos, 2 * (ra - 1.3 * self.NoFlyZone.agent.size),
                                          2 * (rb - 1.3 * self.NoFlyZone.agent.size),
                                          facecolor='w', edgecolor='palevioletred', linestyle='-.')
                ax0.add_patch(ellipse)
                ax0.set_xlim((0, 40))
                ax0.set_ylim((0, 40))
                ax0.axis('equal')
            for x, y, ra, rb in zip(self.NoFlyZone.x_NFZ[:-1], self.NoFlyZone.y_NFZ[:-1],
                                    self.NoFlyZone.a_NFZ[:-1], self.NoFlyZone.b_NFZ[:-1]):
                p_pos = [x, y]
                ra = (ra - 1.3 * self.NoFlyZone.agent.size - 0.09)
                rb = (rb - 1.3 * self.NoFlyZone.agent.size - 0.09)
                ellipse = mpathes.Ellipse(p_pos, 2 * ra, 2 * rb, facecolor='palevioletred')
                ax0.add_patch(ellipse)
            ellipse = mpathes.Ellipse([self.NoFlyZone.x_NFZ[-1], self.NoFlyZone.y_NFZ[-1]],
                                      2 * (self.NoFlyZone.a_NFZ[-1] - 1.3 * self.NoFlyZone.agent.size),
                                      2 * (self.NoFlyZone.b_NFZ[-1] - 1.3 * self.NoFlyZone.agent.size),
                                      facecolor='palevioletred')
            ax0.add_patch(ellipse)
        for i in range(self.UAV_config.N + 1):
            p_pos = np.array([self.x_pos[i], self.y_pos[i]])
            r = self.NoFlyZone.agent.size
            circle = mpathes.Circle(p_pos, r, color='salmon' if i == 0 else 'rosybrown')
            ax0.add_patch(circle)
        self.x = copy.deepcopy(self.x_pos)
        self.y = copy.deepcopy(self.y_pos)
        self.r = self.NoFlyZone.agent.size
        self.n = self.UAV_config.N
        return 0

    def update_initial_trajectory_terminal(self):
        x = self.x_pos[1:-1]
        y = self.y_pos[1:-1]
        v = self.v_value[1:-1]
        return x, y, v

    def update_initial_trajectory_noterminal(self):
        x_pos = self.x_pos
        y_pos = self.y_pos
        v_value = self.v_value
        no_fly_zone = self.NoFlyZone
        deta_s = np.sqrt((x_pos[0] - x_pos[1]) ** 2 + (y_pos[0] - y_pos[1]) ** 2)
        thita = get_angle_between_two_points(x_pos[-2], y_pos[-2], x_pos[-1], y_pos[-1])
        xf = x_pos[-1] + deta_s * np.cos(thita)
        yf = y_pos[-1] + deta_s * np.sin(thita)
        if max(no_fly_zone.Flag_NFZ) == 1:
            for m in range(no_fly_zone.M):
                if (xf - no_fly_zone.x_NFZ[m]) ** 2 / no_fly_zone.a_NFZ[m] ** 2 + (
                        yf - no_fly_zone.y_NFZ[m]) ** 2 / no_fly_zone.b_NFZ[m] ** 2 <= 1:
                    new_x = no_fly_zone.a_NFZ[m] * no_fly_zone.b_NFZ[m] * (xf - no_fly_zone.x_NFZ[m]) / np.sqrt(
                        no_fly_zone.b_NFZ[m] ** 2 * (xf - no_fly_zone.x_NFZ[m]) ** 2 + no_fly_zone.a_NFZ[m] ** 2 * (
                                    yf - no_fly_zone.y_NFZ[m]) ** 2) + no_fly_zone.x_NFZ[m]
                    new_y = no_fly_zone.a_NFZ[m] * no_fly_zone.b_NFZ[m] * (yf - no_fly_zone.y_NFZ[m]) / np.sqrt(
                        no_fly_zone.b_NFZ[m] ** 2 * (xf - no_fly_zone.x_NFZ[m]) ** 2 + no_fly_zone.a_NFZ[m] ** 2 * (
                                    yf - no_fly_zone.y_NFZ[m]) ** 2) + no_fly_zone.y_NFZ[m]
                    xf = new_x
                    yf = new_y
                    break
        x = np.concatenate([x_pos[1:], (xf, )])
        y = np.concatenate([y_pos[1:], (yf, )])
        v = np.concatenate([v_value[1:], (0.2,)])
        return x, y, v

    def run(self):
        """
        :return: run the MPC in a loop and present the figure
        """
        self.fig, self.ax0 = plt.subplots()
        plt.ion()
        flag_mpc = 1  # whether processing MPC MPC进行标志
        flag1 = 0
        s_gone = 0
        num_mpc = 0
        flag_terminal_set = 0

        while flag_mpc:
            num_mpc = num_mpc + 1
            # if mpc optimization method cannot achieve the final answer
            original_sf = self.UAV_config.sf
            original_N = self.UAV_config.N

            while True:
                self.x_pos, self.y_pos, self.thita_value, self.v_value, \
                self.at, self.ah, self.avoid_circle1,\
                self.flag_noactive = optimization(self.UAV_config, self.NoFlyZone,
                                                  self.initial_trajectory)
                # plot
                self.plt(self.fig, self.ax0, num_mpc=num_mpc,
                         flag_noactive=self.flag_noactive)
                plt.show()
                plt.pause(0.3)
                if self.flag_noactive:
                    self.initial_trajectory.flag = False
                    self.UAV_config.sf += 5 * (self.UAV_config.sf
                                               / self.UAV_config.N)
                    self.UAV_config.N += 5
                else:
                    break
            self.UAV_config.N = original_N
            self.UAV_config.sf = original_sf

            # update x0, y0, psi0, and v0 of UAV_config
            self.UAV_config.x0 = self.x_pos[1]
            self.UAV_config.y0 = self.y_pos[1]
            self.UAV_config.psi0 = self.thita_value[1]
            self.UAV_config.v0 = self.v_value[1]
            s_gone = s_gone + np.sqrt((self.x_pos[1] - self.x_pos[0]) ** 2 +
                                      (self.y_pos[1] - self.y_pos[0]) ** 2)


            # if near the target, then change sf and N
            if np.sqrt((self.x_pos[-1] - self.UAV_config.xf) ** 2 + (self.y_pos[-1] - self.UAV_config.yf) ** 2) <= \
                    self.UAV_config.sf / self.UAV_config.N:
                flag_terminal_set = True
                if self.UAV_config.N <= 5:
                    flag_mpc = 0
                self.UAV_config.sf -= self.UAV_config.sf/self.UAV_config.N
                self.UAV_config.N -= 1
            if flag_terminal_set:
                self.UAV_config.sf -= self.UAV_config.sf / self.UAV_config.N
                self.UAV_config.N -= 1

            # update the initial trajectory
            self.initial_trajectory.flag = True
            if flag_terminal_set:
                self.initial_trajectory.x, self.initial_trajectory.y,\
                 self.initial_trajectory.V = self.update_initial_trajectory_terminal()
            else:
                self.initial_trajectory.x, self.initial_trajectory.y,\
                 self.initial_trajectory.V = self.update_initial_trajectory_noterminal()



if __name__ == '__main__':
    '''MpcLayer = Mpc()
    import pickle
    pickle_file = open('MpcLayer2.pkl', 'wb')
    pickle.dump(MpcLayer, pickle_file)
    pickle_file.close()'''
    import pickle
    pickle_file = open('MpcLayer2.pkl', 'rb')
    MpcLayer = pickle.load(pickle_file)
    pickle_file.close()
    print(MpcLayer.initial_trajectory.Config)
    print(MpcLayer.NoFlyZone.x_NFZ, '\n', MpcLayer.NoFlyZone.y_NFZ, '\n', MpcLayer.NoFlyZone.a_NFZ, '\n',
          MpcLayer.NoFlyZone.b_NFZ)
    print(MpcLayer.NoFlyZone.agent.state.p_pos)
    print(MpcLayer.NoFlyZone.agent.state.theta)

    MpcLayer.run()



