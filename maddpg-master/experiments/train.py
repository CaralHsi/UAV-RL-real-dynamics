import argparse
import numpy as np
import tensorflow as tf
import time
import copy
import pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpathes
import sys

sys.path.append('../')
np.set_printoptions(threshold=1e6)

# 禁飞区 多个  8个 partial observable
# omega 平滑
# 测试用例 给定
# theta 随机
# 禁飞区 圆 椭圆


# keras_version=='2.2.4'
# tensorflow_version=='1.13.1'


import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
from safety_layer.safety_layer import SafetyLayer
from safety_layer.MPC_layer_new import MpcLayer

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="my_UAV_world", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=np.floor(130), help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=600000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=0.5 * 1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--use-safety-layer", action="store_true", default=False, help="whether use safety_layer")
    parser.add_argument("--use-mpc-layer", action="store_true", default=True, help="whether use MPC_layer")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./ckpt_my_UAV_world_6_landmarks_safety_layer/test.ckpt", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=10, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=True)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def mlp_model_safety_layer(input, num_outputs, scope, reuse=False, num_units=10, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            done_callback=scenario.done, constraint_value_callback=scenario.constraints_value,
                            is_any_collision_callback=scenario.is_any_collision)
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist, safety_layer=None):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg'), safety_layer=safety_layer))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy == 'ddpg'), safety_layer=safety_layer))
    return trainers


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        obs_n = env.reset()
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))
        # Pretrain the safety_layer
        safety_layer = None
        if arglist.use_safety_layer:
            safety_layer = SafetyLayer(env, len(env.world.landmarks) - 1, mlp_model_safety_layer,
                                       env.observation_space[0].shape,
                                       env.action_space, trainers[0].action)
            # set safety_layer for trainer[0]
            trainers[0].set_safety_layer(safety_layer)
        if arglist.use_mpc_layer:
            safety_layer = MpcLayer(env)
            # set safety_layer for trainer[0]
            trainers[0].set_safety_layer(safety_layer)



        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        episode_step = 0
        train_step = 0
        cumulative_constraint_violations = 0
        t_start = time.time()
        data_save = []
        num_done = 0

        # pickle env
        # env0 = copy.deepcopy(env)
        '''file_path = open('env.pkl', 'rb')
        import pickle
        for i in range(len(env.world.landmarks)):
            env.world.landmarks[i] = pickle.load(file_path)
        for i in range(len(env.world.agents)):
            env.world.agents[i] = pickle.load(file_path)
        obs_n = []
        agents = env.world.agents
        for agent in agents:
            obs_n.append(env._get_obs(agent))'''

        print('Starting iterations...')
        while True:
            # get constraint_values
            c_n = env.get_constraint_values()
            is_any_collision = env.is_any_collision()
            if is_any_collision[0]:
                cumulative_constraint_violations = cumulative_constraint_violations + 1
            '''if c_n[0][0] > 0:
                print("there is a c_n > 0")'''
            # get action
            action_n = [agent.action_real(obs, c, env) for agent, obs, c in zip(trainers, obs_n, c_n)]
            action_real = [action_n[0][0]]
            if_call = [action_n[0][2]]
            action_n = [action_n[0][1]]
            data_save.append(np.concatenate([obs_n[0], action_n[0], action_n[0]]))
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n, if_call=if_call)
            '''is_any_collision_new = env.is_any_collision()
            if is_any_collision_new[0]:
                env.is_any_collision()
                dist = np.sqrt(np.sum(np.square(env.agents[0].state.p_pos - env.world.landmarks[0].state.p_pos))) -\
                       (env.agents[0].size + env.world.landmarks[0].size)
                # print("aaa", env.agents[0].state.p_pos, dist)'''

            # new c_n
            # new_c_n = env.get_constraint_values()
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len) or \
                       (env.agents[0].state.p_pos[0] - env.world.landmarks[-1].state.p_pos[0]) > 1.5
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                if done:
                    num_done = num_done + 1

                data_save.append(np.concatenate([obs_n[0], action_n[0], action_n[0]]))
                data_save = np.array(data_save)
                '''np.savetxt("data_save.txt", data_save)'''  # 缺省按照'%.18e'格式保存数据，以空格分隔

                # plot x, y, v, theta
                a = data_save
                V = a[:, 1]
                x = a[:, 2]
                y = a[:, 3]
                theta = a[:, 4]
                omega = a[:, 5]
                # action_n = a[:, 26] - a[:, 27]
                # action_real = a[:, 31] - a[:, 32]
                '''fig, ax0 = plt.subplots()
                for i, landmark in enumerate(env.world.landmarks[:-1]):
                    p_pos = landmark.state.p_pos
                    r = landmark.size
                    circle = mpathes.Circle(p_pos, r, facecolor='w', edgecolor='forestgreen', linestyle='-.')
                    ax0.add_patch(circle)
                for i, landmark in enumerate(env.world.landmarks):
                    p_pos = landmark.state.p_pos
                    r = (landmark.size - 0.09) if landmark is not env.world.landmarks[-1] else landmark.size
                    circle = mpathes.Circle(p_pos, r, facecolor='forestgreen')
                    ax0.add_patch(circle)
                for i in range(len(x)):
                    p_pos = np.array([x[i], y[i]])
                    r = env.world.agents[0].size
                    circle = mpathes.Circle(p_pos, r, facecolor='darkgreen')
                    ax0.add_patch(circle)
                ax0.set_xlim((-1, 40))
                ax0.set_ylim((-10, 10))
                ax0.axis('equal')
                ax0.set_title("x-y")
                x1 = [-1, 40]
                y1 = [10, 10]
                y2 = [-10, -10]
                ax0.plot(x1, y1, color='forestgreen', linestyle='-.')
                ax0.plot(x1, y2, color='forestgreen', linestyle='-.')
                plt.show()'''
                '''fig, ax = plt.subplots(ncols=2, nrows=2)
                for i, landmark in enumerate(env.world.landmarks):
                    p_pos = landmark.state.p_pos
                    r = landmark.size
                    circle = mpathes.Circle(p_pos, r)
                    ax[0, 0].add_patch(circle)
                for i in range(len(x)):
                    p_pos = np.array([x[i], y[i]])
                    r = env.world.agents[0].size
                    circle = mpathes.Circle(p_pos, r)
                    ax[0, 0].add_patch(circle)
                ax[0, 0].set_xlim((-1, 20))
                ax[0, 0].set_ylim((-10.3, 10.3))
                ax[0, 0].set_title("x-y")
                ax[0, 0].axis('equal')
                ax[0, 1].plot(theta)
                ax[0, 1].set_title("theta")
                ax[1, 0].plot(omega)
                ax[1, 0].set_title("omega")
                # ax[1, 1].plot(action_n * 0.12)
                # ax[1, 1].set_title("action_n")
                plt.show()'''

                # reset and continue
                data_save = []
                obs_n = env.reset()
                # env0 = copy.deepcopy(env)
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            '''for agent in trainers:
                loss = agent.update(trainers, train_step)'''

            # save model, display training output
            if (done or terminal) and ((len(episode_rewards) - 1) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, num_cumulative_constraints: {}, num_done: {}, time: {}".format(
                        train_step, len(episode_rewards) - 1, np.mean(episode_rewards[-arglist.save_rate:]),
                        cumulative_constraint_violations, num_done, round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, num_cumulative_constraints: {}, num_done: {}, time: {}".format(
                        train_step, len(episode_rewards) - 1, np.mean(episode_rewards[-arglist.save_rate:]),
                        cumulative_constraint_violations,
                        num_done, round(time.time()-t_start, 3)))
                    # print(trainers[0].safety_layer.num_call)
                t_start = time.time()
                num_done = 0
                cumulative_constraint_violations = 0
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
