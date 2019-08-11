import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U
import torch
import mosek
import sys

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from safety_layer.replay_buffer import ReplayBuffer
from torch.optim import Adam
from safety_layer.constraint_model import ConstraintModel
from safety_layer.list import for_each


# Since the actual value of Infinity is ignores, we define it solely
# for symbolic purposes:
inf = 1

# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def c_next(make_obs_ph, act_space, c_ph, c_next_func, num_constraints,
           optimizer, grad_norm_clipping, num_units=64,
           reuse=False, scope="c_next"
           ):
    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        act_pdtype = make_pdtype(act_space[0])
        obs_ph = make_obs_ph
        act_ph = act_pdtype.sample_placeholder([None], name="action")
        c_next_target_ph = []
        for _ in range(num_constraints):
            c_next_target_ph.append(tf.placeholder(tf.float32, [None, 1], name="target"+str(_)))

        c_next_input = tf.concat(obs_ph, 1)
        gs_ = []
        for _ in range(num_constraints):
            gs_.append(c_next_func(c_next_input, int((act_pdtype.param_shape()[0])/2),
                                   scope="c_next_func"+str(_), num_units=num_units))

        c_ = []  # to be testified
        for _ in range(num_constraints):
            temp = c_ph[_] + tf.multiply(gs_[_], act_ph)
            c_.append(tf.reduce_sum(temp, -1))

        c_next_vars = [U.scope_vars(U.absolute_scope_name("c_next_func"+str(_))) for _ in range(num_constraints)]

        diff = [(c_[_] - c_next_target_ph[_]) for _ in range(num_constraints)]
        c_next_loss = [tf.reduce_mean(tf.square(diff[_])) for _ in range(num_constraints)]

        optimize_expr = [U.minimize_and_clip(optimizer, c_next_loss[_], c_next_vars[_], grad_norm_clipping)
                         for _ in range(num_constraints)]

        # Create callable functions
        train = [U.function(inputs=[obs_ph] + [act_ph] + [c_ph[_]] + [c_next_target_ph[_]], outputs=c_next_loss[_], updates=[optimize_expr[_]])]
        c_next_values = [U.function([obs_ph] + [act_ph] + [c_ph[_]], c_[_]) for _ in range(num_constraints)]
        g_next_values = [U.function([obs_ph], gs_[_]) for _ in range(num_constraints)]
        return train, c_next_values, g_next_values


class SafetyLayer:
    def __init__(self, env, num_constraints, model, obs_shape,
                 act_space, action):
        self.name = "safe-layer"
        self._env = env
        self.num_constraints = num_constraints  # = num_landmarks - 1 because the last landmark is the target
        self.max_episode_length = 300
        self.batch_size = 256
        self.lr = 0.007
        self.steps_per_epoch = 6000
        self.epochs = 10
        self.evaluation_steps = 1500
        self.replay_buffer_size = 1000000
        self.num_units = 10
        self._train_global_step = 0
        self._eval_global_step = 0
        self.max_replay_buffer = self.batch_size * self.max_episode_length  # 76800
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)  # 1e6
        self._initialize_constraint_models()
        self.action = action


    def _as_tensor(self, ndarray, requires_grad=False):
        tensor = torch.Tensor(ndarray)
        tensor.requires_grad = requires_grad
        return tensor

    def _eval_mode(self):
        for_each(lambda x: x.eval(), self._models)

    def _train_mode(self):
        for_each(lambda x: x.train(), self._models)

    def _initialize_constraint_models(self):
        self._models = [ConstraintModel(self._env.observation_space[0].shape[0],
                                        1) \
                        for _ in range(self.num_constraints)]
        self._optimizers = [Adam(x.parameters(), lr=self.lr) for x in self._models]



    def _experience(self, act, obs, c, c_next):
        # Store transition in the replay buffer.
        # self.replay_buffer.add(act, obs, c, c_next)
        self.replay_buffer.add({
            "action": act[0],
            "observation": obs[0],
            "c": np.array(c[0]),
            "c_next": np.array(c_next[0])
        })

    def _sample_steps(self, num_steps):
        episode_length = 0
        observation = self._env.reset()
        for step in range(num_steps):

            # action = [np.array([1, 2, 3, 4, 5])]
            # action = [self._env.action_space[0].sample()]
            action = [self.action(observation[0])]
            action_omega = [action[0][3] - action[0][4]]
            c = self._env.get_constraint_values()
            observation_next, _, done, _ = self._env.step(action)
            c_next = self._env.get_constraint_values()

            # store the replay
            self._experience(action_omega, observation, c, c_next)

            # update obs
            observation = observation_next
            episode_length += 1

            if done or (episode_length == self.max_episode_length):
                observation = self._env.reset()
                episode_length = 0

    def _evaluate_batch(self, batch):
        observation = self._as_tensor(batch["observation"])
        action_omega = self._as_tensor(batch["action"])
        c = self._as_tensor(batch["c"])
        c_next = self._as_tensor(batch["c_next"])

        gs = [x(observation) for x in self._models]
        # temp1 = gs[0].view(gs[0].shape[0], 1, -1)
        # temp2 = action.view(action.shape[0], -1, 1)

        c_next_predicted = [c[:, i] + \
                            torch.bmm(x.view(x.shape[0], 1, -1), action_omega.view(action_omega.shape[0], -1, 1)).view(-1) \
                            for i, x in enumerate(gs)]
        losses = [torch.mean((c_next[:, i] - c_next_predicted[i]) ** 2) for i in range(self.num_constraints)]

        return losses

    def _update_batch(self, batch):
        batch = self.replay_buffer.sample(self.batch_size)

        # Update critic
        for_each(lambda x: x.zero_grad(), self._optimizers)
        losses = self._evaluate_batch(batch)
        for_each(lambda x: x.backward(), losses)
        for_each(lambda x: x.step(), self._optimizers)

        return np.asarray([x.item() for x in losses])

    def evaluate(self):
        # Sample steps
        self._sample_steps(self.evaluation_steps)

        self._eval_mode()
        # compute losses
        losses = [list(map(lambda x: x.item(), self._evaluate_batch(batch))) for batch in \
                self.replay_buffer.get_sequential(self.batch_size)]

        losses = np.mean(np.concatenate(losses).reshape(-1, self.num_constraints), axis=0)

        self.replay_buffer.clear()

        self._eval_global_step += 1

        self._train_mode()

        print(f"Validation completed, average loss {losses}")

    def get_safe_action(self, observation, action, environment):
        flag = True
        for i, landmark in enumerate(environment.world.landmarks[0:-1]):
            dist = np.sqrt(np.sum(np.square(environment.world.policy_agents[0].state.p_pos - landmark.state.p_pos))) \
                   - (environment.world.policy_agents[0].size + landmark.size) - 0.03
            if dist <= 0:
                x0 = landmark.state.p_pos[0]
                y0 = landmark.state.p_pos[1]
                landmark0 = landmark
                flag = False
                break
        if flag:
            return action
        x = observation[1]
        y = observation[2]
        V = observation[0]
        theta = observation[3]
        omega = observation[4]
        # print(theta)
        d_omega = action[3] - action[4]
        dt = environment.world.dt
        a1 = x + V * np.cos(theta) * dt + theta * V * np.sin(theta) * dt
        b1 = - V * np.sin(theta) * dt
        a2 = y + V * np.sin(theta) * dt - theta * V * np.cos(theta) * dt
        b2 = V * np.cos(theta) * dt
        c1 = a1 + b1 * theta
        d1 = b1 * dt
        c2 = a2 + b2 * theta
        d2 = b2 * dt
        e1 = -2 * x * c1
        f1 = -2 * x * d1
        e2 = -2 * y * c2
        f2 = -2 * y * d2
        g = d1 * d1 + d2 * d2
        h = 2 * c1 * d1 + 2 * c2 * d2 + f1 + f2
        i = c1 * c1 + c2 * c2 + e1 + e2 + np.square(landmark0.state.p_pos[0]) + \
            np.square(landmark0.state.p_pos[1])
        lower_c = np.square(landmark0.size + 0.01)
        upper_c = inf
        A = landmark0.state.p_pos[0] - x
        B = landmark0.state.p_pos[1] - y
        C = - (landmark0.state.p_pos[0]) * x \
            + np.square(x) \
            - (landmark0.state.p_pos[1]) * y \
            + np.square(y)



        # Make a MOSEK environment
        with mosek.Env() as env:
            # Attach a printer to the environment
            # env.set_Stream(mosek.streamtype.log, streamprinter)

            # Create a task
            with env.Task(0, 0) as task:
                # task.set_Stream(mosek.streamtype.log, streamprinter)
                # Set up and input bounds and linear coefficients
                bkc = [mosek.boundkey.up]
                blc = [-inf]
                buc = [- C - A * c1 - B * c2]
                numvar = 1
                bkx = [mosek.boundkey.fr] * numvar
                blx = [-inf] * numvar
                bux = [inf] * numvar
                temp = 0.12
                c = [- 2.0 * omega - 2 * dt * d_omega]
                asub = [[0]]
                aval = [[A * d1 + B * d2]]

                numvar = len(bkx)
                numcon = len(bkc)

                # Append 'numcon' empty constraints.
                # The constraints will initially have no bounds.
                task.appendcons(numcon)

                # Append 'numvar' variables.
                # The variables will initially be fixed at zero (x=0).
                task.appendvars(numvar)

                for j in range(numvar):
                    # Set the linear term c_j in the objective.
                    task.putcj(j, c[j])
                    # Set the bounds on variable j
                    # blx[j] <= x_j <= bux[j]
                    task.putvarbound(j, bkx[j], blx[j], bux[j])
                    # Input column j of A
                    task.putacol(j,  # Variable (column) index.
                                 # Row index of non-zeros in column j.
                                 asub[j],
                                 aval[j])  # Non-zero Values of column j.
                for i in range(numcon):
                    task.putconbound(i, bkc[i], blc[i], buc[i])

                # Set up and input quadratic objective
                qsubi = [0]
                qsubj = [0]
                qval = [2.0]

                task.putqobj(qsubi, qsubj, qval)

                # Input the objective sense (minimize/maximize)
                task.putobjsense(mosek.objsense.minimize)

                # Optimize
                task.optimize()
                # Print a summary containing information
                # about the solution for debugging purposes
                # task.solutionsummary(mosek.streamtype.msg)

                prosta = task.getprosta(mosek.soltype.itr)
                solsta = task.getsolsta(mosek.soltype.itr)

                # Output a solution
                xx = [0.] * numvar
                task.getxx(mosek.soltype.itr,
                           xx)

                '''if solsta == mosek.solsta.optimal:
                    print("Optimal solution: %s" % xx)
                elif solsta == mosek.solsta.dual_infeas_cer:
                    print("Primal or dual infeasibility.\n")
                elif solsta == mosek.solsta.prim_infeas_cer:
                    print("Primal or dual infeasibility.\n")
                elif mosek.solsta.unknown:
                    print("Unknown solution status")
                else:
                    print("Other solution status")'''

                xx = (xx[0] - omega)/dt
                if xx[0] > temp:
                    xx[0] = temp
                if xx[0] < -temp:
                    xx[0] = -temp

                if np.abs(xx[0]/0.12 - d_omega) < 0.02:
                    return action

                delta_action = xx[0]/0.12 - d_omega
                action[3] = action[3] + delta_action/2
                action[4] = action[4] - delta_action/2

                # temp = action[3] - action[4]

                '''action[3] = + xx[0]/2
                action[4] = - xx[0]/2'''
                return action

    def get_safe_action_old(self, observation, action, c, env):
        self._eval_mode()
        g = [x(self._as_tensor(observation).view(1, -1)) for x in self._models]
        self._train_mode()

        action_omega = action[3] - action[4]
        # Find the lagrange multipliers
        g = [x.data.numpy().reshape(-1) for x in g]
        multipliers = [(np.dot(g_i, action_omega) + c_i) / np.dot(g_i, g_i) for g_i, c_i in zip(g, c)]
        multipliers = [np.clip(x, 0, np.inf) for x in multipliers]
        temp = [(np.dot(g_i, action_omega) + c_i) for g_i, c_i in zip(g, c)]

        # Calculate correction
        correction = np.max(multipliers) * g[np.argmax(multipliers)]

        action[3] = action[3] - correction/2.0
        action[4] = action[4] - correction/2.0
        '''temp = [(np.dot(g_i, action_new) + c_i) for g_i, c_i in zip(g, c)]

        new_obs_n, rew_n, done_n, info_n = env.step([action_new])
        c_next = env.get_constraint_values()'''

        return action

    def train(self):
        print("==========================================================")
        print("Initializing constraint model training...")
        print("==========================================================")
        U.initialize()

        for epoch in range(self.epochs):
            # Just sample episodes for the whole epoch
            self._sample_steps(self.steps_per_epoch)

            # Do the update from memory
            losses = np.mean(np.concatenate([self._update_batch(batch) for batch in \
                                             self.replay_buffer.get_sequential(self.batch_size)]).reshape(-1, self.num_constraints),
                             axis=0)

            self.replay_buffer.clear()
            self._train_global_step += 1

            print(f"Finished epoch {epoch} with losses: {losses}. Running validation ...")
            self.evaluate()
            print("----------------------------------------------------------")

        print("==========================================================")

