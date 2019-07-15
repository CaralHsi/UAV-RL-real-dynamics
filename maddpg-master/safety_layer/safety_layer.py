import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U
import torch

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from safety_layer.replay_buffer import ReplayBuffer
from torch.optim import Adam
from safety_layer.constraint_model import ConstraintModel
from safety_layer.list import for_each



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
        self.epochs = 1
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
                                        5) \
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
            c = self._env.get_constraint_values()
            observation_next, _, done, _ = self._env.step(action)
            c_next = self._env.get_constraint_values()

            # store the replay
            self._experience(action, observation, c, c_next)

            # update obs
            observation = observation_next
            episode_length += 1

            if done or (episode_length == self.max_episode_length):
                observation = self._env.reset()
                episode_length = 0

    def _evaluate_batch(self, batch):
        observation = self._as_tensor(batch["observation"])
        action = self._as_tensor(batch["action"])
        c = self._as_tensor(batch["c"])
        c_next = self._as_tensor(batch["c_next"])

        gs = [x(observation) for x in self._models]
        # temp1 = gs[0].view(gs[0].shape[0], 1, -1)
        # temp2 = action.view(action.shape[0], -1, 1)

        c_next_predicted = [c[:, i] + \
                            torch.bmm(x.view(x.shape[0], 1, -1), action.view(action.shape[0], -1, 1)).view(-1) \
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


    def get_safe_action(self, observation, action, c, env):
        self._eval_mode()
        g = [x(self._as_tensor(observation).view(1, -1)) for x in self._models]
        self._train_mode()

        # Find the lagrange multipliers
        g = [x.data.numpy().reshape(-1) for x in g]
        multipliers = [(np.dot(g_i, action) + c_i) / np.dot(g_i, g_i) for g_i, c_i in zip(g, c)]
        multipliers = [np.clip(x, 0, np.inf) for x in multipliers]
        '''new_obs_n, rew_n, done_n, info_n = env.step([action])
        c_next = env.get_constraint_values()
        c_next_prediction = (np.dot(g[0], action) + c[0])'''

        # Calculate correction
        correction = np.max(multipliers) * g[np.argmax(multipliers)]

        action_new = action - correction * 0.0

        return action_new

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

