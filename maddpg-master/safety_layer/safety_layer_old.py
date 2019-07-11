import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from safety_layer.replay_buffer import ReplayBuffer


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
                 act_space):
        self.name = "safe-layer"
        self._env = env
        self.num_constraints = num_constraints  # = num_landmarks - 1 because the last landmark is the target
        self.max_episode_length = 300
        self.batch_size = 1024
        self.lr = 0.1
        self.steps_per_epoch = 6000
        self.epochs = 250
        self.evaluation_steps = 1500
        self.replay_buffer_size = 1000000
        self.num_units = 10
        self._train_global_step = 0
        self.max_replay_buffer = self.batch_size * self.max_episode_length  # 76800
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)  # 1e6
        obs_ph = U.BatchInput(obs_shape, name="observation").get()
        c_ph = [U.BatchInput([1], name="constraints_value"+str(_)).get() for _ in range(self.num_constraints)]
        self.c_next_train, self.c_next_values, self.g_next_values = c_next(
            scope=self.name,
            make_obs_ph=obs_ph,
            act_space=act_space,
            c_ph=c_ph,
            num_constraints=self.num_constraints,
            c_next_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=self.lr),
            grad_norm_clipping=0.5,
            num_units=self.num_units,
        )

    def _experience(self, act, obs, c, c_next):
        # Store transition in the replay buffer.
        self.replay_buffer.add(act, obs, c, c_next)

    def _sample_steps(self, num_steps):
        episode_length = 0
        observation = self._env.reset()
        for step in range(num_steps):
            # action = [np.array([1, 2, 3, 4, 5])]
            action = [self._env.action_space[0].sample()]
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

    def get_safe_action(self, observation, action, c):
        action_origin_shape = action
        observation = np.expand_dims(observation, axis=0)
        action = np.expand_dims(action, axis=0)
        c = np.expand_dims(c, axis=0)
        g_next_values = [self.g_next_values[_](*[observation]) for _ in range(self.num_constraints)]
        g_next_values = [_.reshape(-1) for _ in g_next_values]
        c_next_values = [self.c_next_values[_](*([observation] + [action] + [c])) for _ in range(self.num_constraints)]

        # Find the lagrange multipliers
        temp = np.dot(g_next_values, np.transpose(action)) + c
        multipliers = [c / np.dot(g, g) for g, c in zip(g_next_values, c_next_values)]
        multipliers = [np.clip(x, 0, np.inf) for x in multipliers]

        # Calculate correction
        correction = np.max(multipliers) * g_next_values[np.argmax(multipliers)]

        action_new = action_origin_shape - correction

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
            '''if len(self.replay_buffer) < self.max_replay_buffer:  # replay buffer is not large enough
                continue
            if not epoch % 100 == 0:  # only update every 100 steps
                continue'''
            self.replay_sample_index = self.replay_buffer.make_index(self.batch_size)
            # collect replay sample from all agents
            index = self.replay_sample_index

            action, obs, c, c_next = self.replay_buffer.sample_index(index)
            obs = np.squeeze(obs, axis=1)
            action = np.squeeze(action, axis=1)
            c = np.squeeze(c, axis=1)
            c_next = np.squeeze(c_next, axis=1)

            # train the c_next network
            c_next_loss = [self.c_next_train[_](*([obs] + [action] + [c] + [c_next]))
                           for _ in range(self.num_constraints)]

            self.replay_buffer.clear()
            self._train_global_step += 1

            print(f"Finished epoch {epoch} with losses: {c_next_loss}. Running validation ...")
            print("----------------------------------------------------------")

        print("==========================================================")

