from typing import Callable, Any, Tuple, Union, Dict, List

import keras.backend as kb
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from robot_environments.pusher import PusherEnv
from robot_environments.reacher import ReacherEnv
from robot_environments.reacher_wall import ReacherWallEnv
import matplotlib.pyplot as plt

kb.set_floatx("float64")


class Critic(Sequential):
    def __init__(
            self,
            state_space_dimension: int,
            hidden_layer_units: List[int]
    ):
        super().__init__()
        self.add(
            Dense(
                hidden_layer_units[0],
                input_shape=(state_space_dimension,),
                activation="relu"
            )
        )
        for units in hidden_layer_units[1:]:
            self.add(
                Dense(
                    units,
                    activation="relu"
                )
            )
        self.add(Dense(1))

    def predict_value(
            self,
            state: List[float]
    ) -> float:
        return self.predict(np.array([state]))[0]


class Actor(Sequential):
    def __init__(
            self,
            state_space_dimension: int,
            action_space_dimension: int,
            hidden_layer_units: List[int],
            action_map: Dict[int, np.array]
    ):
        super().__init__()
        self.add(
            Dense(
                hidden_layer_units[0],
                input_dim=state_space_dimension,
                activation="relu"
            )
        )
        for units in hidden_layer_units[1:]:
            self.add(
                Dense(
                    units,
                    activation="relu"
                )
            )
        self.add(
            Dense(
                action_space_dimension,
                activation="softmax"
            )
        )
        self._action_space_dimension = action_space_dimension
        self._action_map = action_map

    def get_policy(
            self,
            state: List[float]
    ) -> np.array:
        return self.predict(np.array([state])).flatten()

    def sample_action_from_policy(
        self,
        state: List[float]
    ) -> np.array:
        action_probabilities = self.get_policy(state)
        selected_action = np.random.choice(self._action_space_dimension, p=action_probabilities)
        return self._action_map[selected_action]


class PPOLearner:
    def __init__(
            self,
            environment: Union[ReacherEnv, ReacherWallEnv, PusherEnv],
            state_space_dimension: int,
            action_space_dimension: int,
            action_map: Dict[int, np.array],
            critic_hidden_layer_units: List[int],
            actor_hidden_layer_units: List[int],
            horizon: int = 200,
            discount: float = 0.99,
            gae_parameter: float = 0.95,
            clipping_parameter: float = 0.2,
            entropy_coefficient: float = 1e-3,
            critic_discount: float = 0.5,
            normalize_advantages: bool = True,
            max_iterations: int = 200,
            batch_size: int = 64,
            epochs: int = 10,
            learning_rate: float = 1e-3
    ):
        #
        self.environment = environment
        self.actor = Actor(
            state_space_dimension=state_space_dimension,
            action_space_dimension=action_space_dimension,
            hidden_layer_units=actor_hidden_layer_units,
            action_map=action_map
        )
        self.critic = Critic(
            state_space_dimension=state_space_dimension,
            hidden_layer_units=critic_hidden_layer_units
        )

        # Set optimizer attributes
        self.optimizer = Adam(learning_rate=learning_rate)

        # Set hyperparameter attributes
        self.horizon = horizon
        self.discount = discount
        self.gae_parameter = gae_parameter
        self.clipping_parameter = clipping_parameter
        self.entropy_coefficient = entropy_coefficient
        self.critic_discount = critic_discount
        self.normalize_advantages = normalize_advantages
        self.max_iterations = max_iterations
        self.epochs = epochs
        self.batch_size = batch_size

    def calculate_returns(
            self,
            values: np.array,
            rewards: np.array,
    ) -> np.array:
        returns = []
        advantage_estimate = 0
        for i in reversed(range(len(rewards) - 1)):
            delta = rewards[i] + self.discount*values[i + 1] - values[i]
            advantage_estimate = delta + self.discount*self.gae_parameter*advantage_estimate
            returns.insert(0, advantage_estimate + values[i])
        return np.array(returns)

    def calculate_advantages(
            self,
            values: np.array,
            returns: np.array,
    ) -> np.array:
        advantages = returns - values[:-1]
        if self.normalize_advantages:
            return (np.array(advantages) - np.mean(advantages)) / (np.std(advantages) + 1e-10)
        else:
            return advantages

    def make_ppo_loss_callback(
        self,
        policy_probabilities: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        advantages: np.ndarray
    ) -> Callable:
        def loss(
                dummy_target: Any,
                updated_policy_probabilities: np.ndarray
        ) -> float:
            eps = 1e-10
            ratio = kb.exp(
                kb.log(updated_policy_probabilities + eps) - kb.log(policy_probabilities + eps)
            )
            actor_loss = -kb.mean(
                kb.minimum(
                    ratio * advantages,
                    kb.clip(
                        ratio,
                        min_value=1 - self.clipping_parameter,
                        max_value=1 + self.clipping_parameter
                    ) * advantages
                )
            )
            critic_loss = kb.mean(
                kb.square(rewards - values)
            )
            entropy_loss = kb.mean(
                (updated_policy_probabilities * kb.log(updated_policy_probabilities + eps))
            )
            return actor_loss + self.critic_discount*critic_loss + self.entropy_coefficient*entropy_loss
        return loss

    def update_critic(
            self,
            states: np.array,
            returns: np.array
    ) -> None:
        self.critic.compile(
            optimizer=self.optimizer,
            loss="mse"
        )
        self.critic.fit(
            x=states,
            y=returns,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )

    def update_actor(
            self,
            states: np.ndarray,
            policy_probabilities: np.ndarray,
            rewards: np.ndarray,
            values: np.ndarray,
            advantages: np.ndarray
    ) -> None:
        self.actor.compile(
            optimizer=self.optimizer,
            loss=[self.make_ppo_loss_callback(
                policy_probabilities=policy_probabilities,
                rewards=rewards,
                values=values,
                advantages=advantages
            )],
        )
        self.actor.fit(
            x=states,
            y=states,  # dummy argument
            epochs=self.epochs,
            steps_per_epoch=int(np.round(self.horizon / self.batch_size)),
            verbose=2
        )

    def generate_trajectory(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # Initialize trjectory
        states = []
        rewards = []
        values = []
        policy_probabilities = []

        # Get initial state
        state = self.environment._get_obs()

        # Generate a sample trajectory under the current policy
        for step in range(self.horizon):
            # Sample from policy and receive feedback from environment
            action = self.actor.sample_action_from_policy(state)
            new_state, reward, done, info = self.environment.step(action)

            # Store information from step
            states.append(state)
            rewards.append(reward)
            values.append(self.critic.predict_value(state))
            policy_probabilities.append(self.actor.get_policy(state))

            # Update state
            state = new_state

            # Reset state if task is done
            if done:
                self.environment.reset()

        return np.array(states), np.array(values), np.array(rewards), np.array(policy_probabilities)

    def train(self):
        mean_rewards = []
        for iters in range(self.max_iterations):

            # Generate trajectory under current policy
            states, values, rewards, policy_probabilities = self.generate_trajectory()

            # Calculate returns and advantages
            returns = self.calculate_returns(
                values=values,
                rewards=rewards
            )
            advantages = self.calculate_advantages(
                values=values,
                returns=returns
            )

            # Perform gradient update on actor and critic
            self.update_actor(
                states=states[:-1],
                policy_probabilities=policy_probabilities[:-1],
                rewards=rewards[:-1],
                values=values[:-1],
                advantages=advantages
            )
            self.update_critic(
                states=states[:-1],
                returns=returns
            )
            mean_rewards.append(np.mean(rewards))
            if iters % 20 == 0:
                plt.plot(mean_rewards)
                plt.show()









