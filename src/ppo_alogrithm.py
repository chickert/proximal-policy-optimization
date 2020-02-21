from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras.backend as kb
import numpy as np
from typing import Callable, Any, Tuple


class Critic(Sequential):
    def __init__(
            self,
            state_space_dimension: int,
            hidden_layer_size: int
    ):
        super().__init__()
        self.add(
            Dense(
                hidden_layer_size,
                input_dim=state_space_dimension,
                activation="relu"
            )
        )
        self.add(Dense(1))


class Actor(Sequential):
    def __init__(
            self,
            state_space_dimension: int,
            action_space_dimension: int,
            hidden_layer_size: int,
            action_map: Callable
    ):
        super().__init__()
        self.add(
            Dense(
                hidden_layer_size,
                input_dim=state_space_dimension,
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

    def sample_action_from_policy(
        self,
        state: np.array
    ) -> np.array:
        action_probabilities = self.predict(state)
        selected_action = np.random.choice(self._action_space_dimension, p=action_probabilities)
        return self._action_map(selected_action)


class PPOLearner:
    def _init_(
            self,
            actor: Actor,
            critic: Critic,
            horizon: int = 100,
            discount: float = 0.99,
            gae_parameter: float = 0.95,
            clipping_parameter: float = 0.2,
            entropy_coefficient: float = 1e-2,
            critic_discount: float = 1,
            normalize_advantages: bool = True,
            n_actors: int = 1,
            max_iterations: int = 100,
            learning_rate: float = 2e-4,
            batch_size: int = 10,
            epochs: int = 10
    ):
        self.actor = actor
        self.critic = critic
        self.horizon = horizon
        self.discount = discount
        self.gae_parameter = gae_parameter
        self.clipping_parameter = clipping_parameter
        self.entropy_coefficient = entropy_coefficient
        self.critic_discount = critic_discount
        self.normalize_advantages = normalize_advantages
        self.n_actors = n_actors
        self.max_iterations = max_iterations
        self.optimizer = Adam(
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size
        )

    def calculate_returns(
            self,
            values: np.array,
            rewards: np.array,
    ) -> np.array:
        returns = []
        advantage_estimate = 0
        for i in reversed(range(len(rewards))):
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
        policy_probabilities: np.array,
        rewards: np.array,
        values: np.array,
        advantages: np.array
    ) -> Callable:
        def loss(
                dummy_target: Any,
                updated_policy_probabilities: np.array
        ) -> float:
            eps = 1e-10
            ratio = kb.exp(
                kb.log(updated_policy_probabilities + eps) - kb.log(policy_probabilities + eps)
            )
            actor_loss = -kb.mean(
                kb.minimum(
                    ratio,
                    kb.clip(
                        ratio,
                        min_value=1 - self.clipping_parameter,
                        max_value=1 + self.clipping_parameter
                    )
                )
            ) * advantages
            critic_loss = kb.mean(kb.square(rewards - values))
            total_loss = actor_loss + self.critic_discount*critic_loss - self.entropy_coefficient * kb.mean(
                -(updated_policy_probabilities * kb.log(updated_policy_probabilities + eps)))
            return total_loss
        return loss

    def update_critic(
            self,
            states: np.array,
            returns: np.array
    ) -> None:
        self.critic.fit(
            x=states,
            y=returns,
            loss="mse",
            optimizer=self.optimizer
        )

    def update_actor(
            self,
            policy_probabilities: np.array,
            rewards: np.array,
            values: np.array,
            advantages: np.array
    ) -> None:
        self.actor.fit(
            loss=self.make_ppo_loss_callback(
                policy_probabilities=policy_probabilities,
                rewards=rewards,
                values=values,
                advantages=advantages
            ),
            optimizer=self.optimizer
        )

    def generate_trajectory(self, environment) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
        states = []
        actions = []
        rewards = []
        values = []
        policy_probabilities = []
        state = environment._get_obs()
        for step in range(self.horizon):
            # Sample from policy and receive feedback from environment
            action = self.actor.sample_action_from_policy(state)
            new_state, reward, done, info = environment.step(action)

            # Store information from step
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(self.critic.predict(state))
            policy_probabilities.append(self.actor.predict(state))

            # Update state
            state = new_state

            # Reset state if task is done
            if done:
                environment.reset()

        return np.array(states), np.array(actions), np.array(values), np.array(rewards), np.array(policy_probabilities)

    def train(self, environment):
        for _ in range(self.max_iterations):
            states, actions, values, rewards, policy_probabilities = self.generate_trajectory(environment)
            returns = self.calculate_returns(values, rewards)
            advantages = self.calculate_advantages(values, returns)
            self.update_critic(states, returns)
            self.update_actor(policy_probabilities, rewards, values, advantages)













