from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras.backend as kb
import numpy as np
from typing import Callable, Any


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


class PPOLearner:
    def _init_(
            self,
            horizon: int = 100,
            discount: float = 0.99,
            gae_parameter: float = 0.95,
            clipping_parameter: float = 0.2,
            entropy_coefficient: float = 1e-2,
            critic_discount: float = 1,
            normalize_advantages: bool = True,
            n_actors: int = 1,
            learning_rate: float = 2e-4,
            batch_size: int = 10,
            epochs: int = 10
    ):
        self.horizon = horizon
        self.discount = discount
        self.gae_parameter = gae_parameter
        self.clipping_parameter = clipping_parameter
        self.entropy_coefficient = entropy_coefficient
        self.critic_discount = critic_discount
        self.normalize_advantages = normalize_advantages
        self.n_actors = n_actors
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
            critic: Critic,
            states: np.array,
            returns: np.array
    ) -> None:
        critic.fit(
            x=states,
            y=returns,
            loss="mse",
            optimizer=self.optimizer
        )

    def update_actor(
            self,
            actor: Actor,
            policy_probabilities: np.array,
            rewards: np.array,
            values: np.arrayp,
            advantages: np.array
    ) -> None:
        actor.fit(
            loss=self.make_ppo_loss_callback(
                policy_probabilities=policy_probabilities,
                rewards=rewards,
                values=values,
                advantages=advantages
            ),
            optimizer=self.optimizer
        )





