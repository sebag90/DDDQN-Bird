import random

import torch
import torch.nn as nn


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class QNet(nn.Module):
    def __init__(self, uniform_init):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.value = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

        self.advantage = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

        # model weight initialization
        for param in self.parameters():
            nn.init.uniform_(param, -uniform_init, uniform_init)

    def forward(self, x):
        x = self.encoder(x)
        values = self.value(x)
        advantages = self.advantage(x)

        # q values = values + how good this action is compared to the others
        return values + (advantages - advantages.mean(axis=1, keepdim=True))


class DDQN:
    def __init__(
        self,
        lr=0.000001,
        gamma=0.99,
        initial_epsilon=0.9,
        final_epsilon=0.5,
        uniform_init=0.01,
        episodes=100000
    ):
        # initialize networks and buffer memory
        self.policy_network = QNet(uniform_init).to(DEVICE)
        self.value_network = QNet(uniform_init).to(DEVICE)

        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=lr
        )
        self.criterion = nn.MSELoss()

        # save hyperparameters
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.epsilon_reducing_factor = (
            (initial_epsilon - final_epsilon) / episodes
        )

    def update_value_network(self):
        self.value_network.load_state_dict(
            self.policy_network.state_dict()
        )

    def save(self, filepath):
        torch.save(
            {
                "model": self.policy_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            filepath,
        )

    def load(self, filepath):
        data = torch.load(filepath, map_location="cpu")
        self.policy_network.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.update_value_network()

    @classmethod
    def from_pretrained(cls, filepath):
        model = cls()
        model.load(filepath)
        return model

    def update_epsilon(self):
        self.epsilon -= self.epsilon_reducing_factor

    def epsilon_greedy(self, state, actions):
        if random.random() > self.epsilon:
            return random.choice(actions)
        else:
            _, argmax = self.max_action(state)
            return argmax

    @torch.no_grad()
    def max_action(self, state):
        preds = self.policy_network(state)
        return preds, preds.argmax().item()

    def train_step(self, old_states, actions, rewards, new_states, terminals):
        self.optimizer.zero_grad()

        # predict q values for this step
        predictions = self.policy_network(old_states)
        q_values = torch.sum(predictions * actions, axis=1)

        with torch.no_grad():
            # predict q values from next state
            new_q_values, _ = self.value_network(new_states).max(axis=1)

            # terminal state: Q_y = reward
            # non terminal:   Q_y = reward + gamma * argmax(Q_next)
            targets = rewards + (new_q_values * self.gamma) * (1 - terminals)

        # calculate loss and update weights
        loss = self.criterion(q_values, targets)
        loss.backward()
        self.optimizer.step()
