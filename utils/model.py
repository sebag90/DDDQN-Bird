from dataclasses import dataclass
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms


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


class Cropper:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        return transforms.functional.crop(x, *self.args, **self.kwargs)


class BlackWhite:
    def __call__(self, x):
        x[x > 0] = 255
        return x


@dataclass
class MemoryItem:
    old_state: torch.tensor
    action: int
    reward: float
    new_state: torch.tensor
    terminal: bool


class Buffer(list):
    def __init__(self, max_len, batch_size, n_actions):
        self.max_len = max_len
        self.batch_size = batch_size
        self.n_actions = n_actions

    def add(self, state, action, reward, next_state, terminal):
        self.append(
            MemoryItem(state, action, reward, next_state, terminal)
        )

        if len(self) > self.max_len:
            self.pop(0)

    def get_minibatch(self):
        minibatch = random.sample(
            self, min(len(self), self.batch_size)
        )

        old_states = (
            torch.cat(tuple(i.old_state for i in minibatch))
            .to(DEVICE)
        )
        new_states = (
            torch.cat(tuple(i.new_state for i in minibatch))
            .to(DEVICE)
        )
        terminals = (
            torch.tensor([i.terminal for i in minibatch])
            .long()
            .to(DEVICE)
        )
        rewards = (
            torch.tensor([i.reward for i in minibatch])
            .to(DEVICE)
        )
        actions = (
            torch.nn.functional.one_hot(
                torch.tensor([i.action for i in minibatch]),
                num_classes=self.n_actions)
            .to(DEVICE)
        )
        return old_states, actions, rewards, new_states, terminals


class DDQN:
    def __init__(
        self,
        n_actions,
        lr=0.000001,
        gamma=0.99,
        buffer_size=10000,
        batch_size=32,
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
        self.buffer = Buffer(buffer_size, batch_size, n_actions)

        # save hyperparameters
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.epsilon_reducing_factor = (
            (initial_epsilon - final_epsilon) / episodes
        )
        self.n_actions = n_actions

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

    @staticmethod
    def prepare_image(image):
        image_pipe = transforms.Compose(
            [
                transforms.ToTensor(),
                # remove the floor
                Cropper(0, 0, 288, 404),
                transforms.Grayscale(),
                transforms.Resize((84, 84)),
                BlackWhite(),
            ]
        )

        return image_pipe(image).permute(0, 2, 1)

    def train_step(self):
        (old_states,
         actions,
         rewards,
         new_states,
         terminals) = self.buffer.get_minibatch()
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
