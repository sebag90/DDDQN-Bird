from collections import deque
from dataclasses import dataclass
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


@dataclass
class MemoryItem:
    old_state: torch.tensor
    action: int
    reward: float
    new_state: torch.tensor
    terminal: bool

    def values(self):
        return (
            self.old_state,
            self.action,
            self.reward,
            self.new_state,
            self.terminal
        )


class Buffer(deque):
    def __init__(self, buffer_size, batch_size, n_actions):
        super().__init__(maxlen=buffer_size)
        self.batch_size = batch_size
        self.n_actions = n_actions

    def add(self, state, action, reward, next_state, terminal):
        self.append(
            MemoryItem(state, action, reward, next_state, terminal)
        )

    def get_minibatch(self):
        minibatch = random.sample(self, min(len(self), self.batch_size))
        old_states, actions, rewards, new_states, terminals = zip(
            *(i.values() for i in minibatch)
        )

        old_states = torch.cat(old_states).to(DEVICE)
        new_states = torch.cat(new_states).to(DEVICE)
        terminals = torch.tensor(terminals).long().to(DEVICE)
        rewards = torch.tensor(rewards).to(DEVICE)
        actions = nn.functional.one_hot(
            torch.tensor(actions), num_classes=self.n_actions
        ).to(DEVICE)

        return old_states, actions, rewards, new_states, terminals


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


class ImagePipe:
    def __init__(self):
        self.pipe = transforms.Compose(
            [
                transforms.ToTensor(),
                # remove the floor
                Cropper(0, 0, 288, 404),
                transforms.Grayscale(),
                transforms.Resize((84, 84)),
                BlackWhite(),
            ]
        )

    def __call__(self, image):
        return self.pipe(image).permute(0, 2, 1)
