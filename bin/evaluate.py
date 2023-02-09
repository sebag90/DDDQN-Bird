from pathlib import Path

import torch

from environment.RLflappy import RLBird
from model.model import DDQN
from model.utils import ImagePipe


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main(args):
    # initialize Q table and environment
    w = 288
    h = 512

    env = RLBird(w, h, fps=args.fps, pipegapsize=100)
    image_pipe = ImagePipe()

    # load model
    q = DDQN.from_pretrained(Path(args.file))
    q.policy_network.eval()

    terminal = False
    score = 0
    total_reward = 0

    env.reset(verbose=True)
    state = env.get_state()
    state = image_pipe(state)
    state = state.unsqueeze(0).repeat_interleave(4, 1).to(DEVICE)

    while terminal is False:
        # select best action
        preds, action = q.max_action(state)
        preds = preds.squeeze(0)

        # take step
        reward, terminal = env.take_step(action, verbose=True)
        next_state = env.get_state()
        next_state = image_pipe(next_state)

        next_state = (
            torch.cat((state.squeeze(0)[1:, :, :], next_state))
            .unsqueeze(0)
            .to(DEVICE)
        )
        if reward == 1:
            score += reward

        total_reward += reward

        state = next_state
        print(
            f"Score: {score:5} | "
            f"Reward: {round(total_reward, 1):5} | "
            f"Action: {action} | "
            f"No flap: {round(preds[0].item(), 3):6} | "
            f"Flap: {round(preds[1].item(), 3):}"
        )

    print(f"Game Over\nScore: {score}")
