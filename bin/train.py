import datetime
import os
from pathlib import Path
import time

import torch

from environment.RLflappy import RLBird
from utils.model import DDQN


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main(args):
    if args.headless is True:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # initialize Q table and environment
    w = 288
    h = 512
    fps = 99999 if args.headless is True else args.fps

    env = RLBird(w, h, fps=fps, pipegapsize=100)
    q = DDQN(
        lr=args.lr,
        n_actions=len(env.action_space),
        buffer_size=args.buffer_size,
        gamma=args.gamma,
        initial_epsilon=args.init_epsilon,
        final_epsilon=args.final_epsilon,
        uniform_init=args.init,
        episodes=args.episodes
    )

    if args.resume is not None:
        q.load(Path(args.resume))

    # start training
    max_steps = 0
    mean_reward = 0
    max_reward = 0
    average_steps = 0
    episode_counter = 0

    # actions: 1=flap, 0=no flap
    actions = [1, 0]
    t_init = time.time()
    for episode in range(args.episodes):
        terminal = False
        steps = 0
        this_reward = 0

        # reset environment
        env.reset(verbose=True)
        state = env.get_state()
        state = q.prepare_image(state)
        state = state.unsqueeze(0).repeat_interleave(4, 1).to(DEVICE)

        while terminal is False:
            # select action and get current Q value
            action = q.epsilon_greedy(state, actions)

            # take step
            reward, terminal = env.take_step(action, verbose=True)
            next_state = env.get_state()

            # prepare current state and concatenate it to old one
            next_state = q.prepare_image(next_state).to(DEVICE)
            next_state = torch.cat(
                (state.squeeze(0)[1:, :, :], next_state)
            ).unsqueeze(0)
            # add element to buffer replay
            q.buffer.add(
                state, action, reward, next_state, terminal
            )

            # update Q value
            q.train_step()

            steps += 1
            this_reward += reward
            mean_reward += reward

            # update state
            state = next_state

        q.update_epsilon()
        average_steps += steps
        episode_counter += 1

        if steps > max_steps:
            max_steps = steps

        if this_reward > max_reward:
            max_reward = this_reward

        if (episode + 1) % args.print_every == 0:
            trailing = f">{len(str(args.episodes))}d"
            current_step = f"{episode + 1:{trailing}}/{args.episodes}"
            print_time = str(
                datetime.timedelta(seconds=int(time.time() - t_init))
            )
            mean_reward = round(mean_reward/episode_counter, 5)
            print(
                f"episode: {current_step:13} | "
                f"epsilon: {round(q.epsilon, 6):,.6f} | "
                f"max steps: {max_steps:4} | "
                f"avg steps: {average_steps/episode_counter:5.1f} | "
                f"max reward: {max_reward:7.2f} | "
                f"avg reward: {mean_reward:5.1f} | "
                f"time: {print_time}",
                flush=True,
            )
            mean_reward = 0
            average_steps = 0
            episode_counter = 0

        if (episode + 1) % args.update_network == 0:
            q.update_value_network()

        if (episode + 1) % args.checkpoint == 0:
            os.makedirs("checkpoints", exist_ok=True)
            q.save(f"checkpoints/checkpoint.{episode + 1}.pt")

    # save last model
    q.update_value_network()
    os.makedirs("checkpoints", exist_ok=True)
    q.save(f"checkpoints/checkpoint.{episode + 1}.pt")
