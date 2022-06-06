from environment.flappy import FlappyBird


def main(args):
    game = FlappyBird(288, 512, pipegapsize=100)
    game.main()
