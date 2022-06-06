import pygame

from environment.flappy import FlappyBird


class RLBird(FlappyBird):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_assets()
        self.action_space = [1, 0]

    def load_assets(self):
        pygame.init()
        self.images["base"] = pygame.image.load(
            "environment/assets/sprites/base.png"
        ).convert_alpha()

        original_bg = pygame.image.load(self.backgrounds_list[0]).convert()
        self.images["background"] = pygame.Surface(
            (original_bg.get_width(), original_bg.get_height())
        )

        # select random player sprites
        self.images["player"] = (
            pygame.image.load(self.player_list[0][0]).convert_alpha(),
            pygame.image.load(self.player_list[0][1]).convert_alpha(),
            pygame.image.load(self.player_list[0][2]).convert_alpha(),
        )

        self.images["pipe"] = (
            pygame.transform.flip(
                pygame.image.load(self.pipes_list[0]).convert_alpha(),
                False,
                True
            ),
            pygame.image.load(self.pipes_list[0]).convert_alpha(),
        )

        # hitmask for pipes
        self.hitmasks["pipe"] = (
            self.getHitmask(self.images["pipe"][0]),
            self.getHitmask(self.images["pipe"][1]),
        )

        # hitmask for player
        self.hitmasks["player"] = (
            self.getHitmask(self.images["player"][0]),
            self.getHitmask(self.images["player"][1]),
            self.getHitmask(self.images["player"][2]),
        )

    def render(self):
        self.screen.blit(self.images["background"], (0, 0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            self.screen.blit(self.images["pipe"][0], (uPipe["x"], uPipe["y"]))
            self.screen.blit(self.images["pipe"][1], (lPipe["x"], lPipe["y"]))

        self.screen.blit(self.images["base"], (self.basex, self.basey))

        self.screen.blit(
            self.images["player"][self.playerIndex],
            (self.playerx, self.playery)
        )

        pygame.display.update()
        self.fpsclock.tick(self.fps)

    def get_state(self):
        return pygame.surfarray.array3d(pygame.display.get_surface())

    def reset(self, verbose=False):
        self.score = 0
        self.playerIndex = 1

        self.playerx = int(self.screenwidth * 0.2)
        self.playery = self.screenheight // 2

        self.basex = 0
        self.baseShift = (
            self.images["base"].get_width()
            - self.images["background"].get_width()
        )

        # get 2 new pipes to add to upperPipes lowerPipes list
        newPipe1 = self.getRandomPipe()
        newPipe2 = self.getRandomPipe()

        # list of upper pipes
        self.upperPipes = [
            {"x": self.screenwidth / 2 + 200, "y": newPipe1[0]["y"]},
            {
                "x": self.screenwidth / 2 + 200 + (self.screenwidth / 2),
                "y": newPipe2[0]["y"],
            },
        ]

        # list of lowerpipe
        self.lowerPipes = [
            {"x": self.screenwidth / 2 + 200, "y": newPipe1[1]["y"]},
            {
                "x": self.screenwidth / 2 + 200 + (self.screenwidth / 2),
                "y": newPipe2[1]["y"],
            },
        ]

        # player velocity, max velocity,
        # downward acceleration,
        # acceleration on flap
        self.pipeVelX = -4
        # player's velocity along Y, default same as playerFlapped
        self.playerVelY = 0
        self.playerMaxVelY = 10  # max vel along Y, max descend speed
        self.playerMinVelY = -8  # min vel along Y, max ascend speed
        self.playerAccY = 1  # players downward acceleration
        self.playerFlapAcc = -9  # players speed on flapping
        self.playerFlapped = False  # True when player flaps

        if verbose is True:
            self.render()

    def take_step(
        self, action, verbose=False, base_r=0.1, point_r=1, death_r=-1
    ):
        pygame.event.pump()
        reward = base_r

        if action == 1:
            if self.playery > -2 * self.images["player"][0].get_height():
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True

        # check for score
        playerMidPos = self.playerx + self.images["player"][0].get_width() / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe["x"] + self.images["pipe"][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                reward = point_r

        self.basex = -((-self.basex + 100) % self.baseShift)

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False

        self.playerHeight = (
            self.images["player"][self.playerIndex].get_height()
        )
        self.playery += min(
            self.playerVelY, self.basey - self.playery - self.playerHeight
        )

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe["x"] += self.pipeVelX
            lPipe["x"] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 3 > len(self.upperPipes) > 0 and 0 < self.upperPipes[0]["x"] < 5:
            newPipe = self.getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if (
            len(self.upperPipes) > 0
            and self.upperPipes[0]["x"] < -self.images["pipe"][0].get_width()
        ):
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # check for crash here
        crashTest = self.checkCrash(
            {"x": self.playerx, "y": self.playery, "index": self.playerIndex},
            self.upperPipes,
            self.lowerPipes,
        )

        terminal = crashTest[0]
        if terminal is True:
            reward = death_r

        if verbose is True:
            self.render()

        return reward, terminal
