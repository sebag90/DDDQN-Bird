from itertools import cycle
import random
import sys
import pygame
from pygame.locals import *


class FlappyBird:
    def __init__(self, screenwidth, screenheight, fps=30, pipegapsize=100):
        self.screenwidth = screenwidth
        self.screenheight = screenheight
        self.fps = fps
        self.basey = self.screenheight * 0.79
        self.pipegapsize = pipegapsize
        self.screen = pygame.display.set_mode(
            (self.screenwidth, self.screenheight)
        )
        self.fpsclock = pygame.time.Clock()

        self.images, self.sounds, self.hitmasks = {}, {}, {}
        self.player_list = (
            # red bird
            (
                "environment/assets/sprites/redbird-upflap.png",
                "environment/assets/sprites/redbird-midflap.png",
                "environment/assets/sprites/redbird-downflap.png",
            ),
            # blue bird
            (
                "environment/assets/sprites/bluebird-upflap.png",
                "environment/assets/sprites/bluebird-midflap.png",
                "environment/assets/sprites/bluebird-downflap.png",
            ),
            # yellow bird
            (
                "environment/assets/sprites/yellowbird-upflap.png",
                "environment/assets/sprites/yellowbird-midflap.png",
                "environment/assets/sprites/yellowbird-downflap.png",
            ),
        )

        # list of backgrounds
        self.backgrounds_list = (
            "environment/assets/sprites/background-day.png",
            "environment/assets/sprites/background-night.png",
        )

        # list of pipes
        self.pipes_list = (
            "environment/assets/sprites/pipe-green.png",
            "environment/assets/sprites/pipe-red.png",
        )

    def load_assets(self):
        pygame.init()
        # numbers sprites for score display
        nums = list()
        for number in range(10):
            nums.append(
                pygame.image.load(
                    f"environment/assets/sprites/{number}.png"
                ).convert_alpha()
            )
        self.images["numbers"] = tuple(nums)

        # game over sprite
        self.images["gameover"] = pygame.image.load(
            "environment/assets/sprites/gameover.png"
        ).convert_alpha()
        # message sprite for welcome screen
        self.images["message"] = pygame.image.load(
            "environment/assets/sprites/message.png"
        ).convert_alpha()
        # base (ground) sprite
        self.images["base"] = pygame.image.load(
            "environment/assets/sprites/base.png"
        ).convert_alpha()

        # self.sounds
        if "win" in sys.platform:
            soundExt = ".wav"
        else:
            soundExt = ".ogg"

        self.sounds["die"] = pygame.mixer.Sound(
            "environment/assets/audio/die" + soundExt
        )
        self.sounds["hit"] = pygame.mixer.Sound(
            "environment/assets/audio/hit" + soundExt
        )
        self.sounds["point"] = pygame.mixer.Sound(
            "environment/assets/audio/point" + soundExt
        )
        self.sounds["swoosh"] = pygame.mixer.Sound(
            "environment/assets/audio/swoosh" + soundExt
        )
        self.sounds["wing"] = pygame.mixer.Sound(
            "environment/assets/audio/wing" + soundExt
        )

        randBg = random.randint(0, len(self.backgrounds_list) - 1)
        self.images["background"] = pygame.image.load(
            self.backgrounds_list[randBg]
        ).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(self.player_list) - 1)
        self.images["player"] = (
            pygame.image.load(self.player_list[randPlayer][0]).convert_alpha(),
            pygame.image.load(self.player_list[randPlayer][1]).convert_alpha(),
            pygame.image.load(self.player_list[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipeindex = random.randint(0, len(self.pipes_list) - 1)
        self.images["pipe"] = (
            pygame.transform.flip(
                pygame.image.load(self.pipes_list[pipeindex]).convert_alpha(),
                False,
                True,
            ),
            pygame.image.load(self.pipes_list[pipeindex]).convert_alpha(),
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

    def main(self):
        self.load_assets()
        pygame.display.set_caption("Flappy Bird")

        while True:
            # select random background sprites
            movementInfo = self.showWelcomeAnimation()
            crashInfo = self.mainGame(movementInfo)
            self.showGameOverScreen(crashInfo)

    def showWelcomeAnimation(self):
        """Shows welcome screen animation of flappy bird"""
        # index of player to blit on screen
        playerIndex = 0
        playerIndexGen = cycle([0, 1, 2, 1])
        # iterator used to change playerIndex after every 5th iteration
        loopIter = 0

        playerx = int(self.screenwidth * 0.2)
        playery = int(
            (self.screenheight - self.images["player"][0].get_height()) / 2
        )

        messagex = int(
            (self.screenwidth - self.images["message"].get_width()) / 2
        )
        messagey = int(self.screenheight * 0.12)

        basex = 0
        # amount by which base can maximum shift to left
        baseShift = (
            self.images["base"].get_width()
            - self.images["background"].get_width()
        )

        # player shm for up-down motion on welcome screen
        playerShmVals = {"val": 0, "dir": 1}

        while True:
            for event in pygame.event.get():
                if event.type == QUIT or (
                    event.type == KEYDOWN and event.key == K_ESCAPE
                ):
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN and (
                    event.key == K_SPACE or event.key == K_UP
                ):
                    # make first flap sound and return values for mainGame
                    self.sounds["wing"].play()
                    return {
                        "playery": playery + playerShmVals["val"],
                        "basex": basex,
                        "playerIndexGen": playerIndexGen,
                    }

            # adjust playery, playerIndex, basex
            if (loopIter + 1) % 5 == 0:
                playerIndex = next(playerIndexGen)
            loopIter = (loopIter + 1) % 30
            basex = -((-basex + 4) % baseShift)
            self.playerShm(playerShmVals)

            # draw sprites
            self.screen.blit(self.images["background"], (0, 0))
            self.screen.blit(
                self.images["player"][playerIndex],
                (playerx, playery + playerShmVals["val"]),
            )
            self.screen.blit(self.images["message"], (messagex, messagey))
            self.screen.blit(self.images["base"], (basex, self.basey))

            pygame.display.update()
            self.fpsclock.tick(self.fps)

    def mainGame(self, movementInfo):
        score = playerIndex = loopIter = 0
        playerIndexGen = movementInfo["playerIndexGen"]
        playerx, playery = int(self.screenwidth * 0.2), movementInfo["playery"]

        basex = movementInfo["basex"]
        baseShift = (
            self.images["base"].get_width() - self.images["background"].get_width()
        )

        # get 2 new pipes to add to upperPipes lowerPipes list
        newPipe1 = self.getRandomPipe()
        newPipe2 = self.getRandomPipe()

        # list of upper pipes
        upperPipes = [
            {"x": self.screenwidth + 200, "y": newPipe1[0]["y"]},
            {
                "x": self.screenwidth + 200 + (self.screenwidth / 2),
                "y": newPipe2[0]["y"],
            },
        ]

        # list of lowerpipe
        lowerPipes = [
            {"x": self.screenwidth + 200, "y": newPipe1[1]["y"]},
            {
                "x": self.screenwidth + 200 + (self.screenwidth / 2),
                "y": newPipe2[1]["y"],
            },
        ]

        dt = self.fpsclock.tick(self.fps) / 1000
        pipeVelX = -128 * dt

        # player velocity, max velocity,
        # downward acceleration,
        # acceleration on flap

        # player's velocity along Y, default same as playerFlapped
        playerVelY = -9
        playerMaxVelY = 10  # max vel along Y, max descend speed
        playerAccY = 1  # players downward acceleration
        playerRot = 45  # player's rotation
        playerVelRot = 3  # angular speed
        playerRotThr = 20  # rotation threshold
        playerFlapAcc = -9  # players speed on flapping
        playerFlapped = False  # True when player flaps

        while True:
            for event in pygame.event.get():
                if event.type == QUIT or (
                    event.type == KEYDOWN and event.key == K_ESCAPE
                ):
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN and (
                    event.key == K_SPACE or event.key == K_UP
                ):
                    if playery > -2 * self.images["player"][0].get_height():
                        playerVelY = playerFlapAcc
                        playerFlapped = True
                        self.sounds["wing"].play()

            # check for crash here
            crashTest = self.checkCrash(
                {"x": playerx, "y": playery, "index": playerIndex},
                upperPipes,
                lowerPipes,
            )

            if crashTest[0]:
                return {
                    "y": playery,
                    "groundCrash": crashTest[1],
                    "basex": basex,
                    "upperPipes": upperPipes,
                    "lowerPipes": lowerPipes,
                    "score": score,
                    "playerVelY": playerVelY,
                    "playerRot": playerRot,
                }

            # check for score
            playerMidPos = playerx + self.images["player"][0].get_width() / 2
            for pipe in upperPipes:
                pipeMidPos = pipe["x"] + self.images["pipe"][0].get_width() / 2
                if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                    score += 1
                    self.sounds["point"].play()

            # playerIndex basex change
            if (loopIter + 1) % 3 == 0:
                playerIndex = next(playerIndexGen)
            loopIter = (loopIter + 1) % 30
            basex = -((-basex + 100) % baseShift)

            # rotate the player
            if playerRot > -90:
                playerRot -= playerVelRot

            # player's movement
            if playerVelY < playerMaxVelY and not playerFlapped:
                playerVelY += playerAccY
            if playerFlapped:
                playerFlapped = False

                # more rotation to cover the threshold
                # (calculated in visible rotation)
                playerRot = 45

            playerHeight = self.images["player"][playerIndex].get_height()
            playery += min(playerVelY, self.basey - playery - playerHeight)

            # move pipes to left
            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                uPipe["x"] += pipeVelX
                lPipe["x"] += pipeVelX

            # add new pipe when first pipe is about to touch left of screen
            if 3 > len(upperPipes) > 0 and 0 < upperPipes[0]["x"] < 5:
                newPipe = self.getRandomPipe()
                upperPipes.append(newPipe[0])
                lowerPipes.append(newPipe[1])

            # remove first pipe if its out of the screen
            if (
                len(upperPipes) > 0
                and upperPipes[0]["x"] < -self.images["pipe"][0].get_width()
            ):
                upperPipes.pop(0)
                lowerPipes.pop(0)

            # draw sprites
            self.screen.blit(self.images["background"], (0, 0))

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                self.screen.blit(
                    self.images["pipe"][0], (uPipe["x"], uPipe["y"])
                )
                self.screen.blit(
                    self.images["pipe"][1], (lPipe["x"], lPipe["y"])
                )

            self.screen.blit(self.images["base"], (basex, self.basey))
            # print score so player overlaps the score
            self.showScore(score)

            # Player rotation has a threshold
            visibleRot = playerRotThr
            if playerRot <= playerRotThr:
                visibleRot = playerRot

            playerSurface = pygame.transform.rotate(
                self.images["player"][playerIndex], visibleRot
            )
            self.screen.blit(playerSurface, (playerx, playery))

            pygame.display.update()
            self.fpsclock.tick(self.fps)

    def showGameOverScreen(self, crashInfo):
        """crashes the player down and shows gameover image"""
        score = crashInfo["score"]
        playerx = self.screenwidth * 0.2
        playery = crashInfo["y"]
        playerHeight = self.images["player"][0].get_height()
        playerVelY = crashInfo["playerVelY"]
        playerAccY = 2
        playerRot = crashInfo["playerRot"]
        playerVelRot = 7

        basex = crashInfo["basex"]

        upperPipes = crashInfo["upperPipes"]
        lowerPipes = crashInfo["lowerPipes"]

        # play hit and die self.sounds
        self.sounds["hit"].play()
        if not crashInfo["groundCrash"]:
            self.sounds["die"].play()

        while True:
            for event in pygame.event.get():
                if event.type == QUIT or (
                    event.type == KEYDOWN and event.key == K_ESCAPE
                ):
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN and (
                    event.key == K_SPACE or event.key == K_UP
                ):
                    if playery + playerHeight >= self.basey - 1:
                        return

            # player y shift
            if playery + playerHeight < self.basey - 1:
                playery += min(playerVelY, self.basey - playery - playerHeight)

            # player velocity change
            if playerVelY < 15:
                playerVelY += playerAccY

            # rotate only when it's a pipe crash
            if not crashInfo["groundCrash"]:
                if playerRot > -90:
                    playerRot -= playerVelRot

            # draw sprites
            self.screen.blit(self.images["background"], (0, 0))

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                self.screen.blit(
                    self.images["pipe"][0], (uPipe["x"], uPipe["y"])
                )
                self.screen.blit(
                    self.images["pipe"][1], (lPipe["x"], lPipe["y"])
                )

            self.screen.blit(self.images["base"], (basex, self.basey))
            self.showScore(score)

            playerSurface = pygame.transform.rotate(
                self.images["player"][1], playerRot
            )
            self.screen.blit(playerSurface, (playerx, playery))
            self.screen.blit(self.images["gameover"], (50, 180))

            self.fpsclock.tick(self.fps)
            pygame.display.update()

    def playerShm(self, playerShm):
        """oscillates the value of playerShm['val'] between 8 and -8"""
        if abs(playerShm["val"]) == 8:
            playerShm["dir"] *= -1

        if playerShm["dir"] == 1:
            playerShm["val"] += 1
        else:
            playerShm["val"] -= 1

    def getRandomPipe(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        gapY = random.randrange(0, int(self.basey * 0.6 - self.pipegapsize))
        gapY += int(self.basey * 0.2)
        pipeHeight = self.images["pipe"][0].get_height()
        pipeX = self.screenwidth + 10

        return [
            {"x": pipeX, "y": gapY - pipeHeight},  # upper pipe
            {"x": pipeX, "y": gapY + self.pipegapsize},  # lower pipe
        ]

    def showScore(self, score):
        """displays score in center of screen"""
        scoreDigits = [int(x) for x in list(str(score))]
        totalWidth = 0  # total width of all numbers to be printed

        for digit in scoreDigits:
            totalWidth += self.images["numbers"][digit].get_width()

        Xoffset = (self.screenwidth - totalWidth) / 2

        for digit in scoreDigits:
            self.screen.blit(
                self.images["numbers"][digit],
                (Xoffset, self.screenheight * 0.1)
            )
            Xoffset += self.images["numbers"][digit].get_width()

    def checkCrash(self, player, upperPipes, lowerPipes):
        """returns True if player collides with base or pipes."""
        pi = player["index"]
        player["w"] = self.images["player"][0].get_width()
        player["h"] = self.images["player"][0].get_height()

        # if player crashes into ground
        if player["y"] + player["h"] >= self.basey - 1:
            return [True, True]
        elif player["y"] < 0:
            return [True, True]
        else:
            playerRect = pygame.Rect(
                player["x"], player["y"], player["w"], player["h"]
            )
            pipeW = self.images["pipe"][0].get_width()
            pipeH = self.images["pipe"][0].get_height()

            for i, (uPipe, lPipe) in enumerate(zip(upperPipes, lowerPipes)):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe["x"], uPipe["y"], pipeW, pipeH)
                lPipeRect = pygame.Rect(lPipe["x"], lPipe["y"], pipeW, pipeH)

                # player and upper/lower pipe self.hitmasks
                pHitMask = self.hitmasks["player"][pi]
                uHitmask = self.hitmasks["pipe"][0]
                lHitmask = self.hitmasks["pipe"][1]

                # if bird collided with upipe or lpipe
                uCollide = self.pixelCollision(
                    playerRect, uPipeRect, pHitMask, uHitmask
                )
                lCollide = self.pixelCollision(
                    playerRect, lPipeRect, pHitMask, lHitmask
                )

                if uCollide or lCollide:
                    return [True, False]

        return [False, False]

    def pixelCollision(self, rect1, rect2, hitmask1, hitmask2):
        """Checks if two objects collide and not just their rects"""
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in range(rect.width):
            for y in range(rect.height):
                if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                    return True
        return False

    def getHitmask(self, image):
        """returns a hitmask using an image's alpha."""
        mask = []
        for x in range(image.get_width()):
            mask.append([])
            for y in range(image.get_height()):
                mask[x].append(bool(image.get_at((x, y))[3]))
        return mask


if __name__ == "__main__":
    game = FlappyBird(288, 512, pipegapsize=100)
    game.main()
