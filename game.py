import pygame  # for visuals
import neat  # NeuroEvolution of Augmenting Topologies
import os
import random  # for random placement of pipes

pygame.font.init()
# -----------------------------------------------------------------------------------------------#
# loading the images and setting the dimensions
GAME_WIDTH = 500
GAME_HEIGHT = 600

GEN = 0

PIPE_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "pipe.png")))
BG_IMAGE = pygame.image.load(os.path.join("images", "background.png"))
GROUND_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "ground.png")))
BIRD_IMAGE = pygame.image.load(os.path.join("images", "birdie.png"))

BIRD_X = 95
BIRD_Y = 90
BIRD_IMAGE = pygame.transform.scale(BIRD_IMAGE, (BIRD_X, BIRD_Y))
SCORE_FONT = pygame.font.SysFont("comicsans", 50)


# ----------------------------------------------------------------------------------------------#

# creating Classes
class Bird:
    IMG = BIRD_IMAGE
    ANIMATION_TIME = 5  # how fast our game is moving
    BIRD_SPEED = 20  # the speed at which the bird moves up/down

    # BIRD_ROTATION = 25  # the tilting/rotation of bird

    def __init__(self, x, y):  # Constructor
        self.x = x  # x, y are the coordinates of the bird
        self.y = y
        self.tickCount = 0
        self.speed = 0
        self.height = self.y
        self.image = self.IMG
        self.image_count = 0

    def move(self):
        self.tickCount += 1  # one frame passed
        displacement = 1.5 * self.tickCount ** 2 + self.speed * self.tickCount  # keep track of how much we move up/down

        if displacement < 0:
            displacement = displacement - 2.5  # if top is not reached keep moving upwards

        if displacement >= 15:
            displacement = 15

        self.y += displacement  # calculates current height by adding the displacement

    def jump(self):
        self.speed = -10.5  # for moving up we'll use negative velocity
        self.height = self.y
        self.tickCount = 0

    def draw(self, window):
        bird_display_img = self.IMG;
        new_rect = bird_display_img.get_rect(center=self.image.get_rect(topleft=(self.x, self.y)).center)
        window.blit(bird_display_img, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.IMG)  # create this mask for picture perfect collision


######################################################################################################

class Pipe:
    PIPE_GAP = 185
    PIPE_SPEED = 5

    def __init__(self, x):
        self.x = x  # only set x as height is randomly set
        self.height = 0

        self.top = 0
        self.bottom = 0
        self.PIPE_DOWN_IMG = PIPE_IMAGE  # two types of pipe placement;up and down
        self.PIPE_UP_IMG = pygame.transform.flip(PIPE_IMAGE, False, True)  # top
        self.passed = False  # if bird has passed pipe or not

        self.set_random_height()

    def set_random_height(self):
        self.height = random.randrange(50, 250)  # randomizing height
        self.top = self.height - self.PIPE_DOWN_IMG.get_height()
        self.bottom = self.height + self.PIPE_GAP

    def draw(self, window):
        window.blit(self.PIPE_UP_IMG, (self.x, self.top))
        window.blit(self.PIPE_DOWN_IMG, (self.x, self.bottom))

    def move(self):
        self.x -= self.PIPE_SPEED  # moving pipe to left

    def check_collision(self, bird, window):
        bird_mask = bird.get_mask()  # getting bird mask from its class

        # calculating how far the masks are from each other
        d_downpipe_bird = (self.x - bird.x, self.bottom - round(bird.y))
        s_uppipe_bird = (self.x - bird.x, self.top - round(bird.y))

        # finding if bird and either of the pipes collide
        # if the don't collide False is stored
        up_collision = bird_mask.overlap(pygame.mask.from_surface(self.PIPE_UP_IMG), s_uppipe_bird)
        down_collision = bird_mask.overlap(pygame.mask.from_surface(self.PIPE_DOWN_IMG), d_downpipe_bird)

        # if they collide return true
        if up_collision or down_collision:
            return True
        return False


######################################################################################################

class Ground:
    IMAGE = GROUND_IMAGE
    GROUND_DIM = GROUND_IMAGE.get_width()
    GROUND_SPEED = 5  # same as pipe speed

    def __init__(self, y):
        self.y = y  # x is moving to left so we don't pass it
        self.x1 = 0  # points at beginning of the ground
        self.x2 = self.GROUND_DIM  # points at end of the ground

    def draw(self, window):
        window.blit(self.IMAGE, (self.x1, self.y))  # drawing both two grounds
        window.blit(self.IMAGE, (self.x2, self.y))

    def move(self):
        self.x1 = self.x1 - self.GROUND_SPEED  # moving the x points with the speed
        self.x2 = self.x2 - self.GROUND_SPEED

        first_img = self.x1 + self.GROUND_DIM
        second_img = self.x2 + self.GROUND_DIM

        if first_img < 0:  # if the first image has completely moved to the left
            self.x1 = self.x2 + self.GROUND_DIM  # then initialize pointer to second image
        if second_img < 0:  # if the second image has completely moved to the left
            self.x2 = self.x1 + self.GROUND_DIM  # then initialize pointer to first image


######################################################################################################

def draw_window(window, birds, ground, pipes, score, gen):
    window.blit(BG_IMAGE, (0, 0))
    for pipe in pipes:
        pipe.draw(window)

    text = SCORE_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    window.blit(text, (10, 10))

    text = SCORE_FONT.render("Gen: " + str(gen), 1, (255, 255, 255))
    window.blit(text, (GAME_WIDTH - 10 - text.get_width(), 10))

    ground.draw(window)
    for bird in birds:
        bird.draw(window)
    pygame.display.update()


def fitnessFunc(genomes, config):
    global GEN
    GEN += 1
    birds = []
    nets = []
    ge = []

    for _, g in genomes:
        g.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        ge.append(g)

    ground = Ground(500)
    pipes = [Pipe(600)]
    window = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
    clock = pygame.time.Clock()

    score = 0

    is_running = True
    while is_running:
        new_pipe = False
        clock.tick(50)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
                pygame.quit()
                quit()

        pipe_buffer = []  # store the pipes that have passed the screen here
        pipe_indicator = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_UP_IMG.get_width():
                pipe_indicator = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1
            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_indicator].height),
                                       abs(bird.y - pipes[pipe_indicator].bottom)))
            if output[0] > 0.5:
                bird.jump()
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.check_collision(bird, window):
                    ge[x].fitness -= 1  # Every time a bird collides, decrement its fitness score, the punishment!
                    birds.pop(x)
                    nets.pop(x)  # basically deleting that bird because it died.
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:  # if bird is passing the pipe generate new pipe
                    pipe.passed = True
                    new_pipe = True

            if pipe.x + pipe.PIPE_UP_IMG.get_width() < 0:
                pipe_buffer.append(pipe)

            pipe.move()

        if new_pipe:
            score += 1
            for g in ge:
                g.fitness += 5  # If a bird manages to surpass a pipe, increase its fitness score - The Reward!

            pipes.append(Pipe(600))

        for prev_pipe in pipe_buffer:
            pipes.remove(prev_pipe)

        for x, bird in enumerate(birds):
            if bird.y + bird.image.get_height() - 10 >= 500 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        ground.move()

        draw_window(window, birds, ground, pipes, score, GEN)


def run(neat_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, neat_path)
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    winner = population.run(fitnessFunc, 50)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)  # returns the path to the current working directory
    neat_path = os.path.join(local_dir, "neat.txt")
    run(neat_path)