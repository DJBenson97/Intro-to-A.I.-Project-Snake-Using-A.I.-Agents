import random
import matplotlib.pyplot as plt
import numpy as np
import time
from IPython.display import clear_output

# directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

DIRECTIONS = {
    "UP": UP,
    "DOWN": DOWN,
    "LEFT": LEFT,
    "RIGHT": RIGHT
}

WIDTH = 25
HEIGHT = 25


def spawn_food(snake):
    while True:
        food = (random.randint(0, WIDTH - 1), random.randint(0, HEIGHT - 1))
        if food not in snake:
            return food


def get_initial_state():
    snake = [(5, 5), (5, 4), (5, 3)]
    food = spawn_food(snake)

    return {
        "snake": snake,
        "food": food,
        "width": WIDTH,
        "height": HEIGHT
    }


def is_collision(point, state):
    x, y = point

    # wall collision
    if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
        return True

    # self collision
    if point in state["snake"]:
        return True

    return False


def get_valid_moves(state):
    valid = []
    head_x, head_y = state["snake"][0]

    for move_name, (dx, dy) in DIRECTIONS.items():
        new_pos = (head_x + dx, head_y + dy)
        if not is_collision(new_pos, state):
            valid.append(move_name)

    return valid


def update_state(state, move):
    dx, dy = DIRECTIONS[move]
    head_x, head_y = state["snake"][0]
    new_head = (head_x + dx, head_y + dy)

    # collision check
    if is_collision(new_head, state):
        return None, True, 0  # game over

    snake = state["snake"].copy()
    snake.insert(0, new_head)

    if new_head == state["food"]:
        food = spawn_food(snake)
        reward = 1
    else:
        snake.pop()
        food = state["food"]
        reward = 0

    new_state = {
        "snake": snake,
        "food": food,
        "width": WIDTH,
        "height": HEIGHT
    }

    return new_state, False, reward



# matplotlib rendering

def render(state):
    clear_output(wait=True)

    grid = np.zeros((HEIGHT, WIDTH))

    # snake
    for (x, y) in state["snake"]:
        grid[y][x] = 1

    # food
    fx, fy = state["food"]
    grid[fy][fx] = 2

    plt.imshow(grid)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def run_game(agent, render_game=False, max_steps=500):
    state = get_initial_state()
    score = 0

    steps = 0

    while True:
        valid_moves = get_valid_moves(state)

        if not valid_moves:
            break

        move = agent.get_action(state, valid_moves)

        state, done, reward = update_state(state, move)

        if done:
            break

        score += reward
        steps += 1

        if render_game:
            render(state)
            time.sleep(0.1)  # animation speed

        if steps > max_steps:
            break

    return score