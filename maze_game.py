
import pygame
import time
from maze_env import MazeEnv
from q_learning_agent import QLearningAgent

CELL_SIZE = 60
GRID_SIZE = 5
WIDTH = HEIGHT = CELL_SIZE * GRID_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 102, 204)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
YELLOW = (255, 255, 0)

manual_mode = False
env = MazeEnv(GRID_SIZE)
agent = QLearningAgent(env.get_state_space(), env.get_action_space())

for ep in range(300):
    state = env.reset()
    for _ in range(100):
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        if done:
            break

def draw_grid(screen, path, current, env):
    screen.fill(WHITE)
    for row in range(env.size):
        for col in range(env.size):
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if (row, col) in env.obstacles:
                pygame.draw.rect(screen, BLACK, rect)
            elif (row, col) == env.goal:
                pygame.draw.rect(screen, GREEN, rect)
            elif (row, col) in path:
                pygame.draw.rect(screen, BLUE, rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)
    if current:
        pygame.draw.circle(screen, RED, (current[1]*CELL_SIZE + CELL_SIZE//2, current[0]*CELL_SIZE + CELL_SIZE//2), CELL_SIZE//4)

def play_maze_game():
    global manual_mode
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Q-Learning Maze Game")
    clock = pygame.time.Clock()

    path = []
    state = env.reset()
    path.append(state)

    running = True
    done = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    manual_mode = not manual_mode
                    print("Manual Mode:" if manual_mode else "Auto Mode")
                if event.key == pygame.K_r:
                    state = env.reset()
                    path = [state]
                    done = False

        if not done:
            if manual_mode:
                keys = pygame.key.get_pressed()
                action = None
                if keys[pygame.K_LEFT]: action = 0
                elif keys[pygame.K_RIGHT]: action = 1
                elif keys[pygame.K_UP]: action = 2
                elif keys[pygame.K_DOWN]: action = 3
                if action is not None:
                    next_state, reward, done = env.step(action)
                    if next_state != state:
                        path.append(next_state)
                    state = next_state
                    time.sleep(0.1)
            else:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                if next_state != state:
                    path.append(next_state)
                state = next_state
                time.sleep(0.3)

        draw_grid(screen, path, state, env)
        pygame.display.flip()
        clock.tick(10)

    pygame.quit()

if __name__ == "__main__":
    play_maze_game()
