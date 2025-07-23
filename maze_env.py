
import numpy as np

class MazeEnv:
    def __init__(self, size=5):
        self.size = size
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.state = self.start
        self.obstacles = {(1, 1), (2, 2), (3, 1)}

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0: y -= 1
        elif action == 1: y += 1
        elif action == 2: x -= 1
        elif action == 3: x += 1

        next_state = (max(0, min(x, self.size-1)), max(0, min(y, self.size-1)))

        if next_state in self.obstacles:
            next_state = self.state
            reward = -1
        elif next_state == self.goal:
            reward = 10
        else:
            reward = -0.1

        self.state = next_state
        done = (self.state == self.goal)
        return next_state, reward, done

    def get_state_space(self):
        return [(i, j) for i in range(self.size) for j in range(self.size) if (i, j) not in self.obstacles]

    def get_action_space(self):
        return [0, 1, 2, 3]
