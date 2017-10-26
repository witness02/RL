# 一个简单的environment，从左上角出发，右下角为终点
class SimpleEnv:
    actions = ['up', 'down', 'left', 'right']

    def __init__(self, width=6, height=6, reward=-1):
        self.width = width
        self.height = height
        self.reward = reward
        self.cur_position = [0, 0]

    def size(self):
        return self.height, self.width

    def actions_space(self):
        return self.actions

    def total_states(self):
        return self.width * self.height

    def reset(self):
        self.cur_position = [0, 0]

    def render(self):
        print(self.cur_position)

    def legal_position(self, pos):
        return (pos[0] >= 0) and (pos[0] < self.height) and (pos[1] >= 0) and (pos[1] < self.width)

    def state_to_position(self, state):
        return [int(state / self.width), int(state % self.width)]

    def position_to_state(self, position):
        return position[0] * self.width + position[1]

    def is_terminal_state(self, state):
        return state == (self.height * self.width - 1)

    def is_start_state(self, state):
        return state == 0

    def step(self, action):
        cur_state = self.position_to_state(self.cur_position)
        self.step_with_state(action, cur_state)

    def step_with_state(self, action, cur_state):
        self.cur_position = self.state_to_position(cur_state)
        if self.is_terminal_state(cur_state):
            return cur_state, 0
        reward = self.reward
        if action == self.actions[0]:
            next_position = [self.cur_position[0] - 1, self.cur_position[1]]
        elif action == self.actions[1]:
            next_position = [self.cur_position[0] + 1, self.cur_position[1]]
        elif action == self.actions[2]:
            next_position = [self.cur_position[0], self.cur_position[1] - 1]
        else:
            next_position = [self.cur_position[0], self.cur_position[1] + 1]
        if not self.legal_position(next_position):
            next_position = self.cur_position
        self.cur_position = next_position

        if self.is_start_state(cur_state):
            reward = 0
        return self.position_to_state(self.cur_position), reward
