# -*- coding: utf-8 -*-

import turtle


class Env:
    UP, LEFT, DOWN, RIGHT = 0, 1, 2, 3
    ToDigit = {'U': UP, 'L': LEFT, 'R': RIGHT, 'D': DOWN}
    ToChar = {val: key for (key, val) in ToDigit.items()}

    def __init__(self, n, G):
        self.G = G
        self.n = n
        self.n_obs = n*n*(G+1)
        self.n_act = 4
        self.t = None
        self.wn = None
        self.unit = 50
        self.xs = [0, -1, 0, 1]
        self.ys = [1, 0, -1, 0]
        self.x, self.y, self.g = 0, 0, 0

    def reset(self):
        self.x, self.y, self.g = 0, 0, 0
        return self.state()

    def state(self):
        # print(self.x, self.y, self.g)
        return self.x + (self.y + self.g*self.n)*self.n

    def move_player(self, x, y):
        self.t.up()
        self.t.setheading(90)
        self.t.fillcolor('red')
        self.t.goto((x + 0.5) * self.unit, (y + 0.5) * self.unit)

    def draw_box(self, x, y, fillcolor='', line_color='gray'):
        self.t.up()
        self.t.goto(x * self.unit, y * self.unit)
        self.t.color(line_color)
        self.t.fillcolor(fillcolor)
        self.t.setheading(90)
        self.t.down()
        self.t.begin_fill()
        for _ in range(4):
            self.t.forward(self.unit)
            self.t.right(90)
        self.t.end_fill()

    def render(self):
        if self.t is None:
            self.t = turtle.Turtle()
            self.wn = turtle.Screen()
            self.wn.setup(self.unit * self.n + 100,
                          self.unit * self.n + 100)
            self.wn.setworldcoordinates(0, 0, self.unit * self.n,
                                        self.unit * self.n)
            self.t.shape('circle')
            self.t.width(2)
            self.t.speed(0)
            self.t.color('gray')
            for i in range(self.n):
                for j in range(self.n):
                    if i == self.n-1 and j == 0:    # mine
                        self.draw_box(i, j, 'yellow')
                    elif i == 0 and j == self.n-1:  # home
                        self.draw_box(i, j, 'black')
                    else:
                        self.draw_box(i, j, 'white')
            self.t.shape('turtle')
        self.move_player(self.x, self.y)

    def step(self, action):
        reward = 0.0
        done = False
        if action == self.UP and self.y < self.n - 1:
            if self.x == 0 and self.y == self.n - 2:     # home
                reward = float(self.g)
                self.g = 0
            else:
                self.y += self.ys[action]
        elif action == self.DOWN and self.y > 0:
            if self.x == self.n-1 and self.y == 1:      # mine
                # print('before add {}'.format(self.g))
                if self.g < self.G:
                    self.g += 1
            else:
                self.y += self.ys[action]
        elif action == self.LEFT and self.x > 0:
            if self.y == self.n-1 and self.x == 1:      # home
                reward = float(self.g)
                self.g = 0
            else:
                self.x += self.xs[action]
        elif action == self.RIGHT and self.x < self.n - 1:
            if self.y == 0 and self.x == self.n - 2:     # mine
                # print('before add {}'.format(self.g))
                if self.g < self.G:
                    self.g += 1
            else:
                self.x += self.xs[action]

        return self.state(), reward, done

    def close(self):
        ...

