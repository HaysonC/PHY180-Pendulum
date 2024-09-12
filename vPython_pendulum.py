import math
import numpy as np
import matplotlib.pyplot as plt

from vpython import *

def config():
    # Flip the x-axis so that the origin is in the bottom left
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.axis('off')

    # flip in the vpthon
    scene.forward = vector(0, -1, 0)
    scene.up = vector(0, 0, 1)
    scene.width = 800
    scene.height = 800



# sign convention: positive angle is counterclockwise, x is to the right, y is up
"""
========rooftop=========
           \
            \ 
             \
              O
              
-pi/2 -pi/4 0 pi/4 pi/2

"""

class Pendulum:
    def __init__(self, length=1, mass=1, angle=math.pi/4, initial_velocity=0, g=9.8, phi=0):
        self.length = length
        self.mass = mass
        self.angle = angle
        self.velocity = initial_velocity
        self.g = g
        self.phi = phi

    def get_acceleration(self):
        return -self.g/self.length * math.sin(self.angle)

    def update(self, dt):
        self.angle += self.velocity * dt
        self.velocity += self.get_acceleration() * dt

    def get_position(self):
        return self.length * vector(math.cos(self.angle), math.sin(self.angle), 0)


class Simulation:
    def __init__(self, pendulum, dt=0.01):
        self.pendulum = pendulum
        self.dt = dt
        self.positions = []

    def run(self, num_steps):
        for i in range(num_steps):
            self.pendulum.update(self.dt)
            self.positions.append(self.pendulum.get_position())

    def plot(self):
        x = [p.x for p in self.positions]
        y = [p.y for p in self.positions]
        plt.plot(x, y)
        plt.show()


class Visualizer:
    def __init__(self, pendulum, num_steps=1000):
        self.pendulum = pendulum
        self.scene = canvas()
        self.scene.range = 2
        self.pendulum_visual = cylinder(pos=vector(0, 0, 0), axis=pendulum.get_position(), radius=0.1)
        self.simulation = Simulation(pendulum)
        self.num_steps = num_steps

    def run(self):
        for i in range(self.num_steps):
            rate(30)
            self.pendulum.update(0.01)
            self.pendulum_visual.axis = self.pendulum.get_position()
            self.simulation.run(1)

        self.simulation.plot()

if __name__ == "__main__":
    pendulum = Pendulum()
    visualizer = Visualizer(pendulum)
    visualizer.run()





