import os
import sys
import random

r = random.Random()

d = 30

for k in range(100):
    mu = random.randint(-100, 100)
    sigma = random.randint(1, 10)
    for n in range(1000 * 1000):
        print(";".join("{:.5f}".format(r.gauss(mu, sigma)) for _ in range(d)))
