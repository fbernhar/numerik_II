import numpy as np
import matplotlib.pyplot as plt

funcOne = lambda u, t: 5*u
funcTwo = lambda u, t: 10 * u - 50 * (u ** 2)

def newton(f, x, max_it):
    eps = 1.0e-10
    for i in range(max_it):
        df = (f(x + eps) - f(x)) / eps
        x = x - f(x)/df
    return x


def multi_step_fctOne(f,xVal, yVal, hVal, max_it):
    x = [xVal]
    y = [yVal]
    xVal = xVal + hVal
    yVal = yVal + hVal * f(yVal, xVal)

    for i in range(max_it):
        x = np.append(x, xVal)
        y = np.append(y, yVal)
        xVal = xVal + hVal
        yVal = y[-2] + 2 * hVal * f(y[-1], xVal)


    return y, x

def multi_step_fctTwo(f, xVal, yVal, hVal, max_it):

    x = [xVal]
    y = [yVal]
    xVal = xVal + hVal
    yVal = yVal + hVal * f(yVal, xVal)



    for i in range(max_it):
        x = np.append(x, xVal)
        y = np.append(y, yVal)
        xVal = xVal + hVal
        func = lambda u: 4/3 * y[-1] - 1/3 * y[-2] - u + ((2*hVal)/3) * f(u, xVal)
        yVal = newton(func, y[-1], 100)


    return y, x


def multi_step_fctThree(f, xVal, yVal, hVal, max_it):

    x = [xVal]
    y = [yVal]
    xVal = xVal + hVal
    yVal = yVal + hVal * f(yVal, xVal)



    for i in range(max_it):
        x = np.append(x, xVal)
        y = np.append(y, yVal)
        xVal = xVal + hVal
        func = lambda u: 3 * y[-1] - 2 * y[-2] - u + (hVal/12) * (7*f(u, xVal) - 8*f(y[-1], xVal) - 11*f(y[-2], xVal))
        yVal = newton(func, y[-1], 100)


    return y, x


h = 0.001
max_it = 10

plt.figure()
plt.title("First DifferentialEquation;  h = " + str(h) + "; u(0)=0.5")
y, x = multi_step_fctOne(funcOne, 0, 0.5, h, max_it)
plt.plot(x, y, '-+', label='first Multistep Method')
y, x = multi_step_fctTwo(funcOne, 0, 0.5, h, max_it)
plt.plot(x, y, '-o', label='second Multistep Method')
y, x = multi_step_fctThree(funcOne, 0, 0.5, h, max_it)
plt.plot(x, y, label='third Multistep Method')
plt.legend()
plt.grid()
plt.show()



plt.figure()
plt.title("Second DifferentialEquation;  h = " + str(h) + "; u(0)=0.5")
y, x = multi_step_fctOne(funcTwo, 0, 0.5, h, max_it)
plt.plot(x, y, '-+', label='first Multistep Method')
y, x = multi_step_fctTwo(funcTwo, 0, 0.5, h, max_it)
plt.plot(x, y, '-o', label='second Multistep Method')
y, x = multi_step_fctThree(funcTwo, 0, 0.5, h, max_it)
plt.plot(x, y, label='third Multistep Method')
plt.legend()
plt.grid()
plt.show()



plt.figure()
plt.title("Second DifferentialEquation;  h = " + str(h) + "; u(0)=2")
y, x = multi_step_fctOne(funcTwo, 0, 2, h, max_it)
plt.plot(x, y, '-+', label='first Multistep Method')
y, x = multi_step_fctTwo(funcTwo, 0, 2, h, max_it)
plt.plot(x, y, '-o', label='second Multistep Method')
y, x = multi_step_fctThree(funcTwo, 0, 2, h, max_it)
plt.plot(x, y, label='third Multistep Method')
plt.legend()
plt.grid()
plt.show()
