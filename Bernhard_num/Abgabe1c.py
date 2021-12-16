import numpy as np
import matplotlib.pyplot as plt

def ordinaryDifferentialEquation(u, t):
    return np.array([[98*u[0][0] + 1998*u[1][0]], [-99*u[0][0] - 1999*u[1][0]]])

xVal = 0
eVal = 0.1
yVal = [[1], [0]]
hVal = 1

x = []
y = [[], []]
h = []

TOL = 1.0e-2
err = []
errVal = 2 * TOL

def middle_point_rule(yVal, xVal, hVal, TOL):
    k1 = ordinaryDifferentialEquation(yVal, xVal)
    k2 = ordinaryDifferentialEquation(yVal + (0.5 * hVal * k1), xVal + 0.5)

    first_order = yVal + (hVal * k1)
    second_order = yVal + (hVal * k2)
    err_est = abs(np.linalg.norm(second_order)-np.linalg.norm(first_order))

    yVal = second_order
    return yVal, xVal, hVal, err_est


def adaptive_runge_kutta(yVal, xVal, hVal, TOL):
    err_est = 2 * TOL
    while (err_est > TOL):
        k1 = ordinaryDifferentialEquation(yVal, xVal)
        k2 = ordinaryDifferentialEquation(yVal + (1 / 5 * hVal * k1), xVal + 1 / 5 * hVal)
        k3 = ordinaryDifferentialEquation(yVal + (3 / 40 * hVal * k1 + 9 / 40 * hVal * k2), xVal + 3 / 10 * hVal)
        k4 = ordinaryDifferentialEquation(yVal + (3 / 10 * hVal * k1 - 9 / 10 * hVal * k2 + 6 / 5 * hVal * k3), xVal + 3 / 5 * hVal)
        k5 = ordinaryDifferentialEquation(yVal - (11 / 54 * hVal * k1 + 5 / 2 * hVal * k2 - 70 / 27 * hVal * k3 + 35 / 27 * hVal * k4), xVal + hVal)
        k6 = ordinaryDifferentialEquation(yVal + (1631 / 55296 * hVal * k1 + 175 / 512 * hVal * k2 + 575 / 13824 * hVal * k3 + 44275 / 110592 * hVal * k4 + 253 / 4096 * hVal * k5), xVal + 7 / 8 * hVal)

        fourth_order = yVal + hVal * (37 / 378 * k1 + 250 / 621 * k3 + 125 / 594 * k4 + 512 / 1771 * k6)
        fifth_order = yVal + hVal * (2825 / 27648 * k1 + 18575 / 48384 * k3 + 13525 / 55296 * k4 + 277 / 14336 * k5 + 1 / 4 * k6)
        err_est = abs(np.linalg.norm(fifth_order) - np.linalg.norm(fourth_order))
        if (err_est > TOL):
            hVal = 0.8 * hVal * ((TOL / err_est) ** (1 / 6))
        #elif(err_est < TOL/2):
         #  hVal = 1.6 * hVal * ((TOL / err_est) ** (1 / 6))

    yVal = fifth_order
    return yVal, xVal, hVal, err_est


while(xVal < eVal):
    y = np.hstack((y, yVal))
    x = np.hstack((x, xVal))
    h = np.hstack((h, hVal))
    err = np.hstack((err, errVal))
    #yVal, xVal, hVal, errVal = middle_point_rule(yVal, xVal, hVal, TOL)
    yVal, xVal, hVal, errVal = adaptive_runge_kutta(yVal,  xVal, hVal, TOL)
    xVal = xVal + hVal


plt.figure()
plt.plot(x, y[0, :], label='y(t)')


plt.legend()
plt.grid()
plt.title('')
plt.xlabel('t')
plt.ylabel('y(t)')

# use scientific notation
plt.ticklabel_format(style='sci', axis='y')

plt.show()

plt.figure()
plt.plot(x, h, label='Schrittweite')
plt.title('Schrittweite')
plt.grid()
plt.show()

plt.figure()
plt.plot(x, err, label='Fehler')
plt.title('Fehler')
plt.grid()
plt.show()