import numpy as np
import matplotlib.pyplot as plt

m = 2500
K = 1000
d = 0.01
a = 0.2
l = 6
W = 100
w = 2 * np.pi * (38/60)

def ordinaryDifferentialEquation(u, t):
    aVar = (np.exp(a * (u[0][0] - l * np.sin(u[2][0]))) - 1 + np.exp(a * (u[0][0] + l * np.sin(u[2][0]))) - 1)
    bVar = (np.exp(a * (u[0][0] - l * np.sin(u[2][0]))) - np.exp(a * (u[0][0] + l * np.sin(u[2][0]))))
    return np.array([[u[1][0]], [-d * u[1][0] - (K / (m * a)) * aVar + 0.2 * W * np.sin(w * t)], [u[3][0]], [-d * u[3][0] + (K / (m * a)) * ((3 * np.cos(u[2][0])) / l) * bVar]])


xVal = 0
eVal = 20
yVal = [[0], [0], [0.001], [0]]
hVal = 0.1

x = []
y = [[], [], [], []]
h = []


TOL = 1.0e-3
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


############################# Main #######################################################
while(xVal < eVal):
    y = np.hstack((y, yVal))
    x = np.hstack((x, xVal))
    h = np.hstack((h, hVal))
    err = np.hstack((err, errVal))
    yVal, xVal, hVal, errVal = middle_point_rule(yVal, xVal, hVal, TOL)
    xVal = xVal + hVal
##########################################################################################

oVal = max(y[2, :])/y[2, 0] # Verstaerkungsfaktor

# use scientific notation
plt.ticklabel_format(style='sci', axis='y')

plt.figure()
plt.plot(x, y[0, :], label='y(x)')
plt.legend()
plt.grid()

plt.title('Vertikale Auslenkung')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.show()

plt.figure()
plt.plot(x, y[2, :], label='O(x)')


plt.legend()
plt.grid()

plt.title('Verdrehungswinkel')
plt.xlabel('x')
plt.ylabel('O(x)')
plt.show()

plt.figure()
plt.plot(x, h, label='Schrittweite')
plt.title('Schrittweite')
plt.grid()
plt.show()

plt.figure()
plt.plot(x,err, label='Fehler')
plt.title('Fehler')
plt.grid()
plt.show()


