# This is the solution to the cart-pole swing-up problem

import numpy as np

# System global paramters
m1 = 10
m2 = 10
l = 10
g = 9.81

# Problem global parameters
d = 1
d_max = 3
u_max = 10
T = 10


# Dynamics
def x_dot(x, u):
    """Calculate the dynamics given the state, x and the control, u"""
    [_, q2, q1_dot, q2_dot] = x # [horzontal_position, angle, ...]

    q1_ddot = (l*m1*np.sin(q2)*q2_dot**2 + u + m2*g*np.cos(q2)*np.sin(q2)) / (m1 + m2*(1 - np.cos(q2)**2))
    q2_ddot = (l*m2*np.cos(q2)*np.sin(q2)*q2_dot**2 + u*np.cos(q2) + (m1 + m2)*g*np.sin(q2)) / (l*m1 + l*m2*(1 - np.cos(q2)**2))

    return np.array([q1_dot, q2_dot, q1_ddot, q2_ddot])


def trapezoid_quadrature(time_array, func):
    """Using given discrete time array and function, approximate the integral with the trapezoid quadrature"""
    s = 0
    for k in range(len(time_array)-1):
        h_k = time_array[k+1] - time_array[k]
        w_k = func(time_array[k])
        w_kplus1 = func(time_array[k+1])

        s += h_k*(w_k + w_kplus1) / 2
    return s