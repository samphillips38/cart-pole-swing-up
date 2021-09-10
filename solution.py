# This is the solution to the cart-pole swing-up problem

import numpy as np
from scipy import optimize

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
N = 1000


# Dynamics
def f(x, u):
    """Calculate the dynamics given the state, x and the control, u"""
    [_, q2, q1_dot, q2_dot] = x # [horzontal_position, angle, ...]

    q1_ddot = (l*m1*np.sin(q2)*q2_dot**2 + u + m2*g*np.cos(q2)*np.sin(q2)) / (m1 + m2*(1 - np.cos(q2)**2))
    q2_ddot = (l*m2*np.cos(q2)*np.sin(q2)*q2_dot**2 + u*np.cos(q2) + (m1 + m2)*g*np.sin(q2)) / (l*m1 + l*m2*(1 - np.cos(q2)**2))

    return np.array([q1_dot, q2_dot, q1_ddot, q2_ddot])


def trapezoid_integral(time_array, func):
    """Using given discrete time array and function, approximate the integral with the trapezoid quadrature"""
    s = 0
    for k in range(len(time_array)-1):
        h_k = time_array[k+1] - time_array[k]
        w_k = func(time_array[k])
        w_kplus1 = func(time_array[k+1])

        s += h_k*(w_k + w_kplus1) / 2
    return s

def objective_func(data):
    """Calculate the value of the objective function with given data"""
    u = data[0]
    h_k = T/N
    s = 0
    for k in range(len(u) - 1):
        s += u[k]**2 + u[k+1]**2
    return 0.5*h_k*s

def collocation_constraint(data):
    """Constructs a matrix which should be constrained to be equal to 0. This is the dynamics constraint"""
    u = data[0]
    x = data[1:]
    h_k = T/N
    output = np.zeros((5, N))
    for k in range(len(data) - 1):
        x_k = x[:, k] # Structured like [q1, q2, qdot1, qdot2]
        x_kplus1 = x[:, k+1]
        u_k = u[k]
        u_kplus1 = u[k+1]

        # Insert Column into output matrix
        output[1:, k] = h_k * (f(u_kplus1, x_kplus1) + f(u_k, x_k)) - x_kplus1 + x_k 
    
    return output

def path_limit(sign=1):
    """Construct the upper limit to the path constraint.
    If sign = -1 output lower limit.
    """
    output = np.full((5, N), np.inf)
    output[0] = u_max
    output[1] = d_max
    return output*sign

def boundary_start(data):
    """Function to constrain start at origin"""
    x_0 = data[1:, 0]
    return x_0

def boundary_end(data):
    """Function to constrain end point"""
    x_N = data[1:, N]
    end = np.array([d, np.pi, 0, 0]) # End vertically with 0 velocity
    return x_N - end

if __name__=='__main__':
    pass