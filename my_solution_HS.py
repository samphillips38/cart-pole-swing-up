# This is the solution to the cart-pole swing-up problem
import numpy as np
from scipy import optimize
from plot import plot

# System global paramters
m1 = 1
m2 = 3
l = 1
g = 9.81

# Problem global parameters
d = 1.5
d_max = 2
u_max = 600
T = 2.5
N = 51

def f(u, x):
    """Calculate the dynamics given the state, x and the control, u"""
    [_, q2, q1_dot, q2_dot] = x # [horzontal_position, angle, ...]

    q1_ddot = (l*m2*np.sin(q2)*q2_dot**2 + u + m2*g*np.cos(q2)*np.sin(q2)) / (m1 + m2*(1 - np.cos(q2)**2))
    q2_ddot = - (l*m2*np.cos(q2)*np.sin(q2)*q2_dot**2 + u*np.cos(q2) + (m1 + m2)*g*np.sin(q2)) / (l*m1 + l*m2*(1 - np.cos(q2)**2))

    return np.array([q1_dot, q2_dot, q1_ddot, q2_ddot])

def split_data(data) -> np.array:
    """Split supplied 1-dimensional data into components"""
    u = data[:N]

    q1 = data[N:2*N]
    q2 = data[2*N:3*N]
    q1_dot = data[3*N:4*N]
    q2_dot = data[4*N:5*N]

    return (u, np.array([q1, q2, q1_dot, q2_dot]))

def objective_func(data):
    """Calculate the value of the objective function with given data"""
    u, _ = split_data(data)
    h_k = T/N

    u_k = u[0:-2:2]
    u_kplushalf = u[1::2]
    u_kplus1 = u[2::2]

    return h_k*(sum(u_k**2) + 4*sum(u_kplushalf**2) + sum(u_kplus1**2))/6 # Using Simpson quadrature

def constraint(data):
    """Constraints Vector"""
    u, x = split_data(data)
    h_k = T/N

    # Dynamic constraints
    x_k = x[:, 0:-2:2] # Every other element
    x_kplushalf = x[:, 1::2] 
    x_kplus1 = x[:, 2::2]

    u_k = u[0:-2:2]
    u_kplushalf = u[1::2]
    u_kplus1 = u[2::2]

    hermite_interpolant = 0.5*(x_k + x_kplus1) + 0.125*h_k*(f(u_k, x_k) - f(u_kplus1, x_kplus1)) - x_kplushalf
    simpson_collocation = x_k - x_kplus1 + h_k*(f(u_k, x_k) + 4*f(u_kplushalf, x_kplushalf) + f(u_kplus1, x_kplus1))/6

    dynamic_constraints = np.concatenate((hermite_interpolant, simpson_collocation)).flatten()

    # Boundary constraints
    boundary_constraint = np.concatenate((x[:, 0], x[:, -1] - np.array([d, np.pi, 0, 0])))

    return np.concatenate((dynamic_constraints, boundary_constraint))

def path_limit(sign=1):
    """Construct the upper limit to the path constraint.
    If sign = -1 output lower limit.
    """
    output = np.full((5, N), np.inf)
    output[0] = u_max
    output[1] = d_max
    return output.flatten()*sign

def initial_guess() -> np.ndarray:
    """Linearly interpolate between start and end states for initial guess"""
    output = np.zeros((5, N))
    q1 = np.linspace(0, d, N)
    q2 = np.linspace(0, np.pi, N)
    output[1] = q1
    output[2] = q2
    return output.flatten()

def solve():
    """Solve the trajectory optimisation problem using scipy."""
    # Define bounds
    bounds = optimize.Bounds(path_limit(sign=-1), path_limit()) 

    # Define Constraints
    con = optimize.NonlinearConstraint(constraint, np.zeros(4*(N-1) + 8), np.zeros(4*(N-1) + 8))

    # Initial guess
    start = initial_guess()

    # Optimise
    res = optimize.minimize(objective_func, start, method='SLSQP', constraints=[con], bounds=bounds)
    u, x = split_data(res.x)
    [q1, q2, q1_dot, q2_dot] = x

    print("Solution Optimised. Final Objective evaluation: ", objective_func(res.x))

    return (u, q1, q2)

if __name__=='__main__':
    u, q1, q2 = solve()
    plot(u, q1, q2, T, N, l)