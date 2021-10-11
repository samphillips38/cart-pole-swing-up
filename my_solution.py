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
u_max = 60
T = 3.5
N = 100

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
    s = 0
    for k in range(N-1):
        s += u[k]**2 + u[k+1]**2
    return 0.5*h_k*s

def collocation_constraint(data):
    """Constructs a matrix which should be constrained to be equal to 0. This is the dynamics constraint"""
    u, x = split_data(data)
    h_k = T/N

    x_k = np.delete(x, N-1, 1) # Size is (4, N-1)
    u_k = np.delete(u, N-1)

    x_kplus1 = np.delete(x, 0, 1)
    u_kplus1 = np.delete(u, 0)

    output = 0.5 * h_k * (f(u_kplus1, x_kplus1) + f(u_k, x_k)) - x_kplus1 + x_k
    return output.flatten() 

def path_limit(sign=1):
    """Construct the upper limit to the path constraint.
    If sign = -1 output lower limit.
    """
    output = np.full((5, N), np.inf)
    output[0] = u_max
    output[1] = d_max
    return output.flatten()*sign

def boundary_start(data):
    """Function to constrain start at origin"""
    _, x = split_data(data)
    x_0 = x[:, 0]
    return x_0

def boundary_end(data):
    """Function to constrain end point"""
    _, x = split_data(data)
    x_N = x[:, N-1]
    end = np.array([d, np.pi, 0, 0]) # End vertically with 0 velocity
    return x_N - end

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
    dynamic_con = optimize.NonlinearConstraint(collocation_constraint, np.zeros(4*(N-1)), np.zeros(4*(N-1)))
    boundary_start_con = optimize.NonlinearConstraint(boundary_start, np.zeros(4), np.zeros(4))
    boundary_end_con = optimize.NonlinearConstraint(boundary_end, np.zeros(4), np.zeros(4))

    # Initial guess
    start = initial_guess()

    # Optimise
    res = optimize.minimize(objective_func, start, method='SLSQP', constraints=[dynamic_con, boundary_start_con, boundary_end_con], bounds=bounds)
    u, x = split_data(res.x)
    [q1, q2, q1_dot, q2_dot] = x

    print("Solution Optimised. Final Objective evaluation: ", objective_func(res.x))
    
    return (u, q1, q2)

if __name__=='__main__':
    u, q1, q2 = solve()
    plot(u, q1, q2, T, N, l)