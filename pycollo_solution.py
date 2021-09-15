import pycollo as pc
import sympy as sym
import numpy as np
from plot import plot

# System global paramters
m1 = 1
m2 = 3
l = 0.5
g = 9.81

# Problem global parameters
d = 1.5
d_max = 2
u_max = 100
T = 3

# Sympy Symbols for input equations
u = sym.symbols("u")
q1, q2, q1_dot, q2_dot = sym.symbols("q1, q2, q1_dot, q2_dot")
q1_ddot, q2_ddot = sym.symbols("q1_ddot, q2_ddot")


def setup_problem() -> pc.OptimalControlProblem:
    """Construct pycollo optimal control problem"""
    prob = pc.OptimalControlProblem("Cart-Pole Swing-Up")
    phase = prob.new_phase("Phase_1")

    # Add variables and equations
    phase.state_variables = [q1, q2, q1_dot, q2_dot]
    phase.state_equations = [q1_dot, q2_dot, q1_ddot, q2_ddot]
    phase.control_variables = u
    phase.integrand_functions = [u**2]

    # Setup Constraints
    phase.bounds.initial_time = 0
    phase.bounds.final_time = T
    phase.bounds.control_variables = {u: [-u_max, u_max]}
    phase.bounds.integral_variables = [[0, 100]]
    phase.bounds.state_variables = {
        q1: [-d_max, d_max],
        q2: [-10, 10],
        q1_dot: [-10, 10],
        q2_dot: [-10, 10]
    }
    phase.bounds.initial_state_constraints = {
        q1: 0,
        q2: 0,
        q1_dot: 0,
        q2_dot: 0
    }
    phase.bounds.final_state_constraints = {
        q1: d,
        q2: np.pi,
        q1_dot: 0,
        q2_dot: 0
    }

    # Guess
    phase.guess.time = [0, T]
    phase.guess.state_variables = [[0, d], [0, np.pi], [0, 0], [0, 0]]
    phase.guess.control_variables = [[0, 0]]
    phase.guess.integral_variables = [0]
    
    # Dynamics equations and objective function
    q1dd_eqn = (l * m2 * sym.sin(q2) * q2_dot**2 + u + m2 * g * sym.cos(q2) * sym.sin(q2)) / (m1 + m2 * (1 - sym.cos(q2)**2))
    q2dd_eqn = - (l * m2 * sym.cos(q2) * sym.sin(q2) * q2_dot**2 + u * sym.cos(q2) + (m1 + m2) * g * sym.sin(q2)) / (l * m1 + l * m2 * (1 - sym.cos(q2)**2))

    prob.objective_function = phase.integral_variables[0]
    prob.auxiliary_data = {
        q1_ddot: q1dd_eqn,
        q2_ddot: q2dd_eqn,
    }

    return prob

def solve():
    """Solve Optimal Control Problem"""
    prob = setup_problem()

    prob.initialise()
    prob.solve()

    t = 0.5 * prob.solution.tau[0] + 0.5
    q1 = prob.solution.state[0][0]
    q2 = prob.solution.state[0][1]
    u = prob.solution.control[0][0]

    return (u, q1, q2)

if __name__=='__main__':
        u, q1, q2 = solve()
        N = len(u)
        plot(u, q1, q2, T, N, l)