# This is the solution to the cart-pole swing-up problem
import numpy as np
from scipy import sparse
from cyipopt import minimize_ipopt
import cyipopt
from scipy.sparse import dia
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
T = 2.5
N = 60

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

class cart_prob:

    def objective(self, data):
        """Calculate the value of the objective function with given data"""
        u, _ = split_data(data)
        h_k = T/N
        return h_k*(sum(u**2) - 0.5*(u[0]**2 + u[-1]**2))  # This can be done due to constant time step

    def gradient(self, data):
        """Calculate grad vector of objective function."""
        output = data * 2*T / N
        output[N:] = 0
        output[0] /= 2
        output[N-1] /= 2
        return output

    def constraints(self, data):
        """Construct constraint vector"""
        u, x = split_data(data)
        h_k = T/N

        # Dynamic Constraints
        x_k = np.delete(x, N-1, 1) # Size is (4, N-1)
        u_k = np.delete(u, N-1)

        x_kplus1 = np.delete(x, 0, 1)
        u_kplus1 = np.delete(u, 0)

        dynamic_constraint = (0.5 * h_k * (f(u_kplus1, x_kplus1) + f(u_k, x_k)) - x_kplus1 + x_k).flatten()

        # Start and End Boundary Constraints
        boundary_constraint = np.concatenate((x[:, 0], x[:, N-1] - np.array([d, np.pi, 0, 0])))

        return np.concatenate((dynamic_constraint, boundary_constraint))

    def jacobian(self, data):
        """The Jacobian of the Constraints"""
        u, [q1 ,q2 ,q1_dot, q2_dot] = split_data(data)
        output = np.zeros((4*(N-1)+8, 5*N))
        hk = T/N

        for i in range(4*(N - 1)): # For each row in Dynamics section

            # Determine which section of the Jacobian we are in
            if i < N-1: # Constraint Equation: 0.5*hk*(q1_dot[k] + q1_dot[k+1]) + q1[k] - q1[k+1]
                k = i

                # Diff wrt u is zero
                # Diff wrt q1
                output[i, k+N] = 1
                output[i, k+N+1] = -1
                # Diff wrt q2 is zero
                # Diff wrt q1_dot
                output[i, k+3*N:k+3*N+2] = hk/2
                # Diff wrt q2_dot is zero

            elif i < 2*(N - 1): # Constraint Equation: 0.5*hk*(q2_dot[k] + q2_dot[k+1]) + q2[k] - q2[k+1]
                k = i - (N - 1)

                # Diff wrt u is zero
                # Diff wrt q1 is zero
                # Diff wrt q2
                output[i, k+2*N] = 1
                output[i, k+2*N+1] = -1
                # Diff wrt q1_dot is zero
                # Diff wrt q2_dot
                output[i, k+4*N:k+4*N+2] = hk/2

            elif i < 3*(N - 1): # Constraint Equation: 0.5*hk*(q1_ddot[k] + q1_ddot[k+1]) + q1_dot[k] - q1_dot[k+1]
                k = i - 2*(N - 1)

                # Diff wrt u
                output[i, k] = 0.5 * hk / (m1 + m2*(1 - np.cos(q2[k])**2))
                output[i, k+1] = 0.5 * hk / (m1 + m2*(1 - np.cos(q2[k+1])**2))
                # Diff wrt q1 is zero
                # Diff wrt q2 - SEE GOOGLE
                output[i, k+2*N] = 0.5*hk*m2*(l*(4*m1 - m2)*np.cos(q2[k])*q2_dot[k]**2 + l*m2*np.cos(3*q2[k])*q2_dot[k]**2 + 2*g*(m2+2*m1)*np.cos(2*q2[k]) - 2*g*m2 - 4*u[k]*np.sin(2*q2[k]))
                output[i, k+2*N] /= (2*m1 + m2*(1 - np.cos(2*q2[k])))**2

                output[i, k+2*N+1] = 0.5*hk*m2*(l*(4*m1 - m2)*np.cos(q2[k+1])*q2_dot[k+1]**2 + l*m2*np.cos(3*q2[k+1])*q2_dot[k+1]**2 + 2*g*(m2+2*m1)*np.cos(2*q2[k+1]) - 2*g*m2 - 4*u[k+1]*np.sin(2*q2[k+1]))
                output[i, k+2*N+1] /= (2*m1 + m2*(1 - np.cos(2*q2[k+1])))**2

                # Diff wrt q1_dot
                output[i, k+3*N] = 1
                output[i, k+3*N+1] = -1

                # Diff wrt q2_dot
                output[i, k+4*N] = hk*l*m2*np.sin(q2[k])*q2_dot[k] / (m1 + m2*(1 - np.cos(q2[k])**2))
                output[i, k+4*N+1] = hk*l*m2*np.sin(q2[k+1])*q2_dot[k+1] / (m1 + m2*(1 - np.cos(q2[k+1])**2))

            elif i < 4*(N - 1): # Constraint Equation: 0.5*hk*(q2_ddot[k] + q2_ddot[k+1]) + q2_dot[k] - q2_dot[k+1]
                k = i - 3*(N - 1)

                # Diff wrt u
                output[i, k] = - 0.5*hk*np.cos(q2[k]) / (l*m1 + l*m2*(1 - np.cos(q2[k])**2))
                output[i, k+1] = - 0.5*hk*np.cos(q2[k+1]) / (l*m1 + l*m2*(1 - np.cos(q2[k+1])**2))
                # Diff wrt q1 is zero
                # Diff wrt q2 - SEE GOOGLE
                output[i, k+2*N] = u[k]*np.sin(q2[k])*(m2*(np.cos(2*q2[k]) + 3) + 2*m1)
                output[i, k+2*N] -= g*(m1 + m2)*np.cos(q2[k])*(m2*(np.cos(2*q2[k]) - 1) + 2*m1)
                output[i, k+2*N] -= l*m2*(m2*(np.cos(2*q2[k]) - 1)*q2_dot[k]**2 + 2*m1*np.cos(2*q2[k]))
                output[i, k+2*N] *= hk
                output[i, k+2*N] /= (l*(m2*(np.cos(2*q2[k]) - 1) - 2*m1)**2)

                output[i, k+2*N+1] = u[k+1]*np.sin(q2[k+1])*(m2*(np.cos(2*q2[k+1]) + 3) + 2*m1)
                output[i, k+2*N+1] -= g*(m1 + m2)*np.cos(q2[k+1])*(m2*(np.cos(2*q2[k+1]) - 1) + 2*m1)
                output[i, k+2*N+1] -= l*m2*(m2*(np.cos(2*q2[k+1]) - 1)*q2_dot[k+1]**2 + 2*m1*np.cos(2*q2[k+1]))
                output[i, k+2*N+1] *= hk
                output[i, k+2*N+1] /= (l*(m2*(np.cos(2*q2[k+1]) - 1) - 2*m1)**2)

                # Diff wrt q1_dot is zero
                # Diff wrt q2_dot
                output[i, k+4*N] = 1 - hk*l*m2*np.cos(q2[k])*np.sin(q2[k])*q2_dot[k] / (l*m1 + l*m2*(1 - np.cos(q2[k])**2))
                output[i, k+4*N+1] = -1 - hk*l*m2*np.cos(q2[k+1])*np.sin(q2[k+1])*q2_dot[k+1] / (l*m1 + l*m2*(1 - np.cos(q2[k+1])**2))

        # Boundary Constraints
        output[4*(N-1), N] = 1 # Diff of q1, 0
        output[4*(N-1)+1, 2*N] = 1 # Diff of q2, 0
        output[4*(N-1)+2, 3*N] = 1 # Diff of q1 dot, 0
        output[4*(N-1)+3, 4*N] = 1 # Diff of q2 dot, 0

        output[4*(N-1)+4, 2*N-1] = 1 # Diff of q1, 0
        output[4*(N-1)+5, 3*N-1] = 1 # Diff of q2, 0
        output[4*(N-1)+6, 4*N-1] = 1 # Diff of q1 dot, 0
        output[4*(N-1)+7, 5*N-1] = 1 # Diff of q2 dot, 0

        return output


    def hessian(self, data, lagrange, obj_factor):
        """Hessian of the lagrangian"""
        u, [q1 ,q2 ,q1_dot, q2_dot] = split_data(data)
        output = np.zeros((5*N, 5*N))
        h_k = T/N

        for i in range(5*N): # Initially iterate through the diagonal

            # Objective function contribution
            if i == 0 or i == N-1:
                obj_val = h_k * obj_factor
            elif i < N-1:
                obj_val = 2*h_k * obj_factor
            else:
                obj_val = 0

            # Constraint summation contribution - Most elements are zero
            if 2*N <= i and i < 3*N: # Second derivative wrt q2
                k = i % N

                # Second derivative of q1_ddot
                der_q1_ddot = q2_dot[k]**2*l*m2**2*( - 14*np.sin(q2[k]) + 3*np.sin(3*q2[k]) + np.sin(5*q2[k]))
                der_q1_ddot += 24*q2_dot[k]**2*l*m2*m1*(np.sin(q2[k]) + np.sin(3*q2[k]))
                der_q1_ddot += 16*q2_dot[k]**2*l*m1**2*np.sin(q2[k])
                der_q1_ddot += g*m2**2*(-8*np.sin(2*q2[k]) + 4*np.sin(4*q2[k]))
                der_q1_ddot += g*m2*m1*(32*np.sin(2*q2[k]) + 8*np.sin(4*q2[k]))
                der_q1_ddot += 32*g*m1**2*np.sin(2*q2[k]) + 16*u[k]*(m2 + 2*m1)*np.cos(2*q2[k]) + 8*m2*u[k]*np.cos(4*q2[k]) - 24*m2*u[k]
                der_q1_ddot *= m2
                der_q1_ddot /= (16*(m2*np.cos(q2[k])**2 - m2 - m1)**3)

                # Second derivative of q2_ddot
                der_q2_ddot = q2_dot[k]**2*l*m2**3*(-8*np.sin(2*q2[k]) + 4*np.sin(4*q2[k]))
                der_q2_ddot += q2_dot[k]**2*l*m2*m1*(32*np.sin(2*q2[k]) + 8*np.sin(4*q2[k]))
                der_q2_ddot += 32*q2_dot[k]**2*l*m2*m1**2*np.sin(2*q2[k])
                der_q2_ddot += g*m2**3*(-14*np.sin(q2[k]) + 3*np.sin(3*q2[k]) + np.sin(5*q2[k]))
                der_q2_ddot += g*m2**2*m1*(27*np.sin(3*q2[k]) + np.sin(5*q2[k]))
                der_q2_ddot += g*m2*m1**2*(40*np.sin(q2[k]) + 24*np.sin(3*q2[k]))
                der_q2_ddot += 16*g*m1**3*np.sin(q2[k]) - 2*u[k]*(11*m2**2 - 4*m2*m1 - 8*m1**2)*np.cos(q2[k])
                der_q2_ddot += m2**2*u[k]*np.cos(5*q2[k]) + 3*m2*u[k]*(7*m2 + 8*m1)*np.cos(3*q2[k])
                der_q2_ddot /= (16*l*(m1+m2-m2*np.cos(q2[k])**2)**3)

                # Aggregate and multiply by lagrange
                con_val = h_k * (der_q1_ddot*lagrange[k + 2*(N-1)] + der_q2_ddot*lagrange[k + 3*(N-1)])

                if k ==0 or k == N-1:
                    con_val *= 0.5

            elif 4*N <= i and i < 5*N: # Second derivative wrt q2_dot
                k = i % N

                # Second derivative of q1_ddot
                der_q1_ddot = 2*l*m2*np.sin(q2[k]) / (m1 + m2*(1 - np.cos(q2[k])**2))

                # Second derivative of q2_ddot
                der_q2_ddot = - 2*l*m2*np.cos(q2[k])*np.sin(q2[k]) / (l*m1 + l*m2*(1 - np.cos(q2[k])**2))

                # Aggregate and multiply by lagrange
                con_val = h_k * (der_q1_ddot*lagrange[k + 2*(N-1)] + der_q2_ddot*lagrange[k + 3*(N-1)])

                if k == 0 or k == N-1:
                    con_val *= 0.5

            else:
                con_val = 0

            # Add non diagonal constraint elements
            # for i in range(N):
            #     con_val = 



            output[i, i] = obj_factor*obj_val + con_val

        return output
            


def variable_bounds():
    """Return the bounds for the data."""
    output = np.full((5, N), np.inf)
    output[0] = u_max
    output[1] = d_max

    # Return in form [[min, max], [min, max], ...]
    return np.array((-output.flatten(), output.flatten())).T

def path_limit(sign=1):
    """Construct the upper limit to the path constraint.
    If sign = -1 output lower limit.
    """
    output = np.full((5, N), np.inf)
    output[0] = u_max
    output[1] = d_max
    return output.flatten()*sign

def solve():
    """Solve the trajectory optimisation problem using cyipopt."""

    lb = path_limit(-1)
    ub = path_limit()

    cl = np.full(4*(N-1)+8, -0.001)
    cu = np.full(4*(N-1)+8, 0.001)

    x0 = initial_guess()

    nlp = cyipopt.Problem(
                    n=5*N,
                    m=len(cl),
                    problem_obj=cart_prob(),
                    lb=lb,
                    ub=ub,
                    cl=cl,
                    cu=cu,
                    )
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('tol', 1e-7)
    
    x, info = nlp.solve(x0)

    u, [q1, q2, q1_dot, q2_dot] = split_data(x)
    return u, q1, q2


def initial_guess() -> np.ndarray:
    """Linearly interpolate between start and end states for initial guess"""
    output = np.zeros((5, N))
    q1 = np.linspace(0, d, N)
    q2 = np.linspace(0, np.pi, N)
    output[1] = q1
    output[2] = q2
    return output.flatten()

if __name__=='__main__':
    u, q1, q2 = solve()
    plot(u, q1, q2, T, N, l)

    a = np.array([[1, 2, 3]])
    b = sparse.dia_matrix((a, np.array([0])), (3, 3))
    print(b.toarray())