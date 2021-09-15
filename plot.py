import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

def plot(u, q1, q2, T, N, l, cart_w=0.6, cart_h=0.2):
    """Plot and animate the data supplied using matplotlib"""
    mass_y = -l*np.cos(q2)
    mass_x = q1 + l*np.sin(q2)
    t = np.linspace(0, T, N)

    # Axis size
    x_min = min(min(q1), min(mass_x)) - cart_w
    x_max = max(max(q1), max(mass_x)) + cart_w
    y_max = max(abs(mass_y)) + cart_h

    fig = plt.figure() 
    axis = plt.axes(xlim =(x_min, x_max),
                    ylim =(-y_max, y_max)) 
    
    line, = axis.plot([], [], lw = 1, label="Trace")
    box_lines, = axis.plot([], [], lw=2, label="Cart")
    arm_line, = axis.plot([], [], lw=2, label="Arm")
    point, = axis.plot([], [], 'bo', label="Mass")
    force_arrow, = axis.plot([], [], label="Force")
    
    def init(): 
        line.set_data([], []) 
        return line, 
    
    line_x, line_y = [], [] 
    
    # animation function 
    def animate(i): 

        # Line Trace
        if len(line_x) < N:
            line_x.append(mass_x[i]) 
            line_y.append(mass_y[i]) 
        line.set_data(line_x, line_y) 

        # Box
        box_x = [q1[i]-cart_w/2, q1[i]-cart_w/2, q1[i]+cart_w/2, q1[i]+cart_w/2, q1[i]-cart_w/2]
        box_y = [-cart_h/2, cart_h/2, cart_h/2, -cart_h/2, -cart_h/2]
        box_lines.set_data(box_x, box_y)

        # Arm
        arm_line.set_data([q1[i], mass_x[i]], [0, mass_y[i]])

        # Set point position
        point.set_data(mass_x[i], mass_y[i])

        # Force arrow
        multiplier = -u[i]*0.05
        force_x = np.array([
                cart_h,
                cart_h,
                0,
                cart_h,
                cart_h,
                cart_h*5
            ])
        force_y = np.array([0, cart_h/2, 0, -cart_h/2, 0, 0]) * multiplier
        force_x = multiplier * (force_x +  0.5*cart_w/abs(multiplier)) + q1[i]

        force_arrow.set_data(force_x, force_y)

        return (line, box_lines, arm_line, point, force_arrow)
    
    # calling the animation function     
    anim = animation.FuncAnimation(fig, animate, init_func = init, 
                                frames = len(t), interval = 1000*T/N, blit = True, repeat_delay=2) 
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.legend()
    plt.show()