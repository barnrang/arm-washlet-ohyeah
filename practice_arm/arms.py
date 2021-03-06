import autograd.numpy as np
from autograd import grad, jacobian
import functools
from matplotlib import animation, rc

PI2 = 2 * np.pi

def plot3D_line(ax, v1, v2, **kwargs):
    return ax.plot3D(*list(zip(v1, v2)), **kwargs)[0]

class ThreeArmModel:
    def __init__(self, first_arm_length=1, second_arm_length=1, third_arm_length=1):
        super().__init__()
        self.first_arm_length = first_arm_length
        self.second_arm_length = second_arm_length
        self.third_arm_length = third_arm_length
        
        self.theta_1 = 0.
        self.theta_2 = 0.
        self.theta_3 = 0.
        self.set_angle(0.,0.,0.)

    def get_pos(self):
        '''
        Get the position of the final arm of the current state
        '''
        return self.get_arm_pos(3, *self.get_angle())

    def get_arm_pos(self, arm, theta_1, theta_2, theta_3):
        '''
        Get the position of the end of an arm
        '''
        R_01_z_theta_1 = np.array([
            [np.cos(theta_1+np.pi/2), -np.sin(theta_1+np.pi/2), 0, 0],
            [np.sin(theta_1+np.pi/2), np.cos(theta_1+np.pi/2), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        T_01_z = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, self.first_arm_length],
            [0, 0, 0, 1]
        ])
        
        R_01_x_90 = np.array([
            [1, 0, 0, 0],
            [0, np.cos(np.pi/2), -np.sin(np.pi/2), 0],
            [0, np.sin(np.pi/2), np.cos(np.pi/2), 0],
            [0, 0, 0, 1]
        ])
        
        R_12_z_theta_2 = np.array([
            [np.cos(theta_2+np.pi/2), -np.sin(theta_2+np.pi/2), 0, 0],
            [np.sin(theta_2+np.pi/2), np.cos(theta_2+np.pi/2), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        T_12_x = np.array([
            [1, 0, 0, self.second_arm_length],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        R_23_z_theta_3 = np.array([
            [np.cos(theta_3), -np.sin(theta_3), 0, 0],
            [np.sin(theta_3), np.cos(theta_3), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        T_23_x = T_12_x = np.array([
            [1, 0, 0, self.third_arm_length],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        if arm == 1:
            p1 = np.array([0., 0., 0., 1.])
            R_01 = functools.reduce(lambda x,y: np.dot(x,y), 
                                    [R_01_z_theta_1, T_01_z, 
                                    R_01_x_90])

            return np.dot(R_01, p1)[:3]
            
        elif arm == 2:
            p2 = np.array([0., 0., 0., 1.])
            R_02 = functools.reduce(lambda x,y: np.dot(x,y), 
                                    [R_01_z_theta_1, T_01_z, R_01_x_90,
                                    R_12_z_theta_2, T_12_x])
        
            return np.dot(R_02, p2)[:3]

        elif arm == 3:
            p3 = np.array([0., 0., 0., 1.])
            R_03 = functools.reduce(lambda x,y: np.dot(x,y), 
                                [R_01_z_theta_1, T_01_z, R_01_x_90, 
                                R_12_z_theta_2, T_12_x,
                                R_23_z_theta_3, T_23_x])

            return np.dot(R_03, p3)[:3]
        else:
            raise f"Arm no.{arm} not exist!!"

    def plot_arm(self, ax, angles=None):
        if angles is None:
            first_arm_pos = self.get_arm_pos(1, *self.get_angle())
            second_arm_pos = self.get_arm_pos(2, *self.get_angle())
            third_arm_pos = self.get_arm_pos(3, *self.get_angle())
        else:
            first_arm_pos = self.get_arm_pos(1, *angles)
            second_arm_pos = self.get_arm_pos(2, *angles)
            third_arm_pos = self.get_arm_pos(3, *angles)

        line_first = plot3D_line(ax, [0,0,0], first_arm_pos, color='blue')
        line_second = plot3D_line(ax, first_arm_pos, second_arm_pos, color='red')
        line_third = plot3D_line(ax, second_arm_pos, third_arm_pos, color='green')

        return line_first, line_second, line_third

    def plot_reverse_animation(self, fig, ax, x, y, z, threshold=1e-5, MAX=1000):
        frames = 48

        current_angle = self.get_angle()

        # Update using reverse kinematic
        self.update_reverse_kinematic(x, y, z, threshold, MAX)
        new_angle = self.get_angle()

        # Angle change by frame
        diff_angle = new_angle - current_angle
        for i in range(len(diff_angle)):
            if np.abs(diff_angle[i]) > np.pi:
                if diff_angle[i] > 0:
                    diff_angle[i] = diff_angle[i] - PI2
                else:
                    diff_angle[i] = PI2 + diff_angle[i]

        change_angle = diff_angle / frames

        lines = self.plot_arm(ax, current_angle)

        def get_plot(i):
            plot_angle = current_angle + i * change_angle
            first = self.get_arm_pos(1, *plot_angle)
            second = self.get_arm_pos(2, *plot_angle)
            third = self.get_arm_pos(3, *plot_angle)

            lines[0].set_data(*list(zip([0,0], first[:2])))
            lines[0].set_3d_properties([0, first[2]])

            lines[1].set_data(*list(zip(first[:2], second[:2])))
            lines[1].set_3d_properties([first[2], second[2]])

            lines[2].set_data(*list(zip(second[:2], third[:2])))
            lines[2].set_3d_properties([second[2], third[2]])

            return lines

        anim = animation.FuncAnimation(fig, get_plot, frames=48, interval=100, blit=True)

        return anim
    
    def set_angle(self, theta_1=None, theta_2=None, theta_3=None):
        if theta_1 is not None:
            self.theta_1 = ((theta_1 % PI2) + PI2) % PI2
            
        if theta_2 is not None:
            self.theta_2 = ((theta_2 % PI2) + PI2) % PI2

        if theta_3 is not None:
            self.theta_3 = ((theta_3 % PI2) + PI2) % PI2

    def get_angle(self):
        return np.array([self.theta_1, self.theta_2, self.theta_3])
            
    def reverse_kinematic_step(self, x, y, z):
        '''
        Calculate error and Jacobian for updating reverse kinematic
        '''
        def cal_loss(thetas):
            pos = self.get_arm_pos(3, *thetas)
            return pos - np.array([x,y,z])
        
        diff = cal_loss(self.get_angle())
        jacobian_loss = jacobian(cal_loss)
        jac = jacobian_loss(self.get_angle())

        return diff, jac
    
    def update_reverse_kinematic(self, x, y, z, threshold=1e-5, MAX=1000):
        '''
        Update reverse kinetamtic by iteration
        '''
        for i in range(MAX):
            diff, jac = self.reverse_kinematic_step(x, y, z)
            jac_inv = np.linalg.pinv(jac)

            new_angle = self.get_angle() - np.dot(jac_inv, diff)
            self.set_angle(*new_angle)
            if np.sum(diff ** 2) < threshold:
                print(f"Found angles! theta_1 = {self.theta_1} theta_2 = {self.theta_2} theta_3 = {self.theta_3}")
                break
        else:
            print("Angles not found")
