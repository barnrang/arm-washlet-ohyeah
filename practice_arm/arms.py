import autograd.numpy as np
from autograd import grad, jacobian
import functools

class TwoArmModel:
    def __init__(self, first_arm_length=1, second_arm_length=1):
        super().__init__()
        self.first_arm_length = first_arm_length
        self.second_arm_length = second_arm_length
        
        self.theta_1 = 0.
        self.theta_2 = 0.
        
    def get_pos_at_angle(self, theta_1, theta_2):
        p2 = np.array([0,0,0,1])
        
        R_01_z_theta_1 = np.array([
            [np.cos(theta_1+np.pi/2), -np.sin(theta_1+np.pi/2), 0, 0],
            [np.sin(theta_1+np.pi/2), np.cos(theta_1+np.pi/2), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        T_01_z = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
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
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        R_02 = functools.reduce(lambda x,y: np.dot(x,y), 
                                [R_01_z_theta_1, T_01_z, R_01_x_90,
                                R_12_z_theta_2, T_12_x])
        
        p0 = np.dot(R_02, p2)
        
        return p0[:3]
    
    def get_pos(self):
        return self.get_pos_at_angle(self.theta_1, self.theta_2)
    
    def set_angle(self, theta_1=None, theta_2=None):
        if theta_1 is not None:
            self.theta_1 = theta_1 % np.pi
            
        if theta_2 is not None:
            self.theta_2 = theta_2 % np.pi
            
    def reverse_kinematic_step(self, x, y, z):
        def cal_loss(thetas):
            pos = self.get_pos_at_angle(thetas[0], thetas[1])
            return pos - np.array([x,y,z])
        
        diff = cal_loss(np.array([self.theta_1, self.theta_2]))
        jacobian_loss = jacobian(cal_loss)
        jac = jacobian_loss(np.array([self.theta_1, self.theta_2]))
        return diff, jac
    
    def update_reverse_kinematic(self, x, y, z, threshold=1e-5, MAX=1000):
        for i in range(MAX):
            diff, jac = self.reverse_kinematic_step(x, y, z)
            jac_inv = np.linalg.pinv(jac)

            [self.theta_1, self.theta_2] = np.array([self.theta_1, self.theta_2]) - np.dot(jac_inv, diff)
            if np.sum(diff ** 2) < threshold:
                print(f"Found angles! theta_1 = {self.theta_1} theta_2 = {self.theta_2}")
                break
        else:
            print("Angles not found")
        


class ThreeArmModel:
    def __init__(self, first_arm_length=1, second_arm_length=1, third_arm_length=1):
        super().__init__()
        self.first_arm_length = first_arm_length
        self.second_arm_length = second_arm_length
        self.third_arm_length = third_arm_length
        
        self.theta_1 = 0.
        self.theta_2 = 0.
        self.theta_3 = 0.
        
    def get_pos_at_angle(self, theta_1, theta_2, theta_3):
        p3 = np.array([0,0,0,1])
        
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
        
        R_03 = functools.reduce(lambda x,y: np.dot(x,y), 
                                [R_01_z_theta_1, T_01_z, R_01_x_90, 
                                R_12_z_theta_2, T_12_x,
                                R_23_z_theta_3, T_23_x])
        
        p0 = np.dot(R_03, p3)
        
        return p0[:3]
    
    def get_pos(self):
        return self.get_pos_at_angle(self.theta_1, self.theta_2, self.theta_3)
    
    def set_angle(self, theta_1=None, theta_2=None, theta_3=None):
        if theta_1 is not None:
            self.theta_1 = theta_1 % np.pi
            
        if theta_2 is not None:
            self.theta_2 = theta_2 % np.pi

        if theta_3 is not None:
            self.theta_3 = theta_3 % np.pi
            
    def reverse_kinematic_step(self, x, y, z):
        def cal_loss(thetas):
            pos = self.get_pos_at_angle(thetas[0], thetas[1], thetas[2])
            return pos - np.array([x,y,z])
        
        diff = cal_loss(np.array([self.theta_1, self.theta_2, self.theta_3]))
        jacobian_loss = jacobian(cal_loss)
        jac = jacobian_loss(np.array([self.theta_1, self.theta_2, self.theta_3]))
        return diff, jac
    
    def update_reverse_kinematic(self, x, y, z, threshold=1e-5, MAX=1000):
        for i in range(MAX):
            diff, jac = self.reverse_kinematic_step(x, y, z)
            jac_inv = np.linalg.pinv(jac)

            new_angle = np.array([self.theta_1, self.theta_2, self.theta_3]) - np.dot(jac_inv, diff)
            self.set_angle(*new_angle)
            if np.sum(diff ** 2) < threshold:
                print(f"Found angles! theta_1 = {self.theta_1} theta_2 = {self.theta_2} theta_3 = {self.theta_3}")
                break
        else:
            print("Angles not found")
