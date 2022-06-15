'''In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
'''


from forward_kinematics import ForwardKinematicsAgent
from numpy.matlib import identity
from scipy.linalg import pinv
import numpy as np 
from math import atan2

def from_trans(T):
    theta = 0
    x = T[3, 0]
    y = T[3, 1]
    z = T[3, 2]

    if T[0, 0] == 1:
        theta = atan2(T[2, 1], T[1, 1])
    elif T[1, 1] == 1:
        theta = atan2(T[0, 2], T[0, 0])
    elif T[2, 2] == 1:
        theta = atan2(T[1, 0], T[0, 0])

    return np.array([x, y, z, theta])

class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''
        joint_angles = []
        lambda_ = 1
        max_step = 0.1
        N = len(self.chains[effector_name])
        joint_angles = np.random.random(N)
        target = np.matrix([from_trans(transform)]).T
        for i in range(1000):
            Ts = [identity(N)]
            for name in self.chains[effector_name]:
                Ts.append(self.transforms[name])
            Te = np.matrix([from_trans(Ts[-1])]).T
            e = target - Te
            e[e > max_step] = max_step
            e[e < -max_step] = -max_step
            T = np.matrix([from_trans(i) for i in Ts[0:-1]]).T
            J = Te - T
            dT = Te - T
            J[0, :] = dT[2, :]
            J[1, :] = dT[1, :] 
            J[2, :] = dT[0, :]
            J[-1, :] = 1  
            d_theta = np.dot(lambda_, np.dot(pinv(J), e))
            joint_angles += np.asarray(d_theta.T)[0]
            if np.linalg.norm(d_theta) < 1e-4:
                break

        return joint_angles

    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # YOUR CODE HERE
        self.keyframes = ([], [], [])  # the result joint angles have to fill in

if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = identity(4)
    T[-1, 1] = 0.05
    T[-1, 2] = -0.26
    agent.set_transforms('LLeg', T)
    agent.run()
