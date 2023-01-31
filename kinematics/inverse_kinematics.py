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
import numpy as np
from scipy.optimize import fmin
from spark_agent import JOINT_SENSOR_NAMES


class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''
        joint_angles = []
        chain = self.chains[effector_name]
        end_trans_name = chain[-1]
        
        def error_func(angles):
            joint_angles = {x:0 for x in JOINT_SENSOR_NAMES.values()}
            joint_angles.update({c:a for (a,c) in zip(angles, chain)})
            self.forward_kinematics(joint_angles)
            actual_trans = self.transforms[end_trans_name].T
            # print(angles)
            # print(actual_trans)
            e = np.linalg.norm(transform-actual_trans)
            # print(e)
            return e

        initial = np.random.random(len(chain))
        # initial = [0]*len(chain)
        joint_angles = fmin(error_func, initial)
        print(self.transforms[end_trans_name].T)

        return joint_angles

    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        chain = self.chains[effector_name]
        angles = self.inverse_kinematics(effector_name, transform)
        print(angles)

        names = chain
        times = [[0, 5]]*len(chain)
        keys = [[[0, None, None], [x, None, None]] for x in angles]

        self.keyframes = (names, times, keys)  # the result joint angles have to fill in

if __name__ == '__main__':
    agent = InverseKinematicsAgent()

    #joint_angles = {x:0 for x in JOINT_SENSOR_NAMES.values()}
    #joint_angles.update({"LElbowRoll": -1.396})

    #agent.forward_kinematics(joint_angles)
    #actual_trans = agent.transforms["LElbowRoll"].T
    #print(actual_trans)

    # test inverse kinematics
    T = identity(4)
    T[-1, 1] = 0.05
    T[-1, 2] = -0.26

    print(T)
    agent.set_transforms('LLeg', T)
    #agent.set_transforms('LArm', T)
    agent.run()
