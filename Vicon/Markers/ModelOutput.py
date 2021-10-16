#!/usr/bin/env python

# //==============================================================================
# /*
#     Software License Agreement (BSD License)
#     Copyright (c) 2020, AIMVicon
#     (www.aimlab.wpi.edu)

#     All rights reserved.

#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions
#     are met:

#     * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.

#     * Neither the name of authors nor the names of its contributors may
#     be used to endorse or promote products derived from this software
#     without specific prior written permission.

#     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#     "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#     LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#     FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#     COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#     INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#     BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#     LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#     CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#     LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#     ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#     POSSIBILITY OF SUCH DAMAGE.

#     \author    <http://www.aimlab.wpi.edu>
#     \author    <nagoldfarb@wpi.edu>
#     \author    Nathaniel Goldfarb
#     \version   0.1
# */
# //==============================================================================

import copy

from GaitCore import Core as core
from GaitCore.Bio.Sara import Sara
from GaitCore.Bio.Score import Score
from GaitCore.Bio.Joint import Joint
from GaitCore.Core.Angle import Angle

from GaitCore.Bio.Leg import Leg
from GaitCore.Bio.Arm import Arm
from GaitCore.Bio.Trunk import Trunk
from GaitCore.Bio.Joint import Joint

class ModelOutput():

    def __init__(self, model_data: dict = {}, verbose=False, joints: dict = None):

        # Create class-wide variables
        self._raw_model_data = model_data # Model data should already be split up in Vicon.py
        self._joints = joints

        self._set_sara()
        self._set_score()



    def _make_models(self):

        joint_names = ["Hip", "Knee", "Ankle", "Head", "Thorax", "Neck", "Shoulder", "Pelvis", "Spine", "Wrist", "Elbow"]
        left_joints = {}
        right_joints = {}
        for side, joint in zip(("L", "R"), (left_joints, right_joints)):
            for output in joint_names:
                angle = None
                power = None
                moment = None
                force = None
                if side + output + "Angles" in self._raw_model_data.keys():
                    angle = core.PointArray.PointArray(self._raw_model_data[side + output + "Angles"]["X"]["data"],
                                                       self._raw_model_data[side + output + "Angles"]["Y"]["data"],
                                                       self._raw_model_data[side + output + "Angles"]["Z"]["data"])
                if side + output + "Force" in self._raw_model_data.keys():
                    force = core.PointArray.PointArray(self._raw_model_data[side + output + "Force"]["X"]["data"],
                                                       self._raw_model_data[side + output + "Force"]["Y"]["data"],
                                                       self._raw_model_data[side + output + "Force"]["Z"]["data"])
                if side + output + "Moment" in self._raw_model_data.keys():
                    moment = core.PointArray.PointArray(self._raw_model_data[side + output + "Moment"]["X"]["data"],
                                                        self._raw_model_dataata[side + output + "Moment"]["Y"]["data"],
                                                        self._raw_model_data[side + output + "Moment"]["Z"]["data"])
                if side + output + "Power" in self._raw_model_data.keys():
                    power = core.PointArray.PointArray(self._raw_model_data[side + output + "Power"]["X"]["data"],
                                                       self._raw_model_data[side + output + "Power"]["Y"]["data"],
                                                       self._raw_model_data[side + output + "Power"]["Z"]["data"])

                joint[output] = Joint(angle, moment, power, force)
                #joint[output] = core.Newton.Newton(angle, force, moment, power)

            if "Hip" in left_joints and "Knee" in left_joints and "Ankle" in left_joints :
                self._left_leg = Leg(left_joints["Hip"], left_joints["Knee"], left_joints["Ankle"])


            if "Hip" in right_joints and "Knee" in right_joints and "Ankle" in right_joints :
                self._right_leg = Leg(right_joints["Hip"], right_joints["Knee"], right_joints["Ankle"])

            if "Shoulder" in left_joints and "Elbow" in left_joints and "Wrist" in left_joints :
                self._left_arm = Arm(left_joints["Shoulder"], left_joints["Elbow"], left_joints["Wrist"])

            if "Shoulder" in right_joints and "Elbow" in right_joints and "Wrist" in right_joints :
                self._right_arm = Arm(right_joints["Shoulder"], right_joints["Elbow"], right_joints["Wrist"])

            if "Head" in left_joints and "Spine" in left_joints and "Thorax" in left_joints and "Pelvis" in left_joints :
                self._left_trunk = Trunk(left_joints["Head"], left_joints["Spine"], left_joints["Thorax"], left_joints["Pelvis"])

            if "Head" in right_joints and "Spine" in right_joints and "Thorax" in right_joints and "Pelvis" in right_joints :
                self._right_trunk = Trunk(right_joints["Head"], right_joints["Spine"], right_joints["Thorax"], right_joints["Pelvis"] )


    def _set_sara(self):
        """
        Sets SARA for all joints with available data
        SARA  => Symmetrical Axis of Rotation Analysis
        """

        for key, value in self._raw_model_data.items():
            if 'sara' in key:
                # if self._joints.get(key.replace('_sara', '')) == None:
                #   self._joints[key.replace('_sara','')]  = Joint(name = key)
                self._joints.get(key.replace('_sara', '')).sara = Sara(sara_data = value)
                #
    def _set_score(self):
        """
        Sets SCoRE for all joints with available data
        SCoRE => Symmetrical Center of Rotation Estimation
        """

        for key, value in self._raw_model_data.items():
            if 'score' in key:
                # if self._joints.get(key.replace('_score', '')) == None:
                #     self._joints[key.replace('_score', '')]  = Joint(name = key)
            self._joints.get(key.replace('_score', '')).score = Score(score_data= value)

    def make_right_leg(self, hip_joint,        knee_joint,         ankle_joint,
                            hip_angle = None,  knee_angle = None,  ankle_angle=None,
                            hip_force = None,  knee_force = None,  ankle_force = None,
                            hip_moment = None, knee_moment = None, ankle_moment = None,
                            hip_power = None,  knee_power = None,  ankle_power = None):
        if hip_angle is not None: hip_joint._angle = Angle(hip_angle)
        if hip_force is not None: hip_joint._force = hip_force
        if hip_moment is not None: hip_joint._moment = hip_moment
        if hip_power is not None: hip_joint._power = hip_power

        if knee_angle is not None: knee_joint._angle = Angle(knee_angle)
        if knee_force is not None: knee_joint._force = knee_force
        if knee_moment is not None: knee_joint._moment = knee_moment
        if knee_power is not None: knee_joint._power = knee_power

        if ankle_angle is not None: ankle_joint._angle = Angle(ankle_angle)
        if ankle_force is not None: ankle_joint._force = ankle_force
        if ankle_moment is not None: ankle_joint._moment = ankle_moment
        if ankle_power is not None: ankle_joint._power = ankle_power
        
        self._right_leg = Leg(hip_joint, knee_joint, ankle_joint)

    def make_left_leg(self, hip_joint,        knee_joint,         ankle_joint,
                            hip_angle = None,  knee_angle = None,  ankle_angle=None,
                            hip_force = None,  knee_force = None,  ankle_force = None,
                            hip_moment = None, knee_moment = None, ankle_moment = None,
                            hip_power = None,  knee_power = None,  ankle_power = None):
        if hip_angle is not None: hip_joint._angle = Angle(hip_angle)
        if hip_force is not None: hip_joint._force = hip_force
        if hip_moment is not None: hip_joint._moment = hip_moment
        if hip_power is not None: hip_joint._power = hip_power

        if knee_angle is not None: knee_joint._angle = Angle(knee_angle)
        if knee_force is not None: knee_joint._force = knee_force
        if knee_moment is not None: knee_joint._moment = knee_moment
        if knee_power is not None: knee_joint._power = knee_power

        if ankle_angle is not None: ankle_joint._angle = Angle(ankle_angle)
        if ankle_force is not None: ankle_joint._force = ankle_force
        if ankle_moment is not None: ankle_joint._moment = ankle_moment
        if ankle_power is not None: ankle_joint._power = ankle_power
        
        self._left_leg = Leg(hip_joint, knee_joint, ankle_joint)

    def make_right_arm(self, shoulder_joint,        elbow_joint,         wrist_joint,
                            shoulder_angle = None,  elbow_angle = None,  wrist_angle=None,
                            shoulder_force = None,  elbow_force = None,  wrist_force = None,
                            shoulder_moment = None, elbow_moment = None, wrist_moment = None,
                            shoulder_power = None,  elbow_power = None,  wrist_power = None):
        if shoulder_angle is not None: shoulder_joint._angle = Angle(shoulder_angle)
        if shoulder_force is not None: shoulder_joint._force = shoulder_force
        if shoulder_moment is not None: shoulder_joint._moment = shoulder_moment
        if shoulder_power is not None: shoulder_joint._power = shoulder_power

        if elbow_angle is not None: elbow_joint._angle = Angle(elbow_angle)
        if elbow_force is not None: elbow_joint._force = elbow_force
        if elbow_moment is not None: elbow_joint._moment = elbow_moment
        if elbow_power is not None: elbow_joint._power = elbow_power

        if wrist_angle is not None: wrist_joint._angle = Angle(wrist_angle)
        if wrist_force is not None: wrist_joint._force = wrist_force
        if wrist_moment is not None: wrist_joint._moment = wrist_moment
        if wrist_power is not None: wrist_joint._power = wrist_power
        
        self._right_arm = Arm(shoulder_joint, elbow_joint, wrist_joint)

    def make_left_arm(self, shoulder_joint,        elbow_joint,         wrist_joint,
                            shoulder_angle = None,  elbow_angle = None,  wrist_angle=None,
                            shoulder_force = None,  elbow_force = None,  wrist_force = None,
                            shoulder_moment = None, elbow_moment = None, wrist_moment = None,
                            shoulder_power = None,  elbow_power = None,  wrist_power = None):
        if shoulder_angle is not None: shoulder_joint._angle = Angle(shoulder_angle)
        if shoulder_force is not None: shoulder_joint._force = shoulder_force
        if shoulder_moment is not None: shoulder_joint._moment = shoulder_moment
        if shoulder_power is not None: shoulder_joint._power = shoulder_power

        if elbow_angle is not None: elbow_joint._angle = Angle(elbow_angle)
        if elbow_force is not None: elbow_joint._force = elbow_force
        if elbow_moment is not None: elbow_joint._moment = elbow_moment
        if elbow_power is not None: elbow_joint._power = elbow_power

        if wrist_angle is not None: wrist_joint._angle = Angle(wrist_angle)
        if wrist_force is not None: wrist_joint._force = wrist_force
        if wrist_moment is not None: wrist_joint._moment = wrist_moment
        if wrist_power is not None: wrist_joint._power = wrist_power
        
        self._left_arm = Arm(shoulder_joint, elbow_joint, wrist_joint)


    
    def get_right_leg(self):
        """

        :return:
        """
        return self._right_leg

    def get_left_leg(self):
        """

        :return:
        """
        return self._left_leg

    def get_right_arm(self):
        """

        :return:
        """
        return self._right_arm

    def get_left_arm(self):
        """

        :return:
        """
        return self._left_arm

    def get_right_trunk(self):
        """

        :return:
        """
        return self._right_trunk

    def get_left_trunk(self):
        """

        :return:
        """
        return self._left_trunk

    def get_joint(self, name):
        return self._joints.get(name)

    def get_joint_names(self):
        return self._joints.keys()