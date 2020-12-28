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

from GaitCore import Core as core
from GaitCore.Bio.Leg import Leg
from GaitCore.Bio.Arm import Arm
from GaitCore.Bio.Trunk import Trunk
from GaitCore.Bio.Joint import Joint

class ModelOutput(object):

    def __init__(self, data):

        self.joint_names = ["Hip", "Knee", "Ankle", "Head", "Thorax", "Neck", "Shoulder", "Pelvis", "Spine", "Wrist"]
        left_joints = {}
        right_joints = {}

        for side, joint in zip(("L", "R"), (left_joints, right_joints)):
            for output in self.joint_names:
                angle = None
                power = None
                moment = None
                force = None
                if side + output + "Angles" in data.keys():
                    angle = core.PointArray.PointArray(data[side + output + "Angles"]["X"]["data"],
                                       data[side + output + "Angles"]["Y"]["data"],
                                       data[side + output + "Angles"]["Z"]["data"])
                if side + output + "Force" in data.keys():
                    force = core.PointArray.PointArray(data[side + output + "Force"]["X"]["data"],
                                       data[side + output + "Force"]["Y"]["data"],
                                       data[side + output + "Force"]["Z"]["data"])
                if side + output + "Moment" in data.keys():
                    moment = core.PointArray.PointArray(data[side + output + "Moment"]["X"]["data"],
                                        data[side + output + "Moment"]["Y"]["data"],
                                        data[side + output + "Moment"]["Z"]["data"])
                if side + output + "Power" in data.keys():
                    power = core.PointArray.PointArray(data[side + output + "Power"]["X"]["data"],
                                       data[side + output + "Power"]["Y"]["data"],
                                       data[side + output + "Power"]["Z"]["data"])

                joint[output] = Joint(angle, moment, power, force)
                #joint[output] = core.Newton.Newton(angle, force, moment, power)

        self._left_leg = Leg(left_joints["Hip"], left_joints["Knee"], left_joints["Ankle"])
        self._right_leg = Leg(right_joints["Hip"], right_joints["Knee"], right_joints["Ankle"])

        self._left_arm = Arm(left_joints["Shoulder"], left_joints["Elbow"], left_joints["Wrist"])
        self._right_arm = Arm(right_joints["Shoulder"], right_joints["Elbow"], right_joints["Wrist"])

        self._left_trunk = Trunk(left_joints["Head"], left_joints["Spine"], left_joints["Thorax"], left_joints["Plevis"] )
        self._right_trunk = Trunk(right_joints["Head"], right_joints["Spine"], right_joints["Thorax"], right_joints["Plevis"] )

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