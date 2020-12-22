
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
import csv
from typing import List, Any

import pandas
import numpy as np
from .Markers import ModelOutput as modeloutput
from .Markers import Markers as markers
from .Devices import EMG, IMU, Accel, ForcePlate
import matplotlib.pyplot as plt
from Vicon import Markers
from .Markers import Interpolation
import abc

class MocapBase(object):

    def __init__(self, file_path, verbose=False, interpolate=True, maxnanstotal=-1, maxnansrow=-1, sanitize=True):
        self._file_path = file_path
        self._verbose = verbose
        self._interpolate = interpolate
        self._maxanstotal=-1
        self._maxnansrow = maxnansrow
        self._sanitize = sanitize
        self._number_of_frames = 0
        self.data_dict = self.open_file(self._file_path, verbose=verbose, interpolate=interpolate,
                                              maxnanstotal=maxnanstotal, maxnansrow=maxnansrow, sanitize=sanitize)

    def _find_number_of_frames(self, col):
        """
        Finds the number and sets of frames
        :param col: column to search in
        :return: None
        """
        index = col.index("Frame") + 2
        current_number = col[index]

        while current_number.isdigit():
            index += 1
            current_number = col[index]

        self.number_of_frames = col[index - 1]

    @property
    def markers(self):
        return self._markers

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value

    @property
    def number_of_frames(self):
        """

        :return: number of frames
        :rtype: int
        """
        return self._number_of_frames

    @number_of_frames.setter
    def number_of_frames(self, value):
        """

        :param value:
        :return:
        """
        self._number_of_frames = value

    def get_markers(self):
        """
        get the markers
        :return: markers
        :type: dict
        """
        return self.markers

    def get_joints(self):
        """
        get the joints
        :return: model joints
        :type: dict
        """
        return self.data_dict["Joints"]

    def get_joints_keys(self):
        """
        get the joints keys
        :return: model joints keys
        :type: list of keys
        """
        return self.data_dict.keys()

    def _check_keys(self, key_list, key):
        """

        :param dict:
        :param key:
        :return:
        """

        return any(key in s for s in key_list)

    def _filter_number(self, key):
        """

        :param key:
        :return:
        """
        return int(''.join(filter(str.isdigit, key)))

    def _filter_dict(self, sensors, substring):
        """
        filter the dictionary
        :param sensors: Dictionary to parse
        :param substring: substring of the keys to look for in the dict
        :return: keys that contain the substring
        :type: list
        """
        my_list = []
        return list(filter(lambda x: substring in x, sensors.keys()))

    def _make_markers(self):
        markers = self.data_dict["Trajectories"]

    def _make_marker_trajs(self):
        """
        generate IMU models
        :return: None
        """
        self._markers = markers.Markers(self.data_dict["Trajectories"])
        self._markers.make_markers()

    @abc.abstractmethod
    def save(self, filename=None, verbose=False, mark_interpolated=True):
        pass

    def _false_of_n(self, n):
        """Helper function to generate an array of Falses of length N"""
        arr = []
        for i in range(n):
            arr.append(False)
        return arr

    def _len_data(self, category):
        """Returns the length of the data section of a given category"""
        return len(next(next(self.data_dict[category].itervalues()).itervalues())["data"])