
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
from ..Markers import ModelOutput as modeloutput
from ..Markers import Markers as markers
from ..Devices import EMG, IMU, Accel, ForcePlate
import matplotlib.pyplot as plt
from Vicon import Markers
from ..Interpolation import Interpolation
import abc
from ..Interpolation import Akmia
class MocapBase(object):

    def __init__(self, file_path, verbose=False, interpolate=True, maxnanstotal=-1, maxnansrow=-1, sanitize=True, inerpolation_method=Akmia.Akmia):
        self._file_path = file_path
        self._verbose = verbose
        self._interpolate = interpolate
        self._maxanstotal = maxnanstotal
        self._maxnansrow = maxnansrow
        self._sanitize = sanitize
        self._number_of_frames = 0
        self._nan_dict = {}
        self.my_marker_interpolation = inerpolation_method
        #  sanitized is a dictionary to keep track of what subject, if any, have had their fields sanitized
        #  If sanitized[category][subject] exists, that subject has had at least one field sanitized
        self._sanitized = {}
        # self.data_dict = self.open_file(self._file_path, verbose=verbose, interpolate=interpolate,
        #                                       maxnanstotal=maxnanstotal, maxnansrow=maxnansrow, sanitize=sanitize)

    @abc.abstractmethod
    def parse(self):
        raise NotImplementedError

    @abc.abstractmethod
    def open_file(self, file_path, verbose=False, interpolate=True, maxnanstotal=-1, maxnansrow=-1,
                        sanitize=True):
        raise NotImplementedError

    @abc.abstractmethod
    def _seperate_csv_sections(self, all_data):
        raise NotImplementedError


    @abc.abstractmethod
    def _extract_values(self, raw_data, start, end, verbose=False, category="", interpolate=True, maxnanstotal=-1,
                    maxnansrow=-1, sanitize=True):
        raise NotImplementedError

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


    def is_sanitized(self, category, subject):
        if category not in self._sanitized:
            return False
        for x in self._sanitized[category]:
            if subject in x:
                return True
        return False

    def graph(self, category, subject, field, showinterpolated=True, colorinterpolated=True, limits=None):
        """Graphs the data specified. If showinterpolated is set to False, interpolated values will not be shown."""
        if not (category in self.data_dict and subject in self.data_dict[category] and field in
                self.data_dict[category][subject]):
            return  # We don't have any data for this field!
        interpolated = True in self._nan_dict[category][subject][field]
        if not interpolated or (not colorinterpolated and showinterpolated):  # Simplest case - just graph the data
            plt.plot(self.data_dict[category][subject][field]["data"])
            plt.xlabel("Frame")
            plt.ylabel(self.data_dict[category][subject][field]["unit"])
            plt.title("Data in category " + category + ", in subject " + subject + ", in field " + field)
            if limits is not None:
                plt.xlim(limits)
            plt.show()
        else:
            nans = self._nan_dict[category][subject][field]
            data = self.data_dict[category][subject][field]["data"]

            orgdatablocks = []
            interdatablocks = []
            orgtemp = []
            intertemp = []
            for i in range(len(nans)):
                if nans[i]:
                    if len(orgtemp) > 0:
                        orgdatablocks.append(orgtemp)
                        orgtemp = []
                    intertemp.append(i)
                else:
                    if len(intertemp) > 0:
                        interdatablocks.append(intertemp)
                        intertemp = []
                    orgtemp.append(i)
            if len(orgtemp) > 0:
                orgdatablocks.append(orgtemp)
            if len(intertemp) > 0:
                interdatablocks.append(intertemp)

            flagorg = True
            for blk in orgdatablocks:
                if flagorg:
                    plt.plot(blk, data[blk[0]:blk[len(blk) - 1] + 1], "C0", label="Original Data")
                    flagorg = False
                else:
                    plt.plot(blk, data[blk[0]:blk[len(blk) - 1] + 1], "C0")

            if showinterpolated:
                flagint = True
                for blk in interdatablocks:
                    if flagint:
                        plt.plot([blk[0]-1] + blk + [blk[len(blk) - 1] + 1], data[blk[0]-1:blk[len(blk) - 1] + 2], "C1", label="Interpolated Data")
                        flagint = False
                    else:
                        plt.plot([blk[0]-1] + blk + [blk[len(blk) - 1] + 1], data[blk[0]-1:blk[len(blk) - 1] + 2], "C1")
                plt.legend()

            plt.xlabel("Frame")
            plt.ylabel(self.data_dict[category][subject][field]["unit"])
            plt.title("Data in category " + category + ", in subject " + subject + ", in field " + field)
            if limits is not None:
                plt.xlim(limits)
            plt.show()

    def _marker_interpolation(self, value, key, naninfo, category, interpolate, sanitize, verbose):
        ### OLD NOT USED
        #  If we have NaNs and the whole row isn't NaNs...
        #  No interpolation method can do anything with an array of NaNs,
        #  so this way we save ourselves a bit of computation

        nans = np.isnan(value["X"]["data"])
        if True in nans and False in nans and interpolate and naninfo[key]["X"]["interpolate"]:
            if category not in self._nan_dict:
                self._nan_dict[category] = {}
            if key not in self._nan_dict[category]:
                self._nan_dict[category][key] = {}
            self._nan_dict[category][key]["X"] = nans
            self._nan_dict[category][key]["Y"] = nans
            self._nan_dict[category][key]["Z"] = nans
            if verbose:
                print("Interpolating missing values in field X Y Z" + ", in subject " + key + \
                      ", in category " + category + "...")
            # x, y, z = Interpolation.velocity_method(value["X"]["data"], value["Y"]["data"], value["Z"]["data"])
            # value["X"]["data"] = Interpolation.akmia(value["X"], verbose, category, "X", key)
            # value["Y"]["data"] = Interpolation.akmia(value["X"], verbose, category, "X", key)
            # value["Z"]["data"] = Interpolation.akmia(value["X"], verbose, category, "X", key)
        else:
            for sub_key, sub_value in value.items():  # For each field under each subject...
                nans = np.isnan(sub_value["data"])
                if False not in nans:
                    if verbose:
                        print("Could not interpolate field " + sub_key + ", in subject " + key + \
                              ", in category " + category + ", as all values were nans!")
                    if sanitize and sub_key != "":
                        sub_value["data"] = [0 for i in range(len(sub_value["data"]))]
                        if verbose:
                            print("Sanitizing field with all 0s...")
                        if category not in self._sanitized:
                            self._sanitized[category] = []
                        if key not in self._sanitized[category]:
                            self._sanitized[category].append(key)
                if category not in self._nan_dict:
                    self._nan_dict[category] = {}
                if key not in self._nan_dict[category]:
                    self._nan_dict[category][key] = {}
                self._nan_dict[category][key][sub_key] = self._false_of_n(len(sub_value["data"]))


    def set_marker_interpolation(self, method):
        assert issubclass(method, Interpolation.Interpolation)
        self.my_marker_interpolation = method

    def _prepare_interpolation(self, value, key, naninfo, category, interpolate, sanitize, verbose):
        for sub_key, sub_value in value.items():  # For each field under each subject...
            #  If we have NaNs and the whole row isn't NaNs...
            #  No interpolation method can do anything with an array of NaNs,
            #  so this way we save ourselves a bit of computation
            nans = np.isnan(sub_value["data"])
            if True in nans and False in nans and naninfo[key][sub_key]["interpolate"]:
                if category not in self._nan_dict:
                    self._nan_dict[category] = {}
                if key not in self._nan_dict[category]:
                    self._nan_dict[category][key] = {}
                self._nan_dict[category][key][sub_key] = nans
                if verbose:
                    print("Interpolating missing values in field " + sub_key + ", in subject " + key + \
                          ", in category " + category + "...")
                if interpolate:
                    sub_value["data"] = Interpolation.akmia(sub_value, verbose, category, sub_key, key)
            else:
                if False not in nans:
                    if verbose:
                        print("Could not interpolate field " + sub_key + ", in subject " + key + \
                              ", in category " + category + ", as all values were nans!")
                    if sanitize and sub_key != "":
                        sub_value["data"] = [0 for i in range(len(sub_value["data"]))]
                        if verbose:
                            print("Sanitizing field with all 0s...")
                        if category not in self._sanitized:
                            self._sanitized[category] = []
                        if key not in self._sanitized[category]:
                            self._sanitized[category].append(key)
                if category not in self._nan_dict:
                    self._nan_dict[category] = {}
                if key not in self._nan_dict[category]:
                    self._nan_dict[category][key] = {}
                self._nan_dict[category][key][sub_key] = self._false_of_n(len(sub_value["data"]))
