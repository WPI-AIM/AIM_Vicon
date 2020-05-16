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
import Accel
import EMG
import ForcePlate
import IMU
import ModelOutput
import Markers
import pandas
import numpy as np

class Vicon(object):

    def __init__(self, file_path, verbose=False):
        self._file_path = file_path
        self.joint_names = ["Ankle", "Knee", "Hip"]
        self._number_of_frames = 0
        self._T_EMGs = {}
        self._EMGs = {}
        self._force_plates = {}
        self._IMUs = {}
        self._accels = {}

        #  nan_dict is a dictionary with the same format of data_dict,
        #  but it keeps track of the positions of nans that get interpolated.

        #  data is accessed through nan_dict[category][subject][field],
        #  where nan_dict contains category iff there is at least one valid subject for the category,
        #  nan_dict[category] contains subject iff there is at least one valid field for the subject, and
        #  nan_dict[category][subject] contains field iff data_dict[category][subject][field]["data"] exists.

        #  nan_dict[category][subject][field] is a boolean array where
        #  nan_dict[category][subject][field] = np.isnan(data_dict[category][subject][field]["data"])
        #  or, nan_dict[category][subject][field][n] = np.isnan(data_dict[category][subject][field]["data"][n])
        #  iff data_dict[category][subject][field]["data"] contained missing values that were interpolated.

        #  if data_dict[category][subject][field]["data"] did not contain any missing values,
        #  or if data_dict[category][subject][field]["data"] consisted solely of nans,
        #  nan_dict[category][subject][field] is an array consisting only of False,
        #  where len(nan_dict[category][subject][field]) = len(data_dict[category][subject][field]["data"])

        self._nan_dict = {}

        self.data_dict = self.open_vicon_file(self._file_path, verbose)
        self._make_Accelerometers(verbose=verbose)
        self._make_EMGs(verbose=verbose)
        self._make_force_plates(verbose=verbose)
        self._make_IMUs(verbose=verbose)
        self._make_marker_trajs()
        self._make_model(verbose=verbose)

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

    @property
    def accels(self):
        """
        Get the Accels dict
        :return: Accels
        :type: dict
        """
        return self._accels

    @property
    def force_plate(self):
        """
         Get the force plate dict
         :return: Force plates
         :type: dict
        """
        return self._force_plates

    @property
    def IMUs(self):
        """
         Get the IMU dict
         :return: IMU
         :type: dict
        """
        return self._IMUs

    @property
    def T_EMGs(self):
        """
         Get the EMG dict
         :return: T EMG
         :type: dict
        """
        return self._T_EMGs

    @property
    def EMGs(self):
        """
        Get the EMGs dict
        :return: EMGs
        :type: dict
        """
        return self._EMGs

    def get_model_output(self):
        """
        get the model output
        :return: model outputs
        :rtype: ModelOutput.ModelOutput
        """
        return self._model_output

    def get_segments(self):
        """
        get the segments
        :return: model segments
        :type: dict
        """
        return self.data_dict["Segments"]

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

    def get_imu(self, index):
        """
        get the a imu
        :param index: imu number
        :return: imu
        :type: IMU.IMU
        """
        return self.IMUs[index]

    def get_accel(self, index):
        """
        get the a Accel
        :param index: Accel number
        :return: Accel
        :type: Accel.Accel
        """
        return self.accels[index]

    def get_force_plate(self, index):
        """
        get the a force plate
        :param index: force plate number
        :return: Force plate
        :type: ForcePlate.ForcePlate
        """
        return self.force_plate[index]

    def get_emg(self, index):
        """
       Get the EMG values
       :param index: number of sensor
       :return: EMG
       :rtype: EMG.EMG
        """
        return self._EMGs[index]

    def get_all_emgs(self):

        return self._EMGs

    def get_t_emg(self, index):
        """
        Get the T EMG values
        :param index: number of sensor
        :return: EMG
        :rtype: EMG.EMG
        """
        return self._T_EMGs[index]

    def get_all_t_emg(self, index):
        """
        Get the T EMG values
        :param index: number of sensor
        :return: EMG
        :rtype: EMG.EMG
        """
        return self._T_EMGs


    def _filter_dict(self, sensors, substring):
        """
        filter the dictionary
        :param sensors: Dictionary to parse
        :param substring: substring of the keys to look for in the dict
        :return: keys that contain the substring
        :type: list
        """
        return list(filter(lambda x: substring in x, sensors.keys()))


    def _make_model(self, verbose=False):
        """
        generates a model from the model outputs
        :return:
        """
        if "Model Outputs" in self.data_dict:
            self._model_output = ModelOutput.ModelOutput(self.data_dict["Model Outputs"], self.joint_names)
            if verbose:
                print "Model Outputs generated"
        elif verbose:
            print "No Model outputs"

    def _make_force_plates(self, verbose=False):
        """
        generate force plate models
        :return: None
        """
        if "Devices" in self.data_dict:

            sensors = self.data_dict["Devices"]
            keys = self._filter_dict(sensors, 'Force_Plate')  # + ['Combined Moment'] + ['Combined CoP']

            if any("Force_Plate" in word for word in keys) :
                self._force_plates[1] = ForcePlate.ForcePlate("Force_Plate_1",
                                                              sensors["Force_Plate__Force_1"],
                                                              sensors["Force_Plate__Moment_1"],
                                                              sensors["Force_Plate__CoP_1"])

                self._force_plates[2] = ForcePlate.ForcePlate("Force_Plate_2",
                                                              sensors["Force_Plate__Force_2"],
                                                              sensors["Force_Plate__Moment_2"],
                                                              sensors["Force_Plate__CoP_2"])
                print "Force plate models generated"
            elif verbose:
                print "No force plates"
        elif verbose:
            print "A scan for force plates found no Devices"

    def _make_markers(self):
        markers = self.data_dict["Trajectories"]

    def _make_EMGs(self, verbose=False):
        """
        generate EMG models
        :return: None
        """
        if "Devices" in self.data_dict:
            sensors = self.data_dict["Devices"]
            if "EMG" in sensors:
                all_keys = self._filter_dict(sensors, 'EMG')
                T_EMG_keys = self._filter_dict(sensors, 'T_EMG')
                EMG_keys = [x for x in all_keys if x not in T_EMG_keys]
                for e_key, t_key in zip(EMG_keys, T_EMG_keys):
                    self._T_EMGs[int(filter(str.isdigit, t_key))] = EMG.EMG(t_key, sensors[t_key]["EMG"])
                    self._EMGs[int(filter(str.isdigit, e_key))] = EMG.EMG(e_key, sensors[e_key]["IM EMG"])
                print "EMG models generated"
            elif verbose:
                print "No EMGs"
        elif verbose:
            print "A scan for EMGs found no Devices"

    def _make_IMUs(self, verbose=False):
        """
        generate IMU models
        :return: None
        """
        if "Devices" in self.data_dict:
            sensors = self.data_dict["Devices"]
            if "IMU" in sensors:
                keys = self._filter_dict(sensors, 'IMU')
                for key in keys:
                    self._IMUs[int(filter(str.isdigit, key))] = IMU.IMU(key, sensors[key])
                print "IMU models Generated"
            elif verbose:
                print "No IMUs"
        elif verbose:
            print "A scan for IMUs found no Devices"

    def _make_marker_trajs(self):
        """
        generate IMU models
        :return: None
        """
        self._markers = Markers.Markers(self.data_dict["Trajectories"])
        self._markers.make_markers()

    def _make_Accelerometers(self, verbose=False):
        """
        generate the accel objects
        :return: None
        """
        if "Devices" in self.data_dict:
            sensors = self.data_dict["Devices"]
            if "Accel" in sensors:
                keys = self._filter_dict(sensors, 'Accel')
                for key in keys:
                    self._accels[int(filter(str.isdigit, key))] = Accel.Accel(key, sensors[key])
                print "Accel models generated"
            elif verbose:
                print "No Accels"
        elif verbose:
            print "A scan for Accels found no Devices"

    def open_vicon_file(self, file_path, verbose=False):
        """
        parses the Vicon sensor data into a dictionary
        :param file_path: file path
        :param verbose: prints debug statements if True
        :return: dictionary of the sensors
        :rtype: dict
        """
        # open the file and get the column names, axis, and units
        with open(file_path, mode='r') as csv_file:
            reader = csv.reader(csv_file)
            raw_data = list(reader)

        # output_names = ["Devices", "Joints", "Model Outputs", "Segments", "Trajectories"]
        data = {}
        names, segs = self._seperate_csv_sections(raw_data)

        for index, output in enumerate(names):
            data[output] = self._extract_values(raw_data, segs[index], segs[index + 1], verbose=verbose, category=output)

        return data

    def _seperate_csv_sections(self, all_data):
        """"""

        raw_col = [row[0] for row in all_data]
        fitlered_col = [item for item in raw_col if not item.isdigit()]
        fitlered_col = filter(lambda a: a != 'Frame', fitlered_col)
        fitlered_col = filter(lambda a: a != "", fitlered_col)

        if 'Devices' in fitlered_col:
            fitlered_col = fitlered_col[fitlered_col.index("Devices"):]

        inx = []
        for name in fitlered_col:
            inx.append(raw_col.index(name))

        inx.append(len(raw_col))
        return fitlered_col, inx

    def  _fix_col_names(self, names):
        fixed_names = []
        get_index = lambda x: x.index("Sensor") + 7

        for name in names:  # type: str

            # if "Subject".upper() in name.upper():
            #     fixed = ''.join(
            #         [i for i in name.replace("Subject", "").replace(":", "").replace("|", "") if
            #          not i.isdigit()]).strip()
            #     fixed_names.append(fixed)

            if ":" in name:

                index = name.index(":")

                fixed_names.append(name[index+1:])

            elif "AMTI" in name:

                if "Force" in name:
                    unit = "_Force_"
                elif "Moment" in name:
                    unit = "_Moment_"
                elif "CoP" in name:
                    unit = "_CoP_"

                number = name[name.find('#') + 1]
                fixed = "Force_Plate_" + unit + str(number)
                fixed_names.append(fixed)

            elif "Trigno EMG" in name:
                fixed = "T_EMG_" + name[-1]
                fixed_names.append(fixed)

            elif "Accelerometers" in name:
                fixed = "Accel_" + name[get_index(name):]
                fixed_names.append(fixed)

            elif "IMU AUX" in name:
                fixed = "IMU_" + name[get_index(name):]
                fixed_names.append(fixed)

            elif "IMU EMG" in name:
                fixed = "EMG_" + name[get_index(name):]
                fixed_names.append(fixed)
            else:
                fixed_names.append(name)

        return fixed_names

    def _extract_values(self, raw_data, start, end, verbose=False, category=""):
        indices = {}
        data = {}
        current_name = None
        last_frame = None

        column_names = self._fix_col_names(raw_data[start + 2])

        # column_names = raw_data[start + 2]
        remove_numbers = lambda str: ''.join([i for i in str if not i.isdigit()])

        axis = map(remove_numbers, raw_data[start + 3])
        unit = raw_data[start + 4]

        # Build the dict to store everything
        for index, name in enumerate(column_names):

            if index <= 1:
                continue
            else:
                if len(name) > 0:
                    current_name = name

                    data[current_name] = {}
                dir = axis[index]
                indices[(current_name, dir)] = index
                data[current_name][dir] = {}
                data[current_name][dir]["data"] = []
                data[current_name][dir]["unit"] = unit[index]

        # Put all the data in the correct sub dictionary.

        flags = []
        for row in raw_data[start + 5:end - 1]:

            frame = int(row[0])
            for key, value in data.iteritems():
                for sub_key, sub_value in value.iteritems():
                    index = indices[(key, sub_key)]
                    if row[index] is '':
                        val = np.nan
                    elif '!' in row[index]:
                        val = float(row[index][1:])
                        if verbose and (key, sub_key) not in flags:
                            print "Reading previously interpolated data in category " + category +\
                                  ", subject " + key + ", field " + sub_key + "."
                            flags.append((key, sub_key))
                    else:
                        val = float(row[index])
                    sub_value["data"].append(val)
        for key, value in data.iteritems():  # For every subject in the data...
            for sub_key, sub_value in value.iteritems():  # For each field under each subject...
                #  If we have NaNs and the whole row isn't NaNs...
                #  No interpolation method can do anything with an array of NaNs,
                #  so this way we save ourselves a bit of computation
                nans = np.isnan(sub_value["data"])
                if True in nans and False in nans:
                    if category not in self._nan_dict:
                        self._nan_dict[category] = {}
                    if key not in self._nan_dict[category]:
                        self._nan_dict[category][key] = {}
                    self._nan_dict[category][key][sub_key] = nans
                    if verbose:
                        print "Interpolating missing values in field " + sub_key + ", in subject " + key +\
                              ", in category " + category + "..."
                    s = pandas.Series(sub_value["data"])
                    #  Akima interpolation only covers interior NaNs,
                    #  and splines are *way* too imprecise with unset boundary conditions,
                    #  so linear interpolation is used for unset values at the edges
                    s = s.interpolate(method='akima', limit_direction='both')
                    s = s.interpolate(method='linear', limit_direction='both')
                    sub_value["data"] = s.to_list()
                else:
                    if category not in self._nan_dict:
                        self._nan_dict[category] = {}
                    if key not in self._nan_dict[category]:
                        self._nan_dict[category][key] = {}
                    self._nan_dict[category][key][sub_key] = self._false_of_n(len(sub_value["data"]))

        return data

    def _false_of_n(self, n):
        """Helper function to generate an array of Falses of length N"""
        arr = []
        for i in range(n):
            arr.append(False)
        return arr

    def save(self, filename=None, verbose=False, mark_interpolated=True):
        file_path = self._file_path
        if filename is not None:
            file_path = filename
        if verbose and mark_interpolated:
            print "Saving data to " + file_path + ". Interpolated values will be marked with '!'."
        if verbose and not mark_interpolated:
            print "Saving data to " + file_path + ". Interpolated values will not be marked."




if __name__ == '__main__':
    file = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_08/subject_08_walking_01.csv"
    data = Vicon(file)

