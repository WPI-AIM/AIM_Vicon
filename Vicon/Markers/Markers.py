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
#     \author    <ajlewis@wpi.edu>
#     \author    Alek Lewis
#     \version   0.2
# */
# //==============================================================================

from GaitCore import Core as core
import numpy as np
from scipy.optimize import minimize
import math
from numpy import *
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import csv, os

import matplotlib.animation as animation


class Markers(object):
    """
    Creates an object to hold marker values
    """

    def __init__(self, marker_dict, dat_name):
        """

        :param marker_dict: dict of markers
        """
        self._data_dict = marker_dict
        self._raw_markers = {}
        self._rigid_body = {}
        self._marker_names = []
        self._frames = {}
        self._filter_window = 10
        self._filtered_markers = {}
        self._joints = {}
        self._joints_rel = {}
        self._joints_rel_child = {}
        self._balljoints_def = {}
        self._hingejoints_def = {}
        self._dat_name = dat_name

    @property
    def marker_names(self):
        """
        :return marker names
        :return: None
        """
        return self._marker_names

    @property
    def filter_window(self):
        return self._filter_window

    @filter_window.setter
    def filter_window(self, value):
        self._filter_window = value

    @property
    def filtered_markers(self):
        return self._filtered_markers

    @filtered_markers.setter
    def filtered_markers(self, value):
        self._filtered_markers = value

    @property
    def rigid_body(self):
        return self._rigid_body

    @filtered_markers.setter
    def rigid_body(self, value):
        """

        :param value: value to set rigid marker
        :return: rigid marker
        """
        self._rigid_body = value

    def get_marker(self, key):
        """

        :param key: name of the marker key
        :return: the value
        """
        return self._filtered_markers[key]

    def get_marker_keys(self):
        """

        :param key: name of the marker key
        :return: the value
        """
        return self._filtered_markers.keys()

    def make_markers(self):
        """
        Convert the dictioanry into something a that can be easy read
        :return:
        """

        # TODO need to ensure that the frame are being created correctly and fill in missing data with a flag

        to_remove = [item for item in self._data_dict.keys() if "|" in item]
        to_remove += [item for item in self._data_dict.keys() if "Trajectory Count" == item]
        for rr in to_remove:
            self._data_dict.pop(rr, None)

        for key_name, value_name in self._data_dict.items():
            fixed_name = key_name[1 + key_name.find(":"):]
            self._marker_names.append(fixed_name)
            self._raw_markers[fixed_name] = []
            self._filtered_markers[fixed_name] = []

            # This removes some of the values that are not very useful
            # if value_name.keys()[0] == "Magnitude( X )" or value_name.keys()[0] == "Count":
            #     continue

            if "Magnitude( X )" in value_name.keys() or "Count" in value_name.keys():
                continue

            x_arr = value_name["X"]["data"]
            y_arr = value_name["Y"]["data"]
            z_arr = value_name["Z"]["data"]

            # smooth the markers
            x_filt = np.convolve(x_arr, np.ones((self._filter_window,)) / self._filter_window, mode='valid')
            y_filt = np.convolve(y_arr, np.ones((self._filter_window,)) / self._filter_window, mode='valid')
            z_filt = np.convolve(z_arr, np.ones((self._filter_window,)) / self._filter_window, mode='valid')

            # save a copy of both the unfiltered and fitlered markers
            for inx in range(len(x_filt)):
                point = core.Point.Point(x_arr[inx], y_arr[inx], z_arr[inx])
                self._raw_markers[fixed_name].append(point)
                point = core.Point.Point(x_filt[inx], y_filt[inx], z_filt[inx])
                self._filtered_markers[fixed_name].append(point)

    def set_ground_plane(self, rigid_body, offset_height=14):

        markers = self.get_rigid_body("marker_names")
        fit, residual = fit_to_plane(markers)

        pass

    def _is_valid_marker(self, name):
        """If a marker consists of all zero points, it does not contain valid data,
         and thus should not be included in the rigid body."""
        for point in self.get_marker(name):
            if not self._is_z_point(point):
                return True
        return False

    def _is_z_point(self, point):
        return point.x == 0 and point.y == 0 and point.z == 0

    def smart_sort(self, filter=True):
        """
        Gather all the frames and attempt to sort the markers into the rigid markers
        :param filter: use the filtered values or the none fitlered values
        :return:
        """
        no_digits = [''.join(x for x in i if not x.isdigit()) for i in self._marker_names]  # removes digits
        single_item = list(set(no_digits))  # removes redundent items
        keys = self._marker_names

        for name in single_item:
            markers_keys = [s for s in keys if name in s]
            markers_keys.sort()
            markers = []
            for marker in markers_keys:
                if self._is_valid_marker(marker):
                    if filter:
                        markers.append(self._filtered_markers[marker])
                    else:
                        markers.append(self._raw_markers[marker])
            self._rigid_body[name] = markers

    def make_frame(self, _origin, _x, _y, _extra):
        """

        :param _origin:
        :param _x:
        :param _y:
        :param _extra:
        :return:
        """
        frames = []
        for o_ii, x_ii, y_ii in zip(_origin, _x, _y):
            o = np.array([o_ii.x, o_ii.y, o_ii.z]).transpose()
            x = np.array([x_ii.x, x_ii.y, x_ii.z]).transpose()
            y = np.array([y_ii.x, y_ii.y, y_ii.z]).transpose()
            xo = (x - o) / np.linalg.norm(x - o)
            yo = (y - o) / np.linalg.norm(y - o)
            zo = np.cross(xo, yo)
            xo = np.pad(xo, (0, 1), 'constant')
            yo = np.pad(yo, (0, 1), 'constant')
            zo = np.pad(zo, (0, 1), 'constant')
            p = np.pad(o, (0, 1), 'constant')
            p[-1] = 1
            F = np.column_stack((xo, yo, zo, p))
            frames.append(F)
        return frames

    def add_frame(self, name, frame):
        """
        add a fame to a the dictionary
        :param name: name for the dictionary
        :param frame: frame to add the dictionary
        :return:
        """

        self._frames[name] = frame

    def auto_make_frames(self):
        """
        Auto make all the frames based on the order of the markers
        :return:
        """
        for name, value in self._rigid_body.items():
            frame = self.make_frame(value[0], value[1], value[2], value[3])
            self.add_frame(name, frame)

    def auto_make_transform(self, bodies):
        """
        make the frames using the cloud method
        :param bodies: list transformation
        :return:
        """
        for name, value in self._rigid_body.items():
            frames = []
            if name in bodies:
                for ii in range(len(value[0])):
                    frames.append(
                        cloud_to_cloud(bodies[name], [value[0][ii], value[1][ii], value[2][ii], value[3][ii]])[0])
                self.add_frame(name, frames)

    def def_joint(self, name, parentBody, childBody, ballJoint=True):

        try:
            assert parentBody in self.get_rigid_body_keys()
            assert childBody in self.get_rigid_body_keys()
        except AssertionError:
            raise ValueError("At least one specified rigid body is not present in this data!")

        if ballJoint:
            self._balljoints_def[name] = [parentBody, childBody]
        else:
            self._hingejoints_def[name] = [parentBody, childBody]

    def calc_joints(self, try_load=True, verbose=False, strict=True):
        """
        Calculates all defined joints automatically
        :return:
        """

        if try_load and os.path.isfile(self._dat_name + "-joints.csv"):
            self.read_joints(verbose=verbose, strict=strict)
            if verbose:
                print("Reading joints from file " + self._dat_name + "-joints.csv")

        else:
            if verbose:
                print("Manually calculating joints")
            for name, bodies in self._balljoints_def.items():
                (self._joints[name], self._joints_rel[name], self._joints_rel_child[name]) = self._calc_ball_joint(bodies[0], bodies[1])

            for name, bodies in self._hingejoints_def.items():
                (self._joints[name], self._joints_rel[name], self._joints_rel_child[name]) = self._calc_hinge_joint(bodies[0], bodies[1])

    def get_joint(self, name):
        """
        Get a joint
        :param name: Name of the joint
        :return: joint
        """
        return self._joints[name]

    def get_joint_rel(self, name):
        return self._joints_rel[name]

    def get_joint_rel_child(self, name):
        return self._joints_rel_child[name]

    def set_joints(self, j):
        self._joints = j

    def get_joints(self):
        return self._joints

    def get_joints_rel(self):
        return self._joints_rel

    def get_joints_rel_child(self):
        return self._joints_rel_child

    def dist_joints(self, a, b):
        return [dist(self.get_joint(a)[n], self.get_joint(b)[n]) for n in range(len(self._joints[a]))]

    def body_rel_body(self, a, b, t):
        return global_point_to_frame(self.get_frame(b)[t], local_point_to_global(self.get_frame(a)[t], core.Point.vector_to_point([[0], [0], [0]])))

    def body_centroid(self, a, t):
        pos = [self.get_rigid_body(a)[n][t] for n in range(4)]
        pos = [[n.x, n.y, n.z] for n in pos]
        return [avg([n[m] for n in pos]) for m in range(3)]

    def limb_len(self, a, b):
        len_raw = self.dist_joints(a, b)
        x = 0
        for i in len_raw:
            x += i
        return x/len(len_raw)

    def get_frame(self, name):
        """
        get a frame
        :param name: name of the frame
        :return: frame
        """
        return self._frames[name]

    def get_rigid_body(self, name):
        """

        :param name: name of rigid body
        :return: transformation of the rigid body
        """

        return self._rigid_body[name]

    def get_rigid_body_keys(self):
        """

        :param name: name of rigid body
        :return: transformation of the rigid body
        """

        return self._rigid_body.keys()

    def calc_joint_center(self, parent_name, child_name, start, end):
        """
        Calculate the joint center between two frames
        :param child_name:
        :param start:
        :param end:
        :return:
        """

        Tp = self.get_frame(parent_name)[start:end]
        m1 = self.get_rigid_body(child_name)[0][start:end]
        m2 = self.get_rigid_body(child_name)[1][start:end]
        m3 = self.get_rigid_body(child_name)[2][start:end]
        m4 = self.get_rigid_body(child_name)[3][start:end]
        m = [m1, m2, m3, m4]

        global_joint = calc_CoR(m)

        axis = calc_AoR(m)
        local_joint = np.array([[0.0], [0.0], [0.0], [0.0]])

        for T in Tp:
            local_joint += transform_vector(np.linalg.pinv(T), global_joint) / len(Tp)

        return np.vstack((global_joint, [1])), axis, local_joint

    def _calc_ball_joint(self, parent, child_name):
        """
        Calculates the location of a ball joint between the parent and child rigid bodies.
        :param parent: Name of the parent rigid body (Ex: Root)
        :param child: Name of the child rigid body (Ex: L_Femur)
        :return: Tuple where first element is (x,y,z) in global reference for all frames and second element is (x,y,z) in parent reference for all frames
        """
        child = self.get_rigid_body(child_name)
        parent_frame = self.get_frame(parent)
        frames = len(child[0])

        # Obtain the locations of the child markers relative to the Parent rigid body
        child_by_parent = [[global_point_to_frame(parent_frame[n], child[i][n]) for n in range(frames)] for i in
                           range(4)]
        # Identical to child except for the difference in frame of reference
        # Points are accessed through child_by_parent[marker][frame]

        # Calculate the location of the center of rotation of child relative to the parent, over the entire dataset
        jointraw = calc_CoR(child_by_parent)
        # calc_CoR requires that the moving body be rotating around a stationary ball joint.
        # By switching to the parent's reference frame, we can pretend as if the joint is stationary

        joint = []
        joint_by_child = core.Point.point_to_vector(global_point_to_frame(
            self.get_frame(child_name)[0], local_point_to_global(parent_frame[0], core.Point.vector_to_point(jointraw))))
        for n in range(frames):  # Convert back from the hip's frame to the global frame
            jointn_global = local_point_to_global(parent_frame[n], core.Point.vector_to_point(jointraw))
            joint.append([jointn_global.x, jointn_global.y, jointn_global.z])

        return joint, jointraw, joint_by_child

    def _calc_hinge_joint(self, parent, child_name):
        """
        Calculates the location of a hinge joint (i.e. a knee joint) between the parent and child rigid bodies
        :param parent: Name of the parent rigid body (Ex: L_Femur)
        :param child: Name of the child rigid body (Ex: L_Tibia)
        :return: 2D array of every position of the center of the hinge joint for every frame. Array at index frame is [x, y, z].
        """
        child = self.get_rigid_body(child_name)
        parent_frame = self.get_frame(parent)
        parent = self.get_rigid_body(parent)

        frames = len(child[0])

        child_by_parent = [[global_point_to_frame(parent_frame[n], child[i][n]) for n in range(frames)] for i in range(4)]
        parent_by_parent = [[global_point_to_frame(parent_frame[n], parent[i][n]) for n in range(frames)] for i in range(4)]

        # calc_CoR isn't meant to be used with hinge joints
        # It still gives us a point, but the location of this joint is poorly defined along the axis of rotation
        jointraw = calc_CoR(child_by_parent)

        # calc_AoR gives us the slope of the axis of rotation
        # Between this and calc_CoR, we can define the line representing the AoR
        # From there, finding the actual joint is just a matter of finding the point on the line
        # that is the closest to both the child and the parent
        jointaxisraw = calc_AoR(child_by_parent)

        joint_by_parent = []
        for frame in range(frames):  # Find the location on the AoR that is closest to both relevant rigid bodies
            l_parentn_local = [parent_by_parent[n][frame] for n in range(4)]
            l_parentn_local = [[n.x, n.y, n.z] for n in l_parentn_local]

            l_childn_local = [child_by_parent[n][frame] for n in range(4)]
            l_childn_local = [[n.x, n.y, n.z] for n in l_childn_local]

            l_jointn_local = minimize_center(l_childn_local+l_parentn_local, jointaxisraw, [jointraw[0][0], jointraw[1][0], jointraw[2][0]])
            joint_by_parent.append(l_jointn_local)

        joint = []
        joint_by_child = core.Point.point_to_vector(global_point_to_frame(
            self.get_frame(child_name)[0], local_point_to_global(parent_frame[0], core.Point.vector_to_point(jointraw))))
        for n in range(frames):
            p = core.Point.Point(joint_by_parent[n].x[0], joint_by_parent[n].x[1], joint_by_parent[n].x[2])
            l_jointn_global = local_point_to_global(parent_frame[n], p)
            joint.append([l_jointn_global.x, l_jointn_global.y, l_jointn_global.z])

        return joint, [[np.mean([joint_by_parent[n].x[0] for n in range(frames)])],
                       [np.mean([joint_by_parent[n].x[1] for n in range(frames)])],
                       [np.mean([joint_by_parent[n].x[2] for n in range(frames)])]], joint_by_child

    def play(self, joints=False, save=False, name="im", center=False):
        """
        play an animation of the         markers
        :param joints: opital param for joint centers
        :param save: bool to save the animation
        :param center: bool to keep the markers centered
        :return: name of file
        """

        x_total = []
        y_total = []
        z_total = []
        joints_points = []
        fps = 10  # Frame per sec
        keys = self._filtered_markers.keys()
        nfr = len(self._filtered_markers[list(keys)[0]])  # Number of frames
        root0z0 = self._filtered_markers["Root0"][0].z

        for frame in range(nfr):
            x = []
            y = []
            z = []
            for key in keys:
                if self._is_valid_marker(key):
                    point = self._filtered_markers[key][frame]
                    if not center:
                        x += [point.x]
                        y += [point.y]
                        z += [point.z]
                    else:
                        root0 = self._filtered_markers["Root0"][frame]
                        x += [point.x - root0.x]
                        y += [point.y - root0.y]
                        z += [point.z - root0.z + root0z0]
            if len(x) > 0:
                x_total.append(x)
                y_total.append(y)
                z_total.append(z)
            x = []
            y = []
            z = []
            if joints:
                for jointname, joint in self._joints.items():
                    f = frame
                    if frame >= len(joint):
                        f = len(joint) - 1

                    if not center:
                        x.append(joint[frame][0])
                        y.append(joint[frame][1])
                        z.append(joint[frame][2])
                    else:
                        root0 = self._filtered_markers["Root0"][frame]
                        x.append(joint[frame][0] - root0.x)
                        y.append(joint[frame][1] - root0.y)
                        z.append(joint[frame][2] - root0.z + root0z0)
                joints_points.append([x, y, z])

        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111, projection='3d')
        self._ax.set_autoscale_on(False)

        ani = animation.FuncAnimation(self._fig,
                                      self.__animate, nfr,
                                      fargs=(x_total, y_total, z_total, joints_points),
                                      interval=100 / fps)
        if save:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            ani.save(name + '.mp4', writer=writer)
            plt.show()
        else:
            plt.show()

    def __animate(self, frame, x, y, z, centers=None):
        """

        :param frame: interation frame
        :param x:array of x data
        :param y: array of y data
        :param z: array of z data
        :param centers:optioanl data
        :return:
        """

        self._ax.clear()
        self._ax.set_xlabel('X Label')
        self._ax.set_ylabel('Y Label')
        self._ax.set_zlabel('Z Label')
        self._ax.axis([-500, 500, -500, 500])
        self._ax.set_zlim3d(0, 1250)
        self._ax.scatter(x[frame], y[frame], z[frame], c='r', marker='o')
        if len(centers) > 0:
            self._ax.scatter(centers[frame][0], centers[frame][1], centers[frame][2], c='g', marker='o')

    def save_joints(self, verbose=False, jlim=None, doBall=True, doHinge=True):
        file_path = self._dat_name + "-joints.csv"
        if verbose:
            print("Saving joints data to " + file_path)

        with open(file_path, "w", newline='') as f:
            f.seek(0)
            f.truncate()
            writer = csv.writer(f)
            writer.writerow(["Ball Joints"])
            for name, bodies in self._balljoints_def.items():
                if doBall and (jlim is None or (jlim is not None and name in jlim)):
                    writer.writerow([name, bodies[0], bodies[1]])
                    writer.writerow(["rel_parent"])
                    writer.writerow([n[0] for n in self.get_joint_rel(name)])
                    writer.writerow(["rel_child"])
                    writer.writerow([n[0] for n in self.get_joint_rel_child(name)])
                    writer.writerow(["pos", len(self.get_joint(name))])
                    for timestep in self.get_joint(name):
                        writer.writerow(timestep)
            writer.writerow(["Hinge Joints"])
            for name, bodies in self._hingejoints_def.items():
                if doHinge and (jlim is None or (jlim is not None and name in jlim)):
                    writer.writerow([name, bodies[0], bodies[1]])
                    writer.writerow(["rel_parent"])
                    writer.writerow([n[0] for n in self.get_joint_rel(name)])
                    writer.writerow(["rel_child"])
                    writer.writerow([n[0] for n in self.get_joint_rel_child(name)])
                    writer.writerow(["pos", len(self.get_joint(name))])
                    for timestep in self.get_joint(name):
                        writer.writerow(timestep)

    def read_joints(self, filename=None, verbose=False, strict=False):
        if filename is None:
            filename = self._dat_name + "-joints.csv"

        if verbose:
            print("Reading joint data from " + filename)

        with open(filename, "r", newline='') as f:
            reader = csv.reader(f)
            next(reader)
            while True:
                try:
                    name, parent, child = next(reader)
                except ValueError:
                    break
                if parent not in self.get_rigid_body_keys() or child not in self.get_rigid_body_keys():
                    if strict:
                        raise ValueError("A rigid body present in this file isn't present in this spreadsheet!")
                    if verbose:
                        print("WARNING: A rigid body present in this file isn't present in this spreadsheet!")
                if verbose:
                    print("Detected ball joint " + name)

                self._balljoints_def[name] = [parent, child]
                next(reader)
                self._joints_rel[name] = [[float(n)] for n in next(reader)]
                next(reader)
                self._joints_rel_child[name] = [[float(n)] for n in next(reader)]

                self._joints[name] = []
                l = int(next(reader)[1])
                for i in range(l):
                    self._joints[name].append([float(n) for n in next(reader)])

            while True:
                try:
                    name, parent, child = next(reader)
                except:
                    break
                if parent not in self.get_rigid_body_keys() or child not in self.get_rigid_body_keys():
                    if strict:
                        raise ValueError("A rigid body present in this file isn't present in this spreadsheet!")
                    if verbose:
                        print("WARNING: A rigid body present in this file isn't present in this spreadsheet!")
                if verbose:
                    print("Detected hinge joint " + name)

                self._hingejoints_def[name] = [parent, child]
                next(reader)
                self._joints_rel[name] = [[float(n)] for n in next(reader)]
                next(reader)
                self._joints_rel_child[name] = [[float(n)] for n in next(reader)]

                self._joints[name] = []
                l = int(next(reader)[1])
                for i in range(l):
                    self._joints[name].append([float(n) for n in next(reader)])


def avg(n):
    return sum(n)/len(n)



def transform_markers(transforms, markers):
    """

    :param transforms:
    :param markers:
    :return:
    """
    trans_markers = []
    for marker in markers:
        adjusted_locations = []
        for transform, frame in zip(transforms, marker):
            v = np.array([[frame.x, frame.y, frame.z, 1.0]]).T
            v_prime = np.dot(transform, v)
            new_marker = core.Point.Point(v_prime[0][0], v_prime[1][0], v_prime[2][0])
            adjusted_locations.append(new_marker)
        trans_markers.append(adjusted_locations)
    return trans_markers


def global_to_frame(frame, global_vector):
    """Transforms a vector in the global frame to the provided frame."""
    return transform_vector(np.linalg.pinv(frame), global_vector)


def global_point_to_frame(frame, global_point):
    """Transforms a Point object from the global frame to the provided frame."""
    return core.Point.vector_to_point(global_to_frame(frame, core.Point.point_to_vector(global_point)))


def frame_to_global(frame, local_vector):
    """Transforms a vector in the provided local frame to the global frame."""
    return transform_vector(frame, local_vector)


def local_point_to_global(frame, local_point):
    """Transforms a Point object in the provided local frame to the global frame."""
    return core.Point.vector_to_point(frame_to_global(frame, core.Point.point_to_vector(local_point)))


def dist(x, y):
    return ((x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2)**0.5

def make_frame(markers):
    """
    Create a frame based on the marker layout
    Creates a marker assuming a certain layout,
    DEPTRICATED
    :param markers: array of markers
    :return:
    """
    origin = markers[0]
    x_axis = markers[1]
    y_axis = markers[2]

    xo = origin - x_axis
    yo = origin - y_axis
    zo = np.cross(xo, yo)
    xo = np.pad(xo, (0, 1), 'constant')
    yo = np.pad(yo, (0, 1), 'constant')
    zo = np.pad(zo, (0, 1), 'constant')
    p = np.pad(origin, (0, 1), 'constant')
    p[-1] = 1
    return np.column_stack((xo, yo, zo, p))


def get_all_transformation_to_base(parent_frames, child_frames):
    """

    :type world_to_base_frame: np.array
    :param world_to_base_frame:
    :param body_frames:
    :return:
    """

    frames = []
    for parent, child in zip(parent_frames, child_frames):
        frames.append(get_transform_btw_two_frames(parent, child))

    return frames


def get_transform_btw_two_frames(parent_frame, child_frame):
    """

    :param parent_frame: parent frame
    :param child_frame: child frame
    :type parent_frame: np.array
    :type child_frame: np.array
    :return: transformation frame
    """
    return np.linalg.inv(parent_frame) * child_frame


def get_angle_between_vects(v1, v2):
    """
    returns the angle between two vectors
    https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
    :param v1: vector 1
    :param v2: vector 2
    :return: cross product of the vectors
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def transform_vector(frame, vector):
    """
    transform a vector from one frame to another
    :param frame:
    :param vector:
    :return:
    """
    p = np.vstack((vector, [1]))
    return np.dot(frame, p)


def batch_transform_vector(frames, vector):
    """

    :param frames: list of frames
    :param vector: vector to transform
    :return:
    """

    trans_vectors = []

    for T in frames:
        p = np.dot(T, vector)
        # p = np.dot(np.eye(4), vector)
        trans_vectors.append(p)

    return trans_vectors


def unit_vector(vector):
    """
    Returns the unit vector of the vector.
    :param vector: list of points
    :return: norm of points
    :type vector: np.array
    """
    return vector / np.linalg.norm(vector)


def avg_vector(markers):
    """
    averages the marker location based
    :param markers: a marker
    :return: norm of all the markers
    """
    vp_norm = []
    for marker in markers:
        vp = np.array((0.0, 0.0, 0.0))
        for point in marker:
            vp = vp + np.array((point.x, point.y, point.z))
        vp /= len(marker)
        vp_norm.append(vp)
    return vp_norm


def calc_CoR(markers):
    '''
        Calculate the center of rotation given two data
        sets representing two frames on separate rigid bodies connected by a
        spherical joint. The function calculates the position of the CoR in the
        reference rigidi body frame
        For more information on this derivation see "New Least Squares Solutions
        for Estimating the Average Centre of Rotation and the Axis of Rotation"
        by Sahan S. Hiniduma
    '''

    A = __calc_A(markers)
    b = __calc_b(markers)
    Ainv = np.linalg.pinv(2.0 * A)
    return np.dot(Ainv, b)


def calc_AoR(markers):
    """
        Calculate the center of rotation given two data
        sets representing two frames on separate rigid bodies connected by a
        spherical joint. The function calculates the position of the CoR in the
        reference rigidi body frame
        For more information on this derivation see "New Least Squares Solutions
        for Estimating the Average Centre of Rotation and the Axis of Rotation"
        by Sahan S. Hiniduma


    :type markers: list
    :param markers: list of markers, each marker is a list of core.Exoskeleton.Points
    :return: axis of rotation
    :rtype np.array
    """
    A = __calc_A(markers)  # calculates the A matrix
    E_vals, E_vecs = np.linalg.eig(
        A)  # I believe that the np function eig has a different output than the matlab function eigs
    min_E_val_idx = np.argmin(E_vals)
    axis = E_vecs[:, min_E_val_idx]
    return axis


def __calc_A(markers):
    """

    :param markers: array of markers
    :return: A array
    """

    A = np.zeros((3, 3))
    vp_norm = avg_vector(markers)
    for marker, vp_n in zip(markers, vp_norm):  # loop though each marker
        Ak = np.zeros((3, 3))
        for point in marker:  # go through is location of the marker
            v = np.array((point.x, point.y, point.z))
            Ak = Ak + v.reshape((-1, 1)) * v
        Ak = (1.0 / len(marker)) * Ak - vp_n.reshape((-1, 1)) * vp_n
        A = A + Ak
    return A


def __calc_b(markers):
    """
    Function to work with calc_AoR and calc_CoR

    :param markers: array of markers
    :return: b array
    """
    b = np.array((0.0, 0.0, 0.0))
    vp_norm = avg_vector(markers)
    for ii, marker in enumerate(markers):
        invN = 1.0 / len(marker)
        v2_sum = 0
        v3_sum = np.array((0.0, 0.0, 0.0))
        for point in marker:
            # v = np.array((point.x, point.y, point.z))
            # print np.dot(v.reshape((-1,1)),v.reshape((-1,1)))
            v2 = (point.x * point.x + point.y * point.y + point.z * point.z)
            v2_sum = v2_sum + invN * v2
            v3_sum = v3_sum + invN * (v2 * np.array((point.x, point.y, point.z)))
        b = b + v3_sum - v2_sum * vp_norm[ii]

    return b.reshape((-1, 1))


def cloud_to_cloud(A_, B_):
    """
    Get the transformation between two frames of marker sets.
    http://nghiaho.com/?page_id=671
    :param A_: rigid body markers set
                Example: [core.Point(0.0, 50.0, 0.0),
                          core.Point(-70.0, 50.0, 0.0),
                          core.Point(-70, 0, 0),
                          core.Point(0.0, 50.0, 100.0)]

    :param B_: currnet position of the markers
    :return: transformation matrix and RSME error
    """
    A = np.asmatrix(points_to_matrix(A_))
    B = np.asmatrix(points_to_matrix(B_))

    assert len(A) == len(B)

    N = A.shape[0];  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T * U.T

    p = -R * centroid_A.T + centroid_B.T

    A2 = (R * A.T) + np.tile(p, (1, N))
    A2 = A2.T
    err = A2 - B
    err = np.multiply(err, err)
    err = sum(err)
    rmse = sqrt(err / N)

    T = np.zeros((4, 4))
    T[:3, :3] = R
    for ii in range(3):
        T[ii, 3] = p[ii]
    T[3, 3] = 1.0

    return T, rmse


def get_center(markers, R):
    """
    Get the marker set
    :param markers: List of markers
    :param R: Rotation matix
    :return: transformed point
    """

    x1 = np.array((markers[0][0].x, markers[0][0].y, markers[0][0].z)).reshape((-1, 1))
    x2 = np.array((markers[1][0].x, markers[1][0].y, markers[1][0].z)).reshape((-1, 1))
    xc = -np.dot(np.linalg.pinv(R + np.eye(3)), (x2 - np.dot(R, x1)))

    return xc


def minimize_center(vectors, axis, initial):
    """
    Optimize the center of the rotation of the axis by finding the closest point
    in the line to the frames. This method uses a lest squares operation
    :param vectors: list of vector
    :param axis: axis to search on
    :param initial: interital guess
    :type vectors: List
    :type axis: List
    :type initial: List
    :return: optimized center
    """

    def objective(x):
        C = 0
        for vect in vectors:
            C += np.sqrt(np.power((x[0] - vect[0]), 2) + np.power((x[1] - vect[1]), 2) + np.power((x[2] - vect[2]), 2))
        return C

    def constraint(x):
        return np.array((x[0], x[1], x[2])) - x[3] * axis - initial

    N = 1000
    b = (-N, N)
    bnds = (b, b, b, b)
    con = {'type': 'eq', 'fun': constraint}
    cons = ([con])
    solution = minimize(objective, np.append(initial, 0), method='SLSQP', \
                        bounds=bnds, constraints=cons)
    return solution


def calc_mass_vect(markers):
    """
    find the average vector to  frame
    :param markers: list of points
    :return: center of the markers
    """
    x = 0
    y = 0
    z = 0
    for point in markers:
        x += point.x
        y += point.y
        z += point.z

    vect = np.array((x / len(markers),
                     y / len(markers),
                     z / len(markers)))
    return vect


def calc_vector_between_points(start_point, end_point):
    """
    calculate the vector between two points
    :param start_point: first point
    :param end_point: sencound point
    :return:
    """
    return end_point - start_point


def get_distance(point1, point2):
    """
    Get the distance between two points
    :type point1: Point
    :type point2l Point
    :param point1: first point
    :param point2: secound point
    :return: distance between two Points
    """
    return np.sqrt(np.sum(np.power((point1 - point2).toarray(),2) ))


def R_to_axis_angle(matrix):
    """Convert the rotation matrix into the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        x = Qzy-Qyz
        y = Qxz-Qzx
        z = Qyx-Qxy
        r = hypot(x,hypot(y,z))
        t = Qxx+Qyy+Qzz
        theta = atan2(r,t-1)
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @return:    The 3D rotation axis and angle.
    @rtype:     numpy 3D rank-1 array, float
    """

    # Axes.
    axis = np.zeros(3)
    axis[0] = matrix[2, 1] - matrix[1, 2]
    axis[1] = matrix[0, 2] - matrix[2, 0]
    axis[2] = matrix[1, 0] - matrix[0, 1]

    # Angle.
    r = np.hypot(axis[0], np.hypot(axis[1], axis[2]))
    t = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    theta = math.atan2(r, t - 1)

    # Normalise the axis.
    axis = axis / r

    # Return the data.
    return axis, theta


def sphereFit(frames):
    """
    Fit a sphere to a serise of transformations
    :param frames: list of frames
    :return: raduis and center of rotation
    """
    spX = []
    spY = []
    spZ = []
    for frame in frames:
        spX.append(frame[0])
        spY.append(frame[1])
        spZ.append(frame[2])

    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX), 4))
    A[:, 0] = spX * 2
    A[:, 1] = spY * 2
    A[:, 2] = spZ * 2
    A[:, 3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX), 1))
    f[:, 0] = (spX * spX) + (spY * spY) + (spZ * spZ)
    C, residules, rank, singval = np.linalg.lstsq(A, f)

    #   solve for the radius
    t = (C[0] * C[0]) + (C[1] * C[1]) + (C[2] * C[2]) + C[3]
    radius = np.sqrt(t)
    return radius, C[:3]


def points_to_matrix(points):
    """
    converts the points to an array
    :param points:
    :return:
    """

    cloud = np.zeros((len(points), 3))
    for index, point in enumerate(points):
        cloud[index, :] = [point.x, point.y, point.z]

    return cloud


def get_rmse(marker_set, body, frame):
    """
    Get the RMSE of the transform and a body location
    :param marker_set: The location of the markers on the rigid body
    :param body: The location of the markers in the global frame
    :param frame: The frame to get the error
    :return:
    """
    f = [body[0][frame], body[1][frame], body[2][frame], body[3][frame]]
    T, err = cloud_to_cloud(marker_set, f)
    return err


def fit_to_plane(points):
    """
    fit a plan to an array of points using regression
    https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    :param points: list of points
    :return:
    """

    tmp_A = []
    tmp_b = []
    for point in points:
        tmp_A.append([point.x, point.y, 1])
        tmp_b.append(point.z)
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * b

    errors = b - A * fit
    residual = np.linalg.norm(errors)
    fit = unit_vector(fit)
    return fit, residual


if __name__ == '__main__':
    DataSets1 = [core.Point(531.6667, - 508.9951, 314.4273),
                 core.Point(510.5082, - 457.7791, 357.1969),
                 core.Point(463.9945, - 476.0904, 356.1137),
                 core.Point(552.4579, - 566.4891, 393.5611),
                 core.Point(505.9442, - 584.8004, 392.4779)]

    DataSets2 = [[-55.4398, 406.9759, - 487.4170],
                 [-117.4716, 384.3339, -510.7755],
                 [-99.5008, 336.9028, - 511.4401],
                 [-84.8805, 394.2636, - 393.6067],
                 [-67.3354, 347.3059, - 393.7805]]

    marker = [core.Point(0.0, 50.0, 0.0),
              core.Point(-70.0, 50.0, 0.0),
              core.Point(-70, 0, 0),
              core.Point(0.0, 50.0, 100.0),
              core.Point(0.0, 0.0, 100.0)]

    # print cloudtocloud(marker, DataSets1)

    # marker0 = np.asarray([3.6, 5.4, 1.69]).transpose()
    # marker1 = np.asarray([4.0, 6.0, 1.75]).transpose()
    # marker2 = np.asarray([3.8, 7.2, 1.59]).transpose()
    # marker3 = np.asarray([3.4, 7.9, 1.34]).transpose()
    #
    # frame = np.asarray([marker0, marker1, marker2, marker3])
    # make_frame(frame)
    # vect = get_angle_between_vects(marker1, marker2)
    # print transform_vector(frame, marker0)
