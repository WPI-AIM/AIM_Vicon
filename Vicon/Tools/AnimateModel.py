#!/usr/bin/env python3

import matplotlib.pyplot as plot
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D as plot3d

from GaitCore.Core.PointArray import PointArray

class AnimateModel():
    def __init__(self,  window_title: str = "3D Marker Animation",
                        chart_title: str = "3D Marker Positions",
                        x_label: str = "X",
                        y_label: str = "Y",
                        z_label: str = "Z",
                        x_limit: tuple = None,      # (lower bound, upper bound)
                        y_limit: tuple = None,      # (lower bound, upper bound)
                        z_limit: tuple = None):     # (lower bound, upper bound)
        self._length = 0
        
        self.markers = {}
        self.units = ""

        self.fig = plot.figure()
        self.fig.canvas.set_window_title(window_title)

        self.window_title = window_title
        self.chart_title = chart_title
        self.x_label = x_label + " (in " + self.units + ")"
        self.y_label = y_label + " (in " + self.units + ")"
        self.z_label = z_label + " (in " + self.units + ")"

        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel(x_label + " (in " + self.units + ")")
        self.ax.set_ylabel(y_label + " (in " + self.units + ")")
        self.ax.set_zlabel(z_label + " (in " + self.units + ")")
        self.ax.set_title = chart_title

        self.x_limit = x_limit
        self.y_limit = y_limit
        self.z_limit = z_limit

    def import_markers(self, data: dict):
        # Define unit list to make sure all units are the same
        units_list = []
        length_list = []
        
        # Erase existing markers
        self.markers = {}

        # Define markers
        for key, value in data.items():
            self.markers[key] = PointArray(
                x = value.get('X').get('data'),
                y = value.get('Y').get('data'),
                z = value.get('Z').get('data'))
            units_list.append(value.get('X').get('unit'))
            units_list.append(value.get('Y').get('unit'))
            units_list.append(value.get('Z').get('unit'))

            length_list.append(len(self.markers[key]))
        
        # Check to make sure if all units are the same
        if len(units_list) < 0:
            is_same_units = True
        is_same_units = all(unit == units_list[0] for unit in units_list)

        if is_same_units:
            self.units = units_list[0]
        else:
            raise Exception("Error in AnimateModel: Marker units are not all the same!")

        if all(length == length_list[0] for length in length_list):
            if self._length == 0:
                self._length = length_list[0]
            if self._length != length_list[0]:
                raise Exception("Error in AnimateModel: Marker sizes are not all the same")
        else:
            raise Exception("Error in AnimateModel: Marker sizes are not all the same")

    def draw(self, interval: int = 2, save_animation: bool = False):

        ani = animation.FuncAnimation(  fig = self.fig, 
                                        func = self._animate, 
                                        frames = self._length, 
                                        interval = interval,
                                        repeat = True)

        if save_animation:
            print("Save animation")

        plot.show()

    def _animation_init(self):
        print("animation init not implemented")

    def _animate(self, i: int):

        # --- Clear and Set Up Figure ---- #
        self.ax.clear()
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.set_zlabel(self.z_label)
        self.ax.set_title = self.chart_title

        if self.x_limit is not None:
            self.ax.set_xlim3d(list(self.x_limit))
        if self.y_limit is not None:
            self.ax.set_ylim3d(list(self.y_limit))
        if self.z_limit is not None:
            self.ax.set_zlim3d(list(self.z_limit))

        # --- Animate Markers --- #
        marker_poses_x = []
        marker_poses_y = []
        marker_poses_z = []
        for key, value in self.markers.items():
            marker_poses_x.append(value.get(i).x)
            marker_poses_y.append(value.get(i).y)
            marker_poses_z.append(value.get(i).z)
        
        self.ax.scatter(marker_poses_x,
                        marker_poses_y,
                        marker_poses_z,
                        c = 'r', marker = 'o')