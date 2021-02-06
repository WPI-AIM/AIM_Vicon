#!/usr/bin/env python3

import matplotlib.pyplot as plot
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D as plot3d

from GaitCore.Core.PointArray import PointArray

class AnimateModel():
    def __init__(self,  window_title: str = "3D Marker Animation",
                        chart_title: str = "3D Marker Positions"):
        self.markers = {}
        self.units = ""

        self.fig = plot.figure()
        self.fig.canvas.set_window_title(window_title)

        self.ax = self.fig.add_subplot(111, aspect='equal', projection='3d')

        self.ax.set_title = chart_title

    def import_markers(self, data: dict):
        # Define unit list to make sure all units are the same
        units_list = []
        
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
        
        # Check to make sure if all units are the same
        if len(units_list) < 0:
            is_same_units = True
        is_same_units = all(unit == units_list[0] for unit in units_list)

        if is_same_units:
            self.units = units_list[0]
        else:
            raise Exception("Error in AnimateModel: Marker units are not all the same!")


    def draw(self, save_animation: bool = False):

        # ani = animation.FuncAnimation(self.fig)

        if save_animation:
            print("Save animation")

        plot.show()

    def _animation_init(self):
        print()

    def _animate(self, i):
        print()