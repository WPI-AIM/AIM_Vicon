from . import Interpolation

class Akmia(Interpolation.Interpolation):

    def __init__(self, data):
        super(Akmia, self).__init__(data)

    def interpolate(self, verbose):

        for key, value in self.data.items():  # For every subject in the data...
            for sub_key, sub_value in value.items():
                if not ("Magnitude( X )" in value.keys()) and not ("Count" in value.keys()):
                    Interpolation.akmia(sub_value, verbose, "Trajectory", sub_key, key)

