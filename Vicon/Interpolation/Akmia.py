from . import Interpolation

class Akmia(Interpolation.Interpolation):

    def __init__(self, data):
        super(Akmia, self).__init__(data)

    @property
    def interpolate(self, naninfo, sanitize, verbose):

        for key, value in self.data.items():  # For every subject in the data...

            if not ("Magnitude( X )" in value.keys()) and not ("Count" in value.keys()):
                Interpolation.akmia(value, key, naninfo, "Trajectory", False, sanitize, verbose)

