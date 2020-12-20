# AIM_Vicon


## Authors
- [Nathaniel Goldfarb](https://github.com/nag92) (nagoldfarb@wpi.edu)
- [Alek Lewis](https://github.com/ajlewis02) (ajlewis@wpi.edu)


A package to read in Vicon data for analysis. This package can be used to read in a CSV file generated from 
the Vicon motion capture system. It will automatically attempt to interpolate missing data, and can save data back to a csv.


## Dependence
* python 2.7
* numpy
* matplotlib
* pandas
* scipy

## External Dependence 
This package requires:

* [AIM_GaitCore](https://github.com/WPI-AIM/AIM_GaitCore.git)



## Notes
- subject prefix removed from marker name i.e (subject:RKNEE -> RKNEE)
- New devices connected to the Vicon should extend the Device class

## Installation
This package can be installed via pip:
```bash
pip install git+https://github.com/WPI-AIM/AIM_Vicon.git
```

(If you have both Python 2 and Python 3 installed you'll need to specify `pip3` - `pip` defaults to Python 2 if installed.)

## Updating
This package can be updated through pip:
```bash
pip install --upgrade git+https://github.com/WPI-AIM/AIM_Vicon.git
```

If necessary, GaitCore can also be updated through pip:
```bash
pip install --upgrade git+https://github.com/WPI-AIM/AIM_GaitCore.git
```

##Usage

### Vicon

#### Reading Data
Vicon automatically reads data from the provided file when constructed.
The constructor the following flags: ``verbose`` (defaults to ``False``), ``interpolate`` (defaults to ``True``),
``maxnanstotal``, (defaults to -1), ``maxnansrow`` (defaults to -1), and ``sanitize`` (defaults to ``True``).

If ``verbose`` is set to ``True``, it will print status updates and warnings while reading data. 

If ``interpolate`` is set to ``True``, it will attempt to interpolate missing data points. If ``interpolate`` is set to 
``False``, or if a field cannot be interpolated, missing data points will be set to ``np.nan``.

If ``maxnanstotal`` or ``maxnansrow`` are set to non-negative values, they will provide a cap on the maximum allowed
nans total or in a row, respectively. If a field violates either rule, that field will not be
interpolated. Setting either of these flags to 0 is equivalent to setting ``interpolate`` to ``False``.

If ``sanitize`` is set to ``True``, Vicon will replace any field consisting entirely of NaNs with
0s. Vicon will keep track of every object that contains a santized field. This can be checked through the ``is_sanitized``
method.

Note: Some objects, as read by Vicon, contain empty fields named ``""`` that contain no data.
These fields will be sanitized, but an object will *not* be marked as having been sanitized for having
one of these fields.

#### Saving Data
The ``Vicon.save()`` method will save the data previously read.
It accepts three flags: ``filename``, which defaults to ``None``, ``verbose``, which defaults to ``False``, and
``mark_interpolated``, which defaults to ``True``.

If ``filename`` is not provided, it will default to the file path specified on construction. ***WARNING: Saving to a
file will overwrite it.***

``verbose`` controls whether or not the save method will print status updates and warnings.

If ``mark_interpolated`` is set to ``True``, any values that were generated through interpolation will be preceded by '!'.
Vicon is able to read this, and a future Vicon object reading this value will display a warning with ``verbose`` set to ``True``.


###Markers
A ``Markers`` object can be obtained through the ``Vicon.get_markers()`` method.
It contains information about the markers' positions, and contains methods for calculating
information about the rigid bodies and the joint centers.

####Getting a Rigid Body
The ``smart_sort`` function will automatically group markers into their rigid bodies.
Once sorted, it is possible to retrieve the data of all markers associated with a given rigid body, using the ``get_rigid_bodies``
function.

####Transformation Matrices
The ``auto_make_transform(frames)`` function will automatically make the transformation matrices for every
rigid body for which a frame of reference is provided. 
A frame of reference is an array of points, which represent the locations of the markers on the rigid body
relative to 0,0 on that rigid body.

The ``get_frame`` function will return the transformation matrices for a given rigid body for all frames.
``get_frame(RigidBody)[n]`` gives the transformation matrix for the specified rigid body during frame n.

The transformation matrices are of the form local to global - that is to say, where ``T = markers.get_frame(RigidBody)[n]``,
``np.dot(T, [[0], [0], [0], [1]])`` will return a vector representing the location of the specified rigid body during frame n.

####Frame Shifting
There are a few static methods which automatically perform frame-shifting operations, requiring only that the user specify points and frames.

``Markers.global_to_frame(frame, vector)`` transforms a vector in the global reference frame to the reference frame specified.

``Markers.global_point_to_frame(frame, vector)`` transforms a Point object from the global frame to the provided frame.

``Markers.local_to_global(frame, vector)`` is the inverse of ``global_to_frame``, and likewise ``Markers.local_point_to_global(frame, point)``
is the inverse of ``global_point_to_frame``.

``Markers.get_transform_btw_two_frames(parent_frame, child_frame)`` returns the transformation matrix from the parent
frame of reference to the child frame of reference.

####Defining and Calculating Joint Centers

The ``def_joint`` function allows the user to define their own joints with the rigid bodies in the data.
``def_joint("r_hip", "hip", "r_femur", ballJoint=True)`` creates a ball joint named *r_hip* between the rigid bodies *hip* and *r_femur*.

The ``calculate_joints`` function will automatically calculate the positions of all defined joint locations.
Additionally, joints may be calculated directly through the ``_calc_ball_joint`` and ``_calc_hinge_joint`` methods.

Joint positions calculated through the ``calculate_joints`` function can be accessed using the ``get_joint(name)`` method.
Joint positions are represented as a 2D array, consisting of the ``[x, y, z]`` position of a joint for each timestep.

The joint position relative to the parent and child rigid bodies can be accessed through the ``get_joint_rel`` and
``get_joint_rel_child`` methods, respectively. Each returns a 1D array of the ``[x, y, z]`` position of the joint center,
relative to either the parent or child rigid body, in that rigid body's reference frame.

####Playing the Markers

The ``play`` function will create a matplotlib animation of the markers. If the ``calculate_joints`` 
function has been run, and the ``joints`` flag is set to ``True``, this animation will include the calculated joint locations
in green. If the ``center`` flag is set to ``True``, the body will be anchored to the center of the screen. This is highly recommended
for any dataset where the markers move a large distance from their starting position.

## Examples

### Interpolation
Vicon's behavior upon encountering missing data is highly customizable.
#### Default Case
With no flags specified, Vicon will attempt to interpolate any amount of missing data in all fields.
```python
import Vicon

data = Vicon.Vicon("path/to/file")
```

#### No Interpolation
Vicon can be configured to never attempt to interpolate missing data, instead filling any holes with ``np.nan``.
```python
import Vicon

data = Vicon.Vicon("path/to/file", interpolate=False)  # No interpolation!
```

#### Interpolate only small holes
By using the ``maxnansrow`` field, Vicon can be configured to only attempt to interpolate fields
if they do not have any holes larger than specified.
```python
import Vicon

data = Vicon.Vicon("path/to/file", maxnansrow=100)  # If any field is missing more than 100 values in a row, it will not be interpolated.
```

#### Interpolate only mostly complete data
By using the ``maxnanstotal`` field, Vicon will only attempt to interpolate a field if it does not have
too many missing values.
```python
import Vicon

data = Vicon.Vicon("path/to/file", maxnanstotal=1000)  # If any field is missing more than 1000 values in total, it will not be interpolated.
```

### Saving data
Vicon can save data into a CSV file. This can be done to save the results of any interpolation, or perhaps
to copy a CSV file very inefficiently.

#### Default Case
In the default case, Vicon will overwrite the file that it read from on creation. 
Any values produced via interpolation will be preceded with '!'. If Vicon encounters such a value while in verbose mode, 
it will print a warning.
```python
import Vicon

data = Vicon.Vicon("path/to/file")

...

data.save()
```

#### Save to new file
Vicon can write saved data to a new file.
```python
import Vicon

data = Vicon.Vicon("path/to/file")

...

data.save(filename="path/to/new/file")
```

#### Do not mark interpolated values
Vicon can be configured to not mark previously interpolated values. **If this is done, any future Vicon object will not be able to distinguish previously interpolated values from real data.**
```python
import Vicon

data = Vicon.Vicon("path/to/file")

...

data.save(mark_interpolated=False)
```
### Playing the markers


```python
import Vicon
file = "path to CSV file"
data = Vicon.Vicon(file)
markers = data.get_markers()
markers.smart_sort() # sort the markers into bodies by the names 
markers.play()
```

### Playing markers with joints
```python
import Vicon
import Core
import Markers

v = Vicon.Vicon("Path/To/File")
markers = v.get_markers()
markers.smart_sort()

frames = {"Root": [Core.Point.Point(0, 14, 0),
                   Core.Point.Point(56, 0, 0),
                   Core.Point.Point(14, 63, 0),
                   Core.Point.Point(56, 63, 0)], "L_Foot": [Core.Point.Point(0, 0, 0),
                                                            Core.Point.Point(70, 0, 0),
                                                            Core.Point.Point(28, 70, 0),
                                                            Core.Point.Point(70, 63, 0)],
          "L_Tibia": [Core.Point.Point(0, 0, 0),
                      Core.Point.Point(0, 63, 0),
                      Core.Point.Point(70, 14, 0),
                      Core.Point.Point(35, 49, 0)], "L_Femur": [Core.Point.Point(0, 0, 0),
                                                                Core.Point.Point(70, 0, 0),
                                                                Core.Point.Point(0, 42, 0),
                                                                Core.Point.Point(70, 56, 0)],
          "R_Foot": [Core.Point.Point(0, 0, 0),
                     Core.Point.Point(56, 0, 0),
                     Core.Point.Point(0, 49, 0),
                     Core.Point.Point(42, 70, 0)], "R_Tibia": [Core.Point.Point(0, 0, 0),
                                                               Core.Point.Point(42, 0, 0),
                                                               Core.Point.Point(7, 49, 0),
                                                               Core.Point.Point(63, 70, 0)],
          "R_Femur": [Core.Point.Point(7, 0, 0),
                      Core.Point.Point(56, 0, 0),
                      Core.Point.Point(0, 70, 0),
                      Core.Point.Point(42, 49, 0)]}

markers.auto_make_transform(frames)

# If any of the markers on the rigid bodies are missing data, the joint calculation will be inaccurate.
# With interpolation and sanitizing on, markers will only be missing data if they have been sanitized.
for body in markers._rigid_body.keys():
    if ("R_" in body or "L_" in body) and v.is_sanitized("Trajectories", body):
        print(body + " is missing data! Adjacent joint locations might not be correct!")

markers.calc_joints()

markers.play(joints=True)

```

### Get rigid body
Rigid bodies are organized  by marker then frame. 
The markers are of type Point. 

```python
import Vicon
file = "path to CSV file"
data = Vicon.Vicon(file)
markers = data.get_markers()
markers.smart_sort() # optional param to remove subject name
shank_frame = markers.get_rigid_body("name of body") # returns an array of markers 
## Get the X corr of a marker 2 in frame 100
x = shank_frame[2][100].x
```


### Get rigid body transform
Rigid bodies are organized  by marker then frame. 
The markers are of type Point. 

```python
import Vicon
import Core
import Markers

file = "path to CSV file"
data = Vicon.Vicon(file)
markers = data.get_markers()
markers.smart_sort() # optional param to remove subject name

# Do several bodies, use the marker location on the rigidbody
frames = {"Root": [Core.Point.Point(0, 14, 0),
                   Core.Point.Point(56, 0, 0),
                   Core.Point.Point(14, 63, 0),
                   Core.Point.Point(56, 63, 0)], "L_Foot": [Core.Point.Point(0, 0, 0),
                                                            Core.Point.Point(70, 0, 0),
                                                            Core.Point.Point(28, 70, 0),
                                                            Core.Point.Point(70, 63, 0)],
          "L_Tibia": [Core.Point.Point(0, 0, 0),
                      Core.Point.Point(0, 63, 0),
                      Core.Point.Point(70, 14, 0),
                      Core.Point.Point(35, 49, 0)], "L_Femur": [Core.Point.Point(0, 0, 0),
                                                                Core.Point.Point(70, 0, 0),
                                                                Core.Point.Point(0, 42, 0),
                                                                Core.Point.Point(70, 56, 0)],
          "R_Foot": [Core.Point.Point(0, 0, 0),
                     Core.Point.Point(56, 0, 0),
                     Core.Point.Point(0, 49, 0),
                     Core.Point.Point(42, 70, 0)], "R_Tibia": [Core.Point.Point(0, 0, 0),
                                                               Core.Point.Point(42, 0, 0),
                                                               Core.Point.Point(7, 49, 0),
                                                               Core.Point.Point(63, 70, 0)],
          "R_Femur": [Core.Point.Point(7, 0, 0),
                      Core.Point.Point(56, 0, 0),
                      Core.Point.Point(0, 70, 0),
                      Core.Point.Point(42, 49, 0)]}

markers.auto_make_transform(frames)

hip_frames = markers.get_frame("Root")
l_femur = markers.get_rigid_body("L_Femur")

#Get the position of all the markers on the left femur rigid body relative to the hip rigid body on frame 0
rel_pos = [Markers.global_point_to_frame(hip_frames[0], l_femur[n][0]) for n in range(4)]
```

### Get model outputs 
Currently only works with lower body model.

```python
import Vicon
file = "path to CSV file"
data = Vicon.Vicon(file)
model = data.get_model_output()
model.left_leg().hip.angle.x
```

### Get force plates

```python
import Vicon
file = "path to CSV file"
data = Vicon.Vicon(file)
fp = data.get_force_plate(1).get_forces() # pass in 1 or 2 to get the foce plates
```


