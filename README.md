# AIM_Vicon


## Authors
- [Nathaniel Goldfarb](https://github.com/nag92) (nagoldfarb@wpi.edu)


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

##Usage

### Reading Data
Vicon automatically reads data from the provided file when constructed.
The constructor the following flags: ``verbose`` (defaults to ``False``), ``interpolate`` (defaults to ``True``),
``maxnanstotal``, (defaults to -1), and ``maxnansrow`` (defaults to -1).

If ``verbose`` is set to ``True``, it will print status updates and warnings while reading data. 

If ``interpolate`` is set to ``True``, it will attempt to interpolate missing data points. If ``interpolate`` is set to 
``False``, or if a field cannot be interpolated, missing data points will be set to ``np.nan``.

If ``maxnanstotal`` or ``maxnansrow`` are set to non-negative values, they will provide a cap on the maximum allowed
nans total or in a row, respectively. If a field violates either rule, that field will not be
interpolated. Setting either of these flags to 0 is equivalent to setting ``interpolate`` to ``False``.

### Saving Data
The ``Vicon.save()`` method will save the data previously read.
It accepts three flags: ``filename``, which defaults to ``None``, ``verbose``, which defaults to ``False``, and
``mark_interpolated``, which defaults to ``True``.

If ``filename`` is not provided, it will default to the file path specified on construction. ***WARNING: Saving to a
file will overwrite it.***

``verbose`` controls whether or not the save method will print status updates and warnings.

If ``mark_interpolated`` is set to ``True``, any values that were generated through interpolation will be preceded by '!'.
Vicon is able to read this, and a future Vicon object reading this value will display a warning with ``verbose`` set to ``True``.


## Examples

### Interpolation
Vicon's behavior when it encounters missing data is highly customizable.
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
import Core as core
file = "path to CSV file"
data = Vicon.Vicon(file)
markers = data.get_markers()
markers.smart_sort() # optional param to remove subject name

# Do several bodies, use the marker location on the rigidbody
frames["hip"] = [core.Point(0.0, 0.0, 0.0),
                 core.Point(70.0, 0, 0.0),
                 core.Point(0, 42.0, 0),
                 core.Point(35.0, 70.0, 0.0)]

frames["RightThigh"] = [core.Point(0.0, 0.0, 0.0),
                        core.Point(56.0, 0, 0.0),
                        core.Point(0, 49.0, 0),
                        core.Point(56.0, 63.0, 0.0)]

frames["RightShank"] = [core.Point(0.0, 0.0, 0.0),
                        core.Point(56.0, 0, 0.0),
                        core.Point(0, 42.0, 0),
                        core.Point(56.0, 70.0, 0.0)]

markers.auto_make_transform(frames)


# Get just one transform and the RMSE error 
# Can be used to get the transformation between ANY two sets of markers 
m = markers.get_rigid_body("ben:hip")
f = [m[0][frame], m[1][frame], m[2][frame], m[3][frame]]
T, err = Markers.cloud_to_cloud(hip_marker, f)
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


