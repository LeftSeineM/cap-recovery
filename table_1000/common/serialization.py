"""JSON and HDF5 attribute values from numerical runtime reports."""
import json


def native_value(value):
    """Convert a NumPy scalar/array; leave ordinary values to the JSON encoder."""
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _default(value):
    converted = native_value(value)
    if converted is value:
        raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')
    return converted


def dumps(value, **kwargs):
    return json.dumps(value, default=_default, **kwargs)


def dump(value, stream, **kwargs):
    return json.dump(value, stream, default=_default, **kwargs)


def attribute(value):
    """Store scalar attributes natively and structured attributes as JSON."""
    value = native_value(value)
    return value if isinstance(value, (int, float, str)) else dumps(value)
