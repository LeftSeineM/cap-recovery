"""A self-contained compressed document with shared repeated numeric values.

References stay inside one calls file. Resolving a failure never needs its original
trajectory, another batch, or a collision-world dump. This is a storage encoding,
not a second record schema or a reader for historical formats.
"""
import gzip
import json
import pickle
from pathlib import Path

from table_1000.common.serialization import dump

SHARE_BYTES = 192
SCHEMA = 2  # calls documents with #114's search-tree facts


def save_document(path, document):
    values, indices, memo = [], {}, {}
    def pack(value):
        if not isinstance(value, (dict, list)):
            return value
        if id(value) in memo:
            return memo[id(value)]
        # A local structural key only: the persisted document remains JSON and
        # readers never unpickle data. Match a repeated subtree before walking
        # its thousands of numeric leaves again.
        key = pickle.dumps(value, protocol=5)
        if len(key) < SHARE_BYTES:
            result = value
        else:
            if key not in indices:
                packed = ({name: pack(item) for name, item in value.items()} if isinstance(value, dict)
                          else [pack(item) for item in value])
                indices[key] = len(values)
                values.append(packed)
            result = {'$ref': indices[key]}
        memo[id(value)] = result
        return result
    result = {key: pack(value) for key, value in document.items()}
    result['values'] = values
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', compresslevel=6) as stream:
        dump(result, stream, separators=(',', ':'))


def load_document(path):
    with gzip.open(path, 'rt') as stream:
        document = json.load(stream)
    if document['version'] != SCHEMA:
        raise ValueError(f'{path}: unsupported calls schema; convert the old batch first')
    values = document.pop('values')
    resolved = {}
    def unpack(value):
        if isinstance(value, list):
            return [unpack(item) for item in value]
        if isinstance(value, dict):
            if set(value) == {'$ref'}:
                index = value['$ref']
                if index not in resolved:
                    resolved[index] = unpack(values[index])
                return resolved[index]
            return {key: unpack(item) for key, item in value.items()}
        return value
    return {key: unpack(value) for key, value in document.items()}
