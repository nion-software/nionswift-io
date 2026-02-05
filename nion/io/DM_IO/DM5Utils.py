from __future__ import annotations

import dataclasses
import datetime
import json
import types
import typing

import h5py
import numpy
import pytz

from nion.data import DataAndMetadata
from nion.data import Calibration

NP_NUMERICAL_TYPES = numpy.uint8 | numpy.uint16 | numpy.uint64 | numpy.uint32 | numpy.float32 | numpy.float64 | numpy.int16 | numpy.int32 | numpy.int64
DM_FILE_TYPES = numpy.ndarray | numpy.void | numpy.bytes_ | NP_NUMERICAL_TYPES | numpy.bool_
VOID_FIELD_DICT_TYPES = dict[str, dict[str, str | int]]
DM_DICT_TYPES = typing.Tuple[int, ...] | int | str | typing.List[typing.Any] | float | bytes | dict[str, VOID_FIELD_DICT_TYPES]
SEQUENCE_TYPES = tuple['SEQUENCE_TYPES', ...] | list['SEQUENCE_TYPES'] | numpy.generic | float | int | bool | str | bytes
FieldInfo: typing.TypeAlias = typing.Union[tuple[numpy.dtype[typing.Any], int], tuple[numpy.dtype[typing.Any], int, typing.Any]]
DTypeLike = typing.Union[numpy.dtype[typing.Any], type[numpy.generic], str]

# swift to dm conversions


def swift_to_dm_metadata(data_and_metadata: DataAndMetadata.DataAndMetadata) -> typing.Tuple[dict[str, DM_DICT_TYPES], dict[str, DM_DICT_TYPES]]:
    metadata = dict(data_and_metadata.metadata)  # in order to preserve as much of a dm5 file structure as possible the importer stores a dict representation in the metadata
    dm_metadata = metadata.pop('__dm_metadata__', dict())  # the dict representation is removed, with the rest of the metadata being used for ImageTags
    return metadata, dm_metadata


@dataclasses.dataclass
class DMFormatDataAndMetadata:
    data: numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[typing.Any]]
    metadata: dict[str, DM_DICT_TYPES]
    dm_metadata: dict[str, DM_DICT_TYPES]
    data_descriptor: DataAndMetadata.DataDescriptor
    dimensional_calibrations: typing.Sequence[Calibration.Calibration]
    collection_dimension_count: int
    datum_dimension_count: int
    needs_slice: bool
    is_single_dimension: bool
    is_sequence: bool


def swift_to_dm_format(data_and_metadata: DataAndMetadata.DataAndMetadata) -> DMFormatDataAndMetadata:
    """Changes the swift data and metadata into the same representation of dimensions as how digital micrograph stores it"""
    data: numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[typing.Any]] = data_and_metadata.data
    dimensional_calibrations: typing.Sequence[Calibration.Calibration] = data_and_metadata.dimensional_calibrations
    collection_dimension_count: int = data_and_metadata.collection_dimension_count
    datum_dimension_count: int = data_and_metadata.datum_dimension_count
    needs_slice: bool = False
    if data_and_metadata.data.dtype != numpy.uint8 and data_and_metadata.datum_dimension_count == 1:
        if len(data_and_metadata.data.shape) == 3:
            data = numpy.moveaxis(data_and_metadata.data, 2, 0)
            dimensional_calibrations = (data_and_metadata.dimensional_calibrations[2],) + tuple(data_and_metadata.dimensional_calibrations[0:2])
        if len(data_and_metadata.data.shape) == 2:
            data = numpy.moveaxis(data_and_metadata.data, 1, 0)
            data = numpy.expand_dims(data, axis=1)
            dimensional_calibrations = (data_and_metadata.dimensional_calibrations[1], Calibration.Calibration(),
                                        data_and_metadata.dimensional_calibrations[0])
            collection_dimension_count, datum_dimension_count = (2, 1)
            needs_slice = True

    needs_slice = needs_slice or (collection_dimension_count == 2 and datum_dimension_count == 1)
    is_single_dimension = (collection_dimension_count + (1 if data_and_metadata.data_descriptor.is_sequence else 0)) == 1
    metadata, dm_metadata = swift_to_dm_metadata(data_and_metadata)
    data_descriptor = data_and_metadata.data_descriptor
    is_sequence = data_descriptor.is_sequence

    return DMFormatDataAndMetadata(data, metadata, dm_metadata, data_descriptor, dimensional_calibrations, collection_dimension_count, datum_dimension_count, needs_slice, is_single_dimension, is_sequence)


def get_datetime_as_str(modified: datetime.datetime | None, timezone: str | None, timezone_offset: str | None) -> tuple[str | None, str | None]:
    if modified:
        timezone_str = None
        if timezone:
            try:
                tz = pytz.timezone(timezone)
                timezone_str = tz.tzname(modified)
            except (pytz.AmbiguousTimeError, pytz.NonExistentTimeError):
                timezone_str = None

        if timezone_str is None and timezone_offset:
            timezone_str = timezone_offset

        timezone_str = "" if timezone_str is None else " " + timezone_str
        return modified.strftime("%x"), modified.strftime("%X") + timezone_str
    return None, None


def _convert_sequence_to_void(sequence_value: SEQUENCE_TYPES) \
        -> tuple[numpy.dtype | None, (DM_FILE_TYPES | None) | tuple[DM_FILE_TYPES, ...]]:
    """Converts a tuple/list to a form that can be stored in a numpy.void. Returns numpy.dtype, and a tuple of values to use to create the void.

    Stores a recursive dictionary of the original container type (list or tuple) in the dtype metadata so it the original data can be rebuilt from the numpy void.
    """
    if not isinstance(sequence_value, (list, tuple)):
        np_dtype = None
        numpy_value = None
        if isinstance(sequence_value, numpy.generic):
            np_dtype = sequence_value.dtype
            numpy_value = np_dtype.type(sequence_value)
        elif isinstance(sequence_value, bool):
            np_dtype = numpy.dtype(numpy.bool_)
            numpy_value = numpy.bool_(sequence_value)
        elif isinstance(sequence_value, int):
            np_dtype = numpy.min_scalar_type(sequence_value)
            numpy_value = np_dtype.type(sequence_value)
        elif isinstance(sequence_value, float):
            np_dtype = numpy.result_type(sequence_value)
            numpy_value = np_dtype.type(sequence_value)
        elif isinstance(sequence_value, str):
            np_dtype = numpy.dtype(f"U{len(sequence_value)}")  # Array-protocol str from numpy
            numpy_value = numpy.str_(sequence_value)
        elif isinstance(sequence_value, bytes):
            np_dtype = numpy.dtype(f"S{len(sequence_value)}")
            numpy_value = numpy.bytes_(sequence_value)

        if np_dtype is not None and numpy_value is not None:
            return np_dtype, typing.cast(DM_FILE_TYPES, numpy_value)  # base case of recursion
        else:
            return None, None

    # The data is a list or tuple
    fields: typing.List[typing.Tuple[str, typing.Any]] = []
    values: typing.List[typing.Any] = []
    children_meta: typing.List[dict[str, typing.Any]] = []

    for i, element in enumerate(sequence_value):
        name = f"f{i}"
        np_dtype, value = _convert_sequence_to_void(element)
        if value is None or np_dtype is None:
            continue  # Don't store unsupported types
        fields.append((name, np_dtype))
        values.append(value)
        if np_dtype.metadata and "container" in np_dtype.metadata:
            children_meta.append({
                "container": np_dtype.metadata["container"],
                "children": np_dtype.metadata.get("children", [])
            })
        else:
            children_meta.append({"container": "scalar"})

    container_type = "list" if isinstance(sequence_value, list) else "tuple"
    metadata = {"container": container_type, "children": children_meta}

    np_dtype = numpy.dtype(fields, metadata=metadata)
    value = tuple(values)

    return np_dtype, value


def _convert_void_to_sequence(void: numpy.void, np_dtype: numpy.dtype, meta: dict[str, typing.Any]) \
        -> list[typing.Any] | tuple[typing.Any, ...] | typing.Any:
    """Recursively convert a void to a list or tuple, using metadata dictionary to determine the container type"""
    if np_dtype.fields is None:
        return void.item()
    assert np_dtype.names is not None
    sequence: list[typing.Any] = []
    container = meta.get("container", "list")
    for i, name in enumerate(np_dtype.names):
        child_field = dict(np_dtype.fields).get(name)

        child_dtype = child_field[0] if child_field is not None else None
        field_val = void[name]

        if child_dtype and child_dtype.fields is not None:
            child_meta = meta.get('children', [])[i]
            sequence.append(_convert_void_to_sequence(field_val, child_dtype, child_meta))
        else:
            if isinstance(field_val, numpy.generic):
                sequence.append(field_val.item())
            else:
                sequence.append(field_val)

    return sequence if container == "list" else tuple(sequence)

# h5py utility functions


def decode_bytes_to_str(data: bytes) -> str:
    try:
        return data.decode()
    except UnicodeDecodeError:
        return data.decode('latin1')  # latin1 has to be used in place of utf-8 because sometimes there are non utf-8 bytes in dm5 files


def get_or_create_group(base_group: h5py.Group, name: str) -> h5py.Group:
    """Creates a group if one doesn't already exist and returns it or return the existing group."""
    group = base_group.get(name)
    if group is None:
        return base_group.create_group(name)
    return group


def get_from_nested_dict(dictionary: dict[str, typing.Any], path: list[str], default: typing.Any = None) -> typing.Any:
    current_dict = dictionary
    for key in path:
        item = current_dict.get(key)
        if isinstance(item, dict):
            current_dict = item
        elif item is None:
            return default
        else:
            return item

    if current_dict is None:
        return default

    if isinstance(current_dict, dict):
        if current_dict.get('__data__'):
            return current_dict['__data__']
    return current_dict


def save_attr_to_group(name: str, swift_value: SEQUENCE_TYPES | DM_FILE_TYPES, group: h5py.Group,
                       dtype: DTypeLike | list[tuple[str, ...]] | None = None) -> None:
    """Save data to a group's attribute, converting python types to numpy types."""
    dm_value: DM_FILE_TYPES | None = None

    if isinstance(swift_value, DM_FILE_TYPES):
        dm_value = swift_value
    elif isinstance(swift_value, str):
        dm_value = numpy.bytes_(swift_value.encode())
    elif isinstance(swift_value, bool):
        dm_value = numpy.bool_(swift_value)
    elif isinstance(swift_value, float):
        if dtype is None:
            dm_value = numpy.float64(swift_value)
        elif numpy.issubdtype(dtype, numpy.floating):
            dm_value = numpy.asarray(swift_value, dtype=dtype)[()]
    elif isinstance(swift_value, int):
        if dtype is None:
            dm_value = numpy.int64(swift_value)
        elif numpy.issubdtype(dtype, numpy.integer):
            dm_value = numpy.asarray(swift_value, dtype=dtype)[()]
    elif isinstance(swift_value, (tuple, list)):
        if dtype is not None:
            np_dtype = numpy.dtype(dtype)
            if np_dtype.fields is None:
                raise TypeError(f"Expected a structured dtype for sequence data {type(swift_value)}.")
            dm_value = numpy.array(tuple(swift_value), dtype=np_dtype)[()]  # Construction of void type
        else:
            void_dtype, data = _convert_sequence_to_void(swift_value)
            if void_dtype is not None and hasattr(void_dtype, "metadata") and isinstance(void_dtype.metadata, types.MappingProxyType):
                # dtype metadata is not saved by h5py so instead the metadata is saved as an attribute named '__meta__'
                meta = dict(void_dtype.metadata) or {"container": "scalar", "children": []}
                save_attr_to_group(name=name + ".__meta__", swift_value=json.dumps(meta), group=group)
                if data:
                    dm_value = numpy.array(data, dtype=void_dtype)[()]
                else:  # If the data is an empty tuple h5py throws an error unless it is done without the dtype
                    dm_value = numpy.array(data)[()]

    if dm_value is not None:
        try:
            if group.attrs.get(name) is not None:
                group.attrs[name] = dm_value
            else:
                group.attrs.create(name, dm_value)
        except OSError as e:
            print(f"Failed to save {name}, {e}")
    else:
        raise TypeError(f"{swift_value!r}, {type(swift_value)!r} is not supported.")

# Serialization Functions


def _serialize_dm_attrs_into_swift_metadata(data: DM_FILE_TYPES | int | float) \
        -> typing.Dict[str, DM_DICT_TYPES | dict[str, typing.Any]] | int | float:
    """Converts data in dm5 attrs, DM_FILE_TYPES, into a dict with the data at dict['__data__'] in a type that can be stored
    in swift metadata.

    Information to rebuild the original type is stored in the dict with the converted data.
    Data can be int or float when there is a recursive call in the ndarray or void, otherwise it is a DM_FILE_TYPE.
    This will raise a TypeError if the type of data was not in DM_FILE_TYPES as an unhandled case.
    """

    def _serialize_void_dtype_into_swift_metadata(fields: typing.Optional[typing.Mapping[str, FieldInfo]]) \
            -> VOID_FIELD_DICT_TYPES:

        void_dict: VOID_FIELD_DICT_TYPES = {}
        if fields is None:
            return void_dict

        for name, info in fields.items():
            dtype, alignment = info[0], info[1]
            void_dict[name] = {
                '__dtype__': dtype.str,
                '__alignment__': alignment,
            }
        return void_dict

    serialized: typing.Dict[str, DM_DICT_TYPES | dict[str, typing.Any]]
    if isinstance(data, numpy.ndarray):
        serialized = {
            '__data__': [_serialize_dm_attrs_into_swift_metadata(x) for x in data.tolist()],
            '__dtype__': data.dtype.str,
            '__shape__': data.shape,
        }
    elif isinstance(data, numpy.void) and data.dtype.fields is not None:
        serialized = {
            '__data__': {
                '__data__': [_serialize_dm_attrs_into_swift_metadata(x) for x in data.tolist()],
                '__fields__': _serialize_void_dtype_into_swift_metadata(data.dtype.fields),
            },
            '__dtype__': data.dtype.str,
            '__shape__': data.shape,
        }
    elif isinstance(data, numpy.bytes_):
        serialized = {
            '__data__': decode_bytes_to_str(data),
            '__dtype__': data.dtype.str,
        }
    elif isinstance(data, (numpy.integer, numpy.bool_, numpy.floating)):
        serialized = {
            '__data__': data.item(),
            '__dtype__': data.dtype.str,
        }
    elif isinstance(data, (float, int)):
        return data
    else:
        raise TypeError(f"{data}, {type(data)} is not supported.")
    return serialized


def _deserialize_dm_attrs_from_swift_metadata(serialized: typing.Mapping[str, DM_DICT_TYPES]) \
        -> DM_FILE_TYPES:
    """Convert the swift metadata back to dm5 attrs data that was serialized using serialize_dm_attrs_into_swift_metadata.

    Uses the stored information in the dict about the original type, and then converts the data to be that type again.
    This will raise an exception if the dictionary passed was not one of the possible serialized versions from serialize_dm_attrs_into_swift_metadata.
    """

    def deserialize_dtype(serialized_void: VOID_FIELD_DICT_TYPES) -> numpy.dtype | None:

        names: list[str] = []
        formats: list[numpy.dtype[typing.Any]] = []
        offsets: list[int] = []

        for name, value in serialized_void.items():
            if isinstance(value, dict):
                dtype_value = value.get('__dtype__')
                if isinstance(dtype_value, str):
                    names.append(name)
                    formats.append(numpy.dtype(dtype_value))
                    alignment = value.get('__alignment__')
                    if isinstance(alignment, int):
                        offsets.append(alignment)
                    else:
                        offsets.append(0)

        if not names:
            return None

        if any(offsets):
            return numpy.dtype({'names': names, 'formats': formats, 'offsets': offsets})
        else:
            return numpy.dtype(list(zip(names, formats)))

    shape = serialized.get('__shape__')
    dtype = serialized.get('__dtype__')
    data = serialized.get('__data__')

    assert isinstance(dtype, str)  # Can only be false if the function was misused
    assert data is not None
    np_dtype = numpy.dtype(dtype) if dtype else None
    return_data: DM_FILE_TYPES | None = None
    assert np_dtype is not None
    if isinstance(np_dtype, numpy.dtypes.VoidDType):
        assert isinstance(data, dict)
        value = data.get('__data__')
        if isinstance(value, tuple):
            void_data = value
        elif isinstance(value, list):
            void_data = tuple(value)
        else:
            void_data = (value, )
            if np_dtype.fields is None or not hasattr(np_dtype.fields, "keys") or len(void_data) != len(np_dtype.fields.keys()):
                raise ValueError(f"Unable to deserialize {data}: Mismatched number of fields.")
        void_fields = data.get('__fields__')
        if isinstance(void_fields, dict):
            data_dtype = deserialize_dtype(void_fields)
            return_data = numpy.array(void_data, dtype=data_dtype)[()]
    elif numpy.issubdtype(np_dtype, numpy.ndarray):
        shape = typing.cast(typing.Tuple[int], shape)
        return_data = numpy.array(data).reshape(shape)
    elif numpy.issubdtype(np_dtype, numpy.bytes_):
        data = typing.cast(str, data)
        return_data = numpy.bytes_(data.encode())
    elif numpy.issubdtype(np_dtype, numpy.bool_):
        return_data = numpy.bool_(data)
    elif numpy.issubdtype(np_dtype, numpy.integer) or numpy.issubdtype(np_dtype, numpy.floating):
        return_data = numpy.asarray(data, dtype=np_dtype)[()]

    if return_data is not None:
        return return_data
    raise TypeError(f"{dtype!r}, {shape!r} {data!r} {type(data)!r} is not supported.")


def squash_metadata_dict(metadata_dict: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Removes the stored types made by serialize_dm_attrs_into_swift_metadata leaving only the data in the swift metadata dict.

    This is done so the swift metadata structure for everything but dm_metadata is identical to what would be seen in dm3/4.
    """

    def _convert_attrs(attrs_dict: typing.Dict[str, typing.Any], base_dict: dict[str, typing.Any]) -> None:
        for key, value in attrs_dict.items():
            if isinstance(value, dict):
                data = value.get('__data__')  # Serialized attributes store the value at the key data
                if data is not None:
                    while isinstance(data, dict) and data.get('__data__') is not None:
                        data = data['__data__']  # The np.void data is another level deeper
                    value = data
            assert (isinstance(key, str))
            base_dict.update({key: value})

    def _recursive_squash_dict(base_dict: dict[str, typing.Any]) -> dict[str, typing.Any]:
        new_dict: dict[str, int | float | dict[str, typing.Any]] = {}
        for key, value in base_dict.items():
            if key == '__attrs__':
                _convert_attrs(value, new_dict)
            elif isinstance(value, dict):
                new_dict[key] = _recursive_squash_dict(base_dict[key])
        return new_dict

    return _recursive_squash_dict(metadata_dict)

# h5py Groups to dictionaries


def convert_group_to_dict(group: h5py.Group) -> dict[str, typing.Any]:
    """Converts h5py groups to a dict, with nested groups becoming nested dicts, and stores attrs as a nested dict, ignores datasets.

    Recursively visit all the nodes in the group converting attrs data to a type that can be stored in swift metadata DM_DICT_TYPES.
    """
    def _convert_attrs_to_dict(attrs: h5py.AttributeManager) -> dict[str, typing.Any]:
        """Converts group attributes into a dict, serializing the data from dm5 types"""
        attrs_dict: dict[str, typing.Any] = dict()
        for key, value in attrs.items():
            if isinstance(key, bytes):  # Some of the keys were encoded so this was required
                key = decode_bytes_to_str(key)
            assert (isinstance(key, str))
            if '.__meta__' in key:
                continue
            if isinstance(value, numpy.void) and key + '.__meta__' in attrs.keys():
                meta_value = attrs[key + '.__meta__']
                if isinstance(meta_value, (numpy.bytes_, bytes, bytearray)):
                    meta_json = decode_bytes_to_str(bytes(meta_value))
                else:
                    meta_json = str(meta_value)
                meta = json.loads(meta_json)
                value = _convert_void_to_sequence(value, value.dtype, meta)
            else:
                value = _serialize_dm_attrs_into_swift_metadata(value)

            attrs_dict[key] = value

        return attrs_dict

    def _recursive_group_to_dict(base_group: h5py.Group) -> dict[str, typing.Any]:
        base_dict: dict[str, typing.Any] = dict()
        attributes = base_group.attrs
        if len(attributes.items()) != 0:
            base_dict['__attrs__'] = _convert_attrs_to_dict(attributes)
        for key, value in base_group.items():
            if isinstance(value, h5py.Group):
                base_dict[key] = _recursive_group_to_dict(value)
            elif isinstance(value, h5py.Dataset):
                pass
            else:
                raise TypeError(f"Unknown type found in group {base_group}, value: {value} type: {type(value)}")
        return base_dict

    return _recursive_group_to_dict(group)


def convert_dict_to_group(base_dict: typing.Dict[str, typing.Any], group: h5py.Group) -> h5py.Group:
    """Converts a dict to group going though any nested dicts rebuilding the groups, and attached attrs. Datasets are currently ignored.

    If a nested dict named attrs is encountered then it will use that to create attributes for the parent group
    """
    def _convert_dict_to_attrs(attrs_dict: typing.Dict[str, typing.Any], base_group: h5py.Group) -> None:
        for key, value in attrs_dict.items():
            if isinstance(value, dict):
                value = _deserialize_dm_attrs_from_swift_metadata(value)
            save_attr_to_group(name=key, swift_value=value, group=base_group)

    def _recursive_dict_to_group(recursive_dict: typing.Dict[str, typing.Any], top_group: h5py.Group) -> h5py.Group:
        for key, value in recursive_dict.items():
            if key is None or key == '':  # ensure the key is a valid name for h5py
                continue
            if key == '__attrs__':
                _convert_dict_to_attrs(value, top_group)
            elif isinstance(value, dict):
                new_group = get_or_create_group(top_group, key)
                _recursive_dict_to_group(recursive_dict[key], new_group)
            elif isinstance(value, (str, list, tuple, float, int)):
                save_attr_to_group(name=key, swift_value=value, group=top_group)
        return top_group

    return _recursive_dict_to_group(base_dict, group)
