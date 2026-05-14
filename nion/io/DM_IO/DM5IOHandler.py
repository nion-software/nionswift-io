import dataclasses
import datetime
import gettext
import typing
import h5py
import numpy

from nion.io.DM_IO import dm3_image_utils
from nion.io.DM_IO import DM5Utils
from nion.data import DataAndMetadata
from nion.data import Calibration
from nion.utils import DateTime
_ = gettext.gettext

_NDArray = numpy.typing.NDArray[typing.Any]


def _read_calibration(calibration: h5py.Group) -> Calibration.Calibration:
    origin = calibration.attrs.get('Origin', 0.0)
    scale = calibration.attrs.get('Scale', 1.0)
    units = calibration.attrs.get('Units')
    units_str = ""
    if isinstance(units, bytes):
        units_str = DM5Utils.decode_bytes_to_str(units)
    return Calibration.Calibration(-origin * scale, scale, units_str)


def _read_dimensional_calibrations(image_data: h5py.Group) -> list[Calibration.Calibration]:
    calibrations = list[Calibration.Calibration]()
    for _, dimension in image_data.get("Calibrations/Dimension", dict()).items():
        calibrations.append(_read_calibration(dimension))
    return list(reversed(calibrations))


def _read_image_data(file: h5py.File) -> typing.Tuple[h5py.Dataset, h5py.Group, h5py.Group]:
    # Find the index in the image list where the image data is stored
    document_object = file.get("DocumentObjectList/[0]", dict())
    if not hasattr(document_object, "attrs"):
        raise IOError(f"ERROR reading {file.filename}: Malformed file. Unable to determine suitable image source as document object had no attributes.")
    image_source_index = document_object.attrs.get("ImageSource")
    image_source = file.get(f"ImageSourceList/[{image_source_index}]", dict())
    if not hasattr(image_source, "attrs"):
        raise IOError(f"ERROR reading {file.filename}: Malformed file. Unable to determine suitable image source as it had no attributes.")
    image_ref = image_source.attrs.get("ImageRef")
    image_data = file.get(f"ImageList/[{image_ref}]/ImageData")
    if None in (image_source_index, image_ref, image_data):
        raise IOError(f"ERROR reading {file.filename}: Malformed file. Unable to determine suitable image source.")

    dataset = image_data.get("Data", None)
    assert dataset is not None  # No data found in at image data.

    return dataset, image_data, image_ref


def _convert_data_shape(dataset: h5py.Dataset, image_data: h5py.Group,
                        calibrations: list[Calibration.Calibration], is_sequence: bool, is_spectrum: bool) \
        -> tuple[_NDArray, list[Calibration.Calibration], DataAndMetadata.DataDescriptor]:
    """Convert the DM data into the correct shape

    Returns data, calibrations, DataDescriptor in the converted format.
    The conditions were derived from the dm3/4 code.
    """
    # The order of DM datasets is reversed from our version so it has to be reshaped.
    datatype = image_data.attrs.get("DataType", None)

    dimensions_group = image_data.get("Dimensions", None)
    dimensions: list[int] = list(dataset.shape)
    if dimensions_group is not None:
        dimensions = [int(dimensions_group.attrs.get(f"[{i}]")) for i in range(len(dimensions_group.attrs))]

    if datatype is None:
        # TODO export a png image as dm5 from DM to see what the pixel depth is and use that to fallback instead
        print("WARNING: No datatype found for image data, fallback RGBA.")
        datatype = 23  # RGBA_UINT8_3_DATA

    data = numpy.ndarray(dataset.shape, dtype=dataset.dtype)
    dataset.read_direct(data)

    # The order of DM datasets is reversed from our version so it has to be reshaped.
    data_shape = dimensions[::-1]

    move_axis: tuple[int, int] | None = None
    collection_dimension_count = 0  # default case for non-collection non-sequence,spectrum or image
    datum_dimension_count = len(data_shape)
    if data.dtype == numpy.uint8:  # TODO this condition is not tested for
        collection_dimension_count, datum_dimension_count = (0, len(data_shape[:-1]))
    elif len(data_shape) == 3:
        if is_spectrum:
            if data_shape[1] == 1:  # Sequence of spectra or 1d collection of spectra
                collection_dimension_count, datum_dimension_count = (1, 1)
                data.squeeze(axis=1)  # Remove the middle one dimension axis
                data_shape.pop(1)
                move_axis = (0, 1)
                calibrations = [calibrations[2], calibrations[0]]
            else:  # 2d Collection of spectra
                collection_dimension_count, datum_dimension_count = (2, 1)
                move_axis = (0, 2)
                calibrations = list(calibrations[1:]) + [calibrations[0]]
        else:  # Sequence or 1d collection of images
            collection_dimension_count, datum_dimension_count = (1, 2)
    elif len(data_shape) == 4:  # 2d collection of images
        collection_dimension_count, datum_dimension_count = (2, 2)

    if is_spectrum:
        collection_dimension_count += datum_dimension_count - 1
        datum_dimension_count = 1

    if is_sequence and collection_dimension_count > 0:
        collection_dimension_count -= 1
    else:
        is_sequence = False

    data = data.reshape(tuple(data_shape))
    if move_axis is not None:
        data = numpy.moveaxis(data, move_axis[0], move_axis[1])

    if datatype == 23:  # RGBA_UINT8_3_DATA
        assert data.dtype == numpy.uint32
        # The stored type of pixels are uint32. This needs to be converted to a nested array of uint8s. The alpha also has to be discarded so it is RGB.
        uint8_data = data.view(numpy.uint8)  # Reinterpret the stored dtype (uint32) as 4 uint8s. [RGBA, RGBA] -> [R,G,B,A,R,G,B,A]
        uint8_data = uint8_data.reshape(data.shape + (-1,))  # Pack the new 4 uint8s into an array of the original shape. [R,G,B,A,R,G,B,A] -> [[R,G,B,A],[R,G,B,A]]
        data = uint8_data[..., :-1]  # Remove the alpha from each pixel. [[R,G,B,A],[R,G,B,A]] -> [[R,G,B],[R,G,B]]

    return data, calibrations, DataAndMetadata.DataDescriptor(is_sequence, collection_dimension_count, datum_dimension_count)


@dataclasses.dataclass
class TimeInfo:
    timestamp: datetime.datetime
    timezone: str | None
    timezone_offset: str | None


def _read_datetime(image_tags: dict[str, typing.Any]) -> TimeInfo | None:
    timestamp = None
    timestamp_str = DM5Utils.get_from_nested_dict(image_tags, ["__attrs__", "Timestamp"])
    if timestamp_str:
        timestamp = dm3_image_utils.get_datetime_from_timestamp_str(timestamp_str)
    timezone = DM5Utils.get_from_nested_dict(image_tags, ["__attrs__", "Timezone"])
    timezone_offset = DM5Utils.get_from_nested_dict(image_tags, ["__attrs__", "TimezoneOffset"])

    if timestamp is None or not isinstance(timezone, str) or not isinstance(timezone_offset, str):
        filetime = DM5Utils.get_from_nested_dict(image_tags, ["Databar", "__attrs__", "Acquisition Time (OS)"])
        if filetime is not None:
            timestamp = DateTime.get_datetime_from_windows_filetime(filetime)
            timezone = "UTC"
            timezone_offset = "+0000"

    if image_tags and image_tags.get("__attrs__"):
        if DM5Utils.get_from_nested_dict(image_tags, ["__attrs__", "Timestamp"]):
            image_tags["__attrs__"].pop('Timestamp')
        if DM5Utils.get_from_nested_dict(image_tags, ["__attrs__", "TimezoneOffset"]):
            image_tags["__attrs__"].pop('TimezoneOffset')
        if DM5Utils.get_from_nested_dict(image_tags, ["__attrs__", "Timezone"]):
            image_tags["__attrs__"].pop('Timezone')
        if len(image_tags["__attrs__"]) == 0:
            image_tags.pop("__attrs__")
    if timestamp is not None:
        return TimeInfo(timestamp, timezone, timezone_offset)
    else:
        return None


def _read_metadata(image_tags: dict[str, typing.Any], meta_data_attrs: dict[str, typing.Any], unread_dm_metadata_dict: dict[str, typing.Any]) \
        -> dict[str, typing.Any]:
    metadata = dict[str, typing.Any]()
    voltage = DM5Utils.get_from_nested_dict(image_tags, ["Microscope Info", "__attrs__", "Voltage"])
    if voltage is not None:
        metadata.setdefault("hardware_source", dict())["autostem"] = {"high_tension": float(voltage)}

    dm_metadata_signal = DM5Utils.get_from_nested_dict(meta_data_attrs, ["Signal"], "")
    if dm_metadata_signal.lower() == "eels":
        metadata.setdefault("hardware_source", dict())["signal_type"] = dm_metadata_signal

    metadata.update(DM5Utils.squash_metadata_dict(image_tags))
    metadata["__dm_metadata__"] = unread_dm_metadata_dict
    return metadata


def load_image(b_file: typing.BinaryIO) -> DataAndMetadata.DataAndMetadata:
    with h5py.File(b_file, "r") as file:

        dataset, image_data, image_ref = _read_image_data(file)
        dimensional_calibrations = _read_dimensional_calibrations(image_data)

        brightness = image_data.get("Calibrations/Brightness")
        intensity_calibration = _read_calibration(brightness) if brightness else None

        if file.get(f"ImageList/[{image_ref}]/ImageTags") is None:  # Handle no metadata for image
            print(f"WARNING: {file.filename} is missing the ImageTags. Loaded without, but this may produce unexpected results.")
            data, dimensional_calibrations, data_descriptor = _convert_data_shape(dataset, image_data, dimensional_calibrations, False, False)
            return DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations)

        unread_dm_metadata_dict = DM5Utils.convert_group_to_dict(file)
        thumbnails = unread_dm_metadata_dict.get("Thumbnails")
        if thumbnails is not None:
            unread_dm_metadata_dict["Thumbnails"] = {}
            for thumbnail in thumbnails:
                image_item = DM5Utils.get_from_nested_dict(unread_dm_metadata_dict, ["ImageList", f"[{thumbnail}]"])
                if image_item is not None:
                    del unread_dm_metadata_dict["ImageList"][f"[{thumbnail}]"]
            if isinstance(unread_dm_metadata_dict.get("ImageList"), dict):
                unread_dm_metadata_dict["ImageList"]["[0]"] = unread_dm_metadata_dict.get("ImageList", dict()).pop(f"[{image_ref}]")

        image_tags = DM5Utils.get_from_nested_dict(unread_dm_metadata_dict, ["ImageList", f"[{image_ref}]", "ImageTags"], dict())
        meta_data_attrs = DM5Utils.get_from_nested_dict(image_tags, ["Meta Data", "__attrs__"], dict())

        is_spectrum = DM5Utils.get_from_nested_dict(meta_data_attrs, ['Format'], '').lower() in ("spectrum", "spectrum image")
        is_sequence = DM5Utils.get_from_nested_dict(meta_data_attrs, ['IsSequence', "__data__"], False)

        data, dimensional_calibrations, data_descriptor = _convert_data_shape(dataset, image_data, dimensional_calibrations, is_sequence, is_spectrum)
        timeinfo = _read_datetime(image_tags)
        properties = _read_metadata(image_tags, meta_data_attrs, unread_dm_metadata_dict)

        while len(dimensional_calibrations) < data_descriptor.expected_dimension_count:
            dimensional_calibrations.append(Calibration.Calibration())

        return DataAndMetadata.new_data_and_metadata(data,
                                                     data_descriptor=data_descriptor,
                                                     dimensional_calibrations=dimensional_calibrations,
                                                     intensity_calibration=intensity_calibration,
                                                     metadata=properties,
                                                     timestamp=timeinfo.timestamp if timeinfo else None,
                                                     timezone=timeinfo.timezone if timeinfo else None,
                                                     timezone_offset=timeinfo.timezone_offset if timeinfo else None)


def _save_dimensional_calibrations(calibrations: h5py.Group, dimensional_calibrations: typing.Sequence[Calibration.Calibration]) -> None:
    """Set up the dimension list with the attributes"""
    dimension_list = DM5Utils.get_or_create_group(calibrations, "Dimension")
    for i, dimensional_calibration in enumerate(reversed(dimensional_calibrations)):
        origin = 0.0 if dimensional_calibration.scale == 0.0 else -dimensional_calibration.offset / dimensional_calibration.scale
        dimension = DM5Utils.get_or_create_group(dimension_list, f"[{i}]")
        DM5Utils.save_attr_to_group(name="Origin", value=origin, dtype=numpy.float32, group=dimension)
        DM5Utils.save_attr_to_group(name="Scale", value=dimensional_calibration.scale, dtype=numpy.float32, group=dimension)  # dm5 stores scale as a float32. This can introduce floating point issues as python uses 64-bit floats
        DM5Utils.save_attr_to_group(name="Units", value=dimensional_calibration.units, group=dimension)


def _save_intensity_calibration(calibrations: h5py.Group, intensity_calibration: Calibration.Calibration) -> None:
    origin = 0.0 if intensity_calibration.scale == 0.0 else -intensity_calibration.offset / intensity_calibration.scale
    brightness = DM5Utils.get_or_create_group(calibrations, "Brightness")
    DM5Utils.save_attr_to_group(name="Origin", value=origin, dtype=numpy.float32, group=brightness)
    DM5Utils.save_attr_to_group(name="Scale", value=intensity_calibration.scale, dtype=numpy.float32, group=brightness)
    DM5Utils.save_attr_to_group(name="Units", value=intensity_calibration.units, group=brightness)


def _save_datetime(image_tags: h5py.Group, modified: datetime.datetime | None, timezone: str | None, timezone_offset: str | None) -> None:
    date_str, time_str = DM5Utils.get_datetime_as_str(modified, timezone, timezone_offset)
    if date_str is None or time_str is None:
        return
    if modified:
        DM5Utils.save_attr_to_group(name="Timestamp", value=modified.isoformat(), group=image_tags)
    if timezone:
        DM5Utils.save_attr_to_group(name="Timezone", value=timezone, group=image_tags)
    if timezone_offset:
        DM5Utils.save_attr_to_group(name="TimezoneOffset", value=timezone_offset, group=image_tags)

    if image_tags.get('Databar') is not None:
        data_bar = DM5Utils.get_or_create_group(image_tags, name="Databar")
        DM5Utils.save_attr_to_group(name="Acquisition Date", value=date_str, group=data_bar)
        DM5Utils.save_attr_to_group(name="Acquisition Time", value=time_str, group=data_bar)


def _save_metadata_group(image_tags: h5py.Group, dm_format: DM5Utils.DMFormatDataAndMetadata) -> None:
    is_eels = DM5Utils.get_from_nested_dict(dm_format.metadata, ["hardware_source", "signal_type"], "").lower() == "eels"
    is_eels_spectrum = is_eels and len(dm_format.data_shape) == 1 or (len(dm_format.data_shape) == 2 and dm_format.data_shape[0] == 1)
    is_eels_spectrum_image = not is_eels and dm_format.is_two_dimensional_spectrum

    format_value: str | None = None
    signal_value: str | None = None

    if is_eels_spectrum:
        format_value = "Spectrum"
        signal_value = "EELS"
    elif is_eels_spectrum_image:
        format_value = "Spectrum Image"
        signal_value = "EELS"
    if dm_format.datum_dimension_count == 1:  # 1d data is always marked as spectrum
        format_value = "Spectrum image" if dm_format.collection_dimension_count == 2 else "Spectrum"

    # Check if the meta_data_group is needed before creating it in the file
    if signal_value or format_value or (dm_format.is_summed_image and dm_format.is_sequence):
        meta_data_group = DM5Utils.get_or_create_group(image_tags, "Meta Data")
        if signal_value:
            DM5Utils.save_attr_to_group("Signal", value=signal_value, group=meta_data_group)
        if format_value:
            DM5Utils.save_attr_to_group("Format", value=format_value, group=meta_data_group)
        if dm_format.is_summed_image and dm_format.is_sequence:
            DM5Utils.save_attr_to_group("IsSequence", value=True, group=meta_data_group)


def _save_document_object_list_and_annotations(base_group: h5py.Group, dm_metadata: DM5Utils.DMFormatDataAndMetadata) -> None:
    document_object_list = DM5Utils.get_or_create_group(base_group, "DocumentObjectList")
    data_document_object = DM5Utils.get_or_create_group(document_object_list, "[0]")
    DM5Utils.save_attr_to_group(name="ImageSource", value=0, group=data_document_object, dtype=numpy.uint64)
    DM5Utils.save_attr_to_group(name="AnnotationType", value=20, group=data_document_object, dtype=numpy.int32)  # Annotation type 20 is image display
    display_type = 1 if (dm_metadata.data_descriptor.datum_dimension_count > 1 or dm_metadata.data_descriptor.collection_dimension_count > 1) else 3  # Display as graph
    if dm_metadata.is_two_dimensional_spectrum:
        annotation_group_list = DM5Utils.get_or_create_group(data_document_object, "AnnotationGroupList")
        annotation_group = DM5Utils.get_or_create_group(annotation_group_list, "[0]")
        DM5Utils.save_attr_to_group(name="AnnotationType", value=23, group=annotation_group)
        DM5Utils.save_attr_to_group(name="Name", value="SICursor", group=annotation_group)
        DM5Utils.save_attr_to_group(name="Rectangle", value=(0, 0, 1, 1), group=annotation_group, dtype=[('top', '<f4'), ('left', '<f4'), ('bottom', '<f4'), ('right', '<f4')])
    DM5Utils.save_attr_to_group(name="ImageDisplayType", value=display_type, group=data_document_object)


def _save_image_source(base_group: h5py.Group, dm_format: DM5Utils.DMFormatDataAndMetadata, image_list_index: int) -> None:
    image_source_list = DM5Utils.get_or_create_group(base_group, "ImageSourceList")
    image_source = DM5Utils.get_or_create_group(image_source_list, "[0]")  # This location is stored in the DocumentObjectList
    DM5Utils.save_attr_to_group(name="ClassName", value="ImageSource:Simple", group=image_source)
    DM5Utils.save_attr_to_group(name="ImageRef", value=image_list_index, group=image_source, dtype=numpy.uint32)  # The reference in the ImageList
    id_group = DM5Utils.get_or_create_group(image_source, name="Id")
    DM5Utils.save_attr_to_group(name="[0]", value=0, group=id_group, dtype=numpy.uint32)
    if dm_format.is_summed_image:
        DM5Utils.save_attr_to_group(name="ClassName", value="ImageSource:Summed", group=image_source)
        DM5Utils.save_attr_to_group(name="Do Sum", value=True, group=image_source)
        DM5Utils.save_attr_to_group(name="LayerEnd", value=0, group=image_source)
        DM5Utils.save_attr_to_group(name="LayerStart", value=0, group=image_source)
        DM5Utils.save_attr_to_group(name="Summed Dimension", value=len(dm_format.data_shape) - 1, group=image_source)


def _save_required_groups(base_group: h5py.Group, dm_format: DM5Utils.DMFormatDataAndMetadata, name: str) -> None:
    source_image = DM5Utils.get_or_create_group(base_group, "/ImageList/[0]")
    image_data = DM5Utils.get_or_create_group(source_image, "ImageData")
    DM5Utils.save_attr_to_group(name="DataType", value=dm_format.dm_datatype_id, group=image_data)
    DM5Utils.save_attr_to_group(name="PixelDepth", value=dm_format.data.dtype.itemsize, group=image_data)
    dimensions_group = DM5Utils.get_or_create_group(image_data, "Dimensions")
    for i, entry in enumerate(reversed(dm_format.data_shape)):
        DM5Utils.save_attr_to_group(name=f"[{i}]", value=entry, group=dimensions_group, dtype=numpy.uint32)

    DM5Utils.save_attr_to_group(name="Name", value=f"{name}", group=source_image)
    _ = DM5Utils.get_or_create_group(base_group, name="Thumbnails")


def _save_image_tags(image_tags: h5py.Group, dm_format: DM5Utils.DMFormatDataAndMetadata) -> None:
    metadata = dm_format.metadata
    acquisition = DM5Utils.get_from_nested_dict(metadata, ["EELS", "Acquisition"])
    if acquisition:  # Until a solution to store datasets exists the datasets referenced will cause DM to not load, so we remove the referencing groups
        if acquisition.get("HQ Dark Correction"):
            acquisition.pop("HQ Dark Correction")
        if acquisition.get("Saturation fraction"):
            acquisition.pop("Saturation fraction")

    DM5Utils.convert_dict_to_group(dm_format.metadata, image_tags)  # The metadata dictionary is treated as the image tags


def save_image(data_and_metadata: DataAndMetadata.DataAndMetadata, file: typing.BinaryIO) -> None:
    dm_format = DM5Utils.get_dm_format_data_and_metadata(data_and_metadata)
    image_list_index = 0

    with (h5py.File(file, "w") as f):
        name: str = f.name if isinstance(f.name, str) else DM5Utils.decode_bytes_to_str(f.name)
        name = name.split("\\")[-1:][0].replace(".dm5", "")
        base_group = DM5Utils.convert_dict_to_group(dm_format.dm_metadata, f)
        image_list = DM5Utils.get_or_create_group(base_group, "ImageList")
        source_image = DM5Utils.get_or_create_group(image_list, f"[{image_list_index}]")  # The image should be in ImageList:[1], 0 is reserved for thumbnails
        image_data = DM5Utils.get_or_create_group(source_image, "ImageData")
        DM5Utils.create_dataset_chunked_writer(image_data, "Data", dm_format.data, dm_format.data_shape, dm_format.move_axis)
        calibrations = DM5Utils.get_or_create_group(image_data, "Calibrations")

        if dm_format.dimensional_calibrations and len(dm_format.dimensional_calibrations) == len(dm_format.data_shape):
            _save_dimensional_calibrations(calibrations, dm_format.dimensional_calibrations)

        if data_and_metadata.intensity_calibration:
            _save_intensity_calibration(calibrations, data_and_metadata.intensity_calibration)

        image_tags = DM5Utils.get_or_create_group(source_image, "ImageTags")
        _save_image_tags(image_tags, dm_format)

        _save_datetime(image_tags, data_and_metadata.timestamp, data_and_metadata.timezone, data_and_metadata.timezone_offset)
        _save_metadata_group(image_tags, dm_format)
        _save_document_object_list_and_annotations(base_group, dm_format)
        _save_image_source(base_group, dm_format, image_list_index)
        if not dm_format.dm_metadata:  # When dm_metadata exists the required groups were already saved, otherwise they need to be rebuilt
            _save_required_groups(base_group, dm_format, name)
