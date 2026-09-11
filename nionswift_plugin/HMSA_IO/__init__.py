"""Read MSA/MAS hyper-dimensional spectral (HMSA) file pairs."""

from __future__ import annotations

import gettext
import math
import pathlib
import typing
import xml.etree.ElementTree as ElementTree

import numpy
import numpy.typing

from nion.data import DataAndMetadata


_ = gettext.gettext

_XML_LANGUAGE = "{http://www.w3.org/XML/1998/namespace}lang"

_DATA_TYPES = {
    "byte": numpy.dtype(numpy.uint8),
    "int16": numpy.dtype(numpy.int16),
    "uint16": numpy.dtype(numpy.uint16),
    "int32": numpy.dtype(numpy.int32),
    "uint32": numpy.dtype(numpy.uint32),
    "int64": numpy.dtype(numpy.int64),
    "float": numpy.dtype(numpy.float32),
    "double": numpy.dtype(numpy.float64),
}

_DATA_CLASS_DIMENSIONS = {
    ("Analysis", "0D"): ((), ()),
    ("Analysis", "1D"): ((), ("Channel",)),
    ("Analysis", "2D"): ((), ("U", "V")),
    ("AnalysisList", "0D"): (("Analysis",), ()),
    ("AnalysisList", "1D"): (("Analysis",), ("Channel",)),
    ("AnalysisList", "2D"): (("Analysis",), ("U", "V")),
    ("ImageRaster", "2D"): (("X", "Y"), ()),
    ("ImageRaster", "2D/Spectral"): (("X", "Y"), ("Channel",)),
    ("ImageRaster", "2D/Hyperimage"): (("X", "Y"), ("U", "V")),
}


def _paired_paths(file_path: str | pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    path = pathlib.Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".hmsa":
        xml_path = path.with_suffix(".xml")
        hmsa_path = path
    elif suffix == ".xml":
        xml_path = path
        hmsa_path = path.with_suffix(".hmsa")
    else:
        raise ValueError("HMSA input must have an .hmsa or .xml extension")

    if not xml_path.is_file():
        raise FileNotFoundError(f"HMSA XML descriptor is missing: {xml_path}")
    if not hmsa_path.is_file():
        raise FileNotFoundError(f"HMSA binary data file is missing: {hmsa_path}")
    return xml_path, hmsa_path


def _required_child(element: ElementTree.Element, tag: str) -> ElementTree.Element:
    child = element.find(tag)
    if child is None:
        raise ValueError(f"Required HMSA element is missing: {tag}")
    return child


def _integer_text(element: ElementTree.Element) -> int:
    if element.text is None:
        raise ValueError(f"HMSA element has no numeric value: {element.tag}")
    try:
        return int(element.text.strip())
    except ValueError as error:
        raise ValueError(f"Invalid HMSA integer in {element.tag}: {element.text!r}") from error


def _dimensions(data_element: ElementTree.Element, tag: str) -> list[tuple[str, int]]:
    dimensions_element = _required_child(data_element, tag)
    dimensions: list[tuple[str, int]] = []
    for dimension in dimensions_element.findall("Dimension"):
        name = dimension.get("Name")
        if not name:
            raise ValueError(f"Unnamed HMSA dimension in {tag}")
        size = _integer_text(dimension)
        if size <= 0:
            raise ValueError(f"HMSA dimension {name!r} must be greater than zero")
        if any(existing_name == name for existing_name, _ in dimensions):
            raise ValueError(f"Duplicate HMSA dimension {name!r} in {tag}")
        dimensions.append((name, size))
    return dimensions


def _canonical_dimensions(
    dimensions: list[tuple[str, int]],
    expected_names: tuple[str, ...],
    tag: str,
) -> list[tuple[str, int]]:
    dimensions_by_name = dict(dimensions)
    if set(dimensions_by_name) != set(expected_names):
        actual_names = tuple(dimensions_by_name)
        raise ValueError(f"HMSA {tag} must be {expected_names}; found {actual_names}")
    return [(name, dimensions_by_name[name]) for name in expected_names]


def _nion_dimensions(dimensions: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Return spatial dimensions in Nion's row/column (height/width) order."""
    dimension_names = tuple(name for name, _ in dimensions)
    if dimension_names in (("X", "Y"), ("U", "V")):
        return list(reversed(dimensions))
    return dimensions


def _data_type(data_element: ElementTree.Element) -> numpy.dtype[typing.Any]:
    datum_type = _required_child(data_element, "DatumType")
    name = (datum_type.text or "").strip()
    try:
        dtype = _DATA_TYPES[name].newbyteorder("<")
    except KeyError as error:
        raise ValueError(f"Unsupported HMSA datum type: {name!r}") from error

    try:
        declared_size = int(datum_type.attrib["SizeInBytes"])
    except (KeyError, ValueError) as error:
        raise ValueError("HMSA DatumType must declare a valid SizeInBytes") from error
    if declared_size != dtype.itemsize:
        raise ValueError(
            f"HMSA datum size mismatch: {name!r} declares {declared_size}, expected {dtype.itemsize}"
        )
    return dtype


def _validate_uid(root: ElementTree.Element, hmsa_path: pathlib.Path) -> str:
    xml_uid = root.get("UID")
    if not xml_uid:
        raise ValueError("HMSA XML descriptor does not declare a UID")
    try:
        expected_uid = bytes.fromhex(xml_uid)
    except ValueError as error:
        raise ValueError(f"Invalid HMSA UID: {xml_uid!r}") from error
    if len(expected_uid) != 8:
        raise ValueError("HMSA UID must contain exactly eight bytes")

    with hmsa_path.open("rb") as hmsa_file:
        actual_uid = hmsa_file.read(8)
    if actual_uid != expected_uid:
        raise ValueError(
            f"HMSA UID mismatch: XML declares {xml_uid.upper()}, "
            f"binary file contains {actual_uid.hex().upper()}"
        )
    return xml_uid.upper()


def _read_array(
    data_element: ElementTree.Element,
    hmsa_path: pathlib.Path,
    dtype: numpy.dtype[typing.Any],
    collection_dimensions: list[tuple[str, int]],
    datum_dimensions: list[tuple[str, int]],
) -> numpy.typing.NDArray[typing.Any]:
    storage_dimensions = datum_dimensions + collection_dimensions
    storage_shape = tuple(size for _, size in storage_dimensions)
    item_count = math.prod(storage_shape) if storage_shape else 1

    inline_value = (data_element.text or "").strip()
    if not storage_shape and inline_value:
        try:
            value = dtype.type(inline_value)
        except (OverflowError, ValueError):
            try:
                value = dtype.type(float(inline_value))
            except (OverflowError, ValueError) as error:
                raise ValueError(f"Invalid inline HMSA scalar value: {inline_value!r}") from error
        return numpy.asarray(value, dtype=dtype)

    data_offset = _integer_text(_required_child(data_element, "DataOffset"))
    data_length = _integer_text(_required_child(data_element, "DataLength"))
    expected_length = item_count * dtype.itemsize
    if data_offset < 8:
        raise ValueError("HMSA DataOffset must not overlap the eight-byte UID")
    if data_length != expected_length:
        raise ValueError(
            f"HMSA DataLength mismatch: descriptor declares {data_length} bytes, "
            f"dimensions require {expected_length} bytes"
        )
    if data_offset + data_length > hmsa_path.stat().st_size:
        raise ValueError("HMSA binary data extends beyond the end of the file")

    data = numpy.fromfile(hmsa_path, dtype=dtype, count=item_count, offset=data_offset)
    if data.size != item_count:
        raise ValueError(f"HMSA binary data is incomplete: read {data.size} of {item_count} values")

    if not storage_shape:
        return data.reshape(())

    data = data.reshape(storage_shape, order="F")
    nion_dimensions = _nion_dimensions(collection_dimensions) + _nion_dimensions(datum_dimensions)
    storage_dimension_names = [name for name, _ in storage_dimensions]
    axes = tuple(storage_dimension_names.index(name) for name, _ in nion_dimensions)
    if axes != tuple(range(data.ndim)):
        data = numpy.transpose(data, axes)
    return numpy.ascontiguousarray(data)


def _ensure_nion_scalar_dtype(
    data: numpy.typing.NDArray[typing.Any],
) -> numpy.typing.NDArray[typing.Any]:
    # Nion reserves trailing uint8 dimensions of length 3 or 4 for RGB(A).
    # HMSA's supported datum classes are scalar, so widen these ambiguous
    # arrays losslessly instead of allowing them to be misclassified as color.
    if data.dtype == numpy.uint8 and data.ndim > 1 and data.shape[-1] in (3, 4):
        return data.astype(numpy.uint16)
    return data


def _metadata_value(element: ElementTree.Element) -> typing.Any:
    children = list(element)
    attributes = dict(element.attrib)
    text = (element.text or "").strip()
    if not children and not attributes:
        return text

    result: dict[str, typing.Any] = {}
    if attributes:
        result["attributes"] = attributes
    if text:
        result["value"] = text
    for child in children:
        child_value = _metadata_value(child)
        existing = result.get(child.tag)
        if existing is None:
            result[child.tag] = child_value
        elif isinstance(existing, list):
            existing.append(child_value)
        else:
            result[child.tag] = [existing, child_value]
    return result


def _float_child(element: ElementTree.Element, tag: str, default: float) -> float:
    child = element.find(tag)
    if child is None or child.text is None:
        return default
    try:
        return float(child.text)
    except ValueError as error:
        raise ValueError(f"Invalid HMSA calibration value in {tag}: {child.text!r}") from error


def _linear_spectral_calibration(root: ElementTree.Element) -> tuple[float, float, str | None] | None:
    for calibration in root.findall("./Conditions/Detector/Calibration"):
        if calibration.get("Class") == "Linear":
            gain = _float_child(calibration, "Gain", 1.0)
            offset = _float_child(calibration, "Offset", 0.0)
            unit_element = calibration.find("Unit")
            units = unit_element.text.strip() if unit_element is not None and unit_element.text else None
            return offset, gain, units
    return None


def _raster_calibrations(root: ElementTree.Element) -> dict[str, tuple[float, float, str | None]]:
    calibrations: dict[str, tuple[float, float, str | None]] = {}
    for acquisition in root.findall("./Conditions/Acquisition"):
        if acquisition.get("Class") != "Raster/XY":
            continue
        for dimension_name, tag in (("X", "XStepSize"), ("Y", "YStepSize")):
            element = acquisition.find(tag)
            if element is not None and element.text:
                try:
                    scale = float(element.text)
                except ValueError as error:
                    raise ValueError(f"Invalid HMSA raster calibration in {tag}: {element.text!r}") from error
                calibrations[dimension_name] = (0.0, scale, element.get("Unit"))
        break
    return calibrations


def _create_calibrations(
    api: typing.Any,
    root: ElementTree.Element,
    nion_dimensions: list[tuple[str, int]],
) -> list[typing.Any]:
    dimension_names = [name for name, _ in nion_dimensions]
    calibration_values = _raster_calibrations(root)
    spectral_calibration = _linear_spectral_calibration(root)
    if spectral_calibration is not None:
        calibration_values["Channel"] = spectral_calibration

    calibrations = []
    for dimension_name in dimension_names:
        values = calibration_values.get(dimension_name)
        if values is None:
            calibrations.append(api.create_calibration())
        else:
            offset, scale, units = values
            calibrations.append(api.create_calibration(offset=offset, scale=scale, units=units))
    return calibrations


def _create_metadata(
    root: ElementTree.Element,
    data_element: ElementTree.Element,
    uid: str,
) -> dict[str, typing.Any]:
    hmsa_metadata: dict[str, typing.Any] = {
        "version": root.get("Version", ""),
        "uid": uid,
        "language": root.get(_XML_LANGUAGE, ""),
        "data_name": data_element.get("Name", ""),
        "data_template": data_element.tag,
        "data_class": data_element.get("Class", ""),
        "datum_type": (_required_child(data_element, "DatumType").text or "").strip(),
    }
    header = root.find("Header")
    if header is not None:
        hmsa_metadata["header"] = _metadata_value(header)
    conditions = root.find("Conditions")
    if conditions is not None:
        hmsa_metadata["conditions"] = _metadata_value(conditions)
    return {"hmsa": hmsa_metadata}


def read_hmsa(api: typing.Any, file_path: str | pathlib.Path) -> DataAndMetadata.DataAndMetadata:
    """Read one dataset from a paired HMSA XML/binary file."""
    xml_path, hmsa_path = _paired_paths(file_path)
    try:
        root = ElementTree.parse(xml_path).getroot()
    except ElementTree.ParseError as error:
        raise ValueError(f"Invalid HMSA XML descriptor: {xml_path}") from error
    if root.tag != "MSAHyperDimensionalDataFile":
        raise ValueError(f"Unexpected HMSA XML root element: {root.tag!r}")
    if root.get("Version") != "1.0":
        raise ValueError(f"Unsupported HMSA version: {root.get('Version')!r}")

    uid = _validate_uid(root, hmsa_path)
    data_container = _required_child(root, "Data")
    data_elements = list(data_container)
    if len(data_elements) != 1:
        raise ValueError(f"This HMSA reader currently requires exactly one dataset; found {len(data_elements)}")
    data_element = data_elements[0]
    data_class = data_element.get("Class", "")
    data_class_key = (data_element.tag, data_class)
    try:
        expected_collection_names, expected_datum_names = _DATA_CLASS_DIMENSIONS[data_class_key]
    except KeyError as error:
        raise ValueError(f"Unsupported HMSA dataset class: {data_element.tag}/{data_class}") from error

    dtype = _data_type(data_element)
    collection_dimensions = _canonical_dimensions(
        _dimensions(data_element, "CollectionDimensions"),
        expected_collection_names,
        "CollectionDimensions",
    )
    datum_dimensions = _canonical_dimensions(
        _dimensions(data_element, "DatumDimensions"),
        expected_datum_names,
        "DatumDimensions",
    )
    data = _read_array(data_element, hmsa_path, dtype, collection_dimensions, datum_dimensions)
    data = _ensure_nion_scalar_dtype(data)
    data_descriptor = api.create_data_descriptor(False, len(collection_dimensions), len(datum_dimensions))
    nion_dimensions = _nion_dimensions(collection_dimensions) + _nion_dimensions(datum_dimensions)
    calibrations = _create_calibrations(api, root, nion_dimensions)
    metadata = _create_metadata(root, data_element, uid)
    return api.create_data_and_metadata(
        data,
        dimensional_calibrations=calibrations,
        metadata=metadata,
        data_descriptor=data_descriptor,
    )


class HMSAIODelegate:
    """Nion Swift I/O delegate for read-only HMSA data."""

    def __init__(self, api: typing.Any) -> None:
        self.__api = api
        self.io_handler_id = "hmsa-io-handler"
        self.io_handler_name = _("MSA/MAS Hyper-dimensional Spectral Files")
        # Register only the distinctive extension so unrelated XML files are not claimed.
        self.io_handler_extensions = ["hmsa"]

    def read_data_and_metadata(self, extension: str, file_path: str) -> DataAndMetadata.DataAndMetadata:
        return read_hmsa(self.__api, file_path)

    def can_write_data_and_metadata(
        self,
        data_and_metadata: DataAndMetadata.DataAndMetadata,
        extension: str,
    ) -> bool:
        return False

    def write_data_and_metadata(
        self,
        data_and_metadata: DataAndMetadata.DataAndMetadata,
        file_path_str: str,
        extension: str,
    ) -> None:
        raise NotImplementedError("HMSA export is not supported")


class HMSAIOExtension:
    """Register the HMSA I/O delegate with Nion Swift."""

    extension_id = "nion.swift.extensions.hmsa_io"

    def __init__(self, api_broker: typing.Any) -> None:
        api = api_broker.get_api(version="~1.0")
        self.__io_handler_ref = api.create_data_and_metadata_io_handler(HMSAIODelegate(api))

    def close(self) -> None:
        self.__io_handler_ref.close()
        self.__io_handler_ref = None
