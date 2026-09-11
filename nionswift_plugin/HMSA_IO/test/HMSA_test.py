import pathlib
import tempfile
import typing
import unittest
import xml.etree.ElementTree as ElementTree

import numpy
import numpy.typing

from nion.data import Calibration
from nion.data import DataAndMetadata
from nionswift_plugin import HMSA_IO


_UID = bytes.fromhex("0102030405060708")
_PYHMSA_RESOURCES = pathlib.Path(__file__).parent / "resources"

_DATUM_TYPE_NAMES = {
    numpy.dtype(numpy.uint8): "byte",
    numpy.dtype(numpy.int16): "int16",
    numpy.dtype(numpy.uint16): "uint16",
    numpy.dtype(numpy.int32): "int32",
    numpy.dtype(numpy.uint32): "uint32",
    numpy.dtype(numpy.int64): "int64",
    numpy.dtype(numpy.float32): "float",
    numpy.dtype(numpy.float64): "double",
}


class API:
    def create_calibration(
        self,
        offset: float | None = None,
        scale: float | None = None,
        units: str | None = None,
    ) -> Calibration.Calibration:
        return Calibration.Calibration(offset, scale, units)

    def create_data_descriptor(
        self,
        is_sequence: bool,
        collection_dimension_count: int,
        datum_dimension_count: int,
    ) -> DataAndMetadata.DataDescriptor:
        return DataAndMetadata.DataDescriptor(
            is_sequence,
            collection_dimension_count,
            datum_dimension_count,
        )

    def create_data_and_metadata(
        self,
        data: numpy.typing.NDArray[typing.Any],
        intensity_calibration: Calibration.Calibration | None = None,
        dimensional_calibrations: list[Calibration.Calibration] | None = None,
        metadata: dict[str, typing.Any] | None = None,
        timestamp: typing.Any = None,
        data_descriptor: DataAndMetadata.DataDescriptor | None = None,
    ) -> DataAndMetadata.DataAndMetadata:
        return DataAndMetadata.new_data_and_metadata(
            data,
            intensity_calibration=intensity_calibration,
            dimensional_calibrations=dimensional_calibrations,
            metadata=metadata,
            timestamp=timestamp,
            data_descriptor=data_descriptor,
        )


def _add_number(parent: ElementTree.Element, tag: str, value: int) -> ElementTree.Element:
    element = ElementTree.SubElement(parent, tag, {"DataType": "int64"})
    element.text = str(value)
    return element


def _write_pair(
    directory: pathlib.Path,
    desired_data: numpy.typing.NDArray[typing.Any],
    data_tag: str,
    data_class: str,
    collection_dimensions: list[tuple[str, int]],
    datum_dimensions: list[tuple[str, int]],
    *,
    declared_length: int | None = None,
    include_calibrations: bool = False,
    inline_value: str | None = None,
) -> pathlib.Path:
    dtype = numpy.dtype(desired_data.dtype).newbyteorder("<")
    if collection_dimensions and datum_dimensions:
        axes = tuple(range(len(collection_dimensions), desired_data.ndim)) + tuple(
            range(len(collection_dimensions))
        )
        storage_data = numpy.transpose(desired_data, axes)
    else:
        storage_data = desired_data
    binary_data = numpy.asarray(storage_data, dtype=dtype).tobytes(order="F")

    root = ElementTree.Element(
        "MSAHyperDimensionalDataFile",
        {
            "Version": "1.0",
            "UID": _UID.hex().upper(),
            "{http://www.w3.org/XML/1998/namespace}lang": "en-US",
        },
    )
    header = ElementTree.SubElement(root, "Header")
    ElementTree.SubElement(header, "Title").text = "Synthetic spectrum image"
    conditions = ElementTree.SubElement(root, "Conditions")
    if include_calibrations:
        acquisition = ElementTree.SubElement(conditions, "Acquisition", {"Class": "Raster/XY", "ID": "Scan"})
        x_step = ElementTree.SubElement(acquisition, "XStepSize", {"DataType": "float", "Unit": "um"})
        x_step.text = "0.25"
        y_step = ElementTree.SubElement(acquisition, "YStepSize", {"DataType": "float", "Unit": "um"})
        y_step.text = "0.5"
        detector = ElementTree.SubElement(conditions, "Detector", {"Class": "Spectrometer/XEDS", "ID": "EDS"})
        calibration = ElementTree.SubElement(detector, "Calibration", {"Class": "Linear"})
        ElementTree.SubElement(calibration, "Unit").text = "eV"
        gain = ElementTree.SubElement(calibration, "Gain", {"DataType": "float"})
        gain.text = "2.5"
        offset = ElementTree.SubElement(calibration, "Offset", {"DataType": "float"})
        offset.text = "-237.0"

    data_container = ElementTree.SubElement(root, "Data")
    data_element = ElementTree.SubElement(data_container, data_tag, {"Class": data_class, "Name": "Test"})
    data_element.text = inline_value
    _add_number(data_element, "DataOffset", 8)
    _add_number(data_element, "DataLength", len(binary_data) if declared_length is None else declared_length)
    datum_type = ElementTree.SubElement(data_element, "DatumType", {"SizeInBytes": str(dtype.itemsize)})
    datum_type.text = _DATUM_TYPE_NAMES[numpy.dtype(dtype.name)]

    datum_dimensions_element = ElementTree.SubElement(data_element, "DatumDimensions")
    for name, size in datum_dimensions:
        dimension = ElementTree.SubElement(
            datum_dimensions_element,
            "Dimension",
            {"DataType": "uint32", "Name": name},
        )
        dimension.text = str(size)

    collection_dimensions_element = ElementTree.SubElement(data_element, "CollectionDimensions")
    for name, size in collection_dimensions:
        dimension = ElementTree.SubElement(
            collection_dimensions_element,
            "Dimension",
            {"DataType": "uint32", "Name": name},
        )
        dimension.text = str(size)
    ElementTree.SubElement(data_element, "IncludeConditions")

    xml_path = directory / "test.xml"
    hmsa_path = directory / "test.hmsa"
    ElementTree.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    hmsa_path.write_bytes(_UID + binary_data)
    return hmsa_path


class TestHMSAIOClass(unittest.TestCase):
    def test_reads_pyhmsa_reference_spectral_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = pathlib.Path(temporary_directory)
            xml_path = directory / "reference.xml"
            hmsa_path = directory / "reference.hmsa"
            xml_path.write_text(
                (_PYHMSA_RESOURCES / "imageraster2dspectral.xml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            hmsa_path.write_bytes(
                bytes.fromhex((_PYHMSA_RESOURCES / "imageraster2dspectral.hmsa.hex").read_text(encoding="ascii"))
            )

            xdata = HMSA_IO.read_hmsa(API(), hmsa_path)

            self.assertEqual((6, 5, 7), xdata.data_shape)
            self.assertEqual(0, xdata.data[0, 0, 0])
            self.assertEqual(15, xdata.data[-1, -1, -1])
            self.assertEqual(2, xdata.collection_dimension_count)
            self.assertEqual(1, xdata.datum_dimension_count)

    def test_reads_spectral_image_in_collection_then_datum_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            hmsa_data = numpy.arange(2 * 3 * 4, dtype=numpy.uint16).reshape((2, 3, 4))
            hmsa_path = _write_pair(
                pathlib.Path(temporary_directory),
                hmsa_data,
                "ImageRaster",
                "2D/Spectral",
                [("X", 2), ("Y", 3)],
                [("Channel", 4)],
                include_calibrations=True,
            )

            xdata = HMSA_IO.HMSAIODelegate(API()).read_data_and_metadata("hmsa", str(hmsa_path))

            expected_data = numpy.transpose(hmsa_data, (1, 0, 2))
            self.assertTrue(numpy.array_equal(expected_data, xdata.data))
            self.assertEqual((3, 2, 4), xdata.data_shape)
            self.assertEqual(2, xdata.collection_dimension_count)
            self.assertEqual(1, xdata.datum_dimension_count)
            self.assertEqual(0.5, xdata.dimensional_calibrations[0].scale)
            self.assertEqual(0.25, xdata.dimensional_calibrations[1].scale)
            self.assertEqual(-237.0, xdata.dimensional_calibrations[2].offset)
            self.assertEqual(2.5, xdata.dimensional_calibrations[2].scale)
            self.assertEqual("eV", xdata.dimensional_calibrations[2].units)
            self.assertEqual("Synthetic spectrum image", xdata.metadata["hmsa"]["header"]["Title"])

    def test_reads_one_dimensional_analysis_from_xml_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            desired_data = numpy.array([3, 5, 8, 13], dtype=numpy.int32)
            hmsa_path = _write_pair(
                pathlib.Path(temporary_directory),
                desired_data,
                "Analysis",
                "1D",
                [],
                [("Channel", 4)],
            )

            xdata = HMSA_IO.read_hmsa(API(), hmsa_path.with_suffix(".xml"))

            self.assertTrue(numpy.array_equal(desired_data, xdata.data))
            self.assertEqual(0, xdata.collection_dimension_count)
            self.assertEqual(1, xdata.datum_dimension_count)

    def test_reads_inline_scalar_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            hmsa_path = _write_pair(
                pathlib.Path(temporary_directory),
                numpy.asarray(123, dtype=numpy.int64),
                "Analysis",
                "0D",
                [],
                [],
                inline_value="123.0",
            )

            xdata = HMSA_IO.read_hmsa(API(), hmsa_path)

            self.assertEqual((), xdata.data_shape)
            self.assertEqual(123, int(xdata.data))
            self.assertEqual(0, xdata.collection_dimension_count)
            self.assertEqual(0, xdata.datum_dimension_count)

    def test_reads_hyperimage_in_nion_spatial_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            hmsa_data = numpy.arange(2 * 3 * 4 * 5, dtype=numpy.uint8).reshape((2, 3, 4, 5))
            hmsa_path = _write_pair(
                pathlib.Path(temporary_directory),
                hmsa_data,
                "ImageRaster",
                "2D/Hyperimage",
                [("X", 2), ("Y", 3)],
                [("U", 4), ("V", 5)],
            )

            xdata = HMSA_IO.read_hmsa(API(), hmsa_path)

            expected_data = numpy.transpose(hmsa_data, (1, 0, 3, 2))
            self.assertTrue(numpy.array_equal(expected_data, xdata.data))
            self.assertEqual(numpy.dtype(numpy.uint16), xdata.data_dtype)
            self.assertEqual((3, 2, 5, 4), xdata.data_shape)
            self.assertEqual(2, xdata.collection_dimension_count)
            self.assertEqual(2, xdata.datum_dimension_count)

    def test_rejects_uid_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            hmsa_path = _write_pair(
                pathlib.Path(temporary_directory),
                numpy.array([1, 2], dtype=numpy.uint8),
                "Analysis",
                "1D",
                [],
                [("Channel", 2)],
            )
            hmsa_path.write_bytes(bytes(8) + hmsa_path.read_bytes()[8:])

            with self.assertRaisesRegex(ValueError, "UID mismatch"):
                HMSA_IO.read_hmsa(API(), hmsa_path)

    def test_rejects_inconsistent_data_length(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            hmsa_path = _write_pair(
                pathlib.Path(temporary_directory),
                numpy.array([1, 2], dtype=numpy.uint8),
                "Analysis",
                "1D",
                [],
                [("Channel", 2)],
                declared_length=1,
            )

            with self.assertRaisesRegex(ValueError, "DataLength mismatch"):
                HMSA_IO.read_hmsa(API(), hmsa_path)

    def test_rejects_dimensions_that_do_not_match_dataset_class(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            hmsa_path = _write_pair(
                pathlib.Path(temporary_directory),
                numpy.array([1, 2], dtype=numpy.uint8),
                "Analysis",
                "1D",
                [],
                [("U", 2)],
            )

            with self.assertRaisesRegex(ValueError, "DatumDimensions must be"):
                HMSA_IO.read_hmsa(API(), hmsa_path)

    def test_requires_xml_and_binary_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            hmsa_path = pathlib.Path(temporary_directory) / "missing.hmsa"
            hmsa_path.write_bytes(_UID)

            with self.assertRaisesRegex(FileNotFoundError, "XML descriptor is missing"):
                HMSA_IO.read_hmsa(API(), hmsa_path)

    def test_delegate_is_read_only(self) -> None:
        delegate = HMSA_IO.HMSAIODelegate(API())
        xdata = DataAndMetadata.new_data_and_metadata(numpy.zeros(1))
        self.assertFalse(delegate.can_write_data_and_metadata(xdata, "hmsa"))
