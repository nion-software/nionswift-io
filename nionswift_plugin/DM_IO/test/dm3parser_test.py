# -*- coding: utf-8 -*-
from __future__ import annotations

import array
import datetime
import io
import itertools
import logging
import pkgutil
import os
import unittest
import shutil
import types
import typing

import h5py
import numpy
import numpy.typing

from nionswift_plugin.DM_IO import parse_dm3
from nionswift_plugin.DM_IO import dm3_image_utils

from nion.data import Calibration
from nion.data import DataAndMetadata


def is_equal(r: typing.Any, data: typing.Any) -> bool:
    # we use this to compare the read and written data
    if isinstance(r, (list, tuple)):
        return len(r) == len(data) and all(is_equal(x, y) for x, y in zip(r, data))
    elif isinstance(r, dict):
        return r.keys() == data.keys() and all(is_equal(r[k], data[k]) for k in r)
    elif isinstance(r, array.array) and isinstance(data, parse_dm3.DataChunkWriter):
        return numpy.array_equal(numpy.array(r), data.data)
    else:
        return bool(r == data)


class TestDM3ImportExportClass(unittest.TestCase):

    def check_write_then_read_matches(self, data: typing.Any, write_func: typing.Callable[..., typing.Any], read_func: typing.Callable[..., typing.Any], _assert: bool = True) -> typing.Any:
        # we confirm that reading a written element returns the same value
        s = io.BytesIO()
        header = write_func(s, data)
        s.seek(0)
        if header is not None:
            r, hy = read_func(s)
        else:
            r = read_func(s)
        if _assert:
            self.assertTrue(is_equal(r, data))
        return r

    def test_dm_read_struct_types(self) -> None:
        s = io.BytesIO()
        types = [2, 2, 2]
        parse_dm3.dm_write_struct_types(s, types)
        s.seek(0)
        in_types, headerlen = parse_dm3.dm_read_struct_types(s)
        self.assertEqual(in_types, types)

    def test_simpledata(self) -> None:
        self.check_write_then_read_matches(45, parse_dm3.dm_write_types[parse_dm3.get_dmtype_for_name('long')], parse_dm3.dm_read_types[parse_dm3.get_dmtype_for_name('long')])
        self.check_write_then_read_matches(2**30, parse_dm3.dm_write_types[parse_dm3.get_dmtype_for_name('uint')], parse_dm3.dm_read_types[parse_dm3.get_dmtype_for_name('uint')])
        self.check_write_then_read_matches(34.56, parse_dm3.dm_write_types[parse_dm3.get_dmtype_for_name('double')], parse_dm3.dm_read_types[parse_dm3.get_dmtype_for_name('double')])

    def test_read_string(self) -> None:
        data = "MyString"
        ret = self.check_write_then_read_matches(data, parse_dm3.dm_write_types[parse_dm3.get_dmtype_for_name('array')], parse_dm3.dm_read_types[parse_dm3.get_dmtype_for_name('array')], False)
        self.assertEqual(data, dm3_image_utils.fix_strings(ret))

    def test_array_simple(self) -> None:
        dat = parse_dm3.DataChunkWriter(numpy.array([0] * 256, dtype=numpy.int8))
        self.check_write_then_read_matches(dat, parse_dm3.dm_write_types[parse_dm3.get_dmtype_for_name('array')], parse_dm3.dm_read_types[parse_dm3.get_dmtype_for_name('array')])

    def test_array_struct(self) -> None:
        dat = parse_dm3.StructArray(['h', 'h', 'h'])
        dat.raw_data = array.array('b', [0, 0] * 3 * 8)  # two bytes x 3 'h's x 8 elements
        self.check_write_then_read_matches(dat, parse_dm3.dm_write_types[parse_dm3.get_dmtype_for_name('array')], parse_dm3.dm_read_types[parse_dm3.get_dmtype_for_name('array')])

    def test_tagdata(self) -> None:
        for d in [45, 2**30, 34.56, parse_dm3.DataChunkWriter(numpy.array([0] * 256, dtype=numpy.int8))]:
            self.check_write_then_read_matches(d, parse_dm3.dm_write_tag_data, parse_dm3.dm_read_tag_data)

    def test_tagroot_dict(self) -> None:
        mydata = dict[str, typing.Any]()
        self.check_write_then_read_matches(mydata, parse_dm3.dm_write_tag_root, parse_dm3.dm_read_tag_root)
        mydata = {"Bob": 45, "Henry": 67, "Joe": 56}
        self.check_write_then_read_matches(mydata, parse_dm3.dm_write_tag_root, parse_dm3.dm_read_tag_root)

    def test_tagroot_dict_complex(self) -> None:
        mydata = {"Bob": 45, "Henry": 67, "Joe": {
                  "hi": [34, 56, 78, 23], "Nope": 56.7, "d": parse_dm3.DataChunkWriter(numpy.array([0] * 32, dtype=numpy.uint16))}}
        self.check_write_then_read_matches(mydata, parse_dm3.dm_write_tag_root, parse_dm3.dm_read_tag_root)

    def test_tagroot_list(self) -> None:
        # note any strings here get converted to 'H' arrays!
        mydata = list[int]()
        self.check_write_then_read_matches(mydata, parse_dm3.dm_write_tag_root, parse_dm3.dm_read_tag_root)
        mydata = [45,  67,  56]
        self.check_write_then_read_matches(mydata, parse_dm3.dm_write_tag_root, parse_dm3.dm_read_tag_root)

    def test_struct(self) -> None:
        # note any strings here get converted to 'H' arrays!
        mydata = tuple[typing.Any]()
        self.check_write_then_read_matches(mydata, parse_dm3.dm_write_types[parse_dm3.get_dmtype_for_name('struct')], parse_dm3.dm_read_types[parse_dm3.get_dmtype_for_name('struct')])
        mydata = (3, 4, 56.7)
        self.check_write_then_read_matches(mydata, parse_dm3.dm_write_types[parse_dm3.get_dmtype_for_name('struct')], parse_dm3.dm_read_types[parse_dm3.get_dmtype_for_name('struct')])

    def test_image(self) -> None:
        im = numpy.random.randint(low=0, high=65536, size=(32,), dtype=numpy.uint16)
        im_tag = {"Data": parse_dm3.DataChunkWriter(im),
                  "Dimensions": [23, 45]}
        s = io.BytesIO()
        parse_dm3.dm_write_tag_root(s, im_tag)
        s.seek(0)
        ret = parse_dm3.dm_read_tag_root(s)
        self.assertTrue(is_equal(ret["Data"], im_tag["Data"]))
        self.assertEqual(im_tag["Dimensions"], ret["Dimensions"])

    def test_data_write_read_round_trip(self) -> None:
        def db_make_directory_if_needed(directory_path: str) -> None:
            if os.path.exists(directory_path):
                if not os.path.isdir(directory_path):
                    raise OSError("Path is not a directory:", directory_path)
            else:
                os.makedirs(directory_path)

        class numpy_array_type:
            def __init__(self, shape: tuple[int, ...], dtype: numpy.typing.DTypeLike) -> None:
                self.data = numpy.ones(shape, dtype)
            def __enter__(self) -> numpy_array_type:
                return self
            def __exit__(self, exception_type: typing.Optional[typing.Type[BaseException]], value: typing.Optional[BaseException], traceback: typing.Optional[types.TracebackType]) -> typing.Optional[bool]:
                return None

        class h5py_array_type:
            def __init__(self, shape: tuple[int, ...], dtype: numpy.typing.DTypeLike) -> None:
                current_working_directory = os.getcwd()
                self.__workspace_dir = os.path.join(current_working_directory, "__Test")
                db_make_directory_if_needed(self.__workspace_dir)
                self.f = h5py.File(os.path.join(self.__workspace_dir, "file.h5"), "a")
                self.data = self.f.create_dataset("data", data=numpy.ones(shape, dtype))
            def __enter__(self) -> h5py_array_type:
                return self
            def __exit__(self, exception_type: typing.Optional[typing.Type[BaseException]], value: typing.Optional[BaseException], traceback: typing.Optional[types.TracebackType]) -> typing.Optional[bool]:
                self.f.close()
                shutil.rmtree(self.__workspace_dir)
                return None

        array_types = numpy_array_type, h5py_array_type
        dtypes = (numpy.float32, numpy.float64, numpy.complex64, numpy.complex128, numpy.int16, numpy.uint16, numpy.int32, numpy.uint32)
        shape_data_descriptors = (
            ((6,), DataAndMetadata.DataDescriptor(False, 0, 1)),        # spectrum
            ((6, 4), DataAndMetadata.DataDescriptor(False, 1, 1)),      # 1d collection of spectra
            ((6, 8, 10), DataAndMetadata.DataDescriptor(False, 2, 1)),  # 2d collection of spectra
            ((6, 4), DataAndMetadata.DataDescriptor(True, 0, 1)),       # sequence of spectra
            ((6, 4), DataAndMetadata.DataDescriptor(False, 0, 2)),      # image
            ((6, 4, 2), DataAndMetadata.DataDescriptor(False, 1, 2)),   # 1d collection of images
            ((6, 5, 4, 2), DataAndMetadata.DataDescriptor(False, 2, 2)),   # 2d collection of images. not possible?
            ((6, 8, 10), DataAndMetadata.DataDescriptor(True, 0, 2)),   # sequence of images
        )
        for version in (3, 4):
            for array_type in array_types:
                for dtype in dtypes:
                    for shape, data_descriptor_in in shape_data_descriptors:
                        for signal_type in ((None, "eels") if data_descriptor_in.datum_dimension_count ==1 else (None,)):
                            # print(f"--- {array_type} {dtype} {shape} {data_descriptor_in} {signal_type}")
                            s = io.BytesIO()
                            with array_type(shape, dtype) as a:
                                data_in = a.data
                                dimensional_calibrations_in = list()
                                for index, dimension in enumerate(shape):
                                    dimensional_calibrations_in.append(Calibration.Calibration(1.0 + 0.1 * index, 2.0 + 0.2 * index, "µ" + "n" * index))
                                intensity_calibration_in = Calibration.Calibration(4, 5, "six")
                                metadata_in = dict[typing.Any, typing.Any]()
                                if signal_type:
                                    metadata_in.setdefault("hardware_source", dict())["signal_type"] = signal_type
                                xdata_in = DataAndMetadata.new_data_and_metadata(data_in, data_descriptor=data_descriptor_in, dimensional_calibrations=dimensional_calibrations_in, intensity_calibration=intensity_calibration_in, metadata=metadata_in)
                                dm3_image_utils.save_image(xdata_in, s, version)
                                s.seek(0)
                                xdata = dm3_image_utils.load_image(s)
                                self.assertTrue(numpy.array_equal(data_in, xdata.data))
                                self.assertEqual(data_descriptor_in, xdata.data_descriptor)
                                self.assertEqual(tuple(dimensional_calibrations_in), tuple(xdata.dimensional_calibrations))
                                self.assertEqual(intensity_calibration_in, xdata.intensity_calibration)

    def test_rgb_data_write_read_round_trip(self) -> None:
        for version in (3, 4):
            s = io.BytesIO()
            data_in = (numpy.random.randn(6, 4, 3) * 255).astype(numpy.uint8)
            data_descriptor_in = DataAndMetadata.DataDescriptor(False, 0, 2)
            dimensional_calibrations_in = [Calibration.Calibration(1, 2, "nm"), Calibration.Calibration(2, 3, u"µm")]
            intensity_calibration_in = Calibration.Calibration(4, 5, "six")
            metadata_in: dict[str, typing.Any] = {"abc": None, "": "", "one": [], "two": {}, "three": [1, None, 2]}
            xdata_in = DataAndMetadata.new_data_and_metadata(data_in, data_descriptor=data_descriptor_in, dimensional_calibrations=dimensional_calibrations_in, intensity_calibration=intensity_calibration_in, metadata=metadata_in)
            dm3_image_utils.save_image(xdata_in, s, version)
            s.seek(0)
            xdata = dm3_image_utils.load_image(s)
            self.assertTrue(numpy.array_equal(data_in, xdata.data))
            self.assertEqual(data_descriptor_in, xdata.data_descriptor)

    def test_calibrations_write_read_round_trip(self) -> None:
        for version in (3, 4):
            s = io.BytesIO()
            data_in = numpy.ones((6, 4), numpy.float32)
            data_descriptor_in = DataAndMetadata.DataDescriptor(False, 0, 2)
            dimensional_calibrations_in = (Calibration.Calibration(1.1, 2.1, "nm"), Calibration.Calibration(2, 3, u"µm"))
            intensity_calibration_in = Calibration.Calibration(4.4, 5.5, "six")
            metadata_in = dict[str, typing.Any]()
            xdata_in = DataAndMetadata.new_data_and_metadata(data_in, data_descriptor=data_descriptor_in, dimensional_calibrations=dimensional_calibrations_in, intensity_calibration=intensity_calibration_in, metadata=metadata_in)
            dm3_image_utils.save_image(xdata_in, s, version)
            s.seek(0)
            xdata = dm3_image_utils.load_image(s)
            self.assertEqual(tuple(dimensional_calibrations_in), tuple(xdata.dimensional_calibrations))
            self.assertEqual(intensity_calibration_in, xdata.intensity_calibration)

    def test_data_timestamp_write_read_round_trip(self) -> None:
        for version in (3, 4):
            s = io.BytesIO()
            data_in = numpy.ones((6, 4), numpy.float32)
            data_descriptor_in = DataAndMetadata.DataDescriptor(False, 0, 2)
            dimensional_calibrations_in = [Calibration.Calibration(1.1, 2.1, "nm"), Calibration.Calibration(2, 3, u"µm")]
            intensity_calibration_in = Calibration.Calibration(4.4, 5.5, "six")
            metadata_in = dict[str, typing.Any]()
            timestamp_in = datetime.datetime(2013, 11, 18, 14, 5, 4, 0)
            timezone_in = "America/Los_Angeles"
            timezone_offset_in = "-0700"
            xdata_in = DataAndMetadata.new_data_and_metadata(data_in, data_descriptor=data_descriptor_in, dimensional_calibrations=dimensional_calibrations_in, intensity_calibration=intensity_calibration_in, metadata=metadata_in, timestamp=timestamp_in, timezone=timezone_in, timezone_offset=timezone_offset_in)
            dm3_image_utils.save_image(xdata_in, s, version)
            s.seek(0)
            xdata = dm3_image_utils.load_image(s)
            self.assertEqual(timestamp_in, xdata.timestamp)
            self.assertEqual(timezone_in, xdata.timezone)
            self.assertEqual(timezone_offset_in, xdata.timezone_offset)

    def test_metadata_write_read_round_trip(self) -> None:
        for version in (3, 4):
            s = io.BytesIO()
            data_in = numpy.ones((6, 4), numpy.float32)
            data_descriptor_in = DataAndMetadata.DataDescriptor(False, 0, 2)
            dimensional_calibrations_in = [Calibration.Calibration(1, 2, "nm"), Calibration.Calibration(2, 3, u"µm")]
            intensity_calibration_in = Calibration.Calibration(4, 5, "six")
            metadata_in = {
                "abc": 1, "def": "abc",
                "efg": {
                    "one": 1, "two": "TWO",
                    "three": [3, 4, 5], "threef": [3.0, 4.0, 5.0],
                    "four": (32, 32), "fourf": (33.0, 34.0),
                    "six": ((1, 2), (3, 4)), "sixf": ((1.0, 2.0), (3.0, 4.0)),
                    "seven": [[1, 2], [3, 4]], "sevenf": [[1.0, 2.0], [3.0, 4.0]],
                    # the following will not work until there is a schema or other type hinting to distinguish
                    # this from the "six" case.
                    # "eight": (1, 2, 3, 4), "eightf": (1.0, 2.0, 3.0, 4.0),
                }
            }
            xdata_in = DataAndMetadata.new_data_and_metadata(data_in, data_descriptor=data_descriptor_in, dimensional_calibrations=dimensional_calibrations_in, intensity_calibration=intensity_calibration_in, metadata=metadata_in)
            dm3_image_utils.save_image(xdata_in, s, version)
            s.seek(0)
            xdata = dm3_image_utils.load_image(s)
            self.assertEqual(metadata_in, xdata.metadata)

    def test_metadata_difficult_types_write_read_round_trip(self) -> None:
        for version in (3, 4):
            s = io.BytesIO()
            data_in = numpy.ones((6, 4), numpy.float32)
            data_descriptor_in = DataAndMetadata.DataDescriptor(False, 0, 2)
            dimensional_calibrations_in = [Calibration.Calibration(1, 2, "nm"), Calibration.Calibration(2, 3, u"µm")]
            intensity_calibration_in = Calibration.Calibration(4, 5, "six")
            metadata_in: dict[str, typing.Any] = {"abc": None, "": "", "one": [], "two": {}, "three": [1, None, 2]}
            xdata_in = DataAndMetadata.new_data_and_metadata(data_in, data_descriptor=data_descriptor_in, dimensional_calibrations=dimensional_calibrations_in, intensity_calibration=intensity_calibration_in, metadata=metadata_in)
            dm3_image_utils.save_image(xdata_in, s, version)
            s.seek(0)
            xdata = dm3_image_utils.load_image(s)
            metadata_expected = {"one": [], "two": {}, "three": [1, 2]}
            self.assertEqual(metadata_expected, xdata.metadata)

    def test_metadata_export_large_integer(self) -> None:
        for version in (3, 4):
            s = io.BytesIO()
            data_in = numpy.ones((6, 4), numpy.float32)
            data_descriptor_in = DataAndMetadata.DataDescriptor(False, 0, 2)
            dimensional_calibrations_in = [Calibration.Calibration(1, 2, "nm"), Calibration.Calibration(2, 3, u"µm")]
            intensity_calibration_in = Calibration.Calibration(4, 5, "six")
            metadata_in: dict[str, typing.Any] = {"abc": 999999999999}
            xdata_in = DataAndMetadata.new_data_and_metadata(data_in, data_descriptor=data_descriptor_in, dimensional_calibrations=dimensional_calibrations_in, intensity_calibration=intensity_calibration_in, metadata=metadata_in)
            dm3_image_utils.save_image(xdata_in, s, version)
            s.seek(0)
            xdata = dm3_image_utils.load_image(s)
            metadata_expected = {"abc": 999999999999}
            self.assertEqual(metadata_expected, xdata.metadata)

    def test_signal_type_round_trip(self) -> None:
        for version in (3, 4):
            s = io.BytesIO()
            data_in = numpy.ones((12,), numpy.float32)
            data_descriptor_in = DataAndMetadata.DataDescriptor(False, 0, 1)
            dimensional_calibrations_in = [Calibration.Calibration(1, 2, "eV")]
            intensity_calibration_in = Calibration.Calibration(4, 5, "e")
            metadata_in = {"hardware_source": {"signal_type": "EELS"}}
            xdata_in = DataAndMetadata.new_data_and_metadata(data_in, data_descriptor=data_descriptor_in, dimensional_calibrations=dimensional_calibrations_in, intensity_calibration=intensity_calibration_in, metadata=metadata_in)
            dm3_image_utils.save_image(xdata_in, s, version)
            s.seek(0)
            xdata = dm3_image_utils.load_image(s)
            metadata_expected = {'hardware_source': {'signal_type': 'EELS'}, 'Meta Data': {'Format': 'Spectrum', 'Signal': 'EELS'}}
            self.assertEqual(metadata_expected, xdata.metadata)

    def test_reference_images_load_properly(self) -> None:
        shape_data_descriptors = (
            ((3,), DataAndMetadata.DataDescriptor(False, 0, 1)),        # spectrum
            ((3, 2), DataAndMetadata.DataDescriptor(False, 1, 1)),      # 1d collection of spectra
            ((3, 4, 5), DataAndMetadata.DataDescriptor(False, 2, 1)),   # 2d collection of spectra
            ((3, 2), DataAndMetadata.DataDescriptor(True, 0, 1)),       # sequence of spectra
            ((3, 2), DataAndMetadata.DataDescriptor(False, 0, 2)),      # image
            ((4, 3, 2), DataAndMetadata.DataDescriptor(False, 1, 2)),   # 1d collection of images
            ((3, 4, 5), DataAndMetadata.DataDescriptor(True, 0, 2)),    # sequence of images
        )
        for (shape, data_descriptor), version in itertools.product(shape_data_descriptors, (3, 4)):
            dimensional_calibrations = list()
            for index, dimension in enumerate(shape):
                dimensional_calibrations.append(Calibration.Calibration(1.0 + 0.1 * index, 2.0 + 0.2 * index, "µ" + "n" * index))
            intensity_calibration = Calibration.Calibration(4, 5, "six")
            data = numpy.arange(numpy.prod(shape), dtype=numpy.float32).reshape(shape)

            name = f"ref_{'T' if data_descriptor.is_sequence else 'F'}_{data_descriptor.collection_dimension_count}_{data_descriptor.datum_dimension_count}.dm{version}"

            # import pathlib
            # xdata = DataAndMetadata.new_data_and_metadata(data, dimensional_calibrations=dimensional_calibrations, intensity_calibration=intensity_calibration, data_descriptor=data_descriptor)
            # file_path = pathlib.Path(__file__).parent / "resources" / name
            # with file_path.open('wb') as f:
            #     dm3_image_utils.save_image(xdata, f, version)

            try:
                _data = pkgutil.get_data(__name__, f"resources/{name}")
                assert _data is not None
                s = io.BytesIO(_data)
                xdata = dm3_image_utils.load_image(s)
                self.assertAlmostEqual(intensity_calibration.scale, xdata.intensity_calibration.scale, 6)
                self.assertAlmostEqual(intensity_calibration.offset, xdata.intensity_calibration.offset, 6)
                self.assertEqual(intensity_calibration.units, xdata.intensity_calibration.units)
                for c1, c2 in zip(dimensional_calibrations, xdata.dimensional_calibrations):
                    self.assertAlmostEqual(c1.scale, c2.scale, 6)
                    self.assertAlmostEqual(c1.offset, c2.offset, 6)
                    self.assertEqual(c1.units, c2.units)
                self.assertEqual(data_descriptor, xdata.data_descriptor)
                self.assertTrue(numpy.array_equal(data, xdata.data))
                # print(f"{name} {data_descriptor} PASS")
            except Exception as e:
                print(f"{name} {data_descriptor} FAIL")
                raise

    def disabled_test_specific_file(self) -> None:
        file_path = "/path/to/test.dm3"
        with open(file_path, "rb") as f:
            xdata = dm3_image_utils.load_image(f)
        # file_path = "/path/to/test-new.dm3"
        # with open(file_path, "wb") as f:
        #     dm3_image_utils.save_image(xdata, f, 3)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
