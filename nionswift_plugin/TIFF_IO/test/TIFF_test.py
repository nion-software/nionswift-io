# -*- coding: utf-8 -*-
import io
import logging
import typing
import unittest

import numpy

from nion.data import Calibration
from nion.data import DataAndMetadata
from nionswift_plugin import TIFF_IO


class API:
    def create_calibration(self, offset: float=None, scale: float=None, units: str=None) -> Calibration.Calibration:
        return Calibration.Calibration(offset, scale, units)

    def create_data_descriptor(self, is_sequence: bool, collection_dimension_count: int, datum_dimension_count: int) -> DataAndMetadata.DataDescriptor:
        return DataAndMetadata.DataDescriptor(is_sequence, collection_dimension_count, datum_dimension_count)

    def create_data_and_metadata(self, data: numpy.ndarray, intensity_calibration: Calibration.Calibration = None,
                                 dimensional_calibrations: typing.List[Calibration.Calibration] = None, metadata: dict = None,
                                 timestamp: str = None, data_descriptor: DataAndMetadata.DataDescriptor = None) -> DataAndMetadata.DataAndMetadata:
        return DataAndMetadata.new_data_and_metadata(data, intensity_calibration, dimensional_calibrations, metadata, timestamp, data_descriptor)


class TestTIFFIOClass(unittest.TestCase):

    def test_image_j_produces_proper_data_types(self):
        io_delegate = TIFF_IO.TIFFIODelegate_ImageJ(API())
        for t in (numpy.int16, numpy.uint16, numpy.int32, numpy.uint32, numpy.int64, numpy.uint64, numpy.float32, numpy.float64):
            b = io.BytesIO()
            xdata_w = DataAndMetadata.new_data_and_metadata(numpy.zeros((16, 16), dtype=t))
            io_delegate.write_data_and_metadata_stream(xdata_w, b)
            b.seek(0)
            xdata_r = io_delegate.read_data_and_metadata_from_stream(b)
            self.assertEqual(xdata_w.data_shape, xdata_r.data_shape)
            self.assertIn(xdata_r.data.dtype, (numpy.uint16, numpy.float32))

    def test_baseline_produces_proper_data_types(self):
        io_delegate = TIFF_IO.TIFFIODelegate_Baseline(API())
        for t in (numpy.int16, numpy.uint16, numpy.int32, numpy.uint32, numpy.int64, numpy.uint64, numpy.float32, numpy.float64):
            b = io.BytesIO()
            xdata_w = DataAndMetadata.new_data_and_metadata(numpy.zeros((16, 16), dtype=t))
            io_delegate.write_data_and_metadata_stream(xdata_w, b)
            b.seek(0)
            xdata_r = io_delegate.read_data_and_metadata_from_stream(b)
            self.assertEqual(xdata_w.data_shape, xdata_r.data_shape)
            self.assertEqual(xdata_r.data.dtype, numpy.uint16)

    def test_baseline_writes_reads_rgb_roundtrip(self):
        io_delegate = TIFF_IO.TIFFIODelegate_Baseline(API())
        d = numpy.zeros((16, 16, 3), dtype=numpy.uint8)
        d[..., 1] = 1
        d[..., 2] = 2
        b = io.BytesIO()
        xdata_w = DataAndMetadata.new_data_and_metadata(d)
        io_delegate.write_data_and_metadata_stream(xdata_w, b)
        b.seek(0)
        xdata_r = io_delegate.read_data_and_metadata_from_stream(b)
        self.assertEqual(xdata_w.data_shape, xdata_r.data_shape)
        self.assertEqual(xdata_w.data.dtype, xdata_r.data.dtype)
        self.assertTrue(numpy.array_equal(xdata_w.data, xdata_r.data))

    def test_baseline_writes_reads_rgba_roundtrip(self):
        io_delegate = TIFF_IO.TIFFIODelegate_Baseline(API())
        d = numpy.zeros((16, 16, 4), dtype=numpy.uint8)
        d[..., 1] = 1
        d[..., 2] = 2
        d[..., 2] = 4
        b = io.BytesIO()
        xdata_w = DataAndMetadata.new_data_and_metadata(d)
        io_delegate.write_data_and_metadata_stream(xdata_w, b)
        b.seek(0)
        xdata_r = io_delegate.read_data_and_metadata_from_stream(b)
        self.assertEqual(xdata_w.data_shape, xdata_r.data_shape)
        self.assertEqual(xdata_w.data.dtype, xdata_r.data.dtype)
        self.assertTrue(numpy.array_equal(xdata_w.data, xdata_r.data))

    def test_baseline_scales_to_uint16(self):
        io_delegate = TIFF_IO.TIFFIODelegate_Baseline(API())
        d = numpy.zeros((16, 16), dtype=numpy.float32)
        d[0, 0] = 0.5
        d[1, 1] = 1.0
        b = io.BytesIO()
        xdata_w = DataAndMetadata.new_data_and_metadata(d)
        io_delegate.write_data_and_metadata_stream(xdata_w, b)
        b.seek(0)
        xdata_r = io_delegate.read_data_and_metadata_from_stream(b)
        self.assertEqual(xdata_w.data_shape, xdata_r.data_shape)
        self.assertEqual(numpy.uint16, xdata_r.data.dtype)
        self.assertEqual(numpy.amin(xdata_r.data), 0)
        self.assertEqual(numpy.amax(xdata_r.data), 65535)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
