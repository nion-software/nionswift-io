# -*- coding: utf-8 -*-
"""
Created on Sun May 19 07:58:10 2013

@author: matt
"""

import array
import io
import logging
import unittest
import sys

import numpy

from nionswift_plugin.DM_IO import parse_dm3
from nionswift_plugin.DM_IO import dm3_image_utils

from nion.data import Calibration
from nion.data import DataAndMetadata


class TestDM3ImportExportClass(unittest.TestCase):

    def check_write_then_read_matches(self, data, func, _assert=True):
        # we confirm that reading a written element returns the same value
        s = io.BytesIO()
        header = func(s, outdata=data)
        s.seek(0)
        if header is not None:
            r, hy = func(s)
        else:
            r = func(s)
        if _assert:
            self.assertEqual(r, data)
        return r

    def test_dm_read_struct_types(self):
        s = io.BytesIO()
        types = [2, 2, 2]
        parse_dm3.dm_read_struct_types(s, outtypes=types)
        s.seek(0)
        in_types, headerlen = parse_dm3.dm_read_struct_types(s)
        self.assertEqual(in_types, types)

    def test_simpledata(self):
        self.check_write_then_read_matches(45, parse_dm3.dm_types[parse_dm3.get_dmtype_for_name('long')])
        self.check_write_then_read_matches(2**30, parse_dm3.dm_types[parse_dm3.get_dmtype_for_name('uint')])
        self.check_write_then_read_matches(34.56, parse_dm3.dm_types[parse_dm3.get_dmtype_for_name('double')])

    def test_read_string(self):
        data = "MyString"
        ret = self.check_write_then_read_matches(data, parse_dm3.dm_types[parse_dm3.get_dmtype_for_name('array')], False)
        self.assertEqual(data, dm3_image_utils.fix_strings(ret))

    def test_array_simple(self):
        dat = array.array('b', [0]*256)
        self.check_write_then_read_matches(dat, parse_dm3.dm_types[parse_dm3.get_dmtype_for_name('array')])

    def test_array_struct(self):
        dat = parse_dm3.structarray(['h', 'h', 'h'])
        dat.raw_data = array.array('b', [0, 0] * 3 * 8)  # two bytes x 3 'h's x 8 elements
        self.check_write_then_read_matches(dat, parse_dm3.dm_types[parse_dm3.get_dmtype_for_name('array')])

    def test_tagdata(self):
        for d in [45, 2**30, 34.56, array.array('b', [0]*256)]:
            self.check_write_then_read_matches(d, parse_dm3.parse_dm_tag_data)

    def test_tagroot_dict(self):
        mydata = {}
        self.check_write_then_read_matches(mydata, parse_dm3.parse_dm_tag_root)
        mydata = {"Bob": 45, "Henry": 67, "Joe": 56}
        self.check_write_then_read_matches(mydata, parse_dm3.parse_dm_tag_root)

    def test_tagroot_dict_complex(self):
        mydata = {"Bob": 45, "Henry": 67, "Joe": {
                  "hi": [34, 56, 78, 23], "Nope": 56.7, "d": array.array('I', [0] * 32)}}
        self.check_write_then_read_matches(mydata, parse_dm3.parse_dm_tag_root)

    def test_tagroot_list(self):
        # note any strings here get converted to 'H' arrays!
        mydata = []
        self.check_write_then_read_matches(mydata, parse_dm3.parse_dm_tag_root)
        mydata = [45,  67,  56]
        self.check_write_then_read_matches(mydata, parse_dm3.parse_dm_tag_root)

    def test_struct(self):
        # note any strings here get converted to 'H' arrays!
        mydata = tuple()
        f = parse_dm3.dm_types[parse_dm3.get_dmtype_for_name('struct')]
        self.check_write_then_read_matches(mydata, f)
        mydata = (3, 4, 56.7)
        self.check_write_then_read_matches(mydata, f)

    def test_image(self):
        im = array.array('h')
        if sys.version < '3':
            im.fromstring(numpy.random.bytes(64))
        else:
            im.frombytes(numpy.random.bytes(64))
        im_tag = {"Data": im,
                  "Dimensions": [23, 45]}
        s = io.BytesIO()
        parse_dm3.parse_dm_tag_root(s, outdata=im_tag)
        s.seek(0)
        ret = parse_dm3.parse_dm_tag_root(s)
        self.assertEqual(im_tag["Data"], ret["Data"])
        self.assertEqual(im_tag["Dimensions"], ret["Dimensions"])
        self.assertTrue((im_tag["Data"] == ret["Data"]))

    def test_data_write_read_round_trip(self):
        dtypes = (numpy.float32, numpy.float64, numpy.complex64, numpy.complex128, numpy.int16, numpy.uint16, numpy.int32, numpy.uint32)
        shape_data_descriptors = (
            ((6,), DataAndMetadata.DataDescriptor(False, 0, 1)),        # spectrum
            ((6, 4), DataAndMetadata.DataDescriptor(False, 1, 1)),      # 1d collection of spectra
            ((6, 8, 10), DataAndMetadata.DataDescriptor(False, 2, 1)),  # 2d collection of spectra
            ((6, 4), DataAndMetadata.DataDescriptor(True, 0, 1)),       # sequence of spectra
            ((6, 4), DataAndMetadata.DataDescriptor(False, 0, 2)),      # image
            ((6, 4, 2), DataAndMetadata.DataDescriptor(False, 1, 2)),   # 1d collection of images
            # ((6, 4, 2), DataAndMetadata.DataDescriptor(False, 2, 2)),   # 2d collection of images. not possible?
            ((6, 8, 10), DataAndMetadata.DataDescriptor(True, 0, 2)),   # sequence of images
        )
        for dtype in dtypes:
            for shape, data_descriptor_in in shape_data_descriptors:
                s = io.BytesIO()
                data_in = numpy.ones(shape, dtype)
                dimensional_calibrations_in = list()
                for index, dimension in enumerate(shape):
                    dimensional_calibrations_in.append(Calibration.Calibration(1.0 + 0.1 * index, 2.0 + 0.2 * index, "µ" + "n" * index))
                intensity_calibration_in = Calibration.Calibration(4, 5, "six")
                metadata_in = dict()
                dm3_image_utils.save_image(data_in, data_descriptor_in, dimensional_calibrations_in, intensity_calibration_in, metadata_in, None, None, None, s)
                s.seek(0)
                data_out, data_descriptor_out, dimensional_calibrations_out, intensity_calibration_out, _, _ = dm3_image_utils.load_image(s)
                self.assertTrue(numpy.array_equal(data_in, data_out))
                self.assertEqual(data_descriptor_in, data_descriptor_out)
                dimensional_calibrations_out = [Calibration.Calibration(*d) for d in dimensional_calibrations_out]
                self.assertEqual(dimensional_calibrations_in, dimensional_calibrations_out)
                self.assertEqual(intensity_calibration_in, Calibration.Calibration(*intensity_calibration_out))

    def test_rgb_data_write_read_round_trip(self):
        s = io.BytesIO()
        data_in = (numpy.random.randn(6, 4, 3) * 255).astype(numpy.uint8)
        data_descriptor_in = DataAndMetadata.DataDescriptor(False, 0, 2)
        dimensional_calibrations_in = [Calibration.Calibration(1, 2, "nm"), Calibration.Calibration(2, 3, u"µm")]
        intensity_calibration_in = Calibration.Calibration(4, 5, "six")
        metadata_in = {"abc": None, "": "", "one": [], "two": {}, "three": [1, None, 2]}
        dm3_image_utils.save_image(data_in, data_descriptor_in, dimensional_calibrations_in, intensity_calibration_in, metadata_in, None, None, None, s)
        s.seek(0)
        data_out, data_descriptor_out, dimensional_calibrations_out, intensity_calibration_out, title_out, metadata_out = dm3_image_utils.load_image(s)
        self.assertTrue(numpy.array_equal(data_in, data_out))
        self.assertEqual(data_descriptor_in, data_descriptor_out)
        # s = "/Users/cmeyer/Desktop/EELS_CL.dm3"
        # data_out, data_descriptor, dimensional_calibrations_out, intensity_calibration_out, title_out, metadata_out = dm3_image_utils.load_image(s)

    def test_calibrations_write_read_round_trip(self):
        s = io.BytesIO()
        data_in = numpy.ones((6, 4), numpy.float32)
        data_descriptor_in = DataAndMetadata.DataDescriptor(False, 0, 2)
        dimensional_calibrations_in = [Calibration.Calibration(1.1, 2.1, "nm"), Calibration.Calibration(2, 3, u"µm")]
        intensity_calibration_in = Calibration.Calibration(4.4, 5.5, "six")
        metadata_in = dict()
        dm3_image_utils.save_image(data_in, data_descriptor_in, dimensional_calibrations_in, intensity_calibration_in, metadata_in, None, None, None, s)
        s.seek(0)
        data_out, data_descriptor_out, dimensional_calibrations_out, intensity_calibration_out, title_out, metadata_out = dm3_image_utils.load_image(s)
        dimensional_calibrations_out = [Calibration.Calibration(*d) for d in dimensional_calibrations_out]
        self.assertEqual(dimensional_calibrations_in, dimensional_calibrations_out)
        intensity_calibration_out = Calibration.Calibration(*intensity_calibration_out)
        self.assertEqual(intensity_calibration_in, intensity_calibration_out)

    def test_metadata_write_read_round_trip(self):
        s = io.BytesIO()
        data_in = numpy.ones((6, 4), numpy.float32)
        data_descriptor_in = DataAndMetadata.DataDescriptor(False, 0, 2)
        dimensional_calibrations_in = [Calibration.Calibration(1, 2, "nm"), Calibration.Calibration(2, 3, u"µm")]
        intensity_calibration_in = Calibration.Calibration(4, 5, "six")
        metadata_in = {"abc": 1, "def": "abc", "efg": { "one": 1, "two": "TWO", "three": [3, 4, 5] }}
        dm3_image_utils.save_image(data_in, data_descriptor_in, dimensional_calibrations_in, intensity_calibration_in, metadata_in, None, None, None, s)
        s.seek(0)
        data_out, data_descriptor_out, dimensional_calibrations_out, intensity_calibration_out, title_out, metadata_out = dm3_image_utils.load_image(s)
        self.assertEqual(metadata_in, metadata_out)

    def test_metadata_difficult_types_write_read_round_trip(self):
        s = io.BytesIO()
        data_in = numpy.ones((6, 4), numpy.float32)
        data_descriptor_in = DataAndMetadata.DataDescriptor(False, 0, 2)
        dimensional_calibrations_in = [Calibration.Calibration(1, 2, "nm"), Calibration.Calibration(2, 3, u"µm")]
        intensity_calibration_in = Calibration.Calibration(4, 5, "six")
        metadata_in = {"abc": None, "": "", "one": [], "two": {}, "three": [1, None, 2]}
        dm3_image_utils.save_image(data_in, data_descriptor_in, dimensional_calibrations_in, intensity_calibration_in, metadata_in, None, None, None, s)
        s.seek(0)
        data_out, data_descriptor_out, dimensional_calibrations_out, intensity_calibration_out, title_out, metadata_out = dm3_image_utils.load_image(s)
        metadata_expected = {"one": [], "two": {}, "three": [1, 2]}
        self.assertEqual(metadata_out, metadata_expected)

    def test_metadata_export_large_integer(self):
        s = io.BytesIO()
        data_in = numpy.ones((6, 4), numpy.float32)
        data_descriptor_in = DataAndMetadata.DataDescriptor(False, 0, 2)
        dimensional_calibrations_in = [Calibration.Calibration(1, 2, "nm"), Calibration.Calibration(2, 3, u"µm")]
        intensity_calibration_in = Calibration.Calibration(4, 5, "six")
        metadata_in = {"abc": 999999999999}
        dm3_image_utils.save_image(data_in, data_descriptor_in, dimensional_calibrations_in, intensity_calibration_in, metadata_in, None, None, None, s)
        s.seek(0)
        data_out, data_descriptor_out, dimensional_calibrations_out, intensity_calibration_out, title_out, metadata_out = dm3_image_utils.load_image(s)
        metadata_expected = {"abc": 999999999999}
        self.assertEqual(metadata_out, metadata_expected)

    def test_signal_type_round_trip(self):
        s = io.BytesIO()
        data_in = numpy.ones((12,), numpy.float32)
        data_descriptor_in = DataAndMetadata.DataDescriptor(False, 0, 1)
        dimensional_calibrations_in = [Calibration.Calibration(1, 2, "eV")]
        intensity_calibration_in = Calibration.Calibration(4, 5, "e")
        metadata_in = {"hardware_source": {"signal_type": "EELS"}}
        dm3_image_utils.save_image(data_in, data_descriptor_in, dimensional_calibrations_in, intensity_calibration_in, metadata_in, None, None, None, s)
        s.seek(0)
        data_out, data_descriptor_out, dimensional_calibrations_out, intensity_calibration_out, title_out, metadata_out = dm3_image_utils.load_image(s)
        metadata_expected = {'hardware_source': {'signal_type': 'EELS'}, 'Meta Data': {'Format': 'Spectrum', 'Signal': 'EELS'}}
        self.assertEqual(metadata_out, metadata_expected)

    def disabled_test_series_data_ordering(self):
        s = "/Users/cmeyer/Downloads/NEW_7FocalSeriesImages_Def_50000nm.dm3"
        data_out, data_descriptor_out, dimensional_calibrations_out, intensity_calibration_out, title_out, metadata_out = dm3_image_utils.load_image(s)
        import pprint
        pprint.pprint(metadata_out)
        print(data_out.shape)

# some functions for processing multiple files.
# useful for testing reading and writing a large number of files.
import os


def process_dm3(path, mode):
    opath = path + ".out.dm3"
    data = odata = None
    if mode == 0 or mode == 1:  # just open source
        # path=opath
        with open(path, 'rb') as f:
            data = parse_dm3.parse_dm_header(f)
    if mode == 1:  # open source, write to out
        with open(opath, 'wb') as f:
            parse_dm3.parse_dm_header(f, outdata=data)
    elif mode == 2:  # open both
        with open(path, 'rb') as f:
            data = parse_dm3.parse_dm_header(f)
        with open(opath, 'rb') as f:
            odata = parse_dm3.parse_dm_header(f)
        # this ensures keys in root only are the same
        assert(sorted(odata) == sorted(data))
    return data, odata


def process_all(mode):
    for f in [x for x in os.listdir(".")
              if x.endswith(".dm3")
              if not x.endswith("out.dm3")]:
        print("reading", f, "...")
        data, odata = process_dm3(f, mode)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
    # process_all(1)
