# ParseDM3File reads in a DM3 file and translates it into a dictionary
# this module treats that dictionary as an image-file and extracts the
# appropriate image data as numpy arrays.
# It also tries to create files from numpy arrays that DM can read.
#
# Some notes:
# Only complex64 and complex128 types are converted to structarrays,
# ie they're arrays of structs. Everything else, (including RGB) are
# standard arrays.
# There is a seperate DatatType and PixelDepth stored for images different
# from the tag file datatype. I think these are used more than the tag
# datratypes in describing the data.
# from .parse_dm3 import *

import copy
import numpy

from nion.data import DataAndMetadata

from . import parse_dm3

def u(x=None):
    return str(x if x is not None else str())

unicode_type = str

long_type = int

def str_to_utf16_bytes(s):
    return s.encode('utf-16')

structarray_to_np_map = {
    ('d', 'd'): numpy.complex128,
    ('f', 'f'): numpy.complex64}

np_to_structarray_map = {v: k for k, v in iter(structarray_to_np_map.items())}

# we want to amp any image type to a single np array type
# but a sinlge np array type could map to more than one dm type.
# For the moment, we won't be strict about, eg, discriminating
# int8 from bool, or even unit32 from RGB. In the future we could
# convert np bool type eg to DM bool and treat y,x,3 int8 images
# as RGB.

# note uint8 here returns the same data type as int8 0 could be that the
# only way they're differentiated is via this type, not the raw type
# in the tag file? And 8 is missing!
dm_image_dtypes = {
    1: ("int16", numpy.int16),
    2: ("float32", numpy.float32),
    3: ("Complex64", numpy.complex64),
    6: ("uint8", numpy.int8),
    7: ("int32", numpy.int32),
    9: ("int8", numpy.int8),
    10: ("uint16", numpy.uint16),
    11: ("uint32", numpy.uint32),
    12: ("float64", numpy.float64),
    13: ("Complex128", numpy.complex128),
    14: ("Bool", numpy.int8),
    23: ("RGB", numpy.int32)
}


def imagedatadict_to_ndarray(imdict):
    """
    Converts the ImageData dictionary, imdict, to an nd image.
    """
    arr = imdict['Data']
    im = None
    if isinstance(arr, parse_dm3.array.array):
        im = numpy.asarray(arr, dtype=arr.typecode)
    elif isinstance(arr, parse_dm3.structarray):
        t = tuple(arr.typecodes)
        im = numpy.frombuffer(
            arr.raw_data,
            dtype=structarray_to_np_map[t])
    # print "Image has dmimagetype", imdict["DataType"], "numpy type is", im.dtype
    assert dm_image_dtypes[imdict["DataType"]][1] == im.dtype
    assert imdict['PixelDepth'] == im.dtype.itemsize
    im = im.reshape(imdict['Dimensions'][::-1])
    if imdict["DataType"] == 23:  # RGB
        im = im.view(numpy.uint8).reshape(im.shape + (-1, ))[..., :-1]  # strip A
        # NOTE: RGB -> BGR would be [:, :, ::-1]
    return im


def platform_independent_char(dtype):
    # windows and linux/macos treat dtype.char differently.
    # on linux/macos where 'l' has size 8, ints of size 4 are reported as 'i'
    # on windows where 'l' has size 4, ints of size 4 are reported as 'l'
    # this function fixes that issue.
    if numpy.dtype('int').itemsize == numpy.dtype('int32').itemsize and dtype.char == 'l': return 'i'
    if numpy.dtype('uint').itemsize == numpy.dtype('uint32').itemsize and dtype.char == 'L': return 'I'
    return dtype.char


def ndarray_to_imagedatadict(nparr):
    """
    Convert the numpy array nparr into a suitable ImageList entry dictionary.
    Returns a dictionary with the appropriate Data, DataType, PixelDepth
    to be inserted into a dm3 tag dictionary and written to a file.
    """
    ret = {}
    dm_type = None
    for k, v in iter(dm_image_dtypes.items()):
        if v[1] == nparr.dtype.type:
            dm_type = k
            break
    if dm_type is None and nparr.dtype == numpy.uint8 and nparr.shape[-1] in (3, 4):
        ret["DataType"] = 23
        ret["PixelDepth"] = 4
        if nparr.shape[2] == 4:
            rgb_view = nparr.view(numpy.int32).reshape(nparr.shape[:-1])  # squash the color into uint32
        else:
            assert nparr.shape[2] == 3
            rgba_image = numpy.empty(nparr.shape[:-1] + (4,), numpy.uint8)
            rgba_image[:,:,0:3] = nparr
            rgba_image[:,:,3] = 255
            rgb_view = rgba_image.view(numpy.int32).reshape(rgba_image.shape[:-1])  # squash the color into uint32
        ret["Dimensions"] = list(rgb_view.shape[::-1])
        ret["Data"] = parse_dm3.array.array(platform_independent_char(rgb_view.dtype), rgb_view.flatten())
    else:
        ret["DataType"] = dm_type
        ret["PixelDepth"] = nparr.dtype.itemsize
        ret["Dimensions"] = list(nparr.shape[::-1])
        if nparr.dtype.type in np_to_structarray_map:
            types = np_to_structarray_map[nparr.dtype.type]
            ret["Data"] = parse_dm3.structarray(types)
            ret["Data"].raw_data = bytes(nparr.data)
        else:
            ret["Data"] = parse_dm3.array.array(platform_independent_char(nparr.dtype), nparr.flatten())
    return ret


def display_keys(tag, indent=None):
    indent = indent if indent is not None else str()
    if isinstance(tag, list) or isinstance(tag, tuple):
        for i, v in enumerate(tag):
            print("{0} {1}:".format(indent, i))
            display_keys(v, indent + "..")
    elif isinstance(tag, dict):
        for k, v in iter(tag.items()):
            print("{0} key: {1}".format(indent, k))
            display_keys(v, indent + "..")
    elif isinstance(tag, bool):
        print("{0} bool: {1}".format(indent, tag))
    elif isinstance(tag, int):
        print("{0} int: {1}".format(indent, tag))
    elif isinstance(tag, float):
        print("{0} float: {1}".format(indent, tag))
    elif isinstance(tag, str):
        print("{0} string: {1}".format(indent, tag))
    else:
        print("{0} {1}: DATA".format(indent, type(tag)))


def fix_strings(d):
    if isinstance(d, dict):
        r = dict()
        for k, v in d.items():
            if k != "Data":
                r[k] = fix_strings(v)
            else:
                r[k] = v
        return r
    elif isinstance(d, list):
        l = list()
        for v in d:
            l.append(fix_strings(v))
        return l
    elif isinstance(d, parse_dm3.array.array):
        if d.typecode == 'H':
            return d.tobytes().decode("utf-16")
        else:
            return d.tolist()
    else:
        return d

def load_image(file):
    """
    Loads the image from the file-like object or string file.
    If file is a string, the file is opened and then read.
    Returns a numpy ndarray of our best guess for the most important image
    in the file.
    """
    if isinstance(file, str) or isinstance(file, unicode_type):
        with open(file, "rb") as f:
            return load_image(f)
    dmtag = parse_dm3.parse_dm_header(file)
    dmtag = fix_strings(dmtag)
    #display_keys(dmtag)
    img_index = -1
    image_tags = dmtag['ImageList'][img_index]
    data = imagedatadict_to_ndarray(image_tags['ImageData'])
    calibrations = []
    calibration_tags = image_tags['ImageData'].get('Calibrations', dict())
    for dimension in calibration_tags.get('Dimension', list()):
        origin, scale, units = dimension.get('Origin', 0.0), dimension.get('Scale', 1.0), dimension.get('Units', str())
        calibrations.append((-origin * scale, scale, units))
    calibrations = tuple(reversed(calibrations))
    if len(data.shape) == 3 and data.dtype != numpy.uint8:
        data = numpy.moveaxis(data, 0, 2)
        data_descriptor = DataAndMetadata.DataDescriptor(False, 1, 2)
        calibrations = tuple(calibrations[1:]) + (calibrations[0],)
    elif data.dtype == numpy.uint8:
        data_descriptor = DataAndMetadata.DataDescriptor(False, 0, len(data.shape[:-1]))
    else:
        data_descriptor = DataAndMetadata.DataDescriptor(False, 0, len(data.shape))
    brightness = calibration_tags.get('Brightness', dict())
    origin, scale, units = brightness.get('Origin', 0.0), brightness.get('Scale', 1.0), brightness.get('Units', str())
    intensity = -origin * scale, scale, units
    title = image_tags.get('Name')
    properties = dict()
    if 'ImageTags' in image_tags:
        properties.update(image_tags['ImageTags'])
        voltage = image_tags['ImageTags'].get('ImageScanned', dict()).get('EHT', dict())
        if voltage:
            properties.setdefault("hardware_source", dict())["autostem"] = { "high_tension_v": float(voltage) }
        dm_metadata_signal = image_tags['ImageTags'].get('Meta Data', dict()).get('Signal')
        if dm_metadata_signal and dm_metadata_signal.lower() == "eels":
            properties.setdefault("hardware_source", dict())["signal_type"] = dm_metadata_signal
        if image_tags['ImageTags'].get('Meta Data', dict()).get('Format') == "Spectrum":
            data_descriptor.collection_dimension_count += data_descriptor.datum_dimension_count - 1
            data_descriptor.datum_dimension_count = 1
        if image_tags['ImageTags'].get('Meta Data', dict()).get('IsSequence', False) and data_descriptor.collection_dimension_count > 0:
            data_descriptor.is_sequence = True
            data_descriptor.collection_dimension_count -= 1
    return data, data_descriptor, tuple(calibrations), intensity, title, properties


def save_image(data, data_descriptor, dimensional_calibrations, intensity_calibration, metadata, modified, timezone, timezone_offset, file):
    """
    Saves the nparray data to the file-like object (or string) file.
    """
    # we need to create a basic DM tree suitable for an image
    # we'll try the minimum: just an data list
    # doesn't work. Do we need a ImageSourceList too?
    # and a DocumentObjectList?
    if len(data.shape) == 3 and data.dtype != numpy.uint8:
        data = numpy.moveaxis(data, 2, 0)
        dimensional_calibrations = (dimensional_calibrations[2],) + tuple(dimensional_calibrations[0:2])
    data_dict = ndarray_to_imagedatadict(data)
    ret = {}
    ret["ImageList"] = [{"ImageData": data_dict}]
    if dimensional_calibrations and len(dimensional_calibrations) == len(data.shape):
        dimension_list = data_dict.setdefault("Calibrations", dict()).setdefault("Dimension", list())
        for dimensional_calibration in reversed(dimensional_calibrations):
            dimension = dict()
            if dimensional_calibration.scale != 0.0:
                origin = -dimensional_calibration.offset / dimensional_calibration.scale
            else:
                origin = 0.0
            dimension['Origin'] = origin
            dimension['Scale'] = dimensional_calibration.scale
            dimension['Units'] = u(dimensional_calibration.units)
            dimension_list.append(dimension)
    if intensity_calibration:
        if intensity_calibration.scale != 0.0:
            origin = -intensity_calibration.offset / intensity_calibration.scale
        else:
            origin = 0.0
        brightness = data_dict.setdefault("Calibrations", dict()).setdefault("Brightness", dict())
        brightness['Origin'] = origin
        brightness['Scale'] = intensity_calibration.scale
        brightness['Units'] = str(intensity_calibration.units)
    if modified:
        timezone_str = None
        if timezone_str is None and timezone:
            try:
                import pytz
                tz = pytz.timezone(timezone)
                timezone_str = tz.tzname(modified)
            except:
                pass
        if timezone_str is None and timezone_offset:
            timezone_str = timezone_offset
        timezone_str = " " + timezone_str if timezone_str is not None else ""
        date_str = modified.strftime("%x")
        time_str = modified.strftime("%X") + timezone_str
        ret["DataBar"] = {"Acquisition Date": date_str, "Acquisition Time": time_str}
    # I think ImageSource list creates a mapping between ImageSourceIds and Images
    ret["ImageSourceList"] = [{"ClassName": "ImageSource:Simple", "Id": [0], "ImageRef": 0}]
    # I think this lists the sources for the DocumentObjectlist. The source number is not
    # the indxe in the imagelist but is either the index in the ImageSourceList or the Id
    # from that list. We also need to set the annotation type to identify it as an data
    ret["DocumentObjectList"] = [{"ImageSource": 0, "AnnotationType": 20}]
    # finally some display options
    ret["Image Behavior"] = {"ViewDisplayID": 8}
    dm_metadata = copy.deepcopy(metadata)
    if metadata.get("hardware_source", dict()).get("signal_type", "").lower() == "eels":
        if len(data.shape) == 1 or (len(data.shape) == 2 and data.shape[0] == 1):
            dm_metadata.setdefault("Meta Data", dict())["Format"] = "Spectrum"
            dm_metadata.setdefault("Meta Data", dict())["Signal"] = "EELS"
    elif data_descriptor.datum_dimension_count == 1:
        dm_metadata.setdefault("Meta Data", dict())["Format"] = "Spectrum"
    if data_descriptor.is_sequence:
        dm_metadata.setdefault("Meta Data", dict())["IsSequence"] = True
    ret["ImageList"][0]["ImageTags"] = dm_metadata
    ret["InImageMode"] = 1
    parse_dm3.parse_dm_header(file, ret)


# logging.debug(image_tags['ImageData']['Calibrations'])
# {u'DisplayCalibratedUnits': True, u'Dimension': [{u'Origin': -0.0, u'Units': u'nm', u'Scale': 0.01171875}, {u'Origin': -0.0, u'Units': u'nm', u'Scale': 0.01171875}, {u'Origin': 0.0, u'Units': u'', u'Scale': 0.01149425096809864}], u'Brightness': {u'Origin': 0.0, u'Units': u'', u'Scale': 1.0}}
