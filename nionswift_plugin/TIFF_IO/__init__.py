"""
    Support for TIFF I/O.

    This module enables import/export of imagej compatible tif files.
    It will also read non-imagej tiffs, but the correct handling and shaping of multidimensional data is limited to
    files created with imagej or files that were exported with Nion Swift.
    Files exported with Nion Swift will keep their metadata when exported. This metadata will also be restored on re-import.
    Currently the support is limited to greyscale/rgb(a) data of 1 to 4 dimensions.

"""

# standard libraries
import gettext
import logging
import warnings

# third party libraries
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from . import tifffile
import numpy
import datetime
import json

# local libraries
# None


_ = gettext.gettext


NION_TAG = 'nion.1'


class TIFFIODelegateBase:

    def __init__(self, api):
        self.__api = api
        self.io_handler_extensions = ["tif", "tiff"]

    def read_data_and_metadata(self, extension, file_path):
        return self.read_data_and_metadata_from_stream(file_path)

    def read_data_and_metadata_from_stream(self, stream):
        x_resolution = y_resolution = unit = x_offset = y_offset = None
        data_element_dict = None
        dimensional_calibrations = intensity_calibration = timestamp = data_descriptor = metadata = None
        # Imagej axes names
        images = channels = slices = frames = None

        with tifffile.TiffFile(stream) as tiffimage:
            # TODO: Check whether support for multiple tif pages is necessary (for imagej compatible tifs it isn't)
            # Non-imagej compatible tifs are written (by tifffile.py) into multiple pages if they have more than 2 dimensions
            # There is also a 'shape' attribute that describes the original shape for those files. This information could be used
            # to properly load these files as well. However, right now, only the first page will be loaded and imported.
            tiffpage = tiffimage.pages[0]
            # Try if image is imagej type
            if tiffimage.is_imagej:
                if tiffpage.tags.get('x_resolution') is not None:
                    x_resolution = tiffpage.tags['x_resolution'].value
                    x_resolution = x_resolution[0] / x_resolution[1]
                if tiffpage.tags.get('y_resolution') is not None:
                    y_resolution = tiffpage.tags['y_resolution'].value
                    y_resolution = y_resolution[0] / y_resolution[1]
                if tiffimage.imagej_metadata.get(NION_TAG) is not None:
                    data_element_dict = json.loads(tiffimage.imagej_metadata[NION_TAG])
                unit = tiffimage.imagej_metadata.get('unit')
                images = tiffimage.imagej_metadata.get('images')
                channels = tiffimage.imagej_metadata.get('channels')
                slices = tiffimage.imagej_metadata.get('slices')
                frames = tiffimage.imagej_metadata.get('frames')
                # samples = tiffimage.imagej_metadata.get('samples')

            # Try to get Nion metadata if file is not imagej type
            if data_element_dict is None:
                description = tiffpage.tags.get('image_description')
                description_dict = {}
                if description is not None:
                    try:
                        description_dict = tifffile.image_description_dict(description.value)
                    except ValueError as detail:
                        print(detail)
                try:
                    nion_properties = description_dict.get(NION_TAG)
                    if nion_properties:
                        data_element_dict = json.loads(nion_properties)
                except Exception as detail:
                    print(detail)

            expected_number_dimensions = None
            if data_element_dict is not None:
                expected_number_dimensions = (data_element_dict.get('collection_dimension_count', 0) +
                                              data_element_dict.get('datum_dimension_count', 1) +
                                              int(data_element_dict.get('is_sequence', False)))

            data = tiffpage.asarray()

            # check and adapt for rgb(a) data
            # last data axis depends on whether data is rgb(a)
            last_data_axis = -1
            is_rgb = False
            if tiffpage.photometric in (tifffile.TIFF.PHOTOMETRIC.RGB, tifffile.TIFF.PHOTOMETRIC.PALETTE):
                # print('Image is rgb type, (shape: {})'.format(data.shape))
                is_rgb = True
                if expected_number_dimensions is not None:
                    expected_number_dimensions += 1
                last_data_axis = -2
                if data.shape[-1] == 3:
                    data = data[...,(2, 1, 0)]
                if data.shape[-1] == 4:
                    data = data[...,(2, 1, 0, 3)]
                # only supports 8-bit color images for now
                data = data.astype(numpy.uint8)

            # if number of axes is wrong or it could not be determined and imagej metadata is there
            # use this information to reshape array
            if (expected_number_dimensions is None or
                expected_number_dimensions != len(data.shape)) and (
                numpy.array([images, channels, slices, frames]).astype('bool').any()):
                datum_dimension_count = 1
                collection_dimension_count = 0
                is_sequence = False
                shape = numpy.array(tiffpage._shape)

                if channels is not None:
                    if shape[2] != channels and shape[0]/channels >= 1:
                        shape[2] = channels
                        shape[0] = int(shape[0]/channels)
                    if slices is not None:
                        if shape[1] != slices and shape[0]/slices >= 1:
                            shape[1] = slices
                            shape[0] = int(shape[0]/slices)

                if shape[0] > 1:
                    is_sequence = True
                if shape[1] > 1:
                    collection_dimension_count += 1
                if shape[2] > 1:
                    collection_dimension_count += 1
                if shape[3] > 1:
                    datum_dimension_count += 1
                # data should always be at least 1d, therefore we don't check for x-dimension

                # reshape data if shape estimate was correct
                if numpy.prod(shape) == data.size:
                    # reshape without length 1 extra dimensions
                    data = data.reshape(tuple(shape[shape>1]))
                    data_element_dict = data_element_dict if data_element_dict is not None else {}
                    data_element_dict['collection_dimension_count'] = collection_dimension_count
                    data_element_dict['datum_dimension_count'] = datum_dimension_count
                    data_element_dict['is_sequence'] = is_sequence
                    expected_number_dimensions = (collection_dimension_count + datum_dimension_count +
                                                  int(is_sequence) + int(is_rgb))
                else:
                    print('Could not reshape data with shape {} to estimated shape {}'.format(data.shape,
                                                                                                  tuple(shape)))

            # if number of dimensions calculated from metadata matches actual number of dimensions, we assume
            # that the data is still valid for being interpreted by shape descriptors
            if expected_number_dimensions is not None and len(data.shape) == expected_number_dimensions:
                # change axis order if necessary
                if data_element_dict.get('collection_dimension_count', 0) > 0:
                    # for 2d data in a collection we move also the second data axis
                    if data_element_dict.get('datum_dimension_count', 1) == 2:
                        data = numpy.moveaxis(data, 0, last_data_axis)
                    # for a collection we need to move the data axis to the last position
                    data = numpy.moveaxis(data, 0, last_data_axis)
            # delete shape estimators from metadata dict to avoid errors during import
            elif expected_number_dimensions is not None and data_element_dict is not None:
                print('Removed shape descriptors to avoid errors during import. ' + str(data_element_dict))
                data_element_dict.pop('collection_dimension_count', None)
                data_element_dict.pop('datum_dimension_count', None)
                data_element_dict.pop('is_sequence', None)

        # rgb axis does not have a calibration, therefore create a shape tuple that is independent from whether
        # data is rgb for adjusting calibrations
        data_shape = data.shape
        if is_rgb:
            data_shape = data_shape[:-1]

        # create data info objects
        if data_element_dict is not None:
            dimensional_calibrations, intensity_calibration, timestamp, data_descriptor, metadata = (
                                                       self.__create_data_info_objects_from_data_element_dict(data_element_dict))

        # remove calibrations if their number is wrong
        if dimensional_calibrations is not None and len(dimensional_calibrations) != len(data_shape):
            dimensional_calibrations = None
        # If no dimensional calibrations were found in the metadata, try to use imagej calibrations
        if dimensional_calibrations is None:
            # if metadata is there try to assign calibrations to the correct axis
            if data_element_dict is not None:
                # If data is a collection, assume x- and y-calibration for collection axis
                if data_element_dict.get('collection_dimension_count', 0) > 0:
                    # for 1d-collection assume we want x-resolution
                    if data_element_dict['collection_dimension_count'] == 1:
                        dimensional_calibrations = [self.__api.create_calibration(offset=x_offset,
                                                        scale=(1 / x_resolution) if x_resolution else None,
                                                        units=unit)]

                    elif data_element_dict['collection_dimension_count'] == 2:
                        dimensional_calibrations = [self.__api.create_calibration(offset=y_offset,
                                                        scale=(1 / y_resolution) if y_resolution else None,
                                                        units=unit),
                                                    self.__api.create_calibration(offset=x_offset,
                                                        scale=(1 / x_resolution) if x_resolution else None,
                                                        units=unit)]
                    # Add "empty" calibrations for remaining axis
                    number_calibrations = len(dimensional_calibrations)
                    for i in range(len(data_shape) - number_calibrations):
                        dimensional_calibrations.append(self.__api.create_calibration())
                else:
                    # Assume that x- and y-calibration is for data
                    if data_element_dict.get('datum_dimension_count', 1) == 1:
                        dimensional_calibrations = [self.__api.create_calibration(offset=x_offset,
                                                        scale=(1 / x_resolution) if x_resolution else None,
                                                        units=unit)]
                    elif data_element_dict.get('datum_dimension_count', 1) == 2:
                        dimensional_calibrations = [self.__api.create_calibration(offset=y_offset,
                                                        scale=(1 / y_resolution) if y_resolution else None,
                                                        units=unit),
                                                    self.__api.create_calibration(offset=x_offset,
                                                        scale=(1 / x_resolution) if x_resolution else None,
                                                        units=unit)]
                    # If calibrations were created, make sure their number is correct
                    if dimensional_calibrations is not None:
                        number_calibrations = len(dimensional_calibrations)
                        for i in range(len(data_shape) - number_calibrations):
                            dimensional_calibrations.insert(0, self.__api.create_calibration())
            # If no metadata is there use calibrations for guessed axes
            else:
                # Data will be at least 1D so we can append x-calibration in any case
                dimensional_calibrations = [self.__api.create_calibration(offset=x_offset,
                                            scale=(1 / x_resolution) if x_resolution else None,
                                            units=unit)]
                # If data has more dimensions also append y-resolution
                dimensional_calibrations.insert(0, self.__api.create_calibration(offset=y_offset,
                                                   scale=(1 / y_resolution) if y_resolution else None,
                                                   units=unit))
                # Add "empty" calibrations for remaining axes
                number_calibrations = len(dimensional_calibrations)
                for i in range(len(data_shape) - number_calibrations):
                    dimensional_calibrations.insert(0, self.__api.create_calibration())

        # If data is 3d and no metadata was found make is_sequence True because a stack of images will be the
        # most likely case of imported 3d data
        if data_descriptor is None and len(data_shape) == 3:
            data_descriptor = self.__api.create_data_descriptor(True, 0, 2)
        # print(dimensional_calibrations, intensity_calibration, timestamp, data_descriptor, metadata)
        # print('Data shape: ' + str(data.shape))
        data_and_metadata = self.__api.create_data_and_metadata(data, intensity_calibration, dimensional_calibrations,
                                                                metadata, timestamp, data_descriptor)
        return data_and_metadata

    def write_data_item(self, data_item, file_path, extension) -> None:
        self.write_data_item_stream(data_item, file_path)

    def __create_data_info_objects_from_data_element_dict(self, metadata_dict):
        dimensional_calibrations = intensity_calibration = timestamp = data_descriptor = metadata = None
        if metadata_dict.get('spatial_calibrations') is not None:
            dimensional_calibrations = []
            for calibration in metadata_dict['spatial_calibrations']:
                dimensional_calibrations.append(self.__api.create_calibration(offset=calibration.get('offset'), scale=calibration.get('scale'), units=calibration.get('units')))
        if metadata_dict.get('intensity_calibration') is not None:
            calibration = metadata_dict['intensity_calibration']
            intensity_calibration = self.__api.create_calibration(offset=calibration.get('offset'),  scale=calibration.get('scale'), units=calibration.get('units'))
        if not None in [metadata_dict.get('collection_dimension_count'), metadata_dict.get('datum_dimension_count')]:
            data_descriptor = self.__api.create_data_descriptor(metadata_dict.get('is_sequence', False), metadata_dict.get('collection_dimension_count'), metadata_dict.get('datum_dimension_count'))
        if metadata_dict.get('properties') is not None:
            metadata = {'hardware_source': metadata_dict['properties']}
        if metadata_dict.get('timestamp') is not None:
            timestamp = datetime.datetime.fromtimestamp(metadata_dict['timestamp'])
        return dimensional_calibrations, intensity_calibration, timestamp, data_descriptor, metadata


class TIFFIODelegate_Baseline(TIFFIODelegateBase):

    def __init__(self, api):
        super().__init__(api)
        self.io_handler_id = "tiff-io-handler-baseline"
        self.io_handler_name = _("TIFF Files (Baseline)")

    def can_write_data_and_metadata(self, data_and_metadata, extension):
        if data_and_metadata.is_sequence:
            return False
        if data_and_metadata.collection_dimension_count == 2 and data_and_metadata.datum_dimension_count == 0:
            return True
        if data_and_metadata.collection_dimension_count == 0 and data_and_metadata.datum_dimension_count == 2:
            return True
        return False

    def write_data_item_stream(self, data_item, stream) -> None:
        self.write_data_and_metadata_stream(data_item.display_xdata, stream)

    def write_data_and_metadata_stream(self, data_and_metadata, stream) -> None:
        data = data_and_metadata.data

        if data is not None:
            # check and adapt for rgb(a) data ordering
            if data_and_metadata.is_data_rgb:
                data = data[...,(2, 1, 0)]
            elif data_and_metadata.is_data_rgba:
                data = data[...,(2, 1, 0, 3)]
            else:
                data_min = numpy.amin(data)
                data_range = numpy.ptp(data)
                if data_range != 0.0:
                    data_01 = (data - data_min) / data_range
                else:
                    data_01 = numpy.zeros(data.shape, numpy.uint16)
                data = (data_01 * 65535).astype(numpy.uint16)

            tifffile.imsave(stream, data, software='Nion Swift')


class TIFFIODelegate_ImageJ(TIFFIODelegateBase):

    def __init__(self, api):
        super().__init__(api)
        self.io_handler_id = "tiff-io-handler-imagej"
        self.io_handler_name = _("TIFF Files (ImageJ)")

    def can_write_data_and_metadata(self, data_and_metadata, extension) -> bool:
        # return data_and_metadata.is_data_2d or data_and_metadata.is_data_1d or data_and_metadata.is_data_3d
        return len(data_and_metadata.data_shape) < 5

    def write_data_item_stream(self, data_item, stream) -> None:
        self.write_data_and_metadata_stream(data_item.xdata, stream)

    def write_data_and_metadata_stream(self, data_and_metadata, stream) -> None:
        data = data_and_metadata.data
        tifffile_metadata = {}

        calibrations = data_and_metadata.dimensional_calibrations

        tifffile_metadata['unit'] = ''
        metadata_dict = self.__extract_data_element_dict_from_data_and_metadata(data_and_metadata)
        tifffile_metadata[NION_TAG] = json.dumps(metadata_dict)

        if data is not None:
            data_shape = data.shape
            # TODO: support data that is a sequence AND a collection

            # create shape that is used for tif so that array is interpreted correctly by imagej
            tifffile_shape = numpy.ones(6, dtype=numpy.int)

            # last data axis depends on whether data is rgb(a) or not
            last_data_axis = -1

            # check and adapt for rgb(a) data
            if data_and_metadata.is_data_rgb:
                data = data[...,(2, 1, 0)]
                data_shape = data_shape[:-1]
                tifffile_shape[-1] = 3
                last_data_axis = -2
            if data_and_metadata.is_data_rgba:
                data_shape = data_shape[:-1]
                data = data[...,(2, 1, 0, 3)]
                tifffile_shape[-1] = 4
                last_data_axis = -2

            if data_and_metadata.collection_dimension_count > 0:
                # if data is a collection, put collection axis in x-and y of tif
                tifffile_shape[4] = data_shape[0]
                # use collection x-calibration as x-calibration in tif
                resolution = (1 / calibrations[0].scale, ) if calibrations[0].scale != 0 else (1, )
                # use x-unit in tif (unfortunately there is no way to save separate units for x- and y)
                unit = calibrations[0].units
                # if data is a 2d-collection, also fill y-axis of tif
                if data_and_metadata.collection_dimension_count == 2:
                    tifffile_shape[3] = data_shape[1]
                    # add collection y-calibration as y-calibration in tif
                    resolution += (1 / calibrations[1].scale, ) if calibrations[1].scale != 0 else (1, )
                # for data x-axis use tif "channel" axis
                tifffile_shape[2] = data_shape[-1]
                # if data is 2d, put y-axis in tif z-axis (there is no better option unfortunately)
                if data_and_metadata.datum_dimension_count == 2:
                    tifffile_shape[1] = data_shape[-2]
            else:
                if data_and_metadata.is_sequence:
                    # Put sequence axis in "time" axis of tif
                    tifffile_shape[0] = data_shape[0]
                # data x-axis goes in tif x-axis
                tifffile_shape[4] = data_shape[-1]
                # use data x-calibration as x-calibration in tif
                resolution = (1 / calibrations[-1].scale, ) if calibrations[-1].scale != 0 else (1, )
                # use x-unit in tif (unfortunately there is no way to save separate units for x- and y)
                unit = calibrations[-1].units
                # if data is 2d, also put y-axis there
                if data_and_metadata.datum_dimension_count == 2:
                    tifffile_shape[3] = data_shape[-2]
                    # use data y-calibration as y-calibration in tif
                    resolution += (1 / calibrations[-2].scale, ) if calibrations[-2].scale != 0 else (1, )

            # change axis order if necessary
            if data_and_metadata.collection_dimension_count > 0:
                # for a collection we need to move the data axis in front of collection axis
                data = numpy.moveaxis(data, last_data_axis, 0)
                # for 2d data also move the second data axis in front
                if data_and_metadata.datum_dimension_count == 2:
                    data = numpy.moveaxis(data, last_data_axis, 0)

            # make sure "resolution" is always a 2-tuple
            if resolution is not None and len(resolution) < 2:
                resolution += (1, )

            # patch "resolution" such that it does not lead to an OverflowError when saving with tifffile.py. The
            # resolution in tif is saved as ratio of two unsigned 32 bit integers. Tifffile.py creates the integer
            # ratio with a maximum denominator of 1e6, which means that for resolutions > 2**32-1/1e6 = 4294.967295
            # there is a risk of getting an OverflowError when trying to save the integer ratio with 32 bit precision.
            # We also have to make the numbers in "resolution" positive.
            if (numpy.array(resolution) < 0).any():
                resolution = tuple(numpy.abs(resolution))

            if (numpy.array(resolution) > (2**32-1)/1e6).any():
                patched_resolution = numpy.array(resolution)
                possible_numbers = (2**32-1)/(1e6-numpy.arange(1e6))
                if resolution[0] > (2**32-1)/1e6:
                    patched_resolution[0] = possible_numbers[numpy.argmin(numpy.abs(possible_numbers-resolution[0]))]
                if resolution[1] > (2**32-1)/1e6:
                    patched_resolution[1] = possible_numbers[numpy.argmin(numpy.abs(possible_numbers-resolution[1]))]
                resolution = tuple(patched_resolution)

            # add unit to tif tags
            if unit is not None:
                tifffile_metadata['unit'] = unit

            data = data.reshape(tuple(tifffile_shape))

            # Change dtype if necessary to make tif compatible with imagej
            if not data.dtype in [numpy.float32, numpy.uint8, numpy.uint16]:
                data = data.astype(numpy.float32)
            try:
                tifffile.imsave(stream, data, resolution=resolution, imagej=True, metadata=tifffile_metadata, software='Nion Swift')
            except Exception as detail:
                tifffile.imsave(stream, data, resolution=resolution, metadata=tifffile_metadata)
                logging.warn('Could not save metadata in tiff. Reason: ' + str(detail))

    def __extract_data_element_dict_from_data_and_metadata(self, data_and_metadata):
        metadata_dict = {}
        dimensional_calibrations = data_and_metadata.dimensional_calibrations
        if dimensional_calibrations is not None:
            calibrations_element = []
            for calibration in dimensional_calibrations:
                calibrations_element.append({'offset': calibration.offset, 'scale': calibration.scale,
                                             'units': calibration.units})
            metadata_dict['spatial_calibrations'] = calibrations_element
        intensity_calibration = data_and_metadata.intensity_calibration
        if intensity_calibration is not None:
            metadata_dict['intensity_calibration'] = {'offset': intensity_calibration.offset,
                                                      'scale': intensity_calibration.scale,
                                                      'units': intensity_calibration.units}
        if data_and_metadata.is_sequence:
            metadata_dict['is_sequence'] = data_and_metadata.is_sequence
        metadata_dict['collection_dimension_count'] = data_and_metadata.collection_dimension_count
        metadata_dict['datum_dimension_count'] = data_and_metadata.datum_dimension_count
        metadata_dict['properties'] = dict(data_and_metadata.metadata.get('hardware_source', {}))
        metadata_dict['timestamp'] = data_and_metadata.timestamp.timestamp()
        return metadata_dict



class TIFFIOExtension(object):

    # required for Nion Swift to recognize this as an extension class.
    extension_id = "nion.swift.extensions.tiff_io"

    def __init__(self, api_broker):
        # grab the api object.
        api = api_broker.get_api(version="~1.0")
        # be sure to keep a reference or it will be closed immediately.
        self.__io_handler1_ref = api.create_data_and_metadata_io_handler(TIFFIODelegate_Baseline(api))
        self.__io_handler2_ref = api.create_data_and_metadata_io_handler(TIFFIODelegate_ImageJ(api))

    def close(self):
        # close will be called when the extension is unloaded. in turn, close any references so they get closed. this
        # is not strictly necessary since the references will be deleted naturally when this object is deleted.
        self.__io_handler1_ref.close()
        self.__io_handler1_ref = None
        self.__io_handler2_ref.close()
        self.__io_handler2_ref = None

