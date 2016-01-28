"""
    Support for TIFF I/O.
"""

# standard libraries
import gettext
import warnings

# third party libraries
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from . import tifffile
import numpy

# local libraries
# None


_ = gettext.gettext


class TIFFIODelegate(object):

    def __init__(self, api):
        self.__api = api
        self.io_handler_id = "tiff-io-handler"
        self.io_handler_name = _("TIFF Files")
        self.io_handler_extensions = ["tif", "tiff"]

    def read_data_and_metadata(self, extension, file_path):
        data = tifffile.imread(file_path)
        if data.dtype == numpy.uint8 and data.shape[-1] == 3 and len(data.shape) > 1:
            data = data[:,:,(2, 1, 0)]
        if data.dtype == numpy.uint8 and data.shape[-1] == 4 and len(data.shape) > 1:
            data = data[:,:,(2, 1, 0, 3)]
        return self.__api.create_data_and_metadata_from_data(data)

    def can_write_data_and_metadata(self, data_and_metadata, extension):
        return data_and_metadata.is_data_2d

    def write_data_and_metadata(self, data_and_metadata, file_path, extension):
        data = data_and_metadata.data
        if data is not None:
            if data.dtype == numpy.uint8 and data.shape[-1] == 3 and len(data.shape) > 1:
                data = data[:,:,(2, 1, 0)]
            if data.dtype == numpy.uint8 and data.shape[-1] == 4 and len(data.shape) > 1:
                data = data[:,:,(2, 1, 0, 3)]
            tifffile.imsave(file_path, data)


class TIFFIOExtension(object):

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.swift.extensions.tiff_io"

    def __init__(self, api_broker):
        # grab the api object.
        api = api_broker.get_api(version="1", ui_version="1")
        # be sure to keep a reference or it will be closed immediately.
        self.__io_handler_ref = api.create_data_and_metadata_io_handler(TIFFIODelegate(api))

    def close(self):
        # close will be called when the extension is unloaded. in turn, close any references so they get closed. this
        # is not strictly necessary since the references will be deleted naturally when this object is deleted.
        self.__io_handler_ref.close()
        self.__io_handler_ref = None

