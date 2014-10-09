"""
    Support for TIFF I/O.
"""

# standard libraries
import gettext
import logging
import warnings

# third party libraries
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tifffile

# local libraries
from nion.swift.model import Image
from nion.swift.model import ImportExportManager


_ = gettext.gettext


class TIFFImportExportHandler(ImportExportManager.ImportExportHandler):

    def __init__(self):
        super(TIFFImportExportHandler, self).__init__(_("TIFF Files"), ["tif", "tiff"])

    def read_data_elements(self, ui, extension, file_path):
        data = tifffile.imread(file_path)
        if Image.is_data_rgb(data):
            data = data[:,:,(2, 1, 0)]
        if Image.is_data_rgba(data):
            data = data[:,:,(2, 1, 0, 3)]
        data_element = dict()
        data_element["data"] = data
        return [data_element]

    def can_write(self, data_item, extension):
        return len(data_item.spatial_shape) == 2

    def write(self, ui, data_item, file_path, extension):
        data = data_item.data
        if data is not None:
            if Image.is_data_rgb(data):
                data = data[:,:,(2, 1, 0)]
            if Image.is_data_rgba(data):
                data = data[:,:,(2, 1, 0, 3)]
            tifffile.imsave(file_path, data)


ImportExportManager.ImportExportManager().register_io_handler(TIFFImportExportHandler())
