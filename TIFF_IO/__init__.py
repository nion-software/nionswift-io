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
from nion.swift.model import ImportExportManager


_ = gettext.gettext


class TIFFImportExportHandler(ImportExportManager.ImportExportHandler):

    def __init__(self):
        super(TIFFImportExportHandler, self).__init__(_("TIFF Files"), ["tif", "tiff"])

    def read_data_elements(self, ui, extension, file_path):
        data = tiffile.imread(file_path)
        data_element = dict()
        data_element["data"] = data
        return [data_element]

    def can_write(self, data_item, extension):
        return len(data_item.spatial_shape) == 2

    def write_data(self, data, extension, file_path):
        tiffile.imsave(data, file_path)

ImportExportManager.ImportExportManager().register_io_handler(TIFFImportExportHandler())
