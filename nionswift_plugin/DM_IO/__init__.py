"""
    Support for DM3 and DM4 I/O.
"""

# standard libraries
import gettext

# third party libraries
from nion.data import DataAndMetadata

# local libraries
from . import dm3_image_utils


_ = gettext.gettext


class DM3IODelegate(object):

    def __init__(self, api):
        self.__api = api
        self.io_handler_id = "dm-io-handler"
        self.io_handler_name = _("DigitalMicrograph Files")
        self.io_handler_extensions = ["dm3", "dm4"]

    def read_data_and_metadata(self, extension, file_path):
        return dm3_image_utils.load_image(file_path)

    def can_write_data_and_metadata(self, data_and_metadata, extension):
        return extension == "dm3"

    def write_data_and_metadata(self, data_and_metadata, file_path, extension):
        data = data_and_metadata.data
        data_descriptor = data_and_metadata.data_descriptor
        dimensional_calibrations = list()
        for dimensional_calibration in data_and_metadata.dimensional_calibrations:
            offset, scale, units = dimensional_calibration.offset, dimensional_calibration.scale, dimensional_calibration.units
            dimensional_calibrations.append(self.__api.create_calibration(offset, scale, units))
        intensity_calibration = data_and_metadata.intensity_calibration
        offset, scale, units = intensity_calibration.offset, intensity_calibration.scale, intensity_calibration.units
        intensity_calibration = self.__api.create_calibration(offset, scale, units)
        metadata = data_and_metadata.metadata
        timestamp = data_and_metadata.timestamp
        timezone = data_and_metadata.timezone
        timezone_offset = data_and_metadata.timezone_offset
        with open(file_path, 'wb') as f:
            xdata = DataAndMetadata.new_data_and_metadata(data,
                                                          data_descriptor=data_descriptor,
                                                          dimensional_calibrations=dimensional_calibrations,
                                                          intensity_calibration=intensity_calibration,
                                                          metadata=metadata,
                                                          timestamp=timestamp,
                                                          timezone=timezone,
                                                          timezone_offset=timezone_offset)
            dm3_image_utils.save_image(xdata, f)


def load_image(file_path):
    return dm3_image_utils.load_image(file_path)


class DM3IOExtension(object):

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.swift.extensions.dm3"

    def __init__(self, api_broker):
        # grab the api object.
        api = api_broker.get_api(version="1", ui_version="1")
        # be sure to keep a reference or it will be closed immediately.
        self.__io_handler_ref = api.create_data_and_metadata_io_handler(DM3IODelegate(api))

    def close(self):
        # close will be called when the extension is unloaded. in turn, close any references so they get closed. this
        # is not strictly necessary since the references will be deleted naturally when this object is deleted.
        self.__io_handler_ref.close()
        self.__io_handler_ref = None

# TODO: How should IO delegate handle title when reading using read_data_and_metadata
