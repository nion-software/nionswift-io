"""
    Support for DM3 and DM4 I/O.
"""

# standard libraries
import gettext

# third party libraries
# None

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
        data, calibrations, intensity, title, properties = dm3_image_utils.load_image(file_path)
        data_element = dict()
        data_element["data"] = data
        dimensional_calibrations = list()
        for calibration in calibrations:
            origin, scale, units = calibration[0], calibration[1], calibration[2]
            scale = 1.0 if scale == 0.0 else scale  # sanity check
            dimensional_calibrations.append(self.__api.create_calibration(-origin * scale, scale, units))
        # data_element["title"] = title
        metadata = dict()
        metadata["hardware_source"] = properties
        return self.__api.create_data_and_metadata_from_data(data, dimensional_calibrations=dimensional_calibrations, metadata=metadata)

    def can_write_data_and_metadata(self, data_and_metadata, extension):
        return extension == "dm3"

    def write_data_and_metadata(self, data_and_metadata, file_path, extension):
        data = data_and_metadata.data
        dimensional_calibrations = data_and_metadata.dimensional_calibrations
        intensity_calibration = data_and_metadata.intensity_calibration
        metadata = data_and_metadata.metadata
        with open(file_path, 'wb') as f:
            dm3_image_utils.save_image(data, dimensional_calibrations, intensity_calibration, metadata, f)


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
