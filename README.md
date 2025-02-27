# Nion Swift I/O

## The Nion Swift I/O library (used in Nion Swift)

Nion Swift I/O is the I/O library for Nion Swift, providing additional
commonly used file formats.

## Quick Start

To write a package to support a custom file format, you can start by
generating a new Nion Swift compatible package using the
`cookiecutter` tool.

See [detailed instructions](https://github.com/nion-software/cookiecutter-nionswift-plug-in) here.

```
cookiecutter gh:nion-software/cookiecutter-nionswift-plug-in
```

The cookiecutter tool will ask a number of questions. We recommend following the convention of attaching "_IO" to the package name. In the case below, the directory will be created as `MyFileFormat_IO`. Of course, you will substitue in your own values and particularly your file format name in the instructions below.

```
  [1/8] title (My Nion Swift Plug-In): MyFileFormat_IO
  [2/8] author (My Name): My Name
  [3/8] github_organization (github-organization): my-organization
  [4/8] github_username (github-username): my-github-username
  [5/8] repo_name (my-github-repo-name): MyFileFormat_IO
  [6/8] org_name (orgnameforpackage): myorg
  [7/8] lib_name (libnameforpackage): MyFileFormat_IO
  [8/8] release_date (2024-04-01):
```

Once the directory is created, edit the `MyFileFormat_IO/nionswift_plugin/MyFileFormat_IO_ui/__init__.py` file with content similar to the code below.

Once install the package using `pip` and you will be able to read and write files with the extension `.mff`, which stands for "MyFileFormat".

```
pip install MyFileFormat_IO
nionswift
```

See the links above for more details of installing the package in developer vs end-user mode.

```python
# standard libraries
import gettext
import pathlib
import typing

# third party libraries
import numpy

# nion swift libraries
from nion.data import Calibration
from nion.data import DataAndMetadata


_ = gettext.gettext


class IODelegate(object):

    def __init__(self, api: typing.Any) -> None:
        self.io_handler_id = "mff-io-handler"
        self.io_handler_name = _("MyFileFormat Files")
        self.io_handler_extensions = ["mff"]

    def read_data_and_metadata(self, extension: str, file_path: str) -> DataAndMetadata.DataAndMetadata:
        with open(file_path, "rb", buffering=8 * 1024 * 1024) as f:
            # does not actually read file, just generates a 100x100 random array.
            return DataAndMetadata.new_data_and_metadata(numpy.random.randn(100, 100))

    def can_write_data_and_metadata(self, data_and_metadata: DataAndMetadata.DataAndMetadata, extension: str) -> bool:
        return extension.lower() in self.io_handler_extensions

    def write_data_and_metadata(self, data_and_metadata: DataAndMetadata.DataAndMetadata, file_path_str: str, extension: str) -> None:
        file_path = pathlib.Path(file_path_str)
        data = data_and_metadata.data
        data_descriptor = data_and_metadata.data_descriptor
        dimensional_calibrations = list()
        for dimensional_calibration in data_and_metadata.dimensional_calibrations:
            offset, scale, units = dimensional_calibration.offset, dimensional_calibration.scale, dimensional_calibration.units
            dimensional_calibrations.append(Calibration.Calibration(offset, scale, units))
        intensity_calibration = data_and_metadata.intensity_calibration
        offset, scale, units = intensity_calibration.offset, intensity_calibration.scale, intensity_calibration.units
        intensity_calibration = Calibration.Calibration(offset, scale, units)
        metadata = data_and_metadata.metadata
        timestamp = data_and_metadata.timestamp
        timezone = data_and_metadata.timezone
        timezone_offset = data_and_metadata.timezone_offset
        # does not actually write file, just writes "file content" to the file.
        with open(file_path, 'wb', buffering=32 * 1024 * 1024) as f:
            f.write("file content".encode("utf-8"))


class IOExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "myorg.mff_file_io"

    def __init__(self, api_broker: typing.Any) -> None:
        # grab the api object.
        api = api_broker.get_api(version="1", ui_version="1")
        # be sure to keep a reference, or it will be closed immediately.
        self.__io_handler_ref = api.create_data_and_metadata_io_handler(IODelegate(api))

    def close(self) -> None:
        # close will be called when the extension is unloaded. in turn, close any references so they get closed. this
        # is not strictly necessary since the references will be deleted naturally when this object is deleted.
        self.__io_handler_ref.close()
        self.__io_handler_ref = None
```

## More Information

-   [Changelog](https://github.com/nion-software/nionswift-io/blob/master/CHANGES.rst)
