Changelog (nionswift-io)
========================

NEXT RELEASE
------------
- Drop support for Python 3.7.
- Fix handling of spectrum image export in DM format (now recognized in DM).

0.14.3 (2021-01-16)
-------------------
- Partially fix issue storing tuple-of-tuple as rectangle in dm3 import/export.

0.14.2 (2020-11-04)
-------------------
- Write out high tension tag with no units.
- Improve writing SI data (always mark 1D datum as a spectrum).

0.14.1 (2020-11-04)
-------------------
- Skipped.

0.14.0 (2019-11-12)
-------------------
- Add support for writing DM4 files. Contributed by Marcel Tencé.

0.13.9 (2019-04-17)
-------------------
- Handle large format data items (HDF5 backed) when exporting to TIFF.
- Improve support for multi-dimensional data as available in latest TIFF library.

0.13.8 (2019-01-31)
-------------------
- Improve treatment of 1D collections/sequences of spectra/images when exporting/import to DM file.

0.13.7 (2018-12-11)
-------------------
- Improve treatment of spectrum images when exporting/import to DM file.

0.13.6 (2018-06-18)
-------------------
- Improve support for exporting large format (HDF5) data items to DM file format.

0.13.5 (2018-05-21)
-------------------
- Add support for timestamp/timezone when exporting/importing to DM file.

0.13.4 (2018-05-20)
-------------------
- Fix bug with large integers (showing up in timestamp).

0.13.2 (2018-05-17)
-------------------
- Improve recognition of 1d data (spectrum) during import. Also sequences.

0.13.0 (2018-05-10)
-------------------
- Initial version online.
