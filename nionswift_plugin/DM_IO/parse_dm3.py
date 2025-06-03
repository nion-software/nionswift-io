from __future__ import annotations

import array
import itertools
import struct
import logging
import re
import typing

import numpy.typing

long_type = int

def str_to_iso8859_bytes(s: str) -> bytes:
    return bytes(s, 'ISO-8859-1')

# marcel 2019-03-14  Adding capability to write dm4 files
# mfm 2013-11-15 initial dm4 support
# this should probably migrate into a class at some point.
# No support for writing dm4 files, but shouldn't be hard -
# just need to make sure functions are symmetric
# mfm 2013-05-21 do we need the numpy array stuff? The python array module
# allows us to store arrays easily and efficiently. How do we deal
# with arrays of complex data? We could use numpy arrays with custom dtypes
# in which case we'd be totally tied to numpy, or else stick with structarray.
# Either way the current setup of treating arrays like numpy arrays in a few
# special cases isn't particularly nice, that could be done outside this module
# and then we'd only have to check for the basic types.
# NB our struct array class is not that different from a very basic array.array
# it has data and a list of data types. We could just store bytes in the data
# instead of lists of tuples?

# mfm 9Feb13 Simpler version than v1, but maybe less robust
# (v1 cnows about names we're trying to extract at extraction time
# this one doesn't). Is easier to follow though
verbose = False

# we treat sizes separately to distinguish 32bit (dm3) and 64 bit (dm4)
# these globals can get changed in parse_dm3_header
version = 3
size_type = "L"

TAG_TYPE_ARRAY = 20
TAG_TYPE_DATA = 21


def get_from_file(f: typing.BinaryIO, stype: str) -> typing.Any:
    #print("reading", stype, "size", struct.calcsize(stype))
    src = f.read(struct.calcsize(stype))
    if len(src) != struct.calcsize(stype):
        print("%s %s %s" % (stype, len(src), struct.calcsize(stype)))
    assert(len(src) == struct.calcsize(stype))
    d = struct.unpack(stype, src)
    if len(d) == 1:
        return d[0]
    else:
        return d


def put_into_file(f: typing.BinaryIO, stype: str, *args: typing.Any) -> None:
    f.write(struct.pack(stype, *args))


class StructArray:
    """
    A class to represent struct arrays. We store the data as a list of
    tuples, with the dm_types telling us the dm id for the  types
    """
    def __init__(self, typecodes: typing.Sequence[str]) -> None:
        self.typecodes = list(typecodes)
        self.raw_data: typing.Optional[array.array[typing.Any]] = None

    def __eq__(self, other: typing.Any) -> bool:
        if not isinstance(other, StructArray):
            return False
        return self.raw_data == other.raw_data and self.typecodes == other.typecodes

    def __ne__(self, other: typing.Any) -> bool:
        if not isinstance(other, StructArray):
            return False
        return self.raw_data != other.raw_data or self.typecodes != other.typecodes

    def __repr__(self) -> str:
        return "structarray({}, {})".format(self.typecodes, self.raw_data)

    def bytelen(self, num_elements: int) -> int:
        return num_elements * struct.calcsize(" ".join(self.typecodes))

    def num_elements(self) -> int:
        b = self.bytelen(1)
        assert(self.raw_data is not None)
        assert(len(self.raw_data) % b == 0)
        return len(self.raw_data) // b

    def from_file(self, f: typing.BinaryIO, num_elements: int) -> None:
        self.raw_data = array.array('b', f.read(self.bytelen(num_elements)))

    def to_file(self, f: typing.BinaryIO) -> None:
        assert(self.raw_data is not None)
        f.write(bytearray(self.raw_data))

    def write(self, f: typing.BinaryIO) -> int:
        # we write type, struct_types, length
        outdmtypes = [get_dmtype_for_structchar(s) for s in self.typecodes]
        put_into_file(f, "> %c" % size_type, get_dmtype_for_name('struct'))
        struct_header = dm_write_struct_types(f, outdmtypes)
        assert isinstance(struct_header, int)  # TODO: clean up dm_read_struct_types return type
        put_into_file(f, "> %c" % size_type, self.num_elements())
        self.to_file(f)
        if verbose:
            print(f"dm_write_array1 end {f.tell()}")
        return struct_header + 2  # type, length


def dm_read_header(f: typing.BinaryIO) -> typing.Any:
    """
    This is the start of the DM file. We check for some
    magic values and then treat the next entry as a tag_root

    If outdata is supplied, we write instead of read using the dictionary outdata as a source
    Hopefully dm_write_header(newf, outdata=dm_read_header(f)) copies f to newf
    """
    # filesize is sizeondisk - 16. But we have 8 bytes of zero at the end of
    # the file.
    # argh. why a global?
    global size_type, version
    if verbose:
        print(f"read_dm_header start {f.tell()}")
    ver = get_from_file(f, "> l")
    assert ver in [3,4], "Version must be 3 or 4, not %s" % ver
    if ver == 3:
        size_type = 'L'  # may be Q?
        version = 3
    if ver == 4:
        size_type = 'Q'  # may be Q?
        version = 4
    file_size, endianness = get_from_file(f, ">%c l" % size_type)
    assert endianness == 1, "Endianness must be 1, not %s"%endianness
    start = f.tell()
    ret = dm_read_tag_root(f)
    end = f.tell()
    # print("fs", file_size, end - start, (end-start)%8)
    # mfm 2013-07-11 the file_size value is not always
    # end-start, sometimes there seems to be an extra 4 bytes,
    # other times not. Let's just ignore it for the moment
    # assert(file_size == end - start)
    enda, endb = get_from_file(f, "> l l")
    assert(enda == endb == 0)
    if verbose:
        print(f"read_dm_header end {f.tell()}")
    return ret


def dm_write_header(f: typing.BinaryIO, file_version: int, outdata: typing.Any) -> None:
    """
    This is the start of the DM file. We check for some
    magic values and then treat the next entry as a tag_root

    If outdata is supplied, we write instead of read using the dictionary outdata as a source
    Hopefully dm_write_header(newf, outdata=dm_read_header(f)) copies f to newf
    """
    # filesize is sizeondisk - 16. But we have 8 bytes of zero at the end of
    # the file.
    # argh. why a global?
    global size_type, version
    if verbose:
        print(f"write_dm_header start {f.tell()}")
    file_size, endianness = 0, 1
    if file_version == 4:
        version = 4
        size_type = 'Q'
    else:
        version = 3
        size_type = 'L'
    put_into_file(f, "> l %c l" % size_type, version, file_size, endianness)
    start = f.tell()
    dm_write_tag_root(f, outdata)
    end = f.tell()
    if version == 4:
        # start is end of 1 int32 1 int64 and 1 int32 header. We want to write 2nd int32
        f.seek(start - 12)
        # the real file size. We started counting after 12-byte version,fs,end
        # and we need to subtract 16 total:
    else:
        # start is end of 3 int32 header. We want to write 2nd int32
        f.seek(start - 8)
        # the real file size. We started counting after 12-byte version,fs,end
        # and we need to subtract 16 total:
    put_into_file(f, "> %c" % size_type, end - start + 4)
    f.seek(end)
    put_into_file(f, "> l l", 0, 0)
    if verbose:
        print(f"write_dm_header end {f.tell()}")


def dm_read_tag_root(f: typing.BinaryIO) -> typing.Any:
    if verbose:
        print(f"read_dm_tag_root start {f.tell()}")
    is_dict, _open, num_tags = get_from_file(f, ("> b b %c" % size_type))
    new_obj: typing.Any
    if is_dict:
        new_obj = {}
        for i in range(num_tags):
            pos = f.tell()
            name, data = dm_read_tag_entry(f)
            assert(name is not None)
            new_obj[name] = data
    else:
        new_obj = []
        for i in range(num_tags):
            pos = f.tell()
            name, data = dm_read_tag_entry(f)
            assert(name is None)
            new_obj.append(data)
    if verbose:
        print(f"read_dm_tag_root end {f.tell()}")
    return new_obj


def dm_write_tag_root(f: typing.BinaryIO, outdata: typing.Any) -> None:
    is_dict = 0 if isinstance(outdata, list) else 1
    _open = 0
    if is_dict:
        num_tags = sum(1 if k is not None and len(k) > 0 and v is not None else 0 for k, v in outdata.items())
    else:
        num_tags = sum(1 if v is not None else 0 for v in outdata)
    if verbose:
        print(f"write_dm_tag_root start {f.tell()} {is_dict} num of tags {num_tags}")
    put_into_file(f, "> b b %c" % size_type, is_dict, _open, num_tags)
    if not is_dict:
        for subdata in outdata:
            if subdata is not None:
                dm_write_tag_entry(f, subdata, None)
    else:
        for key in outdata:
            if key is not None and len(key) > 0:  # don't write out invalid dict's
                value = outdata[key]
                if value is not None:
                    dm_write_tag_entry(f, value, key)
    if verbose:
        print(f"write_dm_tag_root end {f.tell()}")


def dm_read_tag_entry(f: typing.BinaryIO) -> typing.Any:
    if verbose:
        print(f"read_dm_tag_entry start {f.tell()}")
    dtype, name_len = get_from_file(f, "> b H")
    if name_len:
        name = get_from_file(f, ">" + str(name_len) + "s").decode("latin")
    else:
        name = None

    if version == 4:
        extra_tag_flags = get_from_file(f, ">%c" % size_type)

    if dtype == TAG_TYPE_DATA:
        arr = dm_read_tag_data(f)
        if name and hasattr(arr, "__len__") and len(arr) > 0:
            # if we find data which matches this regex we return a
            # string instead of an array
            treat_as_string_names = ['.*Name']
            for regex in treat_as_string_names:
                if re.match(regex, name):
                    if isinstance(arr[0], int):
                        arr = ''.join(chr(x) for x in arr)
                    elif isinstance(arr[0], str):
                        arr = ''.join(arr)
        if verbose:
            print(f"read_dm_tag_entry end {f.tell()}")
        return name, arr
    elif dtype == TAG_TYPE_ARRAY:
        result = dm_read_tag_root(f)
        if verbose:
            print(f"read_dm_tag_entry end {f.tell()}")
        return name, result
    else:
        raise Exception("Unknown data type=" + str(dtype))


def dm_write_tag_entry(f: typing.BinaryIO, outdata: typing.Any, outname: str | None) -> None:
    if verbose:
        print(f"write_dm_tag_entry {outname} start {f.tell()}")
    dtype = TAG_TYPE_ARRAY if isinstance(outdata, (dict, list)) else TAG_TYPE_DATA
    name_len = len(outname) if outname else 0
    put_into_file(f, "> b H", dtype, name_len)
    if outname:
        put_into_file(f, ">" + str(name_len) + "s", str_to_iso8859_bytes(outname))
    start = f.tell()
    if version == 4:
        put_into_file(f, ">%c" % size_type, 0)

    if dtype == TAG_TYPE_DATA:
        dm_write_tag_data(f, outdata)
    else:
        dm_write_tag_root(f, outdata)

    if version == 4:
        end = f.tell()
        f.seek(start)
        put_into_file(f, ">%c" % size_type, end - start - 8)
        f.seek(0, 2)

    if verbose:
        print(f"write_dm_tag_entry {outname} end {f.tell()}")


def dm_read_tag_data(f: typing.BinaryIO) -> typing.Any:
    # todo what is id??
    # it is normally one of 1,3,7,11,19
    # we can parse lists of numbers with them all 1
    # strings work with 3
    # could id be some offset to the start of the data?
    # for simple types we just read data, for strings, we read type, length
    # for structs we read len,num, len0,type0,len1,... =num*2+2
    # structs (15) can be 7,9,11,19
    # arrays (TAG_TYPE_ARRAY) can be 3 or 11
    if verbose:
        print(f"dm_read_tag_data start {f.tell()}")
    _delim2, header_len, data_type = get_from_file(f, "> 4s {size} {size}".format(size=size_type))
    assert(_delim2 == str_to_iso8859_bytes("%%%%"))
    ret, header = dm_read_types[data_type](f)
    assert(header + 1 == header_len)
    if verbose:
        print(f"dm_read_tag_data end {f.tell()}")
    return ret


def dm_write_tag_data(f: typing.BinaryIO, value: typing.Any) -> None:
    if verbose:
        print(f"dm_write_tag_data start {f.tell()}")
    _, data_type = get_structdmtypes_for_python_typeorobject(value)
    if not data_type:
        raise Exception(f"Unsupported type: {type(value)}")
    _delim = "%%%%"
    fm = "> 4s %c %c" % (size_type, size_type)
    put_into_file(f, fm, str_to_iso8859_bytes(_delim), 0, data_type)
    pos = f.tell()
    header = dm_write_types[data_type](f, value)
    if version == 4:
        f.seek(pos-16)  # where our header_len starts
    else:
        f.seek(pos-8)
    put_into_file(f, "> %c" % size_type, header+1)
    f.seek(0, 2)
    if verbose:
        print(f"dm_write_tag_data end {f.tell()}")


# we store the id as a key and the name,
# struct format, python types in a tuple for the value
# mfm 2013-08-02 was using l, L for long and ulong but sizes vary
# on platforms
# can we use i, I instead?
# mfm 2013-11-15 looks like there's two new (or reinstated) types in DM4, 11 and 12.
# Guessing what they are here
dm_simple_names: list[tuple[int, str, str, list[typing.Type[typing.Any]]]] = [
    (8, "bool", "b", [bool]),
    (2, "short", "h", []),
    (3, "long", "i", [int]),
    # (3, "int", "l", [int]),
    (4, "ushort", "H", []),
    (5, "uint", "I", [long_type]),
    # (5, "ulong", "L", [long]),
    (6, "float", "f", []),
    (7, "double", "d", [float]),
    (9, "char", "b", []),
    (10, "octet", "b", []),
    (11, "int64", "q", []),
    (12, "uint64", "Q", []),
]

dm_complex_names = {
    18: "string",
    15: "struct",
    TAG_TYPE_ARRAY: "array"}


def get_dmtype_for_name(name: str) -> int:
    for key, _name, sc, types in dm_simple_names:
        if _name == name:
            return key
    for key, _name in iter(dm_complex_names.items()):
        if _name == name:
            return key
    return 0


def get_structdmtypes_for_python_typeorobject(value: typing.Any) -> tuple[str | None, int]:
    """
    Return structchar, dmtype for the python (or numpy)
    type or object typeorobj.
    For more complex types we only return the dm type
    """
    if isinstance(value, int) and not -2**31 < value < 2**31 - 1:
        return 'q', 11

    for key, name, sc, types in dm_simple_names:
        for t in types:
            if isinstance(value, t):
                return sc, key
    if isinstance(value, str):
        return None, get_dmtype_for_name('array')  # treat all strings as arrays!
    elif isinstance(value, (array.array, DataChunkWriter)):
        return None, get_dmtype_for_name('array')
    elif isinstance(value, tuple):
        return None, get_dmtype_for_name('struct')
    elif isinstance(value, StructArray):
        return None, get_dmtype_for_name('array')
    else:
        logging.warning(f"No appropriate DMType found for {value}, {type(value)}. Trying with float32")
        return None, 6


def get_structchar_for_dmtype(dm_type: int) -> str:
    for key, name, sc, types in dm_simple_names:
        if key == dm_type:
            return sc
    # should be an exception?
    return str()


def get_dmtype_for_structchar(struct_char: str) -> int:
    for key, name, sc, types in dm_simple_names:
        if struct_char == sc:
            return key
    return -1


def standard_dm_read(datatype_num: int, desc: tuple[str, str, list[typing.Type[typing.Any]]]) -> typing.Callable[[typing.BinaryIO], typing.Any]:
    """
    datatype_num is the number of the data type, see dm_simple_names
    above. desc is a (nicename, struct_char) tuple. We return a function
    that parses the data for us.
    """
    nicename, structchar, types = desc

    def dm_read_x(f: typing.BinaryIO) -> typing.Any:
        """Reads (or write if outdata is given) a simple data type.
        returns the data if reading and the number of bytes of header
        """
        if verbose:
            print(f"dm_read start {structchar} at {f.tell()}")
        result = get_from_file(f, "<" + structchar)
        if verbose:
            print(f"dm_read end {f.tell()}")
        return result, 0

    return dm_read_x


def standard_dm_write(datatype_num: int, desc: tuple[str, str, list[typing.Type[typing.Any]]]) -> typing.Callable[[typing.BinaryIO, typing.Any], int]:
    """
    datatype_num is the number of the data type, see dm_simple_names
    above. desc is a (nicename, struct_char) tuple. We return a function
    that parses the data for us.
    """
    nicename, structchar, types = desc

    def dm_write_x(f: typing.BinaryIO, outdata: typing.Any) -> int:
        """Reads (or write if outdata is given) a simple data type.
        returns the data if reading and the number of bytes of header
        """
        if verbose:
            print(f"dm_write start 'structchar' {outdata} at {f.tell()}")
        put_into_file(f, "<" + structchar, outdata)
        if verbose:
            print(f"dm_write end {f.tell()}")
        return 0

    return dm_write_x


dm_read_types = dict[int, typing.Callable[[typing.BinaryIO], typing.Any]]()
dm_write_types = dict[int, typing.Callable[[typing.BinaryIO, typing.Any], int]]()


for key, name, sc, types in dm_simple_names:
    dm_read_types[key] = standard_dm_read(key, (name, sc, types))
    dm_write_types[key] = standard_dm_write(key, (name, sc, types))


# 8 is boolean, and relatively easy:


def dm_read_bool(f: typing.BinaryIO) -> typing.Any:
    if verbose:
        print(f"dm_read_bool start {f.tell()}")
    result = get_from_file(f, "<b")
    if verbose:
        print(f"dm_read_bool end {f.tell()}")
    return result != 0, 0


def dm_write_bool(f: typing.BinaryIO, outdata: typing.Any) -> int:
    if verbose:
        print(f"dm_write_bool start {f.tell()}")
    put_into_file(f, "<b", 1 if outdata else 0)
    if verbose:
        print(f"dm_write_bool end {f.tell()}")
    return 0


dm_read_types[get_dmtype_for_name('bool')] = dm_read_bool
dm_write_types[get_dmtype_for_name('bool')] = dm_write_bool


# string is 18:


# mfm 2013-05-13 looks like this is never used, and all strings are
# treated as array?
def dm_read_string(f: typing.BinaryIO) -> typing.Any:
    header_size = 1  # just a length field
    if verbose:
        print(f"dm_read_string start {f.tell()}")
    slen = get_from_file(f, ">%c" % size_type)
    raws = get_from_file(f, ">" + str(slen) + "s")
    if verbose:
        print(f"dm_read_string end {f.tell()}")
    return str(raws, "utf_16_le"), header_size


# mfm 2013-05-13 looks like this is never used, and all strings are
# treated as array?
def dm_write_string(f: typing.BinaryIO, outdata: typing.Any) -> int:
    header_size = 1  # just a length field
    if verbose:
        print(f"dm_write_string start {f.tell()}")
    outdata = outdata.encode("utf_16_le")
    slen = len(outdata)
    put_into_file(f, ">%c" % size_type, slen)
    put_into_file(f, ">" + str(slen) + "s", str_to_iso8859_bytes(outdata))
    if verbose:
        print(f"dm_write_string end {f.tell()}")
    return header_size


dm_read_types[get_dmtype_for_name('string')] = dm_read_string
dm_write_types[get_dmtype_for_name('string')] = dm_write_string


# struct is 15
def dm_read_struct_types(f: typing.BinaryIO) -> tuple[list[typing.Any], int]:
    types = list[numpy.typing.DTypeLike]()
    _len, nfields = get_from_file(f, "> {size} {size}".format(size=size_type))
    assert(_len == 0)  # is it always?
    for i in range(nfields):
        _len, dtype = get_from_file(f, "> {size} {size}".format(size=size_type))
        types.append(dtype)
        assert(_len == 0)
        assert(dtype != 15)  # we don't allow structs of structs?
    return types, 2+2*nfields


# struct is 15
def dm_write_struct_types(f: typing.BinaryIO, outtypes: list[int]) -> int:
    _len, nfields = 0, len(outtypes)
    put_into_file(f, "> %c %c" % (size_type, size_type), _len, nfields)
    for t in outtypes:
        _len = 0
        put_into_file(f, "> %c %c" % (size_type, size_type), _len, t)
    return 2+2*len(outtypes)


def dm_read_struct(f: typing.BinaryIO) -> typing.Any:
    if verbose:
        print(f"dm_read_struct start {f.tell()}")
    types, header = dm_read_struct_types(f)
    ret = []
    for t in types:
        d, h = dm_read_types[t](f)
        ret.append(d)
    if verbose:
        print(f"dm_read_struct end {f.tell()}")
    if len(ret) == 4:
        # this is a hack to restore flattened rectangles.
        # rectangle structures (tuples) are stored in json as ((t, l), (h, w))
        # rectangles in dm are stored as (t, l, h, w)
        # rectangles in json may also be stored as a list; this works properly in dm tags already
        # long term, there needs to either be a schema or hints to distinguish between rectangles and structs with 4 items.
        ret = [(ret[0], ret[1]), (ret[2], ret[3])]
    return tuple(ret), header


def dm_write_struct(f: typing.BinaryIO, outdata: typing.Any) -> int:
    if verbose:
        print(f"dm_write_struct start {f.tell()}")
    start = f.tell()
    if isinstance(outdata, tuple) and len(outdata) > 1 and all(isinstance(d, (tuple, list)) for d in outdata):
        outdata = tuple(itertools.chain(*outdata))
    write_types = [get_structdmtypes_for_python_typeorobject(x)[1] for x in outdata]
    header = dm_write_struct_types(f, write_types)
    for t, data in zip(write_types, outdata):
        dm_write_types[t](f, data)
    # we write length at the very end
    # but _len is probably not len, it's set to 0 for the
    # file I'm trying...
    write_len = False
    if write_len:
        end = f.tell()
        f.seek(start)
        # dm_read_struct first writes a length which we overwrite here
        # I think the length ignores the length field (4 bytes)
        put_into_file(f, "> l", end-start-4)
        f.seek(0, 2)  # the very end (2 is pos from end)
        assert(f.tell() == end)
    if verbose:
        print(f"dm_write_struct end {f.tell()}")
    return header


dm_read_types[get_dmtype_for_name('struct')] = dm_read_struct
dm_write_types[get_dmtype_for_name('struct')] = dm_write_struct


class DataChunkWriter:
    """Writes data in chunks to a file.

    Tries to avoid writing large chunks of data at once so that memory usage is minimized.

    The 64MB chunk size is a heuristic.
    """

    def __init__(self, data: numpy.typing.NDArray[typing.Any]) -> None:
        self.data = data

    def write(self, f: typing.BinaryIO) -> int:
        """Write the data to the file."""
        # data_array = array.array[typing.Any](platform_independent_char(rgb_view.dtype), rgb_view.flatten())
        # data_array = array.array[typing.Any](platform_independent_char(nparr.dtype), numpy.asarray(nparr).flatten())
        data = self.data
        data_itemsize = data.dtype.itemsize
        data_len_bytes = numpy.prod(data.shape, dtype=numpy.uint64) * data_itemsize
        data_dtype_char = data.dtype.char
        match data_dtype_char, data_itemsize:
            case 'l', 4:
                data_typecode = 'i'
            case 'l', 8:
                data_typecode = 'q'
            case 'L', 4:
                data_typecode = 'I'
            case 'L', 8:
                data_typecode = 'Q'
            case _:
                data_typecode = str(data_dtype_char)
        dtype = get_dmtype_for_structchar(data_typecode)
        assert dtype >= 0, f"typecode {data_typecode}"
        put_into_file(f, "> %c" % size_type, dtype)
        put_into_file(f, "> %c" % size_type, int(data_len_bytes / struct.calcsize(data_typecode)))
        if verbose:
            print(f"dm_write_array2 end {dtype} {len(data)} {data_typecode} {f.tell()}")
        # search for the chunk size by iterating backwards through the shape and finding the
        # largest chunk size that is less than 64MB.
        index_count = 0
        chunk_size = 1
        for n in reversed(data.shape):
            if chunk_size * n > 64 * 1024 * 1024:
                break
            index_count += 1
            chunk_size *= n
        # iterate over the remaining dimensions so that we can write the data in chunks.
        for index in numpy.ndindex(*data.shape[:len(data.shape) - index_count]):
            f.write(data[index].tobytes())
        if verbose:
            print(f"dm_write_array3 end {f.tell()}")
        return 2  # type, length


# array is TAG_TYPE_ARRAY


def dm_read_array(f: typing.BinaryIO) -> typing.Any:
    array_header = 2  # type, length
    # supports arrays of structs and arrays of types,
    # but not arrays of arrays (Is this possible)
    # actually lets just use the array object, which only allows arrays of
    # simple types!

    # arrays of structs are pretty common, eg in a simple image CLUT
    # data["DocumentObjectList"][0]["ImageDisplayInfo"]["CLUT"] is an
    # array of 3 bytes
    # we can't handle arrays of structs easily, as we use lists for
    # taglists, dicts for taggroups and arrays for array data.
    # But array.array only supports simple types. We need a new type, then.
    # let's make a structarray
    if verbose:
        print(f"dm_read_array start {f.tell()}")
    dtype = get_from_file(f, "> {size}".format(size=size_type))
    ret: typing.Any
    if dtype == get_dmtype_for_name('struct'):
        types, struct_header = dm_read_struct_types(f)
        # NB this was '> L', but changing to > {size}. May break things!
        alen = get_from_file(f, "> {size}".format(size=size_type))
        ret = StructArray([get_structchar_for_dmtype(d) for d in types])
        ret.from_file(f, alen)
        if verbose:
            print(f"dm_read_array1 end {f.tell()}")
        return ret, array_header + struct_header
    else:
        # mfm 2013-08-02 struct.calcsize('l') is 4 on win and 8 on Mac!
        # however >l, <l is 4 on both... could be a bug?
        # Can we get around this by adding '>' to out structchar?
        # nope, array only takes a single char. Trying i, I instead
        struct_char = get_structchar_for_dmtype(dtype)
        ret = array.array(struct_char)
        # NB this was '> L', but changing to > {size}. May break things!
        alen = get_from_file(f, "> {size}".format(size=size_type))
        if alen:
            # faster to read <1024f than <f 1024 times. probly
            # stype = "<" + str(alen) + dm_simple_names[dtype][1]
            # ret = get_from_file(f, stype)
            if verbose:
                print(f"dm_read_array2 end {dtype} {alen} {ret.typecode} {f.tell()}")
            ret.fromfile(f, alen)
        # if dtype == get_dmtype_for_name('ushort'):
        #     ret = ret.tobytes().decode("utf-16")
        if verbose:
            print(f"dm_read_array3 end {f.tell()}")
        return ret, array_header


def dm_write_array(f: typing.BinaryIO, outdata: typing.Any) -> int:
    array_header = 2  # type, length
    if verbose:
        print(f"dm_write_array start {f.tell()}")
    if isinstance(outdata, StructArray):
        return outdata.write(f)
    elif isinstance(outdata, DataChunkWriter):
        return outdata.write(f)
    elif isinstance(outdata, str):
        outdata = array.array('H', outdata.encode("utf_16_le"))
        assert(isinstance(outdata, array.array))
        dtype = get_dmtype_for_structchar(outdata.typecode)
        if dtype < 0:
            print("typecode %s" % outdata.typecode)
        assert dtype >= 0
        put_into_file(f, "> %c" % size_type, dtype)
        put_into_file(f, "> %c" % size_type, int(len(outdata.tobytes()) / struct.calcsize(outdata.typecode)))
        if verbose:
            print(f"dm_write_array2 end {dtype} {len(outdata)} {outdata.typecode} {f.tell()}")
        outdata.tofile(f)
        if verbose:
            print(f"dm_write_array3 end {f.tell()}")
        return array_header
    else:
        logging.warning(f"Unsupported type for conversion to array: {outdata}")
        return 0


dm_read_types[get_dmtype_for_name('array')] = dm_read_array
dm_write_types[get_dmtype_for_name('array')] = dm_write_array
