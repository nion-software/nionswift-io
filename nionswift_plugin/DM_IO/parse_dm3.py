import array
import io
import struct
import logging
import re

def u(x=None, y=None):
    return str(x if x is not None else str(), y)

unicode_type = str
long_type = int
file_type = io.IOBase

def str_to_iso8859_bytes(s):
    return bytes(s, 'ISO-8859-1')

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

def get_from_file(f, stype):
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


def put_into_file(f, stype, *args):
    f.write(struct.pack(stype, *args))


class structarray(object):
    """
    A class to represent struct arrays. We store the data as a list of
    tuples, with the dm_types telling us the dm id for the  types
    """
    def __init__(self, typecodes):
        #self.dm_types = dm_types
        self.typecodes = typecodes
        self.raw_data = None

    def __eq__(self, other):
        return self.raw_data == other.raw_data and self.typecodes == other.typecodes

    def __ne__(self, other):
        return self.raw_data != other.raw_data or self.typecodes != other.typecodes

    def __repr__(self):
        return "structarray({}, {})".format(self.typecodes, self.raw_data)

    def bytelen(self, num_elements):
        return num_elements * struct.calcsize(" ".join(self.typecodes))

    def num_elements(self):
        b = self.bytelen(1)
        assert(len(self.raw_data) % b == 0)
        return len(self.raw_data) // b

    def from_file(self, f, num_elements):
        self.raw_data = array.array('b', f.read(self.bytelen(num_elements)))

    def to_file(self, f):
        f.write(bytearray(self.raw_data))


def parse_dm_header(f, outdata=None):
    """
    This is the start of the DM file. We check for some
    magic values and then treat the next entry as a tag_root

    If outdata is supplied, we write instead of read using the dictionary outdata as a source
    Hopefully parse_dm_header(newf, outdata=parse_dm_header(f)) copies f to newf
    """
    # filesize is sizeondisk - 16. But we have 8 bytes of zero at the end of
    # the file.
    if outdata is not None:  # this means we're WRITING to the file
        if verbose:
            print("write_dm_header start", f.tell())
        ver, file_size, endianness = 3, -1, 1
        put_into_file(f, "> l l l", ver, file_size, endianness)
        start = f.tell()
        parse_dm_tag_root(f, outdata)
        end = f.tell()
        # start is end of 3 long header. We want to write 2nd long
        f.seek(start - 8)
        # the real file size. We started counting after 12-byte version,fs,end
        # and we need to subtract 16 total:
        put_into_file(f, "> l", end - start + 4)
        f.seek(end)
        enda, endb = 0, 0
        put_into_file(f, "> l l", enda, endb)
        if verbose:
            print("write_dm_header end", f.tell())
    else:
        if verbose:
            print("read_dm_header start", f.tell())
        ver = get_from_file(f, "> l")
        assert ver in [3,4], "Version must be 3 or 4, not %s" % ver
        # argh. why a global?
        global size_type, version
        if ver == 3:
            size_type = 'L'  # may be Q?
            version = 3
        if ver == 4:
            size_type = 'Q'  # may be Q?
            version = 4
        file_size, endianness = get_from_file(f, ">%c l" % size_type)
        assert endianness == 1, "Endianness must be 1, not %s"%endianness
        start = f.tell()
        ret = parse_dm_tag_root(f, outdata)
        end = f.tell()
        # print("fs", file_size, end - start, (end-start)%8)
        # mfm 2013-07-11 the file_size value is not always
        # end-start, sometimes there seems to be an extra 4 bytes,
        # other times not. Let's just ignore it for the moment
        # assert(file_size == end - start)
        enda, endb = get_from_file(f, "> l l")
        assert(enda == endb == 0)
        if verbose:
            print("read_dm_header end", f.tell())
        return ret


def parse_dm_tag_root(f, outdata=None):
    if outdata is not None:  # this means we're WRITING to the file
        is_dict = 0 if isinstance(outdata, list) else 1
        _open = 0
        if is_dict:
            num_tags = sum(1 if k is not None and len(k) > 0 and v is not None else 0 for k, v in outdata.items())
        else:
            num_tags = sum(1 if v is not None else 0 for v in outdata)
        if verbose:
            print("write_dm_tag_root start {} {} {}".format(f.tell(), is_dict, num_tags))
        put_into_file(f, "> b b l", is_dict, _open, num_tags)
        if not is_dict:
            for subdata in outdata:
                if subdata is not None:
                    parse_dm_tag_entry(f, subdata, None)
        else:
            for key in outdata:
                if key is not None and len(key) > 0:  # don't write out invalid dict's
                    value = outdata[key]
                    if value is not None:
                        parse_dm_tag_entry(f, value, key)
        if verbose:
            print("write_dm_tag_root end", f.tell())
    else:
        if verbose:
            print("read_dm_tag_root start", f.tell())
        is_dict, _open, num_tags = get_from_file(f, ("> b b %c" % size_type))
        if is_dict:
            new_obj = {}
            for i in range(num_tags):
                pos = f.tell()
                name, data = parse_dm_tag_entry(f)
                assert(name is not None)
                new_obj[name] = data
        else:
            new_obj = []
            for i in range(num_tags):
                pos = f.tell()
                name, data = parse_dm_tag_entry(f)
                assert(name is None)
                new_obj.append(data)
        if verbose:
            print("read_dm_tag_root end", f.tell())
        return new_obj


def parse_dm_tag_entry(f, outdata=None, outname=None):
    if outdata is not None:  # this means we're WRITING to the file
        if verbose:
            print("write_dm_tag_entry start", f.tell())
        dtype = TAG_TYPE_ARRAY if isinstance(outdata, (dict, list)) else TAG_TYPE_DATA
        name_len = len(outname) if outname else 0
        put_into_file(f, "> b H", dtype, name_len)
        if outname:
            put_into_file(f, ">" + str(name_len) + "s", str_to_iso8859_bytes(outname))

        if dtype == TAG_TYPE_DATA:
            parse_dm_tag_data(f, outdata)
        else:
            parse_dm_tag_root(f, outdata)
        if verbose:
            print("write_dm_tag_entry end", f.tell())

    else:
        if verbose:
            print("read_dm_tag_entry start", f.tell())
        dtype, name_len = get_from_file(f, "> b H")
        if name_len:
            name = get_from_file(f, ">" + str(name_len) + "s").decode("latin")
        else:
            name = None

        if version == 4:
            extra_tag_flags = get_from_file(f, ">%c" % size_type)

        if dtype == TAG_TYPE_DATA:
            arr = parse_dm_tag_data(f)
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
                print("read_dm_tag_entry end", f.tell())
            return name, arr
        elif dtype == TAG_TYPE_ARRAY:
            result = parse_dm_tag_root(f)
            if verbose:
                print("read_dm_tag_entry end", f.tell())
            return name, result
        else:
            raise Exception("Unknown data type=" + str(dtype))


def parse_dm_tag_data(f, outdata=None):
    # todo what is id??
    # it is normally one of 1,3,7,11,19
    # we can parse lists of numbers with them all 1
    # strings work with 3
    # could id be some offset to the start of the data?
    # for simple types we just read data, for strings, we read type, length
    # for structs we read len,num, len0,type0,len1,... =num*2+2
    # structs (15) can be 7,9,11,19
    # arrays (TAG_TYPE_ARRAY) can be 3 or 11
    if outdata is not None:  # this means we're WRITING to the file
            # can we get away with a limited set that we write?
        # ie can all numbers be doubles or ints, and we have lists
        if verbose:
            print("write_dm_tag_data start", f.tell())
        _, data_type = get_structdmtypes_for_python_typeorobject(outdata)
        if not data_type:
            raise Exception("Unsupported type: {}".format(type(outdata)))
        _delim = "%%%%"
        put_into_file(f, "> 4s l l", str_to_iso8859_bytes(_delim), 0, data_type)
        pos = f.tell()
        header = dm_types[data_type](f, outdata)
        f.seek(pos-8)  # where our header_len starts
        put_into_file(f, "> l", header+1)
        f.seek(0, 2)
        if verbose:
            print("write_dm_tag_data end", f.tell())
    else:
        if verbose:
            print("read_dm_tag_data start", f.tell())
        _delim, header_len, data_type = get_from_file(f, "> 4s {size} {size}".format(size=size_type))
        assert(_delim == str_to_iso8859_bytes("%%%%"))
        ret, header = dm_types[data_type](f)
        assert(header + 1 == header_len)
        if verbose:
            print("read_dm_tag_data end", f.tell())
        return ret


# we store the id as a key and the name,
# struct format, python types in a tuple for the value
# mfm 2013-08-02 was using l, L for long and ulong but sizes vary
# on platforms
# can we use i, I instead?
# mfm 2013-11-15 looks like there's two new (or reinstated) types in DM4, 11 and 12.
# Guessing what they are here
dm_simple_names = [
    (2, "short", "h", []),
    (3, "long", "i", [int]),
    # (3, "int", "l", [int]),
    (4, "ushort", "H", []),
    (5, "uint", "I", [long_type]),
    # (5, "ulong", "L", [long]),
    (6, "float", "f", []),
    (7, "double", "d", [float]),
    (8, "bool", "b", [bool]),
    (9, "char", "b", []),
    (10, "octet", "b", []),
    (11, "int64", "q", []),
    (12, "uint64", "Q", []),
]

dm_complex_names = {
    18: "string",
    15: "struct",
    TAG_TYPE_ARRAY: "array"}


def get_dmtype_for_name(name):
    for key, _name, sc, types in dm_simple_names:
        if _name == name:
            return key
    for key, _name in iter(dm_complex_names.items()):
        if _name == name:
            return key
    return 0


def get_structdmtypes_for_python_typeorobject(typeorobj):
    """
    Return structchar, dmtype for the python (or numpy)
    type or object typeorobj.
    For more complex types we only return the dm type
    """
    # not isinstance is probably a bit more lenient than 'is'
    # ie isinstance(x,str) is nicer than type(x) is str.
    # hence we use isinstance when available
    if isinstance(typeorobj, type):
        comparer = lambda test: test is typeorobj
    else:
        comparer = lambda test: isinstance(typeorobj, test)

    if comparer(int) and not -2**31 < typeorobj < 2**31 - 1:
        return 'q', 11

    for key, name, sc, types in dm_simple_names:
        for t in types:
            if comparer(t):
                return sc, key
    if comparer(str):
        return None, get_dmtype_for_name('array')  # treat all strings as arrays!
    elif comparer(unicode_type):
        return None, get_dmtype_for_name('array')  # treat all strings as arrays!
    elif comparer(array.array):
        return None, get_dmtype_for_name('array')
    elif comparer(tuple):
        return None, get_dmtype_for_name('struct')
    elif comparer(structarray):
        return None, get_dmtype_for_name('array')
    logging.warn("No appropriate DMType found for %s, %s", typeorobj, type(typeorobj))
    return None


def get_structchar_for_dmtype(dm_type):
    for key, name, sc, types in dm_simple_names:
        if key == dm_type:
            return sc
    return None


def get_dmtype_for_structchar(struct_char):
    for key, name, sc, types in dm_simple_names:
        if struct_char == sc:
            return key
    return -1


def standard_dm_read(datatype_num, desc):
    """
    datatype_num is the number of the data type, see dm_simple_names
    above. desc is a (nicename, struct_char) tuple. We return a function
    that parses the data for us.
    """
    nicename, structchar, types = desc

    def dm_read_x(f, outdata=None):
        """Reads (or write if outdata is given) a simple data type.
        returns the data if reading and the number of bytes of header
        """
        if outdata is not None:  # this means we're WRITING to the file
            if verbose:
                print("dm_write start", structchar, outdata, "at", f.tell())
            put_into_file(f, "<" + structchar, outdata)
            if verbose:
                print("dm_write end", f.tell())
            return 0
        else:
            if verbose:
                print("dm_read start", structchar, "at", f.tell())
            result = get_from_file(f, "<" + structchar)
            if verbose:
                print("dm_read end", f.tell())
            return result, 0

    return dm_read_x

dm_types = {}
for key, name, sc, types in dm_simple_names:
    dm_types[key] = standard_dm_read(key, (name, sc, types))
# 8 is boolean, and relatively easy:


def dm_read_bool(f, outdata=None):
    if outdata:  # this means we're WRITING to the file
        if verbose:
            print("dm_write_bool start", f.tell())
        put_into_file(f, "<b", 1 if outdata else 0)
        if verbose:
            print("dm_write_bool end", f.tell())
        return 0
    else:
        if verbose:
            print("dm_read_bool start", f.tell())
        result = get_from_file(f, "<b")
        if verbose:
            print("dm_read_bool end", f.tell())
        return result != 0, 0
dm_types[get_dmtype_for_name('bool')] = dm_read_bool
# string is 18:


# mfm 2013-05-13 looks like this is never used, and all strings are
# treated as array?
def dm_read_string(f, outdata=None):
    header_size = 1  # just a length field
    if outdata is not None:  # this means we're WRITING to the file
        if verbose:
            print("dm_write_string start", f.tell())
        outdata = outdata.encode("utf_16_le")
        slen = len(outdata)
        put_into_file(f, ">L", slen)
        put_into_file(f, ">" + str(slen) + "s", str_to_iso8859_bytes(outdata))
        if verbose:
            print("dm_write_string end", f.tell())
        return header_size
    else:
        assert(False)
        if verbose:
            print("dm_read_string start", f.tell())
        slen = get_from_file(f, ">L")
        raws = get_from_file(f, ">" + str(slen) + "s")
        if verbose:
            print("dm_read_string end", f.tell())
        return u(raws, "utf_16_le"), header_size

dm_types[get_dmtype_for_name('string')] = dm_read_string


# struct is 15
def dm_read_struct_types(f, outtypes=None):
    if outtypes is not None:
        _len, nfields = 0, len(outtypes)
        put_into_file(f, "> l l", _len, nfields)
        for t in outtypes:
            _len = 0
            put_into_file(f, "> l l", _len, t)
        return 2+2*len(outtypes)
    else:
        types = []
        _len, nfields = get_from_file(f, "> {size} {size}".format(size=size_type))
        assert(_len == 0)  # is it always?
        for i in range(nfields):
            _len, dtype = get_from_file(f, "> {size} {size}".format(size=size_type))
            types.append(dtype)
            assert(_len == 0)
            assert(dtype != 15)  # we don't allow structs of structs?
        return types, 2+2*nfields


def dm_read_struct(f, outdata=None):
    if outdata is not None:  # this means we're WRITING to the file
        if verbose:
            print("dm_write_struct start", f.tell())
        start = f.tell()
        types = [get_structdmtypes_for_python_typeorobject(x)[1]
                 for x in outdata]
        header = dm_read_struct_types(f, types)
        for t, data in zip(types, outdata):
            dm_types[t](f, data)
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
            print("dm_write_struct end", f.tell())
        return header
    else:
        if verbose:
            print("dm_read_struct start", f.tell())
        types, header = dm_read_struct_types(f)
        ret = []
        for t in types:
            d, h = dm_types[t](f)
            ret.append(d)
        if verbose:
            print("dm_read_struct end", f.tell())
        return tuple(ret), header

dm_types[get_dmtype_for_name('struct')] = dm_read_struct


# array is TAG_TYPE_ARRAY
def dm_read_array(f, outdata=None):
    array_header = 2  # type, length
    if outdata is not None:  # this means we're WRITING to the file
        if verbose:
            print("dm_write_array start", f.tell())
        if isinstance(outdata, structarray):
            # we write type, struct_types, length
            outdmtypes = [get_dmtype_for_structchar(s) for s in outdata.typecodes]
            put_into_file(f, "> l", get_dmtype_for_name('struct'))
            struct_header = dm_read_struct_types(f, outtypes=outdmtypes)
            put_into_file(f, "> L", outdata.num_elements())
            outdata.to_file(f)
            if verbose:
                print("dm_write_array1 end", f.tell())
            return struct_header + array_header
        elif isinstance(outdata, (str, unicode_type, array.array)):
            if isinstance(outdata, (str, unicode_type)):
                outdata = array.array('H', outdata.encode("utf_16_le"))
            assert(isinstance(outdata, array.array))
            dtype = get_dmtype_for_structchar(outdata.typecode)
            if dtype < 0:
                print("typecode %s" % outdata.typecode)
            assert dtype >= 0
            put_into_file(f, "> l", dtype)
            put_into_file(f, "> L", int(len(outdata.tobytes()) / struct.calcsize(outdata.typecode)))
            if verbose:
                print("dm_write_array2 end", dtype, len(outdata), outdata.typecode, f.tell())
            if isinstance(f, file_type):
                outdata.tofile(f)
            else:
                f.write(outdata.tobytes())
            if verbose:
                print("dm_write_array3 end", f.tell())
            return array_header
        else:
            logging.warn("Unsupported type for conversion to array:%s", outdata)

    else:
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
            print("dm_read_array start", f.tell())
        pos = f.tell()
        dtype = get_from_file(f, "> {size}".format(size=size_type))
        if dtype == get_dmtype_for_name('struct'):
            types, struct_header = dm_read_struct_types(f)
            # NB this was '> L', but changing to > {size}. May break things!
            alen = get_from_file(f, "> {size}".format(size=size_type))
            ret = structarray([get_structchar_for_dmtype(d) for d in types])
            ret.from_file(f, alen)
            if verbose:
                print("dm_read_array1 end", f.tell())
            return ret, array_header + struct_header
        else:
            # mfm 2013-08-02 struct.calcsize('l') is 4 on win and 8 on Mac!
            # however >l, <l is 4 on both... could be a bug?
            # Can we get around this by adding '>' to out structchar?
            # nope, array only takes a sinlge char. Trying i, I instead
            struct_char = get_structchar_for_dmtype(dtype)
            ret = array.array(struct_char)
            # NB this was '> L', but changing to > {size}. May break things!
            alen = get_from_file(f, "> {size}".format(size=size_type))
            if alen:
                # faster to read <1024f than <f 1024 times. probly
                # stype = "<" + str(alen) + dm_simple_names[dtype][1]
                # ret = get_from_file(f, stype)
                if verbose:
                    print("dm_read_array2 end", dtype, alen, ret.typecode, f.tell())
                if isinstance(f, file_type):
                    ret.fromfile(f, alen)
                else:
                    ret.fromstring(f.read(alen*struct.calcsize(ret.typecode)))
            # if dtype == get_dmtype_for_name('ushort'):
            #     ret = ret.tobytes().decode("utf-16")
            if verbose:
                print("dm_read_array3 end", f.tell())
            return ret, array_header

dm_types[get_dmtype_for_name('array')] = dm_read_array
