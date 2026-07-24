import h5py
import numpy as np

def check(condition, msg):
    if condition:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
        raise AssertionError(f"Failed: {msg}")

print("=== Edge Case Verification ===\n")

f = h5py.File('edge_test.h5', 'r')

print("=== Root Attributes ===")
attrs = dict(f.attrs)
print(f"  Root attrs: {list(attrs.keys())}")

# Test int_array attribute
if 'int_array' in f.attrs:
    data = f.attrs['int_array']
    print(f"  int_array: {data}")
    check(list(data) == [1, 2, 3, 4, 5], "int_array values correct")
    check(data.dtype == np.int32, "int_array dtype is int32")
else:
    print("  FAIL: int_array not found")

# Test float_array attribute
if 'float_array' in f.attrs:
    data = f.attrs['float_array']
    print(f"  float_array: {data}")
    check(np.allclose(data, [1.1, 2.2, 3.3]), "float_array values correct")
    check(data.dtype == np.float64, "float_array dtype is float64")
else:
    print("  FAIL: float_array not found")

# Test str_array attribute
if 'str_array' in f.attrs:
    data = f.attrs['str_array']
    print(f"  str_array: {data}")
    expected = [b'alpha', b'beta', b'gamma']
    check(list(data) == expected, "str_array values correct")
else:
    print("  FAIL: str_array not found")

print("\n=== Datasets ===")

# Test chunked_data
if 'chunked_data' in f:
    ds = f['chunked_data']
    print(f"\n  chunked_data: shape={ds.shape}, dtype={ds.dtype}")
    check(ds.shape[0] == 1000, "chunked_data has 1000 elements")
    check(ds.dtype == np.float64, "chunked_data dtype is float64")
    sample = ds[0:5]
    expected = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    check(np.allclose(sample, expected), "chunked_data first 5 values correct")
    print(f"  chunked_data chunks: {ds.chunks}")
else:
    print("  FAIL: chunked_data not found")

# Test soft link data_link
if 'data_link' in f:
    obj = f['data_link']
    print(f"\n  data_link type: {type(obj)}")
    if isinstance(obj, h5py.Dataset):
        print(f"  data_link (resolved): shape={obj.shape}, dtype={obj.dtype}")
        check(obj.shape[0] == 1000, "data_link resolved correctly")
    else:
        print(f"  data_link is not a dataset: {type(obj)}")
else:
    print("  FAIL: data_link not found")

# Test tiny_data
if 'tiny_data' in f:
    ds = f['tiny_data']
    print(f"\n  tiny_data: shape={ds.shape}, dtype={ds.dtype}")
    check(ds.shape[0] == 3, "tiny_data has 3 elements")
    check(ds.dtype == np.int8, "tiny_data dtype is int8")
    check(list(ds[:]) == [1, 2, 3], "tiny_data values correct")
else:
    print("  FAIL: tiny_data not found")

# Test uint8_data
if 'uint8_data' in f:
    ds = f['uint8_data']
    print(f"\n  uint8_data: shape={ds.shape}, dtype={ds.dtype}")
    check(ds.shape[0] == 3, "uint8_data has 3 elements")
    check(ds.dtype == np.uint8, "uint8_data dtype is uint8")
    check(list(ds[:]) == [0, 127, 255], "uint8_data values correct")
else:
    print("  FAIL: uint8_data not found")

# Test uint16_data
if 'uint16_data' in f:
    ds = f['uint16_data']
    print(f"\n  uint16_data: shape={ds.shape}, dtype={ds.dtype}")
    check(ds.shape[0] == 3, "uint16_data has 3 elements")
    check(ds.dtype == np.uint16, "uint16_data dtype is uint16")
    check(list(ds[:]) == [0, 32767, 65535], "uint16_data values correct")
else:
    print("  FAIL: uint16_data not found")

# Test uint64_data
if 'uint64_data' in f:
    ds = f['uint64_data']
    print(f"\n  uint64_data: shape={ds.shape}, dtype={ds.dtype}")
    check(ds.shape[0] == 2, "uint64_data has 2 elements")
    check(ds.dtype == np.uint64, "uint64_data dtype is uint64")
    data = list(ds[:])
    check(data[0] == 0, "uint64_data first element is 0")
    check(data[1] == 18446744073709551615, "uint64_data max value correct")
else:
    print("  FAIL: uint64_data not found")

# Test int8_data
if 'int8_data' in f:
    ds = f['int8_data']
    print(f"\n  int8_data: shape={ds.shape}, dtype={ds.dtype}")
    check(ds.shape[0] == 3, "int8_data has 3 elements")
    check(ds.dtype == np.int8, "int8_data dtype is int8")
    check(list(ds[:]) == [-128, 0, 127], "int8_data values correct")
else:
    print("  FAIL: int8_data not found")

# Test int16_data
if 'int16_data' in f:
    ds = f['int16_data']
    print(f"\n  int16_data: shape={ds.shape}, dtype={ds.dtype}")
    check(ds.shape[0] == 3, "int16_data has 3 elements")
    check(ds.dtype == np.int16, "int16_data dtype is int16")
    check(list(ds[:]) == [-32768, 0, 32767], "int16_data values correct")
else:
    print("  FAIL: int16_data not found")

# Test bool_data
if 'bool_data' in f:
    ds = f['bool_data']
    print(f"\n  bool_data: shape={ds.shape}, dtype={ds.dtype}")
    check(ds.shape[0] == 5, "bool_data has 5 elements")
    data = list(ds[:])
    print(f"  bool_data values: {data}")
    expected = [1, 0, 1, 1, 0] if ds.dtype == np.uint8 else [True, False, True, True, False]
    check(list(data) == expected, "bool_data values correct")
else:
    print("  FAIL: bool_data not found")

# Test single_int
if 'single_int' in f:
    ds = f['single_int']
    print(f"\n  single_int: shape={ds.shape}, dtype={ds.dtype}")
    check(ds.shape[0] == 1, "single_int has 1 element")
    check(ds.dtype == np.int32, "single_int dtype is int32")
    check(list(ds[:]) == [42], "single_int value correct")
else:
    print("  FAIL: single_int not found")

# Test int64_data
if 'int64_data' in f:
    ds = f['int64_data']
    print(f"\n  int64_data: shape={ds.shape}, dtype={ds.dtype}")
    check(ds.shape[0] == 3, "int64_data has 3 elements")
    check(ds.dtype == np.int64, "int64_data dtype is int64")
    data = list(ds[:])
    check(data[0] == -9223372036854775808, "int64_data min value correct")
    check(data[1] == 0, "int64_data zero correct")
    check(data[2] == 9223372036854775807, "int64_data max value correct")
else:
    print("  FAIL: int64_data not found")

print("\n=== File Structure Walk ===")
def walk(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"  Group: {name}")
        for an in obj.attrs:
            print(f"    @{an}")
    else:
        print(f"  Dataset: {name} (shape={obj.shape}, dtype={obj.dtype})")
f.visititems(walk)

f.close()

print("\n=== All edge case tests completed ===")
