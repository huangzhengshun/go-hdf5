import h5py
import sys

errors = []

def check(condition, msg):
    if not condition:
        errors.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  PASS: {msg}")

f = h5py.File('comprehensive_test.h5', 'r')
print("=== Root Group ===")

root_attrs = dict(f.attrs)
print(f"  Root attrs: {root_attrs}")
check(root_attrs.get('version') == 1, "root attr version == 1")
check(root_attrs.get('description') == b'Comprehensive h5py test file' or
      root_attrs.get('description') == 'Comprehensive h5py test file',
      "root attr description correct")

print("\n=== integers group ===")
check('integers' in f, "integers group exists")
int_grp = f['integers']
check(int_grp.attrs.get('category') in (b'integer types', 'integer types'),
      "integers.category attr")
check(bool(int_grp.attrs.get('enabled')) == True, "integers.enabled == True")

check('int32_1d' in int_grp, "int32_1d dataset exists")
ds = int_grp['int32_1d']
print(f"  int32_1d shape: {ds.shape}, dtype: {ds.dtype}")
data = ds[()]
check(len(data) == 10, "int32_1d has 10 elements")
check(data[0] == 1 and data[9] == 10, "int32_1d values correct")
check(ds.attrs.get('units') in (b'count', 'count'), "int32_1d units attr")
check(ds.attrs.get('min') == 1, "int32_1d min attr")
check(ds.attrs.get('max') == 10, "int32_1d max attr")
check(bool(ds.attrs.get('valid')) == True, "int32_1d valid == True")

check('int64_1d' in int_grp, "int64_1d dataset exists")
ds = int_grp['int64_1d']
print(f"  int64_1d shape: {ds.shape}, dtype: {ds.dtype}")
data = ds[()]
check(len(data) == 5, "int64_1d has 5 elements")
check(data[0] == -100 and data[4] == 100, "int64_1d values correct")
check(ds.attrs.get('type') in (b'signed', 'signed'), "int64_1d type attr")
check(bool(ds.attrs.get('valid')) == True, "int64_1d valid == True")

check('uint32_1d' in int_grp, "uint32_1d dataset exists (added during reopen)")
ds = int_grp['uint32_1d']
print(f"  uint32_1d shape: {ds.shape}, dtype: {ds.dtype}")
data = ds[()]
check(len(data) == 3, "uint32_1d has 3 elements")
check(data[0] == 100 and data[2] == 300, "uint32_1d values correct")
check(bool(ds.attrs.get('unsigned')) == True, "uint32_1d unsigned == True")

print("\n=== floats group ===")
check('floats' in f, "floats group exists")
float_grp = f['floats']
check(float_grp.attrs.get('category') in (b'floating point types', 'floating point types'),
      "floats.category attr")

check('float32_1d' in float_grp, "float32_1d dataset exists")
ds = float_grp['float32_1d']
print(f"  float32_1d shape: {ds.shape}, dtype: {ds.dtype}")
data = ds[()]
check(len(data) == 4, "float32_1d has 4 elements")
check(abs(data[0] - 1.5) < 0.001, "float32_1d values correct")
check(ds.attrs.get('precision') in (b'single', 'single'), "float32_1d precision attr")

check('float64_1d' in float_grp, "float64_1d dataset exists")
ds = float_grp['float64_1d']
print(f"  float64_1d shape: {ds.shape}, dtype: {ds.dtype}")
data = ds[()]
check(len(data) == 2, "float64_1d has 2 elements")
check(abs(data[0] - 3.141592653589793) < 0.0000001, "float64_1d pi correct")
check(abs(data[1] - 2.718281828459045) < 0.0000001, "float64_1d e correct")
check(ds.attrs.get('precision') in (b'double', 'double'), "float64_1d precision attr")
check(ds.attrs.get('constants') in (b'pi,e', 'pi,e'), "float64_1d constants attr")

print("\n=== multidim group ===")
check('multidim' in f, "multidim group exists")
multi_grp = f['multidim']
check(multi_grp.attrs.get('description') in (b'Multi-dimensional datasets', 'Multi-dimensional datasets'),
      "multidim.description")

check('int32_2d' in multi_grp, "int32_2d dataset exists")
ds = multi_grp['int32_2d']
print(f"  int32_2d shape: {ds.shape}, dtype: {ds.dtype}")
check(len(ds.shape) == 1, "int32_2d has 1 dimension (flattened)")
check(ds.shape[0] == 9, "int32_2d has 9 elements")
data = ds[()]
check(data[0] == 1 and data[4] == 5 and data[8] == 9, "int32_2d values correct")
check(ds.attrs.get('shape') in (b'3x3', '3x3'), "int32_2d shape attr")

print("\n=== nested groups ===")
check('nested' in f, "nested group exists")
nested = f['nested']
check(nested.attrs.get('level') == 1, "nested.level == 1")
check(bool(nested.attrs.get('updated')) == True, "nested.updated == True (added during reopen)")

check('level2a' in nested, "level2a exists")
l2a = nested['level2a']
check(l2a.attrs.get('name') in (b'level2a', 'level2a'), "level2a.name")
check(abs(l2a.attrs.get('value') - 42.0) < 0.001, "level2a.value")
check(bool(l2a.attrs.get('active')) == True, "level2a.active == True")

check('level2b' in nested, "level2b exists")
l2b = nested['level2b']
check(l2b.attrs.get('name') in (b'level2b', 'level2b'), "level2b.name")

check('level3' in l2a, "level3 exists")
l3 = l2a['level3']
check(l3.attrs.get('depth') == 3, "level3.depth == 3")
check('deep_data' in l3, "level3.deep_data exists")
ds = l3['deep_data']
data = ds[()]
check(len(data) == 3 and data[0] == 10 and data[2] == 30, "deep_data values correct")
check(ds.attrs.get('location') in (b'deep', 'deep'), "deep_data location attr")

print("\n=== empty groups ===")
check('empty_groups' in f, "empty_groups exists")
eg = f['empty_groups']
check('totally_empty' in eg, "totally_empty exists")
te = eg['totally_empty']
check(len(te.keys()) == 0, "totally_empty has no children")

check('only_attrs' in eg, "only_attrs exists")
oa = eg['only_attrs']
check(len(oa.keys()) == 0, "only_attrs has no children")
check(bool(oa.attrs.get('has_data')) == False, "only_attrs.has_data == False")
check(oa.attrs.get('purpose') in (b'attribute only test', 'attribute only test'),
      "only_attrs.purpose")

print("\n=== multiple_datasets ===")
check('multiple_datasets' in f, "multiple_datasets exists")
md = f['multiple_datasets']
check(md.attrs.get('dataset_count') == 4, "md.dataset_count == 4")

for i in range(4):
    name = f'ds_{i}'
    check(name in md, f"{name} exists")
    if name in md:
        ds = md[name]
        data = ds[()]
        expected = [i*10 + j for j in range(5)]
        check(list(data) == expected, f"{name} values correct")
        check(ds.attrs.get('index') == i, f"{name} index attr")
        check(bool(ds.attrs.get('active')) == (i % 2 == 0), f"{name} active attr")

print("\n=== strings group ===")
check('strings' in f, "strings group exists")
sg = f['strings']
check(sg.attrs.get('encoding') in (b'ascii', 'ascii'), "strings.encoding")
check(sg.attrs.get('count') == 3, "strings.count")
check(sg.attrs.get('sample') in (b'hello world', 'hello world'), "strings.sample")
check(bool(sg.attrs.get('has_string_data')) == True, "strings.has_string_data == True")

check('string_array' in sg, "string_array exists")
ds = sg['string_array']
print(f"  string_array shape: {ds.shape}, dtype: {ds.dtype}")
check(len(ds.shape) == 1, "string_array has 1 dimension")
check(ds.shape[0] == 4, "string_array has 4 elements")
data = ds[()]
print(f"  string_array data: {data}")
expected = [b'hello', b'world', b'hdf5', b'test']
if isinstance(data[0], bytes):
    check(list(data) == expected, "string_array values correct")
else:
    check(list(data) == [s.decode('ascii') for s in expected], "string_array values correct")
check(ds.attrs.get('length') == 4, "string_array length attr")

print("\n=== reopen_added group (added during phase 2) ===")
check('reopen_added' in f, "reopen_added group exists")
ra = f['reopen_added']
check(bool(ra.attrs.get('added_during_reopen')) == True, "reopen_added.added_during_reopen == True")
check(ra.attrs.get('timestamp') in (b'phase2', 'phase2'), "reopen_added.timestamp")
check('reopen_data' in ra, "reopen_data exists")
ds = ra['reopen_data']
data = ds[()]
check(len(data) == 3, "reopen_data has 3 elements")
check(abs(data[0] - 1.1) < 0.001 and abs(data[2] - 3.3) < 0.001, "reopen_data values correct")
check(ds.attrs.get('source') in (b'reopen', 'reopen'), "reopen_data source attr")

print("\n=== File Structure Walk ===")
def walk(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"  Group: {name}")
        for an in obj.attrs:
            print(f"    @{an}")
    else:
        print(f"  Dataset: {name} (shape={obj.shape}, dtype={obj.dtype})")

f.visititems(walk)

print("\n=== Object Header Verification ===")
def verify_objects(name, obj):
    try:
        obj.id
        check(True, f"Object {name} is accessible")
    except Exception as e:
        check(False, f"Object {name} failed: {e}")

f.visititems(verify_objects)

f.close()

print("\n" + "=" * 50)
if errors:
    print(f"FAILED: {len(errors)} errors")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED!")
    sys.exit(0)
