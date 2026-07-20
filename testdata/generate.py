#!/usr/bin/env python3
"""Generate HDF5 test files for go-hdf5 testing."""

import numpy as np

try:
    import h5py
except ImportError:
    print("h5py not installed. Install with: pip install h5py numpy")
    exit(1)

# Use latest file format to get Link messages instead of symbol tables
# track_order ensures creation order is preserved
def create_file(name, libver='latest'):
    return h5py.File(name, 'w', libver=libver, track_order=True)

def create_file_v0(name):
    """Create file with v0 superblock (earliest format)."""
    return h5py.File(name, 'w', libver='earliest')

# Minimal test file - simplest possible HDF5 file
with create_file('minimal.h5') as f:
    f.create_dataset('data', data=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))

# Various integer types
with create_file('integers.h5') as f:
    for dtype in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']:
        f.create_dataset(dtype, data=np.array([1, 2, 3, 4, 5], dtype=dtype))

# Various float types
with create_file('floats.h5') as f:
    f.create_dataset('float32', data=np.array([1.5, 2.5, 3.5], dtype=np.float32))
    f.create_dataset('float64', data=np.array([1.5, 2.5, 3.5], dtype=np.float64))

# Multidimensional arrays
with create_file('multidim.h5') as f:
    f.create_dataset('2d', data=np.arange(12).reshape(3, 4).astype(np.int32))
    f.create_dataset('3d', data=np.arange(24).reshape(2, 3, 4).astype(np.float64))

# Chunked datasets (no compression)
with create_file('chunked.h5') as f:
    data = np.arange(100).reshape(10, 10).astype(np.float64)
    f.create_dataset('chunked', data=data, chunks=(5, 5))

# Compressed datasets
with create_file('compressed.h5') as f:
    data = np.random.rand(100, 100).astype(np.float64)
    f.create_dataset('gzip', data=data, chunks=(10, 10), compression='gzip', compression_opts=6)
    f.create_dataset('shuffle_gzip', data=data, chunks=(10, 10), compression='gzip', shuffle=True)

# Groups and hierarchy
with create_file('groups.h5') as f:
    grp1 = f.create_group('group1')
    grp2 = f.create_group('group2')
    grp1.create_dataset('data', data=np.array([1, 2, 3]))
    subgrp = grp1.create_group('subgroup')
    subgrp.create_dataset('nested', data=np.array([4, 5, 6]))

# Strings
with create_file('strings.h5') as f:
    # Fixed-length strings
    dt = h5py.string_dtype(encoding='utf-8', length=10)
    f.create_dataset('fixed', data=['hello', 'world'], dtype=dt)
    # Variable-length strings
    f.create_dataset('variable', data=['hello', 'variable length world'])

# Attributes
with create_file('attributes.h5') as f:
    ds = f.create_dataset('data', data=np.array([1, 2, 3]))
    ds.attrs['int_attr'] = 42
    ds.attrs['float_attr'] = 3.14
    # Use fixed-length string for easier parsing (variable-length requires global heap)
    ds.attrs.create('string_attr', 'hello', dtype=h5py.string_dtype(encoding='utf-8', length=10))
    f.attrs.create('file_attr', 'file level attribute', dtype=h5py.string_dtype(encoding='utf-8', length=30))

# Compact storage (small dataset stored in object header)
with create_file('compact.h5') as f:
    # Small datasets typically use compact storage
    f.create_dataset('compact', data=np.array([1, 2, 3, 4]), dtype=np.int32)

# Variable-length string attributes (uses global heap)
with create_file('varlen_attrs.h5') as f:
    ds = f.create_dataset('data', data=np.array([1, 2, 3]))
    # Default string attrs use variable-length strings
    ds.attrs['description'] = 'A variable length string attribute'
    ds.attrs['author'] = 'Test Author'
    ds.attrs['notes'] = 'This is a longer string that tests variable-length storage in the global heap'

# V0 superblock format files (earliest/legacy format)
with create_file_v0('v0_minimal.h5') as f:
    f.create_dataset('data', data=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))

with create_file_v0('v0_integers.h5') as f:
    f.create_dataset('int32', data=np.array([1, 2, 3, 4, 5], dtype=np.int32))
    f.create_dataset('int64', data=np.array([10, 20, 30], dtype=np.int64))

with create_file_v0('v0_attributes.h5') as f:
    ds = f.create_dataset('data', data=np.array([1, 2, 3]))
    ds.attrs['int_attr'] = 42
    ds.attrs['float_attr'] = 3.14
    ds.attrs['string_attr'] = 'hello'

# V0 superblock with nested groups, datasets, and attributes
with create_file_v0('v0_nested_attrs.h5') as f:
    # Root-level attributes
    f.attrs['file_version'] = 1
    f.attrs['file_description'] = 'Test file for nested attributes'

    # Level 1 group with attributes
    grp1 = f.create_group('sensors')
    grp1.attrs['sensor_count'] = 3
    grp1.attrs['location'] = 'building_a'

    # Dataset in level 1 group with attributes
    temp_data = np.array([22.5, 23.1, 22.8, 23.5, 24.0], dtype=np.float64)
    temp_ds = grp1.create_dataset('temperature', data=temp_data)
    temp_ds.attrs['units'] = 'celsius'
    temp_ds.attrs['calibration_date'] = '2024-01-15'
    temp_ds.attrs['min_value'] = 22.5
    temp_ds.attrs['max_value'] = 24.0

    # Another dataset in level 1 group
    humidity_data = np.array([45, 48, 52, 50, 47], dtype=np.int32)
    humidity_ds = grp1.create_dataset('humidity', data=humidity_data)
    humidity_ds.attrs['units'] = 'percent'
    humidity_ds.attrs['sensor_id'] = 101

    # Level 2 nested group with attributes
    subgrp = grp1.create_group('metadata')
    subgrp.attrs['created_by'] = 'test_generator'
    subgrp.attrs['version'] = 2

    # Dataset in level 2 group with attributes
    timestamps = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int64)
    ts_ds = subgrp.create_dataset('timestamps', data=timestamps)
    ts_ds.attrs['timezone'] = 'UTC'
    ts_ds.attrs['epoch'] = 1704067200

    # Second level 1 group
    grp2 = f.create_group('config')
    grp2.attrs['active'] = 1
    config_ds = grp2.create_dataset('settings', data=np.array([1, 2, 3, 4], dtype=np.int32))
    config_ds.attrs['readonly'] = 0
    config_ds.attrs['priority'] = 5

# V0 superblock with deeply nested groups (5 levels)
with create_file_v0('v0_deep_nested.h5') as f:
    # Level 1
    l1 = f.create_group('level1')
    l1.attrs['depth'] = 1
    l1.create_dataset('data1', data=np.array([1, 2, 3]))

    # Level 2
    l2 = l1.create_group('level2')
    l2.attrs['depth'] = 2
    l2.create_dataset('data2', data=np.array([4, 5, 6]))

    # Level 3
    l3 = l2.create_group('level3')
    l3.attrs['depth'] = 3
    l3.create_dataset('data3', data=np.array([7, 8, 9]))

    # Level 4
    l4 = l3.create_group('level4')
    l4.attrs['depth'] = 4
    l4.create_dataset('data4', data=np.array([10, 11, 12]))

    # Level 5
    l5 = l4.create_group('level5')
    l5.attrs['depth'] = 5
    l5.create_dataset('data5', data=np.array([13, 14, 15]))

    # Sibling groups at each level
    l1.create_group('sibling1')
    l2.create_group('sibling2')
    l3.create_group('sibling3')

# Compound type attributes
with create_file('compound_attrs.h5') as f:
    ds = f.create_dataset('data', data=np.array([1, 2, 3]))
    # Create a compound type for a 3D point
    compound_dt = np.dtype([('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
    ds.attrs.create('point', np.array((1.0, 2.0, 3.0), dtype=compound_dt))
    # Create a more complex compound with different types
    complex_dt = np.dtype([('id', 'i4'), ('value', 'f8'), ('count', 'i4')])
    ds.attrs.create('record', np.array((42, 3.14, 100), dtype=complex_dt))

# Array type attributes
with create_file('array_attrs.h5') as f:
    ds = f.create_dataset('data', data=np.array([1, 2, 3]))
    # 2x2 matrix of int32
    ds.attrs['matrix'] = np.array([[1, 2], [3, 4]], dtype=np.int32)
    # 1D array of float64
    ds.attrs['vector'] = np.array([1.0, 2.0, 3.0], dtype=np.float64)

# Soft links test file
with create_file('softlink.h5') as f:
    # Direct target dataset
    f.create_dataset('target_dataset', data=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64))
    f['link_to_dataset'] = h5py.SoftLink('/target_dataset')

    # Group with nested data and link back
    grp = f.create_group('target_group')
    grp.create_dataset('nested', data=np.array([10, 20, 30], dtype=np.int32))
    grp['link_back'] = h5py.SoftLink('/target_dataset')
    f['link_to_group'] = h5py.SoftLink('/target_group')

    # Chained links: link_to_link -> link_to_dataset -> target_dataset
    f['link_to_link'] = h5py.SoftLink('/link_to_dataset')

# External links target file
with create_file('external_target.h5') as f:
    f.create_dataset('data', data=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64))
    grp = f.create_group('subgroup')
    grp.create_dataset('nested_data', data=np.array([10, 20, 30], dtype=np.int64))

# External links source file
with create_file('external_source.h5') as f:
    # Local data for comparison
    f.create_dataset('local_data', data=np.array([100, 200, 300], dtype=np.int64))
    # External link to dataset
    f['link_to_data'] = h5py.ExternalLink('external_target.h5', '/data')
    # External link to group
    f['link_to_subgroup'] = h5py.ExternalLink('external_target.h5', '/subgroup')
    # External link to nested dataset
    f['link_to_nested'] = h5py.ExternalLink('external_target.h5', '/subgroup/nested_data')

# === EDGE CASE TEST FILES ===

# Circular soft link (self-referencing)
with create_file('circular_self.h5') as f:
    f.create_dataset('real_data', data=np.array([1, 2, 3], dtype=np.int32))
    f['circular'] = h5py.SoftLink('/circular')  # Points to itself

# Multi-level circular soft links (A -> B -> C -> A)
with create_file('circular_chain.h5') as f:
    f.create_dataset('real_data', data=np.array([1, 2, 3], dtype=np.int32))
    # Create cycle: link_a -> link_b -> link_c -> link_a
    f['link_a'] = h5py.SoftLink('/link_b')
    f['link_b'] = h5py.SoftLink('/link_c')
    f['link_c'] = h5py.SoftLink('/link_a')

# Soft link to non-existent target
with create_file('dangling_link.h5') as f:
    f.create_dataset('real_data', data=np.array([1, 2, 3], dtype=np.int32))
    f['missing'] = h5py.SoftLink('/does_not_exist')
    f['missing_nested'] = h5py.SoftLink('/nonexistent/path/deep')

# Deep chain of soft links (test recursion limits)
# Use v0 format (earliest) to avoid fractal heap for links
with create_file_v0('deep_chain.h5') as f:
    f.create_dataset('target', data=np.array([42], dtype=np.int32))
    # Create chain: link_10 -> link_9 -> ... -> link_1 -> target
    prev_name = '/target'
    for i in range(1, 11):
        link_name = f'link_{i}'
        f[link_name] = h5py.SoftLink(prev_name)
        prev_name = '/' + link_name

# External link to missing file
with create_file('external_missing.h5') as f:
    f.create_dataset('real_data', data=np.array([1, 2, 3], dtype=np.int32))
    f['missing_file'] = h5py.ExternalLink('nonexistent_file.h5', '/data')

# Cross-file circular references (A.h5 -> B.h5 -> A.h5)
with create_file('circular_ext_a.h5') as f:
    f.create_dataset('data_a', data=np.array([1, 2, 3], dtype=np.int32))
    f['to_b'] = h5py.ExternalLink('circular_ext_b.h5', '/data_b')
    f['circular_back'] = h5py.ExternalLink('circular_ext_b.h5', '/back_to_a')

with create_file('circular_ext_b.h5') as f:
    f.create_dataset('data_b', data=np.array([4, 5, 6], dtype=np.int32))
    f['to_a'] = h5py.ExternalLink('circular_ext_a.h5', '/data_a')
    f['back_to_a'] = h5py.ExternalLink('circular_ext_a.h5', '/to_b')  # Creates cycle

# Soft link to root
with create_file('link_to_root.h5') as f:
    f.create_dataset('data', data=np.array([1, 2, 3], dtype=np.int32))
    grp = f.create_group('subgroup')
    grp.create_dataset('nested', data=np.array([4, 5, 6], dtype=np.int32))
    f['root_link'] = h5py.SoftLink('/')

# V1 format (earliest) with soft links - uses symbol tables
with create_file_v0('v1_softlinks.h5') as f:
    f.create_dataset('target', data=np.array([1.0, 2.0, 3.0], dtype=np.float64))
    f['soft_link'] = h5py.SoftLink('/target')
    grp = f.create_group('mygroup')
    grp.create_dataset('nested', data=np.array([10, 20, 30], dtype=np.int32))
    f['link_to_group'] = h5py.SoftLink('/mygroup')

# Mixed soft + external link chain
with create_file('mixed_chain.h5') as f:
    f.create_dataset('local', data=np.array([1, 2, 3], dtype=np.int32))
    # soft -> external target file data
    f['ext_link'] = h5py.ExternalLink('external_target.h5', '/data')
    f['soft_to_ext'] = h5py.SoftLink('/ext_link')

# B-tree v2 chunked dataset (force v2 with latest libver)
# This creates a file that uses B-tree v2 for chunk indexing
with h5py.File('btree_v2.h5', 'w', libver='latest') as f:
    # Create a chunked dataset that will use B-tree v2
    data = np.arange(10000).reshape(100, 100).astype(np.float64)
    f.create_dataset('chunked', data=data, chunks=(10, 10))
    # Also create a smaller one for quick testing
    small_data = np.arange(100).reshape(10, 10).astype(np.int32)
    f.create_dataset('small', data=small_data, chunks=(5, 5))

# B-tree v2 with compression (type 11 - with filter info)
with h5py.File('btree_v2_compressed.h5', 'w', libver='latest') as f:
    data = np.arange(10000).reshape(100, 100).astype(np.float64)
    f.create_dataset('compressed', data=data, chunks=(10, 10), compression='gzip', compression_opts=6)

print("Generated test files:")
print("  - minimal.h5")
print("  - integers.h5")
print("  - floats.h5")
print("  - multidim.h5")
print("  - chunked.h5")
print("  - compressed.h5")
print("  - groups.h5")
print("  - strings.h5")
print("  - attributes.h5")
print("  - compact.h5")
print("  - varlen_attrs.h5 (variable-length string attributes)")
print("  - v0_minimal.h5 (v0 superblock)")
print("  - v0_integers.h5 (v0 superblock)")
print("  - v0_attributes.h5 (v0 superblock)")
print("  - v0_nested_attrs.h5 (v0 superblock with nested groups/datasets/attributes)")
print("  - v0_deep_nested.h5 (v0 superblock with 5 levels of nesting)")
print("  - compound_attrs.h5 (compound type attributes)")
print("  - array_attrs.h5 (array type attributes)")
print("  - softlink.h5 (soft links)")
print("  - external_target.h5 (external link target)")
print("  - external_source.h5 (external links)")
print()
print("Edge case test files:")
print("  - circular_self.h5 (self-referencing soft link)")
print("  - circular_chain.h5 (A->B->C->A cycle)")
print("  - dangling_link.h5 (soft link to missing target)")
print("  - deep_chain.h5 (50-level soft link chain)")
print("  - external_missing.h5 (external link to missing file)")
print("  - circular_ext_a.h5 + circular_ext_b.h5 (cross-file cycle)")
print("  - link_to_root.h5 (soft link to /)")
print("  - v1_softlinks.h5 (v1 format soft links)")
print("  - mixed_chain.h5 (soft + external chain)")
print("  - btree_v2.h5 (B-tree v2 chunked dataset)")
print("  - btree_v2_compressed.h5 (B-tree v2 with compression)")
