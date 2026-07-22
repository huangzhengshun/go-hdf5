import h5py

def decode_str(value):
    if isinstance(value, bytes):
        return value.decode('utf-8')
    elif isinstance(value, (list, tuple, h5py.Dataset)):
        return [decode_str(v) for v in value]
    return value

def print_group(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"\nGroup: {name}")
        print(f"  Attributes:")
        for attr_name in obj.attrs:
            value = obj.attrs[attr_name]
            decoded = decode_str(value)
            print(f"    {attr_name}: {decoded} (raw type: {type(value).__name__})")
        print(f"  Children: {[child for child in obj]}")
    elif isinstance(obj, h5py.Dataset):
        print(f"\nDataset: {name}")
        print(f"  Shape: {obj.shape}")
        print(f"  Dtype: {obj.dtype}")
        print(f"  Data: {obj[:]}")

def main():
    path = "nested_groups_test.h5"
    
    with h5py.File(path, 'r') as f:
        print("=== HDF5 File Structure ===")
        f.visititems(print_group)
        
        print("\n=== Verifying Attributes ===")
        
        # Check level1 attributes
        level1 = f["level1"]
        assert level1.attrs["level"] == 1, f"level1 level should be 1, got {level1.attrs['level']}"
        assert decode_str(level1.attrs["description"]) == "First level group", f"level1 description mismatch"
        assert level1.attrs["count"] == 10, f"level1 count should be 10, got {level1.attrs['count']}"
        print("✓ level1 attributes verified")
        
        # Check level2 attributes
        level2 = f["level1/level2"]
        assert level2.attrs["level"] == 2, f"level2 level should be 2, got {level2.attrs['level']}"
        assert level2.attrs["data_count"] == 100, f"level2 data_count should be 100, got {level2.attrs['data_count']}"
        assert decode_str(list(level2.attrs["tags"])) == ["group", "data"], f"level2 tags mismatch"
        print("✓ level2 attributes verified")
        
        # Check level3 attributes
        level3 = f["level1/level2/level3"]
        assert level3.attrs["level"] == 3, f"level3 level should be 3, got {level3.attrs['level']}"
        assert decode_str(list(level3.attrs["tags"])) == ["important", "test", "nested"], f"level3 tags mismatch"
        assert list(level3.attrs["scores"]) == [95.5, 88.0, 76.3, 82.1], f"level3 scores mismatch"
        print("✓ level3 attributes verified")
        
        # Check dataset
        data = f["level1/level2/level3/data"]
        assert list(data[:]) == [1, 2, 3, 4, 5], f"data mismatch"
        print("✓ Dataset verified")
        
        print("\n=== All verifications passed! ===")

if __name__ == "__main__":
    main()
