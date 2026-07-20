package hdf5

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/huangzhengshun/go-hdf5/internal/message"
)

func TestCreateDatasetInt(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hdf5-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test_dataset_int.h5")

	// Create new HDF5 file
	f, err := Create(testFile)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	// Create a dataset with integer data
	data := []int32{1, 2, 3, 4, 5}
	_, err = f.Root().CreateDataset("integers", data)
	if err != nil {
		t.Fatalf("CreateDataset failed: %v", err)
	}

	f.Close()

	// Reopen and verify
	f2, err := Open(testFile)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f2.Close()

	// Open the dataset
	ds, err := f2.Root().OpenDataset("integers")
	if err != nil {
		t.Fatalf("OpenDataset failed: %v", err)
	}

	// Verify shape
	shape := ds.Shape()
	if len(shape) != 1 || shape[0] != 5 {
		t.Errorf("Expected shape [5], got %v", shape)
	}

	// Read data back
	var result []int32
	if err := ds.Read(&result); err != nil {
		t.Fatalf("Read failed: %v", err)
	}

	for i, v := range result {
		if v != data[i] {
			t.Errorf("Element %d: expected %d, got %d", i, data[i], v)
		}
	}
}

func TestCreateDatasetFloat64(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hdf5-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test_dataset_float.h5")

	f, err := Create(testFile)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	// Create a dataset with float data
	data := []float64{1.1, 2.2, 3.3, 4.4, 5.5}
	_, err = f.Root().CreateDataset("floats", data)
	if err != nil {
		t.Fatalf("CreateDataset failed: %v", err)
	}

	f.Close()

	// Reopen and verify
	f2, err := Open(testFile)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f2.Close()

	ds, err := f2.Root().OpenDataset("floats")
	if err != nil {
		t.Fatalf("OpenDataset failed: %v", err)
	}

	var result []float64
	if err := ds.Read(&result); err != nil {
		t.Fatalf("Read failed: %v", err)
	}

	for i, v := range result {
		if v != data[i] {
			t.Errorf("Element %d: expected %f, got %f", i, data[i], v)
		}
	}
}

func TestCreateDatasetWithType(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hdf5-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test_dataset_typed.h5")

	f, err := Create(testFile)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	// Create dataset with explicit type
	dtype := message.NewFixedPointDatatype(4, true, message.OrderLE)
	ds, err := f.Root().CreateDatasetWithType("typed", []uint64{10}, dtype)
	if err != nil {
		t.Fatalf("CreateDatasetWithType failed: %v", err)
	}

	// Write data
	data := []int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	if err := ds.Write(data); err != nil {
		t.Fatalf("Write failed: %v", err)
	}

	f.Close()

	// Reopen and verify
	f2, err := Open(testFile)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f2.Close()

	ds2, err := f2.Root().OpenDataset("typed")
	if err != nil {
		t.Fatalf("OpenDataset failed: %v", err)
	}

	var result []int32
	if err := ds2.Read(&result); err != nil {
		t.Fatalf("Read failed: %v", err)
	}

	for i, v := range result {
		if v != data[i] {
			t.Errorf("Element %d: expected %d, got %d", i, data[i], v)
		}
	}
}

func TestCreateDatasetInGroup(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hdf5-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test_dataset_group.h5")

	f, err := Create(testFile)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	// Create a group
	grp, err := f.Root().CreateGroup("data")
	if err != nil {
		t.Fatalf("CreateGroup failed: %v", err)
	}

	// Create dataset in the group
	data := []float32{1.0, 2.0, 3.0}
	_, err = grp.CreateDataset("values", data)
	if err != nil {
		t.Fatalf("CreateDataset in group failed: %v", err)
	}

	f.Close()

	// Reopen and verify
	f2, err := Open(testFile)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f2.Close()

	// Open dataset via path
	ds, err := f2.OpenDataset("/data/values")
	if err != nil {
		t.Fatalf("OpenDataset failed: %v", err)
	}

	var result []float32
	if err := ds.Read(&result); err != nil {
		t.Fatalf("Read failed: %v", err)
	}

	for i, v := range result {
		if v != data[i] {
			t.Errorf("Element %d: expected %f, got %f", i, data[i], v)
		}
	}
}

func TestCreateMultipleDatasets(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hdf5-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test_multi_ds.h5")

	f, err := Create(testFile)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	// Create multiple datasets
	_, err = f.Root().CreateDataset("ds1", []int32{1, 2, 3})
	if err != nil {
		t.Fatalf("CreateDataset ds1 failed: %v", err)
	}

	_, err = f.Root().CreateDataset("ds2", []float64{1.1, 2.2})
	if err != nil {
		t.Fatalf("CreateDataset ds2 failed: %v", err)
	}

	_, err = f.Root().CreateDataset("ds3", []uint8{255, 128, 64})
	if err != nil {
		t.Fatalf("CreateDataset ds3 failed: %v", err)
	}

	f.Close()

	// Reopen and verify all exist
	f2, err := Open(testFile)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f2.Close()

	for _, name := range []string{"ds1", "ds2", "ds3"} {
		_, err := f2.Root().OpenDataset(name)
		if err != nil {
			t.Errorf("OpenDataset %s failed: %v", name, err)
		}
	}
}

func TestCreateChunkedDataset(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hdf5-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test_chunked.h5")

	f, err := Create(testFile)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	// Create a chunked dataset (data fits in single chunk)
	data := []int32{1, 2, 3, 4, 5}
	_, err = f.Root().CreateDataset("chunked_data", data, WithChunks(10))
	if err != nil {
		t.Fatalf("CreateDataset with chunks failed: %v", err)
	}

	f.Close()

	// Reopen and verify
	f2, err := Open(testFile)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f2.Close()

	ds, err := f2.Root().OpenDataset("chunked_data")
	if err != nil {
		t.Fatalf("OpenDataset failed: %v", err)
	}

	// Verify shape
	shape := ds.Shape()
	if len(shape) != 1 || shape[0] != 5 {
		t.Errorf("Expected shape [5], got %v", shape)
	}

	// Read data back
	var result []int32
	if err := ds.Read(&result); err != nil {
		t.Fatalf("Read failed: %v", err)
	}

	for i, v := range result {
		if v != data[i] {
			t.Errorf("Element %d: expected %d, got %d", i, data[i], v)
		}
	}
}

func TestCreateChunkedDatasetFloat64(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hdf5-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test_chunked_float.h5")

	f, err := Create(testFile)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	// Create a chunked dataset with float64 data
	data := []float64{1.1, 2.2, 3.3, 4.4, 5.5}
	_, err = f.Root().CreateDataset("chunked_floats", data, WithChunks(10))
	if err != nil {
		t.Fatalf("CreateDataset with chunks failed: %v", err)
	}

	f.Close()

	// Reopen and verify
	f2, err := Open(testFile)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f2.Close()

	ds, err := f2.Root().OpenDataset("chunked_floats")
	if err != nil {
		t.Fatalf("OpenDataset failed: %v", err)
	}

	// Read data back
	var result []float64
	if err := ds.Read(&result); err != nil {
		t.Fatalf("Read failed: %v", err)
	}

	for i, v := range result {
		if v != data[i] {
			t.Errorf("Element %d: expected %f, got %f", i, data[i], v)
		}
	}
}

func TestCreateMultiChunkDataset(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hdf5-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test_multi_chunk.h5")

	f, err := Create(testFile)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	// Create a dataset with 100 elements, chunk size 10 = 10 chunks
	data := make([]int32, 100)
	for i := range data {
		data[i] = int32(i)
	}
	_, err = f.Root().CreateDataset("multi_chunk", data, WithChunks(10))
	if err != nil {
		t.Fatalf("CreateDataset with multi-chunk failed: %v", err)
	}

	f.Close()

	// Reopen and verify
	f2, err := Open(testFile)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f2.Close()

	ds, err := f2.Root().OpenDataset("multi_chunk")
	if err != nil {
		t.Fatalf("OpenDataset failed: %v", err)
	}

	// Verify shape
	shape := ds.Shape()
	if len(shape) != 1 || shape[0] != 100 {
		t.Errorf("Expected shape [100], got %v", shape)
	}

	// Read data back
	var result []int32
	if err := ds.Read(&result); err != nil {
		t.Fatalf("Read failed: %v", err)
	}

	if len(result) != 100 {
		t.Fatalf("Expected 100 elements, got %d", len(result))
	}

	for i, v := range result {
		if v != data[i] {
			t.Errorf("Element %d: expected %d, got %d", i, data[i], v)
		}
	}
}
