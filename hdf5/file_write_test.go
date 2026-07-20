package hdf5

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCreate(t *testing.T) {
	// Create temp directory
	tmpDir, err := os.MkdirTemp("", "hdf5-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test.h5")

	// Create new HDF5 file
	f, err := Create(testFile)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	// Verify file is writable
	if !f.writable {
		t.Error("File should be writable")
	}

	// Verify root group exists
	root := f.Root()
	if root == nil {
		t.Error("Root group should not be nil")
	}

	// Close the file
	if err := f.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// Verify file was created
	if _, err := os.Stat(testFile); os.IsNotExist(err) {
		t.Fatal("File was not created")
	}

	// Try to open the file for reading
	f2, err := Open(testFile)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f2.Close()

	// Verify superblock version
	if f2.superblock.Version < 2 {
		t.Errorf("Expected superblock version >= 2, got %d", f2.superblock.Version)
	}

	// Verify root group can be accessed
	root2 := f2.Root()
	if root2 == nil {
		t.Error("Root group should not be nil after reopen")
	}
}

func TestCreateWithOptions(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hdf5-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test_options.h5")

	// Create with 4-byte offsets
	f, err := Create(testFile, WithOffsetSize(4), WithLengthSize(4))
	if err != nil {
		t.Fatalf("Create with options failed: %v", err)
	}

	if f.superblock.OffsetSize != 4 {
		t.Errorf("Expected offset size 4, got %d", f.superblock.OffsetSize)
	}
	if f.superblock.LengthSize != 4 {
		t.Errorf("Expected length size 4, got %d", f.superblock.LengthSize)
	}

	f.Close()

	// Reopen and verify
	f2, err := Open(testFile)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f2.Close()

	if f2.superblock.OffsetSize != 4 {
		t.Errorf("Expected offset size 4 after reopen, got %d", f2.superblock.OffsetSize)
	}
}

func TestCreateFlush(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hdf5-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test_flush.h5")

	f, err := Create(testFile)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	// Flush before close
	if err := f.Flush(); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	f.Close()

	// Verify file is still valid
	f2, err := Open(testFile)
	if err != nil {
		t.Fatalf("Open after flush failed: %v", err)
	}
	f2.Close()
}

func TestOpenReadWrite(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hdf5-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test_rw.h5")

	// Create initial file with a dataset
	f1, err := Create(testFile)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	root1 := f1.Root()
	data1 := []int32{1, 2, 3, 4, 5}
	_, err = root1.CreateDataset("dataset1", data1)
	if err != nil {
		t.Fatalf("CreateDataset failed: %v", err)
	}
	f1.Close()

	// Reopen for read-write and add another dataset
	f2, err := OpenReadWrite(testFile)
	if err != nil {
		t.Fatalf("OpenReadWrite failed: %v", err)
	}

	if !f2.IsWritable() {
		t.Error("File should be writable after OpenReadWrite")
	}

	root2 := f2.Root()

	// Verify we can read existing dataset
	members, err := root2.Members()
	if err != nil {
		t.Fatalf("Members failed: %v", err)
	}
	if len(members) != 1 || members[0] != "dataset1" {
		t.Errorf("Expected [dataset1], got %v", members)
	}

	// Add a new dataset
	data2 := []float64{1.1, 2.2, 3.3}
	_, err = root2.CreateDataset("dataset2", data2)
	if err != nil {
		t.Fatalf("CreateDataset on reopened file failed: %v", err)
	}

	f2.Close()

	// Verify both datasets exist
	f3, err := Open(testFile)
	if err != nil {
		t.Fatalf("Open after modify failed: %v", err)
	}
	defer f3.Close()

	members, err = f3.Root().Members()
	if err != nil {
		t.Fatalf("Members after modify failed: %v", err)
	}

	if len(members) != 2 {
		t.Errorf("Expected 2 members, got %d: %v", len(members), members)
	}

	// Verify dataset values
	ds1, err := f3.Root().OpenDataset("dataset1")
	if err != nil {
		t.Fatalf("OpenDataset dataset1 failed: %v", err)
	}
	var readData1 []int32
	if err := ds1.Read(&readData1); err != nil {
		t.Fatalf("Read dataset1 failed: %v", err)
	}
	for i, v := range data1 {
		if readData1[i] != v {
			t.Errorf("dataset1[%d]: got %d, want %d", i, readData1[i], v)
		}
	}

	ds2, err := f3.Root().OpenDataset("dataset2")
	if err != nil {
		t.Fatalf("OpenDataset dataset2 failed: %v", err)
	}
	var readData2 []float64
	if err := ds2.Read(&readData2); err != nil {
		t.Fatalf("Read dataset2 failed: %v", err)
	}
	for i, v := range data2 {
		if readData2[i] != v {
			t.Errorf("dataset2[%d]: got %f, want %f", i, readData2[i], v)
		}
	}
}

func TestOpenReadWriteAddGroup(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hdf5-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test_rw_group.h5")

	// Create initial file
	f1, err := Create(testFile)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}
	f1.Close()

	// Reopen and add groups and datasets
	f2, err := OpenReadWrite(testFile)
	if err != nil {
		t.Fatalf("OpenReadWrite failed: %v", err)
	}

	root := f2.Root()

	// Create a group
	grp, err := root.CreateGroup("mygroup")
	if err != nil {
		t.Fatalf("CreateGroup failed: %v", err)
	}

	// Add dataset to group
	data := []int32{10, 20, 30}
	_, err = grp.CreateDataset("data", data)
	if err != nil {
		t.Fatalf("CreateDataset in group failed: %v", err)
	}

	f2.Close()

	// Verify structure
	f3, err := Open(testFile)
	if err != nil {
		t.Fatalf("Open after modify failed: %v", err)
	}
	defer f3.Close()

	// Check group exists
	grp2, err := f3.Root().OpenGroup("mygroup")
	if err != nil {
		t.Fatalf("OpenGroup failed: %v", err)
	}

	// Check dataset in group
	ds, err := grp2.OpenDataset("data")
	if err != nil {
		t.Fatalf("OpenDataset in group failed: %v", err)
	}

	var readData []int32
	if err := ds.Read(&readData); err != nil {
		t.Fatalf("Read failed: %v", err)
	}

	for i, v := range data {
		if readData[i] != v {
			t.Errorf("data[%d]: got %d, want %d", i, readData[i], v)
		}
	}
}
