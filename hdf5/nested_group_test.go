package hdf5

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNestedGroupDataset(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "nested_test.h5")

	// Create a new HDF5 file
	f, err := Create(filePath)
	if err != nil {
		t.Fatalf("Failed to create file: %v", err)
	}
	defer f.Close()

	// Create root -> group1 -> group2
	group1, err := f.Root().CreateGroup("group1")
	if err != nil {
		t.Fatalf("Failed to create group1: %v", err)
	}

	group2, err := group1.CreateGroup("group2")
	if err != nil {
		t.Fatalf("Failed to create group2: %v", err)
	}

	// Create dataset in nested group
	data := []float64{1.0, 2.0, 3.0}
	_, err = group2.CreateDataset("data", data)
	if err != nil {
		t.Fatalf("Failed to create dataset in group2: %v", err)
	}

	// Flush to write changes
	if err := f.Flush(); err != nil {
		t.Fatalf("Failed to flush: %v", err)
	}

	// Reopen the file to verify
	f2, err := Open(filePath)
	if err != nil {
		t.Fatalf("Failed to reopen file: %v", err)
	}
	defer f2.Close()

	// Try to open the nested group and dataset
	openedGroup1, err := f2.OpenGroup("/group1")
	if err != nil {
		t.Fatalf("Failed to open group1: %v", err)
	}

	openedGroup2, err := openedGroup1.OpenGroup("group2")
	if err != nil {
		t.Fatalf("Failed to open group2: %v", err)
	}

	// List members of group2
	members, err := openedGroup2.Members()
	if err != nil {
		t.Fatalf("Failed to get members of group2: %v", err)
	}

	if len(members) == 0 {
		t.Errorf("Expected group2 to have members, got empty list")
	}

	found := false
	for _, name := range members {
		if name == "data" {
			found = true
			break
		}
	}

	if !found {
		t.Errorf("Expected to find dataset 'data' in group2, got members: %v", members)
	}

	// Try to open the dataset directly
	_, err = f2.OpenDataset("/group1/group2/data")
	if err != nil {
		t.Errorf("Failed to open dataset /group1/group2/data: %v", err)
	}
}

func TestDeepNestedGroups(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "deep_nested_test.h5")

	f, err := Create(filePath)
	if err != nil {
		t.Fatalf("Failed to create file: %v", err)
	}
	defer f.Close()

	// Create deep nesting: root -> a -> b -> c
	a, err := f.Root().CreateGroup("a")
	if err != nil {
		t.Fatalf("Failed to create group a: %v", err)
	}

	b, err := a.CreateGroup("b")
	if err != nil {
		t.Fatalf("Failed to create group b: %v", err)
	}

	c, err := b.CreateGroup("c")
	if err != nil {
		t.Fatalf("Failed to create group c: %v", err)
	}

	// Create dataset in deepest group
	data := []int{1, 2, 3}
	_, err = c.CreateDataset("values", data)
	if err != nil {
		t.Fatalf("Failed to create dataset in group c: %v", err)
	}

	if err := f.Flush(); err != nil {
		t.Fatalf("Failed to flush: %v", err)
	}

	// Reopen and verify
	f2, err := Open(filePath)
	if err != nil {
		t.Fatalf("Failed to reopen file: %v", err)
	}
	defer f2.Close()

	// Open deep path directly
	_, err = f2.OpenDataset("/a/b/c/values")
	if err != nil {
		t.Errorf("Failed to open dataset /a/b/c/values: %v", err)
	}
}

func TestNestedGroupCreateThenWrite(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "nested_write_test.h5")

	f, err := Create(filePath)
	if err != nil {
		t.Fatalf("Failed to create file: %v", err)
	}
	defer f.Close()

	// Create groups
	g1, err := f.Root().CreateGroup("level1")
	if err != nil {
		t.Fatalf("Failed to create level1: %v", err)
	}

	g2, err := g1.CreateGroup("level2")
	if err != nil {
		t.Fatalf("Failed to create level2: %v", err)
	}

	// Create dataset directly with data
	writeData := make([]int64, 10)
	for i := range writeData {
		writeData[i] = int64(i)
	}
	_, err = g2.CreateDataset("data", writeData)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}

	if err := f.Flush(); err != nil {
		t.Fatalf("Failed to flush: %v", err)
	}

	// Verify after reopen
	f2, err := Open(filePath)
	if err != nil {
		t.Fatalf("Failed to reopen file: %v", err)
	}
	defer f2.Close()

	openedDs, err := f2.OpenDataset("/level1/level2/data")
	if err != nil {
		t.Errorf("Failed to open /level1/level2/data: %v", err)
	}

	readData, err := openedDs.ReadInt64()
	if err != nil {
		t.Errorf("Failed to read data: %v", err)
	}

	if len(readData) != 10 {
		t.Errorf("Expected 10 elements, got %d", len(readData))
	}
}

func TestMultipleDatasetsInNestedGroup(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "multi_dataset_test.h5")

	f, err := Create(filePath)
	if err != nil {
		t.Fatalf("Failed to create file: %v", err)
	}
	defer f.Close()

	// Create nested group
	ng, err := f.Root().CreateGroup("nested")
	if err != nil {
		t.Fatalf("Failed to create nested group: %v", err)
	}

	// Create multiple datasets
	_, err = ng.CreateDataset("ds1", []float64{1.1, 2.2, 3.3})
	if err != nil {
		t.Fatalf("Failed to create ds1: %v", err)
	}

	_, err = ng.CreateDataset("ds2", []int{10, 20, 30})
	if err != nil {
		t.Fatalf("Failed to create ds2: %v", err)
	}

	if err := f.Flush(); err != nil {
		t.Fatalf("Failed to flush: %v", err)
	}

	// Verify
	f2, err := Open(filePath)
	if err != nil {
		t.Fatalf("Failed to reopen file: %v", err)
	}
	defer f2.Close()

	openedNg, err := f2.OpenGroup("/nested")
	if err != nil {
		t.Fatalf("Failed to open nested group: %v", err)
	}

	members, err := openedNg.Members()
	if err != nil {
		t.Fatalf("Failed to get members: %v", err)
	}

	if len(members) != 2 {
		t.Errorf("Expected 2 datasets, got %d: %v", len(members), members)
	}
}

func TestCleanup(t *testing.T) {
	// Clean up test files
	testFiles := []string{
		"nested_test.h5",
		"deep_nested_test.h5",
		"nested_write_test.h5",
		"multi_dataset_test.h5",
	}

	for _, fname := range testFiles {
		if _, err := os.Stat(fname); err == nil {
			os.Remove(fname)
		}
	}
}
