package hdf5

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCreateGroup(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hdf5-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test_group.h5")

	// Create new HDF5 file
	f, err := Create(testFile)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	// Create a group
	grp, err := f.Root().CreateGroup("mygroup")
	if err != nil {
		t.Fatalf("CreateGroup failed: %v", err)
	}

	if grp.Name() != "mygroup" {
		t.Errorf("Expected group name 'mygroup', got '%s'", grp.Name())
	}

	if grp.Path() != "/mygroup" {
		t.Errorf("Expected path '/mygroup', got '%s'", grp.Path())
	}

	// Close the file
	if err := f.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// Reopen and verify
	f2, err := Open(testFile)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f2.Close()

	// Try to open the group
	grp2, err := f2.Root().OpenGroup("mygroup")
	if err != nil {
		t.Fatalf("OpenGroup failed: %v", err)
	}

	if grp2.Name() != "mygroup" {
		t.Errorf("Expected group name 'mygroup' after reopen, got '%s'", grp2.Name())
	}
}

func TestCreateNestedGroups(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hdf5-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test_nested.h5")

	// Create new HDF5 file
	f, err := Create(testFile)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	// Create nested groups
	grp1, err := f.Root().CreateGroup("level1")
	if err != nil {
		t.Fatalf("CreateGroup level1 failed: %v", err)
	}

	grp2, err := grp1.CreateGroup("level2")
	if err != nil {
		t.Fatalf("CreateGroup level2 failed: %v", err)
	}

	if grp2.Path() != "/level1/level2" {
		t.Errorf("Expected path '/level1/level2', got '%s'", grp2.Path())
	}

	f.Close()

	// Reopen and verify
	f2, err := Open(testFile)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f2.Close()

	// Navigate to nested group
	grp2Reopened, err := f2.Root().OpenGroup("level1/level2")
	if err != nil {
		t.Fatalf("OpenGroup level1/level2 failed: %v", err)
	}

	if grp2Reopened.Name() != "level2" {
		t.Errorf("Expected name 'level2', got '%s'", grp2Reopened.Name())
	}
}

func TestCreateMultipleGroups(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hdf5-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test_multi.h5")

	f, err := Create(testFile)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	// Create multiple groups at root level
	_, err = f.Root().CreateGroup("group1")
	if err != nil {
		t.Fatalf("CreateGroup group1 failed: %v", err)
	}

	_, err = f.Root().CreateGroup("group2")
	if err != nil {
		t.Fatalf("CreateGroup group2 failed: %v", err)
	}

	_, err = f.Root().CreateGroup("group3")
	if err != nil {
		t.Fatalf("CreateGroup group3 failed: %v", err)
	}

	f.Close()

	// Reopen and verify all groups exist
	f2, err := Open(testFile)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f2.Close()

	for _, name := range []string{"group1", "group2", "group3"} {
		_, err := f2.Root().OpenGroup(name)
		if err != nil {
			t.Errorf("OpenGroup %s failed: %v", name, err)
		}
	}
}
