package hdf5

import (
	"fmt"
	"path/filepath"
	"testing"
)

func TestMaxNestedDepth(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "max_depth_test.h5")

	f, err := Create(filePath)
	if err != nil {
		t.Fatalf("Failed to create file: %v", err)
	}
	defer f.Close()

	maxDepth := 100
	currentGroup := f.Root()
	var pathParts []string

	for i := 0; i < maxDepth; i++ {
		name := fmt.Sprintf("level%d", i+1)
		pathParts = append(pathParts, name)

		newGroup, err := currentGroup.CreateGroup(name)
		if err != nil {
			t.Errorf("Failed to create group at depth %d: %v", i+1, err)
			break
		}
		currentGroup = newGroup
	}

	if err := f.Flush(); err != nil {
		t.Fatalf("Failed to flush: %v", err)
	}

	f2, err := Open(filePath)
	if err != nil {
		t.Fatalf("Failed to reopen file: %v", err)
	}
	defer f2.Close()

	fullPath := "/" + joinPathParts(pathParts)
	t.Logf("Testing path: %s", fullPath)

	// Try to open the deepest group
	openedGroup, err := f2.OpenGroup(fullPath)
	if err != nil {
		t.Errorf("Failed to open group at depth %d: %v", maxDepth, err)
	} else {
		t.Logf("Successfully opened group at depth %d", maxDepth)
		_ = openedGroup
	}

	// Create a dataset in the deepest group
	f3, err := OpenReadWrite(filePath)
	if err != nil {
		t.Fatalf("Failed to open file for read/write: %v", err)
	}
	defer f3.Close()

	deepestGroup, err := f3.OpenGroup(fullPath)
	if err != nil {
		t.Errorf("Failed to open deepest group for write: %v", err)
		return
	}

	data := []int{1, 2, 3}
	_, err = deepestGroup.CreateDataset("data", data)
	if err != nil {
		t.Errorf("Failed to create dataset at depth %d: %v", maxDepth, err)
		return
	}

	if err := f3.Flush(); err != nil {
		t.Fatalf("Failed to flush after adding dataset: %v", err)
	}

	// Verify the dataset can be opened
	f4, err := Open(filePath)
	if err != nil {
		t.Fatalf("Failed to reopen file after adding dataset: %v", err)
	}
	defer f4.Close()

	datasetPath := fullPath + "/data"
	_, err = f4.OpenDataset(datasetPath)
	if err != nil {
		t.Errorf("Failed to open dataset at depth %d: %v", maxDepth, err)
	} else {
		t.Logf("Successfully opened dataset at depth %d", maxDepth)
	}

	t.Logf("Maximum tested nested depth: %d", maxDepth)
}

func joinPathParts(parts []string) string {
	result := ""
	for i, part := range parts {
		if i == 0 {
			result = part
		} else {
			result = result + "/" + part
		}
	}
	return result
}
