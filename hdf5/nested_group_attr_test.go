package hdf5

import (
	"os"
	"testing"
)

func TestNestedGroupsWithAttributes(t *testing.T) {
	path := "test_nested_groups_attr.h5"
	defer os.Remove(path)

	f, err := Create(path)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}
	defer f.Close()

	level1, err := f.Root().CreateGroup("level1")
	if err != nil {
		t.Fatalf("CreateGroup level1 failed: %v", err)
	}
	err = level1.CreateAttribute("level", int64(1))
	if err != nil {
		t.Fatalf("CreateAttribute on level1 failed: %v", err)
	}
	err = level1.CreateAttribute("description", "First level group")
	if err != nil {
		t.Fatalf("CreateAttribute description on level1 failed: %v", err)
	}

	level2, err := level1.CreateGroup("level2")
	if err != nil {
		t.Fatalf("CreateGroup level2 failed: %v", err)
	}
	err = level2.CreateAttribute("level", int64(2))
	if err != nil {
		t.Fatalf("CreateAttribute on level2 failed: %v", err)
	}
	err = level2.CreateAttribute("data_count", int64(100))
	if err != nil {
		t.Fatalf("CreateAttribute data_count on level2 failed: %v", err)
	}

	level3, err := level2.CreateGroup("level3")
	if err != nil {
		t.Fatalf("CreateGroup level3 failed: %v", err)
	}
	err = level3.CreateAttribute("level", int64(3))
	if err != nil {
		t.Fatalf("CreateAttribute on level3 failed: %v", err)
	}
	err = level3.CreateAttribute("tags", []string{"important", "test", "nested"})
	if err != nil {
		t.Fatalf("CreateAttribute tags on level3 failed: %v", err)
	}

	_, err = level3.CreateDataset("data", []int{1, 2, 3, 4, 5}, nil)
	if err != nil {
		t.Fatalf("CreateDataset on level3 failed: %v", err)
	}

	if err := f.Flush(); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	f2, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f2.Close()

	level1_read, err := f2.OpenGroup("level1")
	if err != nil {
		t.Fatalf("OpenGroup level1 failed: %v", err)
	}
	if !level1_read.HasAttr("level") {
		t.Error("level1 should have 'level' attribute")
	}
	if !level1_read.HasAttr("description") {
		t.Error("level1 should have 'description' attribute")
	}

	level2_read, err := level1_read.OpenGroup("level2")
	if err != nil {
		t.Fatalf("OpenGroup level2 failed: %v", err)
	}
	if !level2_read.HasAttr("level") {
		t.Error("level2 should have 'level' attribute")
	}
	if !level2_read.HasAttr("data_count") {
		t.Error("level2 should have 'data_count' attribute")
	}

	level3_read, err := level2_read.OpenGroup("level3")
	if err != nil {
		t.Fatalf("OpenGroup level3 failed: %v", err)
	}
	if !level3_read.HasAttr("level") {
		t.Error("level3 should have 'level' attribute")
	}
	if !level3_read.HasAttr("tags") {
		t.Error("level3 should have 'tags' attribute")
	}

	_, err = level3_read.OpenDataset("data")
	if err != nil {
		t.Fatalf("OpenDataset data failed: %v", err)
	}

	t.Logf("Successfully created and verified nested groups with attributes")
}
