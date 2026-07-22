package hdf5

import (
	"os"
	"testing"
)

func TestGroupCreateAttribute(t *testing.T) {
	path := "test_group_attribute.h5"
	defer os.Remove(path)

	f, err := Create(path)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}
	defer f.Close()

	grp, err := f.Root().CreateGroup("my_group")
	if err != nil {
		t.Fatalf("CreateGroup failed: %v", err)
	}

	err = grp.CreateAttribute("description", "This is a test group")
	if err != nil {
		t.Fatalf("CreateAttribute (string) failed: %v", err)
	}

	err = grp.CreateAttribute("value", int64(42))
	if err != nil {
		t.Fatalf("CreateAttribute (int) failed: %v", err)
	}

	err = grp.CreateAttribute("scores", []float64{95.5, 88.0, 76.3})
	if err != nil {
		t.Fatalf("CreateAttribute (float array) failed: %v", err)
	}

	if err := f.Flush(); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	f2, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f2.Close()

	grp2, err := f2.OpenGroup("my_group")
	if err != nil {
		t.Fatalf("OpenGroup failed: %v", err)
	}

	attrNames := grp2.Attrs()
	if len(attrNames) != 3 {
		t.Errorf("Expected 3 attributes, got %d: %v", len(attrNames), attrNames)
	}

	descAttr := grp2.Attr("description")
	if descAttr == nil {
		t.Fatalf("Attribute 'description' not found")
	}
	desc, err := descAttr.ReadScalarString()
	if err != nil {
		t.Fatalf("ReadScalarString failed: %v", err)
	}
	if desc != "This is a test group" {
		t.Errorf("Expected description 'This is a test group', got '%s'", desc)
	}

	valAttr := grp2.Attr("value")
	if valAttr == nil {
		t.Fatalf("Attribute 'value' not found")
	}
	val, err := valAttr.ReadScalarInt64()
	if err != nil {
		t.Fatalf("ReadScalarInt64 failed: %v", err)
	}
	if val != 42 {
		t.Errorf("Expected value 42, got %d", val)
	}

	t.Logf("Successfully created and read group attributes")
}
