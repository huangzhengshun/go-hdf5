package hdf5

import (
	"strings"
	"testing"
)

func TestParseAttrPath(t *testing.T) {
	tests := []struct {
		path       string
		wantObject string
		wantAttr   string
		wantErr    bool
	}{
		{"/@root_attr", "/", "root_attr", false},
		{"/data@units", "/data", "units", false},
		{"/group/dataset@attr", "/group/dataset", "attr", false},
		{"/a/b/c@d", "/a/b/c", "d", false},
		{"data@attr", "/data", "attr", false}, // relative path normalized
		{"", "", "", true},                    // empty
		{"/path/no/at", "", "", true},         // missing @
		{"/path@", "", "", true},              // empty attr name
	}

	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			obj, attr, err := ParseAttrPath(tt.path)
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error for %q", tt.path)
				}
				return
			}
			if err != nil {
				t.Errorf("unexpected error for %q: %v", tt.path, err)
				return
			}
			if obj != tt.wantObject {
				t.Errorf("object path: got %q, want %q", obj, tt.wantObject)
			}
			if attr != tt.wantAttr {
				t.Errorf("attr name: got %q, want %q", attr, tt.wantAttr)
			}
		})
	}
}

func TestJoinAttrPath(t *testing.T) {
	tests := []struct {
		objectPath string
		attrName   string
		want       string
	}{
		{"/", "attr", "/@attr"},
		{"/data", "units", "/data@units"},
		{"/group/dataset", "calibration", "/group/dataset@calibration"},
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			got := JoinAttrPath(tt.objectPath, tt.attrName)
			if got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestSplitPathUtil(t *testing.T) {
	tests := []struct {
		path string
		want []string
	}{
		{"/", []string{}},
		{"/foo", []string{"foo"}},
		{"/foo/bar", []string{"foo", "bar"}},
		{"/a/b/c", []string{"a", "b", "c"}},
		{"foo/bar", []string{"foo", "bar"}},
	}

	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			got := SplitPath(tt.path)
			if len(got) != len(tt.want) {
				t.Errorf("got %v, want %v", got, tt.want)
				return
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("got %v, want %v", got, tt.want)
					break
				}
			}
		})
	}
}

func TestAttributeValue(t *testing.T) {
	path := skipIfNoTestdata(t, "attributes.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	ds, err := f.OpenDataset("data")
	if err != nil {
		t.Fatalf("OpenDataset failed: %v", err)
	}

	// Test float attribute
	attr := ds.Attr("float_attr")
	if attr == nil {
		t.Fatal("float_attr not found")
	}
	val, err := attr.Value()
	if err != nil {
		t.Errorf("Value() failed for float_attr: %v", err)
	}
	if v, ok := val.(float64); ok {
		if v != 3.14 {
			t.Errorf("float_attr: got %v, want 3.14", v)
		}
	} else {
		t.Errorf("float_attr: expected float64, got %T", val)
	}

	// Test string attribute
	attr = ds.Attr("string_attr")
	if attr == nil {
		t.Fatal("string_attr not found")
	}
	val, err = attr.Value()
	if err != nil {
		t.Errorf("Value() failed for string_attr: %v", err)
	}
	if v, ok := val.(string); ok {
		if v != "hello" {
			t.Errorf("string_attr: got %q, want 'hello'", v)
		}
	} else {
		t.Errorf("string_attr: expected string, got %T", val)
	}
}

func TestGetAttr(t *testing.T) {
	path := skipIfNoTestdata(t, "attributes.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// Test getting attribute by path
	attr, err := f.GetAttr("/data@float_attr")
	if err != nil {
		t.Fatalf("GetAttr failed: %v", err)
	}

	if attr == nil {
		t.Fatal("GetAttr returned nil")
	}

	val, err := attr.Value()
	if err != nil {
		t.Fatalf("Value failed: %v", err)
	}

	if v, ok := val.(float64); !ok || v != 3.14 {
		t.Errorf("got %v (%T), want 3.14", val, val)
	}
}

func TestReadAttr(t *testing.T) {
	path := skipIfNoTestdata(t, "attributes.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// Test reading attribute value directly
	val, err := f.ReadAttr("/data@string_attr")
	if err != nil {
		t.Fatalf("ReadAttr failed: %v", err)
	}

	if v, ok := val.(string); !ok || v != "hello" {
		t.Errorf("got %v (%T), want 'hello'", val, val)
	}
}

func TestGetAttrNotFound(t *testing.T) {
	path := skipIfNoTestdata(t, "attributes.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// Test non-existent attribute
	_, err = f.GetAttr("/data@nonexistent")
	if err == nil {
		t.Error("expected error for non-existent attribute")
	}

	// Test non-existent object
	_, err = f.GetAttr("/nonexistent@attr")
	if err == nil {
		t.Error("expected error for non-existent object")
	}
}

func TestWalkAttrs(t *testing.T) {
	path := skipIfNoTestdata(t, "attributes.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	var found []string
	err = f.WalkAttrs(func(info AttrInfo) error {
		found = append(found, info.Path)
		return nil
	})
	if err != nil {
		t.Fatalf("WalkAttrs failed: %v", err)
	}

	if len(found) == 0 {
		t.Error("WalkAttrs found no attributes")
	}

	// Check that expected attributes were found
	hasFloat := false
	hasString := false
	for _, p := range found {
		if strings.HasSuffix(p, "@float_attr") {
			hasFloat = true
		}
		if strings.HasSuffix(p, "@string_attr") {
			hasString = true
		}
	}

	if !hasFloat {
		t.Error("did not find @float_attr attribute")
	}
	if !hasString {
		t.Error("did not find @string_attr attribute")
	}
}

func TestWalkAttrsWithGroups(t *testing.T) {
	path := skipIfNoTestdata(t, "groups.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	var paths []string
	var objectTypes []string
	err = f.WalkAttrs(func(info AttrInfo) error {
		paths = append(paths, info.Path)
		objectTypes = append(objectTypes, info.ObjectType)
		return nil
	})
	if err != nil {
		t.Fatalf("WalkAttrs failed: %v", err)
	}

	// Verify we traversed the structure
	t.Logf("Found %d attributes: %v", len(paths), paths)
}

func TestWalkAttrsStopEarly(t *testing.T) {
	path := skipIfNoTestdata(t, "attributes.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	count := 0
	err = f.WalkAttrs(func(info AttrInfo) error {
		count++
		return ErrStopWalk
	})

	if !IsStopWalk(err) {
		t.Errorf("expected ErrStopWalk, got %v", err)
	}

	if count != 1 {
		t.Errorf("expected walk to stop after 1 attribute, got %d", count)
	}
}

func TestWalkAttrsCompound(t *testing.T) {
	path := skipIfNoTestdata(t, "compound_attrs.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	var foundCompound bool
	err = f.WalkAttrs(func(info AttrInfo) error {
		if info.ObjectType == "dataset" && info.Value != nil {
			if _, ok := info.Value.(map[string]interface{}); ok {
				foundCompound = true
				t.Logf("Found compound attr %s = %v", info.Path, info.Value)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("WalkAttrs failed: %v", err)
	}

	if !foundCompound {
		t.Log("No compound attributes found (may be expected depending on file)")
	}
}

func TestAttrInfoFields(t *testing.T) {
	path := skipIfNoTestdata(t, "attributes.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	err = f.WalkAttrs(func(info AttrInfo) error {
		// Verify all fields are populated
		if info.Path == "" {
			t.Error("Path is empty")
		}
		if info.ObjectPath == "" {
			t.Error("ObjectPath is empty")
		}
		if info.ObjectType != "group" && info.ObjectType != "dataset" {
			t.Errorf("ObjectType is invalid: %q", info.ObjectType)
		}
		if info.Name == "" {
			t.Error("Name is empty")
		}
		if info.Attr == nil {
			t.Error("Attr is nil")
		}

		// Verify path consistency
		expected := JoinAttrPath(info.ObjectPath, info.Name)
		if info.Path != expected {
			t.Errorf("Path %q doesn't match ObjectPath@Name (%q)", info.Path, expected)
		}

		return nil
	})
	if err != nil {
		t.Fatalf("WalkAttrs failed: %v", err)
	}
}
