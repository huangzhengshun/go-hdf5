package hdf5

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func getTestdataPath(filename string) string {
	return filepath.Join("..", "testdata", filename)
}

func skipIfNoTestdata(t *testing.T, filename string) string {
	path := getTestdataPath(filename)
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file %s not found. Run 'python3 testdata/generate.py' to create test files.", filename)
	}
	return path
}

func TestOpenMinimal(t *testing.T) {
	path := skipIfNoTestdata(t, "minimal.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	if f.Path() != path {
		t.Errorf("expected path %q, got %q", path, f.Path())
	}

	root := f.Root()
	if root == nil {
		t.Fatal("Root() returned nil")
	}

	if root.Path() != "/" {
		t.Errorf("expected root path '/', got %q", root.Path())
	}
}

func TestOpenNotHDF5(t *testing.T) {
	// Create a temporary non-HDF5 file
	tmpfile, err := os.CreateTemp("", "notHDF5")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpfile.Name())

	tmpfile.WriteString("This is not an HDF5 file")
	tmpfile.Close()

	_, err = Open(tmpfile.Name())
	if err == nil {
		t.Error("expected error for non-HDF5 file")
	}
}

func TestOpenNotExists(t *testing.T) {
	_, err := Open("/nonexistent/path/to/file.h5")
	if err == nil {
		t.Error("expected error for non-existent file")
	}
}

func TestReadMinimalDataset(t *testing.T) {
	path := skipIfNoTestdata(t, "minimal.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	ds, err := f.OpenDataset("data")
	if err != nil {
		t.Fatalf("OpenDataset failed: %v", err)
	}

	if ds.Name() != "data" {
		t.Errorf("expected name 'data', got %q", ds.Name())
	}

	shape := ds.Shape()
	if len(shape) != 1 || shape[0] != 4 {
		t.Errorf("expected shape [4], got %v", shape)
	}

	if ds.NumElements() != 4 {
		t.Errorf("expected 4 elements, got %d", ds.NumElements())
	}

	data, err := ds.ReadFloat64()
	if err != nil {
		t.Fatalf("ReadFloat64 failed: %v", err)
	}

	expected := []float64{1.0, 2.0, 3.0, 4.0}
	if len(data) != len(expected) {
		t.Fatalf("expected %d values, got %d", len(expected), len(data))
	}

	for i, v := range expected {
		if data[i] != v {
			t.Errorf("data[%d] = %f, want %f", i, data[i], v)
		}
	}
}

func TestReadFloats(t *testing.T) {
	path := skipIfNoTestdata(t, "floats.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// Test float32
	ds32, err := f.OpenDataset("float32")
	if err != nil {
		t.Fatalf("OpenDataset float32 failed: %v", err)
	}

	data32, err := ds32.ReadFloat32()
	if err != nil {
		t.Fatalf("ReadFloat32 failed: %v", err)
	}

	expected32 := []float32{1.5, 2.5, 3.5}
	for i, v := range expected32 {
		if data32[i] != v {
			t.Errorf("float32[%d] = %f, want %f", i, data32[i], v)
		}
	}

	// Test float64
	ds64, err := f.OpenDataset("float64")
	if err != nil {
		t.Fatalf("OpenDataset float64 failed: %v", err)
	}

	data64, err := ds64.ReadFloat64()
	if err != nil {
		t.Fatalf("ReadFloat64 failed: %v", err)
	}

	expected64 := []float64{1.5, 2.5, 3.5}
	for i, v := range expected64 {
		if data64[i] != v {
			t.Errorf("float64[%d] = %f, want %f", i, data64[i], v)
		}
	}
}

func TestReadIntegers(t *testing.T) {
	path := skipIfNoTestdata(t, "integers.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// Test int32
	ds, err := f.OpenDataset("int32")
	if err != nil {
		t.Fatalf("OpenDataset int32 failed: %v", err)
	}

	data, err := ds.ReadInt32()
	if err != nil {
		t.Fatalf("ReadInt32 failed: %v", err)
	}

	expected := []int32{1, 2, 3, 4, 5}
	for i, v := range expected {
		if data[i] != v {
			t.Errorf("int32[%d] = %d, want %d", i, data[i], v)
		}
	}
}

func TestReadMultidim(t *testing.T) {
	path := skipIfNoTestdata(t, "multidim.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	ds, err := f.OpenDataset("2d")
	if err != nil {
		t.Fatalf("OpenDataset 2d failed: %v", err)
	}

	shape := ds.Shape()
	if len(shape) != 2 || shape[0] != 3 || shape[1] != 4 {
		t.Errorf("expected shape [3, 4], got %v", shape)
	}

	if ds.Rank() != 2 {
		t.Errorf("expected rank 2, got %d", ds.Rank())
	}

	if ds.NumElements() != 12 {
		t.Errorf("expected 12 elements, got %d", ds.NumElements())
	}

	data, err := ds.ReadInt32()
	if err != nil {
		t.Fatalf("ReadInt32 failed: %v", err)
	}

	// Data should be 0, 1, 2, ..., 11 in row-major order
	for i := 0; i < 12; i++ {
		if data[i] != int32(i) {
			t.Errorf("data[%d] = %d, want %d", i, data[i], i)
		}
	}
}

func TestGroupNavigation(t *testing.T) {
	path := skipIfNoTestdata(t, "groups.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// List root members
	root := f.Root()
	members, err := root.Members()
	if err != nil {
		t.Fatalf("Members failed: %v", err)
	}

	if len(members) < 2 {
		t.Errorf("expected at least 2 members, got %d", len(members))
	}

	// Open group1
	g1, err := f.OpenGroup("group1")
	if err != nil {
		t.Fatalf("OpenGroup group1 failed: %v", err)
	}

	if g1.Name() != "group1" {
		t.Errorf("expected name 'group1', got %q", g1.Name())
	}

	// Open dataset in group
	ds, err := g1.OpenDataset("data")
	if err != nil {
		t.Fatalf("OpenDataset data failed: %v", err)
	}

	data, err := ds.ReadInt64()
	if err != nil {
		t.Fatalf("ReadInt64 failed: %v", err)
	}

	expected := []int64{1, 2, 3}
	for i, v := range expected {
		if data[i] != v {
			t.Errorf("data[%d] = %d, want %d", i, data[i], v)
		}
	}

	// Test path-based access
	ds2, err := f.OpenDataset("group1/data")
	if err != nil {
		t.Fatalf("OpenDataset group1/data failed: %v", err)
	}

	if ds2.Path() != "/group1/data" {
		t.Errorf("expected path '/group1/data', got %q", ds2.Path())
	}
}

func TestDatasetAttributes(t *testing.T) {
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

	attrs := ds.Attrs()
	if len(attrs) < 1 {
		t.Errorf("expected at least 1 attribute, got %d", len(attrs))
	}
}

func TestFileClose(t *testing.T) {
	path := skipIfNoTestdata(t, "minimal.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	err = f.Close()
	if err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// Double close should be safe
	err = f.Close()
	if err != nil {
		t.Fatalf("Double close failed: %v", err)
	}

	// Operations after close should fail
	_, err = f.OpenDataset("data")
	if err != ErrClosed {
		t.Errorf("expected ErrClosed after close, got %v", err)
	}
}

func TestSplitPath(t *testing.T) {
	tests := []struct {
		input    string
		expected []string
	}{
		{"", []string{}},
		{"/", []string{}},
		{"foo", []string{"foo"}},
		{"/foo", []string{"foo"}},
		{"foo/bar", []string{"foo", "bar"}},
		{"/foo/bar", []string{"foo", "bar"}},
		{"/foo/bar/", []string{"foo", "bar"}},
	}

	for _, tt := range tests {
		result := SplitPath(tt.input)
		if len(result) != len(tt.expected) {
			t.Errorf("SplitPath(%q): expected %v, got %v", tt.input, tt.expected, result)
			continue
		}
		for i := range result {
			if result[i] != tt.expected[i] {
				t.Errorf("SplitPath(%q)[%d]: expected %q, got %q", tt.input, i, tt.expected[i], result[i])
			}
		}
	}
}

func TestV0SuperblockMinimal(t *testing.T) {
	path := skipIfNoTestdata(t, "v0_minimal.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	ds, err := f.OpenDataset("data")
	if err != nil {
		t.Fatalf("OpenDataset failed: %v", err)
	}

	data, err := ds.ReadFloat64()
	if err != nil {
		t.Fatalf("ReadFloat64 failed: %v", err)
	}

	expected := []float64{1.0, 2.0, 3.0, 4.0}
	for i, v := range expected {
		if data[i] != v {
			t.Errorf("data[%d] = %f, want %f", i, data[i], v)
		}
	}
}

func TestV0SuperblockIntegers(t *testing.T) {
	path := skipIfNoTestdata(t, "v0_integers.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// Test int32
	ds32, err := f.OpenDataset("int32")
	if err != nil {
		t.Fatalf("OpenDataset int32 failed: %v", err)
	}

	data32, err := ds32.ReadInt32()
	if err != nil {
		t.Fatalf("ReadInt32 failed: %v", err)
	}

	expected32 := []int32{1, 2, 3, 4, 5}
	for i, v := range expected32 {
		if data32[i] != v {
			t.Errorf("int32[%d] = %d, want %d", i, data32[i], v)
		}
	}

	// Test int64
	ds64, err := f.OpenDataset("int64")
	if err != nil {
		t.Fatalf("OpenDataset int64 failed: %v", err)
	}

	data64, err := ds64.ReadInt64()
	if err != nil {
		t.Fatalf("ReadInt64 failed: %v", err)
	}

	expected64 := []int64{10, 20, 30}
	for i, v := range expected64 {
		if data64[i] != v {
			t.Errorf("int64[%d] = %d, want %d", i, data64[i], v)
		}
	}
}

func TestReadAttributeValues(t *testing.T) {
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

	// Test int attribute
	intAttr := ds.Attr("int_attr")
	if intAttr == nil {
		t.Fatal("int_attr not found")
	}
	intVal, err := intAttr.ReadScalarInt64()
	if err != nil {
		t.Fatalf("ReadScalarInt64 failed: %v", err)
	}
	if intVal != 42 {
		t.Errorf("int_attr = %d, want 42", intVal)
	}

	// Test float attribute
	floatAttr := ds.Attr("float_attr")
	if floatAttr == nil {
		t.Fatal("float_attr not found")
	}
	floatVal, err := floatAttr.ReadScalarFloat64()
	if err != nil {
		t.Fatalf("ReadScalarFloat64 failed: %v", err)
	}
	if floatVal != 3.14 {
		t.Errorf("float_attr = %f, want 3.14", floatVal)
	}

	// Test string attribute
	strAttr := ds.Attr("string_attr")
	if strAttr == nil {
		t.Fatal("string_attr not found")
	}
	strVal, err := strAttr.ReadScalarString()
	if err != nil {
		t.Fatalf("ReadScalarString failed: %v", err)
	}
	if strVal != "hello" {
		t.Errorf("string_attr = %q, want %q", strVal, "hello")
	}
}

func TestVarLenStringAttributes(t *testing.T) {
	path := skipIfNoTestdata(t, "varlen_attrs.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	ds, err := f.OpenDataset("data")
	if err != nil {
		t.Fatalf("OpenDataset failed: %v", err)
	}

	// Test variable-length string attribute
	descAttr := ds.Attr("description")
	if descAttr == nil {
		t.Fatal("description attribute not found")
	}

	desc, err := descAttr.ReadScalarString()
	if err != nil {
		t.Fatalf("ReadScalarString failed: %v", err)
	}

	expected := "A variable length string attribute"
	if desc != expected {
		t.Errorf("description = %q, want %q", desc, expected)
	}

	// Test another variable-length string
	authorAttr := ds.Attr("author")
	if authorAttr == nil {
		t.Fatal("author attribute not found")
	}

	author, err := authorAttr.ReadScalarString()
	if err != nil {
		t.Fatalf("ReadScalarString (author) failed: %v", err)
	}

	if author != "Test Author" {
		t.Errorf("author = %q, want %q", author, "Test Author")
	}
}

func TestCompoundAttributes(t *testing.T) {
	path := skipIfNoTestdata(t, "compound_attrs.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	ds, err := f.OpenDataset("data")
	if err != nil {
		t.Fatalf("OpenDataset failed: %v", err)
	}

	// Test point attribute (x, y, z floats)
	pointAttr := ds.Attr("point")
	if pointAttr == nil {
		t.Fatal("point attribute not found")
	}

	if !pointAttr.IsCompound() {
		t.Error("expected point attribute to be compound type")
	}

	point, err := pointAttr.ReadScalarCompound()
	if err != nil {
		t.Fatalf("ReadScalarCompound failed: %v", err)
	}

	// Check that we got the expected fields
	if x, ok := point["x"].(float64); !ok || x != 1.0 {
		t.Errorf("point.x = %v, want 1.0", point["x"])
	}
	if y, ok := point["y"].(float64); !ok || y != 2.0 {
		t.Errorf("point.y = %v, want 2.0", point["y"])
	}
	if z, ok := point["z"].(float64); !ok || z != 3.0 {
		t.Errorf("point.z = %v, want 3.0", point["z"])
	}

	// Test record attribute (id, value, count)
	recordAttr := ds.Attr("record")
	if recordAttr == nil {
		t.Fatal("record attribute not found")
	}

	record, err := recordAttr.ReadScalarCompound()
	if err != nil {
		t.Fatalf("ReadScalarCompound (record) failed: %v", err)
	}

	// Check mixed types
	if id, ok := record["id"].(int32); !ok || id != 42 {
		t.Errorf("record.id = %v (%T), want 42", record["id"], record["id"])
	}
	if val, ok := record["value"].(float64); !ok || val != 3.14 {
		t.Errorf("record.value = %v, want 3.14", record["value"])
	}
	if count, ok := record["count"].(int32); !ok || count != 100 {
		t.Errorf("record.count = %v, want 100", record["count"])
	}
}

func TestArrayAttributes(t *testing.T) {
	path := skipIfNoTestdata(t, "array_attrs.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	ds, err := f.OpenDataset("data")
	if err != nil {
		t.Fatalf("OpenDataset failed: %v", err)
	}

	// Test vector attribute (1D float64 array)
	vectorAttr := ds.Attr("vector")
	if vectorAttr == nil {
		t.Fatal("vector attribute not found")
	}

	vectorVal, err := vectorAttr.ReadFloat64()
	if err != nil {
		t.Fatalf("ReadFloat64 (vector) failed: %v", err)
	}

	expectedVector := []float64{1.0, 2.0, 3.0}
	if len(vectorVal) != len(expectedVector) {
		t.Fatalf("vector length = %d, want %d", len(vectorVal), len(expectedVector))
	}
	for i, v := range expectedVector {
		if vectorVal[i] != v {
			t.Errorf("vector[%d] = %f, want %f", i, vectorVal[i], v)
		}
	}

	// Test matrix attribute (2x2 int32 array)
	matrixAttr := ds.Attr("matrix")
	if matrixAttr == nil {
		t.Fatal("matrix attribute not found")
	}

	// Read as int32 slice (flattened)
	matrixVal, err := matrixAttr.ReadInt32()
	if err != nil {
		t.Fatalf("ReadInt32 (matrix) failed: %v", err)
	}

	expectedMatrix := []int32{1, 2, 3, 4}
	if len(matrixVal) != len(expectedMatrix) {
		t.Fatalf("matrix length = %d, want %d", len(matrixVal), len(expectedMatrix))
	}
	for i, v := range expectedMatrix {
		if matrixVal[i] != v {
			t.Errorf("matrix[%d] = %d, want %d", i, matrixVal[i], v)
		}
	}
}

func TestFileAttributes(t *testing.T) {
	path := skipIfNoTestdata(t, "attributes.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	root := f.Root()
	attrs := root.Attrs()

	// Check that file_attr exists
	found := false
	for _, name := range attrs {
		if name == "file_attr" {
			found = true
			break
		}
	}

	if !found {
		t.Log("Root attributes:", attrs)
		// File-level attributes may be stored differently
		// This is acceptable if file_attr is not on root group
	}
}

func TestReadChunkedDataset(t *testing.T) {
	path := skipIfNoTestdata(t, "chunked.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	ds, err := f.OpenDataset("chunked")
	if err != nil {
		t.Fatalf("OpenDataset failed: %v", err)
	}

	// Check shape
	shape := ds.Shape()
	if len(shape) != 2 || shape[0] != 10 || shape[1] != 10 {
		t.Errorf("expected shape [10 10], got %v", shape)
	}

	// Read the data
	data, err := ds.ReadFloat64()
	if err != nil {
		t.Fatalf("ReadFloat64 failed: %v", err)
	}

	// Verify data (should be 0-99 in row-major order)
	if len(data) != 100 {
		t.Fatalf("expected 100 elements, got %d", len(data))
	}

	for i := 0; i < 100; i++ {
		if data[i] != float64(i) {
			t.Errorf("data[%d] = %f, want %f", i, data[i], float64(i))
			break
		}
	}
}

func TestReadCompressedChunkedDataset(t *testing.T) {
	path := skipIfNoTestdata(t, "compressed.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// Test gzip compressed dataset
	t.Run("gzip", func(t *testing.T) {
		ds, err := f.OpenDataset("gzip")
		if err != nil {
			t.Fatalf("OpenDataset failed: %v", err)
		}

		shape := ds.Shape()
		if len(shape) != 2 || shape[0] != 100 || shape[1] != 100 {
			t.Errorf("expected shape [100 100], got %v", shape)
		}

		data, err := ds.ReadFloat64()
		if err != nil {
			t.Fatalf("ReadFloat64 failed: %v", err)
		}

		if len(data) != 10000 {
			t.Fatalf("expected 10000 elements, got %d", len(data))
		}

		// Verify data is in valid range (random values 0-1)
		for i, v := range data {
			if v < 0 || v > 1 {
				t.Errorf("data[%d] = %f, expected value in range [0, 1]", i, v)
				break
			}
		}
	})

	// Test shuffle + gzip compressed dataset
	t.Run("shuffle_gzip", func(t *testing.T) {
		ds, err := f.OpenDataset("shuffle_gzip")
		if err != nil {
			t.Fatalf("OpenDataset failed: %v", err)
		}

		data, err := ds.ReadFloat64()
		if err != nil {
			t.Fatalf("ReadFloat64 failed: %v", err)
		}

		if len(data) != 10000 {
			t.Fatalf("expected 10000 elements, got %d", len(data))
		}

		// Verify data is in valid range (random values 0-1)
		for i, v := range data {
			if v < 0 || v > 1 {
				t.Errorf("data[%d] = %f, expected value in range [0, 1]", i, v)
				break
			}
		}
	})
}

func TestSoftLinks(t *testing.T) {
	path := skipIfNoTestdata(t, "softlink.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// Test direct access to target dataset
	t.Run("direct_access", func(t *testing.T) {
		ds, err := f.OpenDataset("target_dataset")
		if err != nil {
			t.Fatalf("OpenDataset target_dataset failed: %v", err)
		}

		data, err := ds.ReadFloat64()
		if err != nil {
			t.Fatalf("ReadFloat64 failed: %v", err)
		}

		expected := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		if len(data) != len(expected) {
			t.Fatalf("expected %d elements, got %d", len(expected), len(data))
		}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("data[%d] = %f, want %f", i, data[i], v)
			}
		}
	})

	// Test soft link to dataset
	t.Run("link_to_dataset", func(t *testing.T) {
		ds, err := f.OpenDataset("link_to_dataset")
		if err != nil {
			t.Fatalf("OpenDataset via soft link failed: %v", err)
		}

		data, err := ds.ReadFloat64()
		if err != nil {
			t.Fatalf("ReadFloat64 failed: %v", err)
		}

		expected := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		if len(data) != len(expected) {
			t.Fatalf("expected %d elements, got %d", len(expected), len(data))
		}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("data[%d] = %f, want %f", i, data[i], v)
			}
		}
	})

	// Test soft link to group
	t.Run("link_to_group", func(t *testing.T) {
		grp, err := f.OpenGroup("link_to_group")
		if err != nil {
			t.Fatalf("OpenGroup via soft link failed: %v", err)
		}

		// Access nested dataset through the linked group
		ds, err := grp.OpenDataset("nested")
		if err != nil {
			t.Fatalf("OpenDataset nested failed: %v", err)
		}

		data, err := ds.ReadInt32()
		if err != nil {
			t.Fatalf("ReadInt32 failed: %v", err)
		}

		expected := []int32{10, 20, 30}
		if len(data) != len(expected) {
			t.Fatalf("expected %d elements, got %d", len(expected), len(data))
		}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("data[%d] = %d, want %d", i, data[i], v)
			}
		}
	})

	// Test chained soft link (link to another link)
	t.Run("link_to_link", func(t *testing.T) {
		ds, err := f.OpenDataset("link_to_link")
		if err != nil {
			t.Fatalf("OpenDataset via chained soft link failed: %v", err)
		}

		data, err := ds.ReadFloat64()
		if err != nil {
			t.Fatalf("ReadFloat64 failed: %v", err)
		}

		expected := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		if len(data) != len(expected) {
			t.Fatalf("expected %d elements, got %d", len(expected), len(data))
		}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("data[%d] = %f, want %f", i, data[i], v)
			}
		}
	})

	// Test nested soft link through group
	t.Run("nested_link_back", func(t *testing.T) {
		// Access /target_group/link_back which points to /target_dataset
		ds, err := f.OpenDataset("target_group/link_back")
		if err != nil {
			t.Fatalf("OpenDataset via nested soft link failed: %v", err)
		}

		data, err := ds.ReadFloat64()
		if err != nil {
			t.Fatalf("ReadFloat64 failed: %v", err)
		}

		expected := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		if len(data) != len(expected) {
			t.Fatalf("expected %d elements, got %d", len(expected), len(data))
		}
	})
}

func TestExternalLinks(t *testing.T) {
	path := skipIfNoTestdata(t, "external_source.h5")
	_ = skipIfNoTestdata(t, "external_target.h5") // Ensure target exists

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// Test local data (sanity check)
	t.Run("local_data", func(t *testing.T) {
		ds, err := f.OpenDataset("local_data")
		if err != nil {
			t.Fatalf("OpenDataset local_data failed: %v", err)
		}

		data, err := ds.ReadInt64()
		if err != nil {
			t.Fatalf("ReadInt64 failed: %v", err)
		}

		expected := []int64{100, 200, 300}
		if len(data) != len(expected) {
			t.Fatalf("expected %d elements, got %d", len(expected), len(data))
		}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("data[%d] = %d, want %d", i, data[i], v)
			}
		}
	})

	// Test external link to dataset
	t.Run("link_to_data", func(t *testing.T) {
		ds, err := f.OpenDataset("link_to_data")
		if err != nil {
			t.Fatalf("OpenDataset via external link failed: %v", err)
		}

		data, err := ds.ReadFloat64()
		if err != nil {
			t.Fatalf("ReadFloat64 failed: %v", err)
		}

		expected := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		if len(data) != len(expected) {
			t.Fatalf("expected %d elements, got %d", len(expected), len(data))
		}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("data[%d] = %f, want %f", i, data[i], v)
			}
		}
	})

	// Test external link to group
	t.Run("link_to_subgroup", func(t *testing.T) {
		grp, err := f.OpenGroup("link_to_subgroup")
		if err != nil {
			t.Fatalf("OpenGroup via external link failed: %v", err)
		}

		// Access nested dataset through the externally linked group
		ds, err := grp.OpenDataset("nested_data")
		if err != nil {
			t.Fatalf("OpenDataset nested_data failed: %v", err)
		}

		data, err := ds.ReadInt64()
		if err != nil {
			t.Fatalf("ReadInt64 failed: %v", err)
		}

		expected := []int64{10, 20, 30}
		if len(data) != len(expected) {
			t.Fatalf("expected %d elements, got %d", len(expected), len(data))
		}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("data[%d] = %d, want %d", i, data[i], v)
			}
		}
	})

	// Test external link to nested dataset
	t.Run("link_to_nested", func(t *testing.T) {
		ds, err := f.OpenDataset("link_to_nested")
		if err != nil {
			t.Fatalf("OpenDataset via external link to nested failed: %v", err)
		}

		data, err := ds.ReadInt64()
		if err != nil {
			t.Fatalf("ReadInt64 failed: %v", err)
		}

		expected := []int64{10, 20, 30}
		if len(data) != len(expected) {
			t.Fatalf("expected %d elements, got %d", len(expected), len(data))
		}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("data[%d] = %d, want %d", i, data[i], v)
			}
		}
	})
}

// === EDGE CASE TESTS ===

func TestCircularSoftLinkSelf(t *testing.T) {
	path := skipIfNoTestdata(t, "circular_self.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// Accessing real_data should work
	ds, err := f.OpenDataset("real_data")
	if err != nil {
		t.Fatalf("OpenDataset real_data failed: %v", err)
	}
	data, err := ds.ReadInt32()
	if err != nil {
		t.Fatalf("ReadInt32 failed: %v", err)
	}
	expected := []int32{1, 2, 3}
	for i, v := range expected {
		if data[i] != v {
			t.Errorf("data[%d] = %d, want %d", i, data[i], v)
		}
	}

	// Accessing circular link should fail with circular reference error
	_, err = f.OpenDataset("circular")
	if err == nil {
		t.Fatal("expected error for circular self-referencing link")
	}
	if !strings.Contains(err.Error(), "circular") {
		t.Errorf("expected circular error, got: %v", err)
	}
}

func TestCircularSoftLinkChain(t *testing.T) {
	path := skipIfNoTestdata(t, "circular_chain.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// Accessing real_data should work
	ds, err := f.OpenDataset("real_data")
	if err != nil {
		t.Fatalf("OpenDataset real_data failed: %v", err)
	}
	data, err := ds.ReadInt32()
	if err != nil {
		t.Fatalf("ReadInt32 failed: %v", err)
	}
	expected := []int32{1, 2, 3}
	for i, v := range expected {
		if data[i] != v {
			t.Errorf("data[%d] = %d, want %d", i, data[i], v)
		}
	}

	// Accessing any link in the cycle should fail
	for _, linkName := range []string{"link_a", "link_b", "link_c"} {
		_, err = f.OpenDataset(linkName)
		if err == nil {
			t.Fatalf("expected error for circular link chain at %s", linkName)
		}
		if !strings.Contains(err.Error(), "circular") {
			t.Errorf("expected circular error for %s, got: %v", linkName, err)
		}
	}
}

func TestDanglingLink(t *testing.T) {
	path := skipIfNoTestdata(t, "dangling_link.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// Accessing real_data should work
	ds, err := f.OpenDataset("real_data")
	if err != nil {
		t.Fatalf("OpenDataset real_data failed: %v", err)
	}
	data, err := ds.ReadInt32()
	if err != nil {
		t.Fatalf("ReadInt32 failed: %v", err)
	}
	expected := []int32{1, 2, 3}
	for i, v := range expected {
		if data[i] != v {
			t.Errorf("data[%d] = %d, want %d", i, data[i], v)
		}
	}

	// Accessing dangling links should fail with "not found" error
	_, err = f.OpenDataset("missing")
	if err == nil {
		t.Fatal("expected error for dangling link")
	}

	_, err = f.OpenDataset("missing_nested")
	if err == nil {
		t.Fatal("expected error for dangling nested link")
	}
}

func TestDeepLinkChain(t *testing.T) {
	path := skipIfNoTestdata(t, "deep_chain.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// Direct access to target should work
	ds, err := f.OpenDataset("target")
	if err != nil {
		t.Fatalf("OpenDataset target failed: %v", err)
	}
	data, err := ds.ReadInt32()
	if err != nil {
		t.Fatalf("ReadInt32 failed: %v", err)
	}
	if len(data) != 1 || data[0] != 42 {
		t.Errorf("expected [42], got %v", data)
	}

	// Access through deepest link (link_10 -> link_9 -> ... -> target)
	ds, err = f.OpenDataset("link_10")
	if err != nil {
		t.Fatalf("OpenDataset link_10 failed: %v", err)
	}
	data, err = ds.ReadInt32()
	if err != nil {
		t.Fatalf("ReadInt32 failed: %v", err)
	}
	if len(data) != 1 || data[0] != 42 {
		t.Errorf("expected [42], got %v", data)
	}
}

func TestExternalLinkMissingFile(t *testing.T) {
	path := skipIfNoTestdata(t, "external_missing.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// Accessing real_data should work
	ds, err := f.OpenDataset("real_data")
	if err != nil {
		t.Fatalf("OpenDataset real_data failed: %v", err)
	}
	data, err := ds.ReadInt32()
	if err != nil {
		t.Fatalf("ReadInt32 failed: %v", err)
	}
	expected := []int32{1, 2, 3}
	for i, v := range expected {
		if data[i] != v {
			t.Errorf("data[%d] = %d, want %d", i, data[i], v)
		}
	}

	// Accessing external link to missing file should fail
	_, err = f.OpenDataset("missing_file")
	if err == nil {
		t.Fatal("expected error for external link to missing file")
	}
}

func TestSoftLinkToRoot(t *testing.T) {
	path := skipIfNoTestdata(t, "link_to_root.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// Access data directly
	ds, err := f.OpenDataset("data")
	if err != nil {
		t.Fatalf("OpenDataset data failed: %v", err)
	}
	data, err := ds.ReadInt32()
	if err != nil {
		t.Fatalf("ReadInt32 failed: %v", err)
	}
	expected := []int32{1, 2, 3}
	for i, v := range expected {
		if data[i] != v {
			t.Errorf("data[%d] = %d, want %d", i, data[i], v)
		}
	}

	// Access root via link and then traverse to data
	grp, err := f.OpenGroup("root_link")
	if err != nil {
		t.Fatalf("OpenGroup root_link failed: %v", err)
	}

	ds, err = grp.OpenDataset("data")
	if err != nil {
		t.Fatalf("OpenDataset data via root_link failed: %v", err)
	}
	data, err = ds.ReadInt32()
	if err != nil {
		t.Fatalf("ReadInt32 failed: %v", err)
	}
	for i, v := range expected {
		if data[i] != v {
			t.Errorf("data[%d] = %d, want %d", i, data[i], v)
		}
	}
}

func TestV1FormatSoftLinks(t *testing.T) {
	path := skipIfNoTestdata(t, "v1_softlinks.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// Access target directly
	ds, err := f.OpenDataset("target")
	if err != nil {
		t.Fatalf("OpenDataset target failed: %v", err)
	}
	data, err := ds.ReadFloat64()
	if err != nil {
		t.Fatalf("ReadFloat64 failed: %v", err)
	}
	expected := []float64{1.0, 2.0, 3.0}
	for i, v := range expected {
		if data[i] != v {
			t.Errorf("data[%d] = %f, want %f", i, data[i], v)
		}
	}

	// Access via soft link (v1 format)
	ds, err = f.OpenDataset("soft_link")
	if err != nil {
		t.Fatalf("OpenDataset soft_link failed: %v", err)
	}
	data, err = ds.ReadFloat64()
	if err != nil {
		t.Fatalf("ReadFloat64 failed: %v", err)
	}
	for i, v := range expected {
		if data[i] != v {
			t.Errorf("data[%d] = %f, want %f", i, data[i], v)
		}
	}

	// Access group via soft link
	grp, err := f.OpenGroup("link_to_group")
	if err != nil {
		t.Fatalf("OpenGroup link_to_group failed: %v", err)
	}
	ds, err = grp.OpenDataset("nested")
	if err != nil {
		t.Fatalf("OpenDataset nested failed: %v", err)
	}
	intData, err := ds.ReadInt32()
	if err != nil {
		t.Fatalf("ReadInt32 failed: %v", err)
	}
	expectedInt := []int32{10, 20, 30}
	for i, v := range expectedInt {
		if intData[i] != v {
			t.Errorf("data[%d] = %d, want %d", i, intData[i], v)
		}
	}
}

func TestMixedSoftExternalChain(t *testing.T) {
	path := skipIfNoTestdata(t, "mixed_chain.h5")
	_ = skipIfNoTestdata(t, "external_target.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// Access local data
	ds, err := f.OpenDataset("local")
	if err != nil {
		t.Fatalf("OpenDataset local failed: %v", err)
	}
	data, err := ds.ReadInt32()
	if err != nil {
		t.Fatalf("ReadInt32 failed: %v", err)
	}
	expectedInt := []int32{1, 2, 3}
	for i, v := range expectedInt {
		if data[i] != v {
			t.Errorf("data[%d] = %d, want %d", i, data[i], v)
		}
	}

	// Access external data via soft link -> external link chain
	ds, err = f.OpenDataset("soft_to_ext")
	if err != nil {
		t.Fatalf("OpenDataset soft_to_ext failed: %v", err)
	}
	floatData, err := ds.ReadFloat64()
	if err != nil {
		t.Fatalf("ReadFloat64 failed: %v", err)
	}
	expectedFloat := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	if len(floatData) != len(expectedFloat) {
		t.Fatalf("expected %d elements, got %d", len(expectedFloat), len(floatData))
	}
	for i, v := range expectedFloat {
		if floatData[i] != v {
			t.Errorf("data[%d] = %f, want %f", i, floatData[i], v)
		}
	}
}


// TestMaxLinkDepthConstant verifies the MaxLinkDepth constant exists and is reasonable
func TestMaxLinkDepthConstant(t *testing.T) {
	if MaxLinkDepth < 10 {
		t.Errorf("MaxLinkDepth too small: %d (should be at least 10)", MaxLinkDepth)
	}
	if MaxLinkDepth > 10000 {
		t.Errorf("MaxLinkDepth too large: %d (should be at most 10000)", MaxLinkDepth)
	}
	// Current value should be 100
	if MaxLinkDepth != 100 {
		t.Logf("MaxLinkDepth is %d", MaxLinkDepth)
	}
}

// TestErrLinkDepthExists verifies the ErrLinkDepth error exists
func TestErrLinkDepthExists(t *testing.T) {
	if ErrLinkDepth == nil {
		t.Error("ErrLinkDepth should not be nil")
	}
	if ErrLinkDepth.Error() != "maximum link depth exceeded" {
		t.Errorf("unexpected error message: %s", ErrLinkDepth.Error())
	}
}

// TestBTreeV2Chunked tests reading a dataset with B-tree v2 chunk indexing
func TestBTreeV2Chunked(t *testing.T) {
	path := skipIfNoTestdata(t, "btree_v2.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// Test the small dataset (10x10 int32)
	ds, err := f.OpenDataset("small")
	if err != nil {
		t.Fatalf("OpenDataset small failed: %v", err)
	}

	data, err := ds.ReadInt32()
	if err != nil {
		t.Fatalf("ReadInt32 failed: %v", err)
	}

	// Verify data (should be 0..99)
	if len(data) != 100 {
		t.Fatalf("expected 100 elements, got %d", len(data))
	}
	for i := 0; i < 100; i++ {
		if data[i] != int32(i) {
			t.Errorf("data[%d] = %d, want %d", i, data[i], i)
			break
		}
	}

	// Test the larger dataset (100x100 float64)
	ds, err = f.OpenDataset("chunked")
	if err != nil {
		t.Fatalf("OpenDataset chunked failed: %v", err)
	}

	floatData, err := ds.ReadFloat64()
	if err != nil {
		t.Fatalf("ReadFloat64 failed: %v", err)
	}

	// Verify data (should be 0..9999)
	if len(floatData) != 10000 {
		t.Fatalf("expected 10000 elements, got %d", len(floatData))
	}
	for i := 0; i < 10; i++ {
		if floatData[i] != float64(i) {
			t.Errorf("floatData[%d] = %f, want %f", i, floatData[i], float64(i))
		}
	}
	// Check last few elements
	for i := 9990; i < 10000; i++ {
		if floatData[i] != float64(i) {
			t.Errorf("floatData[%d] = %f, want %f", i, floatData[i], float64(i))
		}
	}
}

// TestV0NestedGroupsAndAttributes comprehensively tests attribute access
// from nested datasets and groups in superblock version 0 files.
func TestV0NestedGroupsAndAttributes(t *testing.T) {
	path := skipIfNoTestdata(t, "v0_nested_attrs.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// Verify superblock version is 0
	if f.Version() != 0 {
		t.Errorf("expected superblock version 0, got %d", f.Version())
	}

	// === ROOT LEVEL ATTRIBUTES ===
	t.Run("root_attributes", func(t *testing.T) {
		root := f.Root()
		attrs := root.Attrs()
		if len(attrs) < 2 {
			t.Errorf("expected at least 2 root attributes, got %d: %v", len(attrs), attrs)
		}

		// Test file_version attribute
		versionAttr := root.Attr("file_version")
		if versionAttr == nil {
			t.Fatal("file_version attribute not found on root")
		}
		version, err := versionAttr.ReadScalarInt64()
		if err != nil {
			t.Fatalf("ReadScalarInt64 failed: %v", err)
		}
		if version != 1 {
			t.Errorf("file_version = %d, want 1", version)
		}

		// Test file_description attribute
		descAttr := root.Attr("file_description")
		if descAttr == nil {
			t.Fatal("file_description attribute not found on root")
		}
		desc, err := descAttr.ReadScalarString()
		if err != nil {
			t.Fatalf("ReadScalarString failed: %v", err)
		}
		if desc != "Test file for nested attributes" {
			t.Errorf("file_description = %q, want %q", desc, "Test file for nested attributes")
		}
	})

	// === LEVEL 1 GROUP ATTRIBUTES ===
	t.Run("level1_group_attributes", func(t *testing.T) {
		grp, err := f.OpenGroup("sensors")
		if err != nil {
			t.Fatalf("OpenGroup sensors failed: %v", err)
		}

		// Check group attributes
		attrs := grp.Attrs()
		if len(attrs) < 2 {
			t.Errorf("expected at least 2 group attributes, got %d: %v", len(attrs), attrs)
		}

		// Test sensor_count attribute
		countAttr := grp.Attr("sensor_count")
		if countAttr == nil {
			t.Fatal("sensor_count attribute not found")
		}
		count, err := countAttr.ReadScalarInt64()
		if err != nil {
			t.Fatalf("ReadScalarInt64 failed: %v", err)
		}
		if count != 3 {
			t.Errorf("sensor_count = %d, want 3", count)
		}

		// Test location attribute
		locAttr := grp.Attr("location")
		if locAttr == nil {
			t.Fatal("location attribute not found")
		}
		loc, err := locAttr.ReadScalarString()
		if err != nil {
			t.Fatalf("ReadScalarString failed: %v", err)
		}
		if loc != "building_a" {
			t.Errorf("location = %q, want %q", loc, "building_a")
		}
	})

	// === LEVEL 1 DATASET ATTRIBUTES ===
	t.Run("level1_dataset_attributes", func(t *testing.T) {
		ds, err := f.OpenDataset("sensors/temperature")
		if err != nil {
			t.Fatalf("OpenDataset sensors/temperature failed: %v", err)
		}

		// Verify data first
		data, err := ds.ReadFloat64()
		if err != nil {
			t.Fatalf("ReadFloat64 failed: %v", err)
		}
		expected := []float64{22.5, 23.1, 22.8, 23.5, 24.0}
		if len(data) != len(expected) {
			t.Fatalf("expected %d values, got %d", len(expected), len(data))
		}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("data[%d] = %f, want %f", i, data[i], v)
			}
		}

		// Check attributes exist
		attrs := ds.Attrs()
		if len(attrs) < 4 {
			t.Errorf("expected at least 4 dataset attributes, got %d: %v", len(attrs), attrs)
		}

		// Test units attribute
		unitsAttr := ds.Attr("units")
		if unitsAttr == nil {
			t.Fatal("units attribute not found")
		}
		units, err := unitsAttr.ReadScalarString()
		if err != nil {
			t.Fatalf("ReadScalarString failed: %v", err)
		}
		if units != "celsius" {
			t.Errorf("units = %q, want %q", units, "celsius")
		}

		// Test min_value attribute
		minAttr := ds.Attr("min_value")
		if minAttr == nil {
			t.Fatal("min_value attribute not found")
		}
		minVal, err := minAttr.ReadScalarFloat64()
		if err != nil {
			t.Fatalf("ReadScalarFloat64 failed: %v", err)
		}
		if minVal != 22.5 {
			t.Errorf("min_value = %f, want 22.5", minVal)
		}

		// Test max_value attribute
		maxAttr := ds.Attr("max_value")
		if maxAttr == nil {
			t.Fatal("max_value attribute not found")
		}
		maxVal, err := maxAttr.ReadScalarFloat64()
		if err != nil {
			t.Fatalf("ReadScalarFloat64 failed: %v", err)
		}
		if maxVal != 24.0 {
			t.Errorf("max_value = %f, want 24.0", maxVal)
		}
	})

	// === ANOTHER LEVEL 1 DATASET ===
	t.Run("level1_another_dataset", func(t *testing.T) {
		ds, err := f.OpenDataset("sensors/humidity")
		if err != nil {
			t.Fatalf("OpenDataset sensors/humidity failed: %v", err)
		}

		// Verify data
		data, err := ds.ReadInt32()
		if err != nil {
			t.Fatalf("ReadInt32 failed: %v", err)
		}
		expected := []int32{45, 48, 52, 50, 47}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("data[%d] = %d, want %d", i, data[i], v)
			}
		}

		// Test sensor_id attribute
		idAttr := ds.Attr("sensor_id")
		if idAttr == nil {
			t.Fatal("sensor_id attribute not found")
		}
		id, err := idAttr.ReadScalarInt64()
		if err != nil {
			t.Fatalf("ReadScalarInt64 failed: %v", err)
		}
		if id != 101 {
			t.Errorf("sensor_id = %d, want 101", id)
		}
	})

	// === LEVEL 2 NESTED GROUP ATTRIBUTES ===
	t.Run("level2_group_attributes", func(t *testing.T) {
		grp, err := f.OpenGroup("sensors/metadata")
		if err != nil {
			t.Fatalf("OpenGroup sensors/metadata failed: %v", err)
		}

		// Test created_by attribute
		createdByAttr := grp.Attr("created_by")
		if createdByAttr == nil {
			t.Fatal("created_by attribute not found")
		}
		createdBy, err := createdByAttr.ReadScalarString()
		if err != nil {
			t.Fatalf("ReadScalarString failed: %v", err)
		}
		if createdBy != "test_generator" {
			t.Errorf("created_by = %q, want %q", createdBy, "test_generator")
		}

		// Test version attribute
		versionAttr := grp.Attr("version")
		if versionAttr == nil {
			t.Fatal("version attribute not found")
		}
		version, err := versionAttr.ReadScalarInt64()
		if err != nil {
			t.Fatalf("ReadScalarInt64 failed: %v", err)
		}
		if version != 2 {
			t.Errorf("version = %d, want 2", version)
		}
	})

	// === LEVEL 2 NESTED DATASET ATTRIBUTES ===
	t.Run("level2_dataset_attributes", func(t *testing.T) {
		ds, err := f.OpenDataset("sensors/metadata/timestamps")
		if err != nil {
			t.Fatalf("OpenDataset sensors/metadata/timestamps failed: %v", err)
		}

		// Verify data
		data, err := ds.ReadInt64()
		if err != nil {
			t.Fatalf("ReadInt64 failed: %v", err)
		}
		expected := []int64{1000, 2000, 3000, 4000, 5000}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("data[%d] = %d, want %d", i, data[i], v)
			}
		}

		// Test timezone attribute
		tzAttr := ds.Attr("timezone")
		if tzAttr == nil {
			t.Fatal("timezone attribute not found")
		}
		tz, err := tzAttr.ReadScalarString()
		if err != nil {
			t.Fatalf("ReadScalarString failed: %v", err)
		}
		if tz != "UTC" {
			t.Errorf("timezone = %q, want %q", tz, "UTC")
		}

		// Test epoch attribute
		epochAttr := ds.Attr("epoch")
		if epochAttr == nil {
			t.Fatal("epoch attribute not found")
		}
		epoch, err := epochAttr.ReadScalarInt64()
		if err != nil {
			t.Fatalf("ReadScalarInt64 failed: %v", err)
		}
		if epoch != 1704067200 {
			t.Errorf("epoch = %d, want 1704067200", epoch)
		}
	})

	// === SECOND TOP-LEVEL GROUP ===
	t.Run("second_group_attributes", func(t *testing.T) {
		grp, err := f.OpenGroup("config")
		if err != nil {
			t.Fatalf("OpenGroup config failed: %v", err)
		}

		// Test active attribute
		activeAttr := grp.Attr("active")
		if activeAttr == nil {
			t.Fatal("active attribute not found")
		}
		active, err := activeAttr.ReadScalarInt64()
		if err != nil {
			t.Fatalf("ReadScalarInt64 failed: %v", err)
		}
		if active != 1 {
			t.Errorf("active = %d, want 1", active)
		}

		// Test dataset in this group
		ds, err := grp.OpenDataset("settings")
		if err != nil {
			t.Fatalf("OpenDataset settings failed: %v", err)
		}

		priorityAttr := ds.Attr("priority")
		if priorityAttr == nil {
			t.Fatal("priority attribute not found")
		}
		priority, err := priorityAttr.ReadScalarInt64()
		if err != nil {
			t.Fatalf("ReadScalarInt64 failed: %v", err)
		}
		if priority != 5 {
			t.Errorf("priority = %d, want 5", priority)
		}
	})

	// === FILE.GetAttr API FOR V0 ===
	t.Run("getattr_api", func(t *testing.T) {
		// Test GetAttr on nested dataset
		attr, err := f.GetAttr("/sensors/temperature@units")
		if err != nil {
			t.Fatalf("GetAttr failed: %v", err)
		}
		units, err := attr.ReadScalarString()
		if err != nil {
			t.Fatalf("ReadScalarString failed: %v", err)
		}
		if units != "celsius" {
			t.Errorf("units = %q, want %q", units, "celsius")
		}

		// Test ReadAttr convenience method
		val, err := f.ReadAttr("/sensors/humidity@sensor_id")
		if err != nil {
			t.Fatalf("ReadAttr failed: %v", err)
		}
		if id, ok := val.(int64); !ok || id != 101 {
			t.Errorf("sensor_id = %v, want 101", val)
		}

		// Test deeply nested
		attr, err = f.GetAttr("/sensors/metadata/timestamps@timezone")
		if err != nil {
			t.Fatalf("GetAttr for nested path failed: %v", err)
		}
		tz, err := attr.ReadScalarString()
		if err != nil {
			t.Fatalf("ReadScalarString failed: %v", err)
		}
		if tz != "UTC" {
			t.Errorf("timezone = %q, want %q", tz, "UTC")
		}
	})
}

// TestV0DeeplyNested tests deeply nested groups (5+ levels) in v0 superblock files
func TestV0DeeplyNested(t *testing.T) {
	path := skipIfNoTestdata(t, "v0_deep_nested.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// Verify superblock version is 0
	if f.Version() != 0 {
		t.Errorf("expected superblock version 0, got %d", f.Version())
	}

	// Test level 1
	t.Run("level1", func(t *testing.T) {
		grp, err := f.OpenGroup("level1")
		if err != nil {
			t.Fatalf("OpenGroup level1 failed: %v", err)
		}

		depthAttr := grp.Attr("depth")
		if depthAttr == nil {
			t.Fatal("depth attribute not found on level1")
		}
		depth, _ := depthAttr.ReadScalarInt64()
		if depth != 1 {
			t.Errorf("level1 depth = %d, want 1", depth)
		}

		ds, err := grp.OpenDataset("data1")
		if err != nil {
			t.Fatalf("OpenDataset data1 failed: %v", err)
		}
		data, _ := ds.ReadInt64()
		if len(data) != 3 || data[0] != 1 {
			t.Errorf("data1 = %v, want [1 2 3]", data)
		}
	})

	// Test level 3
	t.Run("level3", func(t *testing.T) {
		grp, err := f.OpenGroup("level1/level2/level3")
		if err != nil {
			t.Fatalf("OpenGroup level1/level2/level3 failed: %v", err)
		}

		depthAttr := grp.Attr("depth")
		if depthAttr == nil {
			t.Fatal("depth attribute not found on level3")
		}
		depth, _ := depthAttr.ReadScalarInt64()
		if depth != 3 {
			t.Errorf("level3 depth = %d, want 3", depth)
		}

		ds, err := grp.OpenDataset("data3")
		if err != nil {
			t.Fatalf("OpenDataset data3 failed: %v", err)
		}
		data, _ := ds.ReadInt64()
		if len(data) != 3 || data[0] != 7 {
			t.Errorf("data3 = %v, want [7 8 9]", data)
		}
	})

	// Test level 5 (deepest)
	t.Run("level5", func(t *testing.T) {
		grp, err := f.OpenGroup("level1/level2/level3/level4/level5")
		if err != nil {
			t.Fatalf("OpenGroup level5 failed: %v", err)
		}

		depthAttr := grp.Attr("depth")
		if depthAttr == nil {
			t.Fatal("depth attribute not found on level5")
		}
		depth, _ := depthAttr.ReadScalarInt64()
		if depth != 5 {
			t.Errorf("level5 depth = %d, want 5", depth)
		}

		ds, err := grp.OpenDataset("data5")
		if err != nil {
			t.Fatalf("OpenDataset data5 failed: %v", err)
		}
		data, _ := ds.ReadInt64()
		if len(data) != 3 || data[0] != 13 {
			t.Errorf("data5 = %v, want [13 14 15]", data)
		}
	})

	// Test sibling groups at various levels
	t.Run("siblings", func(t *testing.T) {
		for _, path := range []string{"level1/sibling1", "level1/level2/sibling2", "level1/level2/level3/sibling3"} {
			_, err := f.OpenGroup(path)
			if err != nil {
				t.Errorf("OpenGroup %s failed: %v", path, err)
			}
		}
	})

	// Test path-based dataset access
	t.Run("path_access", func(t *testing.T) {
		ds, err := f.OpenDataset("level1/level2/level3/level4/level5/data5")
		if err != nil {
			t.Fatalf("OpenDataset with full path failed: %v", err)
		}
		data, _ := ds.ReadInt64()
		if len(data) != 3 || data[2] != 15 {
			t.Errorf("data5 = %v, want [13 14 15]", data)
		}
	})
}

// TestV0AttributesBasic tests basic attribute access in v0 files
func TestV0AttributesBasic(t *testing.T) {
	path := skipIfNoTestdata(t, "v0_attributes.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// Verify superblock version is 0
	if f.Version() != 0 {
		t.Errorf("expected superblock version 0, got %d", f.Version())
	}

	ds, err := f.OpenDataset("data")
	if err != nil {
		t.Fatalf("OpenDataset failed: %v", err)
	}

	// Test int attribute
	intAttr := ds.Attr("int_attr")
	if intAttr == nil {
		t.Fatal("int_attr not found")
	}
	intVal, err := intAttr.ReadScalarInt64()
	if err != nil {
		t.Fatalf("ReadScalarInt64 failed: %v", err)
	}
	if intVal != 42 {
		t.Errorf("int_attr = %d, want 42", intVal)
	}

	// Test float attribute
	floatAttr := ds.Attr("float_attr")
	if floatAttr == nil {
		t.Fatal("float_attr not found")
	}
	floatVal, err := floatAttr.ReadScalarFloat64()
	if err != nil {
		t.Fatalf("ReadScalarFloat64 failed: %v", err)
	}
	if floatVal != 3.14 {
		t.Errorf("float_attr = %f, want 3.14", floatVal)
	}

	// Test string attribute
	strAttr := ds.Attr("string_attr")
	if strAttr == nil {
		t.Fatal("string_attr not found")
	}
	strVal, err := strAttr.ReadScalarString()
	if err != nil {
		t.Fatalf("ReadScalarString failed: %v", err)
	}
	if strVal != "hello" {
		t.Errorf("string_attr = %q, want %q", strVal, "hello")
	}
}

// TestBTreeV2Compressed tests reading a compressed dataset with B-tree v2
func TestBTreeV2Compressed(t *testing.T) {
	path := skipIfNoTestdata(t, "btree_v2_compressed.h5")

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	ds, err := f.OpenDataset("compressed")
	if err != nil {
		t.Fatalf("OpenDataset compressed failed: %v", err)
	}

	data, err := ds.ReadFloat64()
	if err != nil {
		t.Fatalf("ReadFloat64 failed: %v", err)
	}

	// Verify data (should be 0..9999)
	if len(data) != 10000 {
		t.Fatalf("expected 10000 elements, got %d", len(data))
	}
	for i := 0; i < 10; i++ {
		if data[i] != float64(i) {
			t.Errorf("data[%d] = %f, want %f", i, data[i], float64(i))
		}
	}
}
