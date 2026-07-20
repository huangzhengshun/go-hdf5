package hdf5

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCreateDatasetWithScalarAttribute(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hdf5-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test_attr_scalar.h5")

	// Create file with dataset and scalar attribute
	f, err := Create(testFile)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	data := []int32{1, 2, 3, 4, 5}
	_, err = f.Root().CreateDataset("data", data,
		WithAttribute("scale", float64(1.5)),
		WithAttribute("offset", int32(100)),
	)
	if err != nil {
		t.Fatalf("CreateDataset with attributes failed: %v", err)
	}

	f.Close()

	// Reopen and verify
	f2, err := Open(testFile)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f2.Close()

	ds, err := f2.Root().OpenDataset("data")
	if err != nil {
		t.Fatalf("OpenDataset failed: %v", err)
	}

	// Check attributes exist
	attrs := ds.Attrs()
	if len(attrs) != 2 {
		t.Errorf("Expected 2 attributes, got %d: %v", len(attrs), attrs)
	}

	// Read scale attribute
	scaleAttr := ds.Attr("scale")
	if scaleAttr == nil {
		t.Fatal("scale attribute not found")
	}

	scaleVal, err := scaleAttr.ReadScalarFloat64()
	if err != nil {
		t.Fatalf("ReadScalarFloat64 failed: %v", err)
	}
	if scaleVal != 1.5 {
		t.Errorf("scale: expected 1.5, got %f", scaleVal)
	}

	// Read offset attribute
	offsetAttr := ds.Attr("offset")
	if offsetAttr == nil {
		t.Fatal("offset attribute not found")
	}

	offsetVal, err := offsetAttr.ReadScalarInt64()
	if err != nil {
		t.Fatalf("ReadScalarInt64 failed: %v", err)
	}
	if offsetVal != 100 {
		t.Errorf("offset: expected 100, got %d", offsetVal)
	}
}

func TestCreateDatasetWithArrayAttribute(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hdf5-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test_attr_array.h5")

	// Create file with dataset and array attribute
	f, err := Create(testFile)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	data := []float64{1.0, 2.0, 3.0}
	calibration := []float64{0.5, 1.0, 1.5}
	_, err = f.Root().CreateDataset("measurements", data,
		WithAttribute("calibration", calibration),
	)
	if err != nil {
		t.Fatalf("CreateDataset with array attribute failed: %v", err)
	}

	f.Close()

	// Reopen and verify
	f2, err := Open(testFile)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f2.Close()

	ds, err := f2.Root().OpenDataset("measurements")
	if err != nil {
		t.Fatalf("OpenDataset failed: %v", err)
	}

	// Read calibration attribute
	calAttr := ds.Attr("calibration")
	if calAttr == nil {
		t.Fatal("calibration attribute not found")
	}

	calVals, err := calAttr.ReadFloat64()
	if err != nil {
		t.Fatalf("ReadFloat64 failed: %v", err)
	}

	if len(calVals) != 3 {
		t.Errorf("Expected 3 values, got %d", len(calVals))
	}

	for i, expected := range calibration {
		if calVals[i] != expected {
			t.Errorf("calibration[%d]: expected %f, got %f", i, expected, calVals[i])
		}
	}
}

func TestCreateDatasetWithIntegerAttribute(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hdf5-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test_attr_int.h5")

	f, err := Create(testFile)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	data := []int32{10, 20, 30}
	indices := []int32{0, 1, 2}
	_, err = f.Root().CreateDataset("indexed_data", data,
		WithAttribute("indices", indices),
		WithAttribute("count", int64(3)),
	)
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

	ds, err := f2.Root().OpenDataset("indexed_data")
	if err != nil {
		t.Fatalf("OpenDataset failed: %v", err)
	}

	// Verify indices attribute
	indicesAttr := ds.Attr("indices")
	if indicesAttr == nil {
		t.Fatal("indices attribute not found")
	}

	indicesVals, err := indicesAttr.ReadInt32()
	if err != nil {
		t.Fatalf("ReadInt32 failed: %v", err)
	}

	for i, expected := range indices {
		if indicesVals[i] != expected {
			t.Errorf("indices[%d]: expected %d, got %d", i, expected, indicesVals[i])
		}
	}

	// Verify count attribute
	countAttr := ds.Attr("count")
	if countAttr == nil {
		t.Fatal("count attribute not found")
	}

	countVal, err := countAttr.ReadScalarInt64()
	if err != nil {
		t.Fatalf("ReadScalarInt64 failed: %v", err)
	}
	if countVal != 3 {
		t.Errorf("count: expected 3, got %d", countVal)
	}
}

func TestCreateDatasetWithStringAttribute(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hdf5-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test_attr_string.h5")

	f, err := Create(testFile)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	data := []float64{1.0, 2.0, 3.0}
	_, err = f.Root().CreateDataset("data", data,
		WithAttribute("units", "meters"),
		WithAttribute("description", "Test measurements"),
	)
	if err != nil {
		t.Fatalf("CreateDataset with string attributes failed: %v", err)
	}

	f.Close()

	// Reopen and verify
	f2, err := Open(testFile)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f2.Close()

	ds, err := f2.Root().OpenDataset("data")
	if err != nil {
		t.Fatalf("OpenDataset failed: %v", err)
	}

	// Check attributes exist
	attrs := ds.Attrs()
	if len(attrs) != 2 {
		t.Errorf("Expected 2 attributes, got %d: %v", len(attrs), attrs)
	}

	// Read units attribute
	unitsAttr := ds.Attr("units")
	if unitsAttr == nil {
		t.Fatal("units attribute not found")
	}

	unitsVal, err := unitsAttr.ReadScalarString()
	if err != nil {
		t.Fatalf("ReadScalarString failed: %v", err)
	}
	if unitsVal != "meters" {
		t.Errorf("units: expected 'meters', got '%s'", unitsVal)
	}

	// Read description attribute
	descAttr := ds.Attr("description")
	if descAttr == nil {
		t.Fatal("description attribute not found")
	}

	descVal, err := descAttr.ReadScalarString()
	if err != nil {
		t.Fatalf("ReadScalarString failed: %v", err)
	}
	if descVal != "Test measurements" {
		t.Errorf("description: expected 'Test measurements', got '%s'", descVal)
	}
}

func TestCreateDatasetWithStringArrayAttribute(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hdf5-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test_attr_string_array.h5")

	f, err := Create(testFile)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	data := []int32{1, 2, 3}
	labels := []string{"alpha", "beta", "gamma"}
	_, err = f.Root().CreateDataset("labeled_data", data,
		WithAttribute("labels", labels),
	)
	if err != nil {
		t.Fatalf("CreateDataset with string array attribute failed: %v", err)
	}

	f.Close()

	// Reopen and verify
	f2, err := Open(testFile)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f2.Close()

	ds, err := f2.Root().OpenDataset("labeled_data")
	if err != nil {
		t.Fatalf("OpenDataset failed: %v", err)
	}

	// Read labels attribute
	labelsAttr := ds.Attr("labels")
	if labelsAttr == nil {
		t.Fatal("labels attribute not found")
	}

	labelsVal, err := labelsAttr.ReadString()
	if err != nil {
		t.Fatalf("ReadString failed: %v", err)
	}

	if len(labelsVal) != 3 {
		t.Fatalf("Expected 3 labels, got %d", len(labelsVal))
	}

	for i, expected := range labels {
		if labelsVal[i] != expected {
			t.Errorf("labels[%d]: expected '%s', got '%s'", i, expected, labelsVal[i])
		}
	}
}
