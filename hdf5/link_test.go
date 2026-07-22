package hdf5

import (
	"os"
	"testing"
)

func TestCreateSoftLink(t *testing.T) {
	path := "test_softlink.h5"
	defer os.Remove(path)

	f, err := Create(path)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}
	defer f.Close()

	_, err = f.Root().CreateDataset("data", []int{1, 2, 3})
	if err != nil {
		t.Fatalf("CreateDataset failed: %v", err)
	}

	err = f.Root().CreateSoftLink("data_link", "/data")
	if err != nil {
		t.Fatalf("CreateSoftLink failed: %v", err)
	}

	if err := f.Flush(); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	f2, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f2.Close()

	_, err = f2.OpenDataset("data_link")
	if err != nil {
		t.Fatalf("OpenDataset via soft link failed: %v", err)
	}

	t.Logf("Successfully opened dataset via soft link")
}

func TestCreateExternalLink(t *testing.T) {
	externalPath := "test_external_data.h5"
	mainPath := "test_external_link.h5"
	defer os.Remove(externalPath)
	defer os.Remove(mainPath)

	extFile, err := Create(externalPath)
	if err != nil {
		t.Fatalf("Create external file failed: %v", err)
	}
	_, err = extFile.Root().CreateDataset("external_data", []float64{1.1, 2.2, 3.3})
	if err != nil {
		extFile.Close()
		t.Fatalf("CreateDataset in external file failed: %v", err)
	}
	if err := extFile.Close(); err != nil {
		t.Fatalf("Close external file failed: %v", err)
	}

	mainFile, err := Create(mainPath)
	if err != nil {
		t.Fatalf("Create main file failed: %v", err)
	}
	defer mainFile.Close()

	err = mainFile.Root().CreateExternalLink("external_ref", externalPath, "/external_data")
	if err != nil {
		t.Fatalf("CreateExternalLink failed: %v", err)
	}

	if err := mainFile.Flush(); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	t.Logf("Successfully created external link")
}
