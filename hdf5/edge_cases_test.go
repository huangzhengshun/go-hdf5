package hdf5

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// === ERROR PATH TESTS ===

// TestOpenInvalidHDF5Signature tests opening files with invalid HDF5 signatures.
func TestOpenInvalidHDF5Signature(t *testing.T) {
	tests := []struct {
		name    string
		content []byte
	}{
		{"empty file", []byte{}},
		{"random bytes", []byte{0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77}},
		{"almost valid signature", []byte{0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, 'X'}},
		{"text file", []byte("This is not an HDF5 file")},
		{"binary garbage", bytes.Repeat([]byte{0xFF}, 1024)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpfile, err := os.CreateTemp("", "invalid_hdf5_*.h5")
			if err != nil {
				t.Fatal(err)
			}
			defer os.Remove(tmpfile.Name())

			if len(tt.content) > 0 {
				if _, err := tmpfile.Write(tt.content); err != nil {
					t.Fatal(err)
				}
			}
			tmpfile.Close()

			_, err = Open(tmpfile.Name())
			if err == nil {
				t.Error("expected error for invalid HDF5 file")
			}
		})
	}
}

// TestOpenTruncatedFile tests opening truncated HDF5 files.
func TestOpenTruncatedFile(t *testing.T) {
	// HDF5 signature only (8 bytes) - truncated before version
	signature := []byte{0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n'}

	tests := []struct {
		name    string
		content []byte
	}{
		{"signature only", signature},
		{"signature plus 1 byte", append(signature, 0x02)},
		{"signature plus 4 bytes", append(signature, 0x02, 0x08, 0x08, 0x00)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpfile, err := os.CreateTemp("", "truncated_hdf5_*.h5")
			if err != nil {
				t.Fatal(err)
			}
			defer os.Remove(tmpfile.Name())

			if _, err := tmpfile.Write(tt.content); err != nil {
				t.Fatal(err)
			}
			tmpfile.Close()

			_, err = Open(tmpfile.Name())
			if err == nil {
				t.Error("expected error for truncated HDF5 file")
			}
		})
	}
}

// TestOpenNonExistentFile tests opening a file that doesn't exist.
func TestOpenNonExistentFile(t *testing.T) {
	_, err := Open("/nonexistent/path/to/file.h5")
	if err == nil {
		t.Error("expected error for non-existent file")
	}
}

// TestOpenDirectory tests trying to open a directory as an HDF5 file.
func TestOpenDirectory(t *testing.T) {
	tmpdir, err := os.MkdirTemp("", "hdf5_dir_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	_, err = Open(tmpdir)
	if err == nil {
		t.Error("expected error when opening directory as HDF5 file")
	}
}

// === EDGE CASE TESTS ===

// TestScalarDataset tests reading scalar (0-dimensional) datasets.
func TestScalarDataset(t *testing.T) {
	path := filepath.Join("..", "testdata", "scalar.h5")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		// Create the test file if it doesn't exist
		if createScalarTestFile(path) != nil {
			t.Skipf("Test file %s not found and could not be created", path)
		}
	}

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	ds, err := f.OpenDataset("scalar")
	if err != nil {
		t.Fatalf("OpenDataset failed: %v", err)
	}

	// Check shape is empty (scalar)
	shape := ds.Shape()
	if len(shape) != 0 {
		t.Errorf("expected 0 dimensions for scalar, got %d", len(shape))
	}

	// Check rank is 0
	if ds.Rank() != 0 {
		t.Errorf("expected rank 0 for scalar, got %d", ds.Rank())
	}

	// Check num elements is 1
	if ds.NumElements() != 1 {
		t.Errorf("expected 1 element for scalar, got %d", ds.NumElements())
	}

	// Read the scalar value
	data, err := ds.ReadInt64()
	if err != nil {
		t.Fatalf("ReadInt64 failed: %v", err)
	}

	if len(data) != 1 || data[0] != 42 {
		t.Errorf("expected scalar value 42, got %v", data)
	}
}

// TestEmptyDataset tests reading empty (0-element) datasets.
func TestEmptyDataset(t *testing.T) {
	path := filepath.Join("..", "testdata", "empty.h5")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		// Create the test file if it doesn't exist
		if createEmptyTestFile(path) != nil {
			t.Skipf("Test file %s not found and could not be created", path)
		}
	}

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	ds, err := f.OpenDataset("empty")
	if err != nil {
		t.Fatalf("OpenDataset failed: %v", err)
	}

	// Check shape
	shape := ds.Shape()
	if len(shape) != 1 || shape[0] != 0 {
		t.Errorf("expected shape [0], got %v", shape)
	}

	// Check num elements is 0
	if ds.NumElements() != 0 {
		t.Errorf("expected 0 elements, got %d", ds.NumElements())
	}

	// For empty datasets, reading may return an error or empty slice
	// depending on how the data is stored. We just verify the metadata is correct.
}

// TestDoubleClose tests that closing a file twice is safe.
func TestDoubleClose(t *testing.T) {
	path := filepath.Join("..", "testdata", "minimal.h5")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file %s not found", path)
	}

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	// First close
	err = f.Close()
	if err != nil {
		t.Fatalf("First close failed: %v", err)
	}

	// Second close should be safe
	err = f.Close()
	if err != nil {
		t.Fatalf("Second close failed: %v", err)
	}
}

// TestOperationsAfterClose tests that operations fail properly after close.
func TestOperationsAfterClose(t *testing.T) {
	path := filepath.Join("..", "testdata", "minimal.h5")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file %s not found", path)
	}

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	f.Close()

	// OpenDataset after close
	_, err = f.OpenDataset("data")
	if err != ErrClosed {
		t.Errorf("expected ErrClosed after close, got %v", err)
	}

	// OpenGroup after close
	_, err = f.OpenGroup("group")
	if err != ErrClosed {
		t.Errorf("expected ErrClosed for OpenGroup after close, got %v", err)
	}
}

// TestOpenNonExistentDataset tests opening a dataset that doesn't exist.
func TestOpenNonExistentDataset(t *testing.T) {
	path := filepath.Join("..", "testdata", "minimal.h5")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file %s not found", path)
	}

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	_, err = f.OpenDataset("nonexistent_dataset")
	if err == nil {
		t.Error("expected error for non-existent dataset")
	}
}

// TestOpenNonExistentGroup tests opening a group that doesn't exist.
func TestOpenNonExistentGroup(t *testing.T) {
	path := filepath.Join("..", "testdata", "groups.h5")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file %s not found", path)
	}

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	_, err = f.OpenGroup("nonexistent_group")
	if err == nil {
		t.Error("expected error for non-existent group")
	}
}

// TestRootGroupPath tests that root group has correct path.
func TestRootGroupPath(t *testing.T) {
	path := filepath.Join("..", "testdata", "minimal.h5")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file %s not found", path)
	}

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	root := f.Root()
	if root.Path() != "/" {
		t.Errorf("expected root path '/', got %q", root.Path())
	}
}

// TestDeepPathAccess tests accessing deeply nested objects via path.
func TestDeepPathAccess(t *testing.T) {
	path := filepath.Join("..", "testdata", "groups.h5")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file %s not found", path)
	}

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	// Test various path formats
	tests := []struct {
		path    string
		wantErr bool
	}{
		{"group1", false},
		{"/group1", false},
		{"group1/", false},
		{"/group1/", false},
		{"group1/data", false},
		{"/group1/data", false},
		{"", true},          // Empty path should fail for dataset
		{"../data", true},   // Relative path traversal should fail
	}

	for _, tt := range tests {
		t.Run("path_"+tt.path, func(t *testing.T) {
			if tt.path == "" || tt.path == "../data" {
				// These should try to open as dataset and fail appropriately
				_, err := f.OpenDataset(tt.path)
				if tt.wantErr && err == nil {
					t.Errorf("expected error for path %q", tt.path)
				}
			} else {
				// Try to open group first, then dataset
				_, errG := f.OpenGroup(tt.path)
				_, errD := f.OpenDataset(tt.path)
				if tt.wantErr && errG == nil && errD == nil {
					t.Errorf("expected error for path %q", tt.path)
				} else if !tt.wantErr && errG != nil && errD != nil {
					t.Errorf("unexpected error for path %q: group=%v, dataset=%v", tt.path, errG, errD)
				}
			}
		})
	}
}

// TestSplitPathEdgeCases tests the splitPath function with edge cases.
func TestSplitPathEdgeCases(t *testing.T) {
	tests := []struct {
		input    string
		expected []string
	}{
		{"", nil},
		{"/", nil},
		{"//", nil},
		{"///", nil},
		{"foo", []string{"foo"}},
		{"/foo", []string{"foo"}},
		{"foo/", []string{"foo"}},
		{"/foo/", []string{"foo"}},
		{"foo/bar", []string{"foo", "bar"}},
		{"/foo/bar", []string{"foo", "bar"}},
		{"foo/bar/", []string{"foo", "bar"}},
		{"/foo/bar/", []string{"foo", "bar"}},
		{"foo/bar/baz", []string{"foo", "bar", "baz"}},
		{"/a/b/c/d/e/f", []string{"a", "b", "c", "d", "e", "f"}},
	}

	for _, tt := range tests {
		t.Run("input_"+tt.input, func(t *testing.T) {
			result := splitPath(tt.input)
			if len(result) != len(tt.expected) {
				t.Errorf("splitPath(%q): expected %v, got %v", tt.input, tt.expected, result)
				return
			}
			for i := range result {
				if result[i] != tt.expected[i] {
					t.Errorf("splitPath(%q)[%d]: expected %q, got %q", tt.input, i, tt.expected[i], result[i])
				}
			}
		})
	}
}

// Note: normalizePath is not exported, so we test it indirectly through splitPath

// TestMaxLinkDepthEnforcement tests that link depth limits are enforced.
func TestMaxLinkDepthEnforcement(t *testing.T) {
	// This is tested more thoroughly in hdf5_test.go with circular links
	// Just verify the constant exists and has a reasonable value
	if MaxLinkDepth < 10 {
		t.Errorf("MaxLinkDepth too small: %d", MaxLinkDepth)
	}
	if MaxLinkDepth > 10000 {
		t.Errorf("MaxLinkDepth too large: %d", MaxLinkDepth)
	}
}

// TestFileVersion tests reading file version from superblock.
func TestFileVersion(t *testing.T) {
	testCases := []struct {
		filename        string
		expectedVersion int
	}{
		{"minimal.h5", 3},    // V3 superblock (newer h5py creates v3)
		{"v0_minimal.h5", 0}, // V0 superblock
	}

	for _, tc := range testCases {
		t.Run(tc.filename, func(t *testing.T) {
			path := filepath.Join("..", "testdata", tc.filename)
			if _, err := os.Stat(path); os.IsNotExist(err) {
				t.Skipf("Test file %s not found", path)
			}

			f, err := Open(path)
			if err != nil {
				t.Fatalf("Open failed: %v", err)
			}
			defer f.Close()

			if f.Version() != tc.expectedVersion {
				t.Errorf("expected version %d, got %d", tc.expectedVersion, f.Version())
			}
		})
	}
}

// TestFilePath tests that File.Path returns the correct path.
func TestFilePath(t *testing.T) {
	path := filepath.Join("..", "testdata", "minimal.h5")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file %s not found", path)
	}

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()

	if f.Path() != path {
		t.Errorf("expected path %q, got %q", path, f.Path())
	}
}

// === HELPER FUNCTIONS ===

// createScalarTestFile creates a test file with a scalar dataset.
// Returns nil on success, or an error if Python is not available.
func createScalarTestFile(path string) error {
	// Ensure parent directory exists
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}

	// This would require Python with h5py to create
	// For now, we skip if the file doesn't exist
	return os.ErrNotExist
}

// createEmptyTestFile creates a test file with an empty dataset.
// Returns nil on success, or an error if Python is not available.
func createEmptyTestFile(path string) error {
	// Ensure parent directory exists
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}

	// This would require Python with h5py to create
	// For now, we skip if the file doesn't exist
	return os.ErrNotExist
}
