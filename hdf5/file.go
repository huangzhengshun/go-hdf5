package hdf5

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/huangzhengshun/go-hdf5/internal/alloc"
	"github.com/huangzhengshun/go-hdf5/internal/binary"
	"github.com/huangzhengshun/go-hdf5/internal/object"
	"github.com/huangzhengshun/go-hdf5/internal/superblock"
)

// File represents an open HDF5 file.
type File struct {
	path          string
	file          *os.File
	reader        *binary.Reader
	superblock    *superblock.Superblock
	root          *Group
	closed        bool
	externalFiles map[string]*File // Cache of opened external files

	// Write support fields
	writable   bool
	writer     *binary.Writer
	allocator  *alloc.Allocator  // Space allocator for writing
	groupCache map[string]*Group // Cache of created groups for parent lookup
	mu         sync.RWMutex      // Protects groupCache and other shared state
}

// Open opens an HDF5 file for reading.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening file: %w", err)
	}

	// Parse superblock
	sb, err := superblock.Read(f)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("reading superblock: %w", err)
	}

	// Create reader with correct configuration
	reader := binary.NewReader(f, sb.ReaderConfig())

	hdf := &File{
		path:       path,
		file:       f,
		reader:     reader,
		superblock: sb,
	}

	// Load root group
	root, err := hdf.openGroupAt(sb.RootGroupAddress, "/")
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("opening root group: %w", err)
	}
	hdf.root = root

	return hdf, nil
}

// Close closes the HDF5 file and all opened external files.
func (f *File) Close() error {
	if f.closed {
		return nil
	}
	f.closed = true

	// Handle writable file finalization
	if f.writable {
		if err := f.closeWritable(); err != nil {
			f.file.Close()
			return err
		}
	}

	// Close all external files
	for _, extFile := range f.externalFiles {
		extFile.Close()
	}
	f.externalFiles = nil

	return f.file.Close()
}

// Root returns the root group of the file.
func (f *File) Root() *Group {
	return f.root
}

// Path returns the file path.
func (f *File) Path() string {
	return f.path
}

// Version returns the superblock version.
func (f *File) Version() int {
	return int(f.superblock.Version)
}

// OpenGroup opens a group by path.
func (f *File) OpenGroup(path string) (*Group, error) {
	if f.closed {
		return nil, ErrClosed
	}
	return f.root.OpenGroup(path)
}

// OpenDataset opens a dataset by path.
func (f *File) OpenDataset(path string) (*Dataset, error) {
	if f.closed {
		return nil, ErrClosed
	}
	return f.root.OpenDataset(path)
}

// openGroupAt opens a group at the given address.
func (f *File) openGroupAt(address uint64, path string) (*Group, error) {
	header, err := object.Read(f.reader, address)
	if err != nil {
		return nil, fmt.Errorf("reading object header: %w", err)
	}

	group := &Group{
		file:   f,
		path:   path,
		header: header,
		addr:   address,
	}

	// Add to group cache for parent lookup
	f.mu.Lock()
	if f.groupCache != nil {
		f.groupCache[path] = group
	}
	f.mu.Unlock()

	return group, nil
}

// openDatasetAt opens a dataset at the given address.
func (f *File) openDatasetAt(address uint64, path string) (*Dataset, error) {
	header, err := object.Read(f.reader, address)
	if err != nil {
		return nil, fmt.Errorf("reading object header: %w", err)
	}

	return newDataset(f, path, header)
}

// splitPath splits a path into its components.
// This is an internal wrapper around SplitPath for compatibility.
func splitPath(path string) []string {
	return SplitPath(path)
}

// GetAttr returns an attribute by path.
// Path format: /group/object@attribute_name
//
// Examples:
//   - "/@root_attr" - attribute on root group
//   - "/data@units" - attribute on dataset 'data'
//   - "/sensors/temp@calibration" - attribute on nested dataset
func (f *File) GetAttr(path string) (*Attribute, error) {
	if f.closed {
		return nil, ErrClosed
	}

	objectPath, attrName, err := ParseAttrPath(path)
	if err != nil {
		return nil, err
	}

	// Get the object (group or dataset) at the path
	obj, err := f.getAttributeHolder(objectPath)
	if err != nil {
		return nil, fmt.Errorf("opening object %s: %w", objectPath, err)
	}

	// Get the attribute from the object
	attr := obj.Attr(attrName)
	if attr == nil {
		return nil, ErrAttributeNotFound
	}
	return attr, nil
}

// ReadAttr reads an attribute value by path.
// This is a convenience method that combines GetAttr and Attribute.Value().
//
// Examples:
//
//	val, err := f.ReadAttr("/@version")
//	val, err := f.ReadAttr("/dataset@units")
func (f *File) ReadAttr(path string) (interface{}, error) {
	attr, err := f.GetAttr(path)
	if err != nil {
		return nil, err
	}
	return attr.Value()
}

// attributeHolder is an interface for objects that can have attributes.
type attributeHolder interface {
	Attr(name string) *Attribute
}

// getAttributeHolder returns the group or dataset at the given path.
func (f *File) getAttributeHolder(path string) (attributeHolder, error) {
	if path == "/" {
		return f.root, nil
	}

	// Try opening as a group first
	group, err := f.OpenGroup(path)
	if err == nil {
		return group, nil
	}

	// If that failed, try as a dataset
	dataset, err := f.OpenDataset(path)
	if err == nil {
		return dataset, nil
	}

	return nil, ErrNotFound
}

// findByAbsolutePath navigates an absolute path and returns the target's address.
// This is used for resolving soft links. The visited map tracks paths to detect cycles.
func (f *File) findByAbsolutePath(absPath string, visited map[string]bool) (uint64, bool, error) {
	res, err := f.findByAbsolutePathFull(absPath, visited)
	if err != nil {
		return 0, false, err
	}
	return res.address, res.isDataset, nil
}

// findByAbsolutePathFull navigates an absolute path and returns the full resolution info.
// This handles cases where the target is in an external file.
func (f *File) findByAbsolutePathFull(absPath string, visited map[string]bool) (*linkResolution, error) {
	parts := splitPath(absPath)
	if len(parts) == 0 {
		// Path is "/" - return root group
		// Root group address comes from superblock
		return &linkResolution{
			address:   f.superblock.RootGroupAddress,
			isDataset: false,
			file:      nil,
		}, nil
	}

	current := f.root
	currentFile := f

	for i, name := range parts {
		res, err := current.findChildFull(name, visited)
		if err != nil {
			return nil, fmt.Errorf("resolving %q in path %s: %w", name, absPath, err)
		}

		// If this component resolved to an external file, switch context
		if res.file != nil {
			currentFile = res.file
		}

		if i == len(parts)-1 {
			// Last component - return this resolution
			return res, nil
		}

		// Not the last component - must be a group to continue traversal
		if res.isDataset {
			return nil, fmt.Errorf("%q is not a group in path %s", name, absPath)
		}

		// Open the next group in the appropriate file
		nextGroup, err := currentFile.openGroupAt(res.address, "")
		if err != nil {
			return nil, fmt.Errorf("opening group %q: %w", name, err)
		}
		current = nextGroup
	}

	// Should not reach here
	return nil, fmt.Errorf("empty path")
}

// openExternalFile opens an external file by name, relative to the current file's directory.
// Files are cached to avoid repeated opens.
func (f *File) openExternalFile(filename string) (*File, error) {
	// Check cache first
	if f.externalFiles != nil {
		if extFile, ok := f.externalFiles[filename]; ok {
			return extFile, nil
		}
	}

	// Resolve path relative to current file's directory
	baseDir := filepath.Dir(f.path)
	extPath := filepath.Join(baseDir, filename)

	// Open the external file
	extFile, err := Open(extPath)
	if err != nil {
		return nil, fmt.Errorf("opening external file %q: %w", extPath, err)
	}

	// Cache it
	if f.externalFiles == nil {
		f.externalFiles = make(map[string]*File)
	}
	f.externalFiles[filename] = extFile

	return extFile, nil
}

// resolveExternalLink resolves an external link and returns the target's address and file.
// The visited map tracks paths to detect cycles across files.
func (f *File) resolveExternalLink(extFile string, extPath string, visited map[string]bool) (uint64, bool, *File, error) {
	// Check depth limit
	if len(visited) >= MaxLinkDepth {
		return 0, false, nil, ErrLinkDepth
	}

	// Create a unique key for cycle detection
	linkKey := extFile + ":" + extPath
	if visited[linkKey] {
		return 0, false, nil, fmt.Errorf("circular external link detected: %s", linkKey)
	}
	visited[linkKey] = true

	// Open the external file
	targetFile, err := f.openExternalFile(extFile)
	if err != nil {
		return 0, false, nil, err
	}

	// Resolve the path in the external file
	addr, isDataset, err := targetFile.findByAbsolutePath(extPath, visited)
	if err != nil {
		return 0, false, nil, fmt.Errorf("resolving path %q in external file %q: %w", extPath, extFile, err)
	}

	return addr, isDataset, targetFile, nil
}
