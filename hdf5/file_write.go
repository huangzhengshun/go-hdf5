package hdf5

import (
	"encoding/binary"
	"os"

	"github.com/huangzhengshun/go-hdf5/internal/alloc"
	binpkg "github.com/huangzhengshun/go-hdf5/internal/binary"
	"github.com/huangzhengshun/go-hdf5/internal/object"
	"github.com/huangzhengshun/go-hdf5/internal/superblock"
)

// Note: encoding/binary is still needed for Create() which uses binary.LittleEndian

// Create creates a new HDF5 file at the given path.
// The file will be created with a V2 superblock and V2 object headers.
func Create(path string, opts ...FileOption) (*File, error) {
	options := defaultFileOptions()
	for _, opt := range opts {
		opt(options)
	}

	// Create the file
	osFile, err := os.Create(path)
	if err != nil {
		return nil, err
	}

	// Create writer
	cfg := binpkg.Config{
		ByteOrder:  binary.LittleEndian,
		OffsetSize: options.offsetSize,
		LengthSize: options.lengthSize,
	}
	writer := binpkg.NewWriter(osFile, cfg)

	// Create superblock
	sb := superblock.NewSuperblock()
	sb.OffsetSize = uint8(options.offsetSize)
	sb.LengthSize = uint8(options.lengthSize)

	// Write superblock (will need to update EOF and root group address later)
	sbSize := sb.Size()

	// Calculate root group address (right after superblock)
	rootGroupAddr := uint64(sbSize)
	sb.RootGroupAddress = rootGroupAddr

	// Create root group object header (empty group)
	rootMessages := object.NewEmptyGroupHeader()

	// Calculate header size to determine EOF
	// Use minimum chunk size for compatibility with h5py
	headerSize := object.HeaderSizeWithMinChunk(writer, rootMessages, object.MinGroupChunkSize)
	eofAddr := uint64(sbSize + headerSize)
	sb.EOFAddress = eofAddr

	// Now write the superblock with correct addresses
	if _, err := sb.Write(writer); err != nil {
		osFile.Close()
		os.Remove(path)
		return nil, err
	}

	// Write root group object header with minimum chunk size
	if _, err := object.WriteHeaderWithMinChunk(writer, rootMessages, object.MinGroupChunkSize); err != nil {
		osFile.Close()
		os.Remove(path)
		return nil, err
	}

	// Create allocator starting at EOF
	allocator := alloc.New(eofAddr)

	// Create File structure
	f := &File{
		path:       path,
		file:       osFile,
		superblock: sb,
		writable:   true,
		writer:     writer,
		allocator:  allocator,
		groupCache: make(map[string]*Group),
	}

	// Create root group
	f.root = &Group{
		file:   f,
		path:   "/",
		header: nil, // Will be loaded on demand
		addr:   rootGroupAddr,
	}

	// Add root group to cache
	f.mu.Lock()
	f.groupCache["/"] = f.root
	f.mu.Unlock()

	return f, nil
}

// Flush writes any pending changes to disk.
func (f *File) Flush() error {
	if !f.writable {
		return nil
	}

	// Update superblock with current EOF from allocator
	f.superblock.EOFAddress = f.allocator.EOFAddr()

	// Rewrite superblock at beginning of file
	w := f.writer.At(0)
	if _, err := f.superblock.Write(w); err != nil {
		return err
	}

	// Sync to disk
	return f.file.Sync()
}

// allocate reserves space in the file and returns the address.
func (f *File) allocate(size int64) uint64 {
	return f.allocator.Alloc(uint64(size))
}

// AllocStats returns allocation statistics (for debugging/testing).
func (f *File) AllocStats() alloc.Stats {
	if f.allocator == nil {
		return alloc.Stats{}
	}
	return f.allocator.Stats()
}

// closeWritable handles closing a writable file.
func (f *File) closeWritable() error {
	// Flush pending changes
	if err := f.Flush(); err != nil {
		return err
	}

	return nil
}

// OpenReadWrite opens an existing HDF5 file for reading and writing.
// This allows adding new groups, datasets, and attributes to existing files.
func OpenReadWrite(path string) (*File, error) {
	// Open file with read-write permissions
	osFile, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		return nil, err
	}

	// Parse existing superblock
	sb, err := superblock.Read(osFile)
	if err != nil {
		osFile.Close()
		return nil, err
	}

	// Create reader with correct configuration
	readerCfg := sb.ReaderConfig()
	reader := binpkg.NewReader(osFile, readerCfg)

	// Create writer with same configuration as reader
	// This ensures we use the same byte order, offset size, and length size
	writer := binpkg.NewWriter(osFile, readerCfg)

	// Create allocator starting at current EOF
	allocator := alloc.New(sb.EOFAddress)

	// Create File structure
	f := &File{
		path:       path,
		file:       osFile,
		reader:     reader,
		superblock: sb,
		writable:   true,
		writer:     writer,
		allocator:  allocator,
		groupCache: make(map[string]*Group),
	}

	// Load root group
	root, err := f.openGroupAt(sb.RootGroupAddress, "/")
	if err != nil {
		osFile.Close()
		return nil, err
	}
	f.root = root

	// Add root group to cache
	f.mu.Lock()
	f.groupCache["/"] = root
	f.mu.Unlock()

	return f, nil
}

// IsWritable returns true if the file was opened for writing.
func (f *File) IsWritable() bool {
	return f.writable
}
