package hdf5

import (
	"encoding/binary"
	"fmt"
	"os"
	"path"

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

	// Create reader for reading back data
	reader := binpkg.NewReader(osFile, cfg)

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

	// Create root group
	f.root = &Group{
		file:              f,
		path:              "/",
		header:            nil, // Will be loaded on demand
		addr:              rootGroupAddr,
		pendingLinks:      nil,
		pendingAttributes: nil,
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

	f.mu.Lock()

	// Collect all groups that need rewrite
	rewriteSet := make(map[string]*Group)
	if f.root.pendingAttributes != nil || f.root.pendingLinks != nil {
		rewriteSet[f.root.path] = f.root
	}
	for _, grp := range f.groupCache {
		if grp.pendingAttributes != nil || grp.pendingLinks != nil {
			rewriteSet[grp.path] = grp
		}
	}

	// If no groups need rewriting, nothing to do
	if len(rewriteSet) == 0 {
		f.mu.Unlock()
		return nil
	}

	// Load existing links and attributes for all groups before reallocating addresses
	// Both must be loaded to avoid losing existing data during serialization
	for _, grp := range rewriteSet {
		if grp.pendingLinks == nil {
			grp.loadExistingLinks()
		}
		if grp.pendingAttributes == nil {
			grp.loadExistingAttributes()
		}
	}

	f.mu.Unlock()

	// Cascading: when a group's address changes, its parent must also be rewritten.
	// We need to add all ancestor groups up to the root.
	// Repeat until no new parents are added.
	changed := true
	for changed {
		changed = false
		for _, grp := range rewriteSet {
			if grp.path == "/" {
				continue
			}
			parent := grp.findParent()
			if parent == nil {
				continue
			}
			if _, exists := rewriteSet[parent.path]; !exists {
				// Parent needs to be rewritten because child's address will change
				if parent.pendingLinks == nil {
					parent.loadExistingLinks()
				}
				rewriteSet[parent.path] = parent
				changed = true
			}
		}
	}

	// Convert set to slice and sort by path depth (deepest first)
	var toRewrite []*Group
	for _, grp := range rewriteSet {
		toRewrite = append(toRewrite, grp)
	}
	for i := 0; i < len(toRewrite)-1; i++ {
		for j := i + 1; j < len(toRewrite); j++ {
			if pathDepth(toRewrite[i].path) < pathDepth(toRewrite[j].path) {
				toRewrite[i], toRewrite[j] = toRewrite[j], toRewrite[i]
			}
		}
	}

	// First pass: serialize all headers to memory buffers and calculate actual sizes
	type groupInfo struct {
		grp         *Group
		headerBytes []byte
		oldAddr     uint64
	}
	var groupInfos []groupInfo

	for _, grp := range toRewrite {
		messages := object.NewGroupHeader(grp.pendingLinks)
		for _, attr := range grp.pendingAttributes {
			messages = append(messages, attr)
		}

		headerBytes, err := object.SerializeHeader(f.writer, messages, object.MinGroupChunkSize)
		if err != nil {
			return fmt.Errorf("serializing group %s: %w", grp.path, err)
		}

		groupInfos = append(groupInfos, groupInfo{grp: grp, headerBytes: headerBytes, oldAddr: grp.addr})
	}

	// Second pass: allocate new addresses (deepest first)
	for i := range groupInfos {
		newAddr := f.allocate(int64(len(groupInfos[i].headerBytes)))
		groupInfos[i].grp.addr = newAddr
	}

	// Third pass: update parent links to point to new addresses
	for _, gi := range groupInfos {
		if gi.grp.path != "/" {
			parent := gi.grp.findParent()
			if parent != nil && parent.pendingLinks != nil {
				for _, link := range parent.pendingLinks {
					if link.Name == path.Base(gi.grp.path) {
						link.ObjectAddress = gi.grp.addr
						break
					}
				}
			}
		}
	}

	// Fourth pass: re-serialize all headers with updated links
	groupInfos = nil
	for _, grp := range toRewrite {
		messages := object.NewGroupHeader(grp.pendingLinks)
		for _, attr := range grp.pendingAttributes {
			messages = append(messages, attr)
		}

		headerBytes, err := object.SerializeHeader(f.writer, messages, object.MinGroupChunkSize)
		if err != nil {
			return fmt.Errorf("re-serializing group %s: %w", grp.path, err)
		}

		groupInfos = append(groupInfos, groupInfo{grp: grp, headerBytes: headerBytes, oldAddr: grp.addr})
	}

	// Fifth pass: write all headers (deepest first)
	var rootAddr uint64
	for _, gi := range groupInfos {
		w := f.writer.At(int64(gi.grp.addr))
		if err := w.WriteBytes(gi.headerBytes); err != nil {
			return fmt.Errorf("writing group %s: %w", gi.grp.path, err)
		}

		if gi.grp.path == "/" {
			rootAddr = gi.grp.addr
		}

		// Clear pending changes and stale header after writing
		// The header is stale because the group was rewritten at a new address
		gi.grp.pendingLinks = nil
		gi.grp.pendingAttributes = nil
		gi.grp.header = nil
	}

	// Update superblock with root group address and EOF
	if rootAddr != 0 {
		f.superblock.RootGroupAddress = rootAddr
	}
	f.superblock.EOFAddress = f.allocator.EOFAddr()

	// Sync all writes to disk
	if err := f.file.Sync(); err != nil {
		return err
	}

	// Rewrite superblock at beginning of file
	w := f.writer.At(0)
	if _, err := f.superblock.Write(w); err != nil {
		return err
	}

	return f.file.Sync()
}

// memoryWriter is a simple in-memory writer for buffering
type memoryWriter struct {
	buf []byte
}

func (m *memoryWriter) WriteAt(p []byte, off int64) (n int, err error) {
	if int(off)+len(p) > len(m.buf) {
		newBuf := make([]byte, int(off)+len(p))
		copy(newBuf, m.buf)
		m.buf = newBuf
	}
	copy(m.buf[off:], p)
	return len(p), nil
}

// pathDepth returns the depth of a path
func pathDepth(p string) int {
	depth := 0
	for _, c := range p {
		if c == '/' {
			depth++
		}
	}
	return depth
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
