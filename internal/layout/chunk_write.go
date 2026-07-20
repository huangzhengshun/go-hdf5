package layout

import (
	"github.com/huangzhengshun/go-hdf5/internal/binary"
)

// ChunkWriter handles writing chunked dataset data and indices.
type ChunkWriter struct {
	w            *binary.Writer
	chunkDims    []uint32
	elementSize  uint32
	filterMask   uint32 // 0 = all filters applied
	allocator    func(size int64) uint64
}

// NewChunkWriter creates a new chunk writer.
func NewChunkWriter(w *binary.Writer, chunkDims []uint32, elementSize uint32, allocator func(size int64) uint64) *ChunkWriter {
	return &ChunkWriter{
		w:           w,
		chunkDims:   chunkDims,
		elementSize: elementSize,
		filterMask:  0,
		allocator:   allocator,
	}
}

// ChunkSize returns the size in bytes of one chunk.
func (cw *ChunkWriter) ChunkSize() uint64 {
	size := uint64(cw.elementSize)
	for _, dim := range cw.chunkDims {
		size *= uint64(dim)
	}
	return size
}

// WriteSingleChunk writes the entire data as a single chunk and returns the chunk address.
// This is used when the dataset is smaller than or equal to one chunk.
func (cw *ChunkWriter) WriteSingleChunk(data []byte) (uint64, error) {
	// Allocate space for the chunk
	addr := cw.allocator(int64(len(data)))

	// Write the chunk data
	w := cw.w.At(int64(addr))
	if err := w.WriteBytes(data); err != nil {
		return 0, err
	}

	return addr, nil
}

// WriteSingleChunkIndex writes a single chunk index structure.
// Returns the address of the index.
func (cw *ChunkWriter) WriteSingleChunkIndex(chunkAddr uint64, chunkSize uint32) (uint64, error) {
	// Single Chunk Index format (for layout version 4, chunk index type 0):
	// - Filtered chunk size (if filters present): Length size bytes
	// - Filter mask (if filters present): 4 bytes
	// - Chunk address: Offset size bytes

	// For now, assume no filters (simplified)
	indexSize := cw.w.OffsetSize()
	indexAddr := cw.allocator(int64(indexSize))

	w := cw.w.At(int64(indexAddr))
	if err := w.WriteOffset(chunkAddr); err != nil {
		return 0, err
	}

	return indexAddr, nil
}

// FixedArrayHeader represents the header for a Fixed Array chunk index.
type FixedArrayHeader struct {
	Signature      [4]byte // "FAHD"
	Version        uint8   // Currently 0
	ClientID       uint8   // 0 = non-filtered chunks, 1 = filtered chunks
	EntrySize      uint8   // Size of each element entry
	PageBits       uint8   // log2 of entries per page
	MaxNumEntries  uint64  // Maximum number of entries in array
	DataBlockAddr  uint64  // Address of data block
}

// WriteFixedArrayIndex writes a fixed array chunk index.
// chunkAddrs contains the address of each chunk in storage order.
func (cw *ChunkWriter) WriteFixedArrayIndex(chunkAddrs []uint64, chunkSizes []uint32) (uint64, error) {
	numChunks := len(chunkAddrs)
	if numChunks == 0 {
		return 0, nil
	}

	// For non-filtered chunks, entry size = offset size
	entrySize := cw.w.OffsetSize()
	offsetSize := cw.w.OffsetSize()
	lengthSize := cw.w.LengthSize()

	// Calculate page bits - for small arrays use smaller page size
	pageBits := uint8(10) // Match h5py's default
	if numChunks > 1024 {
		pageBits = 12
	}

	// First, write the Fixed Array Header to get its address
	// Header size: signature(4) + version(1) + clientID(1) + entrySize(1) + pageBits(1) +
	//              maxEntries(lengthSize) + dataBlockAddr(offsetSize) + checksum(4)
	headerSize := 4 + 1 + 1 + 1 + 1 + lengthSize + offsetSize + 4
	headerAddr := cw.allocator(int64(headerSize))

	// Now write the data block with proper signature
	// Data block size: signature(4) + version(1) + clientID(1) + headerAddr(offsetSize) +
	//                  entries(numChunks * entrySize) + checksum(4)
	dataBlockSize := 4 + 1 + 1 + offsetSize + numChunks*entrySize + 4
	dataBlockAddr := cw.allocator(int64(dataBlockSize))

	// Build FADB (data block) in memory to compute checksum
	fadbData := make([]byte, dataBlockSize)
	idx := 0

	// Signature "FADB"
	copy(fadbData[idx:], []byte("FADB"))
	idx += 4

	// Version
	fadbData[idx] = 0
	idx++

	// Client ID (0 = non-filtered chunks)
	fadbData[idx] = 0
	idx++

	// Header address
	putUint64LE(fadbData[idx:], headerAddr, offsetSize)
	idx += offsetSize

	// Write each chunk address (the element entries)
	for _, addr := range chunkAddrs {
		putUint64LE(fadbData[idx:], addr, offsetSize)
		idx += offsetSize
	}

	// Compute and add checksum
	fadbChecksum := binary.Lookup3Checksum(fadbData[:idx])
	putUint32LE(fadbData[idx:], fadbChecksum)
	idx += 4

	// Write FADB to file
	w := cw.w.At(int64(dataBlockAddr))
	if err := w.WriteBytes(fadbData); err != nil {
		return 0, err
	}

	// Build FAHD (header) in memory to compute checksum
	fahdData := make([]byte, headerSize)
	idx = 0

	// Signature "FAHD"
	copy(fahdData[idx:], []byte("FAHD"))
	idx += 4

	// Version
	fahdData[idx] = 0
	idx++

	// Client ID (0 = non-filtered chunks)
	fahdData[idx] = 0
	idx++

	// Entry size
	fahdData[idx] = uint8(entrySize)
	idx++

	// Page bits
	fahdData[idx] = pageBits
	idx++

	// Max number of entries
	putUint64LE(fahdData[idx:], uint64(numChunks), lengthSize)
	idx += lengthSize

	// Data block address
	putUint64LE(fahdData[idx:], dataBlockAddr, offsetSize)
	idx += offsetSize

	// Compute and add checksum
	fahdChecksum := binary.Lookup3Checksum(fahdData[:idx])
	putUint32LE(fahdData[idx:], fahdChecksum)
	idx += 4

	// Write FAHD to file
	hw := cw.w.At(int64(headerAddr))
	if err := hw.WriteBytes(fahdData); err != nil {
		return 0, err
	}

	return headerAddr, nil
}

// WriteChunks writes multiple chunks and returns their addresses.
func (cw *ChunkWriter) WriteChunks(chunks [][]byte) ([]uint64, error) {
	addrs := make([]uint64, len(chunks))

	for i, chunk := range chunks {
		addr, err := cw.WriteSingleChunk(chunk)
		if err != nil {
			return nil, err
		}
		addrs[i] = addr
	}

	return addrs, nil
}

// WriteExtensibleArrayIndex writes an extensible array chunk index.
// This is the format that h5py uses for multi-chunk datasets.
// chunkAddrs contains the address of each chunk in storage order.
func (cw *ChunkWriter) WriteExtensibleArrayIndex(chunkAddrs []uint64) (uint64, error) {
	numChunks := len(chunkAddrs)
	if numChunks == 0 {
		return 0, nil
	}

	// For non-filtered chunks, element size = offset size
	elemSize := cw.w.OffsetSize()
	offsetSize := cw.w.OffsetSize()
	lengthSize := cw.w.LengthSize()

	// Calculate bits for number of elements in index block
	// We'll store all elements directly in the index block for simplicity
	idxBlkElmtsBits := uint8(0)
	for (1 << idxBlkElmtsBits) < numChunks {
		idxBlkElmtsBits++
	}
	// h5py commonly uses 2 bits for small arrays
	if idxBlkElmtsBits < 2 {
		idxBlkElmtsBits = 2
	}

	// Max elements bits - enough to address all chunks
	maxElmtsBits := idxBlkElmtsBits
	for (1 << maxElmtsBits) < numChunks {
		maxElmtsBits++
	}
	if maxElmtsBits < 4 {
		maxElmtsBits = 4 // Minimum for reasonable addressability
	}

	// Calculate sizes
	numIdxElmts := 1 << idxBlkElmtsBits
	idxBlockSize := 4 + 1 + 1 + offsetSize + numIdxElmts*elemSize + 4
	headerSize := 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 6*lengthSize + offsetSize + 4

	// Allocate space for both structures
	idxBlockAddr := cw.allocator(int64(idxBlockSize))
	headerAddr := cw.allocator(int64(headerSize))

	// Build index block in memory
	idxData := make([]byte, idxBlockSize)
	idx := 0

	// Signature "EAIB"
	copy(idxData[idx:], []byte("EAIB"))
	idx += 4

	// Version
	idxData[idx] = 0
	idx++

	// Client ID (0 = non-filtered chunks)
	idxData[idx] = 0
	idx++

	// Header address
	putUint64LE(idxData[idx:], headerAddr, offsetSize)
	idx += offsetSize

	// Chunk addresses (elements)
	for _, addr := range chunkAddrs {
		putUint64LE(idxData[idx:], addr, offsetSize)
		idx += offsetSize
	}

	// Pad remaining slots with undefined address
	for i := numChunks; i < numIdxElmts; i++ {
		putUint64LE(idxData[idx:], 0xFFFFFFFFFFFFFFFF, offsetSize)
		idx += offsetSize
	}

	// Compute and add checksum
	idxChecksum := binary.Lookup3Checksum(idxData[:idx])
	putUint32LE(idxData[idx:], idxChecksum)
	idx += 4

	// Write index block
	iw := cw.w.At(int64(idxBlockAddr))
	if err := iw.WriteBytes(idxData); err != nil {
		return 0, err
	}

	// Build header in memory
	hdrData := make([]byte, headerSize)
	idx = 0

	// Signature "EAHD"
	copy(hdrData[idx:], []byte("EAHD"))
	idx += 4

	// Version
	hdrData[idx] = 0
	idx++

	// Client ID (0 = non-filtered chunks)
	hdrData[idx] = 0
	idx++

	// Element size
	hdrData[idx] = uint8(elemSize)
	idx++

	// Max number of elements bits
	hdrData[idx] = maxElmtsBits
	idx++

	// Index block element count bits
	hdrData[idx] = idxBlkElmtsBits
	idx++

	// Data block min element count bits
	hdrData[idx] = 1
	idx++

	// Super block min element count bits
	hdrData[idx] = 0
	idx++

	// Data block page max element count bits
	hdrData[idx] = 0
	idx++

	// Number of secondary blocks (0)
	putUint64LE(hdrData[idx:], 0, lengthSize)
	idx += lengthSize

	// Secondary block size (0)
	putUint64LE(hdrData[idx:], 0, lengthSize)
	idx += lengthSize

	// Number of data blocks (0)
	putUint64LE(hdrData[idx:], 0, lengthSize)
	idx += lengthSize

	// Data block size (0)
	putUint64LE(hdrData[idx:], 0, lengthSize)
	idx += lengthSize

	// Max index set
	putUint64LE(hdrData[idx:], uint64(numChunks-1), lengthSize)
	idx += lengthSize

	// Number of elements
	putUint64LE(hdrData[idx:], uint64(numChunks), lengthSize)
	idx += lengthSize

	// Index block address
	putUint64LE(hdrData[idx:], idxBlockAddr, offsetSize)
	idx += offsetSize

	// Compute and add checksum
	hdrChecksum := binary.Lookup3Checksum(hdrData[:idx])
	putUint32LE(hdrData[idx:], hdrChecksum)
	idx += 4

	// Write header
	hw := cw.w.At(int64(headerAddr))
	if err := hw.WriteBytes(hdrData); err != nil {
		return 0, err
	}

	return headerAddr, nil
}

// Helper functions for building byte arrays
func putUint64LE(b []byte, v uint64, size int) {
	for i := 0; i < size; i++ {
		b[i] = byte(v >> (8 * i))
	}
}

func putUint32LE(b []byte, v uint32) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
}

// SplitIntoChunks splits contiguous data into chunks based on chunk dimensions.
func SplitIntoChunks(data []byte, dataDims []uint64, chunkDims []uint32, elementSize uint32) [][]byte {
	// Calculate number of chunks in each dimension
	numChunksPerDim := make([]uint64, len(dataDims))
	totalChunks := uint64(1)
	for i, dataDim := range dataDims {
		numChunksPerDim[i] = (dataDim + uint64(chunkDims[i]) - 1) / uint64(chunkDims[i])
		totalChunks *= numChunksPerDim[i]
	}

	// For 1D case, simple splitting
	if len(dataDims) == 1 {
		chunkSize := uint64(chunkDims[0]) * uint64(elementSize)
		chunks := make([][]byte, 0, totalChunks)

		for offset := uint64(0); offset < uint64(len(data)); offset += chunkSize {
			end := offset + chunkSize
			if end > uint64(len(data)) {
				end = uint64(len(data))
			}
			chunks = append(chunks, data[offset:end])
		}
		return chunks
	}

	// For multi-dimensional, more complex logic needed
	// For now, return single chunk for simplicity
	return [][]byte{data}
}
