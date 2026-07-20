package heap

import (
	"github.com/huangzhengshun/go-hdf5/internal/binary"
)

// GlobalHeapWriter helps write global heap collections.
type GlobalHeapWriter struct {
	w         *binary.Writer
	allocator func(size int64) uint64
	objects   [][]byte
}

// NewGlobalHeapWriter creates a new global heap writer.
func NewGlobalHeapWriter(w *binary.Writer, allocator func(size int64) uint64) *GlobalHeapWriter {
	return &GlobalHeapWriter{
		w:         w,
		allocator: allocator,
		objects:   nil,
	}
}

// AddObject adds an object to the heap and returns its index.
// Objects are 1-indexed (0 is reserved for end marker).
func (ghw *GlobalHeapWriter) AddObject(data []byte) uint16 {
	ghw.objects = append(ghw.objects, data)
	return uint16(len(ghw.objects)) // 1-indexed
}

// AddString adds a null-terminated string to the heap.
func (ghw *GlobalHeapWriter) AddString(s string) uint16 {
	// Add null terminator
	data := make([]byte, len(s)+1)
	copy(data, s)
	data[len(s)] = 0
	return ghw.AddObject(data)
}

// Write writes all objects to a new global heap collection.
// Returns the address of the heap and a map of object index to GlobalHeapID.
func (ghw *GlobalHeapWriter) Write() (uint64, map[uint16]GlobalHeapID, error) {
	if len(ghw.objects) == 0 {
		return 0, nil, nil
	}

	// Calculate collection size
	// Header: signature(4) + version(1) + reserved(3) + collectionSize(lengthSize)
	headerSize := 4 + 1 + 3 + ghw.w.LengthSize()

	// Objects size
	objectsSize := 0
	for _, obj := range ghw.objects {
		// Object header: index(2) + refcount(2) + reserved(4) + size(lengthSize)
		objHeaderSize := 2 + 2 + 4 + ghw.w.LengthSize()
		// Object data + padding to 8-byte boundary
		dataSize := len(obj)
		padding := (8 - (dataSize % 8)) % 8
		objectsSize += objHeaderSize + dataSize + padding
	}

	// End marker: index(2) = 0
	endMarkerSize := 2

	// Padding to make total collection size 8-byte aligned
	totalSize := headerSize + objectsSize + endMarkerSize
	collectionPadding := (8 - (totalSize % 8)) % 8
	collectionSize := totalSize + collectionPadding

	// Allocate space
	heapAddr := ghw.allocator(int64(collectionSize))

	// Write the heap
	w := ghw.w.At(int64(heapAddr))

	// Signature "GCOL"
	if err := w.WriteBytes([]byte("GCOL")); err != nil {
		return 0, nil, err
	}

	// Version 1
	if err := w.WriteUint8(1); err != nil {
		return 0, nil, err
	}

	// Reserved (3 bytes)
	if err := w.WriteBytes([]byte{0, 0, 0}); err != nil {
		return 0, nil, err
	}

	// Collection size
	if err := w.WriteLength(uint64(collectionSize)); err != nil {
		return 0, nil, err
	}

	// Write objects
	heapIDs := make(map[uint16]GlobalHeapID)
	for i, obj := range ghw.objects {
		index := uint16(i + 1) // 1-indexed

		// Heap Object Index
		if err := w.WriteUint16(index); err != nil {
			return 0, nil, err
		}

		// Reference Count (1 for single reference)
		if err := w.WriteUint16(1); err != nil {
			return 0, nil, err
		}

		// Reserved (4 bytes)
		if err := w.WriteBytes([]byte{0, 0, 0, 0}); err != nil {
			return 0, nil, err
		}

		// Object Size
		if err := w.WriteLength(uint64(len(obj))); err != nil {
			return 0, nil, err
		}

		// Object Data
		if err := w.WriteBytes(obj); err != nil {
			return 0, nil, err
		}

		// Padding to 8-byte boundary
		padding := (8 - (len(obj) % 8)) % 8
		if padding > 0 {
			if err := w.WriteBytes(make([]byte, padding)); err != nil {
				return 0, nil, err
			}
		}

		heapIDs[index] = GlobalHeapID{
			CollectionAddress: heapAddr,
			ObjectIndex:       uint32(index),
		}
	}

	// End marker (index 0)
	if err := w.WriteUint16(0); err != nil {
		return 0, nil, err
	}

	// Collection padding
	if collectionPadding > 0 {
		if err := w.WriteBytes(make([]byte, collectionPadding)); err != nil {
			return 0, nil, err
		}
	}

	return heapAddr, heapIDs, nil
}

// WriteGlobalHeapID writes a global heap ID to the writer.
// Format: collection address (offset-sized) + object index (4 bytes)
func WriteGlobalHeapID(w *binary.Writer, id GlobalHeapID) error {
	if err := w.WriteOffset(id.CollectionAddress); err != nil {
		return err
	}
	if err := w.WriteUint32(id.ObjectIndex); err != nil {
		return err
	}
	return nil
}

// GlobalHeapIDSize returns the size of a global heap ID.
func GlobalHeapIDSize(offsetSize int) int {
	return offsetSize + 4
}
