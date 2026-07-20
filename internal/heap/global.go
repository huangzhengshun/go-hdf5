package heap

import (
	"fmt"

	"github.com/huangzhengshun/go-hdf5/internal/binary"
)

// GlobalHeap represents an HDF5 global heap collection.
// Global heaps store variable-length data like variable-length strings.
type GlobalHeap struct {
	CollectionSize uint64
	objects        map[uint16][]byte // index -> object data
}

// GlobalHeapID represents a reference to an object in the global heap.
// This is stored in variable-length data fields.
type GlobalHeapID struct {
	CollectionAddress uint64 // Address of the global heap collection
	ObjectIndex       uint32 // Index of the object within the collection
}

// ReadGlobalHeap reads a global heap collection at the given address.
func ReadGlobalHeap(r *binary.Reader, address uint64) (*GlobalHeap, error) {
	if address == 0 || address == 0xFFFFFFFFFFFFFFFF {
		return nil, fmt.Errorf("invalid global heap address")
	}

	hr := r.At(int64(address))

	// Check signature "GCOL"
	sig, err := hr.ReadBytes(4)
	if err != nil {
		return nil, fmt.Errorf("reading global heap signature: %w", err)
	}
	if string(sig) != "GCOL" {
		return nil, fmt.Errorf("invalid global heap signature: %q", string(sig))
	}

	// Version (1 byte)
	version, err := hr.ReadUint8()
	if err != nil {
		return nil, err
	}
	if version != 1 {
		return nil, fmt.Errorf("unsupported global heap version: %d", version)
	}

	// Reserved (3 bytes)
	hr.Skip(3)

	// Collection size (length-sized)
	collectionSize, err := hr.ReadLength()
	if err != nil {
		return nil, err
	}

	heap := &GlobalHeap{
		CollectionSize: collectionSize,
		objects:        make(map[uint16][]byte),
	}

	// Read objects until we hit index 0 or run out of collection space
	// The collection size includes the header (signature + version + reserved + size)
	headerSize := uint64(4 + 1 + 3 + r.LengthSize())
	remainingSize := collectionSize - headerSize

	for remainingSize > 0 {
		// Heap Object Index (2 bytes)
		index, err := hr.ReadUint16()
		if err != nil {
			break
		}

		// Index 0 marks end of objects
		if index == 0 {
			break
		}

		// Reference Count (2 bytes) - we don't use this for reading
		_, err = hr.ReadUint16()
		if err != nil {
			break
		}

		// Reserved (4 bytes)
		hr.Skip(4)

		// Object Size (length-sized)
		objectSize, err := hr.ReadLength()
		if err != nil {
			break
		}

		// Object Data
		if objectSize > 0 {
			data, err := hr.ReadBytes(int(objectSize))
			if err != nil {
				break
			}
			heap.objects[index] = data
		}

		// Objects are padded to 8-byte boundaries
		padding := (8 - (objectSize % 8)) % 8
		hr.Skip(int64(padding))

		// Calculate how much we consumed
		// 2 (index) + 2 (refcount) + 4 (reserved) + lengthSize + objectSize + padding
		consumed := uint64(2 + 2 + 4 + r.LengthSize()) + objectSize + padding
		if consumed > remainingSize {
			break
		}
		remainingSize -= consumed
	}

	return heap, nil
}

// GetObject retrieves an object by index from the global heap.
func (h *GlobalHeap) GetObject(index uint16) ([]byte, error) {
	if h == nil {
		return nil, fmt.Errorf("nil global heap")
	}
	data, ok := h.objects[index]
	if !ok {
		return nil, fmt.Errorf("object index %d not found in global heap", index)
	}
	// Return a copy to prevent modification
	result := make([]byte, len(data))
	copy(result, data)
	return result, nil
}

// GetString retrieves a null-terminated string from the global heap.
func (h *GlobalHeap) GetString(index uint16) (string, error) {
	data, err := h.GetObject(index)
	if err != nil {
		return "", err
	}

	// Find null terminator
	end := len(data)
	for i := 0; i < len(data); i++ {
		if data[i] == 0 {
			end = i
			break
		}
	}

	return string(data[:end]), nil
}

// ParseGlobalHeapID parses a global heap ID from raw bytes.
// The format is: collection address (offset-sized) + object index (4 bytes)
func ParseGlobalHeapID(data []byte, offsetSize int) (GlobalHeapID, error) {
	if len(data) < offsetSize+4 {
		return GlobalHeapID{}, fmt.Errorf("global heap ID too short: need %d bytes, have %d", offsetSize+4, len(data))
	}

	var addr uint64
	switch offsetSize {
	case 2:
		addr = uint64(data[0]) | uint64(data[1])<<8
	case 4:
		addr = uint64(data[0]) | uint64(data[1])<<8 | uint64(data[2])<<16 | uint64(data[3])<<24
	case 8:
		addr = uint64(data[0]) | uint64(data[1])<<8 | uint64(data[2])<<16 | uint64(data[3])<<24 |
			uint64(data[4])<<32 | uint64(data[5])<<40 | uint64(data[6])<<48 | uint64(data[7])<<56
	default:
		return GlobalHeapID{}, fmt.Errorf("unsupported offset size: %d", offsetSize)
	}

	index := uint32(data[offsetSize]) | uint32(data[offsetSize+1])<<8 |
		uint32(data[offsetSize+2])<<16 | uint32(data[offsetSize+3])<<24

	return GlobalHeapID{
		CollectionAddress: addr,
		ObjectIndex:       index,
	}, nil
}
