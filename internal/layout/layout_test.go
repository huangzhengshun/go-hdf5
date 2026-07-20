package layout

import (
	"bytes"
	"testing"

	"github.com/huangzhengshun/go-hdf5/internal/binary"
	"github.com/huangzhengshun/go-hdf5/internal/message"
)

type bytesReaderAt []byte

func (b bytesReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if off >= int64(len(b)) {
		return 0, nil
	}
	n := copy(p, b[off:])
	return n, nil
}

func TestCompactRead(t *testing.T) {
	data := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	layoutMsg := &message.DataLayout{
		Class:       message.LayoutCompact,
		CompactData: data,
	}

	dataspace := &message.Dataspace{
		SpaceType:  message.DataspaceSimple,
		Rank:       1,
		Dimensions: []uint64{8},
	}

	datatype := &message.Datatype{
		Class: message.ClassFixedPoint,
		Size:  1,
	}

	compact := NewCompact(layoutMsg, dataspace, datatype)

	if compact.Class() != message.LayoutCompact {
		t.Errorf("expected compact class, got %d", compact.Class())
	}

	if compact.Size() != len(data) {
		t.Errorf("expected size %d, got %d", len(data), compact.Size())
	}

	result, err := compact.Read()
	if err != nil {
		t.Fatalf("Read failed: %v", err)
	}

	if !bytes.Equal(result, data) {
		t.Errorf("data mismatch: got %v, want %v", result, data)
	}

	// Verify it returns a copy
	result[0] = 0xFF
	result2, _ := compact.Read()
	if result2[0] == 0xFF {
		t.Error("Read should return a copy, not the original slice")
	}
}

func TestContiguousRead(t *testing.T) {
	// Create fake file data with contiguous storage
	fileData := make(bytesReaderAt, 1024)
	// Put data at offset 100
	dataOffset := int64(100)
	testData := []byte{10, 20, 30, 40, 50, 60, 70, 80}
	copy(fileData[dataOffset:], testData)

	reader := binary.NewReader(fileData, binary.Config{
		ByteOrder:  nil, // Use default
		OffsetSize: 8,
		LengthSize: 8,
	})

	layoutMsg := &message.DataLayout{
		Class:   message.LayoutContiguous,
		Address: uint64(dataOffset),
		Size:    uint64(len(testData)),
	}

	dataspace := &message.Dataspace{
		SpaceType:  message.DataspaceSimple,
		Rank:       1,
		Dimensions: []uint64{8},
	}

	datatype := &message.Datatype{
		Class: message.ClassFixedPoint,
		Size:  1,
	}

	contiguous := NewContiguous(layoutMsg, dataspace, datatype, reader)

	if contiguous.Class() != message.LayoutContiguous {
		t.Errorf("expected contiguous class, got %d", contiguous.Class())
	}

	if contiguous.Address() != uint64(dataOffset) {
		t.Errorf("expected address %d, got %d", dataOffset, contiguous.Address())
	}

	if contiguous.Size() != uint64(len(testData)) {
		t.Errorf("expected size %d, got %d", len(testData), contiguous.Size())
	}

	result, err := contiguous.Read()
	if err != nil {
		t.Fatalf("Read failed: %v", err)
	}

	if !bytes.Equal(result, testData) {
		t.Errorf("data mismatch: got %v, want %v", result, testData)
	}
}

func TestContiguousSizeFromDataspace(t *testing.T) {
	fileData := make(bytesReaderAt, 1024)

	reader := binary.NewReader(fileData, binary.Config{
		OffsetSize: 8,
		LengthSize: 8,
	})

	// Layout with no explicit size
	layoutMsg := &message.DataLayout{
		Class:   message.LayoutContiguous,
		Address: 100,
		Size:    0, // Will be calculated
	}

	dataspace := &message.Dataspace{
		SpaceType:  message.DataspaceSimple,
		Rank:       1,
		Dimensions: []uint64{10},
	}

	datatype := &message.Datatype{
		Class: message.ClassFixedPoint,
		Size:  4, // 4 bytes per element
	}

	contiguous := NewContiguous(layoutMsg, dataspace, datatype, reader)

	// Size should be calculated as 10 * 4 = 40
	if contiguous.Size() != 40 {
		t.Errorf("expected size 40, got %d", contiguous.Size())
	}
}

func TestCalculateDataSize(t *testing.T) {
	tests := []struct {
		name      string
		dataspace *message.Dataspace
		datatype  *message.Datatype
		expected  uint64
	}{
		{
			name:      "nil dataspace",
			dataspace: nil,
			datatype:  &message.Datatype{Size: 4},
			expected:  0,
		},
		{
			name:      "nil datatype",
			dataspace: &message.Dataspace{SpaceType: message.DataspaceSimple, Dimensions: []uint64{10}},
			datatype:  nil,
			expected:  0,
		},
		{
			name:      "scalar",
			dataspace: &message.Dataspace{SpaceType: message.DataspaceScalar},
			datatype:  &message.Datatype{Size: 8},
			expected:  8,
		},
		{
			name:      "1D",
			dataspace: &message.Dataspace{SpaceType: message.DataspaceSimple, Dimensions: []uint64{100}},
			datatype:  &message.Datatype{Size: 4},
			expected:  400,
		},
		{
			name:      "2D",
			dataspace: &message.Dataspace{SpaceType: message.DataspaceSimple, Dimensions: []uint64{10, 20}},
			datatype:  &message.Datatype{Size: 8},
			expected:  1600,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := calculateDataSize(tt.dataspace, tt.datatype)
			if result != tt.expected {
				t.Errorf("expected %d, got %d", tt.expected, result)
			}
		})
	}
}
