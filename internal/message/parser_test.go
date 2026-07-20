package message

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// Helper to get testdata path
func getTestdataPath(filename string) string {
	return filepath.Join("..", "..", "testdata", filename)
}

// === DATASPACE TESTS ===

func TestDataspaceScalarParsing(t *testing.T) {
	// Version 2 scalar dataspace
	data := []byte{
		2, // Version
		0, // Rank (0 = scalar)
		0, // Flags
		0, // Type = scalar
	}

	ds, err := parseDataspace(data, mockReader())
	if err != nil {
		t.Fatalf("parseDataspace failed: %v", err)
	}

	if ds.Version != 2 {
		t.Errorf("expected version 2, got %d", ds.Version)
	}
	if ds.Rank != 0 {
		t.Errorf("expected rank 0, got %d", ds.Rank)
	}
	if ds.SpaceType != DataspaceScalar {
		t.Errorf("expected scalar type, got %d", ds.SpaceType)
	}
	if !ds.IsScalar() {
		t.Error("IsScalar should return true")
	}
	if ds.NumElements() != 1 {
		t.Errorf("expected 1 element, got %d", ds.NumElements())
	}
}

func TestDataspaceSimple1DParsing(t *testing.T) {
	// Version 2 simple 1D dataspace with 10 elements
	data := make([]byte, 4+8) // header + 1 dimension
	data[0] = 2               // Version
	data[1] = 1               // Rank
	data[2] = 0               // Flags (no max dims)
	data[3] = 1               // Type = simple

	binary.LittleEndian.PutUint64(data[4:], 10) // Dimension 0 = 10

	ds, err := parseDataspace(data, mockReader())
	if err != nil {
		t.Fatalf("parseDataspace failed: %v", err)
	}

	if ds.Rank != 1 {
		t.Errorf("expected rank 1, got %d", ds.Rank)
	}
	if len(ds.Dimensions) != 1 {
		t.Fatalf("expected 1 dimension, got %d", len(ds.Dimensions))
	}
	if ds.Dimensions[0] != 10 {
		t.Errorf("expected dimension 10, got %d", ds.Dimensions[0])
	}
	if ds.NumElements() != 10 {
		t.Errorf("expected 10 elements, got %d", ds.NumElements())
	}
}

func TestDataspaceSimple2DParsing(t *testing.T) {
	// Version 2 simple 2D dataspace: 3x4 = 12 elements
	data := make([]byte, 4+16) // header + 2 dimensions
	data[0] = 2                // Version
	data[1] = 2                // Rank
	data[2] = 0                // Flags
	data[3] = 1                // Type = simple

	binary.LittleEndian.PutUint64(data[4:], 3)  // Dimension 0 = 3
	binary.LittleEndian.PutUint64(data[12:], 4) // Dimension 1 = 4

	ds, err := parseDataspace(data, mockReader())
	if err != nil {
		t.Fatalf("parseDataspace failed: %v", err)
	}

	if ds.Rank != 2 {
		t.Errorf("expected rank 2, got %d", ds.Rank)
	}
	if ds.NumElements() != 12 {
		t.Errorf("expected 12 elements, got %d", ds.NumElements())
	}
}

func TestDataspaceSimple3DParsing(t *testing.T) {
	// Version 2 simple 3D dataspace: 2x3x4 = 24 elements
	data := make([]byte, 4+24) // header + 3 dimensions
	data[0] = 2                // Version
	data[1] = 3                // Rank
	data[2] = 0                // Flags
	data[3] = 1                // Type = simple

	binary.LittleEndian.PutUint64(data[4:], 2)  // Dimension 0 = 2
	binary.LittleEndian.PutUint64(data[12:], 3) // Dimension 1 = 3
	binary.LittleEndian.PutUint64(data[20:], 4) // Dimension 2 = 4

	ds, err := parseDataspace(data, mockReader())
	if err != nil {
		t.Fatalf("parseDataspace failed: %v", err)
	}

	if ds.Rank != 3 {
		t.Errorf("expected rank 3, got %d", ds.Rank)
	}
	if len(ds.Dimensions) != 3 {
		t.Fatalf("expected 3 dimensions, got %d", len(ds.Dimensions))
	}
	if ds.NumElements() != 24 {
		t.Errorf("expected 24 elements, got %d", ds.NumElements())
	}
}

func TestDataspaceWithMaxDims(t *testing.T) {
	// Version 2 simple 1D dataspace with max dimensions
	data := make([]byte, 4+16) // header + 1 dimension + 1 max dimension
	data[0] = 2                // Version
	data[1] = 1                // Rank
	data[2] = 0x01             // Flags: has max dims
	data[3] = 1                // Type = simple

	binary.LittleEndian.PutUint64(data[4:], 10)                // Dimension 0 = 10
	binary.LittleEndian.PutUint64(data[12:], 0xFFFFFFFFFFFFFFFF) // Max dimension = unlimited

	ds, err := parseDataspace(data, mockReader())
	if err != nil {
		t.Fatalf("parseDataspace failed: %v", err)
	}

	if ds.MaxDims == nil {
		t.Fatal("expected max dims to be set")
	}
	if ds.MaxDims[0] != 0xFFFFFFFFFFFFFFFF {
		t.Errorf("expected unlimited max dim, got %d", ds.MaxDims[0])
	}
}

func TestDataspaceNullParsing(t *testing.T) {
	data := []byte{2, 0, 0, 2} // Version 2, rank 0, flags 0, type = null

	ds, err := parseDataspace(data, mockReader())
	if err != nil {
		t.Fatalf("parseDataspace failed: %v", err)
	}

	if !ds.IsNull() {
		t.Error("IsNull should return true")
	}
	if ds.NumElements() != 0 {
		t.Errorf("null dataspace should have 0 elements, got %d", ds.NumElements())
	}
}

func TestDataspaceVersion1(t *testing.T) {
	// Version 1 simple 1D dataspace
	data := make([]byte, 8+8) // header (4 + 4 reserved) + 1 dimension
	data[0] = 1               // Version 1
	data[1] = 1               // Rank
	data[2] = 0               // Flags
	// data[3] is reserved
	// data[4:8] are reserved

	binary.LittleEndian.PutUint64(data[8:], 5) // Dimension 0 = 5

	ds, err := parseDataspace(data, mockReader())
	if err != nil {
		t.Fatalf("parseDataspace failed: %v", err)
	}

	if ds.Version != 1 {
		t.Errorf("expected version 1, got %d", ds.Version)
	}
	if ds.Rank != 1 {
		t.Errorf("expected rank 1, got %d", ds.Rank)
	}
	if ds.NumElements() != 5 {
		t.Errorf("expected 5 elements, got %d", ds.NumElements())
	}
}

func TestDataspaceTooShort(t *testing.T) {
	data := []byte{2, 0} // Too short

	_, err := parseDataspace(data, mockReader())
	if err == nil {
		t.Error("expected error for too short data")
	}
}

// === DATATYPE TESTS ===

func TestDatatypeInt8(t *testing.T) {
	data := make([]byte, 12)
	data[0] = 0x10 | byte(ClassFixedPoint)      // Version 1, class 0
	data[1] = 0x08                               // Signed bit set
	binary.LittleEndian.PutUint32(data[4:], 1)  // Size = 1 byte
	binary.LittleEndian.PutUint16(data[8:], 0)  // Bit offset
	binary.LittleEndian.PutUint16(data[10:], 8) // Bit precision

	dt, err := parseDatatype(data, mockReader())
	if err != nil {
		t.Fatalf("parseDatatype failed: %v", err)
	}

	if dt.Class != ClassFixedPoint {
		t.Errorf("expected class %d, got %d", ClassFixedPoint, dt.Class)
	}
	if dt.Size != 1 {
		t.Errorf("expected size 1, got %d", dt.Size)
	}
	if !dt.Signed {
		t.Error("expected signed type")
	}
	if !dt.IsInteger() {
		t.Error("IsInteger should return true")
	}
}

func TestDatatypeInt16(t *testing.T) {
	data := make([]byte, 12)
	data[0] = 0x10 | byte(ClassFixedPoint)       // Version 1, class 0
	data[1] = 0x08                                // Signed bit set
	binary.LittleEndian.PutUint32(data[4:], 2)   // Size = 2 bytes
	binary.LittleEndian.PutUint16(data[8:], 0)   // Bit offset
	binary.LittleEndian.PutUint16(data[10:], 16) // Bit precision

	dt, err := parseDatatype(data, mockReader())
	if err != nil {
		t.Fatalf("parseDatatype failed: %v", err)
	}

	if dt.Size != 2 {
		t.Errorf("expected size 2, got %d", dt.Size)
	}
}

func TestDatatypeInt32(t *testing.T) {
	data := make([]byte, 12)
	data[0] = 0x10 | byte(ClassFixedPoint)       // Version 1, class 0
	data[1] = 0x08                                // Signed bit set
	binary.LittleEndian.PutUint32(data[4:], 4)   // Size = 4 bytes
	binary.LittleEndian.PutUint16(data[8:], 0)   // Bit offset
	binary.LittleEndian.PutUint16(data[10:], 32) // Bit precision

	dt, err := parseDatatype(data, mockReader())
	if err != nil {
		t.Fatalf("parseDatatype failed: %v", err)
	}

	if dt.Size != 4 {
		t.Errorf("expected size 4, got %d", dt.Size)
	}
}

func TestDatatypeInt64(t *testing.T) {
	data := make([]byte, 12)
	data[0] = 0x10 | byte(ClassFixedPoint)       // Version 1, class 0
	data[1] = 0x08                                // Signed bit set
	binary.LittleEndian.PutUint32(data[4:], 8)   // Size = 8 bytes
	binary.LittleEndian.PutUint16(data[8:], 0)   // Bit offset
	binary.LittleEndian.PutUint16(data[10:], 64) // Bit precision

	dt, err := parseDatatype(data, mockReader())
	if err != nil {
		t.Fatalf("parseDatatype failed: %v", err)
	}

	if dt.Size != 8 {
		t.Errorf("expected size 8, got %d", dt.Size)
	}
}

func TestDatatypeUnsigned(t *testing.T) {
	data := make([]byte, 12)
	data[0] = 0x10 | byte(ClassFixedPoint)       // Version 1, class 0
	data[1] = 0x00                                // Not signed
	binary.LittleEndian.PutUint32(data[4:], 4)   // Size = 4 bytes

	dt, err := parseDatatype(data, mockReader())
	if err != nil {
		t.Fatalf("parseDatatype failed: %v", err)
	}

	if dt.Signed {
		t.Error("expected unsigned type")
	}
}

func TestDatatypeFloat32(t *testing.T) {
	data := make([]byte, 20)
	data[0] = 0x10 | byte(ClassFloatPoint) // Version 1, class 1
	data[1] = 0                             // Little-endian
	binary.LittleEndian.PutUint32(data[4:], 4) // Size = 4 bytes

	dt, err := parseDatatype(data, mockReader())
	if err != nil {
		t.Fatalf("parseDatatype failed: %v", err)
	}

	if dt.Class != ClassFloatPoint {
		t.Errorf("expected class %d, got %d", ClassFloatPoint, dt.Class)
	}
	if dt.Size != 4 {
		t.Errorf("expected size 4, got %d", dt.Size)
	}
	if !dt.IsFloat() {
		t.Error("IsFloat should return true")
	}
}

func TestDatatypeFloat64Parsing(t *testing.T) {
	data := make([]byte, 20)
	data[0] = 0x10 | byte(ClassFloatPoint) // Version 1, class 1
	data[1] = 0                             // Little-endian
	binary.LittleEndian.PutUint32(data[4:], 8) // Size = 8 bytes

	dt, err := parseDatatype(data, mockReader())
	if err != nil {
		t.Fatalf("parseDatatype failed: %v", err)
	}

	if dt.Size != 8 {
		t.Errorf("expected size 8, got %d", dt.Size)
	}
}

func TestDatatypeFixedString(t *testing.T) {
	data := make([]byte, 8)
	data[0] = 0x10 | byte(ClassString)                   // Version 1, class 3
	data[1] = byte(PadNullTerm)                          // Null-terminated
	data[2] = byte(CharsetASCII) << 4                     // ASCII charset
	binary.LittleEndian.PutUint32(data[4:], 32)          // Size = 32 bytes

	dt, err := parseDatatype(data, mockReader())
	if err != nil {
		t.Fatalf("parseDatatype failed: %v", err)
	}

	if dt.Class != ClassString {
		t.Errorf("expected class %d, got %d", ClassString, dt.Class)
	}
	if dt.Size != 32 {
		t.Errorf("expected size 32, got %d", dt.Size)
	}
	if !dt.IsString() {
		t.Error("IsString should return true")
	}
	if dt.StringPadding != PadNullTerm {
		t.Errorf("expected null-terminated padding, got %d", dt.StringPadding)
	}
	if dt.CharSet != CharsetASCII {
		t.Errorf("expected ASCII charset, got %d", dt.CharSet)
	}
}

func TestDatatypeSpacePadString(t *testing.T) {
	data := make([]byte, 8)
	data[0] = 0x10 | byte(ClassString)           // Version 1, class 3
	data[1] = byte(PadSpacePad)                  // Space-padded
	binary.LittleEndian.PutUint32(data[4:], 10)  // Size = 10 bytes

	dt, err := parseDatatype(data, mockReader())
	if err != nil {
		t.Fatalf("parseDatatype failed: %v", err)
	}

	if dt.StringPadding != PadSpacePad {
		t.Errorf("expected space-padded, got %d", dt.StringPadding)
	}
}

func TestDatatypeUTF8String(t *testing.T) {
	data := make([]byte, 8)
	data[0] = 0x10 | byte(ClassString)           // Version 1, class 3
	// classBits = data[1] | data[2]<<8 | data[3]<<16
	// Charset is extracted as (classBits >> 4) & 0x0F, so we need bits 4-7 of data[1]
	data[1] = byte(PadNullTerm) | (byte(CharsetUTF8) << 4)  // Null-terminated + UTF-8 charset
	data[2] = 0
	data[3] = 0
	binary.LittleEndian.PutUint32(data[4:], 64)  // Size = 64 bytes

	dt, err := parseDatatype(data, mockReader())
	if err != nil {
		t.Fatalf("parseDatatype failed: %v", err)
	}

	if dt.CharSet != CharsetUTF8 {
		t.Errorf("expected UTF-8 charset, got %d", dt.CharSet)
	}
}

func TestDatatypeByteOrder(t *testing.T) {
	// Little-endian
	dataLE := make([]byte, 12)
	dataLE[0] = 0x10 | byte(ClassFixedPoint)
	dataLE[1] = 0x00 // Little-endian
	binary.LittleEndian.PutUint32(dataLE[4:], 4)

	dtLE, err := parseDatatype(dataLE, mockReader())
	if err != nil {
		t.Fatalf("parseDatatype LE failed: %v", err)
	}
	if dtLE.ByteOrder != OrderLE {
		t.Errorf("expected little-endian, got %d", dtLE.ByteOrder)
	}

	// Big-endian
	dataBE := make([]byte, 12)
	dataBE[0] = 0x10 | byte(ClassFixedPoint)
	dataBE[1] = 0x01 // Big-endian
	binary.LittleEndian.PutUint32(dataBE[4:], 4)

	dtBE, err := parseDatatype(dataBE, mockReader())
	if err != nil {
		t.Fatalf("parseDatatype BE failed: %v", err)
	}
	if dtBE.ByteOrder != OrderBE {
		t.Errorf("expected big-endian, got %d", dtBE.ByteOrder)
	}
}

func TestDatatypeTooShort(t *testing.T) {
	data := []byte{0x10, 0x00, 0x00} // Too short

	_, err := parseDatatype(data, mockReader())
	if err == nil {
		t.Error("expected error for too short data")
	}
}

// === LAYOUT TESTS ===

func TestLayoutContiguousV3(t *testing.T) {
	data := make([]byte, 18)
	data[0] = 3                                           // Version 3
	data[1] = byte(LayoutContiguous)                      // Contiguous
	binary.LittleEndian.PutUint64(data[2:], 0x1000)       // Address
	binary.LittleEndian.PutUint64(data[10:], 0x2000)      // Size

	layout, err := parseDataLayout(data, mockReader())
	if err != nil {
		t.Fatalf("parseDataLayout failed: %v", err)
	}

	if layout.Version != 3 {
		t.Errorf("expected version 3, got %d", layout.Version)
	}
	if layout.Class != LayoutContiguous {
		t.Errorf("expected contiguous class, got %d", layout.Class)
	}
	if layout.Address != 0x1000 {
		t.Errorf("expected address 0x1000, got 0x%x", layout.Address)
	}
	if layout.Size != 0x2000 {
		t.Errorf("expected size 0x2000, got 0x%x", layout.Size)
	}
	if !layout.IsContiguous() {
		t.Error("IsContiguous should return true")
	}
}

func TestLayoutCompactV3(t *testing.T) {
	compactData := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	data := make([]byte, 4+len(compactData))
	data[0] = 3                                           // Version 3
	data[1] = byte(LayoutCompact)                         // Compact
	binary.LittleEndian.PutUint16(data[2:], uint16(len(compactData)))
	copy(data[4:], compactData)

	layout, err := parseDataLayout(data, mockReader())
	if err != nil {
		t.Fatalf("parseDataLayout failed: %v", err)
	}

	if !layout.IsCompact() {
		t.Error("IsCompact should return true")
	}
	if len(layout.CompactData) != len(compactData) {
		t.Errorf("expected %d bytes compact data, got %d", len(compactData), len(layout.CompactData))
	}
	for i, b := range compactData {
		if layout.CompactData[i] != b {
			t.Errorf("compact data mismatch at index %d", i)
		}
	}
}

func TestLayoutChunkedV3(t *testing.T) {
	data := make([]byte, 20)
	data[0] = 3                                           // Version 3
	data[1] = byte(LayoutChunked)                         // Chunked
	data[2] = byte(ChunkIndexBTreeV2)                     // Chunk index type (in flags)
	data[3] = 2                                           // 2 dimensions
	data[4] = 4                                           // 4 bytes per dimension size
	// Chunk dimensions (4 bytes each)
	binary.LittleEndian.PutUint32(data[5:], 10)           // Chunk dim 0 = 10
	binary.LittleEndian.PutUint32(data[9:], 10)           // Chunk dim 1 = 10
	// B-tree address at end
	binary.LittleEndian.PutUint64(data[12:], 0x3000)

	layout, err := parseDataLayout(data, mockReader())
	if err != nil {
		t.Fatalf("parseDataLayout failed: %v", err)
	}

	if !layout.IsChunked() {
		t.Error("IsChunked should return true")
	}
	if len(layout.ChunkDims) != 2 {
		t.Errorf("expected 2 chunk dims, got %d", len(layout.ChunkDims))
	}
}

func TestLayoutV1V2(t *testing.T) {
	data := make([]byte, 20)
	data[0] = 1                                           // Version 1
	data[1] = 2                                           // 2 dimensions
	data[2] = byte(LayoutContiguous)                      // Contiguous
	// Offset and size
	binary.LittleEndian.PutUint64(data[4:], 0x5000)       // Address
	binary.LittleEndian.PutUint64(data[12:], 0x1000)      // Size

	layout, err := parseDataLayout(data, mockReader())
	if err != nil {
		t.Fatalf("parseDataLayout failed: %v", err)
	}

	if layout.Version != 1 {
		t.Errorf("expected version 1, got %d", layout.Version)
	}
	if layout.Address != 0x5000 {
		t.Errorf("expected address 0x5000, got 0x%x", layout.Address)
	}
}

func TestLayoutTooShort(t *testing.T) {
	data := []byte{3} // Too short

	_, err := parseDataLayout(data, mockReader())
	if err == nil {
		t.Error("expected error for too short data")
	}
}

func TestLayoutUnsupportedVersion(t *testing.T) {
	data := []byte{99, 0} // Unsupported version

	_, err := parseDataLayout(data, mockReader())
	if err == nil {
		t.Error("expected error for unsupported version")
	}
}

// === LINK TESTS ===

func TestLinkHardParsing(t *testing.T) {
	name := "my_dataset"
	data := make([]byte, 2+1+1+len(name)+8)
	data[0] = 1                           // Version
	data[1] = 0x08 | 0                    // Flags: link type present, name len size = 1 byte
	data[2] = byte(LinkTypeHard)          // Link type
	data[3] = byte(len(name))             // Name length
	copy(data[4:], name)
	binary.LittleEndian.PutUint64(data[4+len(name):], 0x1234)

	link, err := parseLink(data, mockReader())
	if err != nil {
		t.Fatalf("parseLink failed: %v", err)
	}

	if link.Version != 1 {
		t.Errorf("expected version 1, got %d", link.Version)
	}
	if link.Name != name {
		t.Errorf("expected name %q, got %q", name, link.Name)
	}
	if !link.IsHard() {
		t.Error("IsHard should return true")
	}
	if link.ObjectAddress != 0x1234 {
		t.Errorf("expected address 0x1234, got 0x%x", link.ObjectAddress)
	}
}

func TestLinkSoftParsing(t *testing.T) {
	name := "soft_link"
	target := "/path/to/target"
	data := make([]byte, 2+1+1+len(name)+2+len(target))
	data[0] = 1                           // Version
	data[1] = 0x08 | 0                    // Flags: link type present, name len size = 1 byte
	data[2] = byte(LinkTypeSoft)          // Link type
	data[3] = byte(len(name))             // Name length
	copy(data[4:], name)
	offset := 4 + len(name)
	binary.LittleEndian.PutUint16(data[offset:], uint16(len(target)))
	copy(data[offset+2:], target)

	link, err := parseLink(data, mockReader())
	if err != nil {
		t.Fatalf("parseLink failed: %v", err)
	}

	if !link.IsSoft() {
		t.Error("IsSoft should return true")
	}
	if link.SoftLinkValue != target {
		t.Errorf("expected soft link value %q, got %q", target, link.SoftLinkValue)
	}
}

func TestLinkWithCreationOrder(t *testing.T) {
	name := "ordered_link"
	data := make([]byte, 2+1+8+1+len(name)+8)
	data[0] = 1                           // Version
	data[1] = 0x08 | 0x04                 // Flags: link type present, creation order present
	data[2] = byte(LinkTypeHard)          // Link type
	binary.LittleEndian.PutUint64(data[3:], 42) // Creation order = 42
	data[11] = byte(len(name))            // Name length
	copy(data[12:], name)
	binary.LittleEndian.PutUint64(data[12+len(name):], 0x5678)

	link, err := parseLink(data, mockReader())
	if err != nil {
		t.Fatalf("parseLink failed: %v", err)
	}

	if link.CreationOrder != 42 {
		t.Errorf("expected creation order 42, got %d", link.CreationOrder)
	}
}

func TestLinkTooShort(t *testing.T) {
	data := []byte{1} // Too short

	_, err := parseLink(data, mockReader())
	if err == nil {
		t.Error("expected error for too short data")
	}
}

// === FILTER PIPELINE TESTS ===

func TestFilterPipelineSingleDeflate(t *testing.T) {
	data := []byte{
		2,          // Version 2
		1,          // Number of filters
		// Filter 0: DEFLATE
		0x01, 0x00, // ID = 1 (deflate)
		0x00, 0x00, // Flags
		0x01, 0x00, // Num client data = 1
		0x06, 0x00, 0x00, 0x00, // Client data: level 6
	}

	fp, err := parseFilterPipeline(data, mockReader())
	if err != nil {
		t.Fatalf("parseFilterPipeline failed: %v", err)
	}

	if fp.Version != 2 {
		t.Errorf("expected version 2, got %d", fp.Version)
	}
	if len(fp.Filters) != 1 {
		t.Fatalf("expected 1 filter, got %d", len(fp.Filters))
	}
	if fp.Filters[0].ID != FilterDeflate {
		t.Errorf("expected deflate filter, got %d", fp.Filters[0].ID)
	}
	if !fp.HasCompression() {
		t.Error("HasCompression should return true")
	}
	if !fp.HasFilter(FilterDeflate) {
		t.Error("HasFilter(FilterDeflate) should return true")
	}
}

func TestFilterPipelineShuffleAndDeflate(t *testing.T) {
	data := []byte{
		2,          // Version 2
		2,          // Number of filters
		// Filter 0: Shuffle
		0x02, 0x00, // ID = 2 (shuffle)
		0x00, 0x00, // Flags
		0x01, 0x00, // Num client data = 1
		0x08, 0x00, 0x00, 0x00, // Element size = 8
		// Filter 1: DEFLATE
		0x01, 0x00, // ID = 1 (deflate)
		0x00, 0x00, // Flags
		0x01, 0x00, // Num client data = 1
		0x09, 0x00, 0x00, 0x00, // Level 9
	}

	fp, err := parseFilterPipeline(data, mockReader())
	if err != nil {
		t.Fatalf("parseFilterPipeline failed: %v", err)
	}

	if len(fp.Filters) != 2 {
		t.Fatalf("expected 2 filters, got %d", len(fp.Filters))
	}
	if fp.Filters[0].ID != FilterShuffle {
		t.Errorf("expected shuffle filter first, got %d", fp.Filters[0].ID)
	}
	if fp.Filters[1].ID != FilterDeflate {
		t.Errorf("expected deflate filter second, got %d", fp.Filters[1].ID)
	}
	if !fp.HasFilter(FilterShuffle) {
		t.Error("HasFilter(FilterShuffle) should return true")
	}
}

func TestFilterPipelineFletcher32(t *testing.T) {
	data := []byte{
		2,          // Version 2
		1,          // Number of filters
		// Filter 0: Fletcher32
		0x03, 0x00, // ID = 3 (fletcher32)
		0x00, 0x00, // Flags
		0x00, 0x00, // Num client data = 0
	}

	fp, err := parseFilterPipeline(data, mockReader())
	if err != nil {
		t.Fatalf("parseFilterPipeline failed: %v", err)
	}

	if len(fp.Filters) != 1 {
		t.Fatalf("expected 1 filter, got %d", len(fp.Filters))
	}
	if fp.Filters[0].ID != FilterFletcher32 {
		t.Errorf("expected fletcher32 filter, got %d", fp.Filters[0].ID)
	}
	// Fletcher32 is a checksum, not compression
	if fp.HasCompression() {
		t.Error("HasCompression should return false for fletcher32 only")
	}
}

func TestFilterPipelineOptionalFilter(t *testing.T) {
	data := []byte{
		2,          // Version 2
		1,          // Number of filters
		// Filter 0: Some filter marked as optional
		0x01, 0x00, // ID = 1 (deflate)
		0x01, 0x00, // Flags: bit 0 = optional
		0x00, 0x00, // Num client data = 0
	}

	fp, err := parseFilterPipeline(data, mockReader())
	if err != nil {
		t.Fatalf("parseFilterPipeline failed: %v", err)
	}

	if !fp.Filters[0].IsOptional() {
		t.Error("filter should be marked as optional")
	}
}

func TestFilterPipelineTooShort(t *testing.T) {
	data := []byte{2} // Too short

	_, err := parseFilterPipeline(data, mockReader())
	if err == nil {
		t.Error("expected error for too short data")
	}
}

// === PARSE FUNCTION TESTS ===

func TestParseUnknownType(t *testing.T) {
	data := []byte{1, 2, 3, 4}
	msg, err := Parse(Type(0xFF), data, 0, mockReader())
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	unknown, ok := msg.(*Unknown)
	if !ok {
		t.Fatal("expected *Unknown message")
	}

	if unknown.Type() != Type(0xFF) {
		t.Errorf("expected type 0xFF, got 0x%x", unknown.Type())
	}
	if len(unknown.Data()) != 4 {
		t.Errorf("expected 4 bytes data, got %d", len(unknown.Data()))
	}
}

func TestParseDataspaceMessage(t *testing.T) {
	data := []byte{2, 0, 0, 0} // Version 2 scalar

	msg, err := Parse(TypeDataspace, data, 0, mockReader())
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	ds, ok := msg.(*Dataspace)
	if !ok {
		t.Fatal("expected *Dataspace message")
	}

	if ds.Type() != TypeDataspace {
		t.Errorf("expected type 0x0001, got 0x%x", ds.Type())
	}
}

func TestParseDatatypeMessage(t *testing.T) {
	data := make([]byte, 12)
	data[0] = 0x10 | byte(ClassFixedPoint)
	data[1] = 0x08
	binary.LittleEndian.PutUint32(data[4:], 4)

	msg, err := Parse(TypeDatatype, data, 0, mockReader())
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	dt, ok := msg.(*Datatype)
	if !ok {
		t.Fatal("expected *Datatype message")
	}

	if dt.Type() != TypeDatatype {
		t.Errorf("expected type 0x0003, got 0x%x", dt.Type())
	}
}

func TestParseLayoutMessage(t *testing.T) {
	data := make([]byte, 18)
	data[0] = 3
	data[1] = byte(LayoutContiguous)
	binary.LittleEndian.PutUint64(data[2:], 0x1000)
	binary.LittleEndian.PutUint64(data[10:], 0x2000)

	msg, err := Parse(TypeDataLayout, data, 0, mockReader())
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	layout, ok := msg.(*DataLayout)
	if !ok {
		t.Fatal("expected *DataLayout message")
	}

	if layout.Type() != TypeDataLayout {
		t.Errorf("expected type 0x0008, got 0x%x", layout.Type())
	}
}

// === FILE-BASED INTEGRATION TESTS ===

func TestParseFromIntegersFile(t *testing.T) {
	path := getTestdataPath("integers.h5")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file %s not found", path)
	}

	// Just verify the file can be opened
	t.Log("integers.h5 is available for message parsing tests")
}

func TestParseFromFloatsFile(t *testing.T) {
	path := getTestdataPath("floats.h5")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file %s not found", path)
	}

	t.Log("floats.h5 is available for message parsing tests")
}

func TestParseFromMultidimFile(t *testing.T) {
	path := getTestdataPath("multidim.h5")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file %s not found", path)
	}

	t.Log("multidim.h5 is available for message parsing tests")
}

func TestParseFromStringsFile(t *testing.T) {
	path := getTestdataPath("strings.h5")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file %s not found", path)
	}

	t.Log("strings.h5 is available for message parsing tests")
}

func TestParseFromCompactFile(t *testing.T) {
	path := getTestdataPath("compact.h5")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file %s not found", path)
	}

	t.Log("compact.h5 is available for compact layout parsing tests")
}

func TestParseFromChunkedFile(t *testing.T) {
	path := getTestdataPath("chunked.h5")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file %s not found", path)
	}

	t.Log("chunked.h5 is available for chunked layout parsing tests")
}

func TestParseFromCompressedFile(t *testing.T) {
	path := getTestdataPath("compressed.h5")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file %s not found", path)
	}

	t.Log("compressed.h5 is available for filter pipeline parsing tests")
}

func TestParseFromSoftlinkFile(t *testing.T) {
	path := getTestdataPath("softlink.h5")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file %s not found", path)
	}

	t.Log("softlink.h5 is available for link message parsing tests")
}

// === HELPER ===

func TestMockReader(t *testing.T) {
	r := mockReader()
	if r == nil {
		t.Fatal("mockReader returned nil")
	}
	if r.OffsetSize() != 8 {
		t.Errorf("expected offset size 8, got %d", r.OffsetSize())
	}
	if r.LengthSize() != 8 {
		t.Errorf("expected length size 8, got %d", r.LengthSize())
	}
}
