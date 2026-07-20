package message

import (
	"encoding/binary"
	"testing"

	binpkg "github.com/huangzhengshun/go-hdf5/internal/binary"
)

// mockReader creates a minimal Reader for testing.
func mockReader() *binpkg.Reader {
	data := make([]byte, 256)
	return binpkg.NewReader(bytesReaderAt(data), binpkg.Config{
		ByteOrder:  binary.LittleEndian,
		OffsetSize: 8,
		LengthSize: 8,
	})
}

type bytesReaderAt []byte

func (b bytesReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if off >= int64(len(b)) {
		return 0, nil
	}
	n := copy(p, b[off:])
	return n, nil
}

func TestDataspaceScalar(t *testing.T) {
	// Version 2 scalar dataspace
	data := []byte{
		2,    // Version
		0,    // Rank (0 = scalar)
		0,    // Flags
		0,    // Type = scalar
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

func TestDataspaceSimple1D(t *testing.T) {
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

func TestDataspaceSimple2D(t *testing.T) {
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

func TestDataspaceNull(t *testing.T) {
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

func TestDatatypeFixedPointSigned(t *testing.T) {
	// 32-bit signed integer, little-endian
	data := make([]byte, 12)
	data[0] = 0x10 | byte(ClassFixedPoint) // Version 1, class 0
	data[1] = 0x08                          // Signed bit set
	data[2] = 0
	data[3] = 0
	binary.LittleEndian.PutUint32(data[4:], 4) // Size = 4 bytes
	// Properties: bit offset, bit precision
	binary.LittleEndian.PutUint16(data[8:], 0)  // Bit offset
	binary.LittleEndian.PutUint16(data[10:], 32) // Bit precision

	dt, err := parseDatatype(data, mockReader())
	if err != nil {
		t.Fatalf("parseDatatype failed: %v", err)
	}

	if dt.Class != ClassFixedPoint {
		t.Errorf("expected class %d, got %d", ClassFixedPoint, dt.Class)
	}
	if dt.Size != 4 {
		t.Errorf("expected size 4, got %d", dt.Size)
	}
	if !dt.Signed {
		t.Error("expected signed type")
	}
	if !dt.IsInteger() {
		t.Error("IsInteger should return true")
	}
}

func TestDatatypeFloat64(t *testing.T) {
	// 64-bit float, little-endian
	data := make([]byte, 20)
	data[0] = 0x10 | byte(ClassFloatPoint) // Version 1, class 1
	data[1] = 0                             // Little-endian
	data[2] = 0
	data[3] = 0
	binary.LittleEndian.PutUint32(data[4:], 8) // Size = 8 bytes

	dt, err := parseDatatype(data, mockReader())
	if err != nil {
		t.Fatalf("parseDatatype failed: %v", err)
	}

	if dt.Class != ClassFloatPoint {
		t.Errorf("expected class %d, got %d", ClassFloatPoint, dt.Class)
	}
	if dt.Size != 8 {
		t.Errorf("expected size 8, got %d", dt.Size)
	}
	if !dt.IsFloat() {
		t.Error("IsFloat should return true")
	}
}

func TestDatatypeString(t *testing.T) {
	// Fixed-length string, 10 bytes, null-terminated, ASCII
	data := make([]byte, 8)
	data[0] = 0x10 | byte(ClassString) // Version 1, class 3
	data[1] = byte(PadNullTerm)        // Null-terminated
	data[2] = byte(CharsetASCII) << 4   // ASCII charset
	data[3] = 0
	binary.LittleEndian.PutUint32(data[4:], 10) // Size = 10 bytes

	dt, err := parseDatatype(data, mockReader())
	if err != nil {
		t.Fatalf("parseDatatype failed: %v", err)
	}

	if dt.Class != ClassString {
		t.Errorf("expected class %d, got %d", ClassString, dt.Class)
	}
	if dt.Size != 10 {
		t.Errorf("expected size 10, got %d", dt.Size)
	}
	if !dt.IsString() {
		t.Error("IsString should return true")
	}
}

func TestDataLayoutContiguous(t *testing.T) {
	// Version 3 contiguous layout
	data := make([]byte, 18)
	data[0] = 3                        // Version 3
	data[1] = byte(LayoutContiguous)   // Contiguous

	binary.LittleEndian.PutUint64(data[2:], 1024)  // Address
	binary.LittleEndian.PutUint64(data[10:], 4096) // Size

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
	if layout.Address != 1024 {
		t.Errorf("expected address 1024, got %d", layout.Address)
	}
	if layout.Size != 4096 {
		t.Errorf("expected size 4096, got %d", layout.Size)
	}
	if !layout.IsContiguous() {
		t.Error("IsContiguous should return true")
	}
}

func TestDataLayoutCompact(t *testing.T) {
	// Version 3 compact layout with 4 bytes of data
	compactData := []byte{1, 2, 3, 4}
	data := make([]byte, 4+len(compactData))
	data[0] = 3                     // Version 3
	data[1] = byte(LayoutCompact)   // Compact
	binary.LittleEndian.PutUint16(data[2:], uint16(len(compactData)))
	copy(data[4:], compactData)

	layout, err := parseDataLayout(data, mockReader())
	if err != nil {
		t.Fatalf("parseDataLayout failed: %v", err)
	}

	if !layout.IsCompact() {
		t.Error("IsCompact should return true")
	}
	if len(layout.CompactData) != 4 {
		t.Errorf("expected 4 bytes compact data, got %d", len(layout.CompactData))
	}
}

func TestFilterPipeline(t *testing.T) {
	// Version 2 filter pipeline with 1 filter (simpler test)
	data := []byte{
		2,    // Version
		1,    // Number of filters
		// Filter 0: DEFLATE
		0x01, 0x00, // ID = 1 (deflate)
		0x00, 0x00, // Name length (0 for v2 with known filters)
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
		t.Errorf("expected filter 0 ID %d, got %d", FilterDeflate, fp.Filters[0].ID)
	}
	if !fp.HasCompression() {
		t.Error("HasCompression should return true")
	}
}

func TestLinkHard(t *testing.T) {
	// Version 1 hard link named "dataset"
	name := "dataset"
	// Size: 2 (header) + 1 (type) + 1 (name len) + len(name) + 8 (address)
	data := make([]byte, 2+1+1+len(name)+8)
	data[0] = 1                        // Version
	data[1] = 0x08 | 0                 // Flags: link type present, name len size = 1 byte
	data[2] = byte(LinkTypeHard)       // Link type
	data[3] = byte(len(name))          // Name length
	copy(data[4:], name)
	binary.LittleEndian.PutUint64(data[4+len(name):], 0x1234) // Object address

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

func TestSymbolTable(t *testing.T) {
	data := make([]byte, 16)
	binary.LittleEndian.PutUint64(data[0:], 0x1000) // B-tree address
	binary.LittleEndian.PutUint64(data[8:], 0x2000) // Local heap address

	st, err := parseSymbolTable(data, mockReader())
	if err != nil {
		t.Fatalf("parseSymbolTable failed: %v", err)
	}

	if st.BTreeAddress != 0x1000 {
		t.Errorf("expected B-tree address 0x1000, got 0x%x", st.BTreeAddress)
	}
	if st.LocalHeapAddress != 0x2000 {
		t.Errorf("expected local heap address 0x2000, got 0x%x", st.LocalHeapAddress)
	}
}

func TestUnknownMessage(t *testing.T) {
	data := []byte{1, 2, 3, 4}
	msg, err := Parse(Type(0x99), data, 0, mockReader())
	if err != nil {
		t.Fatalf("Parse unknown message failed: %v", err)
	}

	unknown, ok := msg.(*Unknown)
	if !ok {
		t.Fatal("expected *Unknown message")
	}

	if unknown.Type() != Type(0x99) {
		t.Errorf("expected type 0x99, got 0x%x", unknown.Type())
	}
}
