package object

import (
	"bytes"
	"testing"

	"github.com/huangzhengshun/go-hdf5/internal/binary"
	"github.com/huangzhengshun/go-hdf5/internal/message"
)

func TestHeaderGetMessage(t *testing.T) {
	h := &Header{
		Version: 2,
		Messages: []message.Message{
			&message.Dataspace{Rank: 2, Dimensions: []uint64{10, 20}},
			&message.Datatype{Class: message.ClassFixedPoint, Size: 4},
		},
	}

	// Test finding existing message
	ds := h.GetMessage(message.TypeDataspace)
	if ds == nil {
		t.Error("expected to find dataspace message")
	}
	if space, ok := ds.(*message.Dataspace); !ok || space.Rank != 2 {
		t.Error("wrong dataspace returned")
	}

	// Test finding non-existent message
	fp := h.GetMessage(message.TypeFilterPipeline)
	if fp != nil {
		t.Error("expected nil for missing filter pipeline message")
	}
}

func TestHeaderGetMessages(t *testing.T) {
	attr1 := &message.Attribute{Name: "attr1"}
	attr2 := &message.Attribute{Name: "attr2"}
	h := &Header{
		Version: 2,
		Messages: []message.Message{
			&message.Dataspace{Rank: 1},
			attr1,
			attr2,
		},
	}

	// Get all attribute messages
	attrs := h.GetMessages(message.TypeAttribute)
	if len(attrs) != 2 {
		t.Errorf("expected 2 attributes, got %d", len(attrs))
	}

	// Get single dataspace
	spaces := h.GetMessages(message.TypeDataspace)
	if len(spaces) != 1 {
		t.Errorf("expected 1 dataspace, got %d", len(spaces))
	}

	// Get non-existent message type
	links := h.GetMessages(message.TypeLink)
	if len(links) != 0 {
		t.Errorf("expected 0 links, got %d", len(links))
	}
}

func TestHeaderDataspace(t *testing.T) {
	h := &Header{
		Messages: []message.Message{
			&message.Dataspace{Rank: 3, Dimensions: []uint64{2, 3, 4}},
		},
	}

	ds := h.Dataspace()
	if ds == nil {
		t.Fatal("expected dataspace")
	}
	if ds.Rank != 3 {
		t.Errorf("expected rank 3, got %d", ds.Rank)
	}
	if len(ds.Dimensions) != 3 || ds.Dimensions[0] != 2 {
		t.Errorf("unexpected dimensions: %v", ds.Dimensions)
	}

	// Test header without dataspace
	h2 := &Header{Messages: nil}
	if h2.Dataspace() != nil {
		t.Error("expected nil dataspace")
	}
}

func TestHeaderDatatype(t *testing.T) {
	h := &Header{
		Messages: []message.Message{
			&message.Datatype{Class: message.ClassFloatPoint, Size: 8},
		},
	}

	dt := h.Datatype()
	if dt == nil {
		t.Fatal("expected datatype")
	}
	if dt.Class != message.ClassFloatPoint {
		t.Errorf("expected float class, got %d", dt.Class)
	}
	if dt.Size != 8 {
		t.Errorf("expected size 8, got %d", dt.Size)
	}

	// Test header without datatype
	h2 := &Header{Messages: nil}
	if h2.Datatype() != nil {
		t.Error("expected nil datatype")
	}
}

func TestHeaderDataLayout(t *testing.T) {
	h := &Header{
		Messages: []message.Message{
			&message.DataLayout{Class: message.LayoutContiguous, Address: 1234},
		},
	}

	dl := h.DataLayout()
	if dl == nil {
		t.Fatal("expected data layout")
	}
	if dl.Class != message.LayoutContiguous {
		t.Errorf("expected contiguous layout, got %d", dl.Class)
	}

	// Test header without data layout
	h2 := &Header{Messages: nil}
	if h2.DataLayout() != nil {
		t.Error("expected nil data layout")
	}
}

func TestHeaderFilterPipeline(t *testing.T) {
	h := &Header{
		Messages: []message.Message{
			&message.FilterPipeline{
				Filters: []message.FilterInfo{{ID: 1}},
			},
		},
	}

	fp := h.FilterPipeline()
	if fp == nil {
		t.Fatal("expected filter pipeline")
	}
	if len(fp.Filters) != 1 {
		t.Errorf("expected 1 filter, got %d", len(fp.Filters))
	}

	// Test header without filter pipeline
	h2 := &Header{Messages: nil}
	if h2.FilterPipeline() != nil {
		t.Error("expected nil filter pipeline")
	}
}

func TestReadInvalidHeader(t *testing.T) {
	// Create buffer with invalid header (not v1 or v2)
	buf := bytes.NewBuffer(nil)
	buf.Write([]byte{99, 0, 0, 0}) // Invalid version/signature

	r := binary.NewReader(bytes.NewReader(buf.Bytes()), binary.DefaultConfig())

	_, err := Read(r, 0)
	if err == nil {
		t.Error("expected error for invalid header")
	}
}

func TestReadV2InvalidSignature(t *testing.T) {
	// Create buffer that looks like v2 but has wrong signature
	buf := bytes.NewBuffer(nil)
	buf.WriteString("XXXX") // Wrong signature (not OHDR)

	r := binary.NewReader(bytes.NewReader(buf.Bytes()), binary.DefaultConfig())

	_, err := Read(r, 0)
	if err == nil {
		t.Error("expected error for invalid v2 signature")
	}
}

func TestErrorTypes(t *testing.T) {
	// Verify error variables are defined
	if ErrInvalidHeader == nil {
		t.Error("ErrInvalidHeader should be defined")
	}
	if ErrUnsupportedVersion == nil {
		t.Error("ErrUnsupportedVersion should be defined")
	}
	if ErrChecksumMismatch == nil {
		t.Error("ErrChecksumMismatch should be defined")
	}
}

func TestSignatureV2(t *testing.T) {
	expected := []byte{'O', 'H', 'D', 'R'}
	if !bytes.Equal(SignatureV2, expected) {
		t.Errorf("expected SignatureV2 to be %q, got %q", expected, SignatureV2)
	}
}
