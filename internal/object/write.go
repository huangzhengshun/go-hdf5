package object

import (
	"github.com/huangzhengshun/go-hdf5/internal/binary"
	"github.com/huangzhengshun/go-hdf5/internal/message"
)

// MinGroupChunkSize is the minimum chunk size for group object headers.
// This matches what h5py uses for compatibility.
const MinGroupChunkSize = 120

// WriteHeader writes a V2 object header at the current writer position.
// It buffers the header data to compute the checksum.
// Returns the total bytes written.
func WriteHeader(w *binary.Writer, messages []message.Message) (int64, error) {
	return WriteHeaderWithMinChunk(w, messages, 0)
}

// WriteHeaderWithMinChunk writes a V2 object header with a minimum chunk size.
// Note: In HDF5, the chunk size field contains only the size of messages + padding (NOT checksum).
func WriteHeaderWithMinChunk(w *binary.Writer, messages []message.Message, minChunkSize int) (int64, error) {
	startPos := w.Pos()

	// First pass: calculate an estimate for the buffer size
	var messagesSize int
	for _, msg := range messages {
		messagesSize += messageHeaderSize(w, msg)
		if s, ok := msg.(message.Serializable); ok {
			messagesSize += s.SerializedSize(w)
		}
	}

	// Chunk size = messages + padding (NOT including checksum, per HDF5 spec)
	// minChunkSize is the minimum size for messages + padding
	chunkSize := messagesSize
	if minChunkSize > 0 && chunkSize < minChunkSize {
		chunkSize = minChunkSize
	}

	paddingSize := chunkSize - messagesSize
	if paddingSize < 0 {
		paddingSize = 0
	}

	// NIL message has a minimum size of 4 bytes (type + size + flags).
	// If padding is 1-3 bytes, we need to increase it to 4 to fit a NIL message.
	if paddingSize > 0 && paddingSize < 4 {
		paddingSize = 4
		chunkSize = messagesSize + paddingSize
	}

	// Determine chunk size field size and flags
	// HDF5 flag bits 0-1 encode: 0→1 byte, 1→2 bytes, 2→4 bytes, 3→8 bytes
	var chunkSizeFieldSize int
	var flags uint8
	if chunkSize <= 0xFF {
		chunkSizeFieldSize = 1
		flags = 0
	} else if chunkSize <= 0xFFFF {
		chunkSizeFieldSize = 2
		flags = 1
	} else if chunkSize <= 0xFFFFFFFF {
		chunkSizeFieldSize = 4
		flags = 2
	} else {
		chunkSizeFieldSize = 8
		flags = 3
	}

	// Calculate estimated total header size for buffering
	// signature(4) + version(1) + flags(1) + chunkSize(var) + messages + padding + checksum(4)
	estimatedHeaderSize := 4 + 1 + 1 + chunkSizeFieldSize + messagesSize + paddingSize + 4

	// Create a buffer larger than estimated to accommodate any differences
	buf := make([]byte, estimatedHeaderSize+1024)
	bufWriter := &bufferWriterAt{buf: buf}
	bw := binary.NewWriter(bufWriter, binary.Config{
		ByteOrder:  w.ByteOrder(),
		OffsetSize: w.OffsetSize(),
		LengthSize: w.LengthSize(),
	})

	// Write signature
	if err := bw.WriteBytes(SignatureV2); err != nil {
		return 0, err
	}

	// Write version
	if err := bw.WriteUint8(2); err != nil {
		return 0, err
	}

	// Write flags
	if err := bw.WriteUint8(flags); err != nil {
		return 0, err
	}

	// Write chunk size
	if err := bw.WriteUintN(uint64(chunkSize), chunkSizeFieldSize); err != nil {
		return 0, err
	}

	// Write messages
	for _, msg := range messages {
		if err := writeV2Message(bw, msg); err != nil {
			return 0, err
		}
	}

	// Write NIL padding if needed
	if paddingSize > 0 {
		// NIL message: type(1) + size(2) + flags(1) + data(size)
		nilDataSize := paddingSize - 4
		if nilDataSize < 0 {
			nilDataSize = 0
		}
		// Type 0x00 = NIL
		if err := bw.WriteUint8(0x00); err != nil {
			return 0, err
		}
		// Size
		if err := bw.WriteUint16(uint16(nilDataSize)); err != nil {
			return 0, err
		}
		// Flags
		if err := bw.WriteUint8(0x00); err != nil {
			return 0, err
		}
		// Padding data (zeros)
		padding := make([]byte, nilDataSize)
		if err := bw.WriteBytes(padding); err != nil {
			return 0, err
		}
	}

	// Calculate checksum: covers everything from OHDR to just before the checksum
	// This is: signature(4) + version(1) + flags(1) + chunkSize(var) + messages + padding
	// = 6 + chunkSizeFieldSize + chunkSize
	checksumDataSize := 6 + chunkSizeFieldSize + chunkSize
	checksumData := bufWriter.buf[:checksumDataSize]
	checksum := binary.Lookup3Checksum(checksumData)

	// Write checksum
	if err := bw.WriteUint32(checksum); err != nil {
		return 0, err
	}

	// Get actual size written
	actualHeaderSize := bw.Pos()

	// Write the complete buffer to the actual writer
	if err := w.WriteBytes(bufWriter.buf[:actualHeaderSize]); err != nil {
		return 0, err
	}

	return w.Pos() - startPos, nil
}

// writeV2Message writes a single V2 message.
func writeV2Message(w *binary.Writer, msg message.Message) error {
	s, ok := msg.(message.Serializable)
	if !ok {
		// Skip non-serializable messages
		return nil
	}

	dataSize := s.SerializedSize(w)

	// Use extended format if size > 65535
	if dataSize > 65535 {
		// Extended format
		if err := w.WriteUint8(0xFF); err != nil {
			return err
		}
		if err := w.WriteUint8(uint8(msg.Type())); err != nil {
			return err
		}
		if err := w.WriteUint32(uint32(dataSize)); err != nil {
			return err
		}
	} else {
		// Normal format
		if err := w.WriteUint8(uint8(msg.Type())); err != nil {
			return err
		}
		if err := w.WriteUint16(uint16(dataSize)); err != nil {
			return err
		}
	}

	// Flags (0 = no special flags)
	if err := w.WriteUint8(0); err != nil {
		return err
	}

	// Message data
	return s.Serialize(w)
}

// messageHeaderSize returns the size of the V2 message header for a given message.
func messageHeaderSize(w *binary.Writer, msg message.Message) int {
	s, ok := msg.(message.Serializable)
	if !ok {
		return 0
	}

	dataSize := s.SerializedSize(w)
	if dataSize > 65535 {
		// Extended format: 0xFF + type + size(4) + flags = 7 bytes
		return 7
	}
	// Normal format: type + size(2) + flags = 4 bytes
	return 4
}

// chunkSizeFieldBytes returns the number of bytes needed to store the chunk size.
// HDF5 flag bits 0-1 encode: 0→1 byte, 1→2 bytes, 2→4 bytes, 3→8 bytes
func chunkSizeFieldBytes(size int64) int {
	if size <= 0xFF {
		return 1
	}
	if size <= 0xFFFF {
		return 2
	}
	if size <= 0xFFFFFFFF {
		return 4
	}
	return 8
}

// bufferWriterAt is a simple in-memory WriterAt for buffering.
type bufferWriterAt struct {
	buf []byte
}

func (b *bufferWriterAt) WriteAt(p []byte, off int64) (n int, err error) {
	if int(off)+len(p) > len(b.buf) {
		// Extend buffer
		newBuf := make([]byte, int(off)+len(p))
		copy(newBuf, b.buf)
		b.buf = newBuf
	}
	copy(b.buf[off:], p)
	return len(p), nil
}

// SerializeHeader serializes a V2 object header to a byte slice.
// This is useful for buffering headers in memory before writing to disk.
func SerializeHeader(w *binary.Writer, messages []message.Message, minChunkSize int) ([]byte, error) {
	buf := make([]byte, 1024*1024)
	bufWriter := &bufferWriterAt{buf: buf}
	bw := binary.NewWriter(bufWriter, binary.Config{
		ByteOrder:  w.ByteOrder(),
		OffsetSize: w.OffsetSize(),
		LengthSize: w.LengthSize(),
	})

	if _, err := WriteHeaderWithMinChunk(bw, messages, minChunkSize); err != nil {
		return nil, err
	}

	actualSize := bw.Pos()
	result := make([]byte, actualSize)
	copy(result, bufWriter.buf[:actualSize])
	return result, nil
}

// HeaderSize calculates the total size of a V2 object header with the given messages.
func HeaderSize(w *binary.Writer, messages []message.Message) int {
	return HeaderSizeWithMinChunk(w, messages, 0)
}

// HeaderSizeWithMinChunk calculates the total size with a minimum chunk size.
// The returned size includes: prefix + chunk (messages + padding) + checksum.
func HeaderSizeWithMinChunk(w *binary.Writer, messages []message.Message, minChunkSize int) int {
	var messagesSize int
	for _, msg := range messages {
		messagesSize += messageHeaderSize(w, msg)
		if s, ok := msg.(message.Serializable); ok {
			messagesSize += s.SerializedSize(w)
		}
	}

	// Chunk size = messages + padding (NOT including checksum, per HDF5 spec)
	chunkSize := messagesSize
	if minChunkSize > 0 && chunkSize < minChunkSize {
		chunkSize = minChunkSize
	}

	paddingSize := chunkSize - messagesSize
	if paddingSize < 0 {
		paddingSize = 0
	}

	// NIL message has a minimum size of 4 bytes (type + size + flags).
	if paddingSize > 0 && paddingSize < 4 {
		paddingSize = 4
		chunkSize = messagesSize + paddingSize
	}

	// Determine chunk size field size
	var chunkSizeFieldSize int
	if chunkSize <= 0xFF {
		chunkSizeFieldSize = 1
	} else if chunkSize <= 0xFFFF {
		chunkSizeFieldSize = 2
	} else if chunkSize <= 0xFFFFFFFF {
		chunkSizeFieldSize = 4
	} else {
		chunkSizeFieldSize = 8
	}

	// signature(4) + version(1) + flags(1) + chunkSize(var) + chunk(messages + padding) + checksum(4)
	return 4 + 1 + 1 + chunkSizeFieldSize + chunkSize + 4
}

// NewEmptyGroupHeader creates messages for an empty group object header.
// Includes LinkInfo and GroupInfo messages for HDF5 library/h5py compatibility.
func NewEmptyGroupHeader() []message.Message {
	return []message.Message{
		message.NewLinkInfo(),
		message.NewGroupInfo(),
	}
}

// NewGroupHeader creates messages for a group with links.
// Includes LinkInfo and GroupInfo messages plus all the provided links.
func NewGroupHeader(links []*message.Link) []message.Message {
	messages := make([]message.Message, 0, len(links)+2)
	messages = append(messages, message.NewLinkInfo())
	messages = append(messages, message.NewGroupInfo())
	for _, link := range links {
		messages = append(messages, link)
	}
	return messages
}

// NewDatasetHeader creates messages for a dataset object header.
func NewDatasetHeader(dataspace *message.Dataspace, datatype *message.Datatype, layout *message.DataLayout) []message.Message {
	messages := []message.Message{
		dataspace,
		datatype,
		layout,
	}
	return messages
}
