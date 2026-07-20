// Package binary provides low-level binary I/O operations for HDF5 file parsing and writing.
package binary

import (
	"encoding/binary"
	"io"
)

// Writer provides methods for writing HDF5 binary data with variable-width
// offset and length fields.
type Writer struct {
	w          io.WriterAt
	order      binary.ByteOrder
	offsetSize int
	lengthSize int
	pos        int64
}

// NewWriter creates a binary writer with the given configuration.
func NewWriter(w io.WriterAt, cfg Config) *Writer {
	return &Writer{
		w:          w,
		order:      cfg.ByteOrder,
		offsetSize: cfg.OffsetSize,
		lengthSize: cfg.LengthSize,
		pos:        0,
	}
}

// At returns a new writer positioned at the given offset.
// The new writer shares the underlying io.WriterAt but has independent position.
func (w *Writer) At(offset int64) *Writer {
	return &Writer{
		w:          w.w,
		order:      w.order,
		offsetSize: w.offsetSize,
		lengthSize: w.lengthSize,
		pos:        offset,
	}
}

// WithSizes returns a new writer with updated offset and length sizes.
func (w *Writer) WithSizes(offsetSize, lengthSize int) *Writer {
	return &Writer{
		w:          w.w,
		order:      w.order,
		offsetSize: offsetSize,
		lengthSize: lengthSize,
		pos:        w.pos,
	}
}

// Pos returns the current write position.
func (w *Writer) Pos() int64 {
	return w.pos
}

// WriteBytes writes the given bytes at the current position.
func (w *Writer) WriteBytes(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	n, err := w.w.WriteAt(data, w.pos)
	w.pos += int64(n)
	return err
}

// WriteUint8 writes an unsigned 8-bit integer.
func (w *Writer) WriteUint8(v uint8) error {
	return w.WriteBytes([]byte{v})
}

// WriteUint16 writes an unsigned 16-bit integer.
func (w *Writer) WriteUint16(v uint16) error {
	buf := make([]byte, 2)
	w.order.PutUint16(buf, v)
	return w.WriteBytes(buf)
}

// WriteUint32 writes an unsigned 32-bit integer.
func (w *Writer) WriteUint32(v uint32) error {
	buf := make([]byte, 4)
	w.order.PutUint32(buf, v)
	return w.WriteBytes(buf)
}

// WriteUint64 writes an unsigned 64-bit integer.
func (w *Writer) WriteUint64(v uint64) error {
	buf := make([]byte, 8)
	w.order.PutUint64(buf, v)
	return w.WriteBytes(buf)
}

// WriteUintN writes an unsigned integer of n bytes (1, 2, 4, or 8).
func (w *Writer) WriteUintN(v uint64, n int) error {
	buf := make([]byte, n)
	w.encodeUint(buf, v, n)
	return w.WriteBytes(buf)
}

// WriteOffset writes a file offset using the configured offset size.
func (w *Writer) WriteOffset(v uint64) error {
	return w.WriteUintN(v, w.offsetSize)
}

// WriteLength writes a length value using the configured length size.
func (w *Writer) WriteLength(v uint64) error {
	return w.WriteUintN(v, w.lengthSize)
}

// encodeUint encodes a variable-width unsigned integer into a buffer.
func (w *Writer) encodeUint(buf []byte, v uint64, size int) {
	switch size {
	case 1:
		buf[0] = uint8(v)
	case 2:
		w.order.PutUint16(buf, uint16(v))
	case 4:
		w.order.PutUint32(buf, uint32(v))
	case 8:
		w.order.PutUint64(buf, v)
	default:
		// Handle arbitrary sizes (little-endian assumed for non-standard)
		for i := 0; i < size; i++ {
			buf[i] = byte(v >> (8 * i))
		}
	}
}

// UndefinedOffset returns the "undefined" sentinel value for offsets.
// In HDF5, undefined addresses are all 1-bits.
func (w *Writer) UndefinedOffset() uint64 {
	switch w.offsetSize {
	case 2:
		return 0xFFFF
	case 4:
		return 0xFFFFFFFF
	case 8:
		return 0xFFFFFFFFFFFFFFFF
	default:
		return uint64(1<<(w.offsetSize*8)) - 1
	}
}

// UndefinedLength returns the "undefined" sentinel value for lengths.
func (w *Writer) UndefinedLength() uint64 {
	switch w.lengthSize {
	case 2:
		return 0xFFFF
	case 4:
		return 0xFFFFFFFF
	case 8:
		return 0xFFFFFFFFFFFFFFFF
	default:
		return uint64(1<<(w.lengthSize*8)) - 1
	}
}

// WriteUndefinedOffset writes the undefined offset sentinel value.
func (w *Writer) WriteUndefinedOffset() error {
	return w.WriteOffset(w.UndefinedOffset())
}

// WriteUndefinedLength writes the undefined length sentinel value.
func (w *Writer) WriteUndefinedLength() error {
	return w.WriteLength(w.UndefinedLength())
}

// Skip advances the position by n bytes without writing.
func (w *Writer) Skip(n int64) {
	w.pos += n
}

// Align advances the position to the next multiple of alignment.
// If already aligned, the position is unchanged.
func (w *Writer) Align(alignment int64) {
	if alignment <= 1 {
		return
	}
	if remainder := w.pos % alignment; remainder != 0 {
		w.pos += alignment - remainder
	}
}

// WritePadding writes zero bytes to align to the given alignment.
func (w *Writer) WritePadding(alignment int64) error {
	if alignment <= 1 {
		return nil
	}
	remainder := w.pos % alignment
	if remainder == 0 {
		return nil
	}
	padding := alignment - remainder
	zeros := make([]byte, padding)
	return w.WriteBytes(zeros)
}

// WriteZeros writes n zero bytes.
func (w *Writer) WriteZeros(n int) error {
	if n <= 0 {
		return nil
	}
	zeros := make([]byte, n)
	return w.WriteBytes(zeros)
}

// OffsetSize returns the configured offset size in bytes.
func (w *Writer) OffsetSize() int {
	return w.offsetSize
}

// LengthSize returns the configured length size in bytes.
func (w *Writer) LengthSize() int {
	return w.lengthSize
}

// ByteOrder returns the configured byte order.
func (w *Writer) ByteOrder() binary.ByteOrder {
	return w.order
}

// SeekableWriterAt wraps an io.WriteSeeker to provide io.WriterAt functionality.
// This is useful when working with os.File which implements WriteSeeker.
type SeekableWriterAt struct {
	ws io.WriteSeeker
}

// NewSeekableWriterAt creates a WriterAt from a WriteSeeker.
func NewSeekableWriterAt(ws io.WriteSeeker) *SeekableWriterAt {
	return &SeekableWriterAt{ws: ws}
}

// WriteAt implements io.WriterAt.
func (s *SeekableWriterAt) WriteAt(p []byte, off int64) (n int, err error) {
	_, err = s.ws.Seek(off, io.SeekStart)
	if err != nil {
		return 0, err
	}
	return s.ws.Write(p)
}
