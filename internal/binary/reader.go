// Package binary provides low-level binary I/O operations for HDF5 file parsing.
package binary

import (
	"encoding/binary"
	"errors"
	"io"
)

// ErrInvalidSize is returned when an invalid offset or length size is specified.
var ErrInvalidSize = errors.New("invalid offset/length size: must be 2, 4, or 8")

// Reader provides methods for reading HDF5 binary data with variable-width
// offset and length fields.
type Reader struct {
	r          io.ReaderAt
	order      binary.ByteOrder
	offsetSize int
	lengthSize int
	pos        int64
}

// Config holds reader configuration, typically derived from the superblock.
type Config struct {
	ByteOrder  binary.ByteOrder
	OffsetSize int // 2, 4, or 8 bytes
	LengthSize int // 2, 4, or 8 bytes
}

// DefaultConfig returns a configuration suitable for initial superblock reading.
// Uses little-endian byte order and 8-byte offsets/lengths.
func DefaultConfig() Config {
	return Config{
		ByteOrder:  binary.LittleEndian,
		OffsetSize: 8,
		LengthSize: 8,
	}
}

// NewReader creates a binary reader with the given configuration.
func NewReader(r io.ReaderAt, cfg Config) *Reader {
	return &Reader{
		r:          r,
		order:      cfg.ByteOrder,
		offsetSize: cfg.OffsetSize,
		lengthSize: cfg.LengthSize,
		pos:        0,
	}
}

// At returns a new reader positioned at the given offset.
// The new reader shares the underlying io.ReaderAt but has independent position.
func (r *Reader) At(offset int64) *Reader {
	return &Reader{
		r:          r.r,
		order:      r.order,
		offsetSize: r.offsetSize,
		lengthSize: r.lengthSize,
		pos:        offset,
	}
}

// WithSizes returns a new reader with updated offset and length sizes.
// This is used after parsing the superblock to configure correct sizes.
func (r *Reader) WithSizes(offsetSize, lengthSize int) *Reader {
	return &Reader{
		r:          r.r,
		order:      r.order,
		offsetSize: offsetSize,
		lengthSize: lengthSize,
		pos:        r.pos,
	}
}

// Pos returns the current read position.
func (r *Reader) Pos() int64 {
	return r.pos
}

// ReadBytes reads exactly n bytes from the current position.
func (r *Reader) ReadBytes(n int) ([]byte, error) {
	if n <= 0 {
		return nil, nil
	}
	buf := make([]byte, n)
	_, err := r.r.ReadAt(buf, r.pos)
	if err != nil {
		return nil, err
	}
	r.pos += int64(n)
	return buf, nil
}

// ReadUint8 reads an unsigned 8-bit integer.
func (r *Reader) ReadUint8() (uint8, error) {
	buf, err := r.ReadBytes(1)
	if err != nil {
		return 0, err
	}
	return buf[0], nil
}

// ReadUint16 reads an unsigned 16-bit integer.
func (r *Reader) ReadUint16() (uint16, error) {
	buf, err := r.ReadBytes(2)
	if err != nil {
		return 0, err
	}
	return r.order.Uint16(buf), nil
}

// ReadUint32 reads an unsigned 32-bit integer.
func (r *Reader) ReadUint32() (uint32, error) {
	buf, err := r.ReadBytes(4)
	if err != nil {
		return 0, err
	}
	return r.order.Uint32(buf), nil
}

// ReadUint64 reads an unsigned 64-bit integer.
func (r *Reader) ReadUint64() (uint64, error) {
	buf, err := r.ReadBytes(8)
	if err != nil {
		return 0, err
	}
	return r.order.Uint64(buf), nil
}

// ReadUintN reads an unsigned integer of n bytes (1, 2, 4, or 8).
func (r *Reader) ReadUintN(n int) (uint64, error) {
	buf, err := r.ReadBytes(n)
	if err != nil {
		return 0, err
	}
	return r.decodeUint(buf, n), nil
}

// ReadOffset reads a file offset using the configured offset size.
func (r *Reader) ReadOffset() (uint64, error) {
	buf, err := r.ReadBytes(r.offsetSize)
	if err != nil {
		return 0, err
	}
	return r.decodeUint(buf, r.offsetSize), nil
}

// ReadLength reads a length value using the configured length size.
func (r *Reader) ReadLength() (uint64, error) {
	buf, err := r.ReadBytes(r.lengthSize)
	if err != nil {
		return 0, err
	}
	return r.decodeUint(buf, r.lengthSize), nil
}

// decodeUint decodes a variable-width unsigned integer.
func (r *Reader) decodeUint(buf []byte, size int) uint64 {
	switch size {
	case 1:
		return uint64(buf[0])
	case 2:
		return uint64(r.order.Uint16(buf))
	case 4:
		return uint64(r.order.Uint32(buf))
	case 8:
		return r.order.Uint64(buf)
	default:
		// Handle arbitrary sizes (little-endian assumed for non-standard)
		var val uint64
		for i := size - 1; i >= 0; i-- {
			val = (val << 8) | uint64(buf[i])
		}
		return val
	}
}

// IsUndefinedOffset checks if an offset value represents the "undefined" sentinel.
// In HDF5, undefined addresses are all 1-bits.
func (r *Reader) IsUndefinedOffset(offset uint64) bool {
	switch r.offsetSize {
	case 2:
		return offset == 0xFFFF
	case 4:
		return offset == 0xFFFFFFFF
	case 8:
		return offset == 0xFFFFFFFFFFFFFFFF
	default:
		// For non-standard sizes, check if all bits are set
		mask := uint64(1<<(r.offsetSize*8)) - 1
		return offset == mask
	}
}

// IsUndefinedLength checks if a length value represents the "undefined" sentinel.
func (r *Reader) IsUndefinedLength(length uint64) bool {
	switch r.lengthSize {
	case 2:
		return length == 0xFFFF
	case 4:
		return length == 0xFFFFFFFF
	case 8:
		return length == 0xFFFFFFFFFFFFFFFF
	default:
		mask := uint64(1<<(r.lengthSize*8)) - 1
		return length == mask
	}
}

// Skip advances the position by n bytes.
func (r *Reader) Skip(n int64) {
	r.pos += n
}

// Align advances the position to the next multiple of alignment.
// If already aligned, the position is unchanged.
func (r *Reader) Align(alignment int64) {
	if alignment <= 1 {
		return
	}
	if remainder := r.pos % alignment; remainder != 0 {
		r.pos += alignment - remainder
	}
}

// Peek reads n bytes without advancing the position.
func (r *Reader) Peek(n int) ([]byte, error) {
	if n <= 0 {
		return nil, nil
	}
	buf := make([]byte, n)
	_, err := r.r.ReadAt(buf, r.pos)
	if err != nil {
		return nil, err
	}
	return buf, nil
}

// OffsetSize returns the configured offset size in bytes.
func (r *Reader) OffsetSize() int {
	return r.offsetSize
}

// LengthSize returns the configured length size in bytes.
func (r *Reader) LengthSize() int {
	return r.lengthSize
}

// ByteOrder returns the configured byte order.
func (r *Reader) ByteOrder() binary.ByteOrder {
	return r.order
}
