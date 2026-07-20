package superblock

import (
	binpkg "github.com/huangzhengshun/go-hdf5/internal/binary"
)

// Write writes a V2/V3 superblock at the current writer position.
// Returns the total bytes written.
func (sb *Superblock) Write(w *binpkg.Writer) (int64, error) {
	startPos := w.Pos()

	// Calculate header size for checksum
	// Signature(8) + Version(1) + OffsetSize(1) + LengthSize(1) + Flags(1) + 4*Offset + Checksum(4)
	headerSize := 12 + 4*w.OffsetSize()

	// Buffer the header for checksum calculation
	buf := make([]byte, headerSize+4) // +4 for checksum
	bufWriter := &bufferWriterAt{buf: buf}
	bw := binpkg.NewWriter(bufWriter, binpkg.Config{
		ByteOrder:  w.ByteOrder(),
		OffsetSize: w.OffsetSize(),
		LengthSize: w.LengthSize(),
	})

	// Write signature
	if err := bw.WriteBytes(Signature); err != nil {
		return 0, err
	}

	// Write version (2 or 3)
	version := sb.Version
	if version < 2 {
		version = 2 // Default to version 2
	}
	if err := bw.WriteUint8(version); err != nil {
		return 0, err
	}

	// Write offset size
	if err := bw.WriteUint8(sb.OffsetSize); err != nil {
		return 0, err
	}

	// Write length size
	if err := bw.WriteUint8(sb.LengthSize); err != nil {
		return 0, err
	}

	// Write file consistency flags
	if err := bw.WriteUint8(sb.FileConsistencyFlags); err != nil {
		return 0, err
	}

	// Write base address
	if err := bw.WriteOffset(sb.BaseAddress); err != nil {
		return 0, err
	}

	// Write superblock extension address (undefined if not used)
	extAddr := sb.SuperblockExtensionAddress
	if extAddr == 0 {
		extAddr = bw.UndefinedOffset()
	}
	if err := bw.WriteOffset(extAddr); err != nil {
		return 0, err
	}

	// Write EOF address
	if err := bw.WriteOffset(sb.EOFAddress); err != nil {
		return 0, err
	}

	// Write root group object header address
	if err := bw.WriteOffset(sb.RootGroupAddress); err != nil {
		return 0, err
	}

	// Calculate checksum over header (before checksum field)
	checksumData := buf[:bw.Pos()]
	checksum := binpkg.Lookup3Checksum(checksumData)

	// Write checksum
	if err := bw.WriteUint32(checksum); err != nil {
		return 0, err
	}

	// Write the complete buffer to the actual writer
	if err := w.WriteBytes(buf[:bw.Pos()]); err != nil {
		return 0, err
	}

	return w.Pos() - startPos, nil
}

// Size returns the size in bytes of a V2/V3 superblock.
func (sb *Superblock) Size() int {
	// Signature(8) + Version(1) + OffsetSize(1) + LengthSize(1) + Flags(1) +
	// BaseAddr(O) + ExtAddr(O) + EOFAddr(O) + RootAddr(O) + Checksum(4)
	offsetSize := int(sb.OffsetSize)
	if offsetSize == 0 {
		offsetSize = 8
	}
	return 12 + 4*offsetSize + 4
}

// NewSuperblock creates a new V3 superblock with default settings.
func NewSuperblock() *Superblock {
	return &Superblock{
		Version:                    3, // Use V3 for best compatibility
		OffsetSize:                 8,
		LengthSize:                 8,
		FileConsistencyFlags:       0,
		BaseAddress:                0,
		SuperblockExtensionAddress: 0, // Will be set to undefined when writing
	}
}

// bufferWriterAt is a simple in-memory WriterAt for buffering.
type bufferWriterAt struct {
	buf []byte
}

func (b *bufferWriterAt) WriteAt(p []byte, off int64) (n int, err error) {
	if int(off)+len(p) > len(b.buf) {
		newBuf := make([]byte, int(off)+len(p))
		copy(newBuf, b.buf)
		b.buf = newBuf
	}
	copy(b.buf[off:], p)
	return len(p), nil
}
