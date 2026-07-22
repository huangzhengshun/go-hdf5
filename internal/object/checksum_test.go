package object

import (
	"bytes"
	"encoding/binary"
	"testing"

	bin "github.com/huangzhengshun/go-hdf5/internal/binary"
	"github.com/huangzhengshun/go-hdf5/internal/message"
)

func TestV2ObjectHeaderChecksum(t *testing.T) {
	buf := make([]byte, 1024)
	bufWriter := &bufferWriterAt{buf: buf}
	w := bin.NewWriter(bufWriter, bin.Config{
		ByteOrder:  binary.LittleEndian,
		OffsetSize: 8,
		LengthSize: 8,
	})

	messages := []message.Message{}

	pos, err := WriteHeader(w, messages)
	if err != nil {
		t.Fatalf("WriteHeader failed: %v", err)
	}

	rawData := buf[:pos]
	t.Logf("Raw header bytes (len=%d): %x", len(rawData), rawData)

	r := bin.NewReader(bytes.NewReader(rawData), bin.Config{
		ByteOrder:  binary.LittleEndian,
		OffsetSize: 8,
		LengthSize: 8,
	})

	readHdr, err := Read(r, 0)
	if err != nil {
		t.Fatalf("Read failed: %v", err)
	}

	t.Logf("Header read successfully: version=%d, refCount=%d, messages=%d",
		readHdr.Version, readHdr.RefCount, len(readHdr.Messages))
}

func TestV2ObjectHeaderChecksumWithMessages(t *testing.T) {
	buf := make([]byte, 1024)
	bufWriter := &bufferWriterAt{buf: buf}
	w := bin.NewWriter(bufWriter, bin.Config{
		ByteOrder:  binary.LittleEndian,
		OffsetSize: 8,
		LengthSize: 8,
	})

	linkInfo := message.NewLinkInfo()
	messages := []message.Message{linkInfo}

	pos, err := WriteHeaderWithMinChunk(w, messages, MinGroupChunkSize)
	if err != nil {
		t.Fatalf("WriteHeaderWithMinChunk failed: %v", err)
	}

	rawData := buf[:pos]
	t.Logf("Raw header bytes (len=%d):", len(rawData))
	for i := 0; i < len(rawData); i += 16 {
		end := i + 16
		if end > len(rawData) {
			end = len(rawData)
		}
		t.Logf("  %04x: %x", i, rawData[i:end])
	}

	storedChecksum := binary.LittleEndian.Uint32(rawData[len(rawData)-4:])
	t.Logf("Stored checksum: %08x", storedChecksum)

	checksumData := rawData[:len(rawData)-4]
	calculatedChecksum := bin.Lookup3Checksum(checksumData)
	t.Logf("Calculated checksum: %08x", calculatedChecksum)

	if storedChecksum != calculatedChecksum {
		t.Errorf("Checksum mismatch: stored=%08x, calculated=%08x", storedChecksum, calculatedChecksum)
	}

	r := bin.NewReader(bytes.NewReader(rawData), bin.Config{
		ByteOrder:  binary.LittleEndian,
		OffsetSize: 8,
		LengthSize: 8,
	})

	readHdr, err := Read(r, 0)
	if err != nil {
		t.Fatalf("Read failed: %v", err)
	}

	t.Logf("Header read successfully: version=%d, refCount=%d, messages=%d",
		readHdr.Version, readHdr.RefCount, len(readHdr.Messages))
}
