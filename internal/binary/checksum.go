package binary

// Lookup3Checksum computes the Jenkins lookup3 hash used by HDF5 for
// metadata checksums (superblock v2/v3, object header v2, etc.).
//
// This implementation follows the HDF5 library's H5_checksum_lookup3
// which uses the hashlittle variant with initial values of 0.
func Lookup3Checksum(data []byte) uint32 {
	length := len(data)
	// Jenkins hashlittle: a = b = c = 0xdeadbeef + length + initval
	// HDF5 uses initval = 0
	initval := uint32(0xdeadbeef) + uint32(length)
	a, b, c := initval, initval, initval
	k := data

	// Process while MORE than 12 bytes remain (NOT >=, matching HDF5's "while (length > 12)")
	// This is critical: HDF5 processes the last 1-12 bytes in the switch statement
	// with final mix, not in the main loop with intermediate mix.
	for len(k) > 12 {
		a += uint32(k[0]) | uint32(k[1])<<8 |
			uint32(k[2])<<16 | uint32(k[3])<<24
		b += uint32(k[4]) | uint32(k[5])<<8 |
			uint32(k[6])<<16 | uint32(k[7])<<24
		c += uint32(k[8]) | uint32(k[9])<<8 |
			uint32(k[10])<<16 | uint32(k[11])<<24
		a, b, c = lookup3Mix(a, b, c)
		k = k[12:]
	}

	// Handle final 0-12 bytes
	switch len(k) {
	case 12:
		c += uint32(k[11]) << 24
		fallthrough
	case 11:
		c += uint32(k[10]) << 16
		fallthrough
	case 10:
		c += uint32(k[9]) << 8
		fallthrough
	case 9:
		c += uint32(k[8])
		fallthrough
	case 8:
		b += uint32(k[7]) << 24
		fallthrough
	case 7:
		b += uint32(k[6]) << 16
		fallthrough
	case 6:
		b += uint32(k[5]) << 8
		fallthrough
	case 5:
		b += uint32(k[4])
		fallthrough
	case 4:
		a += uint32(k[3]) << 24
		fallthrough
	case 3:
		a += uint32(k[2]) << 16
		fallthrough
	case 2:
		a += uint32(k[1]) << 8
		fallthrough
	case 1:
		a += uint32(k[0])
	case 0:
		// No bytes remaining, skip final mix (matching HDF5's "goto done")
		return c
	}

	a, b, c = lookup3Final(a, b, c)
	return c
}

func lookup3Mix(a, b, c uint32) (uint32, uint32, uint32) {
	a -= c
	a ^= rotl32(c, 4)
	c += b
	b -= a
	b ^= rotl32(a, 6)
	a += c
	c -= b
	c ^= rotl32(b, 8)
	b += a
	a -= c
	a ^= rotl32(c, 16)
	c += b
	b -= a
	b ^= rotl32(a, 19)
	a += c
	c -= b
	c ^= rotl32(b, 4)
	b += a
	return a, b, c
}

func lookup3Final(a, b, c uint32) (uint32, uint32, uint32) {
	c ^= b
	c -= rotl32(b, 14)
	a ^= c
	a -= rotl32(c, 11)
	b ^= a
	b -= rotl32(a, 25)
	c ^= b
	c -= rotl32(b, 16)
	a ^= c
	a -= rotl32(c, 4)
	b ^= a
	b -= rotl32(a, 14)
	c ^= b
	c -= rotl32(b, 24)
	return a, b, c
}

func rotl32(x uint32, k uint) uint32 {
	return (x << k) | (x >> (32 - k))
}

// Fletcher32 computes the Fletcher-32 checksum used by HDF5 for data
// verification (filter pipeline checksum).
//
// The input is treated as a sequence of 16-bit words in little-endian order.
// If the input has an odd number of bytes, it is padded with a zero byte.
func Fletcher32(data []byte) uint32 {
	var sum1, sum2 uint32

	// Process 16-bit words
	length := len(data)
	i := 0
	for ; i+1 < length; i += 2 {
		word := uint32(data[i]) | uint32(data[i+1])<<8
		sum1 = (sum1 + word) % 65535
		sum2 = (sum2 + sum1) % 65535
	}

	// Handle odd byte (pad with zero)
	if i < length {
		word := uint32(data[i])
		sum1 = (sum1 + word) % 65535
		sum2 = (sum2 + sum1) % 65535
	}

	return (sum2 << 16) | sum1
}

// VerifyFletcher32 verifies data against an expected Fletcher-32 checksum.
func VerifyFletcher32(data []byte, expected uint32) bool {
	return Fletcher32(data) == expected
}

// VerifyLookup3 verifies data against an expected lookup3 checksum.
func VerifyLookup3(data []byte, expected uint32) bool {
	return Lookup3Checksum(data) == expected
}
