// Package superblock handles parsing of HDF5 superblock structures.
//
// The superblock is the entry point for any HDF5 file, containing critical
// metadata required to read the rest of the file. Every HDF5 file must have
// a superblock, which can be located at one of several standard offsets.
//
// # File Signature
//
// HDF5 files are identified by an 8-byte signature at the start of the
// superblock: 0x89 H D F \r \n 0x1a \n (hex: 89 48 44 46 0D 0A 1A 0A).
// The [Read] function searches for this signature at offsets 0, 512, 1024,
// and 2048 to locate the superblock.
//
// # Superblock Versions
//
// HDF5 defines four superblock versions:
//
//   - Version 0: Original format with fixed-size fields. Uses symbol table
//     entries and B-trees for the root group. Most compatible with old files.
//
//   - Version 1: Similar to v0 with minor additions (indexed storage K value).
//     Also uses symbol table entries for the root group.
//
//   - Version 2: Modern format with compact structure and optional superblock
//     extension. Root group is referenced directly by object header address.
//     Adds file consistency flags.
//
//   - Version 3: Same structure as v2. Used to indicate file consistency
//     flag semantics have changed.
//
// # Superblock Contents
//
// The [Superblock] structure contains:
//
//   - Version: Superblock format version (0-3)
//   - OffsetSize: Bytes used for file offsets (typically 8)
//   - LengthSize: Bytes used for lengths (typically 8)
//   - BaseAddress: Absolute file address of byte 0 (usually 0)
//   - EOFAddress: Logical end-of-file address
//   - RootGroupAddress: Address of the root group object header
//   - V0/V1 fields: B-tree parameters and symbol table addresses
//
// # Usage
//
// Read the superblock from an HDF5 file:
//
//	sb, err := superblock.Read(file)
//	if err == superblock.ErrNotHDF5 {
//	    // Not an HDF5 file
//	}
//
// Create a binary reader configured for this file:
//
//	config := sb.ReaderConfig()
//	reader := binary.NewReader(file, config)
//
// # Writing
//
// For HDF5 file creation, use [Write] to write a v2/v3 superblock:
//
//	superblock.Write(writer, sb)
//
// # Key Types and Functions
//
//   - [Superblock]: Contains all superblock fields
//   - [Read]: Locates and parses the superblock
//   - [Write]: Writes a superblock for file creation
//
// # Errors
//
//   - [ErrNotHDF5]: File does not have a valid HDF5 signature
//   - [ErrUnsupportedVersion]: Superblock version not supported
//   - [ErrInvalidSuperblock]: Superblock structure is invalid
package superblock
