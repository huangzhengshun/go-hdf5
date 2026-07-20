// Diagnostic tool for analyzing HDF5 files
package main

import (
	"fmt"
	"os"

	"github.com/huangzhengshun/go-hdf5/hdf5"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run cmd/diagnose/main.go <file.h5>")
		os.Exit(1)
	}

	filename := os.Args[1]
	fmt.Printf("=== Analyzing %s ===\n\n", filename)

	f, err := hdf5.Open(filename)
	if err != nil {
		fmt.Printf("ERROR: Failed to open file: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	fmt.Printf("Superblock version: %d\n", f.Version())
	fmt.Println()

	// Walk the entire file
	walkGroup(f.Root(), "", 0)
}

func walkGroup(g *hdf5.Group, indent string, depth int) {
	if depth > 20 {
		fmt.Printf("%s[MAX DEPTH REACHED]\n", indent)
		return
	}

	members, err := g.Members()
	if err != nil {
		fmt.Printf("%sERROR getting members: %v\n", indent, err)
		return
	}

	attrs := g.Attrs()
	fmt.Printf("%sGroup %q:\n", indent, g.Path())
	fmt.Printf("%s  Members: %d\n", indent, len(members))
	fmt.Printf("%s  Attrs: %v\n", indent, attrs)

	if len(members) == 0 && len(attrs) == 0 && depth > 0 {
		fmt.Printf("%s  [EMPTY - no members or attrs]\n", indent)
	}

	for _, name := range members {
		// Try as group first
		subg, err := g.OpenGroup(name)
		if err == nil {
			walkGroup(subg, indent+"  ", depth+1)
			continue
		}

		// Try as dataset
		ds, err := g.OpenDataset(name)
		if err == nil {
			fmt.Printf("%s  Dataset %q:\n", indent, name)
			fmt.Printf("%s    Shape: %v\n", indent, ds.Shape())
			fmt.Printf("%s    Attrs: %v\n", indent, ds.Attrs())
			continue
		}

		fmt.Printf("%s  %q: ERROR opening as group or dataset\n", indent, name)
		fmt.Printf("%s    Group error: %v\n", indent, err)
	}
}
