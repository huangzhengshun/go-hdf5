package main

import (
	"fmt"

	"github.com/huangzhengshun/go-hdf5/hdf5"
)

func main() {
	path := "bug_check.h5"

	// Create file
	f, err := hdf5.Create(path)
	if err != nil {
		fmt.Printf("Failed to create file: %v\n", err)
		return
	}

	// Create bool attribute
	f.Root().CreateAttribute("is_active", true)

	// Create dataset
	intData := []int32{1, 2, 3}
	f.Root().CreateDataset("data", intData)

	// Explicitly flush before closing
	fmt.Println("Flushing...")
	err = f.Flush()
	if err != nil {
		fmt.Printf("Flush failed: %v\n", err)
	}

	fmt.Println("Closing...")
	f.Close()

	// Try reading with OpenReadWrite
	fmt.Println("\n=== Reading with OpenReadWrite ===")
	f2, err := hdf5.OpenReadWrite(path)
	if err != nil {
		fmt.Printf("OpenReadWrite failed: %v\n", err)
		return
	}
	defer f2.Close()

	// Test reading dataset
	ds, err := f2.OpenDataset("data")
	if err != nil {
		fmt.Printf("Failed to open dataset: %v\n", err)
	} else {
		fmt.Printf("Dataset 'data' exists, shape: %v\n", ds.Shape())
	}

	// Test reading attribute
	attr := f2.Root().Attr("is_active")
	if attr == nil {
		fmt.Println("Failed to get attribute: is_active")
	} else {
		var val bool
		err := attr.Read(&val)
		if err != nil {
			fmt.Printf("Failed to read bool: %v\n", err)
		} else {
			fmt.Printf("is_active value (bool): %v\n", val)
		}
	}

	fmt.Println("OpenReadWrite test complete")
}
