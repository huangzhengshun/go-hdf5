package main

import (
	"fmt"

	"github.com/huangzhengshun/go-hdf5/hdf5"
)

func main() {
	path := "edge_test.h5"

	// Create file
	f, err := hdf5.Create(path)
	if err != nil {
		fmt.Printf("Failed to create file: %v\n", err)
		return
	}
	defer f.Close()

	// Test 1: Array attributes
	fmt.Println("Creating array attributes...")
	intArray := []int32{1, 2, 3, 4, 5}
	f.Root().CreateAttribute("int_array", intArray)

	floatArray := []float64{1.1, 2.2, 3.3}
	f.Root().CreateAttribute("float_array", floatArray)

	strArray := []string{"alpha", "beta", "gamma"}
	f.Root().CreateAttribute("str_array", strArray)

	// Test 2: Chunked and compressed dataset
	fmt.Println("Creating chunked dataset...")
	largeData := make([]float64, 1000)
	for i := range largeData {
		largeData[i] = float64(i) * 0.1
	}
	f.Root().CreateDataset("chunked_data", largeData,
		hdf5.WithChunks(100),
		hdf5.WithCompression(4),
	)

	// Test 3: Soft link
	fmt.Println("Creating soft link...")
	f.Root().CreateSoftLink("data_link", "/chunked_data")

	// Test 4: Very small dataset
	fmt.Println("Creating tiny dataset...")
	f.Root().CreateDataset("tiny_data", []int8{1, 2, 3})

	// Test 5: Unsigned types
	fmt.Println("Creating unsigned datasets...")
	uint8Data := []uint8{0, 127, 255}
	f.Root().CreateDataset("uint8_data", uint8Data)

	uint16Data := []uint16{0, 32767, 65535}
	f.Root().CreateDataset("uint16_data", uint16Data)

	uint64Data := []uint64{0, 18446744073709551615}
	f.Root().CreateDataset("uint64_data", uint64Data)

	// Test 6: Signed types
	fmt.Println("Creating signed datasets...")
	int8Data := []int8{-128, 0, 127}
	f.Root().CreateDataset("int8_data", int8Data)

	int16Data := []int16{-32768, 0, 32767}
	f.Root().CreateDataset("int16_data", int16Data)

	// Test 7: Boolean dataset
	fmt.Println("Creating bool dataset...")
	boolData := []bool{true, false, true, true, false}
	f.Root().CreateDataset("bool_data", boolData)

	// Test 8: Single element dataset
	fmt.Println("Creating single element dataset...")
	f.Root().CreateDataset("single_int", []int32{42})

	// Test 9: Large int64 dataset
	fmt.Println("Creating int64 dataset...")
	int64Data := []int64{-9223372036854775808, 0, 9223372036854775807}
	f.Root().CreateDataset("int64_data", int64Data)

	// Flush and close
	fmt.Println("Flushing...")
	f.Flush()

	fmt.Printf("Edge test file created: %s\n", path)
}
