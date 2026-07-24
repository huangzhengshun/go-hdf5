package main

import (
	"fmt"

	"github.com/huangzhengshun/go-hdf5/hdf5"
)

func check(condition bool, msg string) {
	if condition {
		fmt.Printf("  PASS: %s\n", msg)
	} else {
		fmt.Printf("  FAIL: %s\n", msg)
		panic(msg)
	}
}

func main() {
	path := "roundtrip_test.h5"

	// Write phase
	fmt.Println("=== Write Phase ===")
	f, err := hdf5.Create(path)
	if err != nil {
		fmt.Printf("Failed to create file: %v\n", err)
		return
	}

	// Write various datasets
	int32Data := []int32{1, 2, 3, 4, 5}
	f.Root().CreateDataset("int32_data", int32Data)

	float64Data := []float64{1.1, 2.2, 3.3}
	f.Root().CreateDataset("float64_data", float64Data)

	uint8Data := []uint8{0, 127, 255}
	f.Root().CreateDataset("uint8_data", uint8Data)

	boolData := []bool{true, false, true}
	f.Root().CreateDataset("bool_data", boolData)

	strData := []string{"hello", "world"}
	f.Root().CreateDataset("string_data", strData)

	// Write attributes
	f.Root().CreateAttribute("int_attr", int32(42))
	f.Root().CreateAttribute("float_attr", float64(3.14))
	f.Root().CreateAttribute("bool_attr", true)
	f.Root().CreateAttribute("str_attr", "test")

	// Create nested group
	nested, _ := f.Root().CreateGroup("nested")
	nested.CreateDataset("nested_data", []int64{10, 20, 30})
	nested.CreateAttribute("nested_attr", int32(99))

	f.Flush()
	f.Close()

	// Read phase
	fmt.Println("\n=== Read Phase ===")
	f2, err := hdf5.Open(path)
	if err != nil {
		fmt.Printf("Failed to open file: %v\n", err)
		return
	}
	defer f2.Close()

	// Read int32_data
	ds, _ := f2.OpenDataset("int32_data")
	var readInt32 []int32
	ds.Read(&readInt32)
	check(len(readInt32) == 5, "int32_data length")
	check(readInt32[0] == 1 && readInt32[4] == 5, "int32_data values")
	fmt.Printf("  int32_data: %v\n", readInt32)

	// Read float64_data
	ds, _ = f2.OpenDataset("float64_data")
	var readFloat64 []float64
	ds.Read(&readFloat64)
	check(len(readFloat64) == 3, "float64_data length")
	check(readFloat64[0] == 1.1 && readFloat64[1] == 2.2, "float64_data values")
	fmt.Printf("  float64_data: %v\n", readFloat64)

	// Read uint8_data
	ds, _ = f2.OpenDataset("uint8_data")
	var readUint8 []uint8
	ds.Read(&readUint8)
	check(len(readUint8) == 3, "uint8_data length")
	check(readUint8[0] == 0 && readUint8[2] == 255, "uint8_data values")
	fmt.Printf("  uint8_data: %v\n", readUint8)

	// Read bool_data
	ds, _ = f2.OpenDataset("bool_data")
	var readBool []bool
	ds.Read(&readBool)
	check(len(readBool) == 3, "bool_data length")
	check(readBool[0] == true && readBool[1] == false, "bool_data values")
	fmt.Printf("  bool_data: %v\n", readBool)

	// Read string_data
	ds, _ = f2.OpenDataset("string_data")
	var readStr []string
	ds.Read(&readStr)
	check(len(readStr) == 2, "string_data length")
	check(readStr[0] == "hello" && readStr[1] == "world", "string_data values")
	fmt.Printf("  string_data: %v\n", readStr)

	// Read attributes
	attr := f2.Root().Attr("int_attr")
	var intVal int32
	attr.Read(&intVal)
	check(intVal == 42, "int_attr")

	attr = f2.Root().Attr("float_attr")
	var floatVal float64
	attr.Read(&floatVal)
	check(floatVal == 3.14, "float_attr")

	attr = f2.Root().Attr("bool_attr")
	var boolVal bool
	attr.Read(&boolVal)
	check(boolVal == true, "bool_attr")

	attr = f2.Root().Attr("str_attr")
	var strVal string
	attr.Read(&strVal)
	check(strVal == "test", "str_attr")

	// Read nested group
	ng, _ := f2.OpenGroup("nested")
	ds, _ = ng.OpenDataset("nested_data")
	var nestedData []int64
	ds.Read(&nestedData)
	check(len(nestedData) == 3, "nested_data length")
	check(nestedData[0] == 10 && nestedData[2] == 30, "nested_data values")

	nestedAttr := ng.Attr("nested_attr")
	var nestedAttrVal int32
	nestedAttr.Read(&nestedAttrVal)
	check(nestedAttrVal == 99, "nested_attr")

	fmt.Println("\n=== Go Round-trip Test Completed ===")
}
