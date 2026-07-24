package main

import (
	"fmt"
	"os"

	"github.com/huangzhengshun/go-hdf5/hdf5"
)

func main() {
	path := "comprehensive_test.h5"
	os.Remove(path)

	f, err := hdf5.Create(path)
	if err != nil {
		fmt.Printf("Create failed: %v\n", err)
		os.Exit(1)
	}

	// Root attributes
	f.Root().CreateAttribute("version", int32(1))
	f.Root().CreateAttribute("description", "Comprehensive h5py test file")

	// Group 1: integers
	intGroup, _ := f.Root().CreateGroup("integers")
	intGroup.CreateAttribute("category", "integer types")

	int32Data := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	intGroup.CreateDataset("int32_1d", int32Data,
		hdf5.WithAttribute("units", "count"),
		hdf5.WithAttribute("min", int32(1)),
		hdf5.WithAttribute("max", int32(10)),
	)

	int64Data := []int64{-100, -50, 0, 50, 100}
	intGroup.CreateDataset("int64_1d", int64Data,
		hdf5.WithAttribute("type", "signed"),
	)

	// Group 2: floats
	floatGroup, _ := f.Root().CreateGroup("floats")
	floatGroup.CreateAttribute("category", "floating point types")

	float32Data := []float32{1.5, 2.5, 3.5, 4.5}
	floatGroup.CreateDataset("float32_1d", float32Data,
		hdf5.WithAttribute("precision", "single"),
	)

	float64Data := []float64{3.141592653589793, 2.718281828459045}
	floatGroup.CreateDataset("float64_1d", float64Data,
		hdf5.WithAttribute("precision", "double"),
		hdf5.WithAttribute("constants", "pi,e"),
	)

	// Group 3: nested groups
	nestedGroup, _ := f.Root().CreateGroup("nested")
	nestedGroup.CreateAttribute("level", int32(1))

	level2a, _ := nestedGroup.CreateGroup("level2a")
	level2a.CreateAttribute("name", "level2a")
	level2a.CreateAttribute("value", float64(42.0))

	level2b, _ := nestedGroup.CreateGroup("level2b")
	level2b.CreateAttribute("name", "level2b")

	level3, _ := level2a.CreateGroup("level3")
	level3.CreateAttribute("depth", int32(3))
	level3Data := []int32{10, 20, 30}
	level3.CreateDataset("deep_data", level3Data,
		hdf5.WithAttribute("location", "deep"),
	)

	// Group 4: empty groups
	emptyGroup, _ := f.Root().CreateGroup("empty_groups")
	emptyGroup.CreateGroup("totally_empty")
	withAttr, _ := emptyGroup.CreateGroup("only_attrs")
	withAttr.CreateAttribute("has_data", int32(0))
	withAttr.CreateAttribute("purpose", "attribute only test")

	// Group 5: multiple datasets
	multiGroup, _ := f.Root().CreateGroup("multiple_datasets")
	multiGroup.CreateAttribute("dataset_count", int32(4))
	for i := 0; i < 4; i++ {
		data := make([]int32, 5)
		for j := 0; j < 5; j++ {
			data[j] = int32(i*10 + j)
		}
		multiGroup.CreateDataset(fmt.Sprintf("ds_%d", i), data,
			hdf5.WithAttribute("index", int32(i)),
		)
	}

	// Group 6: strings (string attributes only for now)
	strGroup, _ := f.Root().CreateGroup("strings")
	strGroup.CreateAttribute("encoding", "ascii")
	strGroup.CreateAttribute("count", int32(3))
	strGroup.CreateAttribute("sample", "hello world")
	// String datasets use variable-length strings which require global heap support
	// strData := []string{"hello", "world", "hdf5"}
	// strGroup.CreateDataset("string_array", strData)

	if err := f.Close(); err != nil {
		fmt.Printf("Close failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("HDF5 file created:", path)
}
