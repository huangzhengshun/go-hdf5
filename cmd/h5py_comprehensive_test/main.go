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

	f.Root().CreateAttribute("version", int32(1))
	f.Root().CreateAttribute("description", "Comprehensive h5py test file")

	// Group 1: integers with bool attributes
	intGroup, _ := f.Root().CreateGroup("integers")
	intGroup.CreateAttribute("category", "integer types")
	intGroup.CreateAttribute("enabled", true)

	int32Data := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	intGroup.CreateDataset("int32_1d", int32Data,
		hdf5.WithAttribute("units", "count"),
		hdf5.WithAttribute("min", int32(1)),
		hdf5.WithAttribute("max", int32(10)),
		hdf5.WithAttribute("valid", true),
	)

	int64Data := []int64{-100, -50, 0, 50, 100}
	intGroup.CreateDataset("int64_1d", int64Data,
		hdf5.WithAttribute("type", "signed"),
		hdf5.WithAttribute("valid", true),
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

	// Group 3: multi-dimensional datasets
	multiGroup, _ := f.Root().CreateGroup("multidim")
	multiGroup.CreateAttribute("description", "Multi-dimensional datasets")

	// 2D int32
	data2d := [][]int32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
	flattened2d := make([]int32, 9)
	idx := 0
	for _, row := range data2d {
		for _, v := range row {
			flattened2d[idx] = v
			idx++
		}
	}
	multiGroup.CreateDataset("int32_2d", flattened2d,
		hdf5.WithAttribute("shape", "3x3"),
		hdf5.WithAttribute("dimensions", int32(2)),
	)

	// Group 4: nested groups
	nestedGroup, _ := f.Root().CreateGroup("nested")
	nestedGroup.CreateAttribute("level", int32(1))

	level2a, _ := nestedGroup.CreateGroup("level2a")
	level2a.CreateAttribute("name", "level2a")
	level2a.CreateAttribute("value", float64(42.0))
	level2a.CreateAttribute("active", true)

	level2b, _ := nestedGroup.CreateGroup("level2b")
	level2b.CreateAttribute("name", "level2b")

	level3, _ := level2a.CreateGroup("level3")
	level3.CreateAttribute("depth", int32(3))
	level3Data := []int32{10, 20, 30}
	level3.CreateDataset("deep_data", level3Data,
		hdf5.WithAttribute("location", "deep"),
	)

	// Group 5: empty groups
	emptyGroup, _ := f.Root().CreateGroup("empty_groups")
	emptyGroup.CreateGroup("totally_empty")
	withAttr, _ := emptyGroup.CreateGroup("only_attrs")
	withAttr.CreateAttribute("has_data", false)
	withAttr.CreateAttribute("purpose", "attribute only test")

	// Group 6: multiple datasets
	multiDsGroup, _ := f.Root().CreateGroup("multiple_datasets")
	multiDsGroup.CreateAttribute("dataset_count", int32(4))
	for i := 0; i < 4; i++ {
		data := make([]int32, 5)
		for j := 0; j < 5; j++ {
			data[j] = int32(i*10 + j)
		}
		multiDsGroup.CreateDataset(fmt.Sprintf("ds_%d", i), data,
			hdf5.WithAttribute("index", int32(i)),
			hdf5.WithAttribute("active", i%2 == 0),
		)
	}

	// Group 7: strings with fixed-length string dataset
	strGroup, _ := f.Root().CreateGroup("strings")
	strGroup.CreateAttribute("encoding", "ascii")
	strGroup.CreateAttribute("count", int32(3))
	strGroup.CreateAttribute("sample", "hello world")
	strGroup.CreateAttribute("has_string_data", true)

	// Fixed-length string dataset
	strData := []string{"hello", "world", "hdf5", "test"}
	strGroup.CreateDataset("string_array", strData,
		hdf5.WithAttribute("length", int32(len(strData))),
	)

	f.Close()

	fmt.Println("Phase 1 done: initial file created")

	// Phase 2: reopen and add more content
	f2, err := hdf5.OpenReadWrite(path)
	if err != nil {
		fmt.Printf("OpenReadWrite failed: %v\n", err)
		os.Exit(1)
	}

	// Add to integers group
	intGroup2, _ := f2.OpenGroup("/integers")
	uintData := []uint32{100, 200, 300}
	intGroup2.CreateDataset("uint32_1d", uintData,
		hdf5.WithAttribute("unsigned", true),
	)

	// Add new root group
	newGroup, _ := f2.Root().CreateGroup("reopen_added")
	newGroup.CreateAttribute("added_during_reopen", true)
	newGroup.CreateAttribute("timestamp", "phase2")

	// Add dataset to new group
	newGroup.CreateDataset("reopen_data", []float64{1.1, 2.2, 3.3},
		hdf5.WithAttribute("source", "reopen"),
	)

	// Add attribute to existing group
	nestedGroup2, _ := f2.OpenGroup("/nested")
	nestedGroup2.CreateAttribute("updated", true)

	f2.Close()

	fmt.Println("Phase 2 done: additional content added")

	fmt.Println("HDF5 file created:", path)
}
