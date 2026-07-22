package main

import (
	"fmt"

	"github.com/huangzhengshun/go-hdf5/hdf5"
)

func main() {
	path := "nested_groups_test.h5"

	f, err := hdf5.Create(path)
	if err != nil {
		fmt.Printf("Create failed: %v\n", err)
		return
	}
	defer f.Close()

	root := f.Root()

	level1, err := root.CreateGroup("level1")
	if err != nil {
		fmt.Printf("CreateGroup level1 failed: %v\n", err)
		return
	}
	err = level1.CreateAttribute("level", int64(1))
	if err != nil {
		fmt.Printf("CreateAttribute level on level1 failed: %v\n", err)
		return
	}
	err = level1.CreateAttribute("description", "First level group")
	if err != nil {
		fmt.Printf("CreateAttribute description on level1 failed: %v\n", err)
		return
	}
	err = level1.CreateAttribute("count", int64(10))
	if err != nil {
		fmt.Printf("CreateAttribute count on level1 failed: %v\n", err)
		return
	}

	level2, err := level1.CreateGroup("level2")
	if err != nil {
		fmt.Printf("CreateGroup level2 failed: %v\n", err)
		return
	}
	err = level2.CreateAttribute("level", int64(2))
	if err != nil {
		fmt.Printf("CreateAttribute level on level2 failed: %v\n", err)
		return
	}
	err = level2.CreateAttribute("data_count", int64(100))
	if err != nil {
		fmt.Printf("CreateAttribute data_count on level2 failed: %v\n", err)
		return
	}
	err = level2.CreateAttribute("tags", []string{"group", "data"})
	if err != nil {
		fmt.Printf("CreateAttribute tags on level2 failed: %v\n", err)
		return
	}

	level3, err := level2.CreateGroup("level3")
	if err != nil {
		fmt.Printf("CreateGroup level3 failed: %v\n", err)
		return
	}
	err = level3.CreateAttribute("level", int64(3))
	if err != nil {
		fmt.Printf("CreateAttribute level on level3 failed: %v\n", err)
		return
	}
	err = level3.CreateAttribute("tags", []string{"important", "test", "nested"})
	if err != nil {
		fmt.Printf("CreateAttribute tags on level3 failed: %v\n", err)
		return
	}
	err = level3.CreateAttribute("scores", []float64{95.5, 88.0, 76.3, 82.1})
	if err != nil {
		fmt.Printf("CreateAttribute scores on level3 failed: %v\n", err)
		return
	}

	_, err = level3.CreateDataset("data", []int{1, 2, 3, 4, 5}, nil)
	if err != nil {
		fmt.Printf("CreateDataset data on level3 failed: %v\n", err)
		return
	}

	if err := f.Flush(); err != nil {
		fmt.Printf("Flush failed: %v\n", err)
		return
	}

	fmt.Printf("Successfully created %s\n", path)
}
