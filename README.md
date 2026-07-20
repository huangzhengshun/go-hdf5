# go-hdf5

一个纯 Go 语言编写的 HDF5 文件读取库，无需 CGO 或外部依赖。

## 安装

```bash
go get github.com/huangzhengshun/go-hdf5
```

## 快速开始

```go
package main

import (
    "fmt"
    "log"

    "github.com/huangzhengshun/go-hdf5/hdf5"
)

func main() {
    // 打开 HDF5 文件
    f, err := hdf5.Open("data.h5")
    if err != nil {
        log.Fatal(err)
    }
    defer f.Close()

    // 读取数据集
    ds, err := f.OpenDataset("/measurements/temperature")
    if err != nil {
        log.Fatal(err)
    }

    // 获取数据集信息
    fmt.Printf("Shape: %v\n", ds.Shape())
    fmt.Printf("Elements: %d\n", ds.NumElements())

    // 以 float64 类型读取数据
    data, err := ds.ReadFloat64()
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Data: %v\n", data)
}
```

## 功能特性

### 已支持

- **数据类型**: 所有整数类型 (int8-64, uint8-64), float32, float64, 字符串 (定长和变长)
- **存储布局**: 连续存储, 分块存储 (B-tree v1 和 v2), 紧凑存储
- **压缩**: Gzip/deflate, shuffle 过滤器
- **结构**: 组, 嵌套组, 软链接, 外部链接
- **属性**: 组和数据集上的属性, 标量和数组, 复合类型
- **文件格式**: Superblock 版本 0-3

### 尚未支持

- SZIP 压缩
- 部分读取 (hyperslabs)
- 虚拟数据集
- 对象/区域引用
- 写入文件

## 使用示例

### 读取数据集

```go
// 通过路径打开数据集
ds, err := f.OpenDataset("/group/subgroup/data")

// 检查数据集属性
fmt.Printf("Dimensions: %v\n", ds.Shape())   // 例如, [100, 200]
fmt.Printf("Rank: %d\n", ds.Rank())          // 例如, 2
fmt.Printf("Total elements: %d\n", ds.NumElements())

// 使用类型特定的方法读取
floats, _ := ds.ReadFloat64()
ints, _ := ds.ReadInt32()
strings, _ := ds.ReadString()

// 或者读取到类型化切片中
var data []float64
err := ds.Read(&data)
```

### 遍历组

```go
// 获取根组
root := f.Root()

// 打开子组
grp, err := root.OpenGroup("sensors")

// 列出所有成员 (组和数据集)
members, err := grp.Members()
for _, name := range members {
    fmt.Println(name)
}

// 直接打开嵌套路径
ds, err := f.OpenDataset("/sensors/temperature/readings")
```

### 读取属性

```go
// 在数据集上
ds, _ := f.OpenDataset("/data")
attr := ds.Attr("units")
if attr != nil {
    units, _ := attr.ReadScalarString()
    fmt.Printf("Units: %s\n", units)
}

// 在组上
grp, _ := f.OpenGroup("/experiment")
attr := grp.Attr("description")
value, _ := attr.Value()  // 根据 HDF5 数据类型自动推断类型

// 列出所有属性
for _, name := range ds.Attrs() {
    fmt.Println(name)
}

// 通过完整路径读取属性
val, err := f.ReadAttr("/data@units")
```

### 复合类型属性

```go
// 读取复合属性 (例如, 带有 x, y, z 字段的点)
attr := ds.Attr("origin")
point, err := attr.ReadScalarCompound()
// point 是 map[string]interface{}{"x": 1.0, "y": 2.0, "z": 3.0}

x := point["x"].(float64)
y := point["y"].(float64)
```

### 遍历所有属性

```go
// 遍历文件中的所有属性
err := f.WalkAttrs(func(info hdf5.AttrInfo) error {
    fmt.Printf("%s = %v\n", info.Path, info.Value)
    // info.Path 格式: "/group/dataset@attribute"
    // info.ObjectPath: "/group/dataset"
    // info.Name: "attribute"
    return nil
})
```

### 跟随链接

```go
// 软链接会自动跟随
ds, err := f.OpenDataset("/link_to_data")  // -> 解析到实际数据集

// 外部链接也支持 (自动打开外部文件)
ds, err := f.OpenDataset("/external_link")  // -> 打开 external_file.h5:/path
```

### 错误处理

```go
import "errors"

ds, err := f.OpenDataset("/nonexistent")
if errors.Is(err, hdf5.ErrNotFound) {
    fmt.Println("Dataset not found")
}

// 常见错误:
// - hdf5.ErrNotFound: 对象不存在
// - hdf5.ErrNotDataset: 尝试将组当作数据集打开
// - hdf5.ErrNotGroup: 尝试将数据集当作组打开
// - hdf5.ErrClosed: 文件已关闭
// - hdf5.ErrLinkDepth: 嵌套软链接/外部链接过多 (循环引用保护)
```

## API 参考

### File

| Method | Description |
|--------|-------------|
| `Open(path string) (*File, error)` | 打开 HDF5 文件进行读取 |
| `Close() error` | 关闭文件 |
| `Root() *Group` | 获取根组 |
| `OpenGroup(path string) (*Group, error)` | 通过绝对路径打开组 |
| `OpenDataset(path string) (*Dataset, error)` | 通过绝对路径打开数据集 |
| `GetAttr(path string) (*Attribute, error)` | 通过路径获取属性 (`/obj@attr`) |
| `ReadAttr(path string) (interface{}, error)` | 通过路径读取属性值 |
| `WalkAttrs(fn WalkAttrsFunc) error` | 遍历文件中的所有属性 |
| `Version() int` | 获取 superblock 版本 |
| `Path() string` | 获取文件路径 |

### Group

| Method | Description |
|--------|-------------|
| `Name() string` | 组名 (路径的最后一部分) |
| `Path() string` | 此组的完整路径 |
| `OpenGroup(path string) (*Group, error)` | 通过相对路径打开子组 |
| `OpenDataset(path string) (*Dataset, error)` | 通过相对路径打开数据集 |
| `Members() ([]string, error)` | 列出所有成员名称 |
| `NumObjects() (int, error)` | 成员数量 |
| `Attrs() []string` | 列出属性名称 |
| `Attr(name string) *Attribute` | 通过名称获取属性 |
| `HasAttr(name string) bool` | 检查属性是否存在 |

### Dataset

| Method | Description |
|--------|-------------|
| `Name() string` | 数据集名称 |
| `Path() string` | 此数据集的完整路径 |
| `Shape() []uint64` | 维度 (标量为 nil) |
| `Rank() int` | 维度数量 |
| `NumElements() uint64` | 元素总数 |
| `IsScalar() bool` | 如果是标量 (单个值) 返回 true |
| `DtypeSize() int` | 元素大小 (字节) |
| `Read(dest interface{}) error` | 读取到类型化切片 |
| `ReadFloat64() ([]float64, error)` | 以 float64 读取 |
| `ReadFloat32() ([]float32, error)` | 以 float32 读取 |
| `ReadInt64() ([]int64, error)` | 以 int64 读取 |
| `ReadInt32() ([]int32, error)` | 以 int32 读取 |
| `ReadInt16() ([]int16, error)` | 以 int16 读取 |
| `ReadInt8() ([]int8, error)` | 以 int8 读取 |
| `ReadUint64() ([]uint64, error)` | 以 uint64 读取 |
| `ReadUint32() ([]uint32, error)` | 以 uint32 读取 |
| `ReadUint16() ([]uint16, error)` | 以 uint16 读取 |
| `ReadUint8() ([]uint8, error)` | 以 uint8 读取 |
| `ReadString() ([]string, error)` | 以字符串读取 |
| `ReadRaw() ([]byte, error)` | 读取原始字节 |
| `Attrs() []string` | 列出属性名称 |
| `Attr(name string) *Attribute` | 获取属性 |

### Attribute

| Method | Description |
|--------|-------------|
| `Name() string` | 属性名称 |
| `Shape() []uint64` | 维度 |
| `NumElements() uint64` | 元素数量 |
| `IsScalar() bool` | 如果是标量返回 true |
| `Value() (interface{}, error)` | 自动推断类型的值 |
| `Read(dest interface{}) error` | 读取到类型化变量 |
| `ReadFloat64() ([]float64, error)` | 以 float64 读取 |
| `ReadInt64() ([]int64, error)` | 以 int64 读取 |
| `ReadString() ([]string, error)` | 以字符串读取 |
| `ReadScalarFloat64() (float64, error)` | 读取标量 float64 |
| `ReadScalarInt64() (int64, error)` | 读取标量 int64 |
| `ReadScalarString() (string, error)` | 读取标量字符串 |
| `ReadCompound() ([]map[string]interface{}, error)` | 读取复合类型 |
| `ReadScalarCompound() (map[string]interface{}, error)` | 读取标量复合类型 |

## 测试

```bash
# 生成测试文件 (需要安装 Python 及 h5py 和 numpy)
cd testdata && python3 generate.py

# 运行测试
go test ./...

# 运行测试并生成覆盖率报告
go test ./... -cover
```

## 许可证

MIT