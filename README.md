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
- **结构**: 组, 多层嵌套组 (已测试支持 100 层), 软链接, 外部链接
- **属性**: 组和数据集上的属性, 标量和数组, 复合类型
- **文件格式**: Superblock 版本 0-3

### 尚未支持

- SZIP 压缩
- 部分读取 (hyperslabs)
- 虚拟数据集
- 对象/区域引用

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

## HDF5 文件 8 字节对齐规则

HDF5 文件格式要求多个部分遵循 8 字节对齐规则，这是保证跨平台兼容性和高效内存访问的重要设计。

### 对齐规则汇总

| 组件 | 对齐要求 | 代码位置 |
|------|---------|---------|
| **全局堆对象 (Global Heap)** | 对象数据和整个堆集合都填充到 8 字节边界 | [internal/heap/global.go](file:///d:/code/go-hdf5/internal/heap/global.go#L104-L106), [internal/heap/global_write.go](file:///d:/code/go-hdf5/internal/heap/global_write.go#L57-L67) |
| **属性消息 V1 (Attribute V1)** | 名称、数据类型、数据空间后面都填充到 8 字节边界 | [internal/message/attribute.go](file:///d:/code/go-hdf5/internal/message/attribute.go#L66-L98) |
| **属性消息 V2/V3** | 无 8 字节对齐要求 | [internal/message/attribute_write.go](file:///d:/code/go-hdf5/internal/message/attribute_write.go) |
| **数据类型名称 (Datatype V1/V2)** | 复合类型成员名称后面填充到 8 字节边界 | [internal/message/datatype.go](file:///d:/code/go-hdf5/internal/message/datatype.go#L294) |
| **对象头部分配** | 使用 8 字节对齐分配 | [internal/alloc/allocator.go](file:///d:/code/go-hdf5/internal/alloc/allocator.go#L107-L119) |
| **二进制读写器** | 提供 `Align(8)` 和 `WritePadding(8)` 方法 | [internal/binary/reader.go](file:///d:/code/go-hdf5/internal/binary/reader.go#L217), [internal/binary/writer.go](file:///d:/code/go-hdf5/internal/binary/writer.go#L176) |

### 对齐计算方式

对齐填充字节数的计算公式为：
```go
padding := (8 - (size % 8)) % 8
```

其中 `size` 是当前数据的字节长度。如果 `size` 已经是 8 的倍数，则 `padding` 为 0。

### 全局堆对象对齐示例

```go
// 对象数据写入后，计算填充
dataSize := len(obj)
padding := (8 - (dataSize % 8)) % 8
if padding > 0 {
    w.WriteBytes(make([]byte, padding)) // 写入零字节填充
}
```

### 属性 V1 对齐示例

属性 V1 格式在每个主要部分后都需要对齐：

```
[版本(1)] [标志(1)] [名称大小(2)] [类型大小(2)] [空间大小(2)]
[名称...] [填充到8字节]
[数据类型...] [填充到8字节]
[数据空间...] [填充到8字节]
[属性数据...]
```

---

## HDF5 文件写入流程

写入 HDF5 文件遵循以下步骤：

### 文件创建阶段

```
1. os.Create(path)                    创建操作系统文件
        ↓
2. binary.NewWriter(osFile, cfg)      创建二进制写入器
        ↓
3. superblock.NewSuperblock()         创建 Superblock (V2)
        ↓
4. 计算 root group 地址 = superblock 大小
        ↓
5. object.NewEmptyGroupHeader()       创建空 root group header
        ↓
6. 计算 EOF 地址 = superblock + header 大小
        ↓
7. sb.Write(writer)                   写入 Superblock
        ↓
8. object.WriteHeader(...)            写入 root group header
        ↓
9. alloc.New(eofAddr)                 创建空间分配器
        ↓
10. 返回 File 对象
```

### 组创建阶段 (`CreateGroup`)

```
1. object.NewEmptyGroupHeader()       创建空组消息 (LinkInfo + GroupInfo)
        ↓
2. object.HeaderSize(...)             计算 header 大小
        ↓
3. allocator.Alloc(...)               分配空间
        ↓
4. object.WriteHeader(...)            写入组 header
        ↓
5. message.NewHardLink(name, addr)    创建硬链接
        ↓
6. parent.addLink(link)               添加到父组
        ↓
7. 返回 Group 对象
```

### 数据集创建阶段 (`CreateDataset`)

```
1. inferDimensionsAndType(data)       推断维度和元素类型
        ↓
2. dtype.GoTypeToDatatype(elemType)   创建 HDF5 数据类型
        ↓
3. message.NewDataspace(dims)        创建数据空间
        ↓
4. dtype.Encode(datatype, data)       编码数据为字节
        ↓
5. 确定存储布局 (连续或分块)
        ├─ 连续: 分配空间 → 写入数据 → NewContiguousLayout
        └─ 分块: 切分数据 → 写入块 → NewChunkedLayout
        ↓
6. object.NewDatasetHeader(...)       创建数据集消息
        ↓
7. allocator.Alloc(...)               分配 header 空间
        ↓
8. object.WriteHeader(...)            写入数据集 header
        ↓
9. parent.addLink(link)               创建硬链接到父组
        ↓
10. 返回 Dataset 对象
```

### 刷新阶段 (`Flush`)

```
1. 更新 superblock.EOFAddress         设置当前 EOF
        ↓
2. writer.At(0)                       定位到文件开头
        ↓
3. sb.Write(writer)                   重写 Superblock
        ↓
4. file.Sync()                        同步到磁盘
```

---

## HDF5 文件读取流程

读取 HDF5 文件遵循以下步骤：

### 文件打开阶段

```
1. os.Open(path)                      打开操作系统文件
        ↓
2. superblock.Read(file)              搜索并读取 Superblock
   ├─ 在偏移位置 0, 512, 1024, 2048 搜索签名
   ├─ 根据版本选择 readV0/V1/V2/V3
   └─ 返回 Superblock 结构
        ↓
3. binary.NewReader(file, cfg)        创建二进制读取器
   ├─ 使用 Superblock 的配置
   ├─ 字节序: LittleEndian
   ├─ OffsetSize: 2/4/8 字节
   └─ LengthSize: 2/4/8 字节
        ↓
4. openGroupAt(rootAddr, "/")         加载 root group
        ↓
5. 返回 File 对象
```

### 对象读取阶段 (`object.Read`)

```
1. reader.At(address)                 定位到对象地址
        ↓
2. Peek(4) 检查签名
   ├─ "OHDR" → readV2()               V2 对象头
   └─ 版本字节 1 → readV1()            V1 对象头
        ↓
3. 解析 header 结构
   ├─ 签名、版本、标志
   ├─ 引用计数
   └─ 消息列表
        ↓
4. 遍历消息并调用 message.Parse()     解析每条消息
   ├─ TypeDataspace → parseDataspace
   ├─ TypeDatatype → parseDatatype
   ├─ TypeDataLayout → parseDataLayout
   ├─ TypeAttribute → parseAttribute
   └─ TypeLink → parseLink
        ↓
5. 返回 Header 结构
```

### 数据集读取阶段

```
1. OpenDataset(path)                  解析路径并定位数据集
        ↓
2. object.Read(reader, address)       读取数据集对象头
        ↓
3. header.Dataspace()                 获取维度信息
        ↓
4. header.Datatype()                  获取数据类型
        ↓
5. header.DataLayout()                获取存储布局
        ↓
6. 根据布局读取数据
   ├─ 连续布局: 直接读取数据
   ├─ 分块布局: 读取 B-tree/索引 → 读取块数据 → 解压缩
   └─ 紧凑布局: 数据存储在 header 中
        ↓
7. dtype.Decode(datatype, rawData)    解码为 Go 类型
        ↓
8. 返回数据
```

### 组读取阶段

```
1. OpenGroup(path)                    解析路径并定位组
        ↓
2. object.Read(reader, address)       读取组对象头
        ↓
3. 获取 Link 消息                      获取成员链接
        ↓
4. 通过 B-tree 或链接消息遍历成员
   ├─ V1 组: 使用符号表和 B-tree v1
   └─ V2 组: 使用对象头中的链接消息和 B-tree v2
        ↓
5. 返回成员列表
```

### 属性读取阶段

```
1. obj.Attr(name)                     在对象头中查找属性
        ↓
2. 解析属性消息
   ├─ 名称
   ├─ 数据类型
   ├─ 数据空间
   └─ 属性数据
        ↓
3. 根据数据类型解码值
        ↓
4. 返回 Attribute 对象
```

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