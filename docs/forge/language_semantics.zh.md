# 分支变量作用域

[English](language_semantics.en.md) · [文档入口](index.zh.md)

这是待发布的 FG-LANG-01 增强，已发布 0.7.0 仍要求在运行期分支外预先定义变量。
当前源码允许运行期 `if/else` 的两个分支定义相同名称的变量，并在分支后读取：

```python
@ti.kernel
def choose(flag: ti.i32) -> ti.i32:
    if flag:
        value = 1
    else:
        value = 2
    return value
```

- 新变量必须在两个分支都定义；嵌套分支按同样规则合并。仅单侧赋值、只在循环内赋值，
  或引用另一分支的局部变量，仍会报错。不推断提前返回使某条路径不可达。
- 合并要求类型和形状完全一致，不隐式提升不同分支的类型。支持标量、向量、矩阵、
  匹配成员与方法的结构体，以及由这些运行期值组成的列表/元组。
- 两边同名的新运行期变量会合并，即使后续未读取它；不同类型的同名临时变量应改名，
  或在分支前明确定义共同变量。已有外层变量的赋值和转换规则保持原样。
- `ti.static` 条件及静态 Python 对象保持原来的块作用域；不变成运行期条件值。
- 分支值在各自路径内写入公共局部存储，不额外求值未选分支。已有 AD 处理公共局部变量。

规则参考 [Triton 的全路径赋值要求](https://triton-lang.org/main/python-api/triton-semantics.html)
和 [Warp 的分支合并实现](https://github.com/NVIDIA/warp/blob/main/warp/_src/codegen.py)。
这是语言增强，不将原 Taichi 的块作用域认定为缺陷。
