<!-- 这一行是 HTML 注释，应该被清洗掉 -->
<!--
  多行 HTML 注释
  也应该被清洗掉
-->

[![Build](https://img.shields.io/badge/build-passing-green.svg)](https://example.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://example.com)

# HashMap 底层原理

这是一段前言，位于第一个二级标题之前，但已经在 H1 之下。

## 数据结构

JDK 8 之后，HashMap 采用 **数组 + 链表 + 红黑树** 的结构。

![结构图](assets/hashmap.png)

当链表长度超过 8 且数组长度大于等于 64 时，链表会转换为红黑树。

## 扩容机制

扩容触发条件是 `size > threshold`，其中 `threshold = capacity * loadFactor`。

```java
// 下面这段代码里有一个井号开头的注释，不应该被误判为标题
final Node<K,V>[] resize() {
    Node<K,V>[] oldTab = table;
    int oldCap = (oldTab == null) ? 0 : oldTab.length;
    // # 这行以井号开头，但在代码块内
    if (oldCap > 0) {
        if (oldCap >= MAXIMUM_CAPACITY) {
            threshold = Integer.MAX_VALUE;
            return oldTab;
        }
    }
    return newTab;
}
```

### 为什么是 0.75

<!-- 表格前的注释 -->

| 负载因子 | 冲突概率 | 空间利用率 |
|---------|---------|-----------|
| 0.5     | 低      | 低        |
| 0.75    | 中      | 中        |
| 1.0     | 高      | 高        |

0.75 是时间和空间成本的折中，来源于泊松分布的计算。

### 为什么是 8

树化阈值 8 的推导见源码注释，理想情况下链表长度服从泊松分布。

## 线程安全

HashMap 不是线程安全的。多线程并发扩容可能导致**死循环**（JDK 7 头插法）
或数据丢失（JDK 8 尾插法）。

> 想要线程安全，用 `ConcurrentHashMap`。

```text
注意：代码块里出现了三个反引号
```
上面这个代码块结束了吗？结束了。
