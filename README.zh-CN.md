# dace-rs

[DACE](https://github.com/dacelib/dace)（Differential Algebra Computational Toolbox，微分代数计算工具箱）的纯 Rust 实现。

`dace-rs` 使用截断的多变量 Taylor 多项式（“微分代数”）进行计算：通过算术运算、初等函数复合以及完整的多项式映射求逆，传播任意多变量函数的高阶展开，并支持可配置的截断 epsilon 与截断阶数。这是对 DACE 2.1 的全量安全 Rust 重写——构建时无需 C 工具链，计算内核中无 unsafe 代码。

[English](README.md) | **简体中文**

[![CI](https://github.com/cislunarspace/dace-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/cislunarspace/dace-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/dace-rs)](https://crates.io/crates/dace-rs)
[![Docs.rs](https://img.shields.io/docsrs/dace-rs)](https://docs.rs/dace-rs)
[![License](https://img.shields.io/crates/l/dace-rs)](#许可证)

## 为什么用 dace-rs

crates.io 上已有的 [`dace`](https://crates.io/crates/dace) 是对上游 C 库的绑定封装：需要 C 工具链并链接 C 内核。`dace-rs` 是对相同算法与数值行为的独立纯 Rust 实现，并与 C 参考实现逐系数对照验证（见[对照验证方法](#对照验证方法)）。同时提供 Rust 原生的人体工学：

- `Da × Da`、`Da × f64`、`f64 × Da` 的运算符 trait（`Add`/`Sub`/`Mul`/`Div`/`Neg` 及赋值形式）；
- DA 值在重新初始化后仍然有效（每个 `Da` 持有自己的上下文；C 会清空一切）；
- 可恢复的失败返回 `Result`，定义域误用携带 C 错误码 panic，信息性降级通过 [`log`] crate 输出——没有粘性错误状态，没有 `exit(1)`；
- 自动域分裂（[`ads::split`]，Rust-only 扩展，C 无对应）通过递归二分与逐块重展开，在大不确定度盒上保持多项式包围的严格性。

## 快速上手

```rust
use dace_rs::Da;

fn main() {
    dace_rs::init(20, 2).unwrap();                 // 20 阶，2 个变量
    let x = Da::variable(1);
    let y = Da::variable(2);

    let f = (1.0 + x.clone() * y.clone()).sin();   // sin(1+xy) 的 Taylor 展开
    println!("{f}");                               // daceWrite 风格的系数列表

    // 系数检查：sin(1+xy) 中 x*y 项的系数为 cos(1)
    assert!((f.get_coefficient(&[1, 1]) - 1.0f64.cos()).abs() < 1e-14);
}
```

映射迭代的完整示例见 [`examples/quickstart.rs`](examples/quickstart.rs)，完整 API 见[ crate 文档](https://docs.rs/dace-rs)。

## 平台支持

所有依赖均为纯 Rust；没有 `build.rs`，不需要 C 工具链。

| 目标平台 | CI |
|---|---|
| Linux x86-64 | `ubuntu-latest` |
| Linux aarch64 | `ubuntu-24.04-arm` |
| Windows x86-64 | `windows-latest` |
| Windows aarch64 | `windows-11-arm` |

算术内核使用固定的累加顺序与默认浮点设置（无 fast-math、无 FMA 收缩），四个平台上的结果逐位一致；只有标量超越函数（`f64::sin` 等）可能因 libm 实现差异相差几个 ulp。

## API 对照（C++ → Rust）

| DACE C++ | dace-rs |
|---|---|
| `DA::DA(var)`、`DA::identity` | [`Da::variable`] / [`Da::identity`] |
| `DA::cons`、`DA::linear`、`DA::gradient` | [`Da::cons`] / [`Da::linear`] / [`Da::gradient`] |
| `DA::getCoefficient`、`setCoefficient` | [`Da::get_coefficient`] / [`Da::set_coefficient`] |
| `DA::deriv`、`DA::integ`、`DA::trim` | [`Da::deriv`] / [`Da::integ`] / [`Da::trim`] |
| `exp, log, sin, tan, asin, ...` | [`dace_rs::exp`] 等自由函数及 `Da` 方法 |
| `BesselJFunction` 等 | [`dace_rs::bessel_j`] 等 |
| `GammaFunction`、`PsiFunction` | [`dace_rs::gamma`] / [`dace_rs::psi`] |
| `DA::norm`、`orderNorm`、`estimNorm`、`bound`、`convRadius` | [`Da::norm`] 等 |
| `DA::compile`、`compiledDA::eval` | [`Da::compile`] / [`CompiledDa::eval`] / [`CompiledDa::eval_da`] |
| `DA::plug`、`DA::eval` | [`Da::plug`] / [`Da::eval`] / [`Da::eval_da`] |
| `DA::replaceVariable`、`scaleVariable`、`translateVariable` | [`Da::replace_variable`] 等 |
| `DA::read`/`write`（blob）、`operator>>`/`<<` | [`Da::to_blob`]/[`Da::from_blob`]、[`Display`]/[`FromStr`] |
| `DASimpleFormatter` | [`SimpleFormat`] 预设与 [`format_da`] |
| `AlgebraicVector<DA>::invert` | [`DaVector::invert`] |
| ——（无 C 对应；Rust-only 扩展） | [`ads::split`] 及 [`AdsConfig`]/[`AdsResult`] |

未移植（上游实验性/默认关闭）：`AlgebraicMatrix`、`dacecompat` 别名、MATLAB 接口。

## 对照验证方法

C 库本身没有测试套件，因此 `dace-rs` 直接与 C 参考实现对照验证：

- **黄金数据**：`dev/golden/gen.sh` 构建 C 库并运行固定用例表（`dev/golden/main.c`），覆盖算术、全部初等与特殊函数、微积分、范数、求值和 blob 导出，上下文包括 `(6,3)`、`(20,6)`、`(6,2)` 及多个常数项取值。提交入库的 `tests/golden/cases.txt` 由 CI 中的 `tests/parity.rs` 回放（无需 C 工具链）；系数在 rtol `1e-10` 内一致（算术类 `1e-13`，多数情形已验证逐位一致）。
- **开发过程中的位级检查**：单项式编码表、乘法/求逆内核、文本格式与初等函数层均在本机与 C 库逐字节比对。

上游 C 库的两个 bug 被有意地不复现；两处分歧均记录在相关函数的 rustdoc 中：

1. `daceReplaceVariable` 用 1 基变量号索引 0 基指数数组（文档语义是 1 基替换；实际实现相当于 `from+1 → val·(to+1)`，且 `from == nvmax` 时静默无效）。dace-rs 实现文档语义。
2. `dacePower` 负数次幂在混叠结果上调用乘法逆，而 Newton 迭代并不混叠安全；C 返回错误系数（C 的 `pow(A,-2)` 与 C 自己的 `minv(sqr(A))` 不一致）。dace-rs 返回正确值。

## 性能基准

核心内核的 criterion 基准位于 `benches/kernels.rs`。`cargo bench` 运行全套（或 `cargo bench --bench kernels` 只跑该目标）；`cargo bench mul/20x6` 可过滤单个基准。

乘法、初等函数（`sin`、`sqrt`、`exp`）、编译与编译求值各有两档配置：6 阶 2 变量（28 个单项式槽位）与 20 阶 6 变量（230,230 个槽位）。映射求逆只测 6/2 与 10/4 两档：20 阶 6 变量下单次近恒等映射求逆约需 29 秒（下述基线机器实测，release 构建），故只记录量级于此，不纳入常规 bench 循环。

基线（criterion 默认设置，Linux x86-64，AMD Ryzen Threadripper 9960X，`bench`/release 配置，median）：

| 基准 | 配置 | Median |
|---|---|---|
| mul/6x2 | 6 阶 2 变量 | 124 ns |
| mul/20x6 | 20 阶 6 变量 | 137 µs |
| sin/6x2 | 6 阶 2 变量 | 969 ns |
| sin/20x6 | 20 阶 6 变量 | 2.68 ms |
| sqrt/6x2 | 6 阶 2 变量 | 973 ns |
| sqrt/20x6 | 20 阶 6 变量 | 2.61 ms |
| exp/6x2 | 6 阶 2 变量 | 969 ns |
| exp/20x6 | 20 阶 6 变量 | 2.64 ms |
| compile/6x2 | 6 阶 2 变量 | 198 ns |
| compile/20x6 | 20 阶 6 变量 | 97.4 µs |
| eval/6x2 | 6 阶 2 变量 | 23.5 ns |
| eval/20x6 | 20 阶 6 变量 | 33.4 ns |
| invert/6x2 | 6 阶 2 变量 | 6.64 µs |
| invert/10x4 | 10 阶 4 变量 | 979 µs |

criterion 结果落盘于 `target/criterion/`；再次运行 `cargo bench` 会自动输出相对上次的变化（`change: [...]`），作为内核改动前后对照的推荐方式。

口径说明与[平台支持](#平台支持)一致：算术内核固定累加顺序、默认浮点设置；微基准不代表普适性能结论。

## 许可证

Apache-2.0。`dace-rs` 是 DACE 的衍生作品（Copyright 2016 Politecnico di Milano 及贡献者；Copyright 2014 Dinamica Srl）。参见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

[`Da::variable`]: https://docs.rs/dace-rs/latest/dace_rs/struct.Da.html#method.variable
[`CompiledDa::eval`]: https://docs.rs/dace-rs/latest/dace_rs/struct.CompiledDa.html#method.eval
[`SimpleFormat`]: https://docs.rs/dace-rs/latest/dace_rs/io/struct.SimpleFormat.html
[`DaVector::invert`]: https://docs.rs/dace-rs/latest/dace_rs/vector/trait.DaVector.html#tymethod.invert
[`ads::split`]: https://docs.rs/dace-rs/latest/dace_rs/ads/fn.split.html
[`AdsConfig`]: https://docs.rs/dace-rs/latest/dace_rs/ads/struct.AdsConfig.html
[`AdsResult`]: https://docs.rs/dace-rs/latest/dace_rs/ads/struct.AdsResult.html
[`log`]: https://docs.rs/log
