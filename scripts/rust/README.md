# Rust Build Scripts

This folder contains helper commands for the root Rust/PyO3 crate.

- `build_extension.py` builds the Python extension with `maturin develop`, using
  the root `Cargo.toml` and explicitly enabling the `python-extension` feature.

Rust-native verification should use Cargo directly from the repository root:

```bash
cargo test --manifest-path Cargo.toml -q
```

