#[rustfmt::skip]
pub const INDEXING: &str = include_str!(concat!(env!("OUT_DIR"), "/src/kernels/indexing.ptx"));
#[rustfmt::skip]
pub const CUSTOM_BINARY: &str = include_str!(concat!(env!("OUT_DIR"), "/src/kernels/custom_binary.ptx"));
