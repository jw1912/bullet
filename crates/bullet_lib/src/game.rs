/// Contains the `InputType` trait for implementing custom input types,
/// as well as several premade input formats that are commonly used.
pub mod inputs;
/// Contains the `OutputBuckets` trait for implementing custom output bucket types,
/// as well as several premade output buckets that are commonly used.
pub mod outputs;

/// Contains data formats
pub mod formats {
    pub use bulletformat;
    pub use montyformat;
    pub use sfbinpack;
}
