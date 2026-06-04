//! Textbook assembly multi-echelon inventory system (Rosling 1989).
//!
//! The `assembly` *version* of the multi-echelon problem: components procured from outside
//! suppliers are assembled into one finished product facing customer demand. By Rosling
//! (1989) an assembly system is equivalent to a serial system, so the exact optimum and
//! optimal echelon base-stock policy come from the literature-verified `serial` solver via
//! the reduction in `rosling.rs`. `env.rs` is verified to reproduce that optimum by
//! simulation (`verification.rs`), so policies can be trained on it with confidence.
//!
//! Scope: equal component lead time (the clean Rosling reduction to a 2-stage serial). It is
//! a sibling of `serial` and `production_assembly_distribution_network` under `multi_echelon/`.

pub mod echelon_base_stock;
pub mod env;
pub mod references;
pub mod rosling;
pub mod verification;
