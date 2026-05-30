//! Textbook serial multi-echelon inventory system (Clark & Scarf 1960).
//!
//! This family is the CLEAN, literature-faithful serial model: N stages in series,
//! i.i.d. customer demand at the most downstream stage, deterministic lead times,
//! linear holding and backorder costs, optimal echelon base-stock policy. It is named
//! for exactly what it is and is verified to reproduce the published optimum by both an
//! exact solver and env simulation, so policies can be trained on `env.rs` with
//! confidence that the dynamics match the literature.
//!
//! It is deliberately separate from `general_network`, which implements the richer
//! Pirhooshyaran & Snyder (2021) general supply-network model (per-node production
//! steps and pipeline holding) and does NOT reduce to this textbook serial system.

pub mod echelon_base_stock;
pub mod env;
pub mod exact;
pub mod verification;
