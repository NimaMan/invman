#![allow(dead_code)]

//! Literature references and verification-status flags for the `multi_echelon/assembly` family.
//!
//! ALGORITHMIC / HONESTY CONTRACT
//! ------------------------------
//! This file is the machine-checkable source of truth for what is and is NOT literature-verified
//! in this family, per `docs/rust.md` ("What counts as literature-verified": a family is
//! literature-verified ONLY when an in-crate test re-runs the env/solver and asserts the freshly
//! computed metric reproduces a number PRINTED IN A PAPER within a stated tolerance).
//!
//! VERDICT FOR THIS FAMILY: NO carried assembly instance is `literature_verified` (all rows below
//! are `literature_verified = false`). The reasons are structural and are deliberately encoded so a
//! future edit cannot silently flip a flag to true without a real published anchor:
//!
//! 1. Rosling (1989) is a STRUCTURAL result, not a worked numeric benchmark. It proves an assembly
//!    system is equivalent to a serial system (with lead-time reordering in the general case) and
//!    characterizes the optimal policy as a balanced echelon base-stock policy under "long-run
//!    balance." It does NOT tabulate an assembly optimal cost or base-stock vector that this
//!    equal-lead-time, 2-stage-reducible env reproduces. (Confirmed against the RePEc/IDEAS abstract
//!    and secondary characterizations; no paper-printed assembly cost/base-stock table is available
//!    to reproduce.)
//! 2. The ONE genuinely PUBLISHED number reachable in this verification chain is Snyder & Shen
//!    Example 6.1 optimal cost 47.65 — but that is a 3-STAGE serial system, re-derived in
//!    `multi_echelon/serial`. The Rosling reduction of an equal-lead-time assembly system yields a
//!    2-STAGE serial system (kit -> finished), which cannot reach the 3-stage Example 6.1 number.
//!
//! WHAT *IS* TRUE (the honest verification basis), strictly stronger than "self-consistent only":
//!   - the equivalence claim is literature-verified at the STRUCTURAL level (Rosling 1989);
//!   - the serial system the assembly reduces to is the same Clark & Scarf model whose published
//!     anchor (Snyder & Shen 47.65) and stockpyl reference optima ARE verified in
//!     `multi_echelon/serial`;
//!   - the assembly `env.rs` reproduces, by Monte-Carlo simulation, the exact serial optimum that
//!     the Rosling reduction + the literature-verified serial solver produce
//!     (`verification.rs`, finished lead time 1).
//! The assembly *instance numbers themselves* (22.759 / 52.536 / 27.530 etc.) are solver-derived,
//! not published — hence `literature_verified = false`.

/// A literature citation carried by this family.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AssemblyLiteratureReference {
    pub source: &'static str,
    pub url: &'static str,
    pub role: &'static str,
}

/// Rosling (1989): the structural assembly -> serial equivalence (the equivalence anchor).
pub const ROSLING_1989: AssemblyLiteratureReference = AssemblyLiteratureReference {
    source: "Rosling, K. (1989). Optimal Inventory Policies for Assembly Systems Under Random \
             Demands. Operations Research 37(4):565-579.",
    url: "https://doi.org/10.1287/opre.37.4.565",
    role: "Structural equivalence: an assembly system is equivalent to a serial system; optimal \
           policy is a balanced echelon base-stock policy. NOT a worked numeric benchmark (no \
           reproducible assembly cost/base-stock table).",
};

/// Clark & Scarf (1960): the serial model + echelon base-stock optimality the reduction targets.
pub const CLARK_SCARF_1960: AssemblyLiteratureReference = AssemblyLiteratureReference {
    source: "Clark, A. J., and H. Scarf (1960). Optimal Policies for a Multi-Echelon Inventory \
             Problem. Management Science 6(4):475-490.",
    url: "https://doi.org/10.1287/mnsc.6.4.475",
    role: "The serial equivalent and its echelon base-stock optimality; solved exactly in \
           multi_echelon/serial::exact.",
};

/// Snyder & Shen Example 6.1: the only PUBLISHED number in the chain (47.65). It is a 3-stage
/// serial instance verified in `multi_echelon/serial`; the 2-stage assembly reduction cannot reach
/// it, so it is NOT an assembly anchor — recorded here only to state where the published number is.
pub const SNYDER_SHEN_SERIAL_ANCHOR: AssemblyLiteratureReference = AssemblyLiteratureReference {
    source: "Snyder, L. V., and Z.-J. M. Shen. Fundamentals of Supply Chain Theory (2nd ed., \
             Wiley 2019), Example 6.1. Published optimal cost 47.65.",
    url: "https://doi.org/10.1002/9781119584445",
    role: "The published serial anchor (3-stage). Lives in multi_echelon/serial; NOT reachable by \
           the 2-stage assembly reduction, hence not an assembly published number.",
};

pub const ASSEMBLY_LITERATURE_REFERENCES: &[AssemblyLiteratureReference] =
    &[ROSLING_1989, CLARK_SCARF_1960, SNYDER_SHEN_SERIAL_ANCHOR];

/// A carried assembly verification instance (the equal-lead-time Rosling-reducible cases used in
/// `verification.rs`). `literature_verified` is the honest per-instance flag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AssemblyReferenceInstance {
    pub name: &'static str,
    /// FALSE for every instance: the cost is solver-derived (Rosling reduction + serial solver),
    /// reproduced by the env simulation — it is NOT a number printed in a paper.
    pub literature_verified: bool,
    pub num_components: usize,
    pub component_lead_time: usize,
    pub finished_lead_time: usize,
    /// Exact optimum of the Rosling serial-equivalent (repo solver-derived, NOT published).
    pub solver_derived_optimal_cost: f64,
    pub notes: &'static str,
}

/// The three `verification.rs` anchor instances, all honestly `literature_verified = false`.
pub const ASSEMBLY_REFERENCE_INSTANCES: &[AssemblyReferenceInstance] = &[
    AssemblyReferenceInstance {
        name: "two_component_poisson_Lc1",
        literature_verified: false,
        num_components: 2,
        component_lead_time: 1,
        finished_lead_time: 1,
        solver_derived_optimal_cost: 22.758925,
        notes: "2 components (kit holding 2), finished holding 3, L_c=L_a=1, p=10, Poisson(5). \
                Reduces to serial [kit 2, finished 3], lead [1,1]. Cost is solver-derived (NOT a \
                published assembly number). Verified-by-equivalence + env-sim reproduction.",
    },
    AssemblyReferenceInstance {
        name: "three_component_poisson_Lc2",
        literature_verified: false,
        num_components: 3,
        component_lead_time: 2,
        finished_lead_time: 1,
        solver_derived_optimal_cost: 52.536229,
        notes: "3 components (kit holding 3), finished holding 7, L_c=2, L_a=1, p=37.12, \
                Poisson(5). Constructed to share Snyder&Shen Example 6.1's two downstream stages, \
                but its cost 52.536 is solver-derived, NOT the published 3-stage 47.65.",
    },
    AssemblyReferenceInstance {
        name: "heterogeneous_components_poisson_Lc2",
        literature_verified: false,
        num_components: 2,
        component_lead_time: 2,
        finished_lead_time: 1,
        solver_derived_optimal_cost: 27.530177,
        notes: "Heterogeneous component holding [0.5,1.5] (kit holding 2), finished holding 4, \
                L_c=2, L_a=1, p=20, Poisson(4). Cost is solver-derived (NOT a published number).",
    },
];

/// The instance used as the verification entry point (still NOT literature-verified — see notes).
pub const VERIFICATION_PROBLEM_INSTANCE: &AssemblyReferenceInstance =
    &ASSEMBLY_REFERENCE_INSTANCES[0];

/// The primary reference is the structural equivalence anchor (Rosling 1989).
pub const PRIMARY_REFERENCE_INSTANCE: &AssemblyLiteratureReference = &ROSLING_1989;

#[cfg(test)]
mod tests {
    use super::*;

    /// HONESTY GUARD: no carried assembly instance may claim `literature_verified = true`.
    /// There is no directly-reproducible PUBLISHED assembly number (Rosling 1989 is structural;
    /// the only published number, Snyder&Shen 47.65, is a 3-stage serial system the 2-stage
    /// assembly reduction cannot reach). Flipping any flag to true requires a real paper-printed
    /// assembly cost/base-stock table — this test fails until one exists and is reproduced.
    #[test]
    fn no_assembly_instance_is_literature_verified() {
        for inst in ASSEMBLY_REFERENCE_INSTANCES {
            assert!(
                !inst.literature_verified,
                "instance {} must stay literature_verified=false: assembly has no directly \
                 reproducible published number (Rosling 1989 is structural; the published 47.65 \
                 anchor is a 3-stage serial system unreachable by the 2-stage assembly reduction)",
                inst.name
            );
        }
    }

    #[test]
    fn references_carry_the_structural_and_serial_anchors() {
        assert_eq!(ASSEMBLY_LITERATURE_REFERENCES.len(), 3);
        assert!(ROSLING_1989.source.contains("Rosling"));
        assert!(CLARK_SCARF_1960.source.contains("Clark"));
        assert!(SNYDER_SHEN_SERIAL_ANCHOR.source.contains("47.65"));
    }
}
