// Identifiers for the three vanilla lost-sales heuristics implemented in this
// module: Myopic-1, Myopic-2, and the Standard Vector Base Stock (SVBS) policy.
//
// `policy_name` returns the canonical short name used both in the verification
// targets ("myopic1"/"myopic2"/"svbs") and in any downstream reporting.
// `all` lists the policies in best-to-worst order for the canonical vanilla
// instance (myopic2 <= myopic1 <= svbs), which is the order used when running
// a full verification sweep.

/// The three vanilla lost-sales heuristics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LostSalesHeuristicPolicyKind {
    Myopic1,
    Myopic2,
    StandardVectorBaseStock,
}

impl LostSalesHeuristicPolicyKind {
    /// Canonical short name for the policy.
    pub fn policy_name(self) -> &'static str {
        match self {
            Self::Myopic1 => "myopic1",
            Self::Myopic2 => "myopic2",
            Self::StandardVectorBaseStock => "svbs",
        }
    }

    /// All policies, ordered best-to-worst for the canonical vanilla instance.
    pub fn all() -> [Self; 3] {
        [Self::Myopic2, Self::Myopic1, Self::StandardVectorBaseStock]
    }
}
