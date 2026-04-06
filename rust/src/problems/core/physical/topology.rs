#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Topology {
    SingleLocation,
    SerialChain,
    DivergentNetwork,
    DirectedNetwork,
    JointMultiItem,
    Custom(String),
}
