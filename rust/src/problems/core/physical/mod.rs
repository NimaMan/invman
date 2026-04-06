pub mod flow;
pub mod material;
pub mod pipeline;
pub mod stock;
pub mod topology;

#[allow(unused_imports)]
pub use flow::{FlowEdgeSpec, FlowMode};
#[allow(unused_imports)]
pub use material::MaterialAttribute;
#[allow(unused_imports)]
pub use pipeline::PipelineSpec;
#[allow(unused_imports)]
pub use stock::{StockNodeSpec, StockRole};
#[allow(unused_imports)]
pub use topology::Topology;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalLayer {
    pub topology: Topology,
    pub stock_nodes: Vec<StockNodeSpec>,
    pub pipelines: Vec<PipelineSpec>,
    pub flow_edges: Vec<FlowEdgeSpec>,
    pub material_attributes: Vec<MaterialAttribute>,
}

impl PhysicalLayer {
    pub fn has_inventory_states(&self) -> bool {
        !self.stock_nodes.is_empty()
    }

    pub fn has_material_movement(&self) -> bool {
        !self.flow_edges.is_empty() || !self.pipelines.is_empty()
    }
}
