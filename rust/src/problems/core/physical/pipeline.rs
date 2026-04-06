#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineSpec {
    pub name: String,
    pub from: String,
    pub to: String,
    pub stages: usize,
}
