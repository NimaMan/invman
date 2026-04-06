#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaterialAttribute {
    pub name: String,
    pub allowed_values: Vec<String>,
}
