#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Discounting {
    None,
    Factor(f64),
}
