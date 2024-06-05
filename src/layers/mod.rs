mod module;
mod batch_norm;
mod tanh;

pub use module::{LinearLayer, Module};
pub use batch_norm::BatchNorm1d;
pub use tanh::Tanh;