pub trait RowAccess {
    type Output;
    fn get_row(&self) -> Option<Vec<Self::Output>>;
}

pub trait ColAccess {
    type Output;
    fn get_col(&self) -> Option<Vec<Self::Output>>;
}

#[derive(Debug)]
pub struct MatDim {
    pub rows: usize,
    pub cols: usize
}

pub trait GetDims {
    fn get_dims(&self) -> MatDim;
}

#[derive(Debug)]
pub enum MatErr {
    MatrixFinalised,
    NonSquareMatrix,
    IncorrectDimensions,
}