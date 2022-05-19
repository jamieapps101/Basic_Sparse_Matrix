pub trait RowAccess {
    type Output;
    fn get_row(&self) -> Option<Vec<Self::Output>>;
}

pub trait ColAccess {
    type Output;
    fn get_col(&self) -> Option<Vec<Self::Output>>;
}

#[derive(Debug,PartialEq,Clone,Copy)]
pub struct MatDim {
    pub rows: usize,
    pub cols: usize
}

impl MatDim {
    pub fn transpose(self) -> Self {
        Self {rows: self.cols, cols: self.rows}
    }
}

impl From<(usize,usize)> for MatDim {
    fn from(d: (usize,usize)) -> Self {
        Self { rows: d.0 , cols: d.1 }
    }
}

impl From<MatDim> for (usize,usize) {
    fn from(d: MatDim) -> Self {
        (d.rows, d.cols)
    }
}

use std::fmt;
impl fmt::Display for MatDim {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(rows: {}, cols: {})",self.rows,self.cols)
    }
}

pub trait GetDims {
    fn get_dims(&self) -> MatDim;
}

#[derive(Debug,PartialEq)]
pub enum MatErr {
    MatrixFinalised,
    MatrixNotFinalised,
    NonSquareMatrix,
    IncorrectDimensions,
    PaddingSizeSmallerThanOriginal,
    OutOfBounds,
}