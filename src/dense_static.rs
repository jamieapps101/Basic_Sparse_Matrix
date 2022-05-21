use crate::util::{GetDims,MatDim};
use std::fmt;

#[derive(PartialEq,Debug)]
pub struct DenseS<T, const ROWS: usize, const COLS: usize> {
    col_count: usize,
    row_count: usize,
    data: [[T; ROWS ]; COLS ]
}


impl<T: Default + Clone + Copy, const ROWS: usize, const COLS: usize> DenseS<T,ROWS,COLS> {
    pub fn new_default() -> Self {
        Self::new(T::default())
    }

    pub fn new(val: T) -> Self {
        Self { col_count: COLS, row_count: ROWS, data: [ [val; ROWS];COLS] }
    }

    pub fn from_data(data: &[&[T]]) -> Self {
        let col_count = data.len();
        let row_count = data[0].len();
        let mut temp = [[T::default(); ROWS]; COLS];
        for i in 0..COLS {
            for j in 0..ROWS {
                temp[i][j] = data[i][j];
            }
        }
        Self {
            col_count,
            row_count,
            data: temp
        }
    }

    pub fn get_col(&self, col_index: usize) -> &[T] {
        &self.data[col_index]
    }

    pub fn get_col_mut(&mut self, col_index: usize) -> &mut [T] {
        &mut self.data[col_index]
    }
}

impl<T, const COLS: usize, const ROWS: usize> GetDims for DenseS<T,COLS,ROWS> {
    fn get_dims(&self) -> MatDim {
        MatDim {
            rows: self.row_count,
            cols: self.col_count,
        }
    }
}

impl<T: fmt::Display + Copy + Default + PartialEq + fmt::Debug, const COLS: usize, const ROWS: usize> fmt::Display for DenseS<T,COLS,ROWS> {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for row_index in 0..self.row_count {
            write!(f, "|").unwrap();
            for col_index in 0..self.col_count {
                let entry = self.get_col(col_index)[row_index];
                write!(f, "{:>5}", entry).unwrap();
            }
            writeln!(f, "|").unwrap();
        }
        Ok(())
    }
}


#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn init() {
        let a = DenseS::<i32,7,5>::new_default();
        let b = DenseS::from_data(&[
            &[0,0,0,0,0,0,0],
            &[0,0,0,0,0,0,0],
            &[0,0,0,0,0,0,0],
            &[0,0,0,0,0,0,0],
            &[0,0,0,0,0,0,0]
        ]);
        assert_eq!(a,b);
    }

    #[test]
    fn get_col() {
        let a = DenseS::<i32,3,3>::from_data(&[
            &[1,2,3],
            &[4,5,6],
            &[7,8,9],
        ]);

        assert_eq!(a.get_col(2), &[7,8,9]);
    }
}