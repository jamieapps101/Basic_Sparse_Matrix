use crate::util::{GetDims,MatDim};
use std::fmt;

#[derive(PartialEq,Debug)]
pub struct Dense<T> {
    col_count: usize,
    row_count: usize,
    data: Vec<Vec<T>>
}


impl<T: Default + Clone + Copy> Dense<T> {
    pub fn new_default_with_dims(col_count: usize, row_count: usize) -> Self {
        Self::new_with_dims(T::default(), col_count, row_count)
    }

    pub fn new_with_dims(val: T, col_count: usize, row_count: usize) -> Self {
        Self { col_count, row_count, data: vec![vec![val; row_count];col_count] }
    }

    pub fn from_data(data: &[&[T]]) -> Self {
        let col_count = data.len();
        let row_count = data[0].len();
        Self {
            col_count,
            row_count,
            data: data.iter().map(|e| Vec::from(*e)).collect(),
        }
    }

    pub fn get_col(&self, col_index: usize) -> &[T] {
        &self.data[col_index]
    }

    pub fn get_col_mut(&mut self, col_index: usize) -> &mut [T] {
        &mut self.data[col_index]
    }
}

impl<T> GetDims for Dense<T> {
    fn get_dims(&self) -> MatDim {
        MatDim {
            rows: self.row_count,
            cols: self.col_count,
        }
    }
}

impl<T: fmt::Display + Copy + Default + PartialEq + fmt::Debug> fmt::Display for Dense<T> {
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
        let a = Dense::<i32>::new_default_with_dims(5,7);
        let b = Dense::from_data(&[
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
        let a = Dense::<i32>::from_data(&[
            &[1,2,3],
            &[4,5,6],
            &[7,8,9],
        ]);

        assert_eq!(a.get_col(2), &[7,8,9]);
    }
}