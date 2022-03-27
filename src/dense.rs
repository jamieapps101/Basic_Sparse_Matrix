use crate::util::{GetDims,MatDim};

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

    pub fn new_from_data(data: &[&[T]]) -> Self {
        let col_count = data.len();
        let row_count = data[0].len();
        Self {
            col_count,
            row_count,
            data: data.iter().map(|e| Vec::from(*e)).collect(),
        }
    }

    pub fn get_col(&self,col_index: usize) -> &[T] {
        &self.data[col_index]
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


#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn init() {
        let a = Dense::<i32>::new_default_with_dims(5,7);
        let b = Dense::new_from_data(&[
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
        let a = Dense::<i32>::new_from_data(&[
            &[1,2,3],
            &[4,5,6],
            &[7,8,9],
        ]);

        assert_eq!(a.get_col(2), &[7,8,9]);
    }
}