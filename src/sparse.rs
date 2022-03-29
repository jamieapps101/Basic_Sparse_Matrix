use std::fmt;
use crate::util::{GetDims,MatDim,MatErr};
use crate::dense::Dense;


#[derive(PartialEq,Clone)]
pub struct Csr<T> {
    dims: MatDim,
    v: Vec<T>,
    col_index: Vec<usize>,
    row_index: Vec<usize>,
    is_finalised: bool
}

#[derive(PartialEq, Debug)]
pub struct CsrEntry<T: std::fmt::Debug> {
    pub v: T,
    pub col_index: usize,
    pub row_index: usize
}

impl<T: Copy + Default + PartialEq + std::fmt::Debug> Csr<T> {
    pub fn new<D: Into<MatDim>>(dims: D) -> Self {
        Self {
            dims: dims.into(),
            v: Vec::new(),
            col_index: Vec::new(),
            row_index: vec![0],
            is_finalised: false,
        }
    }

    pub fn eye<D: Into<MatDim>>(dims: D, value: T) -> Result<Self,MatErr> {
        let dims = dims.into();
        if dims.cols != dims.rows {
            return Err(MatErr::IncorrectDimensions)
        }
        let mut m = Self {
            dims,
            v: Vec::new(),
            col_index: Vec::new(),
            row_index: vec![0],
            is_finalised: false,
        };
        for n in 0..dims.cols {
            m.insert_unchecked(value, n, n);
        }
        Ok(m)
    }

    pub fn get_nnz(&self) -> usize {
        *self.row_index.last().unwrap_or(&0)
    }

    pub fn get_density(&self) -> f32 {
        self.v.len() as f32 / (self.dims.rows*self.dims.cols) as f32
    }

    pub fn get_val_at(&self, at: MatDim) -> Option<&T> {
        let row_start = self.row_index[at.rows];
        let row_end = self.row_index[at.rows+1];
        for (local_index,col_index) in self.col_index[row_start..row_end].iter().enumerate() {
            if col_index == &at.rows {
                return Some(&self.v[row_start+local_index])
            }
        }
        None
    }

    pub fn get_mut_val_at(&mut self, at: MatDim) -> Option<&mut T> {
        let row_start = self.row_index[at.rows];
        let row_end = self.row_index[at.rows+1];
        for (local_index,col_index) in self.col_index[row_start..row_end].iter().enumerate() {
            if col_index == &at.rows {
                return Some(&mut self.v[row_start+local_index])
            }
        }
        None
    }

    pub fn from_data(data: &[&[T]]) -> Self {
        let rows = data.len();
        let cols = data[0].len();
        let mut m = Self::new((rows,cols));
        for (i,row) in data.iter().enumerate() {
            for (j,val) in row.iter().enumerate() {
                m.insert(*val,i,j).unwrap();
            }
        }
        m.finalise()
    }

    /// adds the value of NNZ onto the end of the row_indexes
    pub fn finalise(mut self) -> Self {
        if !self.is_finalised {
            self.is_finalised = true;
            self.row_index.push(self.v.len());
        }
        self
    }

    /// should only be used to insert data in order of appearance, col by col, row by row
    pub fn insert(&mut self, value: T, row: usize, col: usize) ->  Result<(), MatErr> {
        if self.is_finalised {
            return Err(MatErr::MatrixFinalised)
        }
        // todo: add more checks here

        // silently ignore trying to add values of default
        if value != T::default() {
            self.insert_unchecked(value, row, col);
        }
        Ok(())
    }

    /// same as above, but does not do the checking that data is in order
    /// treats default value of T as the value to not store; 0 for most types
    fn insert_unchecked(&mut self, value: T, row: usize, col: usize) {
            self.v.push(value);
            self.col_index.push(col);
            if row > self.row_index.len()-1 {
                if row > self.row_index.len() {
                    self.row_index.push( self.v.len()-1  );
                    for _ in self.row_index.len()..(row+1) {
                        self.row_index.push(*self.row_index.last().unwrap());
                    }
                } else {
                    self.row_index.push(self.v.len()-1);
                }
            }
    }

    pub fn get_row_compact(&self, index: usize) -> Option<Vec<CsrEntry<&T>>> {
        let row_start = self.row_index[index];
        let row_end = self.row_index[index+1];
        Some(self.col_index[row_start..row_end].iter().zip(self.v[row_start..row_end].iter()).map( | (col_index,v) | {
            CsrEntry { v, col_index: *col_index, row_index: index }
        }).collect())
    }

    pub fn get_row_complete(&self, index: usize) -> Option<Vec<T>> {
        if self.row_index.len() == 0 { return None }
        // need to handle special case where only enties in the index row exist with none in row index+1
        if index >= self.row_index.len() { return None }

        let row_start = self.row_index[index];
        let index_with_offset = index+1;
        let row_end;
        if self.row_index.len() == index_with_offset {
            row_end = self.v.len()
        } else {
            row_end = self.row_index[index+1];
        }
        let mut return_vec = Vec::with_capacity(self.dims.cols);
        let mut prev_col = 0;
        for (col,entry) in self.col_index[row_start..row_end].iter().zip(self.v[row_start..row_end].iter()) {
            if col != &0 {
                for _ in prev_col..*col {
                    return_vec.push(T::default());
                }
            }
            prev_col = *col+1;
            return_vec.push(*entry)
        }
        for _ in prev_col..(self.dims.cols) {
            return_vec.push(T::default());
        }
        Some(return_vec)
    }

    pub fn transpose(&self) -> Self {
        let mut t = Self::new(self.dims.transpose());
        for col_index in 0..self.dims.cols {
            for entry_index in 0..self.v.len() {
                // for each col index, find any entries with matching col index
                if col_index==self.col_index[entry_index] {
                    let col = col_index;
                    let val = self.v[entry_index];
                    let mut row = 0;
                    loop {
                        if self.row_index.len() > 1 && self.row_index[row+1] <= entry_index {
                            row += 1;
                        } else {
                            break;
                        }
                    }
                    // insert into new matrix, with reversed row/col index
                    t.insert_unchecked(val, col, row);
                }
            }
        }
        t.finalise()
    }

    pub fn pair_with_tranpose(self) -> (Self,Self) {
        let t = self.transpose();
        (self,t)
    }

    /// This is an expensive function for a Csr style matrix
    pub fn get_col_compact(&self, index: usize) -> Option<Vec<CsrEntry<&T>>> {
        let mut return_vec = Vec::new();
        let mut row_count = 0;
        for i in 0..self.v.len() {
            while self.row_index[row_count] == i {
                row_count +=1;
            }
            if self.col_index[i] == index {
                return_vec.push(CsrEntry{
                    v: &self.v[i],
                    col_index: index,
                    row_index: row_count-1
                })
            }
        }
        return Some(return_vec)
    }

    /// This is an expensive function for a Csr style matrix
    pub fn get_col_complete(&self, index: usize) -> Option<Vec<T>> {
        let mut return_vec = Vec::new();
        let mut row_count = 0;
        for i in 0..self.v.len() {
            if self.col_index[i] == index {
                return_vec.push(self.v[i]);
            } else {
                while self.row_index[row_count] == i {
                    row_count +=1;
                    if i != 0 {
                        return_vec.push(T::default());
                    }
                }
            }
            while self.row_index[row_count] == i {
                row_count +=1;
            }
        }
        return Some(return_vec)
    }

    /// This is an expensive function for a Csr style matrix
    pub fn get_col(&self, index: usize) -> Option<Self> {
        if let Some(col) = self.get_col_compact(index) {
            let mut m = Self::new((self.dims.rows,1));
            for entry in col {
                m.insert_unchecked(*entry.v, entry.row_index, 0);
            }
            Some(m.finalise())
        } else {
            None
        }
    }
}

impl<T> GetDims for Csr<T> {
    fn get_dims(&self) -> MatDim {
        self.dims
    }
}


impl<T: Copy + Default + PartialEq + std::fmt::Debug + std::ops::Add<T,Output=T> + std::ops::Sub<T,Output=T> + std::ops::Mul<T,Output=T>> Csr<T> {
    pub fn mul_dense(&self, rhs: &Dense<T>) -> Result<Self,MatErr> {
        if self.dims.cols != rhs.get_dims().rows {
            return Err(MatErr::IncorrectDimensions)
        }
        let mut result = Self::new((self.dims.rows, rhs.get_dims().cols));
        for row_index in 0..self.dims.rows {
            let row = self.get_row_compact(row_index).unwrap();
            for col_index in 0..rhs.get_dims().cols {
                let mut value = T::default();
                for lhs_entry in &row {
                    let a = lhs_entry.v;
                    let b = rhs.get_col(col_index)[lhs_entry.col_index];
                    let c = (*a)*b;
                    value = value + c;
                }
                // result.insert_unchecked(value, row_index, col_index)
                result.insert(value, row_index, col_index).unwrap();
            }
        }
        Ok(result.finalise())
    }

    pub fn add_sparse(&self, rhs: &Self) -> Result<Self,MatErr> {
        if self.get_dims() != rhs.get_dims() {
            return Err(MatErr::IncorrectDimensions)
        }
        let mut output = Self::new(self.dims);

        // create two iterators over the elements which can asyncronously advance
        // let mut self_row_index = 0;
        let mut row_index = 0;
        loop {
            let self_row = self.get_row_compact(row_index).unwrap();
            let rhs_row = rhs.get_row_compact(row_index).unwrap();
            let mut self_entry_index = 0;
            let mut rhs_entry_index = 0;
            loop {
                let self_has_entry = self_row.len() > 0 && self_row.len() > self_entry_index;
                let rhs_has_entry = rhs_row.len() > 0 && rhs_row.len() > rhs_entry_index;
                if self_has_entry && rhs_has_entry {
                    let self_entry = &self_row[self_entry_index];
                    let rhs_entry  = &rhs_row[rhs_entry_index];
                    if self_entry.col_index > rhs_entry.col_index {
                        output.insert(*rhs_entry.v, rhs_entry.row_index, rhs_entry.col_index).unwrap();
                        rhs_entry_index+=1;
                    } else if self_entry.col_index < rhs_entry.col_index {
                        output.insert(*self_entry.v, self_entry.row_index, self_entry.col_index).unwrap();
                        self_entry_index+=1;
                    } else {
                        // self_entry.col_index == rhs_entry.col_index
                        rhs_entry_index+=1;
                        self_entry_index+=1;
                        let v = *self_entry.v + *rhs_entry.v;
                        output.insert(v, self_entry.row_index, self_entry.col_index).unwrap();
                    }

                } else if self_has_entry && !rhs_has_entry {
                    let entry = &self_row[self_entry_index];
                    output.insert(*entry.v, entry.row_index, entry.col_index).unwrap();
                    self_entry_index+=1;
                } else if !self_has_entry && rhs_has_entry {
                    let entry = &rhs_row[rhs_entry_index];
                    output.insert(*entry.v, entry.row_index, entry.col_index).unwrap();
                    rhs_entry_index+=1;
                } else {
                    // ie !self_has_entry && !rhs_has_entry
                    break;
                }
            }
            row_index+=1;
            if row_index == self.dims.rows {
                break;
            }
        }
        Ok(output.finalise())
    }

    pub fn sub_sparse(&self, rhs: &Self) -> Result<Self,MatErr> {
        if self.get_dims() != rhs.get_dims() {
            return Err(MatErr::IncorrectDimensions)
        }
        let mut output = Self::new(self.dims);

        // create two iterators over the elements which can asyncronously advance
        // let mut self_row_index = 0;
        let mut row_index = 0;
        loop {
            let self_row = self.get_row_compact(row_index).unwrap();
            let rhs_row = rhs.get_row_compact(row_index).unwrap();
            let mut self_entry_index = 0;
            let mut rhs_entry_index = 0;
            loop {
                let self_has_entry = self_row.len() > 0 && self_row.len() > self_entry_index;
                let rhs_has_entry = rhs_row.len() > 0 && rhs_row.len() > rhs_entry_index;
                if self_has_entry && rhs_has_entry {
                    let self_entry = &self_row[self_entry_index];
                    let rhs_entry  = &rhs_row[rhs_entry_index];
                    if self_entry.col_index > rhs_entry.col_index {
                        output.insert(T::default()-*rhs_entry.v, rhs_entry.row_index, rhs_entry.col_index).unwrap();
                        rhs_entry_index+=1;
                    } else if self_entry.col_index < rhs_entry.col_index {
                        output.insert(*self_entry.v, self_entry.row_index, self_entry.col_index).unwrap();
                        self_entry_index+=1;
                    } else {
                        // self_entry.col_index == rhs_entry.col_index
                        rhs_entry_index+=1;
                        self_entry_index+=1;
                        let v = *self_entry.v - *rhs_entry.v;
                        output.insert(v, self_entry.row_index, self_entry.col_index).unwrap();
                    }

                } else if self_has_entry && !rhs_has_entry {
                    let entry = &self_row[self_entry_index];
                    output.insert(*entry.v, entry.row_index, entry.col_index).unwrap();
                    self_entry_index+=1;
                } else if !self_has_entry && rhs_has_entry {
                    let entry = &rhs_row[rhs_entry_index];
                    output.insert(T::default()-*entry.v, entry.row_index, entry.col_index).unwrap();
                    rhs_entry_index+=1;
                } else {
                    // ie !self_has_entry && !rhs_has_entry
                    break;
                }
            }
            row_index+=1;
            if row_index == self.dims.rows {
                break;
            }
        }
        Ok(output.finalise())
    }

    pub fn mul_sparse(&self, rhs: Self) -> Result<Self,MatErr> {
        unimplemented!();
    }

    pub fn sum_elements(&self) -> T {
        let mut sum = T::default();
        for v in &self.v {
            sum = sum + *v;
        }
        sum
    }
}

impl Csr<f32> {
    pub fn cholesky_decomp(&self) -> Result<Self,MatErr> {
        if self.dims.rows != self.dims.cols {
            return Err(MatErr::NonSquareMatrix)
        }
        let mut l = Self::new(self.dims);
        for i in 0..self.dims.rows {
            for j in 0..(i+1) {
                let mut sum: f32 = 0.0;
                for k in 0..j {
                    let a = if let Some(row) = l.get_row_complete(i) {
                        row[k]
                    } else {
                        0.0
                    };
                    let b = if let Some(row) = l.get_row_complete(j) {
                        row[k]
                    } else {
                        0.0
                    };
                    sum += a * b;
                }
                let val_to_insert;
                if i==j {
                    val_to_insert = (self.get_row_complete(i).unwrap()[i]-sum).powf(0.5);
                } else {
                    let temp = self.get_row_complete(i).unwrap()[j]-sum;
                    let a = l.get_row_complete(j).unwrap()[j];
                    val_to_insert = (1.0 / a) * temp;
                }
                l.insert(val_to_insert, i, j).unwrap();
            }
        }
        Ok(l.finalise())
    }
}

impl<T: fmt::Display + Copy + Default + PartialEq + fmt::Debug> fmt::Display for Csr<T> {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for row_index in 0..self.dims.rows {
            write!(f, "|").unwrap();
            if let Some(row) = self.get_row_complete(row_index) {
                for entry in row {
                    write!(f, "{:>5}", entry).unwrap();
                }
            } else {
                for _ in 0..self.dims.cols {
                    write!(f, "{:>5}", T::default()).unwrap();
                }
            }
            write!(f, "|\n").unwrap();
        }
        Ok(())
    }
}

impl<T: fmt::Display + Copy + Default + PartialEq + fmt::Debug > fmt::Debug for Csr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v:         {:?}\n", self.v ).unwrap();
        write!(f, "col_index: {:?}\n", self.col_index ).unwrap();
        write!(f, "row_index: {:?}\n", self.row_index ).unwrap();
        Ok(())
    }
}


#[cfg(test)]
mod test {
    use super::*;
    // check v and col_index are of length NNZ
    // check row index is of length m+1


    #[test]
    fn example_mat_0() {
        let m = Csr::from_data(&[
            &[5,0,0,0],
            &[0,8,0,0],
            &[0,0,3,0],
            &[0,6,0,0],
        ]);

        assert_eq!(m.v.as_slice(),         &[5,8,3,6]);
        assert_eq!(m.col_index.as_slice(), &[0,1,2,1]);
        assert_eq!(m.row_index.as_slice(), &[0,1,2,3,4]);
    }

    #[test]
    fn example_mat_1() {
        let m = Csr::from_data(&[
            &[10,20, 0, 0, 0, 0],
            &[ 0,30, 0,40, 0, 0],
            &[ 0, 0,50,60,70, 0],
            &[ 0, 0, 0, 0, 0,80],
        ]);

        assert_eq!(m.v.as_slice(),         &[10,20,30,40,50,60,70,80]);
        assert_eq!(m.col_index.as_slice(), &[0,1,1,3,2,3,4,5]);
        assert_eq!(m.row_index.as_slice(), &[0,2,4,7,8]);
    }

    #[test]
    fn example_mat_2() {
        let m = Csr::from_data(&[
            &[5],
        ]);

        assert_eq!(m.v.as_slice(),         &[5]);
        assert_eq!(m.col_index.as_slice(), &[0]);
        assert_eq!(m.row_index.as_slice(), &[0,1]);
    }

    #[test]
    fn get_row_by_index_0() {
        let m = Csr::from_data(&[
            &[10,20, 0, 0, 0, 0],
            &[ 0,30, 0,40, 0, 0],
            &[ 0, 0,50,60,70, 0],
            &[ 0, 0, 0, 0, 0,80],
        ]);

        assert_eq!(m.get_row_complete(2), Some(vec![ 0, 0,50,60,70, 0]));

        assert_eq!(m.get_row_compact(2), Some(vec![
            CsrEntry {v: &50, row_index: 2, col_index: 2},
            CsrEntry {v: &60, row_index: 2, col_index: 3},
            CsrEntry {v: &70, row_index: 2, col_index: 4},
        ]));
    }

    #[test]
    fn get_col_by_index_0() {
        let m = Csr::from_data(&[
            &[10,20, 0, 0, 0, 0],
            &[ 0,30, 0,40, 0, 0],
            &[ 0, 0,50,60,70, 0],
            &[ 0, 0, 0, 0, 0,80],
        ]);
        assert_eq!(m.get_col_complete(1), Some(vec![ 20, 30, 0, 0]));
        assert_eq!(m.get_col_complete(4), Some(vec![ 0, 0, 70, 0]));

        assert_eq!(m.get_col_compact(1), Some(vec![
            CsrEntry {v: &20, row_index: 0, col_index: 1},
            CsrEntry {v: &30, row_index: 1, col_index: 1},
        ]));

        assert_eq!(m.get_col_compact(4), Some(vec![
            CsrEntry {v: &70, row_index: 2, col_index: 4},
        ]));

        let c = m.get_col(3).unwrap();
        let c_ref = Csr::from_data(&[
            &[ 0],
            &[40],
            &[60],
            &[ 0],
        ]);
        assert_eq!(c,c_ref)
    }

    #[test]
    fn get_row_by_index_1() {
        let m = Csr::from_data(&[
            &[5,6,7,8,9],
            &[0,0,0,0,0],
            &[0,0,0,0,1],
            &[1,0,0,0,0],
        ]);

        println!("m:\n{:?}\n\n{}\n",m,m);

        assert_eq!(m.get_row_complete(0), Some(vec![ 5,6,7,8,9 ]));
        assert_eq!(m.get_row_complete(1), Some(vec![ 0,0,0,0,0 ]));
        assert_eq!(m.get_row_complete(2), Some(vec![ 0,0,0,0,1 ]));
        assert_eq!(m.get_row_complete(3), Some(vec![ 1,0,0,0,0 ]));

        assert_eq!(m.get_row_compact(0), Some(vec![
            CsrEntry {v: &5, row_index: 0, col_index: 0},
            CsrEntry {v: &6, row_index: 0, col_index: 1},
            CsrEntry {v: &7, row_index: 0, col_index: 2},
            CsrEntry {v: &8, row_index: 0, col_index: 3},
            CsrEntry {v: &9, row_index: 0, col_index: 4},
        ]));

        assert_eq!(m.get_row_compact(1), Some(vec![]));

        assert_eq!(m.get_row_compact(2), Some(vec![
            CsrEntry {v: &1, row_index: 2, col_index: 4},
        ]));

        assert_eq!(m.get_row_compact(3), Some(vec![
            CsrEntry {v: &1, row_index: 3, col_index: 0},
        ]));

    }

    #[test]
    fn get_row_by_index_single() -> Result<(),String> {

        let mut m = Csr::<f32>::new((5, 5));
        m.insert_unchecked(2.0, 0, 0);
        let v = m.get_row_complete(0).unwrap();
        if v[0] != 2.0 {
            let t = v[0];
            return Err(format!("{t} != 2.0"))
        }

        Ok(())
    }

    #[test]
    fn transpose_1x1() {
        let m = Csr::from_data(&[&[10]]);
        let m_transpose_ref = Csr::from_data(&[ &[10], ]);
        let m_transpose = m.transpose();
        assert_eq!(m_transpose,m_transpose_ref);
    }

    #[test]
    fn transpose_nxn() {
        let m = Csr::from_data(&[
            &[10,5,7,9,2],
            &[10,8,5,9,3],
            &[ 0,5,4,6,2],
            &[ 1,2,7,9,2],
            ]);

        let m_transpose_ref = Csr::from_data(&[
            &[10,10, 0, 1],
            &[ 5, 8, 5, 2],
            &[ 7, 5, 4, 7],
            &[ 9, 9, 6, 9],
            &[ 2, 3, 2, 2],
        ]);

        let m_transpose = m.transpose();
        assert_eq!(m_transpose,m_transpose_ref);
    }

    #[test]
    fn transpose_mxn() {
        let m = Csr::from_data(&[
            &[10,20, 0, 0, 0, 0],
            &[ 0,30, 0,40, 0, 0],
            &[ 0, 0,50,60,70, 0],
            &[ 0, 0, 0, 0, 0,80],
            ]);

        let m_transpose_ref = Csr::from_data(&[
            &[10, 0, 0, 0],
            &[20,30, 0, 0],
            &[ 0, 0,50, 0],
            &[ 0,40,60, 0],
            &[ 0, 0,70, 0],
            &[ 0, 0, 0,80],
        ]);

        let m_transpose = m.transpose();
        assert_eq!(m_transpose,m_transpose_ref);
    }

    #[test]
    #[ignore]
    fn display_mat() {
        let m: Csr<f32> = Csr::from_data(&[
            &[  4.0, 12.0,-16.0],
            &[ 12.0, 37.0,-43.0],
            &[-16.0,-43.0, 98.0],
        ]);
        println!("{:5}",m);
    }

    #[test]
    fn cholesky_decomposition_0() {

        let m: Csr<f32> = Csr::from_data(&[
            &[  4.0, 12.0,-16.0],
            &[ 12.0, 37.0,-43.0],
            &[-16.0,-43.0, 98.0],
        ]);

        let lower_l_ref: Csr<f32>  = Csr::from_data(&[
            &[ 2.0,0.0,0.0],
            &[ 6.0,1.0,0.0],
            &[-8.0,5.0,3.0],
        ]);
        let upper_l_ref: Csr<f32>  = Csr::from_data(&[
            &[2.0,6.0,-8.0],
            &[0.0,1.0, 5.0],
            &[0.0,0.0, 3.0],
        ]);


        // let (lower_l,upper_l) = m.cholesky_decomp();
        let lower_l = m.cholesky_decomp().unwrap();
        let upper_l = lower_l.transpose();

        // println!("lower_l:\n{lower_l}");
        // println!("\nlower_l_ref:\n{lower_l_ref}");

        assert_eq!(lower_l, lower_l_ref);
        assert_eq!(upper_l, upper_l_ref);
    }

    #[test]
    fn cholesky_decomposition_1() {
        let m = Csr::from_data(&[
            &[8.0, 0.0, 0.0, 0.0],
            &[0.0, 7.0, 1.0, 0.0],
            &[0.0, 1.0, 3.0, 0.0],
            &[0.0, 0.0, 0.0, 2.0]
        ]);
        println!("m:\n{m}");

        let lower_l_ref: Csr<f32>  = Csr::from_data(&[
            &[2.828427, 0.0       , 0.0      , 0.0       ],
            &[0.0     , 2.6457512 , 0.0      , 0.0       ],
            &[0.0     , 0.37796451, 1.6903086, 0.0       ],
            &[0.0     , 0.0       , 0.0      , 1.4142135]
        ]);
        let lower_l = m.cholesky_decomp().unwrap();
        assert_eq!(lower_l, lower_l_ref);
    }

    #[test]
    fn test_dense_mul() {
        let d = Dense::from_data(&[
            &[ 1, 2, 3, 4],
            &[ 5, 6, 7, 8],
            &[ 9,10,11,12]
        ]);

        let s = Csr::from_data(&[
            &[3,0,2,0],
            &[7,0,0,0],
            &[0,2,0,1],
            &[0,0,1,0],
            &[1,0,0,0],
        ]);

        let output_ref = Csr::from_data(&[
            &[ 9,29,49],
            &[ 7,35,63],
            &[ 8,20,32],
            &[ 3, 7,11],
            &[ 1, 5, 9],
        ]);

        let output = s.mul_dense(&d).unwrap();

        assert_eq!(output,output_ref);
    }

    #[test]
    fn csr_with_empty_row_top() {
        let a = 11;
        let b = 12;
        let c = 13;

        let m = Csr::from_data(&[
            &[0,0,0],
            &[a,b,c],
            &[0,0,0],
        ]);

        assert_eq!(m.is_finalised,true);
        assert_eq!(m.v,vec![a,b,c]);
        assert_eq!(m.col_index,vec![0,1,2]);
        assert_eq!(m.row_index,vec![0,0,3]);
        // todo
        // maybe should be as below, however as no elements are
        // added on the 3rd row, no need to point to the 3rd indexed
        // value. Existing 3 is just the NNZ added as part of finalisation.
        // maybe worth adding this as part of finalisation?
        //assert_eq!(m.row_index,vec![0,0,3,3]);
    }

    #[test]
    fn csr_with_empty_row_middle() {
        let m = Csr::from_data(&[
            &[8,0,2,0,0], // 0
            &[0,0,5,0,0], // 1
            &[0,0,0,0,0], // 2
            &[0,0,0,0,0], // 2
            &[0,0,7,1,2], // 2
            &[0,0,0,0,0], // 2
            &[0,0,0,9,0], //
        ]);

        assert_eq!(m.is_finalised,true);
        assert_eq!(m.v,vec![8,2,5,7,1,2,9]);
        assert_eq!(m.col_index,vec![0,2,2,2,3,4,3]);
        assert_eq!(m.row_index,vec![0,2,3,3,3,6,6,7]);
        // todo
        // maybe should be as below, however as no elements are
        // added on the 3rd row, no need to point to the 3rd indexed
        // value. Existing 3 is just the NNZ added as part of finalisation.
        // maybe worth adding this as part of finalisation?
        //assert_eq!(m.row_index,vec![0,0,3,3]);
    }

    #[test]
    fn test_nnz() {
        let m = Csr::from_data(&[
            &[5,2,1,3],
            &[7,0,1,3],
            &[0,1,0,0],
            &[0,7,4,0],
        ]);

        let a = Dense::from_data(&[
            &[1,0,3,4],
            &[8,0,0,5]
        ]);

        let output_ref = Csr::from_data(&[
            &[20,55],
            &[22,71],
            &[ 0,0],
            &[12,0],
        ]);

        let output = m.mul_dense(&a).unwrap();
        assert_eq!(output,output_ref);
        assert_eq!(output.get_nnz(),5);

    }


    #[test]
    fn add_sparse() {
        let a = Csr::from_data(&[
            &[5,6,7,8,9],
            &[0,0,0,0,0],
            &[0,0,0,0,1],
            &[1,0,0,0,0],
        ]);

        println!("a:\n{}",a);

        let b = Csr::from_data(&[
            &[9,8,7,6,5],
            &[0,0,0,0,0],
            &[1,0,0,0,0],
            &[1,0,0,0,0],
        ]);

        let c_ref = Csr::from_data(&[
            &[14,14,14,14,14],
            &[0,0,0,0,0],
            &[1,0,0,0,1],
            &[2,0,0,0,0],
        ]);

        let c = a.add_sparse(&b).unwrap();
        assert_eq!(c,c_ref);
    }

    #[test]
    fn sub_sparse() {
        let a = Csr::from_data(&[
            &[5,6,7,8,9],
            &[0,0,0,0,0],
            &[0,0,0,0,1],
            &[1,0,0,0,0],
        ]);

        println!("a:\n{}",a);

        let b = Csr::from_data(&[
            &[9,8,7,6,5],
            &[0,0,0,0,0],
            &[1,0,0,0,0],
            &[1,0,0,0,0],
        ]);

        let c_ref = Csr::from_data(&[
            &[-4,-2,0,2,4],
            &[0,0,0,0,0],
            &[-1,0,0,0,1],
            &[0,0,0,0,0],
        ]);

        let c = a.sub_sparse(&b).unwrap();
        assert_eq!(c,c_ref);
    }
}