
pub enum CsrErr {
    MatrixFinalised,
}

#[derive(PartialEq, Debug)]
pub struct CSR<T> {
    col_count: usize,
    row_count: usize,
    v: Vec<T>,
    col_index: Vec<usize>,
    row_index: Vec<usize>,
    is_finalised: bool
}

#[derive(PartialEq, Debug)]
pub struct CsrEntry<T: std::fmt::Debug> {
    v: T,
    col_index: usize,
    row_index: usize
}

impl<T: Copy + Default + PartialEq + std::fmt::Debug> CSR<T> {
    pub fn new(col_count: usize, row_count: usize) -> Self {
        Self {
            col_count, 
            row_count,
            v: Vec::new(),
            col_index: Vec::new(),
            row_index: Vec::new(),
            is_finalised: false,
        }
    }

    pub fn from_data(data: &[&[T]]) -> Self {
        let rows = data.len();
        let cols = data[0].len();
        let mut m = Self::new(cols,rows);
        for (i,row) in data.iter().enumerate() {
            for (j,val) in row.iter().enumerate() {
                m.insert_unchecked(*val,i,j);
            }
        }
        m.finalise()
    }

    /// adds the value of NNZ onto the end of the row_indexes
    fn finalise(mut self) -> Self {
        self.is_finalised = true;
        self.row_index.push(self.v.len());
        self
    }

    /* 
        /// should only be used to insert data in order of appearance, col by col, row by row
        pub fn insert(&mut self, value: T, row: usize, col: usize) ->  Result<(), CsrErr> {
            if self.is_finalised {
                return Err(CsrErr::MatrixFinalised)
            }
        }
    */ 

    /// same as above, but does not do the checking that data is in order
    /// treats default value of T as the value to not store; 0 for most types
    fn insert_unchecked(&mut self, value: T, row: usize, col: usize) {
        if value != T::default() {
            println!("value: {value:?}, row: {row}, col: {col}");
            self.v.push(value);
            self.col_index.push(col);
            // strictly speaking the new row index should always be less than or equal to
            // that can be tested and dealt with in the *insert* function.
            // let current_recorded_row = self.row_index.len()-1;
            if self.row_index.len() == 0 || row > self.row_index.len()-1 {
                self.row_index.push(self.v.len()-1)
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
        let row_start = self.row_index[index];
        let row_end = self.row_index[index+1];
        let mut return_vec = Vec::new();
        let mut prev_col = 0;
        for (col,entry) in self.col_index[row_start..row_end].iter().zip(self.v[row_start..row_end].iter()) {
                for _ in prev_col..*col {
                    return_vec.push(T::default());
                }
                prev_col = *col+1;
                return_vec.push(*entry)
        }
        for _ in prev_col..(self.col_count) {
            return_vec.push(T::default());
        }
        Some(return_vec)
    }

    pub fn transpose(&self) -> Self {
        let mut t = Self::new(self.row_count, self.col_count);
        for col_index in 0..self.col_count {
            for entry_index in 0..self.v.len() {
                // for each col index, find any entries with matching col index
                if col_index==self.col_index[entry_index] {

                    let col = col_index;
                    let val = self.v[entry_index];

                    let mut row = 0;
                    loop {
                        if self.row_index[row+1] <= entry_index {
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

}

impl CSR<f32> {
    pub fn cholesky_decomp(&self) -> (Self,Self) {
        assert_eq!(self.row_count, self.col_count);
        let mut l = Self::new(self.col_count, self.row_count);
        for i in 0..self.row_count {
            for j in 0..i {
                let mut sum: f32 = 0.0;
                for k in 0..j {
                    sum += l.get_row_complete(i).unwrap()[k] * l.get_row_complete(j).unwrap()[k];
                }

                let val_to_insert;
                if i==j {
                    val_to_insert = (self.get_row_complete(i).unwrap()[i]-sum).powf(0.5);
                } else {
                    let temp = self.get_row_complete(i).unwrap()[j]-sum;
                    val_to_insert = 1.0 / l.get_row_complete(j).unwrap()[j] * temp;
                }
                l.insert_unchecked(val_to_insert, i, j);
            }
        }
        // Conjugate transpose (just transpose in this case)
        let l_ct = l.transpose();
        (l,l_ct)
    }
}

#[cfg(test)] 
mod test {
    use super::*;
    // check v and col_index are of length NNZ
    // check row index is of length m+1


    #[test]
    fn example_mat_0() {
        let m = CSR::from_data(&[
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
        let m = CSR::from_data(&[
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
    fn get_row_by_index() {
        let m = CSR::from_data(&[
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
    fn transpose() {
        println!("Create m");
        let m = CSR::from_data(&[
            &[10,20, 0, 0, 0, 0],
            &[ 0,30, 0,40, 0, 0],
            &[ 0, 0,50,60,70, 0],
            &[ 0, 0, 0, 0, 0,80],
            ]);
            
        println!("\nCreate m_transpose_ref");
        let m_transpose_ref = CSR::from_data(&[
            &[10, 0, 0, 0],
            &[20,30, 0, 0],
            &[ 0, 0,50, 0],
            &[ 0,40,60, 0],
            &[ 0, 0,70, 0],
            &[ 0, 0, 0,80],
        ]);

        println!("\nCreate m_transpose");
        let m_transpose = m.transpose();

        assert_eq!(m_transpose,m_transpose_ref);
    }
}