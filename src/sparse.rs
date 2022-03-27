use std::fmt;
use crate::util::{GetDims,MatErr};

#[derive(PartialEq)]
pub struct Csr<T> {
    col_count: usize,
    row_count: usize,
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
        self.insert_unchecked(value, row, col);
        Ok(())
    }
     

    /// same as above, but does not do the checking that data is in order
    /// treats default value of T as the value to not store; 0 for most types
    fn insert_unchecked(&mut self, value: T, row: usize, col: usize) {
        if value != T::default() {
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
        
        let mut return_vec = Vec::new();
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

}

impl<T: Copy + Default + PartialEq + std::fmt::Debug + std::ops::Add<T,Output=T> + std::ops::Mul<T,Output=T>> Csr<T> {
    pub fn mul(&self, rhs: crate::dense::Dense<T>) -> Result<Self,MatErr> {
        if self.col_count != rhs.get_dims().rows {
            return Err(MatErr::IncorrectDimensions)
        }
        let mut result = Self::new(rhs.get_dims().cols, self.row_count);
        for row_index in 0..self.row_count {
            let row = self.get_row_compact(row_index).unwrap();
            for col_index in 0..rhs.get_dims().cols {
                let mut value = T::default();
                for lhs_entry in &row {
                    let a = lhs_entry.v;
                    let b = rhs.get_col(col_index)[lhs_entry.col_index];
                    let c = (*a)*b;
                    value = value + c;
                }
                result.insert_unchecked(value, row_index, col_index)
            }
        }
        Ok(result.finalise())
    }
}

impl Csr<f32> {
    pub fn cholesky_decomp(&self) -> Result<Self,MatErr> {
        if self.row_count != self.col_count {
            return Err(MatErr::NonSquareMatrix)
        }
        let mut l = Self::new(self.col_count, self.row_count);
        for i in 0..self.row_count {
            for j in 0..(i+1) {
                // println!("i={i},j={j}");
                // println!("l (before):\n{l}");
                let mut sum: f32 = 0.0;
                for k in 0..j {
                    // println!(">k={k}");
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
                // println!(">sum={sum}");
                let val_to_insert;
                if i==j {
                    val_to_insert = (self.get_row_complete(i).unwrap()[i]-sum).powf(0.5);
                } else {
                    let temp = self.get_row_complete(i).unwrap()[j]-sum;
                    // println!(">temp={temp}");
                    let a = l.get_row_complete(j).unwrap()[j];
                    // println!(">a={a}");
                    val_to_insert = (1.0 / a) * temp;
                }
                // println!(">val_to_insert={val_to_insert}");
                l.insert_unchecked(val_to_insert, i, j);
                // println!("l (after):\n{l}");
            }
        }
        Ok(l.finalise())
    }
}

impl<T: fmt::Display + Copy + Default + PartialEq + fmt::Debug> fmt::Display for Csr<T> {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for row_index in 0..self.row_count {
            write!(f, "|").unwrap();
            if let Some(row) = self.get_row_complete(row_index) {
                for entry in row {
                    write!(f, "{:>5}", entry).unwrap();
                }
            } else {
                for _ in 0..self.col_count {
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
    fn get_row_by_index() {
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
    fn get_row_by_index_1x1() -> Result<(),String> {

        let mut m = Csr::<f32>::new(5, 5);
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
        let d = crate::dense::Dense::from_data(&[
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

        let output = s.mul(d).unwrap();

        assert_eq!(output,output_ref);
    }
}