use std::fmt;
use crate::util::{GetDims,MatDim,MatErr};
use crate::dense::Dense;
use crate::dense_static::DenseS;


#[derive(PartialEq,std::fmt::Debug)]
pub struct COOEntry<T> {
    row: usize,col: usize, value: T
}


impl<T> From<(usize,usize,T)> for COOEntry<T> {
    fn from(e: (usize,usize,T)) -> Self {
        Self {
            row: e.0,
            col: e.1,
            value: e.2
        }
    }
}

// use core::cmp::Ordering;
use std::cmp::{Ordering,PartialOrd};

impl<T: PartialEq> PartialOrd for COOEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.row == other.row {
            return Some(self.col.cmp(&other.col));
        }
        Some(self.row.cmp(&other.row))
    }
}

pub struct COO<T> {
    entries: Vec<COOEntry<T>>,
    dims: MatDim,
}

impl<T> COO<T> {
    pub fn with_capacity<D: Into<MatDim>>(dims: D, capacity: usize) -> Self {
        Self { entries: Vec::with_capacity(capacity), dims: dims.into() }
    }

    pub fn insert<E: Into<COOEntry<T>>>(&mut self, entry: E) -> Result<(),MatErr> {
        let entry = entry.into();
        if self.dims.rows <= entry.row || self.dims.cols <= entry.col {
            return Err(MatErr::OutOfBounds)
        }
        self.entries.push(entry);
        Ok(())
    }
}


impl<T: Copy+Default+std::fmt::Debug+PartialEq> From<COO<T>> for Csr<T> {
    fn from(mut m: COO<T> ) -> Self {
        let mut csr = Csr::new_with_capacity(m.dims, m.entries.len());
        m.entries.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for e in m.entries {
            println!("e: {e:?}");
            csr.insert(e.value, e.row, e.col).unwrap();
        }
        csr.finalise()
    }
}

#[derive(PartialEq,Clone)]
pub struct Csr<T> {
    dims: MatDim,
    v: Vec<T>,
    col_index: Vec<usize>,
    row_index: Vec<usize>,
    is_finalised: bool,

    iter_v_index: usize,
    iter_row_index: usize
}

#[derive(PartialEq, Debug)]
pub struct CsrEntry<T: std::fmt::Debug> {
    pub v: T,
    pub col_index: usize,
    pub row_index: usize
}

impl<T: std::fmt::Debug> From<(T,usize,usize)> for CsrEntry<T> {
    fn from(e: (T,usize,usize)) -> Self {
        CsrEntry { v: e.0, row_index: e.1, col_index: e.2 }
    }
}

impl<T: std::fmt::Debug + Copy> Iterator for Csr<T> {
    type Item = CsrEntry<T>;
    fn next(&mut self) -> Option<Self::Item> {

        if self.iter_v_index == self.v.len() {
            return None
        }

        while self.row_index[self.iter_row_index] == self.iter_v_index {
            self.iter_row_index+=1;
        }

        let v = self.v[self.iter_v_index];
        let col_index = self.col_index[self.iter_v_index];
        let row_index = self.iter_row_index-1;


        self.iter_v_index += 1;

        Some(CsrEntry { v, col_index, row_index })
    }
}

impl<T: Copy + Default + PartialEq + std::fmt::Debug> Csr<T> {
    pub fn new<D: Into<MatDim>>(dims: D) -> Self {
        Self::new_with_capacity(dims, 0)
    }

    pub fn new_with_capacity<D: Into<MatDim>>(dims: D, capacity: usize) -> Self {
        let dims = dims.into();
        Self {
            dims,
            v: Vec::with_capacity(capacity),
            col_index: Vec::with_capacity(capacity),
            row_index: vec![0],
            is_finalised: false,
            iter_v_index:   0,
            iter_row_index: 0
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
            iter_v_index:   0,
            iter_row_index: 0
        };
        for n in 0..dims.cols {
            m.insert_unchecked(value, n, n);
        }
        Ok(m.finalise())
    }

    pub fn create_diagonal(contents: Vec<T>) -> Self {
        let mut m = Self::new_with_capacity((contents.len(),contents.len()), contents.len());
        for (i,v) in contents.iter().enumerate() {
            m.insert(*v, i, i).unwrap();
        }
        m.finalise()
    }

    pub fn get_nnz(&self) -> usize {
        *self.row_index.last().unwrap_or(&0)
    }

    pub fn get_density(&self) -> f32 {
        self.v.len() as f32 / (self.dims.rows*self.dims.cols) as f32
    }

    pub fn get_val_at<D: Into<MatDim>>(&self, at: D) -> Option<&T> {
        let at = at.into();
        let row_start = self.row_index[at.rows];
        let row_end = self.row_index[at.rows+1];
        for (local_index,col_index) in self.col_index[row_start..row_end].iter().enumerate() {
            if col_index == &at.cols {
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
            if self.dims.rows < self.row_index.len() {
                panic!("big eek")
            }
            let required_spacers = self.dims.rows-self.row_index.len();
            for _ in 0..required_spacers {
                self.row_index.push(self.v.len());
            }
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

    pub fn get_row_compact(&self, index: usize) -> Vec<CsrEntry<&T>> {
        // assert!(self.is_finalised);
        let mut return_row = Vec::with_capacity(self.dims.cols);
        let row_start = self.row_index[index];
        let row_end = if index == self.row_index.len()-1 {
            self.v.len()
        } else {
            self.row_index[index+1]
        };
        for (col_index,v) in self.col_index[row_start..row_end].iter().zip(self.v[row_start..row_end].iter()) {
            return_row.push(CsrEntry { v, col_index: *col_index, row_index: index });
        }
        return_row
    }

    pub fn get_row_complete(&self, index: usize) -> Option<Vec<T>> {
        if self.row_index.is_empty() { return None }
        // need to handle special case where only enties in the index row exist with none in row index+1
        if index >= self.row_index.len() { return None }

        let row_start = self.row_index[index];
        let index_with_offset = index+1;
        let row_end = if self.row_index.len() == index_with_offset {
            self.v.len()
        } else {
            self.row_index[index+1]
        };
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
        Some(return_vec)
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
        Some(return_vec)
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

    pub fn take_submatrix<D: Into<MatDim>>(&self, from: D, to: D) -> Result<Self,MatErr> {
        if !self.is_finalised { return Err(MatErr::MatrixNotFinalised) }
        let from = from.into();
        let to = to.into();
        // check requirements
        // todo: return a nice error here if these aren't true
        assert!(from.cols < to.cols);
        assert!(to.cols <= self.dims.cols+1);
        assert!(from.rows < to.rows);
        assert!(to.rows <= self.dims.rows+1);

        let submat_dims = (to.rows-from.rows,to.cols-from.cols);
        let mut submatrix = Csr::new(submat_dims);

        let mut row_index = 0;
        for i in 0..self.v.len() {
            while self.row_index[row_index] == i {
                row_index += 1;
            }
            if (row_index-1) >= from.rows && (row_index-1) < to.rows {
                let col_index = self.col_index[i];
                if col_index >= from.cols && col_index < to.cols {
                    // TODO: (row_index-1) makes first test pass, (row_index) makes second test pass
                    // something broken in insert_unchecked
                    let local_row_index = (row_index)-from.rows;
                    let local_col_index = col_index-from.cols;
                    submatrix.insert(self.v[i], local_row_index-1, local_col_index).unwrap();
                }
            }
        }

        Ok(submatrix.finalise())
    }

    pub fn reset_iter(&mut self) {
        self.iter_row_index = 0; self.iter_v_index = 0;
    }
}

impl<T> GetDims for Csr<T> {
    fn get_dims(&self) -> MatDim {
        self.dims
    }
}


impl<T: std::iter::Sum + Copy + Default + PartialEq + std::fmt::Debug + std::ops::Add<T,Output=T> + std::ops::Sub<T,Output=T> + std::ops::Mul<T,Output=T>> Csr<T> {
    pub fn mul_dense(&self, rhs: &Dense<T>) -> Result<Self,MatErr> {
        if self.dims.cols != rhs.get_dims().rows {
            return Err(MatErr::IncorrectDimensions)
        }
        let mut result = Self::new((self.dims.rows, rhs.get_dims().cols));
        for row_index in 0..self.dims.rows {
            let row = self.get_row_compact(row_index);
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

    pub fn mul_dense_s<const ROWS: usize, const COLS: usize>(&self, rhs: &DenseS<T,ROWS,COLS>) -> Result<Self,MatErr> {
        if self.dims.cols != ROWS { return Err(MatErr::IncorrectDimensions) }
        let mut result = Self::new((self.dims.rows, rhs.get_dims().cols));
        for row_index in 0..self.dims.rows {
            let row = self.get_row_compact(row_index);
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

    pub fn mul_vector(&self, rhs: &[T], out: &mut [T]) -> Result<(),MatErr> {
        if self.dims.cols != rhs.len() || self.dims.rows != out.len() {
            return Err(MatErr::IncorrectDimensions)
        }
        // transposing means that the row access becomes col access, which in a CSR matrix
        // is considerably faster
        let self_t = self.transpose();
        for (row_index,result_item) in out.iter_mut().enumerate().take(self.dims.rows) {
            let row = self_t.get_col_compact(row_index).unwrap();
            *result_item  = row.iter().map(|row_entry|{
                *row_entry.v * rhs[row_entry.row_index]
            }).sum();
        }
        Ok(())
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
            let self_row = self.get_row_compact(row_index);
            let rhs_row = rhs.get_row_compact(row_index);
            let mut self_entry_index = 0;
            let mut rhs_entry_index = 0;
            loop {
                let self_has_entry = !self_row.is_empty() && self_row.len() > self_entry_index;
                let rhs_has_entry = !rhs_row.is_empty() && rhs_row.len() > rhs_entry_index;
                if self_has_entry && rhs_has_entry {
                    let self_entry = &self_row[self_entry_index];
                    let rhs_entry  = &rhs_row[rhs_entry_index];

                    match self_entry.col_index.cmp(&rhs_entry.col_index) {
                        Ordering::Greater => {
                            output.insert(*rhs_entry.v, rhs_entry.row_index, rhs_entry.col_index).unwrap();
                            rhs_entry_index+=1;
                        }
                        Ordering::Less => {
                            output.insert(*self_entry.v, self_entry.row_index, self_entry.col_index).unwrap();
                            self_entry_index+=1;
                        }
                        Ordering::Equal => {
                            rhs_entry_index+=1;
                            self_entry_index+=1;
                            let v = *self_entry.v + *rhs_entry.v;
                            output.insert(v, self_entry.row_index, self_entry.col_index).unwrap();
                        }
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
            println!("{} vs {}",self.get_dims(),rhs.get_dims());
            return Err(MatErr::IncorrectDimensions)
        }
        let mut output = Self::new(self.dims);

        // create two iterators over the elements which can asyncronously advance
        // let mut self_row_index = 0;
        let mut row_index = 0;
        loop {
            let self_row = self.get_row_compact(row_index);
            let rhs_row = rhs.get_row_compact(row_index);
            let mut self_entry_index = 0;
            let mut rhs_entry_index: usize = 0;
            loop {
                let self_has_entry = !self_row.is_empty() && self_row.len() > self_entry_index;
                let rhs_has_entry = !rhs_row.is_empty() && rhs_row.len() > rhs_entry_index;
                if self_has_entry && rhs_has_entry {
                    let self_entry = &self_row[self_entry_index];
                    let rhs_entry  = &rhs_row[rhs_entry_index];
                    match self_entry.col_index.cmp(&rhs_entry.col_index) {
                        Ordering::Greater => {
                            output.insert(T::default()-*rhs_entry.v, rhs_entry.row_index, rhs_entry.col_index).unwrap();
                            rhs_entry_index+=1;
                        }
                        Ordering::Less => {
                            output.insert(*self_entry.v, self_entry.row_index, self_entry.col_index).unwrap();
                            self_entry_index+=1;
                        }
                        Ordering::Equal => {
                            rhs_entry_index+=1;
                            self_entry_index+=1;
                            let v = *self_entry.v - *rhs_entry.v;
                            output.insert(v, self_entry.row_index, self_entry.col_index).unwrap();
                        }
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

    pub fn mul_sparse(&self, rhs: &Self) -> Result<Self,MatErr> {
        let mut result = Self::new((self.dims.rows,rhs.dims.cols));
        let rhs_t = rhs.transpose();
        for row_index in 0..self.dims.rows {
            let row = self.get_row_compact(row_index);
            for col_index in 0..rhs_t.dims.rows {
                let  col = rhs_t.get_row_compact(col_index);
                let mut row_position = 0;
                let mut col_position = 0;
                let mut val = T::default();
                while col_position != col.len() && row_position != row.len() {
                    match row[row_position].col_index.cmp(&col[col_position].col_index) {
                        Ordering::Equal => {
                            let t = *row[row_position].v * *col[col_position].v;
                            val = val + t;
                            row_position += 1;
                            col_position += 1;
                        }
                        Ordering::Greater => {
                            col_position += 1;
                        }
                        Ordering::Less => {
                            row_position += 1;
                        }
                    }

                }
                if val != T::default() {
                    result.insert(val, row_index, col_index).unwrap();
                }
            }
        }

        Ok(result.finalise())
    }

    pub fn sum_elements(&self) -> T {
        let mut sum = T::default();
        for v in &self.v {
            sum = sum + *v;
        }
        sum
    }

    pub fn mul_scalar(&self, s: T) -> Self {
        let mut output = self.clone();
        output.v.iter_mut().for_each(|v|{
            *v = *v*s;
        });

        output
    }


    pub fn add_padding<D: Into<MatDim>, C: Into<MatDim>>(&self, padded_size: C, at: D) -> Result<Self,MatErr> {
        let padded_size = padded_size.into();
        let offset = at.into();
        if self.dims.cols > padded_size.cols || self.dims.rows > padded_size.cols {
            return Err(MatErr::PaddingSizeSmallerThanOriginal)
        }

        if padded_size.rows < self.dims.rows + offset.rows || padded_size.cols < self.dims.cols + offset.cols {
            println!("padded_size: {padded_size}");
            println!("offset: {offset}");
            println!("self.dims: {}",self.dims);
            return Err(MatErr::IncorrectDimensions)
        }

        let mut return_mat = Self::new(padded_size);
        for e in self.clone() {
            return_mat.insert(e.v, e.row_index+offset.rows, e.col_index+offset.cols).unwrap();
        }
        Ok(return_mat.finalise())
    }
}

impl Csr<f32> {
    pub fn l2_norm(&self) -> f32 {
        self.v.iter().map(|v| v.powi(2)).sum::<f32>().powf(0.5)
    }

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
                let val_to_insert = if i==j {
                    (self.get_row_complete(i).unwrap()[i]-sum).powf(0.5)
                } else {
                    let temp = self.get_row_complete(i).unwrap()[j]-sum;
                    let a = l.get_row_complete(j).unwrap()[j];
                    (1.0 / a) * temp
                };
                l.insert(val_to_insert, i, j).unwrap();
            }
        }
        Ok(l.finalise())
    }

    pub fn qr_decomp(&self) -> (Self,Self) {
        let mut working_mat = self.clone();
        let mut q_queue = Vec::with_capacity(self.dims.cols);

        for i in 0..(self.dims.cols-1) {
            let x = working_mat.get_col(0).unwrap();

            let mut e = Csr::new((self.dims.rows-i,1));
            e.insert(1.0, 0, 0).unwrap();
            let e = e.finalise();
            let a_mag = x.l2_norm();

            let u = x.sub_sparse(&e.mul_scalar(a_mag)).unwrap();
            let v = u.mul_scalar(1.0/u.l2_norm());

            let temp_a = Csr::eye((self.dims.rows-i,self.dims.rows-i), 1.0).unwrap();
            let temp_b = v.mul_sparse(&v.transpose()).unwrap().mul_scalar(2.0);
            let q = temp_a.sub_sparse(&temp_b).unwrap();
            q_queue.push(q.clone());
            let q_a = q.mul_sparse(self).unwrap();
            working_mat = q_a.take_submatrix((1,1), (self.dims.rows-i,self.dims.cols-i)).unwrap();
        }

        let mut q = Csr::eye(self.dims,1.0).unwrap();
        for (i,q_undersized) in q_queue.iter().enumerate() {
            let q_local = if i == 0 {
                q_undersized.clone()
            } else {
                let mut temp = Csr::new(self.dims);
                for j in 0..i {
                    temp.insert(1.0, j, j).unwrap();
                }
                let temp_2 = q_undersized.add_padding(self.dims, (i,i)).unwrap();
                temp.finalise().add_sparse(&temp_2).unwrap()
            };
            q = q.mul_sparse(&q_local.transpose()).unwrap();
        }

        let r = q.transpose().mul_sparse(self).unwrap();
        (q,r)
    }

    /// this uses the QR algorithm
    pub fn eigen_values(&self, iterations: usize) -> Vec<f32> {
        let mut working_a = self.clone();

        for _ in 0..iterations {
            let (q,r) = working_a.qr_decomp();
            working_a = r.mul_sparse(&q).unwrap();
        }

        let mut eigen_values = Vec::new();
        for i in 0..working_a.dims.cols {
            eigen_values.push(*working_a.get_val_at((i,i)).unwrap());
        }

        eigen_values

    }
}

impl<T: fmt::Display + Copy + Default + PartialEq + fmt::Debug> fmt::Display for Csr<T> {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for row_index in 0..self.dims.rows {
            write!(f, "|").unwrap();
            if let Some(row) = self.get_row_complete(row_index) {
                for entry in row {
                    write!(f, "{:>5} ", entry).unwrap();
                }
            } else {
                for _ in 0..self.dims.cols {
                    write!(f, "{:>5} ", T::default()).unwrap();
                }
            }
            writeln!(f, "|").unwrap();
        }
        Ok(())
    }
}

impl<T: fmt::Display + Copy + Default + PartialEq + fmt::Debug > fmt::Debug for Csr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "dims:      {}", self.dims).unwrap();
        writeln!(f, "v:         {:?}", self.v ).unwrap();
        writeln!(f, "col_index: {:?}", self.col_index ).unwrap();
        writeln!(f, "row_index: {:?}", self.row_index ).unwrap();
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
    fn create_mat_by_insert() {
        let mut b = Csr::new((3,3));
        for (col,v) in [5,6,7].iter().enumerate() {
            b.insert(*v, 0, col).unwrap();
        }
        let b = b.finalise();
        let b_ref = Csr::from_data(&[
            &[5,6,7],
            &[0,0,0],
            &[0,0,0],
        ]);

        assert_eq!(b,b_ref);
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

        assert_eq!(m.get_row_compact(2), vec![
            CsrEntry {v: &50, row_index: 2, col_index: 2},
            CsrEntry {v: &60, row_index: 2, col_index: 3},
            CsrEntry {v: &70, row_index: 2, col_index: 4},
        ]);
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
            CsrEntry {v: &70, row_index: 2, col_index: 4}
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

        assert_eq!(m.get_row_compact(0), vec![
            CsrEntry {v: &5, row_index: 0, col_index: 0},
            CsrEntry {v: &6, row_index: 0, col_index: 1},
            CsrEntry {v: &7, row_index: 0, col_index: 2},
            CsrEntry {v: &8, row_index: 0, col_index: 3},
            CsrEntry {v: &9, row_index: 0, col_index: 4},
        ]);

        assert_eq!(m.get_row_compact(1), vec![]);

        assert_eq!(m.get_row_compact(2), vec![
            CsrEntry {v: &1, row_index: 2, col_index: 4},
        ]);

        assert_eq!(m.get_row_compact(3), vec![
            CsrEntry {v: &1, row_index: 3, col_index: 0},
        ]);

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
        assert_eq!(m.row_index,vec![0,0,3,3]);
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

    #[test]
    fn sparse_multiplication() {
        // let a = Csr::from_data(&[
        //     &[1.0,2.0,3.0],
        //     &[4.0,5.0,6.0],
        //     &[7.0,8.0,9.0],
        // ]);

        // let b = Csr::eye((3,3), 1.0).unwrap();
        // // println!("b:\n{b}");

        // let c = a.mul_sparse(&b).unwrap();

        // let c_ref = Csr::from_data(&[
        //     &[1.0,2.0,3.0],
        //     &[4.0,5.0,6.0],
        //     &[7.0,8.0,9.0],
        // ]);

        // // println!("c:\n{c}");
        // assert_eq!(c,c_ref);

        ///// Round 2
        // let a = Csr::from_data(&[
        //     &[1,3,5],
        //     &[3,7,9],
        //     &[5,9,11],
        // ]);

        // let b = Csr::from_data(&[
        //     &[2,4,6],
        //     &[4,8,10],
        //     &[6,10,12],
        // ]);

        // let c = a.mul_sparse(&b).unwrap();

        // let c_ref = Csr::from_data(&[
        //     &[ 44, 78, 96],
        //     &[ 88,158,196],
        //     &[112,202,252],
        // ]);

        // assert_eq!(c,c_ref);

        ///// Round 3
        let a = Csr::from_data(&[
            &[ 0 ],
            &[ 1 ],
            &[ 1 ],
        ]);

        let b = a.transpose();

        let c = a.mul_sparse(&b).unwrap();

        let c_ref = Csr::from_data(&[
            &[0,0,0],
            &[0,1,1],
            &[0,1,1],
        ]);

        assert_eq!(c,c_ref);

    }



    #[test]
    fn mul_scalar() {
        let a = Csr::from_data(&[
            &[1.0,2.0,3.0],
            &[4.0,5.0,6.0],
            &[7.0,8.0,9.0],
        ]);

        let b_ref = Csr::from_data(&[
            &[2.0,4.0,6.0],
            &[8.0,10.0,12.0],
            &[14.0,16.0,18.0],
        ]);

        let b = a.mul_scalar(2.0);
        assert_eq!(b,b_ref);
    }

    #[test]
    fn test_submatrix() {
        let a = Csr::from_data(&[
            &[5,6,7,8,9],
            &[0,0,0,0,0],
            &[0,0,0,0,1],
            &[1,0,0,0,0],
        ]);

        // test feasible opts first
        println!("A");
        let b = a.take_submatrix((0,0), (3,3)).unwrap();
        let b_ref = Csr::from_data(&[
            &[5,6,7],
            &[0,0,0],
            &[0,0,0],
        ]);
        ///////////////////////////////
        assert_eq!(b,b_ref);
        ///////////////////////////////

        println!("B");
        let b = a.take_submatrix((1,2), (4,5)).unwrap();
        let b_ref = Csr::from_data(&[
            &[0,0,0],
            &[0,0,1],
            &[0,0,0],
        ]);
        ///////////////////////////////
        assert_eq!(b,b_ref);
        ///////////////////////////////

        println!("C");
        let b = a.take_submatrix((0,2), (3,5)).unwrap();
        let b_ref = Csr::from_data(&[
            &[7,8,9],
            &[0,0,0],
            &[0,0,1],
        ]);
        ///////////////////////////////
        assert_eq!(b,b_ref);
        ///////////////////////////////
    }

    #[test]
    fn qr_decomp() {
        let a = Csr::from_data(&[
            &[ 12.0,-51.0,  4.0],
            &[  6.0,167.0,-68.0],
            &[ -4.0, 24.0,-41.0],
        ]);

        let (q,r) = a.qr_decomp();

        let q_r = q.mul_sparse(&r).unwrap();
        assert!(a.sub_sparse(&q_r).unwrap().l2_norm() < 0.1 );
    }


    #[test]
    fn test_iterator() {
        let mut a = Csr::from_data(&[
            &[5,0,0,0],
            &[0,8,0,0],
            &[0,0,3,0],
            &[0,6,0,0],
        ]);

        assert_eq!(a.next(), Some((5,0,0).into()));
        assert_eq!(a.next(), Some((8,1,1).into()));
        assert_eq!(a.next(), Some((3,2,2).into()));
        assert_eq!(a.next(), Some((6,3,1).into()));
        assert_eq!(a.next(), None);
    }


    #[test]
    fn add_padding() {
        let a = Csr::from_data(&[
            &[1,1,1],
            &[1,0,0],
            &[1,0,0],
        ]);

        let padded_a = a.add_padding((5,5), (2,2)).unwrap();


        let padded_a_ref = Csr::from_data(&[
            &[0,0,0,0,0],
            &[0,0,0,0,0],
            &[0,0,1,1,1],
            &[0,0,1,0,0],
            &[0,0,1,0,0],
        ]);

        assert_eq!(padded_a,padded_a_ref);
    }


    #[test]
    fn get_eigen_vals() {
        let a = Csr::from_data(&[
            &[ 12.0,-51.0,  4.0],
            &[  6.0,167.0,-68.0],
            &[ -4.0, 24.0,-41.0],
        ]);

        let eigen_values_ref = vec![-34.196675, 16.05999094, 156.13668406];
        for c in 0..20 {
            let mut eigen_values = a.eigen_values(c);
            eigen_values.sort_by(|a,b| a.partial_cmp(b).unwrap() );
            let err = eigen_values.iter().zip(eigen_values_ref.iter()).map(|(v,r)|{
                (v-r).powi(2)
            }).sum::<f32>().powf(0.5);
            println!("({err})");
        }
    }

    #[test]
    fn coo_to_csr() {
        let mut coo = COO::with_capacity((5,6), 8);
        coo.insert((0,0,1.0)).unwrap();
        coo.insert((1,1,2.0)).unwrap();
        coo.insert((1,2,3.0)).unwrap();
        coo.insert((2,2,4.0)).unwrap();
        coo.insert((2,3,5.0)).unwrap();
        coo.insert((3,3,6.0)).unwrap();
        coo.insert((3,4,7.0)).unwrap();
        coo.insert((4,4,8.0)).unwrap();
        coo.insert((4,5,9.0)).unwrap();

        let csr = Csr::from(coo);
        // println!("csr:\n{csr}");


        let ref_csr = Csr::from_data(&[
            &[1.0,0.0,0.0, 0.0,0.0,0.0],
            &[0.0,2.0,3.0, 0.0,0.0,0.0],
            &[0.0,0.0,4.0, 5.0,0.0,0.0],
            &[0.0,0.0,0.0, 6.0,7.0,0.0],
            &[0.0,0.0,0.0, 0.0,8.0,9.0],
        ]);
        // println!("ref_csr:\n{ref_csr}");

        assert_eq!(ref_csr,csr);

    }

    #[test]
    fn create_diagonal() {
        // diagonal with non-zero entries
        let c = vec![1,2,3,4];
        let ref_m = Csr::from_data(&[
            &[1,0,0,0],
            &[0,2,0,0],
            &[0,0,3,0],
            &[0,0,0,4],
        ]);
        let m = Csr::create_diagonal(c);
        assert_eq!(m,ref_m);

        // create digonal with zero entries
        let c = vec![0,1,0,2,0,3,0];
        let ref_m = Csr::from_data(&[
            &[0,0,0,0,0,0,0],
            &[0,1,0,0,0,0,0],
            &[0,0,0,0,0,0,0],
            &[0,0,0,2,0,0,0],
            &[0,0,0,0,0,0,0],
            &[0,0,0,0,0,3,0],
            &[0,0,0,0,0,0,0],
        ]);
        let m = Csr::create_diagonal(c);
        assert_eq!(m,ref_m);
    }

    #[test]
    fn test_mul_vector() {
        let v = vec![0,1,2,3,4];
        let m = Csr::from_data(&[
            &[0,0,0,0],
            &[0,0,0,0],
            &[0,0,0,0],
        ]);
        let mut out = vec![0;5];
        assert_eq!(m.mul_vector(&v,&mut out),Err(MatErr::IncorrectDimensions));

        let m = Csr::from_data(&[
            &[1,0,0,0,0],
            &[0,1,0,0,0],
            &[0,0,1,0,0],
            &[0,0,0,1,0],
            &[0,0,0,0,1],
        ]);
        assert_eq!(m.mul_vector(&v,&mut out), Ok(()));
        assert_eq!(out, v.clone());

        let m = Csr::from_data(&[
            &[1,0,2,0,3],
            &[0,1,0,2,0],
        ]);
        let mut out = vec![0;2];
        assert_eq!(m.mul_vector(&v,&mut out), Ok(()));
        assert_eq!(out, vec![16,7]);
    }
}