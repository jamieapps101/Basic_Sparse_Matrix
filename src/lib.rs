pub mod sparse;
pub mod dense;
pub mod util;


use util::GetDims;
use dense::Dense;
use sparse::Csr;

pub fn solve(a: Csr<f32>, x: Dense<f32>, b: &mut Dense<f32>) {
    /*
    Ax=b
    A->LL*
    LL*x=b
    Ly=b
    L*x=y

    */
    let l = a.cholesky_decomp().unwrap();
    let l_star = l.transpose();
    let y =  forward_substitution(l,b);
 


    // backward substitution

}


/// forward substitution, solve Ly=b for y
fn forward_substitution(l: Csr<f32>, b: &Dense<f32>) -> Dense<f32> {
    let mut y = Dense::<f32>::new_default_with_dims(b.get_dims().cols, b.get_dims().rows);
    // forward substitution, solve Ly=b for y
    for col_index in 0..y.get_dims().cols {
        for row_index in 0..y.get_dims().rows {
            let b_element = b.get_col(col_index)[row_index];
            let mut l_x: f32 = 0.0;
            for entry in l.get_row_compact(row_index).unwrap() {
                if row_index != entry.col_index {
                    l_x = entry.v*y.get_col(col_index)[entry.col_index];
                }
            }
            let l_element = l.get_row_compact(row_index).unwrap().last().unwrap().v;
            let y_element = (b_element-l_x)/l_element;
            y.get_col_mut(col_index)[row_index] = y_element;
        }
    }
    y
}


#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn forward_substitution_test() {
        let b = Dense::<f32>::from_data(&[
            &[5.0],
            &[7.0],
            &[2.0],
            &[9.0],
            &[1.0],
        ]);

        let l = Csr::<f32>::from_data(&[
            &[1.0,0.0,0.0,0.0,0.0],
            &[2.0,1.0,0.0,0.0,0.0],
            &[2.0,3.0,1.0,0.0,0.0],
            &[2.0,3.0,4.0,1.0,0.0],
            &[2.0,3.0,4.0,5.0,1.0],
        ]);

        let y = forward_substitution(l, &b);

        println!("y:\n{y:?}");
    }
}






#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
