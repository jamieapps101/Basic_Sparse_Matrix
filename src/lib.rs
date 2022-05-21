pub mod sparse;
pub mod dense;
pub mod dense_static;
pub mod util;


use util::GetDims;
use dense::Dense;
use sparse::Csr;

pub fn solve(a: Csr<f32>, b: Dense<f32>) -> Dense<f32> {
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
    backward_substitution(l_star, y)
}


/// forward substitution, solve Ly=b for y
fn forward_substitution(l: Csr<f32>, b: Dense<f32>) -> Dense<f32> {
    let dims = b.get_dims();
    let mut y = Dense::<f32>::new_default_with_dims(dims.cols, dims.rows);
    for col_index in 0..y.get_dims().cols {
        for row_index in 0..y.get_dims().rows {
            let b_element = b.get_col(col_index)[row_index];
            let mut l_x = 0.0;
            let row = l.get_row_compact(row_index);
            for entry in &row {
                if entry.col_index != row_index {
                    l_x += entry.v*y.get_col(col_index)[entry.col_index];
                }
            }
            let next_y = (b_element - l_x)/row.last().unwrap().v;
            y.get_col_mut(col_index)[row_index] = next_y;
        }
    }
    y
}

/// bakward substitution, solved L*x=y for x
fn backward_substitution(l_star: Csr<f32>, y: Dense<f32>) -> Dense<f32> {
    let dims = y.get_dims();
    let mut x = Dense::<f32>::new_default_with_dims(dims.cols, dims.rows);
    for col_index in 0..x.get_dims().cols {
        for row_index in (0..x.get_dims().rows).rev() {
            let y_element = y.get_col(col_index)[row_index];
            let mut l_x = 0.0;
            let row = l_star.get_row_compact(row_index);
            for entry in row.iter().skip(1) {
                l_x += entry.v*x.get_col(col_index)[entry.col_index]
            }
            let next_x = (y_element - l_x)/row[0].v;
            x.get_col_mut(col_index)[row_index] = next_x;
        }
    }
    x
}



#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn forward_substitution_test_0() {
        let b = Dense::<f32>::from_data(&[
            &[7.0,3.0,1.0],
        ]);

        let l = Csr::<f32>::from_data(&[
            &[5.0,0.0,0.0],
            &[8.0,2.0,0.0],
            &[3.0,7.0,1.0],
        ]);

        let y_ref = Dense::<f32>::from_data(&[
            &[(7.0/5.0),-4.1,25.5],
        ]);

        let y = forward_substitution(l, b);

        println!("y:\n{y:?}");

        assert_eq!(y,y_ref)
    }

    #[test]
    fn backward_substitution_test_0() {
        let y = Dense::<f32>::from_data(&[
            &[1.0,7.0,3.0],
        ]);

        let l_star = Csr::<f32>::from_data(&[
            &[7.0,1.0,8.0],
            &[0.0,2.0,3.0],
            &[0.0,0.0,5.0],
        ]);

        let x_ref = Dense::<f32>::from_data(&[
            &[(-32.0/35.0), 2.6, 0.6],
        ]);

        let x = backward_substitution(l_star, y);

        println!("x:\n{x:?}");

        assert_eq!(x,x_ref)
    }

    #[test]
    fn solve_test() {
        let b = Dense::from_data(&[&[5.0,2.0,8.0,1.0]]);

        let a = Csr::from_data(&[
            &[8.0, 0.0, 0.0, 0.0],
            &[0.0, 7.0, 1.0, 0.0],
            &[0.0, 1.0, 3.0, 0.0],
            &[0.0, 0.0, 0.0, 2.0]
        ]);

        let x_ref = Dense::from_data(&[
            &[ 0.625, -0.1,2.6999998, 0.5]
            ]);

        let x = solve(a, b);

        assert_eq!(x,x_ref);

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
