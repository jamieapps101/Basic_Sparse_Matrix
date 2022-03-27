pub mod sparse;
pub mod dense;
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
    println!("\nl:\n{l}");
    let l_star = l.transpose();
    println!("\nl_star:\n{l_star}");
    let y =  forward_substitution(l,b);
    println!("\ny:\n{y}");
    let x = backward_substitution(l_star, y);
    println!("\nx:\n{x}");
    x
}


/// forward substitution, solve Ly=b for y
fn forward_substitution(l: Csr<f32>, b: Dense<f32>) -> Dense<f32> {
    let dims = b.get_dims();
    // println!("dims: {dims:?}");
    let mut y = Dense::<f32>::new_default_with_dims(dims.cols, dims.rows);
    for col_index in 0..y.get_dims().cols {
        // println!(">col_index={col_index}");
        for row_index in 0..y.get_dims().rows {
            // println!("->row_index={row_index}");
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

/// bakward substitution, solved L*x=y for x
fn backward_substitution(l_star: Csr<f32>, y: Dense<f32>) -> Dense<f32> {
    println!("backward_substitution");
    let dims = y.get_dims();
    let mut x = Dense::<f32>::new_default_with_dims(dims.cols, dims.rows);
    for col_index in 0..x.get_dims().cols {
        println!(">col_index={col_index}");
        for row_index in (0..x.get_dims().rows).rev() {
            println!("->row_index={row_index}");
            let y_element = y.get_col(col_index)[row_index];

            let mut l_star_x: f32 = 0.0;
            for entry in l_star.get_row_compact(row_index).unwrap() {
                if row_index != entry.col_index {
                    l_star_x = entry.v*y.get_col(col_index)[entry.col_index];
                }
            }
            let l_element = l_star.get_row_compact(row_index).unwrap().last().unwrap().v;
            let x_element = (y_element-l_star_x)/l_element;
            x.get_col_mut(col_index)[row_index] = x_element;
        }
    }
    x
}



#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn forward_substitution_test() {
        let b = Dense::<f32>::from_data(&[
            &[5.0,7.0,2.0],
        ]);

        let l = Csr::<f32>::from_data(&[
            &[1.0,0.0,0.0],
            &[2.0,3.0,0.0],
            &[2.0,3.0,7.0],
        ]);

        let y_ref = Dense::<f32>::from_data(&[
            &[5.0,-1.0,0.71428573],
        ]);

        let y = forward_substitution(l, b);

        println!("y:\n{y:?}");

        assert_eq!(y,y_ref)
    }

    #[test]
    fn backward_substitution_test() {
        let y = Dense::<f32>::from_data(&[
            &[5.0,7.0,2.0],
        ]);

        let l_star = Csr::<f32>::from_data(&[
            &[2.0,3.0,7.0],
            &[2.0,3.0,0.0],
            &[1.0,0.0,0.0],
        ]);

        let x_ref = Dense::<f32>::from_data(&[
            &[-1.2857143, -1.0, -3.0],
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
            &[ 0.625, -0.1,2.7, 0.5]
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
