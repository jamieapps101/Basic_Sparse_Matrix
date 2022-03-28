use rand::prelude::*;
use criterion::*;

use basic_sparse_matrix::{dense::Dense,sparse::Csr};

fn bench(c: &mut Criterion) {
    c.bench_function("iter", move |bencher| {
        let mut a = Csr::<u32>::new(1000,1000);
        let mut x = Dense::<u32>::new_default_with_dims(10,1000);

        // assign random vars
        let mut rng = rand::rngs::StdRng::seed_from_u64(1000);

        for _ in 0..10000 {
            let row = (rng.next_u64() % 1000) as usize;
            let col = (rng.next_u64() % 1000) as usize;
            let v = (rng.next_u64() %  255) as u32;
            a.insert(v,row,col).unwrap();
        }
        let a = a.finalise();

        for _ in 0..100 {
            let col = (rng.next_u64() % 10) as usize;
            let row = (rng.next_u64() % 1000) as usize;
            let v = (rng.next_u64() %  255) as u32;
            x.get_col_mut(col)[row] = v;
        }

        bencher.iter_with_large_drop(|| {
            let output = a.mul_dense(&x).unwrap();
            assert_eq!(output.get_nnz(),10000)
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);