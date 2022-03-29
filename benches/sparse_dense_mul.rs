use rand::prelude::*;
use criterion::*;

use basic_sparse_matrix::{dense::Dense,sparse::Csr};

fn sd_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("sd_mul");
    for i in 0..9 {
        let density = (i+1)*10;
        let elements = 10000*density;
        group.throughput(Throughput::Elements(elements as u64));
        group.bench_with_input(BenchmarkId::from_parameter(elements), &elements, |b, &e| {
            let mut a = Csr::<u32>::new((1000,1000));
            let mut x = Dense::<u32>::new_default_with_dims(10,1000);
            // assign random vars
            let mut rng = rand::rngs::StdRng::seed_from_u64(1000);
            for _ in 0..e {
                let row = (rng.next_u64() % 1000) as usize;
                let col = (rng.next_u64() % 1000) as usize;
                let v = (rng.next_u64() %  255) as u32;
                a.insert(v,row,col).unwrap();
            }
            let a = a.finalise();
            for _ in 0..(e/100) {
                let col = (rng.next_u64() % 10) as usize;
                let row = (rng.next_u64() % 1000) as usize;
                let v = (rng.next_u64() %  255) as u32;
                x.get_col_mut(col)[row] = v;
            }
            b.iter(|| {
                let _output = a.mul_dense(&x).unwrap();
            });
        });
    }
}

fn ss_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("ss_add");
    for i in 0..9 {
        let density = (i+1)*10;
        let elements = 10000*density;
        group.throughput(Throughput::Elements(elements as u64));
        group.bench_with_input(BenchmarkId::from_parameter(elements), &elements, |b, &e| {
            let mut a = Csr::<u32>::new((1000,1000));
            let mut x = Csr::<u32>::new((1000,1000));
            // assign random vars
            let mut rng = rand::rngs::StdRng::seed_from_u64(1000);
            for _ in 0..e {
                let row = (rng.next_u64() % 1000) as usize;
                let col = (rng.next_u64() % 1000) as usize;
                let v = (rng.next_u64() %  255) as u32;
                a.insert(v,row,col).unwrap();
            }
            let a = a.finalise();
            for _ in 0..e {
                let row = (rng.next_u64() % 1000) as usize;
                let col = (rng.next_u64() % 1000) as usize;
                let v = (rng.next_u64() %  255) as u32;
                x.insert(v,row,col).unwrap();
            }
            let x = x.finalise();
            b.iter(|| {
                let _output = a.add_sparse(&x).unwrap();
            });
        });
    }
}



criterion_group!(benches, sd_mul, ss_add);
criterion_main!(benches);