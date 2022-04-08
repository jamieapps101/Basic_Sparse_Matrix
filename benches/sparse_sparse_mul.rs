use rand::prelude::*;
use criterion::*;

use basic_sparse_matrix::sparse::Csr;

fn ss_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("ss_mul");
    for i in 0..5 {
        let density = (i+1)*10*3;
        let elements = density;
        group.throughput(Throughput::Elements(elements as u64));
        group.sample_size(10);
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
                let _output = a.mul_sparse(&x).unwrap();
            });
        });
    }
}

criterion_group!(benches, ss_mul);
criterion_main!(benches);