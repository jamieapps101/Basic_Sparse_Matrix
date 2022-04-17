use rand::prelude::*;
use criterion::*;

use sparse_matrix::sparse::Csr;

fn ss_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("ss_mul");
    for i in [1,2,5,10,20,50,100,200,500,1000,2000,10000] {
        let density = i*50;
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