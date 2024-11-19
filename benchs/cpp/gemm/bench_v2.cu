#include "cutlass/cutlass_gemm.cuh"
#include "tiledcuda/tiledcuda_gemm.cuh"
#include "util.cuh"
#include "utils/cpp/cuda_info.cuh"

#include <cutlass/half.h>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

void run_test_v2(std::ofstream& fout) {
    using Element = cutlass::half_t;
    using InType = __half;
    using AccType = float;

    static constexpr int kM = 4096;
    static constexpr int kN = 4096;
    static constexpr int kK = 2048;

    static constexpr int kTM = 128;
    static constexpr int kTN = 128;
    static constexpr int kTK = 128;

    static constexpr int kWarpPerRow = 2;
    static constexpr int kWarpPerCol = 2;

    using WholeShape = GemmShape<kM, kN, kK>;
    using CtaTileShape = GemmShape<kTM, kTN, kTK>;
    using WarpLayout = tl::RowMajor<kWarpPerRow, kWarpPerCol>;

    static constexpr int kRK = 32;

    using Config = KeGemmTraits<InType, AccType, WholeShape, CtaTileShape, kRK,
                                WarpLayout>;

    auto tiledcuda_gemm_kernel = &tiledcuda_gemm<
        InType, AccType, kM, kN, kK, kTM, kTN, kTK, typename Config::GIteratorA,
        typename Config::SIteratorA, typename Config::SharedA,
        typename Config::RegA, typename Config::G2SLoaderA,
        typename Config::S2RLoaderA, typename Config::GIteratorB,
        typename Config::SIteratorB, typename Config::SharedB,
        typename Config::RegB, typename Config::G2SLoaderB,
        typename Config::S2RLoaderB, typename Config::GlobalC,
        typename Config::SharedC, typename Config::RegC,
        typename Config::R2SStorerC, typename Config::S2GStorerC>;

    static constexpr int smem_size_inputs = kTK * (kTN + kTM) * sizeof(InType);
    static constexpr int smem_size_accumulators = kTM * kTN * sizeof(AccType);
    static constexpr int smem_size = smem_size_inputs > smem_size_accumulators
                                         ? smem_size_inputs
                                         : smem_size_accumulators;

    const int kMaxSmemPerBlock = 48 * 1024;
    if (smem_size > kMaxSmemPerBlock) {
        cudaFuncSetAttribute(tiledcuda_gemm_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
    }

    int block_x = (kM + kTM - 1) / kTM;
    int block_y = (kN + kTN - 1) / kTN;

    dim3 grid(block_x, block_y, 1);
    dim3 block(Config::kThreads, 1, 1);

    thrust::host_vector<Element> h_a(kM * kK);
    for (int i = 0; i < h_a.size(); ++i)
        h_a[i] = static_cast<Element>(rand_float());

    thrust::host_vector<Element> h_b(kK * kN);
    for (int i = 0; i < h_b.size(); ++i)
        h_b[i] = static_cast<Element>(rand_float());

    thrust::host_vector<Element> h_c(kM * kN);
    thrust::fill(h_c.begin(), h_c.end(), 0.);

    thrust::host_vector<AccType> h_c2(kM * kN);
    thrust::fill(h_c2.begin(), h_c2.end(), 0.);

    thrust::host_vector<__half> h_c3(kM * kN);
    thrust::fill(h_c3.begin(), h_c3.end(), 0.);

    thrust::device_vector<Element> d_a = h_a;
    thrust::device_vector<Element> d_b = h_b;
    thrust::device_vector<Element> d_c = h_c;
    thrust::device_vector<AccType> d_c2 = h_c2;
    thrust::device_vector<__half> d_c3 = h_c3;

    const Element* dA = thrust::raw_pointer_cast(d_a.data());
    const Element* dB = thrust::raw_pointer_cast(d_b.data());
    Element* dC = thrust::raw_pointer_cast(d_c.data());

    const InType* dA2 = reinterpret_cast<const InType*>(dA);
    const InType* dB2 = reinterpret_cast<const InType*>(dB);
    AccType* dC2 = thrust::raw_pointer_cast(d_c2.data());

    const __half* dA3 = reinterpret_cast<const __half*>(dA);
    const __half* dB3 = reinterpret_cast<const __half*>(dB);
    __half* dC3 = thrust::raw_pointer_cast(d_c3.data());

    auto cute_gemm_kernel = &cute_gemm<Element, kWarpPerRow, kWarpPerCol, kM,
                                       kN, kK, kTM, kTN, kTK>;

    const int warm_up = 5;
    const int iters = 20;

    benchmarks::CudaTimer timer;

    for (int i = 0; i < warm_up; ++i) {
        cute_gemm_kernel(dA, dB, dC);
    }

    cudaDeviceSynchronize();

    timer.start();
    for (int i = 0; i < iters; ++i) {
        cute_gemm_kernel(dA, dB, dC);
    }
    cudaDeviceSynchronize();
    float cutlass_time = timer.stop() / iters;

    for (int i = 0; i < warm_up; ++i) {
        tiledcuda_gemm_kernel<<<grid, block, smem_size>>>(dA2, dB2, dC2);
    }

    cudaDeviceSynchronize();

    timer.start();
    for (int i = 0; i < iters; ++i) {
        tiledcuda_gemm_kernel<<<grid, block, smem_size>>>(dA2, dB2, dC2);
    }
    cudaDeviceSynchronize();
    float tiledcuda_time = timer.stop() / iters;

    float cublas_time =
        cublas_hgemm(kM, kN, kK, dA3, dB3,
                     thrust::raw_pointer_cast(d_c3.data()), true /*timeit*/);

    std::cout << "[" << kM << ", " << kN << ", " << kK << "]\t[" << kTM << ", "
              << kTN << ", " << kTK << "]\t[" << kWarpPerRow << ", "
              << kWarpPerCol << "]\t" << cublas_time << "(ms)\t" << cutlass_time
              << "(ms)\t" << tiledcuda_time << "(ms)\t"
              << cutlass_time / cublas_time << "\t"
              << tiledcuda_time / cublas_time << std::endl;

    fout << "[" << kM << ", " << kN << ", " << kK << "]\t[" << kTM << ", "
         << kTN << ", " << kTK << "]\t[" << kWarpPerRow << ", " << kWarpPerCol
         << "]\t" << cublas_time << "(ms)\t" << cutlass_time << "(ms)\t"
         << tiledcuda_time << "(ms)\t" << cutlass_time / cublas_time << "\t"
         << tiledcuda_time / cublas_time << std::endl;
}

int main() {
    std::ofstream fout;
    fout.setf(std::ios::fixed);
    fout.precision(4);

    auto dev_name = benchmarks::get_device_name();
    std::stringstream file_name;
    file_name << "bench_" << dev_name << "_gemm_v2.tsv";
    fout.open(file_name.str(), std::ios::out);

    std::cout << "[M, N, K]\t[kTM, kTN, kTK]\t[kWarpPerRow, kWarpPerCol]\t"
                 "CublasTime(ms)\tCutlassTime(ms)\tTiledCUDATime(ms)\t"
                 "CutlassRatio\tTiledCUDARatio"
              << std::endl;

    fout << "[M, N, K]\t[kTM, kTN, kTK]\t[kWarpPerRow, kWarpPerCol]\t"
            "CublasTime(ms)\tCutlassTime(ms)\tTiledCUDATime(ms)\t"
            "CutlassRatio\tTiledCUDARatio"
         << std::endl;

    run_test_v2(fout);

    return 0;
}
