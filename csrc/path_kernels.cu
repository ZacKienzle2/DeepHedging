// Fused Philox path generators.
//
// One thread owns one path; the log-price (and variance) recursion lives
// entirely in registers, so a path costs one kernel launch regardless of
// step count, against one launch per step per tensor op in the eager
// simulators. The (seed, stream) pair of a NoiseSpec maps onto the Philox
// (seed, subsequence) pair as subsequence = (stream << 32) | path, which
// keeps streams disjoint for any path count below 2^32 and makes every
// draw addressable for a backward pass that regenerates noise instead of
// storing it. Output is time-major to match the MarketState contract.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <curand_kernel.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 256;

__global__ void gbm_kernel(
    float* __restrict__ spot,
    const int64_t n_paths,
    const int n_steps,
    const float s0,
    const float drift,
    const float diffusion,
    const uint64_t seed,
    const uint64_t subsequence_base) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= n_paths) {
    return;
  }
  curandStatePhilox4_32_10_t rng;
  curand_init(seed, subsequence_base + static_cast<uint64_t>(tid), 0ULL, &rng);
  float log_return = 0.0f;
  spot[tid] = s0;
  for (int k = 0; k < n_steps; ++k) {
    log_return += drift + diffusion * curand_normal(&rng);
    spot[static_cast<int64_t>(k + 1) * n_paths + tid] = s0 * expf(log_return);
  }
}

__global__ void heston_kernel(
    float* __restrict__ spot,
    float* __restrict__ variance_out,
    const int64_t n_paths,
    const int n_steps,
    const float s0,
    const float v0,
    const float dt,
    const float sqrt_dt,
    const float kappa,
    const float theta,
    const float xi,
    const float rho,
    const float mu,
    const uint64_t seed,
    const uint64_t subsequence_base) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= n_paths) {
    return;
  }
  curandStatePhilox4_32_10_t rng;
  curand_init(seed, subsequence_base + static_cast<uint64_t>(tid), 0ULL, &rng);
  const float rho_complement = sqrtf(1.0f - rho * rho);
  float log_return = 0.0f;
  float variance = v0;
  spot[tid] = s0;
  variance_out[tid] = v0;
  for (int k = 0; k < n_steps; ++k) {
    const float2 gauss = curand_normal2(&rng);
    const float variance_plus = fmaxf(variance, 0.0f);
    const float vol = sqrtf(variance_plus);
    log_return += (mu - 0.5f * variance_plus) * dt + vol * sqrt_dt * gauss.x;
    variance += kappa * (theta - variance_plus) * dt +
                xi * vol * sqrt_dt * (rho * gauss.x + rho_complement * gauss.y);
    const int64_t row = static_cast<int64_t>(k + 1) * n_paths + tid;
    spot[row] = s0 * expf(log_return);
    variance_out[row] = fmaxf(variance, 0.0f);
  }
}

int64_t blocks_for(const int64_t n_paths) {
  return (n_paths + kThreads - 1) / kThreads;
}

void check_arguments(const int64_t n_paths, const int64_t n_steps,
                     const int64_t stream) {
  TORCH_CHECK(n_paths > 0, "n_paths must be positive");
  TORCH_CHECK(n_paths < (1LL << 32), "n_paths must be below 2^32");
  TORCH_CHECK(n_steps > 0, "n_steps must be positive");
  TORCH_CHECK(n_steps < (1LL << 31), "n_steps must be below 2^31");
  TORCH_CHECK(stream >= 0 && stream < (1LL << 32),
              "stream must lie in [0, 2^32)");
}

}  // namespace

torch::Tensor gbm_paths(const int64_t n_paths, const int64_t n_steps,
                        const double s0, const double mu, const double sigma,
                        const double maturity, const int64_t seed,
                        const int64_t stream) {
  check_arguments(n_paths, n_steps, stream);
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto spot = torch::empty({n_steps + 1, n_paths}, options);
  const double dt = maturity / static_cast<double>(n_steps);
  const float drift = static_cast<float>((mu - 0.5 * sigma * sigma) * dt);
  const float diffusion = static_cast<float>(sigma * std::sqrt(dt));
  const uint64_t subsequence_base = static_cast<uint64_t>(stream) << 32;
  gbm_kernel<<<blocks_for(n_paths), kThreads, 0,
               at::cuda::getCurrentCUDAStream()>>>(
      spot.data_ptr<float>(), n_paths, static_cast<int>(n_steps),
      static_cast<float>(s0), drift, diffusion, static_cast<uint64_t>(seed),
      subsequence_base);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return spot;
}

std::vector<torch::Tensor> heston_paths(
    const int64_t n_paths, const int64_t n_steps, const double s0,
    const double v0, const double kappa, const double theta, const double xi,
    const double rho, const double mu, const double maturity,
    const int64_t seed, const int64_t stream) {
  check_arguments(n_paths, n_steps, stream);
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto spot = torch::empty({n_steps + 1, n_paths}, options);
  auto variance = torch::empty({n_steps + 1, n_paths}, options);
  const double dt = maturity / static_cast<double>(n_steps);
  const uint64_t subsequence_base = static_cast<uint64_t>(stream) << 32;
  heston_kernel<<<blocks_for(n_paths), kThreads, 0,
                  at::cuda::getCurrentCUDAStream()>>>(
      spot.data_ptr<float>(), variance.data_ptr<float>(), n_paths,
      static_cast<int>(n_steps), static_cast<float>(s0),
      static_cast<float>(v0), static_cast<float>(dt),
      static_cast<float>(std::sqrt(dt)), static_cast<float>(kappa),
      static_cast<float>(theta), static_cast<float>(xi),
      static_cast<float>(rho), static_cast<float>(mu),
      static_cast<uint64_t>(seed), subsequence_base);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {spot, variance};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gbm_paths", &gbm_paths, "Fused Philox GBM path generation");
  m.def("heston_paths", &heston_paths,
        "Fused Philox Heston path generation");
}
