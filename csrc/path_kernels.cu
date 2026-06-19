// Fused Philox path generators.
//
// One thread owns one path; the log-price (and variance) recursion lives
// entirely in registers, so a path costs one kernel launch regardless of
// step count, against one launch per step per tensor op in the eager
// simulators. The (seed, stream) pair of a NoiseSpec maps onto the Philox
// (seed, subsequence) pair as subsequence = (stream << 32) | path, which
// keeps streams disjoint for any path count below 2^32 and makes every
// draw addressable for a backward pass that regenerates noise instead of
// storing it. The offset variants read the shifted subsequence from device
// memory rather than a launch argument, so a captured graph can draw a
// fresh batch on every replay by updating one element in place. Output is
// time-major to match the MarketState contract.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <curand_kernel.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxJumps = 16;

// The truncated Poisson distribution function travels by value so the
// offset kernels carry no device buffer that a captured graph would have
// to keep alive across replays.
struct JumpCdf {
  float v[kMaxJumps + 1];
};

__device__ void gbm_path(
    float* __restrict__ spot,
    const int64_t n_paths,
    const int n_steps,
    const float s0,
    const float drift,
    const float diffusion,
    const uint64_t seed,
    const uint64_t subsequence_base,
    const int64_t tid) {
  curandStatePhilox4_32_10_t rng;
  curand_init(seed, subsequence_base + static_cast<uint64_t>(tid), 0ULL, &rng);
  float log_return = 0.0f;
  spot[tid] = s0;
  for (int k = 0; k < n_steps; ++k) {
    log_return += drift + diffusion * curand_normal(&rng);
    spot[static_cast<int64_t>(k + 1) * n_paths + tid] = s0 * expf(log_return);
  }
}

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
  gbm_path(spot, n_paths, n_steps, s0, drift, diffusion, seed,
           subsequence_base, tid);
}

__global__ void gbm_offset_kernel(
    float* __restrict__ spot,
    const int64_t n_paths,
    const int n_steps,
    const float s0,
    const float drift,
    const float diffusion,
    const uint64_t seed,
    const int64_t* __restrict__ subsequence_base) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= n_paths) {
    return;
  }
  gbm_path(spot, n_paths, n_steps, s0, drift, diffusion, seed,
           static_cast<uint64_t>(*subsequence_base), tid);
}

__device__ void heston_path(
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
    const uint64_t subsequence_base,
    const int64_t tid) {
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
  heston_path(spot, variance_out, n_paths, n_steps, s0, v0, dt, sqrt_dt, kappa,
              theta, xi, rho, mu, seed, subsequence_base, tid);
}

__global__ void heston_offset_kernel(
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
    const int64_t* __restrict__ subsequence_base) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= n_paths) {
    return;
  }
  heston_path(spot, variance_out, n_paths, n_steps, s0, v0, dt, sqrt_dt, kappa,
              theta, xi, rho, mu, seed,
              static_cast<uint64_t>(*subsequence_base), tid);
}

__global__ void gbm_fold_kernel(
    float* __restrict__ terminal,
    float* __restrict__ running_max,
    float* __restrict__ running_min,
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
  float spot = s0;
  float spot_max = s0;
  float spot_min = s0;
  for (int k = 0; k < n_steps; ++k) {
    log_return += drift + diffusion * curand_normal(&rng);
    spot = s0 * expf(log_return);
    spot_max = fmaxf(spot_max, spot);
    spot_min = fminf(spot_min, spot);
  }
  terminal[tid] = spot;
  running_max[tid] = spot_max;
  running_min[tid] = spot_min;
}

__global__ void heston_fold_kernel(
    float* __restrict__ terminal,
    float* __restrict__ running_max,
    float* __restrict__ running_min,
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
  float spot = s0;
  float spot_max = s0;
  float spot_min = s0;
  for (int k = 0; k < n_steps; ++k) {
    const float2 gauss = curand_normal2(&rng);
    const float variance_plus = fmaxf(variance, 0.0f);
    const float vol = sqrtf(variance_plus);
    log_return += (mu - 0.5f * variance_plus) * dt + vol * sqrt_dt * gauss.x;
    variance += kappa * (theta - variance_plus) * dt +
                xi * vol * sqrt_dt * (rho * gauss.x + rho_complement * gauss.y);
    spot = s0 * expf(log_return);
    spot_max = fmaxf(spot_max, spot);
    spot_min = fminf(spot_min, spot);
  }
  terminal[tid] = spot;
  running_max[tid] = spot_max;
  running_min[tid] = spot_min;
}

__device__ void merton_path(
    float* __restrict__ spot,
    float* __restrict__ jumps_out,
    const int64_t n_paths,
    const int n_steps,
    const float s0,
    const float drift,
    const float diffusion,
    const float jump_mean,
    const float jump_vol,
    const JumpCdf cdf,
    const uint64_t seed,
    const uint64_t subsequence_base,
    const int64_t tid) {
  curandStatePhilox4_32_10_t rng;
  curand_init(seed, subsequence_base + static_cast<uint64_t>(tid), 0ULL, &rng);
  float log_return = 0.0f;
  float cumulative_jumps = 0.0f;
  spot[tid] = s0;
  jumps_out[tid] = 0.0f;
  for (int k = 0; k < n_steps; ++k) {
    const float z_diffusion = curand_normal(&rng);
    const float uniform = curand_uniform(&rng);
    const float z_jump = curand_normal(&rng);
    int count = 0;
    while (count < kMaxJumps && cdf.v[count] < uniform) {
      ++count;
    }
    const float jumps = static_cast<float>(count);
    const float jump_sum = jump_mean * jumps + jump_vol * sqrtf(jumps) * z_jump;
    log_return += drift + diffusion * z_diffusion + jump_sum;
    cumulative_jumps += jumps;
    const int64_t row = static_cast<int64_t>(k + 1) * n_paths + tid;
    spot[row] = s0 * expf(log_return);
    jumps_out[row] = cumulative_jumps;
  }
}

__global__ void merton_kernel(
    float* __restrict__ spot,
    float* __restrict__ jumps_out,
    const int64_t n_paths,
    const int n_steps,
    const float s0,
    const float drift,
    const float diffusion,
    const float jump_mean,
    const float jump_vol,
    const JumpCdf cdf,
    const uint64_t seed,
    const uint64_t subsequence_base) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= n_paths) {
    return;
  }
  merton_path(spot, jumps_out, n_paths, n_steps, s0, drift, diffusion,
              jump_mean, jump_vol, cdf, seed, subsequence_base, tid);
}

__global__ void merton_offset_kernel(
    float* __restrict__ spot,
    float* __restrict__ jumps_out,
    const int64_t n_paths,
    const int n_steps,
    const float s0,
    const float drift,
    const float diffusion,
    const float jump_mean,
    const float jump_vol,
    const JumpCdf cdf,
    const uint64_t seed,
    const int64_t* __restrict__ subsequence_base) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= n_paths) {
    return;
  }
  merton_path(spot, jumps_out, n_paths, n_steps, s0, drift, diffusion,
              jump_mean, jump_vol, cdf, seed,
              static_cast<uint64_t>(*subsequence_base), tid);
}

__global__ void merton_fold_kernel(
    float* __restrict__ terminal,
    float* __restrict__ running_max,
    float* __restrict__ running_min,
    const int64_t n_paths,
    const int n_steps,
    const float s0,
    const float drift,
    const float diffusion,
    const float jump_mean,
    const float jump_vol,
    const JumpCdf cdf,
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
  float spot = s0;
  float spot_max = s0;
  float spot_min = s0;
  for (int k = 0; k < n_steps; ++k) {
    const float z_diffusion = curand_normal(&rng);
    const float uniform = curand_uniform(&rng);
    const float z_jump = curand_normal(&rng);
    int count = 0;
    while (count < kMaxJumps && cdf.v[count] < uniform) {
      ++count;
    }
    const float jumps = static_cast<float>(count);
    const float jump_sum = jump_mean * jumps + jump_vol * sqrtf(jumps) * z_jump;
    log_return += drift + diffusion * z_diffusion + jump_sum;
    spot = s0 * expf(log_return);
    spot_max = fmaxf(spot_max, spot);
    spot_min = fminf(spot_min, spot);
  }
  terminal[tid] = spot;
  running_max[tid] = spot_max;
  running_min[tid] = spot_min;
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

// The offset VALUE is trusted. It lives in device memory, so validating it
// host-side would force a synchronisation that capture forbids. Callers
// write a non-negative shifted stream (stream << 32); a negative value
// would wrap through the unsigned cast into a foreign Philox subsequence.
void check_offset(const torch::Tensor& offset) {
  TORCH_CHECK(offset.is_cuda(), "offset must live on the CUDA device");
  TORCH_CHECK(offset.scalar_type() == torch::kInt64, "offset must be int64");
  TORCH_CHECK(offset.numel() == 1, "offset must hold one element");
}

torch::Tensor gbm_paths_offset(const int64_t n_paths, const int64_t n_steps,
                               const double s0, const double mu,
                               const double sigma, const double maturity,
                               const int64_t seed,
                               const torch::Tensor& offset) {
  check_arguments(n_paths, n_steps, 0);
  check_offset(offset);
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto spot = torch::empty({n_steps + 1, n_paths}, options);
  const double dt = maturity / static_cast<double>(n_steps);
  const float drift = static_cast<float>((mu - 0.5 * sigma * sigma) * dt);
  const float diffusion = static_cast<float>(sigma * std::sqrt(dt));
  gbm_offset_kernel<<<blocks_for(n_paths), kThreads, 0,
                      at::cuda::getCurrentCUDAStream()>>>(
      spot.data_ptr<float>(), n_paths, static_cast<int>(n_steps),
      static_cast<float>(s0), drift, diffusion, static_cast<uint64_t>(seed),
      offset.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return spot;
}

std::vector<torch::Tensor> heston_paths_offset(
    const int64_t n_paths, const int64_t n_steps, const double s0,
    const double v0, const double kappa, const double theta, const double xi,
    const double rho, const double mu, const double maturity,
    const int64_t seed, const torch::Tensor& offset) {
  check_arguments(n_paths, n_steps, 0);
  check_offset(offset);
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto spot = torch::empty({n_steps + 1, n_paths}, options);
  auto variance = torch::empty({n_steps + 1, n_paths}, options);
  const double dt = maturity / static_cast<double>(n_steps);
  heston_offset_kernel<<<blocks_for(n_paths), kThreads, 0,
                         at::cuda::getCurrentCUDAStream()>>>(
      spot.data_ptr<float>(), variance.data_ptr<float>(), n_paths,
      static_cast<int>(n_steps), static_cast<float>(s0),
      static_cast<float>(v0), static_cast<float>(dt),
      static_cast<float>(std::sqrt(dt)), static_cast<float>(kappa),
      static_cast<float>(theta), static_cast<float>(xi),
      static_cast<float>(rho), static_cast<float>(mu),
      static_cast<uint64_t>(seed), offset.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {spot, variance};
}

std::vector<torch::Tensor> gbm_folds(const int64_t n_paths,
                                     const int64_t n_steps, const double s0,
                                     const double mu, const double sigma,
                                     const double maturity, const int64_t seed,
                                     const int64_t stream) {
  check_arguments(n_paths, n_steps, stream);
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto terminal = torch::empty({n_paths}, options);
  auto running_max = torch::empty({n_paths}, options);
  auto running_min = torch::empty({n_paths}, options);
  const double dt = maturity / static_cast<double>(n_steps);
  const float drift = static_cast<float>((mu - 0.5 * sigma * sigma) * dt);
  const float diffusion = static_cast<float>(sigma * std::sqrt(dt));
  const uint64_t subsequence_base = static_cast<uint64_t>(stream) << 32;
  gbm_fold_kernel<<<blocks_for(n_paths), kThreads, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
      terminal.data_ptr<float>(), running_max.data_ptr<float>(),
      running_min.data_ptr<float>(), n_paths, static_cast<int>(n_steps),
      static_cast<float>(s0), drift, diffusion, static_cast<uint64_t>(seed),
      subsequence_base);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {terminal, running_max, running_min};
}

std::vector<torch::Tensor> heston_folds(
    const int64_t n_paths, const int64_t n_steps, const double s0,
    const double v0, const double kappa, const double theta, const double xi,
    const double rho, const double mu, const double maturity,
    const int64_t seed, const int64_t stream) {
  check_arguments(n_paths, n_steps, stream);
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto terminal = torch::empty({n_paths}, options);
  auto running_max = torch::empty({n_paths}, options);
  auto running_min = torch::empty({n_paths}, options);
  const double dt = maturity / static_cast<double>(n_steps);
  const uint64_t subsequence_base = static_cast<uint64_t>(stream) << 32;
  heston_fold_kernel<<<blocks_for(n_paths), kThreads, 0,
                       at::cuda::getCurrentCUDAStream()>>>(
      terminal.data_ptr<float>(), running_max.data_ptr<float>(),
      running_min.data_ptr<float>(), n_paths, static_cast<int>(n_steps),
      static_cast<float>(s0), static_cast<float>(v0), static_cast<float>(dt),
      static_cast<float>(std::sqrt(dt)), static_cast<float>(kappa),
      static_cast<float>(theta), static_cast<float>(xi),
      static_cast<float>(rho), static_cast<float>(mu),
      static_cast<uint64_t>(seed), subsequence_base);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {terminal, running_max, running_min};
}

JumpCdf make_jump_cdf(const double rate) {
  JumpCdf cdf;
  double pmf = std::exp(-rate);
  double cumulative = pmf;
  cdf.v[0] = static_cast<float>(cumulative);
  for (int k = 1; k <= kMaxJumps; ++k) {
    pmf *= rate / static_cast<double>(k);
    cumulative += pmf;
    cdf.v[k] = static_cast<float>(cumulative);
  }
  return cdf;
}

std::vector<torch::Tensor> merton_paths(
    const int64_t n_paths, const int64_t n_steps, const double s0,
    const double sigma, const double jump_intensity, const double jump_mean,
    const double jump_vol, const double mu, const double maturity,
    const int64_t seed, const int64_t stream) {
  check_arguments(n_paths, n_steps, stream);
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto spot = torch::empty({n_steps + 1, n_paths}, options);
  auto jumps = torch::empty({n_steps + 1, n_paths}, options);
  const double dt = maturity / static_cast<double>(n_steps);
  const double mean_jump = std::exp(jump_mean + 0.5 * jump_vol * jump_vol) - 1.0;
  const double compensator = jump_intensity * mean_jump;
  const float drift =
      static_cast<float>((mu - compensator - 0.5 * sigma * sigma) * dt);
  const float diffusion = static_cast<float>(sigma * std::sqrt(dt));
  const JumpCdf cdf = make_jump_cdf(jump_intensity * dt);
  const uint64_t subsequence_base = static_cast<uint64_t>(stream) << 32;
  merton_kernel<<<blocks_for(n_paths), kThreads, 0,
                  at::cuda::getCurrentCUDAStream()>>>(
      spot.data_ptr<float>(), jumps.data_ptr<float>(), n_paths,
      static_cast<int>(n_steps), static_cast<float>(s0), drift, diffusion,
      static_cast<float>(jump_mean), static_cast<float>(jump_vol), cdf,
      static_cast<uint64_t>(seed), subsequence_base);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {spot, jumps};
}

std::vector<torch::Tensor> merton_paths_offset(
    const int64_t n_paths, const int64_t n_steps, const double s0,
    const double sigma, const double jump_intensity, const double jump_mean,
    const double jump_vol, const double mu, const double maturity,
    const int64_t seed, const torch::Tensor& offset) {
  check_arguments(n_paths, n_steps, 0);
  check_offset(offset);
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto spot = torch::empty({n_steps + 1, n_paths}, options);
  auto jumps = torch::empty({n_steps + 1, n_paths}, options);
  const double dt = maturity / static_cast<double>(n_steps);
  const double mean_jump = std::exp(jump_mean + 0.5 * jump_vol * jump_vol) - 1.0;
  const double compensator = jump_intensity * mean_jump;
  const float drift =
      static_cast<float>((mu - compensator - 0.5 * sigma * sigma) * dt);
  const float diffusion = static_cast<float>(sigma * std::sqrt(dt));
  const JumpCdf cdf = make_jump_cdf(jump_intensity * dt);
  merton_offset_kernel<<<blocks_for(n_paths), kThreads, 0,
                         at::cuda::getCurrentCUDAStream()>>>(
      spot.data_ptr<float>(), jumps.data_ptr<float>(), n_paths,
      static_cast<int>(n_steps), static_cast<float>(s0), drift, diffusion,
      static_cast<float>(jump_mean), static_cast<float>(jump_vol), cdf,
      static_cast<uint64_t>(seed), offset.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {spot, jumps};
}

std::vector<torch::Tensor> merton_folds(
    const int64_t n_paths, const int64_t n_steps, const double s0,
    const double sigma, const double jump_intensity, const double jump_mean,
    const double jump_vol, const double mu, const double maturity,
    const int64_t seed, const int64_t stream) {
  check_arguments(n_paths, n_steps, stream);
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto terminal = torch::empty({n_paths}, options);
  auto running_max = torch::empty({n_paths}, options);
  auto running_min = torch::empty({n_paths}, options);
  const double dt = maturity / static_cast<double>(n_steps);
  const double mean_jump = std::exp(jump_mean + 0.5 * jump_vol * jump_vol) - 1.0;
  const double compensator = jump_intensity * mean_jump;
  const float drift =
      static_cast<float>((mu - compensator - 0.5 * sigma * sigma) * dt);
  const float diffusion = static_cast<float>(sigma * std::sqrt(dt));
  const JumpCdf cdf = make_jump_cdf(jump_intensity * dt);
  const uint64_t subsequence_base = static_cast<uint64_t>(stream) << 32;
  merton_fold_kernel<<<blocks_for(n_paths), kThreads, 0,
                       at::cuda::getCurrentCUDAStream()>>>(
      terminal.data_ptr<float>(), running_max.data_ptr<float>(),
      running_min.data_ptr<float>(), n_paths, static_cast<int>(n_steps),
      static_cast<float>(s0), drift, diffusion, static_cast<float>(jump_mean),
      static_cast<float>(jump_vol), cdf, static_cast<uint64_t>(seed),
      subsequence_base);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {terminal, running_max, running_min};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gbm_paths", &gbm_paths, "Fused Philox GBM path generation");
  m.def("gbm_paths_offset", &gbm_paths_offset,
        "Fused Philox GBM paths reading the subsequence from device memory");
  m.def("heston_paths", &heston_paths,
        "Fused Philox Heston path generation");
  m.def("heston_paths_offset", &heston_paths_offset,
        "Fused Philox Heston paths reading the subsequence from device memory");
  m.def("gbm_folds", &gbm_folds,
        "Fused Philox GBM path-statistic folds without the grid");
  m.def("heston_folds", &heston_folds,
        "Fused Philox Heston path-statistic folds without the grid");
  m.def("merton_paths", &merton_paths,
        "Fused Philox Merton jump-diffusion path generation");
  m.def("merton_paths_offset", &merton_paths_offset,
        "Fused Philox Merton paths reading the subsequence from device memory");
  m.def("merton_folds", &merton_folds,
        "Fused Philox Merton path-statistic folds without the grid");
}
