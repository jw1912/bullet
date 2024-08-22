#define _CRT_SECURE_NO_WARNINGS

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>

namespace {
enum class QuantiseMode {
  Truncate,
  Round,
};

constexpr auto InFile =
    "../checkpoints/bullet_r36_768x8-1024x2-1x8-400/params.bin";
constexpr auto OutFile = "bullet_r36_768x8-1024x2-1x8-400.nn";

constexpr bool Factorised = true;
constexpr bool TransposeOutputWeights = true;

constexpr std::uint32_t InputSize = 768;
constexpr std::uint32_t InputBuckets = 8;
constexpr std::uint32_t L1 = 1024;
constexpr std::uint32_t OutputBuckets = 8;

constexpr float Clip = 1.98F;

constexpr std::uint32_t L1Q = 255;
constexpr std::uint32_t OutputQ = 64;

constexpr QuantiseMode Mode = QuantiseMode::Round;

constexpr std::size_t PaddingBlockSize = 64;

// ========================================================================

namespace internal {
template <typename T, std::size_t N, std::size_t... Ns> struct MultiArrayImpl {
  using Type = std::array<typename MultiArrayImpl<T, Ns...>::Type, N>;
};

template <typename T, std::size_t N> struct MultiArrayImpl<T, N> {
  using Type = std::array<T, N>;
};
} // namespace internal

template <typename T, std::size_t... Ns>
using MultiArray = typename internal::MultiArrayImpl<T, Ns...>::Type;

template <typename Param, std::uint32_t InputBuckets> struct Network {
  MultiArray<Param, InputBuckets, InputSize * L1> ftWeights;
  std::array<Param, L1> ftBiases;
  std::array<Param, 2 * L1 * OutputBuckets> l1Weights;
  std::array<Param, OutputBuckets> l1Biases;
};

using RawNetwork = Network<float, InputBuckets + Factorised>;
using QuantisedNetwork = Network<std::int16_t, InputBuckets>;

using RawNetworkUnfactorised = Network<float, InputBuckets>;

template <std::uint32_t Q> [[nodiscard]] inline auto quantise(float v) {
  v = std::clamp(v, -Clip, Clip);
  v *= static_cast<float>(Q);

  if constexpr (Mode == QuantiseMode::Round)
    v = std::round(v);

  assert(std::abs(v) <=
         static_cast<float>(std::numeric_limits<std::int16_t>::max()));

  return static_cast<std::int16_t>(v);
}

template <std::size_t Block> [[nodiscard]] inline auto pad(std::size_t v) {
  return ((v + Block - 1) / Block) * Block;
}
} // namespace

auto main() -> int {
  auto raw = std::make_unique<RawNetwork>();

  {
    std::ifstream in{InFile, std::ios::binary};

    if (!in) {
      std::cerr << "failed to open source network" << std::endl;
      std::cerr << std::strerror(errno) << std::endl;
      return 1;
    }

    if (!in.read(reinterpret_cast<char *>(raw.get()), sizeof(RawNetwork))) {
      std::cerr << "failed to load source network" << std::endl;

      if (in.eof()) {
        std::cerr << "Source network too small";
        if (Factorised && in.gcount() >= sizeof(RawNetworkUnfactorised))
          std::cerr << " - unfactorised network?";
        std::cerr << std::endl;
      } else
        std::cerr << std::strerror(errno) << std::endl;

      return 1;
    }
  }

  auto quantised = std::make_unique<QuantisedNetwork>();

  for (std::uint32_t bucket = 0; bucket < InputBuckets; ++bucket) {
    for (std::uint32_t weight = 0; weight < InputSize * L1; ++weight) {
      auto param = raw->ftWeights[bucket + Factorised][weight];

      if constexpr (Factorised)
        param += raw->ftWeights[0][weight];

      quantised->ftWeights[bucket][weight] = quantise<L1Q>(param);
    }
  }

  for (std::uint32_t bias = 0; bias < L1; ++bias) {
    quantised->ftBiases[bias] = quantise<L1Q>(raw->ftBiases[bias]);
  }

  if constexpr (TransposeOutputWeights) {
    for (std::uint32_t weight = 0; weight < 2 * L1; ++weight) {
      for (std::uint32_t bucket = 0; bucket < OutputBuckets; ++bucket) {
        const auto src = weight * OutputBuckets + bucket;
        const auto dst = bucket * 2 * L1 + weight;

        quantised->l1Weights[dst] = quantise<OutputQ>(raw->l1Weights[src]);
      }
    }
  } else {
    for (std::uint32_t weight = 0; weight < 2 * L1 * OutputBuckets; ++weight) {
      quantised->l1Weights[weight] = quantise<OutputQ>(raw->l1Weights[weight]);
    }
  }

  for (std::uint32_t bias = 0; bias < OutputBuckets; ++bias) {
    quantised->l1Biases[bias] = quantise<L1Q * OutputQ>(raw->l1Biases[bias]);
  }

  {
    std::ofstream out{OutFile, std::ios::binary};

    if (!out.write(reinterpret_cast<const char *>(quantised.get()),
                   sizeof(QuantisedNetwork))) {
      std::cerr << "failed to write transposed network" << std::endl;
      std::cerr << std::strerror(errno) << std::endl;
      return 1;
    }

    if constexpr (PaddingBlockSize > 1) {
      if (const auto padding = pad<PaddingBlockSize>(sizeof(QuantisedNetwork)) -
                               sizeof(QuantisedNetwork);
          padding != 0) {
        static const std::array<std::byte, PaddingBlockSize> empty{};

        if (!out.write(reinterpret_cast<const char *>(empty.data()),
                       static_cast<std::streamsize>(padding))) {
          std::cerr << "failed to write padding" << std::endl;
          std::cerr << std::strerror(errno) << std::endl;
          return 1;
        }
      }
    }
  }

  return 0;
}