// Frozen configuration-mean fits in SYS_CNT ticks, with static tree residuals.
#include "AscendModel/RouteModel/Models/PartialStructuredMemoryModel.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <string_view>

using namespace mlir::ascend::partial_memory;

namespace {
using Segment = std::pair<double, double>;
using Segments = std::vector<Segment>;
double lg(double x) { return std::log2(std::max(x, 1.0)); }
double mod(double x, double y) { return x - y * std::floor(x / y); }
void appendSegment(Segments &s, const Axis &a) {
  if (!s.empty() && a.pitch == s.front().first * s.front().second)
    s.front().first *= a.extent;
  else
    s.insert(s.begin(), {double(a.extent), double(a.pitch)});
}
double tileElements(const Segments &s) {
  double r = 1;
  for (size_t i = s.size() > 3 ? s.size() - 3 : 0; i < s.size(); ++i)
    r *= s[i].first;
  return r;
}
struct Node {
  double thresholdOrValue;
  uint32_t left, right;
  int16_t feature; // -1 denotes a leaf.
};
struct Base {
  const char *group;
  std::array<double, 4> coefficients;
};
struct Frozen {
  const char *const *names;
  size_t featureCount;
  const Node *nodes;
  const uint32_t *roots;
  size_t treeCount;
  const Base *base;
  size_t baseCount;
  double rate, intercept;
  bool strict; // Histogram searchsorted(right) splits use <, HGB uses <=.
};
#include "PartialStructuredMemoryModels.inc"
} // namespace

Features mlir::ascend::partial_memory::extractFeatures(const Geometry &g) {
  Features f;
  const auto &a = g.axes;
  const size_t rank = a.size();
  if (!rank)
    return f;
  const double b = g.bits / 8.0;
  double E = 1, dc = 0, dp = 1, sp = 1, de = 0, ss = 1;
  double minDP = 1e300, maxDP = 0, maxSS = 1;
  int first = -1, last = -1, mask = 0;
  for (size_t i = 0; i < rank; ++i) {
    E *= a[i].extent;
    if (a[i].discrete) {
      dc++;
      dp *= a[i].extent;
      de += a[i].extent;
      minDP = std::min(minDP, double(a[i].pitch));
      maxDP = std::max(maxDP, double(a[i].pitch));
      if (first < 0)
        first = i;
      last = i;
      mask |= 1 << i;
    } else {
      sp *= a[i].extent;
      ss += (a[i].extent - 1) * double(a[i].pitch);
      maxSS = std::max(maxSS, double(a[i].pitch));
    }
  }
  if (first < 0)
    return f;
  const auto &tail = a.back();
  double suffix = 1, C = 1;
  Segments regular, all;
  for (size_t i = rank; i-- > 0;) {
    if (a[i].discrete)
      break;
    suffix *= a[i].extent;
    appendSegment(regular, a[i]);
  }
  for (size_t i = rank; i-- > 0;) {
    if (a[i].discrete || a[i].pitch != C)
      break;
    C *= a[i].extent;
  }
  for (size_t i = rank; i-- > 0;)
    if (!a[i].discrete)
      appendSegment(all, a[i]);
  double step = tail.discrete ? 0 : tail.pitch;
  f = {{"one", 1},
       {"E", E},
       {"bytes", E * b},
       {"bits", double(g.bits)},
       {"rank", double(rank)},
       {"W", double(g.warps)},
       {"D_count", dc},
       {"S_count", rank - dc},
       {"D_elements", de},
       {"D_product", dp},
       {"S_product", sp},
       {"C", C},
       {"suffix", suffix},
       {"rows", E / tail.extent},
       {"tail_n", double(tail.extent)},
       {"tail_bytes", tail.extent * b},
       {"tail_step", step},
       {"tail_step_bytes", step * b},
       {"tail_D", double(tail.discrete)},
       {"tail_unit", double(step == 1)},
       {"S_span", ss},
       {"min_D_pitch", minDP},
       {"max_D_pitch", maxDP},
       {"max_S_step", maxSS},
       {"chunks", E / C},
       {"suffix_rows", E / suffix},
       {"aligned_row_bytes", std::ceil(tail.extent * b / 32) * 32},
       {"format_float", double(g.floatingPoint)}};
  // Base logs are taken before adding later derived features, as in the fit.
  Features initial = f;
  for (const auto &v : initial)
    if (v.first != "one")
      f["log_" + v.first] = lg(v.second);
  for (size_t i = 0; i < 5; ++i) {
    bool present = i < rank;
    int64_t p = present ? a[i].pitch : 1;
    auto k = std::to_string(i);
    f["axis" + k] = present ? lg(a[i].extent) : 0;
    f["pitch" + k] = lg(p);
    f["D" + k] = present && a[i].discrete;
    f["v2pitch" + k] = lg(p & -p);
    f["tail_axis" + k] = present ? lg(a[rank - 1 - i].extent) : 0;
    f["tail_pitch" + k] = present ? lg(a[rank - 1 - i].pitch * b) : 0;
    f["tail_random" + k] = present && a[rank - 1 - i].discrete;
  }
  double suffixSpan = 1, prefix = 1;
  int64_t gcd = 0;
  for (size_t i = 0; i < rank; ++i) {
    gcd = std::gcd(gcd, a[i].pitch * (g.bits / 8));
    if (int(i) > last)
      suffixSpan += (a[i].extent - 1) * double(a[i].pitch);
    else
      prefix *= a[i].extent;
  }
  f["first_D"] = first;
  f["last_D"] = last;
  f["D_mask"] = mask;
  f["suffix_rank"] = rank - last - 1;
  f["prefix_S_rank"] = first;
  f["tail_row_blocks32"] = std::ceil(tail.extent * b / 32);
  f["tail_row_blocks128"] = std::ceil(tail.extent * b / 128);
  f["tail_span_bytes"] = tail.discrete ? b : ((tail.extent - 1) * step + 1) * b;
  f["suffix_span_bytes"] = suffixSpan * b;
  f["address_alignment_gcd"] = gcd;
  f["suffix_span_per_element"] = suffixSpan / suffix;
  f["tail_stride_mod32"] = mod(tail.pitch * b, 32);
  f["tail_stride_mod128"] = mod(tail.pitch * b, 128);
  f["tail_stride_gcd128"] = std::gcd(tail.pitch * (g.bits / 8), int64_t(128));
  f["aligned_tail32"] = mod(tail.extent * b, 32) == 0;
  f["aligned_tail128"] = mod(tail.extent * b, 128) == 0;
  f["prefix_elements"] = prefix;
  for (auto k :
       {"tail_row_blocks32", "tail_row_blocks128", "tail_span_bytes",
        "suffix_span_bytes", "address_alignment_gcd", "suffix_span_per_element",
        "tail_stride_gcd128", "prefix_elements"})
    f[std::string("log_") + k] = lg(f[k]);
  double tile = tileElements(regular);
  double inv = step > 1 ? 1 / (step * b) : 0;
  f["sqrt_E"] = std::sqrt(E);
  f["sqrt_W"] = std::sqrt(g.warps);
  f["inverse_W"] = 1.0 / g.warps;
  f["regular_rank"] = regular.size();
  f["regular_groups"] = E / suffix;
  f["tile_elements"] = tile;
  f["tile_large"] = tile * b >= 32;
  f["static_work"] = tile * b >= 32 ? E / tile : E;
  f["suffix_work"] = suffix * b >= 32 ? E / suffix : E;
  f["stride_work"] = E * inv;
  f["stride_work2"] = E * inv * inv;
  f["clipped_stride_work"] = E * std::max(inv - 1.0 / 128, 0.0);
  f["suffix_bytes"] = suffix * b;
  f["C_bytes"] = C * b;
  f["log_suffix_bytes"] = std::log2(suffix * b);
  f["log_C_bytes"] = std::log2(C * b);
  f["suffix32"] = suffix * b >= 32;
  f["C32"] = C * b >= 32;
  double allTile = tileElements(all);
  f["all_S_bytes"] = sp * b;
  f["all_S32"] = sp * b >= 32;
  f["all_S_work"] = sp * b >= 32 ? E / sp : E;
  f["all_regular_rank"] = all.size();
  f["all_tile_elements"] = allTile;
  f["all_tile32"] = allTile * b >= 32;
  f["all_tile_work"] = allTile * b >= 32 ? E / allTile : E;
  for (auto k : {"S_product", "all_S_bytes", "all_S_work", "all_tile_elements",
                 "all_tile_work"})
    f[std::string("log_") + k] = lg(f[k]);
  for (size_t i = 0; i < 5; ++i) {
    double st = i < rank && !a[i].discrete ? a[i].pitch : 0;
    double ub = i < rank ? 1 : 0;
    for (size_t j = i + 1; j < rank; ++j)
      ub *= a[j].extent;
    double ns = i + 1 < rank && !a[i + 1].discrete ? a[i + 1].pitch : 0;
    double den = ns ? ns * a[i + 1].extent : 0;
    auto k = std::to_string(i);
    f["step_mod32_" + k] = mod(st * b, 32);
    f["step_mod128_" + k] = mod(st * b, 128);
    f["dense_pitch_mod32_" + k] = mod(ub * b, 32);
    f["axis_bytes32_" + k] = i < rank && a[i].extent * b >= 32;
    f["pair_merge_" + k] = st > 0 && den > 0 && st == den;
    f["pair_ratio_" + k] = st && den ? std::log2(std::max(st / den, 1e-9)) : 0;
    f["pair_gap32_" + k] = st && den ? mod((st - den) * b, 32) : 0;
  }
  Segments t(regular.end() - std::min<size_t>(regular.size(), 3),
             regular.end());
  double minDim = 1;
  int minAxis = -1;
  if (!t.empty()) {
    auto it = std::min_element(t.begin(), t.end(), [](Segment x, Segment y) {
      return x.first < y.first;
    });
    minDim = it->first;
    minAxis = it - t.begin();
  }
  f["tile_min_dim"] = minDim;
  f["tile_min_axis"] = minAxis;
  f["tile_scalar_rows"] = t.empty() ? E : E / t.back().first;
  f["tile_outer_work"] = t.empty() ? E : E / tile * minDim;
  for (size_t j = 0; j < 3; ++j) {
    Segment s = j < t.size() ? t[t.size() - 1 - j] : Segment{1, 0};
    auto k = std::to_string(j);
    f["segment_log_n" + k] = lg(s.first);
    f["segment_log_step" + k] = lg(s.second * b);
    f["segment_mod32_" + k] = mod(s.second * b, 32);
  }
  // Explicit static layout hypotheses; not actual backend UB layouts.
  t.insert(t.begin(), 3 - t.size(), {1, 1});
  for (auto layout : {"dense", "row32"}) {
    std::string p = layout;
    std::array<double, 3> u = {
        0, p == "dense" ? t[2].first : std::ceil(t[2].first * b / 32) * 32 / b,
        1};
    u[0] = t[1].first * u[1];
    std::array<bool, 3> align;
    std::array<bool, 2> merge;
    for (size_t i = 0; i < 3; ++i) {
      align[i] = mod(u[i] * b, 32) == 0;
      f[p + "_aligned" + std::to_string(i)] = align[i];
      f[p + "_log_pitch" + std::to_string(i)] = lg(u[i] * b);
    }
    for (size_t i = 0; i < 2; ++i) {
      merge[i] = u[i] == u[i + 1] * t[i + 1].first &&
                 t[i].second == t[i + 1].second * t[i + 1].first;
      f[p + "_merge" + std::to_string(i)] = merge[i];
    }
    bool legal =
        t[2].second == 1
            ? (t[1].second == t[2].first && u[1] == t[2].first ? align[0]
               : merge[0]                                      ? align[1]
                          : align[0] && align[1])
            : (merge[0] && align[1]) || (merge[1] && align[0] && align[2]);
    f[p + "_legal_proxy"] = legal;
    f[p + "_work_proxy"] = legal ? E / tile : E;
  }
  f["known_tail_unit"] = !regular.empty() && t.back().second == 1;
  for (size_t i = 0; i < 3; ++i) {
    auto k = std::to_string(i);
    f["tile_gm_align" + k] = mod(t[i].second * b, 32) == 0;
    f["tile_gm_stride_limit" + k] = t[i].second * b >= 65536;
    f["tile_dim_block" + k] = std::ceil(t[i].first * b / 32);
    f["tile_dim_ge32" + k] = t[i].first * b >= 32;
    f["tile_gm_stride_over_dim" + k] = t[i].second / t[i].first;
  }
  for (size_t i = 0; i < 2; ++i) {
    auto k = std::to_string(i);
    f["tile_gm_gap" + k] = t[i].second - t[i + 1].first * t[i + 1].second;
    f["tile_gm_monotonic" + k] = t[i].second >= t[i + 1].second;
  }
  return f;
}

std::optional<double> mlir::ascend::partial_memory::predict(const Geometry &g,
                                                            Model m) {
  if (g.axes.empty() || g.axes.size() > 5 || g.warps < 1 || g.warps > 64 ||
      (g.warps & (g.warps - 1)) ||
      (g.bits != 8 && g.bits != 16 && g.bits != 32 && g.bits != 64))
    return std::nullopt;
  int64_t E = 1;
  size_t discrete = 0;
  for (const auto &a : g.axes) {
    if (a.extent < 2 || (a.extent & (a.extent - 1)) || a.extent > 2048 ||
        a.pitch <= 0 || a.pitch > (int64_t(1) << 31) || E > 2048 / a.extent)
      return std::nullopt;
    E *= a.extent;
    discrete += a.discrete;
  }
  if (!discrete || (g.axes.size() > 1 && discrete == g.axes.size()) ||
      E != 32 * g.warps)
    return std::nullopt;
  Features f = extractFeatures(g);
  const Frozen &model = kModels[static_cast<size_t>(m)];
  std::string tail = f.at("tail_D")      ? "D"
                     : f.at("tail_unit") ? "S1"
                                         : "Sstride";
  std::string key = std::to_string(g.bits) + "_" + tail;
  std::array<double, 4> terms = {1, double(E), std::sqrt(E), lg(g.warps)};
  if (m == Model::SimdLoad) {
    key += f.at("tail_bytes") < 32 ? "_short" : "_long";
    terms = {1, double(E), f.at("rows"), 0};
  } else if (m == Model::SimdStore) {
    key = std::to_string(g.bits) + "_" +
          std::to_string(int(f.at("regular_rank"))) + "_" +
          (tail == "Sstride" ? "Ss" : tail) + "_" +
          std::to_string(int(f.at("tile_large")));
    terms = {1, double(E), f.at("static_work"), f.at("clipped_stride_work")};
  }
  const Base *base = nullptr, *fallback = nullptr;
  for (size_t i = 0; i < model.baseCount; ++i) {
    if (key == model.base[i].group)
      base = &model.base[i];
    if (std::string_view(model.base[i].group) == "all")
      fallback = &model.base[i];
  }
  if (!base)
    base = fallback;
  if (!base)
    return std::nullopt;
  double t0 = std::inner_product(terms.begin(), terms.end(),
                                 base->coefficients.begin(), 0.0);
  std::vector<double> values;
  for (size_t i = 0; i < model.featureCount; ++i) {
    auto it = f.find(model.names[i]);
    if (it == f.end() || !std::isfinite(it->second))
      return std::nullopt;
    values.push_back(it->second);
  }
  double residual = model.intercept;
  for (size_t i = 0; i < model.treeCount; ++i) {
    uint32_t n = model.roots[i];
    while (model.nodes[n].feature >= 0) {
      const auto &node = model.nodes[n];
      double x = values[node.feature];
      bool left =
          model.strict ? x < node.thresholdOrValue : x <= node.thresholdOrValue;
      n = left ? node.left : node.right;
    }
    residual += model.rate * model.nodes[n].thresholdOrValue;
  }
  double result = std::max(1.0, t0) * std::exp(residual);
  return std::isfinite(result) ? std::optional<double>(result) : std::nullopt;
}
