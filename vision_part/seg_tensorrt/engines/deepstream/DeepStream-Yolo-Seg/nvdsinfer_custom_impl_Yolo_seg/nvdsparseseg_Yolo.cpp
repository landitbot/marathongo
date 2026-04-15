#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool
NvDsInferParseYoloSeg(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList);

extern "C" bool
NvDsInferParseFusionSeg(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList);

namespace {

constexpr size_t kFusionRegMax = 16;
constexpr size_t kFusionMaskChannels = 32;
constexpr float kFusionIouThreshold = 0.70f;
constexpr float kRobotMaskCoverThreshold = 0.50f;

struct FusionAnchor {
  float x;
  float y;
  float stride;
};

struct FusionDetection {
  unsigned int classId {0};
  float confidence {0.0F};
  float x1 {0.0F};
  float y1 {0.0F};
  float x2 {0.0F};
  float y2 {0.0F};
  std::array<float, kFusionMaskChannels> maskCoefficients {};
  const float* proto {nullptr};
  size_t protoC {0};
  size_t protoH {0};
  size_t protoW {0};
  std::vector<float> mask;
};

std::mutex gFusionLogMutex;
uint64_t gFusionFrameCounter = 0;

float clampFloat(float val, float minVal, float maxVal)
{
  assert(minVal <= maxVal);
  return std::min(maxVal, std::max(minVal, val));
}

float sigmoidFloat(float x)
{
  if (x >= 0.0F) {
    const float z = std::exp(-x);
    return 1.0F / (1.0F + z);
  }

  const float z = std::exp(x);
  return z / (1.0F + z);
}

float safeIoU(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2, float by2)
{
  const float ix1 = std::max(ax1, bx1);
  const float iy1 = std::max(ay1, by1);
  const float ix2 = std::min(ax2, bx2);
  const float iy2 = std::min(ay2, by2);

  const float iw = std::max(0.0F, ix2 - ix1);
  const float ih = std::max(0.0F, iy2 - iy1);
  const float inter = iw * ih;

  const float aArea = std::max(0.0F, ax2 - ax1) * std::max(0.0F, ay2 - ay1);
  const float bArea = std::max(0.0F, bx2 - bx1) * std::max(0.0F, by2 - by1);
  const float denom = aArea + bArea - inter;
  if (denom <= 0.0F) {
    return 0.0F;
  }

  return inter / denom;
}

void addSegProposal(const float* output, size_t channelsSize, uint netW, uint netH, size_t n, NvDsInferInstanceMaskInfo& b)
{
  const size_t maskSize = channelsSize - 6;
  b.mask = new float[maskSize];
  b.mask_width = netW / 4;
  b.mask_height = netH / 4;
  b.mask_size = sizeof(float) * maskSize;
  std::memcpy(b.mask, output + n * channelsSize + 6, sizeof(float) * maskSize);
}

void addBBoxProposal(float x1, float y1, float x2, float y2, uint netW, uint netH, int maxIndex, float maxProb,
    NvDsInferInstanceMaskInfo& b)
{
  x1 = clampFloat(x1, 0.0F, static_cast<float>(netW));
  y1 = clampFloat(y1, 0.0F, static_cast<float>(netH));
  x2 = clampFloat(x2, 0.0F, static_cast<float>(netW));
  y2 = clampFloat(y2, 0.0F, static_cast<float>(netH));

  b.left = x1;
  b.width = clampFloat(x2 - x1, 0.0F, static_cast<float>(netW));
  b.top = y1;
  b.height = clampFloat(y2 - y1, 0.0F, static_cast<float>(netH));

  if (b.width < 1.0F || b.height < 1.0F) {
    return;
  }

  b.detectionConfidence = maxProb;
  b.classId = maxIndex;
}

std::vector<NvDsInferInstanceMaskInfo>
decodeTensorYoloSeg(const float* output, size_t outputSize, size_t channelsSize, uint netW, uint netH,
    const std::vector<float>& preclusterThreshold)
{
  std::vector<NvDsInferInstanceMaskInfo> objects;

  for (size_t n = 0; n < outputSize; ++n) {
    const float maxProb = output[n * channelsSize + 4];
    const int maxIndex = static_cast<int>(output[n * channelsSize + 5]);

    if (maxIndex < 0 || static_cast<size_t>(maxIndex) >= preclusterThreshold.size()) {
      continue;
    }

    if (maxProb < preclusterThreshold[maxIndex]) {
      continue;
    }

    const float x1 = output[n * channelsSize + 0];
    const float y1 = output[n * channelsSize + 1];
    const float x2 = output[n * channelsSize + 2];
    const float y2 = output[n * channelsSize + 3];

    NvDsInferInstanceMaskInfo b {};
    addBBoxProposal(x1, y1, x2, y2, netW, netH, maxIndex, maxProb, b);
    if (b.width < 1.0F || b.height < 1.0F) {
      continue;
    }

    addSegProposal(output, channelsSize, netW, netH, n, b);
    objects.push_back(b);
  }

  return objects;
}

bool decodeYoloSegLikeOutput(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR - Could not find output layer" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo& output = outputLayersInfo[0];
  const size_t outputSize = output.inferDims.d[0];
  const size_t channelsSize = output.inferDims.d[1];

  objectList = decodeTensorYoloSeg(
      static_cast<const float*>(output.buffer), outputSize, channelsSize, networkInfo.width, networkInfo.height,
      detectionParams.perClassPreclusterThreshold);
  return true;
}

const NvDsInferLayerInfo* findLayerByName(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo, const std::string& name)
{
  for (const auto& layer : outputLayersInfo) {
    if (layer.layerName != nullptr && name == layer.layerName) {
      return &layer;
    }
  }
  return nullptr;
}

bool getTensorChannelsAndAnchors(const NvDsInferLayerInfo& layer, size_t& channels, size_t& anchors)
{
  const auto& dims = layer.inferDims;
  if (dims.numDims == 2) {
    channels = dims.d[0];
    anchors = dims.d[1];
    return true;
  }

  if (dims.numDims == 3) {
    if (dims.d[0] == 1U) {
      channels = dims.d[1];
      anchors = dims.d[2];
    } else {
      channels = dims.d[0];
      anchors = dims.d[2];
    }
    return true;
  }

  return false;
}

bool getTensorCHW(const NvDsInferLayerInfo& layer, size_t& channels, size_t& height, size_t& width)
{
  const auto& dims = layer.inferDims;
  if (dims.numDims == 3) {
    channels = dims.d[0];
    height = dims.d[1];
    width = dims.d[2];
    return true;
  }

  if (dims.numDims == 4 && dims.d[0] == 1U) {
    channels = dims.d[1];
    height = dims.d[2];
    width = dims.d[3];
    return true;
  }

  return false;
}

std::vector<FusionAnchor> buildFusionAnchors(const NvDsInferNetworkInfo& networkInfo)
{
  static constexpr std::array<int, 3> kStrides = {8, 16, 32};

  std::vector<FusionAnchor> anchors;
  for (int stride : kStrides) {
    const int featureH = static_cast<int>(networkInfo.height) / stride;
    const int featureW = static_cast<int>(networkInfo.width) / stride;

    for (int y = 0; y < featureH; ++y) {
      for (int x = 0; x < featureW; ++x) {
        anchors.push_back(FusionAnchor {x + 0.5F, y + 0.5F, static_cast<float>(stride)});
      }
    }
  }

  return anchors;
}

float decodeFusionDistance(const float* boxData, size_t numAnchors, size_t anchorIdx, size_t coordIdx, size_t regMax)
{
  const size_t base = coordIdx * regMax;
  float maxLogit = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < regMax; ++i) {
    const float logit = boxData[(base + i) * numAnchors + anchorIdx];
    maxLogit = std::max(maxLogit, logit);
  }

  float sum = 0.0F;
  float weighted = 0.0F;
  for (size_t i = 0; i < regMax; ++i) {
    const float e = std::exp(boxData[(base + i) * numAnchors + anchorIdx] - maxLogit);
    sum += e;
    weighted += e * static_cast<float>(i);
  }

  if (sum <= 0.0F) {
    return 0.0F;
  }
  return weighted / sum;
}

float getMappedThreshold(const NvDsInferParseDetectionParams& detectionParams, unsigned int classId)
{
  if (classId < detectionParams.perClassPreclusterThreshold.size()) {
    return detectionParams.perClassPreclusterThreshold[classId];
  }

  return 0.25F;
}

int mapPersonHeadClass(int sourceClass)
{
  switch (sourceClass) {
    case 0:
      return 0;  // person
    case 2:
    case 5:
    case 7:
      return 1;  // merged to car
    default:
      return -1;
  }
}

int mapRobotHeadClass(int sourceClass)
{
  return sourceClass == 0 ? 2 : -1;
}

const char* fusionClassName(unsigned int classId)
{
  switch (classId) {
    case 0:
      return "person";
    case 1:
      return "car";
    case 2:
      return "robot";
    default:
      return "unknown";
  }
}

std::vector<FusionDetection> nonMaxSuppressFusionDetections(std::vector<FusionDetection> detections)
{
  std::sort(detections.begin(), detections.end(), [](const FusionDetection& a, const FusionDetection& b) {
    return a.confidence > b.confidence;
  });

  std::vector<FusionDetection> kept;
  std::vector<bool> suppressed(detections.size(), false);

  for (size_t i = 0; i < detections.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }

    kept.push_back(detections[i]);
    for (size_t j = i + 1; j < detections.size(); ++j) {
      if (suppressed[j] || detections[i].classId != detections[j].classId) {
        continue;
      }

      const float iou = safeIoU(
          detections[i].x1, detections[i].y1, detections[i].x2, detections[i].y2,
          detections[j].x1, detections[j].y1, detections[j].x2, detections[j].y2);
      if (iou > kFusionIouThreshold) {
        suppressed[j] = true;
      }
    }
  }

  return kept;
}

template <typename Mapper>
std::vector<FusionDetection> parseFusionHead(const NvDsInferLayerInfo& boxesLayer, const NvDsInferLayerInfo& scoresLayer,
    const NvDsInferLayerInfo& maskLayer, const NvDsInferLayerInfo& protoLayer, const NvDsInferNetworkInfo& networkInfo,
    const NvDsInferParseDetectionParams& detectionParams, Mapper mapper)
{
  size_t boxChannels = 0;
  size_t numAnchors = 0;
  size_t scoreClasses = 0;
  size_t scoreAnchors = 0;
  size_t maskChannels = 0;
  size_t maskAnchors = 0;
  size_t protoC = 0;
  size_t protoH = 0;
  size_t protoW = 0;

  if (!getTensorChannelsAndAnchors(boxesLayer, boxChannels, numAnchors) ||
      !getTensorChannelsAndAnchors(scoresLayer, scoreClasses, scoreAnchors) ||
      !getTensorChannelsAndAnchors(maskLayer, maskChannels, maskAnchors) ||
      !getTensorCHW(protoLayer, protoC, protoH, protoW)) {
    std::cerr << "ERROR - Fusion parser received unexpected tensor dimensions" << std::endl;
    return {};
  }

  if (boxChannels != 4U * kFusionRegMax || numAnchors != scoreAnchors || numAnchors != maskAnchors ||
      maskChannels != kFusionMaskChannels || protoC != kFusionMaskChannels) {
    std::cerr << "ERROR - Fusion parser tensor shape mismatch" << std::endl;
    return {};
  }

  const auto anchors = buildFusionAnchors(networkInfo);
  if (anchors.size() != numAnchors) {
    std::cerr << "ERROR - Fusion parser anchor count mismatch: expected " << anchors.size()
              << " got " << numAnchors << std::endl;
    return {};
  }

  const float* boxData = static_cast<const float*>(boxesLayer.buffer);
  const float* scoreData = static_cast<const float*>(scoresLayer.buffer);
  const float* maskData = static_cast<const float*>(maskLayer.buffer);
  const float* protoData = static_cast<const float*>(protoLayer.buffer);

  std::vector<FusionDetection> detections;
  detections.reserve(numAnchors / 8);

  for (size_t anchorIdx = 0; anchorIdx < numAnchors; ++anchorIdx) {
    int bestSourceClass = -1;
    float bestScore = 0.0F;

    for (size_t classIdx = 0; classIdx < scoreClasses; ++classIdx) {
      const float score = sigmoidFloat(scoreData[classIdx * numAnchors + anchorIdx]);
      if (score > bestScore) {
        bestScore = score;
        bestSourceClass = static_cast<int>(classIdx);
      }
    }

    if (bestSourceClass < 0) {
      continue;
    }

    const int mappedClass = mapper(bestSourceClass);
    if (mappedClass < 0) {
      continue;
    }

    const float minScore = getMappedThreshold(detectionParams, static_cast<unsigned int>(mappedClass));
    if (bestScore < minScore) {
      continue;
    }

    const float left = decodeFusionDistance(boxData, numAnchors, anchorIdx, 0, kFusionRegMax);
    const float top = decodeFusionDistance(boxData, numAnchors, anchorIdx, 1, kFusionRegMax);
    const float right = decodeFusionDistance(boxData, numAnchors, anchorIdx, 2, kFusionRegMax);
    const float bottom = decodeFusionDistance(boxData, numAnchors, anchorIdx, 3, kFusionRegMax);

    const FusionAnchor& anchor = anchors[anchorIdx];
    FusionDetection det;
    det.classId = static_cast<unsigned int>(mappedClass);
    det.confidence = bestScore;
    det.x1 = clampFloat((anchor.x - left) * anchor.stride, 0.0F, static_cast<float>(networkInfo.width));
    det.y1 = clampFloat((anchor.y - top) * anchor.stride, 0.0F, static_cast<float>(networkInfo.height));
    det.x2 = clampFloat((anchor.x + right) * anchor.stride, 0.0F, static_cast<float>(networkInfo.width));
    det.y2 = clampFloat((anchor.y + bottom) * anchor.stride, 0.0F, static_cast<float>(networkInfo.height));
    if ((det.x2 - det.x1) < 1.0F || (det.y2 - det.y1) < 1.0F) {
      continue;
    }

    for (size_t c = 0; c < kFusionMaskChannels; ++c) {
      det.maskCoefficients[c] = maskData[c * numAnchors + anchorIdx];
    }
    det.proto = protoData;
    det.protoC = protoC;
    det.protoH = protoH;
    det.protoW = protoW;
    detections.push_back(det);
  }

  return nonMaxSuppressFusionDetections(std::move(detections));
}

void buildFusionMask(FusionDetection& det, const NvDsInferNetworkInfo& networkInfo)
{
  if (det.proto == nullptr || det.protoC != kFusionMaskChannels || det.protoH == 0 || det.protoW == 0) {
    return;
  }

  det.mask.assign(det.protoH * det.protoW, 0.0F);

  const float scaleX = static_cast<float>(det.protoW) / static_cast<float>(networkInfo.width);
  const float scaleY = static_cast<float>(det.protoH) / static_cast<float>(networkInfo.height);

  const int left = std::max(0, static_cast<int>(std::floor(det.x1 * scaleX)));
  const int top = std::max(0, static_cast<int>(std::floor(det.y1 * scaleY)));
  const int right = std::min(static_cast<int>(det.protoW), static_cast<int>(std::ceil(det.x2 * scaleX)));
  const int bottom = std::min(static_cast<int>(det.protoH), static_cast<int>(std::ceil(det.y2 * scaleY)));

  if (right <= left || bottom <= top) {
    return;
  }

  for (int y = top; y < bottom; ++y) {
    for (int x = left; x < right; ++x) {
      const size_t idx = static_cast<size_t>(y) * det.protoW + static_cast<size_t>(x);
      float value = 0.0F;
      for (size_t c = 0; c < det.protoC; ++c) {
        value += det.maskCoefficients[c] * det.proto[c * det.protoH * det.protoW + idx];
      }
      det.mask[idx] = value > 0.0F ? 1.0F : 0.0F;
    }
  }
}

void suppressPersonInsideRobot(std::vector<FusionDetection>& detections)
{
  std::vector<size_t> robotIndices;
  std::vector<size_t> personIndices;
  for (size_t i = 0; i < detections.size(); ++i) {
    if (detections[i].classId == 2U) {
      robotIndices.push_back(i);
    } else if (detections[i].classId == 0U) {
      personIndices.push_back(i);
    }
  }

  if (robotIndices.empty() || personIndices.empty()) {
    return;
  }

  std::vector<bool> keep(detections.size(), true);
  for (const size_t personIdx : personIndices) {
    const FusionDetection& person = detections[personIdx];
    const float centerX = (person.x1 + person.x2) * 0.5F;
    const float centerY = (person.y1 + person.y2) * 0.5F;

    bool suppress = false;
    for (const size_t robotIdx : robotIndices) {
      const FusionDetection& robot = detections[robotIdx];
      if (centerX >= robot.x1 && centerX <= robot.x2 && centerY >= robot.y1 && centerY <= robot.y2) {
        suppress = true;
        break;
      }
    }

    if (suppress || person.mask.empty()) {
      keep[personIdx] = !suppress;
      continue;
    }

    size_t personArea = 0;
    for (const float value : person.mask) {
      if (value > 0.5F) {
        ++personArea;
      }
    }

    if (personArea == 0U) {
      continue;
    }

    for (const size_t robotIdx : robotIndices) {
      const FusionDetection& robot = detections[robotIdx];
      if (robot.mask.empty() || robot.mask.size() != person.mask.size()) {
        continue;
      }

      size_t inter = 0;
      for (size_t i = 0; i < person.mask.size(); ++i) {
        if (person.mask[i] > 0.5F && robot.mask[i] > 0.5F) {
          ++inter;
        }
      }

      const float coverRatio = static_cast<float>(inter) / static_cast<float>(personArea);
      if (coverRatio >= kRobotMaskCoverThreshold) {
        suppress = true;
        break;
      }
    }

    if (suppress) {
      keep[personIdx] = false;
    }
  }

  std::vector<FusionDetection> filtered;
  filtered.reserve(detections.size());
  for (size_t i = 0; i < detections.size(); ++i) {
    if (keep[i]) {
      filtered.push_back(std::move(detections[i]));
    }
  }
  detections.swap(filtered);
}

struct CoordSpace {
  bool enabled {false};
  float srcW {0.0F};
  float srcH {0.0F};
  float scale {1.0F};
  float padX {0.0F};
  float padY {0.0F};
};

CoordSpace getCoordSpace(const NvDsInferNetworkInfo& networkInfo)
{
  CoordSpace space;
  const char* srcW = std::getenv("FUSION_SOURCE_WIDTH");
  const char* srcH = std::getenv("FUSION_SOURCE_HEIGHT");
  if (srcW == nullptr || srcH == nullptr) {
    return space;
  }

  space.srcW = std::strtof(srcW, nullptr);
  space.srcH = std::strtof(srcH, nullptr);
  if (space.srcW <= 0.0F || space.srcH <= 0.0F) {
    return space;
  }

  space.scale = std::min(
      static_cast<float>(networkInfo.width) / space.srcW,
      static_cast<float>(networkInfo.height) / space.srcH);
  space.padX = (static_cast<float>(networkInfo.width) - space.srcW * space.scale) * 0.5F;
  space.padY = (static_cast<float>(networkInfo.height) - space.srcH * space.scale) * 0.5F;
  space.enabled = true;
  return space;
}

void convertBoxForLogging(const CoordSpace& space, const FusionDetection& det,
    float& outX1, float& outY1, float& outX2, float& outY2)
{
  if (!space.enabled) {
    outX1 = det.x1;
    outY1 = det.y1;
    outX2 = det.x2;
    outY2 = det.y2;
    return;
  }

  outX1 = clampFloat((det.x1 - space.padX) / space.scale, 0.0F, space.srcW);
  outY1 = clampFloat((det.y1 - space.padY) / space.scale, 0.0F, space.srcH);
  outX2 = clampFloat((det.x2 - space.padX) / space.scale, 0.0F, space.srcW);
  outY2 = clampFloat((det.y2 - space.padY) / space.scale, 0.0F, space.srcH);
}

void convertPointForLogging(const CoordSpace& space, const NvDsInferNetworkInfo& networkInfo,
    size_t protoW, size_t protoH, float px, float py, float& outX, float& outY)
{
  const float netX = (px / static_cast<float>(protoW)) * static_cast<float>(networkInfo.width);
  const float netY = (py / static_cast<float>(protoH)) * static_cast<float>(networkInfo.height);

  if (!space.enabled) {
    outX = netX;
    outY = netY;
    return;
  }

  outX = clampFloat((netX - space.padX) / space.scale, 0.0F, space.srcW);
  outY = clampFloat((netY - space.padY) / space.scale, 0.0F, space.srcH);
}

bool saveSegmentPointsEnabled()
{
  const char* env = std::getenv("FUSION_SEGMENT_POINTS");
  return env != nullptr && std::atoi(env) != 0;
}

int segmentationPointStride()
{
  const char* env = std::getenv("FUSION_SEGMENT_POINT_STRIDE");
  if (env == nullptr) {
    return 4;
  }

  const int stride = std::atoi(env);
  return stride > 0 ? stride : 4;
}

bool isBoundaryPixel(const FusionDetection& det, int x, int y)
{
  const int width = static_cast<int>(det.protoW);
  const int height = static_cast<int>(det.protoH);
  const auto maskAt = [&](int px, int py) -> float {
    return det.mask[static_cast<size_t>(py) * det.protoW + static_cast<size_t>(px)];
  };

  if (maskAt(x, y) <= 0.5F) {
    return false;
  }

  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0) {
        continue;
      }

      const int nx = x + dx;
      const int ny = y + dy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
        return true;
      }

      if (maskAt(nx, ny) <= 0.5F) {
        return true;
      }
    }
  }

  return false;
}

std::string buildSegmentationJson(const FusionDetection& det, const CoordSpace& coordSpace,
    const NvDsInferNetworkInfo& networkInfo)
{
  if (det.mask.empty() || det.protoW == 0U || det.protoH == 0U) {
    return "\"segmentation\":[]";
  }

  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(2);
  oss << "\"segmentation\":[";

  const int stride = segmentationPointStride();
  int emitted = 0;
  bool firstPoint = true;
  for (int y = 0; y < static_cast<int>(det.protoH); ++y) {
    for (int x = 0; x < static_cast<int>(det.protoW); ++x) {
      if (!isBoundaryPixel(det, x, y)) {
        continue;
      }

      if ((emitted % stride) != 0) {
        ++emitted;
        continue;
      }

      float outX = 0.0F;
      float outY = 0.0F;
      convertPointForLogging(
          coordSpace, networkInfo, det.protoW, det.protoH,
          static_cast<float>(x), static_cast<float>(y), outX, outY);

      if (!firstPoint) {
        oss << ',';
      }
      firstPoint = false;
      oss << '[' << outX << ',' << outY << ']';
      ++emitted;
    }
  }

  oss << ']';
  return oss.str();
}

void logFusionCoordinates(const std::vector<FusionDetection>& detections, const NvDsInferNetworkInfo& networkInfo)
{
  const char* stdoutEnv = std::getenv("FUSION_COORD_STDOUT");
  const bool emitStdout = stdoutEnv != nullptr && std::atoi(stdoutEnv) != 0;
  const char* logPath = std::getenv("FUSION_COORD_LOG_PATH");
  const bool saveSegmentation = saveSegmentPointsEnabled();

  if (!emitStdout && (logPath == nullptr || std::strlen(logPath) == 0U)) {
    return;
  }

  const CoordSpace coordSpace = getCoordSpace(networkInfo);

  std::lock_guard<std::mutex> lock(gFusionLogMutex);
  const uint64_t frameId = ++gFusionFrameCounter;
  std::ofstream outFile;
  if (logPath != nullptr && std::strlen(logPath) != 0U) {
    outFile.open(logPath, std::ios::out | std::ios::app);
  }

  for (const auto& det : detections) {
    float x1 = 0.0F;
    float y1 = 0.0F;
    float x2 = 0.0F;
    float y2 = 0.0F;
    convertBoxForLogging(coordSpace, det, x1, y1, x2, y2);

    const float centerX = (x1 + x2) * 0.5F;
    const float centerY = (y1 + y2) * 0.5F;

    std::ostringstream line;
    line.setf(std::ios::fixed);
    line.precision(2);
    line
        << "{\"frame\":" << frameId
        << ",\"class_id\":" << det.classId
        << ",\"class_name\":\"" << fusionClassName(det.classId) << "\""
        << ",\"conf\":" << det.confidence
        << ",\"center\":[" << centerX << "," << centerY << "]"
        << ",\"box\":[" << x1 << "," << y1 << "," << x2 << "," << y2 << "]}";

    if (emitStdout) {
      std::cout << line.str() << std::endl;
    }
    if (outFile.is_open()) {
      if (saveSegmentation) {
        std::string fileLine = line.str();
        fileLine.pop_back();
        fileLine.push_back(',');
        fileLine += buildSegmentationJson(det, coordSpace, networkInfo);
        fileLine.push_back('}');
        outFile << fileLine << '\n';
      } else {
        outFile << line.str() << '\n';
      }
    }
  }
}

bool buildFusionObjects(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
  const auto* boxes1 = findLayerByName(outputLayersInfo, "boxes1");
  const auto* scores1 = findLayerByName(outputLayersInfo, "scores1");
  const auto* mask1 = findLayerByName(outputLayersInfo, "mask1");
  const auto* proto1 = findLayerByName(outputLayersInfo, "proto1");
  const auto* boxes2 = findLayerByName(outputLayersInfo, "boxes2");
  const auto* scores2 = findLayerByName(outputLayersInfo, "scores2");
  const auto* mask2 = findLayerByName(outputLayersInfo, "mask2");
  const auto* proto2 = findLayerByName(outputLayersInfo, "proto2");

  if (boxes1 == nullptr || scores1 == nullptr || mask1 == nullptr || proto1 == nullptr ||
      boxes2 == nullptr || scores2 == nullptr || mask2 == nullptr || proto2 == nullptr) {
    std::cerr << "ERROR - Missing one or more fusion output tensors" << std::endl;
    return false;
  }

  std::vector<FusionDetection> detections = parseFusionHead(
      *boxes1, *scores1, *mask1, *proto1, networkInfo, detectionParams, mapPersonHeadClass);
  std::vector<FusionDetection> robotDetections = parseFusionHead(
      *boxes2, *scores2, *mask2, *proto2, networkInfo, detectionParams, mapRobotHeadClass);

  detections.insert(
      detections.end(),
      std::make_move_iterator(robotDetections.begin()),
      std::make_move_iterator(robotDetections.end()));

  for (auto& det : detections) {
    buildFusionMask(det, networkInfo);
  }

  suppressPersonInsideRobot(detections);
  logFusionCoordinates(detections, networkInfo);

  objectList.clear();
  objectList.reserve(detections.size());
  for (const auto& det : detections) {
    if (det.mask.empty()) {
      continue;
    }

    NvDsInferInstanceMaskInfo object {};
    object.classId = det.classId;
    object.left = det.x1;
    object.top = det.y1;
    object.width = det.x2 - det.x1;
    object.height = det.y2 - det.y1;
    object.detectionConfidence = det.confidence;
    object.mask_width = det.protoW;
    object.mask_height = det.protoH;
    object.mask_size = static_cast<unsigned int>(det.mask.size() * sizeof(float));
    object.mask = new float[det.mask.size()];
    std::copy(det.mask.begin(), det.mask.end(), object.mask);
    objectList.push_back(object);
  }

  return true;
}

}  // namespace

extern "C" bool
NvDsInferParseYoloSeg(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
  return decodeYoloSegLikeOutput(outputLayersInfo, networkInfo, detectionParams, objectList);
}

extern "C" bool
NvDsInferParseFusionSeg(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
  return buildFusionObjects(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloSeg);
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseFusionSeg);
