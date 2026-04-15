#include <gst/gst.h>
#include <glib.h>
#include <gstnvdsinfer.h>
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

volatile std::sig_atomic_t g_stop = 0;

constexpr int kDflBins = 16;
constexpr int kMaskCoeffCount = 32;
constexpr int kDefaultHead1Classes = 80;
constexpr int kDefaultHead2Classes = 3;
constexpr size_t kMaxDisplayLen = 256;
constexpr int kDefaultRecordBitrate = 8000000;

struct AppConfig {
    std::string camera_device = "/dev/video0";
    int camera_width = 1280;
    int camera_height = 720;
    int camera_fps = 30;
    int exposure_auto = -1;
    int exposure_absolute = -1;
    int gain = -1;
    std::string extra_controls;

    int content_width = 960;
    int content_height = 540;
    int infer_width = 960;
    int infer_height = 544;
    int display_width = 960;
    int display_height = 560;

    float head1_threshold = 0.01f;
    float head2_threshold = 0.01f;
    float nms_iou_threshold = 0.45f;
    float mask_threshold = 0.5f;
    unsigned int max_detections = 300;
    int record_bitrate = kDefaultRecordBitrate;

    std::string labels_file = "/workspace/DeepStream-Yolo-Seg/labels.txt";
    std::string nvinfer_config =
        "/workspace/DeepStream-Yolo-Seg/config_infer_primary_fusion_960x544_engine_tensormeta.txt";
    std::string output_file;

    bool preview = false;
    bool verbose = false;
    unsigned int tensor_log_interval = 30;
};

struct Detection {
    float left = 0.0f;
    float top = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    float confidence = 0.0f;
    int class_id = -1;
    int head_index = -1;
    int anchor_index = -1;
    std::array<float, kMaskCoeffCount> mask_coeffs{};
    std::vector<float> mask_data;
    unsigned int mask_width = 0;
    unsigned int mask_height = 0;
};

struct ProbeState {
    AppConfig config;
    std::vector<std::string> labels;
    guint64 frame_counter = 0;
    bool printed_layer_info = false;
    unsigned int tensor_log_interval = 30;
    guint64 total_detection_count = 0;
    guint64 total_decode_ms = 0;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
};

void signal_handler(int) {
    g_stop = 1;
}

std::string dims_to_string(const NvDsInferDims &dims) {
    std::ostringstream oss;
    oss << "[";
    for (unsigned int i = 0; i < dims.numDims; ++i) {
        if (i != 0) {
            oss << "x";
        }
        oss << dims.d[i];
    }
    oss << "]";
    return oss.str();
}

float sigmoid(float x) {
    if (x >= 0.0f) {
        float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    float z = std::exp(x);
    return z / (1.0f + z);
}

float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

float intersection_over_union(const Detection &a, const Detection &b) {
    const float ax2 = a.left + a.width;
    const float ay2 = a.top + a.height;
    const float bx2 = b.left + b.width;
    const float by2 = b.top + b.height;

    const float inter_left = std::max(a.left, b.left);
    const float inter_top = std::max(a.top, b.top);
    const float inter_right = std::min(ax2, bx2);
    const float inter_bottom = std::min(ay2, by2);

    const float inter_w = std::max(0.0f, inter_right - inter_left);
    const float inter_h = std::max(0.0f, inter_bottom - inter_top);
    const float inter_area = inter_w * inter_h;
    if (inter_area <= 0.0f) {
        return 0.0f;
    }

    const float union_area = a.width * a.height + b.width * b.height - inter_area;
    if (union_area <= 0.0f) {
        return 0.0f;
    }
    return inter_area / union_area;
}

std::vector<std::string> load_labels(const std::string &labels_file) {
    std::vector<std::string> labels;
    std::ifstream input(labels_file);
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            labels.push_back(line);
        }
    }

    while (labels.size() < static_cast<size_t>(kDefaultHead1Classes)) {
        labels.push_back("coco_" + std::to_string(labels.size()));
    }

    labels.push_back("robot_0");
    labels.push_back("robot_1");
    labels.push_back("robot_2");
    return labels;
}

int last_dim(const NvDsInferLayerInfo &layer) {
    return static_cast<int>(layer.inferDims.d[layer.inferDims.numDims - 1]);
}

int channel_count_except_last(const NvDsInferLayerInfo &layer) {
    int channels = 1;
    for (unsigned int i = 0; i + 1 < layer.inferDims.numDims; ++i) {
        channels *= static_cast<int>(layer.inferDims.d[i]);
    }
    return channels;
}

float dfl_distance(const float *boxes, int num_anchors, int anchor_idx, int side) {
    float logits[kDflBins];
    float max_logit = -1e30f;
    for (int b = 0; b < kDflBins; ++b) {
        float v = boxes[(side * kDflBins + b) * num_anchors + anchor_idx];
        logits[b] = v;
        max_logit = std::max(max_logit, v);
    }

    float sum = 0.0f;
    float dist = 0.0f;
    for (int b = 0; b < kDflBins; ++b) {
        float e = std::exp(logits[b] - max_logit);
        sum += e;
        dist += e * static_cast<float>(b);
    }
    return sum > 0.0f ? dist / sum : 0.0f;
}

std::vector<Detection> decode_head(
    const float *boxes,
    const float *scores,
    const float *mask_coeffs,
    int num_anchors,
    int class_count,
    int class_offset,
    float score_threshold,
    unsigned int infer_width,
    unsigned int infer_height) {

    std::vector<Detection> detections;
    const std::array<int, 3> strides{8, 16, 32};
    std::vector<int> level_offsets;
    level_offsets.reserve(strides.size() + 1);
    level_offsets.push_back(0);

    int total = 0;
    for (int stride : strides) {
        const int grid_w = static_cast<int>(infer_width) / stride;
        const int grid_h = static_cast<int>(infer_height) / stride;
        total += grid_w * grid_h;
        level_offsets.push_back(total);
    }

    if (total != num_anchors) {
        std::cerr << "[warn] Unexpected anchor count. expected=" << total
                  << " actual=" << num_anchors << std::endl;
    }

    for (int anchor_idx = 0; anchor_idx < num_anchors; ++anchor_idx) {
        float best_score = 0.0f;
        int best_class = -1;
        for (int c = 0; c < class_count; ++c) {
            float prob = sigmoid(scores[c * num_anchors + anchor_idx]);
            if (prob > best_score) {
                best_score = prob;
                best_class = c;
            }
        }

        if (best_class < 0 || best_score < score_threshold) {
            continue;
        }

        int stride = strides.back();
        int local_anchor = anchor_idx;
        int level = static_cast<int>(strides.size()) - 1;
        for (int i = 0; i < static_cast<int>(strides.size()); ++i) {
            if (anchor_idx < level_offsets[i + 1]) {
                stride = strides[i];
                local_anchor = anchor_idx - level_offsets[i];
                level = i;
                break;
            }
        }

        (void) level;
        const int grid_w = static_cast<int>(infer_width) / stride;
        const int grid_x = local_anchor % grid_w;
        const int grid_y = local_anchor / grid_w;

        const float cx = (static_cast<float>(grid_x) + 0.5f) * stride;
        const float cy = (static_cast<float>(grid_y) + 0.5f) * stride;

        const float l = dfl_distance(boxes, num_anchors, anchor_idx, 0) * stride;
        const float t = dfl_distance(boxes, num_anchors, anchor_idx, 1) * stride;
        const float r = dfl_distance(boxes, num_anchors, anchor_idx, 2) * stride;
        const float b = dfl_distance(boxes, num_anchors, anchor_idx, 3) * stride;

        float x1 = clampf(cx - l, 0.0f, static_cast<float>(infer_width - 1));
        float y1 = clampf(cy - t, 0.0f, static_cast<float>(infer_height - 1));
        float x2 = clampf(cx + r, 0.0f, static_cast<float>(infer_width - 1));
        float y2 = clampf(cy + b, 0.0f, static_cast<float>(infer_height - 1));

        float w = x2 - x1;
        float h = y2 - y1;
        if (w < 1.0f || h < 1.0f) {
            continue;
        }

        Detection det;
        det.left = x1;
        det.top = y1;
        det.width = w;
        det.height = h;
        det.confidence = best_score;
        det.class_id = class_offset + best_class;
        det.head_index = class_offset == 0 ? 0 : 1;
        det.anchor_index = anchor_idx;
        if (mask_coeffs) {
            for (int i = 0; i < kMaskCoeffCount; ++i) {
                det.mask_coeffs[i] = mask_coeffs[i * num_anchors + anchor_idx];
            }
        }
        detections.push_back(det);
    }

    return detections;
}

std::vector<Detection> nms_by_class(std::vector<Detection> detections, float iou_threshold,
                                    unsigned int max_detections) {
    std::sort(detections.begin(), detections.end(),
              [](const Detection &a, const Detection &b) {
                  return a.confidence > b.confidence;
              });

    std::vector<Detection> kept;
    kept.reserve(detections.size());

    for (const auto &det : detections) {
        bool suppressed = false;
        for (const auto &picked : kept) {
            if (picked.class_id != det.class_id) {
                continue;
            }
            if (intersection_over_union(det, picked) > iou_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(det);
            if (kept.size() >= max_detections) {
                break;
            }
        }
    }
    return kept;
}

std::vector<Detection> keep_top_k(std::vector<Detection> detections, size_t top_k) {
    std::sort(detections.begin(), detections.end(),
              [](const Detection &a, const Detection &b) {
                  return a.confidence > b.confidence;
              });
    if (detections.size() > top_k) {
        detections.resize(top_k);
    }
    return detections;
}

struct ProtoInfo {
    const float *data = nullptr;
    int channels = 0;
    int height = 0;
    int width = 0;
};

ProtoInfo get_proto_info(const NvDsInferLayerInfo &layer) {
    ProtoInfo info;
    info.data = static_cast<const float *>(layer.buffer);
    if (layer.inferDims.numDims >= 3) {
        info.channels = static_cast<int>(layer.inferDims.d[0]);
        info.height = static_cast<int>(layer.inferDims.d[1]);
        info.width = static_cast<int>(layer.inferDims.d[2]);
    }
    return info;
}

void populate_mask(
    Detection &det,
    const ProtoInfo &proto,
    unsigned int infer_width,
    unsigned int infer_height) {

    if (!proto.data || proto.channels != kMaskCoeffCount || proto.height <= 0 || proto.width <= 0) {
        return;
    }

    det.mask_width = static_cast<unsigned int>(proto.width);
    det.mask_height = static_cast<unsigned int>(proto.height);
    det.mask_data.assign(static_cast<size_t>(proto.width * proto.height), 0.0f);

    const float scale_x = static_cast<float>(proto.width) / static_cast<float>(infer_width);
    const float scale_y = static_cast<float>(proto.height) / static_cast<float>(infer_height);

    const int x0 = std::max(0, std::min(proto.width - 1, static_cast<int>(std::floor(det.left * scale_x))));
    const int y0 = std::max(0, std::min(proto.height - 1, static_cast<int>(std::floor(det.top * scale_y))));
    const int x1 = std::max(0, std::min(proto.width, static_cast<int>(std::ceil((det.left + det.width) * scale_x))));
    const int y1 = std::max(0, std::min(proto.height, static_cast<int>(std::ceil((det.top + det.height) * scale_y))));

    const int pixel_count = proto.width * proto.height;
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            const int pixel_index = y * proto.width + x;
            float value = 0.0f;
            for (int c = 0; c < proto.channels; ++c) {
                value += det.mask_coeffs[c] * proto.data[c * pixel_count + pixel_index];
            }
            det.mask_data[static_cast<size_t>(pixel_index)] = sigmoid(value);
        }
    }
}

void attach_object_meta(
    NvDsBatchMeta *batch_meta,
    NvDsFrameMeta *frame_meta,
    const std::vector<Detection> &detections,
    const std::vector<std::string> &labels,
    float mask_threshold) {

    for (const auto &det : detections) {
        NvDsObjectMeta *obj_meta = nvds_acquire_obj_meta_from_pool(batch_meta);
        if (!obj_meta) {
            continue;
        }

        obj_meta->unique_component_id = 1;
        obj_meta->confidence = det.confidence;
        obj_meta->object_id = UNTRACKED_OBJECT_ID;
        obj_meta->class_id = det.class_id;

        auto &rect = obj_meta->rect_params;
        rect.left = det.left;
        rect.top = det.top;
        rect.width = det.width;
        rect.height = det.height;
        rect.border_width = 3;
        rect.has_bg_color = 0;
        rect.border_color = det.head_index == 0
            ? NvOSD_ColorParams{1.0f, 0.0f, 0.0f, 1.0f}
            : NvOSD_ColorParams{0.0f, 1.0f, 0.0f, 1.0f};

        auto &text = obj_meta->text_params;
        std::ostringstream label_stream;
        if (det.class_id >= 0 && det.class_id < static_cast<int>(labels.size())) {
            label_stream << labels[det.class_id];
        } else {
            label_stream << "cls_" << det.class_id;
        }
        label_stream << " " << std::fixed << std::setprecision(2) << det.confidence;
        text.display_text = g_strdup(label_stream.str().c_str());
        text.x_offset = static_cast<int>(det.left);
        text.y_offset = std::max(0, static_cast<int>(det.top) - 10);
        text.set_bg_clr = 1;
        text.text_bg_clr = NvOSD_ColorParams{0.0f, 0.0f, 0.0f, 1.0f};
        text.font_params.font_name = const_cast<gchar *>("Serif");
        text.font_params.font_size = 11;
        text.font_params.font_color = NvOSD_ColorParams{1.0f, 1.0f, 1.0f, 1.0f};

        if (!det.mask_data.empty()) {
            const size_t mask_bytes = det.mask_data.size() * sizeof(float);
            float *mask = static_cast<float *>(g_malloc(mask_bytes));
            std::memcpy(mask, det.mask_data.data(), mask_bytes);
            obj_meta->mask_params.data = mask;
            obj_meta->mask_params.size = static_cast<unsigned int>(mask_bytes);
            obj_meta->mask_params.threshold = mask_threshold;
            obj_meta->mask_params.width = det.mask_width;
            obj_meta->mask_params.height = det.mask_height;
        }

        nvds_add_obj_meta_to_frame(frame_meta, obj_meta, nullptr);
    }
}

void attach_summary_meta(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta,
                         const std::vector<Detection> &detections,
                         double avg_fps,
                         const AppConfig &cfg) {
    NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    if (!display_meta) {
        return;
    }

    display_meta->num_labels = 1;
    auto &text = display_meta->text_params[0];
    text.display_text = static_cast<gchar *>(g_malloc0(kMaxDisplayLen));
    std::snprintf(text.display_text, kMaxDisplayLen,
                  "fps=%.2f det=%zu frame=%u", avg_fps, detections.size(), frame_meta->frame_num);
    text.x_offset = 10;
    text.y_offset = cfg.display_height > cfg.infer_height
        ? (cfg.infer_height + 2)
        : 12;
    text.font_params.font_name = const_cast<gchar *>("Serif");
    text.font_params.font_size = 12;
    text.font_params.font_color = NvOSD_ColorParams{1.0f, 1.0f, 1.0f, 1.0f};
    text.set_bg_clr = 1;
    text.text_bg_clr = NvOSD_ColorParams{0.0f, 0.0f, 0.0f, 1.0f};
    nvds_add_display_meta_to_frame(frame_meta, display_meta);
}

std::vector<Detection> decode_fusion_tensor_meta(
    NvDsInferTensorMeta *tensor_meta,
    const AppConfig &cfg) {

    std::unordered_map<std::string, const NvDsInferLayerInfo *> layer_map;
    for (guint i = 0; i < tensor_meta->num_output_layers; ++i) {
        auto &layer = tensor_meta->output_layers_info[i];
        layer_map[layer.layerName] = &layer;
    }

    auto require_layer = [&](const char *name) -> const NvDsInferLayerInfo * {
        auto it = layer_map.find(name);
        return it == layer_map.end() ? nullptr : it->second;
    };

    const auto *boxes1 = require_layer("boxes1");
    const auto *scores1 = require_layer("scores1");
    const auto *mask1 = require_layer("mask1");
    const auto *proto1 = require_layer("proto1");
    const auto *boxes2 = require_layer("boxes2");
    const auto *scores2 = require_layer("scores2");
    const auto *mask2 = require_layer("mask2");
    const auto *proto2 = require_layer("proto2");
    if (!boxes1 || !scores1 || !mask1 || !proto1 || !boxes2 || !scores2 || !mask2 || !proto2) {
        std::cerr << "[warn] Missing expected fusion output layers" << std::endl;
        return {};
    }

    const int anchors1 = last_dim(*scores1);
    const int anchors2 = last_dim(*scores2);
    const int classes1 = channel_count_except_last(*scores1);
    const int classes2 = channel_count_except_last(*scores2);
    const int box_channels1 = channel_count_except_last(*boxes1);
    const int box_channels2 = channel_count_except_last(*boxes2);

    if (box_channels1 != 4 * kDflBins || box_channels2 != 4 * kDflBins) {
        std::cerr << "[warn] Unexpected box channel count: "
                  << box_channels1 << ", " << box_channels2 << std::endl;
        return {};
    }

    if (classes1 != kDefaultHead1Classes || classes2 != kDefaultHead2Classes) {
        std::cerr << "[warn] Unexpected class count: "
                  << classes1 << ", " << classes2 << std::endl;
    }

    auto head1 = decode_head(
        static_cast<const float *>(boxes1->buffer),
        static_cast<const float *>(scores1->buffer),
        static_cast<const float *>(mask1->buffer),
        anchors1, classes1, 0, cfg.head1_threshold,
        tensor_meta->network_info.width, tensor_meta->network_info.height);

    auto head2 = decode_head(
        static_cast<const float *>(boxes2->buffer),
        static_cast<const float *>(scores2->buffer),
        static_cast<const float *>(mask2->buffer),
        anchors2, classes2, kDefaultHead1Classes, cfg.head2_threshold,
        tensor_meta->network_info.width, tensor_meta->network_info.height);

    head1.insert(head1.end(), head2.begin(), head2.end());
    head1 = keep_top_k(std::move(head1), 1200);
    auto kept = nms_by_class(std::move(head1), cfg.nms_iou_threshold, cfg.max_detections);

    const auto proto_info_1 = get_proto_info(*proto1);
    const auto proto_info_2 = get_proto_info(*proto2);
    for (auto &det : kept) {
        populate_mask(det,
                      det.head_index == 0 ? proto_info_1 : proto_info_2,
                      tensor_meta->network_info.width,
                      tensor_meta->network_info.height);
    }
    return kept;
}

void log_tensor_layers_once(NvDsInferTensorMeta *tensor_meta) {
    std::cout
        << "[tensor-meta] uid=" << tensor_meta->unique_id
        << " layers=" << tensor_meta->num_output_layers
        << " network=" << tensor_meta->network_info.width
        << "x" << tensor_meta->network_info.height
        << "x" << tensor_meta->network_info.channels
        << " maintain_aspect_ratio=" << tensor_meta->maintain_aspect_ratio
        << " symmetric_padding=" << tensor_meta->symmetric_padding
        << std::endl;

    for (guint i = 0; i < tensor_meta->num_output_layers; ++i) {
        const auto &layer = tensor_meta->output_layers_info[i];
        std::cout
            << "  layer[" << i << "] name=" << (layer.layerName ? layer.layerName : "<null>")
            << " dims=" << dims_to_string(layer.inferDims)
            << " dataType=" << static_cast<int>(layer.dataType)
            << std::endl;
    }
}

GstPadProbeReturn infer_src_probe(GstPad *, GstPadProbeInfo *info, gpointer user_data) {
    auto *state = reinterpret_cast<ProbeState *>(user_data);
    if (!(GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER)) {
        return GST_PAD_PROBE_OK;
    }

    GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buffer) {
        return GST_PAD_PROBE_OK;
    }

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) {
        std::cerr << "[warn] NvDsBatchMeta not found" << std::endl;
        return GST_PAD_PROBE_OK;
    }

    state->frame_counter++;

    for (NvDsMetaList *frame_list = batch_meta->frame_meta_list; frame_list != nullptr;
         frame_list = frame_list->next) {
        auto *frame_meta = reinterpret_cast<NvDsFrameMeta *>(frame_list->data);
        if (!frame_meta) {
            continue;
        }

        bool found_tensor = false;
        for (NvDsMetaList *user_list = frame_meta->frame_user_meta_list; user_list != nullptr;
             user_list = user_list->next) {
            auto *user_meta = reinterpret_cast<NvDsUserMeta *>(user_list->data);
            if (!user_meta || user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META) {
                continue;
            }

            found_tensor = true;
            auto *tensor_meta = reinterpret_cast<NvDsInferTensorMeta *>(user_meta->user_meta_data);
            if (!tensor_meta) {
                continue;
            }

            for (guint i = 0; i < tensor_meta->num_output_layers; ++i) {
                tensor_meta->output_layers_info[i].buffer = tensor_meta->out_buf_ptrs_host[i];
            }

            if (!state->printed_layer_info) {
                log_tensor_layers_once(tensor_meta);
                state->printed_layer_info = true;
            }

            std::cerr << "[frame " << state->frame_counter << "] tensor meta ready" << std::endl;
            auto decode_begin = std::chrono::steady_clock::now();
            std::cerr << "[frame " << state->frame_counter << "] start decode" << std::endl;
            auto detections = decode_fusion_tensor_meta(tensor_meta, state->config);
            auto decode_end = std::chrono::steady_clock::now();
            auto decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                decode_end - decode_begin).count();
            state->total_decode_ms += static_cast<guint64>(decode_ms);
            state->total_detection_count += detections.size();
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::duration<double>>(
                    decode_end - state->start_time).count();
            const double avg_fps = (elapsed > 0.0)
                ? static_cast<double>(state->frame_counter) / elapsed
                : 0.0;
            std::cerr << "[frame " << state->frame_counter << "] decode finished, count="
                      << detections.size() << " ms=" << decode_ms << std::endl;

            if (state->tensor_log_interval > 0 &&
                (state->frame_counter % state->tensor_log_interval) == 0) {
                std::cout << "[frame " << state->frame_counter << "] detections="
                          << detections.size()
                          << " decode_ms=" << decode_ms
                          << " avg_fps=" << std::fixed << std::setprecision(2) << avg_fps
                          << std::endl;
                for (size_t i = 0; i < detections.size() && i < 10; ++i) {
                    const auto &det = detections[i];
                    const std::string label =
                        (det.class_id >= 0 && det.class_id < static_cast<int>(state->labels.size()))
                            ? state->labels[det.class_id]
                            : ("cls_" + std::to_string(det.class_id));
                    std::cout << "  det[" << i << "] head=" << det.head_index
                              << " cls=" << det.class_id
                              << " label=" << label
                              << " conf=" << std::fixed << std::setprecision(3) << det.confidence
                              << " box=(" << det.left << "," << det.top << ","
                              << det.width << "," << det.height << ")" << std::endl;
                }
            }

            attach_object_meta(batch_meta, frame_meta, detections, state->labels,
                               state->config.mask_threshold);
            attach_summary_meta(batch_meta, frame_meta, detections, avg_fps, state->config);
            std::cerr << "[frame " << state->frame_counter << "] metas attached" << std::endl;
        }

        if (!found_tensor && state->tensor_log_interval > 0 &&
            (state->frame_counter % state->tensor_log_interval) == 0) {
            std::cerr << "[warn] No NVDSINFER_TENSOR_OUTPUT_META found on frame "
                      << state->frame_counter << std::endl;
        }
    }

    return GST_PAD_PROBE_OK;
}

std::string make_common_source(const AppConfig &cfg) {
    std::ostringstream oss;
    std::ostringstream controls;
    bool has_controls = false;
    if (cfg.exposure_auto >= 0) {
        controls << "exposure_auto=" << cfg.exposure_auto;
        has_controls = true;
    }
    if (cfg.exposure_absolute >= 0) {
        if (has_controls) {
            controls << ",";
        }
        controls << "exposure_absolute=" << cfg.exposure_absolute;
        has_controls = true;
    }
    if (cfg.gain >= 0) {
        if (has_controls) {
            controls << ",";
        }
        controls << "gain=" << cfg.gain;
        has_controls = true;
    }
    if (!cfg.extra_controls.empty()) {
        if (has_controls) {
            controls << ",";
        }
        controls << cfg.extra_controls;
        has_controls = true;
    }

    oss
        << "v4l2src device=" << cfg.camera_device
        << (has_controls ? " extra-controls=\"c," + controls.str() + "\"" : "")
        << " ! image/jpeg,width=" << cfg.camera_width
        << ",height=" << cfg.camera_height
        << ",framerate=" << cfg.camera_fps << "/1"
        << " ! nvv4l2decoder mjpeg=1"
        << " ! nvvideoconvert compute-hw=2 interpolation-method=4"
        << " ! video/x-raw(memory:NVMM),format=NV12,width=" << cfg.content_width
        << ",height=" << cfg.content_height << " ! ";
    return oss.str();
}

std::string build_record_branch(const AppConfig &cfg);

std::string build_headless_pipeline(const AppConfig &cfg) {
    std::ostringstream oss;
    oss
        << "nvstreammux name=mux batch-size=1 width=" << cfg.infer_width
        << " height=" << cfg.infer_height
        << " live-source=1 batched-push-timeout=33000 nvbuf-memory-type=0"
        << " ! nvinfer name=primary_infer config-file-path=" << cfg.nvinfer_config
        << " ! nvcompositor name=compdisplay sink_0::xpos=0 sink_0::ypos=0 sink_0::width="
        << cfg.infer_width << " sink_0::height=" << cfg.infer_height
        << " ! video/x-raw(memory:NVMM),format=NV12,width=" << cfg.display_width
        << ",height=" << cfg.display_height
        << " ! nvdsosd name=osd process-mode=1 display-bbox=1 display-text=1 display-mask=1 display-clock=1 x-clock-offset=10 y-clock-offset="
        << (cfg.infer_height + 2) << " clock-font-size=12"
        << " ! " << build_record_branch(cfg) << " "
        << "nvcompositor name=compinfer sink_0::xpos=0 sink_0::ypos=0 sink_0::width="
        << cfg.content_width << " sink_0::height=" << cfg.content_height
        << " ! nvvideoconvert compute-hw=2 interpolation-method=4"
        << " ! video/x-raw(memory:NVMM),format=NV12,width=" << cfg.infer_width
        << ",height=" << cfg.infer_height << " ! mux.sink_0 "
        << make_common_source(cfg)
        << "compinfer.sink_0";
    return oss.str();
}

std::string build_preview_pipeline(const AppConfig &cfg) {
    std::ostringstream oss;
    oss
        << "nvstreammux name=mux batch-size=1 width=" << cfg.infer_width
        << " height=" << cfg.infer_height
        << " live-source=1 batched-push-timeout=33000 nvbuf-memory-type=0"
        << " ! nvinfer name=primary_infer config-file-path=" << cfg.nvinfer_config
        << " ! nvcompositor name=compdisplay sink_0::xpos=0 sink_0::ypos=0 sink_0::width="
        << cfg.infer_width << " sink_0::height=" << cfg.infer_height
        << " ! video/x-raw(memory:NVMM),format=NV12,width=" << cfg.display_width
        << ",height=" << cfg.display_height
        << " ! nvdsosd name=osd process-mode=1 display-bbox=1 display-text=1 display-mask=1 display-clock=1 x-clock-offset=10 y-clock-offset="
        << (cfg.infer_height + 2) << " clock-font-size=12"
        << " ! tee name=post "
        << "post. ! " << build_record_branch(cfg) << " "
        << "post. ! queue ! nvvideoconvert compute-hw=1 interpolation-method=4"
        << " ! video/x-raw(memory:NVMM),format=RGBA,width=" << cfg.display_width
        << ",height=" << cfg.display_height
        << " ! nvegltransform ! nveglglessink sync=0 qos=0 "
        << "nvcompositor name=compinfer sink_0::xpos=0 sink_0::ypos=0 sink_0::width="
        << cfg.content_width << " sink_0::height=" << cfg.content_height
        << " ! nvvideoconvert compute-hw=2 interpolation-method=4"
        << " ! video/x-raw(memory:NVMM),format=NV12,width=" << cfg.infer_width
        << ",height=" << cfg.infer_height << " ! mux.sink_0 "
        << make_common_source(cfg)
        << "compinfer.sink_0";
    return oss.str();
}

void print_usage(const char *argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --device PATH           V4L2 device, default /dev/video0\n"
        << "  --camera-width N        Camera width, default 1280\n"
        << "  --camera-height N       Camera height, default 720\n"
        << "  --fps N                 Camera FPS, default 30\n"
        << "  --exposure-auto N       V4L2 exposure_auto control value\n"
        << "  --exposure-absolute N   V4L2 exposure_absolute control value\n"
        << "  --gain N                V4L2 gain control value\n"
        << "  --extra-controls STR    Raw V4L2 extra-controls payload without leading c,\n"
        << "  --config PATH           nvinfer config file path\n"
        << "  --output PATH           Output mp4 file path\n"
        << "  --labels PATH           labels file path, default /workspace/DeepStream-Yolo-Seg/labels.txt\n"
        << "  --head1-threshold X     score threshold for head1, default 0.01\n"
        << "  --head2-threshold X     score threshold for head2, default 0.01\n"
        << "  --mask-threshold X      segmentation mask threshold, default 0.5\n"
        << "  --nms-iou X             NMS IoU threshold, default 0.45\n"
        << "  --max-detections N      max detections after NMS, default 300\n"
        << "  --bitrate N             output H264 bitrate, default 8000000\n"
        << "  --preview               Enable 960x560 preview branch\n"
        << "  --tensor-log-interval N Print detection summary every N frames, default 30\n"
        << "  --verbose               Print the generated pipeline string\n"
        << "  -h, --help              Show this help\n";
}

bool parse_args(int argc, char **argv, AppConfig &cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << std::endl;
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--device") {
            cfg.camera_device = require_value("--device");
        } else if (arg == "--camera-width") {
            cfg.camera_width = std::stoi(require_value("--camera-width"));
        } else if (arg == "--camera-height") {
            cfg.camera_height = std::stoi(require_value("--camera-height"));
        } else if (arg == "--fps") {
            cfg.camera_fps = std::stoi(require_value("--fps"));
        } else if (arg == "--exposure-auto") {
            cfg.exposure_auto = std::stoi(require_value("--exposure-auto"));
        } else if (arg == "--exposure-absolute") {
            cfg.exposure_absolute = std::stoi(require_value("--exposure-absolute"));
        } else if (arg == "--gain") {
            cfg.gain = std::stoi(require_value("--gain"));
        } else if (arg == "--extra-controls") {
            cfg.extra_controls = require_value("--extra-controls");
        } else if (arg == "--config") {
            cfg.nvinfer_config = require_value("--config");
        } else if (arg == "--output") {
            cfg.output_file = require_value("--output");
        } else if (arg == "--labels") {
            cfg.labels_file = require_value("--labels");
        } else if (arg == "--head1-threshold") {
            cfg.head1_threshold = std::stof(require_value("--head1-threshold"));
        } else if (arg == "--head2-threshold") {
            cfg.head2_threshold = std::stof(require_value("--head2-threshold"));
        } else if (arg == "--mask-threshold") {
            cfg.mask_threshold = std::stof(require_value("--mask-threshold"));
        } else if (arg == "--nms-iou") {
            cfg.nms_iou_threshold = std::stof(require_value("--nms-iou"));
        } else if (arg == "--max-detections") {
            cfg.max_detections = static_cast<unsigned int>(
                std::stoi(require_value("--max-detections")));
        } else if (arg == "--bitrate") {
            cfg.record_bitrate = std::stoi(require_value("--bitrate"));
        } else if (arg == "--preview") {
            cfg.preview = true;
        } else if (arg == "--tensor-log-interval") {
            cfg.tensor_log_interval = static_cast<unsigned int>(
                std::stoi(require_value("--tensor-log-interval")));
        } else if (arg == "--verbose") {
            cfg.verbose = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return false;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return false;
        }
    }
    return true;
}

std::string default_output_path() {
    std::time_t now = std::time(nullptr);
    std::tm tm_buf{};
    localtime_r(&now, &tm_buf);
    char filename[64];
    std::strftime(filename, sizeof(filename), "fusion_seg_%Y%m%d_%H%M%S.mp4", &tm_buf);
    return std::string("/workspace/fusion_v4l2_deepstream_app/output/") + filename;
}

std::string build_record_branch(const AppConfig &cfg) {
    std::ostringstream oss;
    oss
        << "queue ! nvv4l2h264enc bitrate=" << cfg.record_bitrate
        << " insert-sps-pps=true iframeinterval=30 idrinterval=30"
        << " ! h264parse ! qtmux faststart=true ! filesink location=" << cfg.output_file
        << " sync=0 async=0";
    return oss.str();
}

}  // namespace

int main(int argc, char **argv) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);

    AppConfig cfg;
    if (!parse_args(argc, argv, cfg)) {
        return 0;
    }

    if (cfg.output_file.empty()) {
        cfg.output_file = default_output_path();
    }
    try {
        std::filesystem::create_directories(std::filesystem::path(cfg.output_file).parent_path());
    } catch (const std::exception &e) {
        std::cerr << "[warn] Failed to create output directory: " << e.what() << std::endl;
    }

    gst_init(&argc, &argv);

    ProbeState probe_state;
    probe_state.config = cfg;
    probe_state.labels = load_labels(cfg.labels_file);
    probe_state.tensor_log_interval = cfg.tensor_log_interval;

    const std::string pipeline_description =
        cfg.preview ? build_preview_pipeline(cfg) : build_headless_pipeline(cfg);

    if (cfg.verbose) {
        std::cout << "Pipeline:\n" << pipeline_description << std::endl;
    }

    GError *error = nullptr;
    GstElement *pipeline = gst_parse_launch(pipeline_description.c_str(), &error);
    if (!pipeline) {
        std::cerr << "Failed to create pipeline: "
                  << (error ? error->message : "unknown error") << std::endl;
        if (error) {
            g_error_free(error);
        }
        return 1;
    }

    GstElement *infer = gst_bin_get_by_name(GST_BIN(pipeline), "primary_infer");
    if (!infer) {
        std::cerr << "Failed to locate nvinfer element in pipeline" << std::endl;
        gst_object_unref(pipeline);
        return 1;
    }

    GstPad *infer_src_pad = gst_element_get_static_pad(infer, "src");
    if (infer_src_pad) {
        gst_pad_add_probe(infer_src_pad, GST_PAD_PROBE_TYPE_BUFFER, infer_src_probe,
                          &probe_state, nullptr);
        gst_object_unref(infer_src_pad);
    }
    gst_object_unref(infer);

    GstStateChangeReturn state_ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (state_ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "Failed to set pipeline to PLAYING" << std::endl;
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
        return 1;
    }

    GstBus *bus = gst_element_get_bus(pipeline);
    bool running = true;
    bool got_eos = false;

    while (running && !g_stop) {
        GstMessage *msg = gst_bus_timed_pop_filtered(
            bus, 200 * GST_MSECOND,
            static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS | GST_MESSAGE_WARNING));

        if (!msg) {
            continue;
        }

        switch (GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_ERROR: {
                GError *err = nullptr;
                gchar *dbg = nullptr;
                gst_message_parse_error(msg, &err, &dbg);
                std::cerr << "[ERROR] "
                          << GST_OBJECT_NAME(msg->src) << ": "
                          << (err ? err->message : "unknown error") << std::endl;
                if (dbg) {
                    std::cerr << "  debug: " << dbg << std::endl;
                }
                if (err) {
                    g_error_free(err);
                }
                g_free(dbg);
                running = false;
                break;
            }
            case GST_MESSAGE_WARNING: {
                GError *err = nullptr;
                gchar *dbg = nullptr;
                gst_message_parse_warning(msg, &err, &dbg);
                std::cerr << "[WARN] "
                          << GST_OBJECT_NAME(msg->src) << ": "
                          << (err ? err->message : "unknown warning") << std::endl;
                if (dbg) {
                    std::cerr << "  debug: " << dbg << std::endl;
                }
                if (err) {
                    g_error_free(err);
                }
                g_free(dbg);
                break;
            }
            case GST_MESSAGE_EOS:
                std::cout << "Received EOS, shutting down." << std::endl;
                running = false;
                got_eos = true;
                break;
            default:
                break;
        }
        gst_message_unref(msg);
    }

    if (!got_eos) {
        std::cout << "Stopping pipeline, sending EOS..." << std::endl;
        gst_element_send_event(pipeline, gst_event_new_eos());
        const auto eos_wait_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < eos_wait_deadline) {
            GstMessage *msg = gst_bus_timed_pop_filtered(
                bus, 500 * GST_MSECOND,
                static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
            if (!msg) {
                continue;
            }

            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS) {
                std::cout << "EOS received after stop request." << std::endl;
                gst_message_unref(msg);
                break;
            }

            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                GError *err = nullptr;
                gchar *dbg = nullptr;
                gst_message_parse_error(msg, &err, &dbg);
                std::cerr << "[ERROR] "
                          << GST_OBJECT_NAME(msg->src) << ": "
                          << (err ? err->message : "unknown error") << std::endl;
                if (dbg) {
                    std::cerr << "  debug: " << dbg << std::endl;
                }
                if (err) {
                    g_error_free(err);
                }
                g_free(dbg);
                gst_message_unref(msg);
                break;
            }
            gst_message_unref(msg);
        }
    }

    gst_element_set_state(pipeline, GST_STATE_NULL);

    const auto end_time = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time - probe_state.start_time).count();
    const double avg_fps = (elapsed > 0.0)
        ? static_cast<double>(probe_state.frame_counter) / elapsed
        : 0.0;
    const double avg_decode_ms = probe_state.frame_counter > 0
        ? static_cast<double>(probe_state.total_decode_ms) / static_cast<double>(probe_state.frame_counter)
        : 0.0;
    std::cout << "[summary] frames=" << probe_state.frame_counter
              << " avg_fps=" << std::fixed << std::setprecision(2) << avg_fps
              << " avg_decode_ms=" << std::fixed << std::setprecision(2) << avg_decode_ms
              << " total_detections=" << probe_state.total_detection_count
              << " output=" << cfg.output_file
              << std::endl;

    gst_object_unref(bus);
    gst_object_unref(pipeline);
    return 0;
}
