#pragma once

#include <faiss/IndexHNSW.h>

#include <vector>

namespace faiss {

struct NaiveEntryPointSelector : EntryPointSelector {
private:
    int d;
    HNSW& hnsw;
    std::vector<int32_t> entry_point_ids;
    std::vector<float> entry_point_distances;

public:
    NaiveEntryPointSelector(int d, HNSW& hnsw);
    int32_t operator()(const float* v) const override;
    void add(const float* x, int n, int32_t id0);
    int debug_get_ep(int i) {return entry_point_ids[i];}
};

struct IndexPEPS : IndexHNSWFlat {
private:
    std::unique_ptr<NaiveEntryPointSelector> eps;
public:
    IndexPEPS();
    IndexPEPS(int d_in, int M, MetricType metric = METRIC_L2);

    int debug_get_ep(int i) {return eps->debug_get_ep(i);};
    void add(idx_t n, const float* x) override;
};
}
