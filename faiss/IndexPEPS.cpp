#include <faiss/IndexPEPS.h>

#include <limits>

namespace {

/**
 * Calculates the squared distances to the d * 2 polar entry points and places them into
 * res. res[i*2 + j] has distance to ith vector and positive if j=0, negative if j=1.
 */
void fill_peps_distances_squared(const float* vec, float* res, int d) {
    float total = 0.0;
    for(int i = 0; i < d; i++) {
        total += vec[i] * vec[i];
    }

    for(int i = 0; i < d; i++) {
        float sub = vec[i] * vec[i];
        float v_pos = vec[i] - 1.0;
        float v_neg = vec[i] + 1.0;
        res[2*i] = total - sub + (v_pos * v_pos);
        res[2*i + 1] = total - sub + (v_neg * v_neg);
    }
}

int get_peps_ep_index(const float* v, int d) {
    float largest = 0.0;
    int index = 0;

    for(int i = 0; i < d; i++) {
        if(std::abs(v[i]) > largest) {
            largest = std::abs(v[i]);
            index = i;
        }
    }

    return index*2 + (v[index] < 0 ? 1 : 0);
}

}

namespace faiss {

NaiveEntryPointSelector::NaiveEntryPointSelector(int d, HNSW& hnsw) 
    : d{d}, hnsw{hnsw}, entry_point_ids(2*d, -1), 
    entry_point_distances(2*d, std::numeric_limits<float>::max()) {}

int32_t NaiveEntryPointSelector::operator()(const float* v) const {
    int32_t candidate = entry_point_ids.at(get_peps_ep_index(v, d));
    return candidate;
}

void NaiveEntryPointSelector::add(const float* x, int n, int32_t id0) {
    float* distances = new float[2*d];
    for(int i = 0; i < n; i++) {
        fill_peps_distances_squared(x + d*i, distances, d);
        for(int j = 0; j < 2*d; j++) {
            if(distances[j] < entry_point_distances[j]) {
                entry_point_ids[j] = id0 + i;
                entry_point_distances[j] = distances[j];
            }
        }
    }

    delete[] distances;
}

void IndexPEPS::add(idx_t n, const float* x) {
    eps->add(x, n, ntotal);
    IndexHNSWFlat::add(n, x);
}

IndexPEPS::IndexPEPS() {}

IndexPEPS::IndexPEPS(int d_in, int M, MetricType metric) 
    : IndexHNSWFlat(d_in, M, metric) {
        eps = std::make_unique<NaiveEntryPointSelector>(d, hnsw);
        hnsw.ep_selector = eps.get();
    }

}