#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <iostream>

namespace huecode
{

    struct OnlineRemapperParams
    {
        int n_clusters = 4; // must be even

        int n_blacks = 2; // clusters on black side
        // n_whites = n_clusters - n_blacks

        // kmeans
        int max_iter = 10;
        double eps = 1.0;
        int attempts = 1;

        // morphology
        int iter_dilate = 1;
        int iter_erode = 1;

        // RGB palette (input as RGB, internally converted to BGR)
        std::vector<cv::Vec3b> rgb_palette = {
            {255, 0, 0},
            {0, 255, 0},
            {0, 0, 255},
            {255, 255, 255},
        };
    };

    namespace detail
    {

        inline void requireBgr8(const cv::Mat &bgr, const char *name)
        {
            if (bgr.empty())
                throw std::runtime_error(std::string(name) + " is empty");
            if (bgr.type() != CV_8UC3)
                throw std::runtime_error(std::string(name) + " must be CV_8UC3 (bgr8)");
        }

        inline void buildPaletteBgr(
            const OnlineRemapperParams &p,
            std::vector<cv::Vec3b> &palette_bgr)
        {
            const int K = p.n_clusters;
            if ((int)p.rgb_palette.size() < K)
                throw std::runtime_error("rgb_palette.size() < n_clusters");

            palette_bgr.resize(K);
            for (int i = 0; i < K; ++i)
            {
                const cv::Vec3b rgb = p.rgb_palette[(size_t)i];
                palette_bgr[i] = cv::Vec3b(rgb[2], rgb[1], rgb[0]); // RGB -> BGR
            }
        }

    } // namespace detail

    // ============================================================
    // RemapImageWithImage equivalent (bin_image only)
    // ============================================================
    inline cv::Mat remapWithImageSeparately(
        const cv::Mat &clustering_bgr,
        const cv::Mat &remapping_bgr,
        const OnlineRemapperParams &p = OnlineRemapperParams())
    {
        const int K = p.n_clusters;
        if (K <= 0 || K % 2 != 0)
            throw std::runtime_error("n_clusters must be positive and even");
        if (p.n_blacks <= 0 || p.n_blacks >= K)
            throw std::runtime_error("invalid n_blacks");

        const int n_whites = K - p.n_blacks;

        detail::requireBgr8(clustering_bgr, "clustering_bgr");
        detail::requireBgr8(remapping_bgr, "remapping_bgr");

        const int crows = clustering_bgr.rows;
        const int ccols = clustering_bgr.cols;
    
        const int rrows = remapping_bgr.rows;
        const int rcols = remapping_bgr.cols;

        // --- grayscale + Otsu ---
        cv::Mat gray;
        cv::cvtColor(clustering_bgr, gray, cv::COLOR_BGR2GRAY);

        cv::Mat otsu_binary;
        cv::threshold(gray, otsu_binary, 0, 255,
                      cv::THRESH_BINARY | cv::THRESH_OTSU);

        // --- morphology ---
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat bin_dilate, bin_erode;
        cv::dilate(otsu_binary, bin_dilate, kernel, {}, p.iter_dilate);
        cv::erode(otsu_binary, bin_erode, kernel, {}, p.iter_erode);

        // --- count samples ---
        int n_black = 0, n_white = 0;
        for (int y = 0; y < crows; ++y)
        {
            const uchar *pd = bin_dilate.ptr<uchar>(y);
            const uchar *pe = bin_erode.ptr<uchar>(y);
            for (int x = 0; x < ccols; ++x)
            {
                n_black += (pd[x] == 0);
                n_white += (pe[x] == 255);
            }
        }
        if (n_black < p.n_blacks || n_white < n_whites)
            throw std::runtime_error("not enough samples for kmeans");

        // --- build samples ---
        cv::Mat black_samples(n_black, 3, CV_32F);
        cv::Mat white_samples(n_white, 3, CV_32F);

        int ib = 0, iw = 0;
        for (int y = 0; y < crows; ++y)
        {
            const uchar *pd = bin_dilate.ptr<uchar>(y);
            const uchar *pe = bin_erode.ptr<uchar>(y);
            const cv::Vec3b *src = clustering_bgr.ptr<cv::Vec3b>(y);

            for (int x = 0; x < ccols; ++x)
            {
                const cv::Vec3b v = src[x];
                if (pd[x] == 0)
                {
                    float *d = black_samples.ptr<float>(ib++);
                    d[0] = v[0];
                    d[1] = v[1];
                    d[2] = v[2];
                }
                else if (pe[x] == 255)
                {
                    float *d = white_samples.ptr<float>(iw++);
                    d[0] = v[0];
                    d[1] = v[1];
                    d[2] = v[2];
                }
            }
        }

        // --- kmeans ---
        // cv::theRNG().state = (uint64_t)p.rng_seed;
        cv::TermCriteria tc(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER,
                            p.max_iter, p.eps);

        cv::Mat black_centers, white_centers;
        cv::Mat black_labels, white_labels;

        cv::kmeans(black_samples, p.n_blacks, black_labels,
                   tc, p.attempts, cv::KMEANS_PP_CENTERS, black_centers);
        cv::kmeans(white_samples, n_whites, white_labels,
                   tc, p.attempts, cv::KMEANS_PP_CENTERS, white_centers);

        cv::Mat centers;
        cv::vconcat(black_centers, white_centers, centers); // K x 3 (float)

        // --- palette ---
        std::vector<cv::Vec3b> palette_bgr;
        detail::buildPaletteBgr(p, palette_bgr);

        // --- remap pairing ---
        std::vector<int> remap_idx(K);
        std::iota(remap_idx.begin(), remap_idx.end(), 0);

        auto palRow32F = [&](int idx) -> cv::Mat
        {
            cv::Mat m(1, 3, CV_32F);
            m.at<float>(0, 0) = static_cast<float>(palette_bgr[idx][0]);
            m.at<float>(0, 1) = static_cast<float>(palette_bgr[idx][1]);
            m.at<float>(0, 2) = static_cast<float>(palette_bgr[idx][2]);
            return m;
        };

        for (int i = 0; i < K; i += 2)
        {
            const cv::Mat p0 = palRow32F(i);
            const cv::Mat p1 = palRow32F(i + 1);

            const double a =
                cv::norm(centers.row(i) - p0) +
                cv::norm(centers.row(i + 1) - p1);
            const double b =
                cv::norm(centers.row(i) - p1) +
                cv::norm(centers.row(i + 1) - p0);

            if (a > b)
                std::swap(remap_idx[i], remap_idx[i + 1]);
        }

        // --- bin value per cluster ---
        std::vector<uchar> bin_val(K);
        for (int k = 0; k < K; ++k)
            bin_val[k] = (remap_idx[k] % 2 == 0 ? 0 : 255);

        // --- assign per pixel ---
        cv::Mat bin_image(remapping_bgr.size(), CV_8UC1);

        std::vector<cv::Vec3f> c(K);
        for (int k = 0; k < K; ++k)
        {
            c[k][0] = centers.at<float>(k, 0);
            c[k][1] = centers.at<float>(k, 1);
            c[k][2] = centers.at<float>(k, 2);
        }

        if (p.iter_dilate == 0 and p.iter_erode == 0 and clustering_bgr.size() == remapping_bgr.size())
        {
            int ib = 0, iw = 0;
            for (int y = 0; y < crows; ++y)
            {
                const uchar *pd = bin_dilate.ptr<uchar>(y);
                uchar *out = bin_image.ptr<uchar>(y);

                for (int x = 0; x < ccols; ++x)
                {
                    const int label = (pd[x] == 0) ? black_labels.at<int>(ib++) : white_labels.at<int>(iw++) + 2;
                    out[x] = bin_val[label];
                }
            }
        }
        else
        {
            for (int y = 0; y < rrows; ++y)
            {
                const cv::Vec3b *src = remapping_bgr.ptr<cv::Vec3b>(y);
                uchar *out = bin_image.ptr<uchar>(y);

                for (int x = 0; x < rcols; ++x)
                {
                    int best_k = 0;
                    int best_d = std::numeric_limits<int>::max();

                    for (int k = 0; k < K; ++k)
                    {
                        const int db = src[x][0] - (int)c[k][0];
                        const int dg = src[x][1] - (int)c[k][1];
                        const int dr = src[x][2] - (int)c[k][2];
                        const int d = db * db + dg * dg + dr * dr;
                        if (d < best_d)
                        {
                            best_d = d;
                            best_k = k;
                        }
                    }
                    out[x] = bin_val[best_k];
                }
            }
        }

        return bin_image;
    }

    inline cv::Mat remapWithImage(
        const cv::Mat &clustering_bgr,
        const cv::Mat &remapping_bgr,
        const OnlineRemapperParams &p = OnlineRemapperParams())
    {
        const int K = p.n_clusters;
        if (K <= 0 || K % 2 != 0)
            throw std::runtime_error("n_clusters must be positive and even");

        detail::requireBgr8(clustering_bgr, "clustering_bgr");
        detail::requireBgr8(remapping_bgr, "remapping_bgr");

        const int crows = clustering_bgr.rows;
        const int ccols = clustering_bgr.cols;
    
        const int rrows = remapping_bgr.rows;
        const int rcols = remapping_bgr.cols;

        // --- build samples ---
        const int n_samples = crows * ccols;
        cv::Mat samples(n_samples, 3, CV_32F);

        int i = 0;
        for (int y = 0; y < crows; ++y)
        {
            const cv::Vec3b *src = clustering_bgr.ptr<cv::Vec3b>(y);

            for (int x = 0; x < ccols; ++x)
            {
                const cv::Vec3b v = src[x];
                float *d = samples.ptr<float>(i++);
                d[0] = v[0];
                d[1] = v[1];
                d[2] = v[2];
            }
        }

        // --- kmeans ---
        // cv::theRNG().state = (uint64_t)p.rng_seed;
        cv::TermCriteria tc(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER,
                            p.max_iter, p.eps);

        cv::Mat centers, labels;

        cv::kmeans(samples, K, labels,
                   tc, p.attempts, cv::KMEANS_PP_CENTERS, centers);

        // --- palette ---
        std::vector<cv::Vec3b> palette_bgr;
        detail::buildPaletteBgr(p, palette_bgr);

        std::vector<cv::Vec3b> c(K);
        for (int k = 0; k < K; ++k)
        {
            c[k][0] = static_cast<int>(centers.at<float>(k, 0));
            c[k][1] = static_cast<int>(centers.at<float>(k, 1));
            c[k][2] = static_cast<int>(centers.at<float>(k, 2));
        }

        std::vector<int> remap_idx(K), tmp_idx(K);
        std::iota(remap_idx.begin(), remap_idx.end(), 0);
        std::iota(tmp_idx.begin(), tmp_idx.end(), 0);
        int best_d = std::numeric_limits<int>::max();

        do 
        {
            int d = 0;

            for (int k = 0; k < K; k++) 
            {
                const int db = (int)c[k][0] - (int)palette_bgr[tmp_idx[k]][0];
                const int dg = (int)c[k][1] - (int)palette_bgr[tmp_idx[k]][1];
                const int dr = (int)c[k][2] - (int)palette_bgr[tmp_idx[k]][2];

                d += db * db + dg * dg + dr * dr;
            }

            if (d < best_d) {
                best_d = d;
                remap_idx = tmp_idx;
            }

        } while (std::next_permutation(tmp_idx.begin(), tmp_idx.end()));

        // --- bin value per cluster ---
        std::vector<uchar> bin_val(K);
        for (int k = 0; k < K; ++k)
            bin_val[k] = (remap_idx[k] % 2 == 0 ? 0 : 255);
        
        // --- assign per pixel ---
        cv::Mat bin_image(remapping_bgr.size(), CV_8UC1);

        if (clustering_bgr.size() == remapping_bgr.size())
        {
            int i = 0;
            for (int y = 0; y < crows; ++y)
            {
                uchar *out = bin_image.ptr<uchar>(y);

                for (int x = 0; x < ccols; ++x)
                {
                    const int label = labels.at<int>(i++);
                    out[x] = bin_val[label];
                }
            }
        }
        else
        {
            for (int y = 0; y < rrows; ++y)
            {
                const cv::Vec3b *src = remapping_bgr.ptr<cv::Vec3b>(y);
                uchar *out = bin_image.ptr<uchar>(y);

                for (int x = 0; x < rcols; ++x)
                {
                    int best_k = 0;
                    int best_d = std::numeric_limits<int>::max();

                    for (int k = 0; k < K; ++k)
                    {
                        const int db = src[x][0] - (int)c[k][0];
                        const int dg = src[x][1] - (int)c[k][1];
                        const int dr = src[x][2] - (int)c[k][2];
                        const int d = db * db + dg * dg + dr * dr;
                        if (d < best_d)
                        {
                            best_d = d;
                            best_k = k;
                        }
                    }
                    out[x] = bin_val[best_k];
                }
            }
        }

        return bin_image;
    }

} // namespace huecode
