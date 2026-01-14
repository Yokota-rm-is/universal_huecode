#ifndef RECOGNITION_RECOGNITION_UNIVERSAL_HUECODE_HPP
#define RECOGNITION_RECOGNITION_UNIVERSAL_HUECODE_HPP

#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "common/objects.hpp"
#include "common/result_object.hpp"
#include "common/detection_visualizer.hpp"
#include "detectors/aruco_detector.hpp"
#include "detectors/qr_zbar_detector.hpp"
#include "recognition/online_remapper.hpp"

namespace recognition
{
    namespace detail
    {
        inline double elapsedSec(std::chrono::steady_clock::time_point t0,
                                 std::chrono::steady_clock::time_point t1)
        {
            return std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        }
    } // namespace detail

    // Universal HueCode:
    // 1) ArUco detect (on original image)
    // 2) clip by detected ArUco corners
    // 3) online remap (bin only)
    // 4) QR(ZBar) detect on bin image
    inline void recognizeUniversalHuecode(
        const cv::Mat &src_bgr,
        ResultObject &result,
        int side_pixels,
        int margin_size,
        huecode::OnlineRemapperParams p,
        bool clustering_separately,
        const std::string &result_image_path = "",
        cv::aruco::PREDEFINED_DICTIONARY_NAME aruco_dictionary = cv::aruco::DICT_4X4_50,
        std::vector<cv::Vec3b> rgb_palette = {})
    {
        // Reset results
        result.aruco.clear();
        result.qr.clear();
        result.duration_aruco = 0.0;
        result.duration_clip = 0.0;
        result.duration_remap = 0.0;
        result.duration_qr = 0.0;

        if (src_bgr.empty())
        {
            return;
        }
        if (src_bgr.type() != CV_8UC3 && src_bgr.type() != CV_8UC1)
        {
            throw std::runtime_error("recognizeUniversalHuecode: src must be CV_8UC3 or CV_8UC1");
        }
        if (side_pixels <= 0)
        {
            throw std::runtime_error("recognizeUniversalHuecode: side_pixels must be > 0");
        }
        if (margin_size < 0)
        {
            throw std::runtime_error("recognizeUniversalHuecode: margin_size must be >= 0");
        }

        // Use the previous hard-coded palette as default to keep backward behavior.
        if (rgb_palette.empty())
        {
            rgb_palette = {
                {0, 0, 255},
                {97, 0, 0},
                {157, 255, 255},
                {255, 255, 0},
            };
        }
        p.rgb_palette = rgb_palette;

        if ((int)p.rgb_palette.size() < p.n_clusters)
        {
            throw std::runtime_error("recognizeUniversalHuecode: rgb_palette.size() < n_clusters");
        }

        cv::Mat bgr;
        if (src_bgr.channels() == 1)
            cv::cvtColor(src_bgr, bgr, cv::COLOR_GRAY2BGR);
        else
            bgr = src_bgr;

        Objects aruco_objs;
        Objects qr_objs;
        cv::Mat H_inv;

        auto save_result_image = [&]()
        {
            if (!result_image_path.empty())
            {
                common::visualization::saveDetectionResultImage(
                    bgr, aruco_objs, qr_objs, H_inv, result_image_path);
            }
        };

        // --------------------------------------
        // 1) ArUco
        // --------------------------------------
        {
            const auto t0 = std::chrono::steady_clock::now();

            aruco_objs = detectors::detectAruco(
                bgr,
                aruco_dictionary,
                nullptr);

            const auto t1 = std::chrono::steady_clock::now();
            result.duration_aruco = detail::elapsedSec(t0, t1);

            if (!aruco_objs.empty() && !aruco_objs.front().data.empty())
                result.aruco = aruco_objs.front().data.front();
            else
                result.aruco = "";
        }

        if (aruco_objs.empty() || aruco_objs.front().corners.size() != 4)
        {
            save_result_image();
            return;
        }

        // --------------------------------------
        // 2) Clip
        // --------------------------------------
        cv::Mat clipped;
        cv::Mat H;
        {
            const auto t0 = std::chrono::steady_clock::now();

            const std::vector<cv::Point2f> src_corners = aruco_objs.front().corners;
            const int minv = margin_size;
            const int maxv = margin_size + side_pixels;

            const std::vector<cv::Point2f> dst_corners{
                cv::Point2f((float)minv, (float)minv),
                cv::Point2f((float)maxv, (float)minv),
                cv::Point2f((float)maxv, (float)maxv),
                cv::Point2f((float)minv, (float)maxv)};

            H = cv::getPerspectiveTransform(src_corners, dst_corners);
            H_inv = H.inv();

            const cv::Size dst_size(side_pixels + 2 * margin_size, side_pixels + 2 * margin_size);
            clipped = cv::Mat(dst_size, bgr.type(), cv::Scalar::all(127));

            cv::warpPerspective(
                bgr, clipped, H, dst_size,
                cv::INTER_LINEAR,
                cv::BORDER_CONSTANT, cv::Scalar::all(255));

            const auto t1 = std::chrono::steady_clock::now();
            result.duration_clip = detail::elapsedSec(t0, t1);
        }

        // --------------------------------------
        // 3) Online remap (bin only)
        // --------------------------------------
        cv::Mat bin_image;
        {
            const auto t0 = std::chrono::steady_clock::now();

            if (clustering_separately)
                bin_image = huecode::remapWithImageSeparately(clipped, clipped, p);
            else
                bin_image = huecode::remapWithImage(clipped, clipped, p);

            const auto t1 = std::chrono::steady_clock::now();
            result.duration_remap = detail::elapsedSec(t0, t1);
        }

        if (bin_image.empty())
        {
            save_result_image();
            return;
        }

        if (bin_image.type() != CV_8UC1)
        {
            cv::Mat tmp;
            if (bin_image.channels() == 3)
                cv::cvtColor(bin_image, tmp, cv::COLOR_BGR2GRAY);
            else
                bin_image.convertTo(tmp, CV_8U);
            bin_image = tmp;
        }

        // --------------------------------------
        // 4) QR(ZBar)
        // --------------------------------------
        {
            const auto t0 = std::chrono::steady_clock::now();

            qr_objs = detectors::detectQrZBar(bin_image);

            const auto t1 = std::chrono::steady_clock::now();
            result.duration_qr = detail::elapsedSec(t0, t1);

            if (!qr_objs.empty() && !qr_objs.front().data.empty())
                result.qr = qr_objs.front().data.front();
            else
                result.qr = "";
        }

        save_result_image();
    }

} // namespace recognition

#endif // RECOGNITION_RECOGNITION_UNIVERSAL_HUECODE_HPP
