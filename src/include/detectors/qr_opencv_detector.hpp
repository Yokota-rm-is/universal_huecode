#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <string>
#include <vector>

#include "common/objects.hpp"

namespace detectors
{

    inline Objects detectQrOpenCV(const cv::Mat &bgr)
    {
        Objects out;
        if (bgr.empty())
            return out;

        cv::QRCodeDetector qrd;

        // detectAndDecodeMulti を優先（なければ single）
        std::vector<std::string> decoded;
        std::vector<std::vector<cv::Point2f>> points;
        std::vector<cv::Mat> straight;

        bool ok = qrd.detectAndDecodeMulti(bgr, decoded, points, straight);
        if (ok)
        {
            for (size_t i = 0; i < decoded.size(); ++i)
            {
                if (points[i].size() != 4)
                    continue;
                Object obj;
                obj.corners = points[i]; // OpenCV QR: TL,TR,BR,BL
                obj.data = {decoded[i]};
                out.push_back(std::move(obj));
            }
            return out;
        }

        // fallback: single
        std::vector<cv::Point2f> pts;
        std::string s = qrd.detectAndDecode(bgr, pts);
        if (!s.empty() && pts.size() == 4)
        {
            Object obj;
            obj.corners = pts;
            obj.data = {s};
            out.push_back(std::move(obj));
        }
        return out;
    }

} // namespace detectors
