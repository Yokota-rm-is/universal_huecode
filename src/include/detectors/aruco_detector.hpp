#pragma once
#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include "common/objects.hpp"

namespace detectors
{
    
    inline Objects detectAruco(
        const cv::Mat &bgr,
        cv::aruco::PREDEFINED_DICTIONARY_NAME dict_type = cv::aruco::DICT_4X4_50,
        const cv::Ptr<cv::aruco::DetectorParameters> &params_opt = nullptr)
    {
        Objects out;
        if (bgr.empty())
            return out;

        cv::Mat gray;
        if (bgr.channels() == 3)
            cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        else
            gray = bgr;

        auto dict = cv::aruco::getPredefinedDictionary(dict_type);

        cv::Ptr<cv::aruco::DetectorParameters> params =
            (params_opt ? params_opt : cv::aruco::DetectorParameters::create());

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners, rejected;

        cv::aruco::detectMarkers(gray, dict, corners, ids, params, rejected);

        for (size_t i = 0; i < corners.size(); ++i)
        {
            if (corners[i].size() != 4)
                continue;
            Object obj;
            obj.corners = corners[i];            // TL,TR,BR,BL
            obj.data = {std::to_string(ids[i])}; // ids[0]
            out.push_back(std::move(obj));
        }
        return out;
    }

} // namespace detectors
