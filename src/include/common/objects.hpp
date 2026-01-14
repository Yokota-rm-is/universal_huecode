#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct Object
{
    // corners: TL,TR,BR,BL order
    std::vector<cv::Point2f> corners;
    std::vector<std::string> data;
};

using Objects = std::vector<Object>;
