#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <zbar.h>
#include <string>
#include <vector>
#include <stdexcept>

#include "common/objects.hpp"

namespace detectors
{

    // Order points as TL, TR, BR, BL
    inline std::vector<cv::Point2f> orderTLTRBRBL(const std::vector<cv::Point2f> &pts4)
    {
        if (pts4.size() != 4)
            return {};
        std::vector<cv::Point2f> o(4);

        auto sum = [](const cv::Point2f &p)
        { return p.x + p.y; };
        auto diff = [](const cv::Point2f &p)
        { return p.x - p.y; };

        int tl = 0, br = 0, tr = 0, bl = 0;
        for (int i = 1; i < 4; i++)
        {
            if (sum(pts4[i]) < sum(pts4[tl]))
                tl = i;
            if (sum(pts4[i]) > sum(pts4[br]))
                br = i;
            if (diff(pts4[i]) > diff(pts4[tr]))
                tr = i;
            if (diff(pts4[i]) < diff(pts4[bl]))
                bl = i;
        }
        o[0] = pts4[tl];
        o[1] = pts4[tr];
        o[2] = pts4[br];
        o[3] = pts4[bl];
        return o;
    }

    inline Objects detectQrZBar(const cv::Mat &bgr)
    {
        Objects out;
        if (bgr.empty())
            return out;

        cv::Mat gray;
        if (bgr.channels() == 3)
            cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        else
            gray = bgr;

        zbar::ImageScanner scanner;
        scanner.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 1);

        zbar::Image zimg(gray.cols, gray.rows, "Y800", gray.data, gray.cols * gray.rows);

        int n = scanner.scan(zimg);
        if (n <= 0)
            return out;

        for (auto sym = zimg.symbol_begin(); sym != zimg.symbol_end(); ++sym)
        {
            if (sym->get_type() != zbar::ZBAR_QRCODE)
                continue;

            std::string data = sym->get_data();

            std::vector<cv::Point2f> pts;
            pts.reserve(sym->get_location_size());
            for (int i = 0; i < sym->get_location_size(); ++i)
            {
                pts.emplace_back((float)sym->get_location_x(i), (float)sym->get_location_y(i));
            }

            std::vector<cv::Point2f> corners;
            if (pts.size() == 4)
            {
                corners = orderTLTRBRBL(pts);
            }
            else if (pts.size() >= 4)
            {
                cv::RotatedRect rr = cv::minAreaRect(pts);
                cv::Point2f box[4];
                rr.points(box);
                std::vector<cv::Point2f> boxv = {box[0], box[1], box[2], box[3]};
                corners = orderTLTRBRBL(boxv);
            }
            else
            {
                corners = {};
            }

            Object obj;
            obj.corners = std::move(corners);
            obj.data = {data};
            out.push_back(std::move(obj));
        }

        zimg.set_data(nullptr, 0);
        return out;
    }

} // namespace detectors
