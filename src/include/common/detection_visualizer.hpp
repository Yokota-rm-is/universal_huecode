#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "common/objects.hpp"

namespace common::visualization
{
    inline void ensureParentDir(const std::string& path)
    {
        namespace fs = std::filesystem;
        fs::path p(path);
        if (p.has_parent_path())
        {
            fs::create_directories(p.parent_path());
        }
    }

    inline cv::Rect clampRectToImage(const cv::Rect& r, const cv::Size& sz)
    {
        cv::Rect imgRect(0, 0, sz.width, sz.height);
        return r & imgRect;
    }

    inline void putTextOutlined(
        cv::Mat& img,
        const std::string& text,
        const cv::Point& org,
        int fontFace,
        double fontScale,
        const cv::Scalar& color_bgr,
        int thickness)
    {
        const int outline = thickness + 3;
        cv::putText(img, text, org, fontFace, fontScale, cv::Scalar(0, 0, 0), outline, cv::LINE_AA);
        cv::putText(img, text, org, fontFace, fontScale, color_bgr, thickness, cv::LINE_AA);
    }

    inline bool getBoundingRectFromCorners(const std::vector<cv::Point2f>& corners, cv::Rect& out_rect)
    {
        if (corners.empty()) return false;
        out_rect = cv::boundingRect(corners);
        return true;
    }

    inline std::vector<cv::Point2f> transformIfNeeded(
        const std::vector<cv::Point2f>& corners,
        const cv::Mat& H_inv)
    {
        if (corners.empty()) return {};
        if (H_inv.empty()) return corners;

        std::vector<cv::Point2f> out;
        cv::perspectiveTransform(corners, out, H_inv);
        return out;
    }

    inline void drawTextBackground(cv::Mat& img, const cv::Rect& bg)
    {
        cv::rectangle(img, bg, cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA);
    }

    inline cv::Point computeLabelOriginOutsideRect(
        const cv::Rect& rect,
        const cv::Size& img_size,
        const cv::Size& text_size,
        int baseline,
        bool place_above,
        int padding = 6)
    {
        int x = rect.x;

        auto orgAbove = [&]() {
            int y = rect.y - padding;
            return cv::Point(x + padding, y);
        };
        auto orgBelow = [&]() {
            int y = rect.y + rect.height + text_size.height + baseline + padding;
            return cv::Point(x + padding, y);
        };

        cv::Point org = place_above ? orgAbove() : orgBelow();

        if (place_above)
        {
            if (org.y - text_size.height - baseline - padding < 0)
            {
                org = orgBelow();
            }
        }
        else
        {
            if (org.y + padding > img_size.height)
            {
                org = orgAbove();
            }
        }

        org.x = std::max(padding, std::min(org.x, img_size.width - padding));
        org.y = std::max(text_size.height + baseline + padding,
                         std::min(org.y, img_size.height - padding));

        return org;
    }

    inline cv::Rect computeTextBackgroundRect(
        const cv::Point& org,
        const cv::Size& text_size,
        int baseline,
        const cv::Size& img_size,
        int padding = 6)
    {
        cv::Rect bg(org.x - padding,
                    org.y - text_size.height - baseline - padding,
                    text_size.width + 2 * padding,
                    text_size.height + baseline + 2 * padding);
        return clampRectToImage(bg, img_size);
    }

    inline void saveDetectionResultImage(
        const cv::Mat& bgr,
        const Objects& primary_objs,
        const Objects& secondary_objs,
        const cv::Mat& H_inv,             // clipped->original
        const std::string& save_path)
    {
        if (bgr.empty() || save_path.empty()) return;

        cv::Mat vis = bgr.clone();

        const int fontFace = cv::FONT_HERSHEY_SIMPLEX;

        const double fontScale = 5;
        const int text_thickness = 10;
        const int box_thickness = 48;

        const cv::Scalar green(0, 255, 0); // primary marker
        const cv::Scalar red(0, 0, 255);   // seconday marker

        // ---- primary marker ----
        bool primary_found = false;
        cv::Rect primary_rect;
        std::string primary_text = "primary: (not detected)";
        if (!primary_objs.empty())
        {
            const auto& o = primary_objs.front();
            if (!o.data.empty() && !o.data.front().empty())
            {
                primary_text = "primary: " + o.data.front();
            }
            if (o.corners.size() >= 4 && getBoundingRectFromCorners(o.corners, primary_rect))
            {
                primary_rect = clampRectToImage(primary_rect, vis.size());
                if (primary_rect.area() > 0) primary_found = true;
            }
        }

        // ---- secondary marker ----
        bool secondary_found = false;
        cv::Rect secondary_rect;
        std::string secondary_text = "secondary: (not detected)";
        if (!secondary_objs.empty())
        {
            const auto& o = secondary_objs.front();
            if (!o.data.empty() && !o.data.front().empty())
            {
                secondary_text = "secondary: " + o.data.front();
            }
            if (o.corners.size() >= 4)
            {
                const auto corners_org = transformIfNeeded(o.corners, H_inv);
                if (getBoundingRectFromCorners(corners_org, secondary_rect))
                {
                    secondary_rect = clampRectToImage(secondary_rect, vis.size());
                    if (secondary_rect.area() > 0) secondary_found = true;
                }
            }
        }

        if (primary_found) cv::rectangle(vis, primary_rect, green, box_thickness, cv::LINE_AA);
        if (secondary_found)    cv::rectangle(vis, secondary_rect, red,   box_thickness / 2, cv::LINE_AA);

        auto drawLabelOutside = [&](const std::string& text,
                                    const cv::Rect& rect,
                                    const cv::Scalar& color,
                                    bool place_above)
        {
            int baseline = 0;
            const cv::Size ts = cv::getTextSize(text, fontFace, fontScale, text_thickness, &baseline);

            const cv::Point org = computeLabelOriginOutsideRect(
                rect, vis.size(), ts, baseline, place_above, /*padding=*/6);

            putTextOutlined(vis, text, org, fontFace, fontScale, color, text_thickness);
        };

        if (primary_found)
        {
            drawLabelOutside(primary_text, primary_rect, green, /*place_above=*/true);
        }

        if (secondary_found)
        {
            drawLabelOutside(secondary_text, secondary_rect, red, /*place_above=*/false);
        }

        ensureParentDir(save_path);
        cv::imwrite(save_path, vis);
    }
} // namespace common::visualization
