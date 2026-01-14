#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>

#include "common/result_object.hpp"
#include "recognition/online_remapper.hpp"
#include "recognition/recognition_universal_huecode.hpp"


namespace fs = std::filesystem;

static std::string buildOutputPath(const std::string& input_image_path,
                                  const std::string& output_option)
{
    if (output_option.empty()) return "";

    fs::path in_path(input_image_path);
    fs::path out_path(output_option);

    // Treat as directory if:
    // - output ends with '/' (or '\') OR
    // - output exists and is a directory
    bool as_dir = false;
    if (!output_option.empty())
    {
        const char last = output_option.back();
        if (last == '/' || last == '\\') as_dir = true;
    }

    std::error_code ec;
    if (fs::exists(out_path, ec) && fs::is_directory(out_path, ec)) as_dir = true;

    if (as_dir)
    {
        fs::create_directories(out_path, ec);

        std::string stem = in_path.stem().string();
        std::string ext  = in_path.extension().string();
        if (ext.empty()) ext = ".png";

        fs::path dst = out_path / (stem + "_out" + ext);
        return dst.string();
    }

    // Otherwise treat as a file path; create parent directories if needed.
    fs::path parent = out_path.parent_path();
    if (!parent.empty())
        fs::create_directories(parent, ec);

    return out_path.string();
}

static cv::aruco::PREDEFINED_DICTIONARY_NAME parseArucoDict(const std::string& s)
{
    // Accept integer id
    bool all_digits = !s.empty();
    for (char c : s) all_digits = all_digits && (c >= '0' && c <= '9');
    if (all_digits) return static_cast<cv::aruco::PREDEFINED_DICTIONARY_NAME>(std::stoi(s));

    // Accept common OpenCV dictionary names
    if (s == "DICT_4X4_50") return cv::aruco::DICT_4X4_50;
    if (s == "DICT_4X4_100") return cv::aruco::DICT_4X4_100;
    if (s == "DICT_4X4_250") return cv::aruco::DICT_4X4_250;
    if (s == "DICT_4X4_1000") return cv::aruco::DICT_4X4_1000;

    if (s == "DICT_5X5_50") return cv::aruco::DICT_5X5_50;
    if (s == "DICT_5X5_100") return cv::aruco::DICT_5X5_100;
    if (s == "DICT_5X5_250") return cv::aruco::DICT_5X5_250;
    if (s == "DICT_5X5_1000") return cv::aruco::DICT_5X5_1000;

    if (s == "DICT_6X6_50") return cv::aruco::DICT_6X6_50;
    if (s == "DICT_6X6_100") return cv::aruco::DICT_6X6_100;
    if (s == "DICT_6X6_250") return cv::aruco::DICT_6X6_250;
    if (s == "DICT_6X6_1000") return cv::aruco::DICT_6X6_1000;

    if (s == "DICT_7X7_50") return cv::aruco::DICT_7X7_50;
    if (s == "DICT_7X7_100") return cv::aruco::DICT_7X7_100;
    if (s == "DICT_7X7_250") return cv::aruco::DICT_7X7_250;
    if (s == "DICT_7X7_1000") return cv::aruco::DICT_7X7_1000;

    if (s == "DICT_ARUCO_ORIGINAL") return cv::aruco::DICT_ARUCO_ORIGINAL;

    throw std::runtime_error("Unknown aruco_dict: " + s);
}

static std::vector<cv::Vec3b> parsePaletteRgb(const YAML::Node& node)
{
    // Expected format:
    // palette_rgb:
    //   - [R, G, B]
    //   - [R, G, B]
    std::vector<cv::Vec3b> pal;
    if (!node || node.IsNull()) return pal;
    if (!node.IsSequence())
        throw std::runtime_error("palette_rgb must be a YAML sequence");

    for (std::size_t i = 0; i < node.size(); ++i)
    {
        const YAML::Node c = node[i];
        if (!c.IsSequence() || c.size() != 3)
            throw std::runtime_error("palette_rgb entries must be [R, G, B]");

        int r = c[0].as<int>();
        int g = c[1].as<int>();
        int b = c[2].as<int>();

        auto inRange = [](int v) { return 0 <= v && v <= 255; };
        if (!inRange(r) || !inRange(g) || !inRange(b))
            throw std::runtime_error("palette_rgb values must be in [0,255]");

        pal.emplace_back((uchar)r, (uchar)g, (uchar)b); // Stored as RGB
    }
    return pal;
}

static int getIntOrDefault(const YAML::Node& n, const char* key, int def)
{
    if (n && n[key]) return n[key].as<int>();
    return def;
}

static double getDoubleOrDefault(const YAML::Node& n, const char* key, double def)
{
    if (n && n[key]) return n[key].as<double>();
    return def;
}

static bool getBoolOrDefault(const YAML::Node& n, const char* key, bool def)
{
    if (n && n[key]) return n[key].as<bool>();
    return def;
}

static std::string getStringOrDefault(const YAML::Node& n, const char* key, const std::string& def)
{
    if (n && n[key]) return n[key].as<std::string>();
    return def;
}

int main(int argc, char** argv)
{
    std::string config_path;
    if (argc != 2)
    {
        std::cerr << "Default config path: config.yaml\n";
        config_path = "config.yaml";
    }
    else config_path = argv[1];

    YAML::Node cfg;
    try
    {
        cfg = YAML::LoadFile(config_path);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to load YAML: " << config_path << "\n";
        std::cerr << "Exception: " << e.what() << "\n";
        return 2;
    }

    const std::string input_path = getStringOrDefault(cfg, "input", "");
    if (input_path.empty())
    {
        std::cerr << "YAML must contain: input\n";
        return 2;
    }

    const std::string output_opt = getStringOrDefault(cfg, "output", "");

    const int side_pixels = getIntOrDefault(cfg, "side_pixels", 300);
    const int margin_size = getIntOrDefault(cfg, "margin_size", 25);
    const bool clustering_separately = getBoolOrDefault(cfg, "clustering_separately", true);

    // ArUco dictionary: allow string name or integer id
    cv::aruco::PREDEFINED_DICTIONARY_NAME aruco_dict = cv::aruco::DICT_4X4_50;
    if (cfg["aruco_dict"])
    {
        if (cfg["aruco_dict"].IsScalar())
        {
            const std::string s = cfg["aruco_dict"].as<std::string>();
            aruco_dict = parseArucoDict(s);
        }
        else
        {
            aruco_dict = static_cast<cv::aruco::PREDEFINED_DICTIONARY_NAME>(cfg["aruco_dict"].as<int>());
        }
    }

    // Palette (RGB)
    std::vector<cv::Vec3b> palette_rgb = parsePaletteRgb(cfg["palette_rgb"]);

    // Remapper params
    huecode::OnlineRemapperParams p;
    const YAML::Node r = cfg["remap"];

    p.n_clusters  = getIntOrDefault(r, "n_clusters", p.n_clusters);
    p.n_blacks    = getIntOrDefault(r, "n_blacks", p.n_blacks);
    p.max_iter    = getIntOrDefault(r, "max_iter", p.max_iter);
    p.eps         = getDoubleOrDefault(r, "eps", p.eps);
    p.attempts    = getIntOrDefault(r, "attempts", p.attempts);
    p.iter_dilate = getIntOrDefault(r, "iter_dilate", p.iter_dilate);
    p.iter_erode  = getIntOrDefault(r, "iter_erode", p.iter_erode);

    cv::Mat bgr = cv::imread(input_path, cv::IMREAD_COLOR);
    if (bgr.empty())
    {
        std::cerr << "Failed to read image: " << input_path << "\n";
        return 1;
    }

    const std::string result_image_path = buildOutputPath(input_path, output_opt);

    ResultObject result;

    try
    {
        recognition::recognizeUniversalHuecode(
            bgr,
            result,
            side_pixels,
            margin_size,
            p,
            clustering_separately,
            result_image_path,
            aruco_dict,
            palette_rgb
        );
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }

    std::cout << "config: " << config_path << "\n";
    std::cout << "input:  " << input_path << "\n";
    std::cout << "aruco:  " << (result.aruco.empty() ? "(empty)" : result.aruco) << "\n";
    std::cout << "qr:    " << (result.qr.empty() ? "(empty)" : result.qr) << "\n";

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "duration_aruco: " << result.duration_aruco << " sec\n";
    std::cout << "duration_clip:  " << result.duration_clip  << " sec\n";
    std::cout << "duration_remap: " << result.duration_remap << " sec\n";
    std::cout << "duration_qr:   " << result.duration_qr   << " sec\n";

    if (!result_image_path.empty())
        std::cout << "result_image:  " << result_image_path << "\n";

    return 0;
}
