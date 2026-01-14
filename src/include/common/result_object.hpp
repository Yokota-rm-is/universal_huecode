#pragma once
#include <string>

struct ResultObject
{
    // ---- formerly ResultObject ----
    std::string aruco; // ids[0] or ""
    std::string qr;   // data[0] or ""

    double duration_aruco = 0.0;
    double duration_clip = 0.0;
    double duration_remap = 0.0;
    double duration_qr = 0.0;
};