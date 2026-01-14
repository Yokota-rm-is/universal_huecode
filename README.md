# Universal HueCode 

## Status
- Marker detection: `src/` will be made public after the paper is accepted
- Marker generation: in progress

## 1) Install dependencies (Ubuntu 22.04)

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake pkg-config \
  libopencv-dev \
  libopencv-contrib-dev \
  libzbar-dev \
  libboost-all-dev \
  libapriltag-dev \
  libeigen3-dev \
  libyaml-cpp-dev \
  libjpeg-dev libpng-dev libtiff-dev \
  libavcodec-dev libavformat-dev libswscale-dev
```

## 2) Build

From the repository root:

```bash
mkdir -p build
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Executable:

```bash
./build/detect
```

## 3) Prepare a config file

Create or edit `config.yaml` (or any `*.yaml`) like this:

```yaml
input: "images/input1.png"
output: "results/"

side_pixels: 300
margin_size: 25
clustering_separately: true

aruco_dict: "DICT_4X4_50"

palette_rgb:
  - [0, 0, 255]
  - [97, 0, 0]
  - [157, 255, 255]
  - [255, 255, 0]

remap:
  n_clusters: 4
  n_blacks: 2
  attempts: 5
  eps: 1.0
  max_iter: 300
  iter_dilate: 0
  iter_erode: 0
```

Sample images included:
- `images/input1.png`
- `images/input2.png`
- `images/input3.png`

To test different images, change `input:` in the YAML.

## 4) Run

### Use default `config.yaml`

```bash
./build/detect
```

### Or specify a YAML path

```bash
./build/detect path/to/config.yaml
```

## 5) Check outputs

- Terminal: prints decoded strings and timing values.
- If `output` is set:
  - If it is a directory (recommended to end with `/`), the tool saves `<input_stem>_out.<ext>` under that directory.
  - If it is a file path, the tool saves exactly to that path.
