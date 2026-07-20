# Video Graphics

Animated graphics for the Face Swap YouTube series, built with [Manim Community](https://www.manim.community/). Each scene in `scenes/` is one clip that can be rendered to MP4 (or MOV with a transparent background) and dropped into the video editor.

## Setup

```bash
# From the repo root (Manim needs Python <= 3.13, so use 3.12 here)
py -3.12 -m venv video_graphics/.venv
video_graphics\.venv\Scripts\activate
pip install -r video_graphics/requirements.txt
```

## Rendering

Run Manim from inside `video_graphics/` so it picks up `manim.cfg` (1080p60 by default):

```bash
cd video_graphics

# Preview quality (fast), opens the file when done
manim -pql scenes/deepfake_equation.py DeepfakeEquation

# Final quality 1080p60
manim -qh scenes/deepfake_equation.py DeepfakeEquation

# Transparent background (.mov) for overlaying in the editor
manim -qh -t scenes/deepfake_equation.py DeepfakeEquation
```

Rendered files land in `video_graphics/media/videos/<scene_file>/<quality>/` (git-ignored).

## Scenes

| Scene | File | Used in script |
| --- | --- | --- |
| `DeepfakeEquation` — "Deep Learning + Fake = Deepfake" | `scenes/deepfake_equation.py` | Part 1, 0:20–1:30 (Deepfake vs Face Swap) |
| `DeepfakeMindmap` — deepfake forms mind map, Face Swap highlighted | `scenes/deepfake_mindmap.py` | Part 1, 0:20–1:30 (examples: đổi mặt, nhép môi, giả giọng) |
| `NeuralNetwork` — numbers flow through layers, features get complex | `scenes/neural_network.py` | Part 1, 0:20–1:30 (học sâu / mạng nơ-ron nhiều lớp) |
| `DeepfakeForms` — forms (image/video/audio) and what deepfake can do | `scenes/deepfake_forms.py` | Part 1, 0:20–1:30 (deepfake xuất hiện dưới nhiều dạng) |
| `FaceSwapDefinition` — identity moves from A to B, then face swap inside the deepfake set | `scenes/face_swap_definition.py` | Part 1, 0:20–1:30 (face swap là một ứng dụng của deepfake) |
| `SourceTargetRoles` — nguồn vs đích, not cut-and-paste, identity vs expression | `scenes/source_target_roles.py` | Part 1, 0:20–1:30 (người nguồn / người đích) |
| `Part1Outro` — flash results, comment CTA, then Phần 2 title card | `scenes/part1_outro.py` | Part 1, 2:15–2:45 (kết thúc phần 1) |

## Adding a new scene

1. Create a new file in `scenes/` (one file per graphic keeps renders organized).
2. Subclass `manim.Scene` and implement `construct()`.
3. Reuse the palette from `deepfake_equation.py` (`#0d1117` background, blue `#4da6ff`, red `#ff5c5c`) so clips look consistent across the series.
