"""Animated neural network: numbers flow through layers, features get complex.

Illustrates VIDEO_SCRIPTS.md Part 1, 0:20-1:30 (lines about deep learning):
a neural network is made of many processing nodes; each node receives
numbers, transforms them and passes them on. Early layers learn simple
features (edges, colors), deeper layers recognize eyes, a nose, or a
whole face.

Render (from the video_graphics/ folder):

    manim -pqh scenes/neural_network.py NeuralNetwork

Render with transparent background (for overlaying in a video editor):

    manim -qh -t scenes/neural_network.py NeuralNetwork
"""

from pathlib import Path

from manim import (
    BOLD,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arc,
    Arrow,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    LaggedStart,
    Line,
    MoveAlongPath,
    Scene,
    Square,
    SVGMobject,
    Text,
    VGroup,
    Write,
)

BACKGROUND = "#0d1117"
TEXT_COLOR = "#f0f2f5"
ACCENT_BLUE = "#4da6ff"
ACCENT_YELLOW = "#f5e663"
NODE_BASE = "#1d2a3a"
EDGE_COLOR = "#3a4a5e"
FONT = "Arial"
ICONS_DIR = Path(__file__).resolve().parents[1] / "assets" / "icons"

LAYER_X = [-3.5, -1.4, 0.7, 2.8]
LAYER_SIZES = [4, 6, 6, 4]
NODE_RADIUS = 0.17
NET_Y = 0.5


def make_node() -> Circle:
    return Circle(
        radius=NODE_RADIUS,
        fill_color=NODE_BASE,
        fill_opacity=1.0,
        stroke_color=ACCENT_BLUE,
        stroke_width=2.5,
    )


def edge_between(a: Circle, b: Circle) -> Line:
    direction = b.get_center() - a.get_center()
    direction = direction / (direction[0] ** 2 + direction[1] ** 2) ** 0.5
    return Line(
        a.get_center() + direction * NODE_RADIUS,
        b.get_center() - direction * NODE_RADIUS,
        color=EDGE_COLOR,
        stroke_width=1.8,
    )


class NeuralNetwork(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND

        title = Text(
            "Mạng nơ-ron nhiều lớp",
            font=FONT,
            weight=BOLD,
            color=TEXT_COLOR,
            font_size=36,
        ).to_edge(UP, buff=0.45)

        # --- Input image as a grid of numbers -------------------------------
        pixel_values = ["0.9", "0.2", "0.7", "0.1", "0.8", "0.4", "0.6", "0.3", "0.5"]
        pixel_shades = [
            "#8fa8c9", "#22303f", "#6d87a8",
            "#182230", "#7d97b8", "#3a4c61",
            "#5c7695", "#2d3d4f", "#4b6076",
        ]
        cell_size = 0.62
        grid = VGroup()
        numbers = VGroup()
        for idx, (val, shade) in enumerate(zip(pixel_values, pixel_shades)):
            row, col = divmod(idx, 3)
            cell = Square(
                side_length=cell_size,
                fill_color=shade,
                fill_opacity=1.0,
                stroke_color=BACKGROUND,
                stroke_width=2,
            )
            cell.move_to(
                LEFT * 5.9
                + UP * NET_Y
                + RIGHT * (col - 1) * cell_size
                + DOWN * (row - 1) * cell_size
            )
            num = Text(val, font=FONT, color=TEXT_COLOR, font_size=19)
            num.move_to(cell)
            grid.add(cell)
            numbers.add(num)

        # --- Network nodes and edges ---------------------------------------
        layers = []
        for x, size in zip(LAYER_X, LAYER_SIZES):
            layer = VGroup()
            for i in range(size):
                node = make_node()
                node.move_to(RIGHT * x + UP * (NET_Y + (i - (size - 1) / 2) * 0.85))
                layer.add(node)
            layers.append(layer)

        edge_groups = []
        for l_from, l_to in zip(layers[:-1], layers[1:]):
            group = VGroup(*[edge_between(a, b) for a in l_from for b in l_to])
            edge_groups.append(group)

        in_arrow = Arrow(
            grid.get_right() + RIGHT * 0.1,
            layers[0].get_center() + LEFT * 0.55,
            color=TEXT_COLOR,
            stroke_width=3.5,
            max_tip_length_to_length_ratio=0.3,
        )

        # --- Output: recognized face ----------------------------------------
        face = VGroup(
            Circle(radius=0.55, stroke_color=ACCENT_BLUE, stroke_width=4),
            Dot(LEFT * 0.2 + UP * 0.14, radius=0.05, color=TEXT_COLOR),
            Dot(RIGHT * 0.2 + UP * 0.14, radius=0.05, color=TEXT_COLOR),
            Arc(radius=0.27, start_angle=-2.5, angle=1.7, stroke_width=4,
                color=TEXT_COLOR),
        )
        face.move_to(RIGHT * 5.4 + UP * NET_Y)
        out_arrow = Arrow(
            layers[-1].get_center() + RIGHT * 0.55,
            face.get_left() + LEFT * 0.1,
            color=TEXT_COLOR,
            stroke_width=3.5,
            max_tip_length_to_length_ratio=0.3,
        )

        # --- Feature labels below the network -------------------------------
        strokes_icon = SVGMobject(ICONS_DIR / "palette.svg")
        strokes_icon.set_color(ACCENT_YELLOW).set_height(0.52)

        parts_icon = SVGMobject(ICONS_DIR / "scan-face.svg")
        parts_icon.set_color(TEXT_COLOR).set_height(0.58)

        def feature_label(icon: VGroup, text: str, x: float) -> VGroup:
            label = Text(text, font=FONT, weight=BOLD, color=TEXT_COLOR, font_size=26)
            group = VGroup(icon, label).arrange(DOWN, buff=0.22)
            group.move_to(RIGHT * x + DOWN * 3.0)
            return group

        simple_features = feature_label(strokes_icon, "Đường nét, màu sắc", -2.45)
        mid_features = feature_label(parts_icon, "Mắt, mũi", 0.7)
        face_label = Text(
            "Khuôn mặt", font=FONT, weight=BOLD, color=TEXT_COLOR, font_size=26
        ).next_to(face, DOWN, buff=0.4)

        # =====================================================================
        # 1. Title, then the input image as numbers.
        self.play(Write(title), run_time=0.9)
        self.play(
            LaggedStart(*[FadeIn(c, scale=0.8) for c in grid], lag_ratio=0.05),
            run_time=0.9,
        )
        self.play(
            LaggedStart(*[Write(n) for n in numbers], lag_ratio=0.05), run_time=0.9
        )
        self.wait(0.4)

        # 2. The network builds up layer by layer.
        self.play(FadeIn(in_arrow, shift=RIGHT * 0.3), run_time=0.5)
        for i, layer in enumerate(layers):
            anims = [GrowFromCenter(n) for n in layer]
            if i > 0:
                anims.append(Create(edge_groups[i - 1]))
            self.play(LaggedStart(*anims, lag_ratio=0.04), run_time=0.7)
        self.wait(0.5)

        # 3. Numbers enter the first layer.
        moving = numbers.copy()
        self.play(
            moving.animate.set_opacity(0.0).shift(RIGHT * 1.6),
            layers[0].animate.set_fill(ACCENT_BLUE, opacity=1.0),
            run_time=0.9,
        )
        self.remove(moving)

        # 4. Activation pulses travel through the network; feature labels
        #    appear as the signal reaches deeper layers.
        reveals = [FadeIn(simple_features, shift=UP * 0.3), FadeIn(mid_features, shift=UP * 0.3), None]
        for i, edges in enumerate(edge_groups):
            pulses = VGroup(
                *[Dot(radius=0.055, color=ACCENT_BLUE).move_to(e.get_start())
                  for e in edges]
            )
            self.play(
                LaggedStart(
                    *[MoveAlongPath(p, e) for p, e in zip(pulses, edges)],
                    lag_ratio=0.01,
                ),
                run_time=0.85,
            )
            arrival = [
                FadeOut(pulses),
                layers[i + 1].animate.set_fill(ACCENT_BLUE, opacity=1.0),
                layers[i].animate.set_fill(NODE_BASE, opacity=1.0),
            ]
            if reveals[i] is not None:
                arrival.append(reveals[i])
            self.play(*arrival, run_time=0.55)

        # 5. The output: the network recognized a face.
        self.play(
            FadeIn(out_arrow, shift=RIGHT * 0.3),
            Create(face),
            FadeIn(face_label, shift=UP * 0.3),
            run_time=1.1,
        )
        self.wait(1.6)

        # 6. Clean exit so the clip cuts nicely in the editor.
        everything = VGroup(
            title, grid, numbers, in_arrow, *layers, *edge_groups,
            simple_features, mid_features, out_arrow, face, face_label,
        )
        self.play(FadeOut(everything), run_time=0.8)
        self.wait(0.3)
