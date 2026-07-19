"""Animated mind map: the forms a deepfake can take, with Face Swap highlighted.

Used in Part 1 of the video (0:20-1:30, "Deepfake va Face Swap" section),
right after the "Deep Learning + Fake = Deepfake" title.

Render (from the video_graphics/ folder):

    manim -pqh scenes/deepfake_mindmap.py DeepfakeMindmap

Render with transparent background (for overlaying in a video editor):

    manim -qh -t scenes/deepfake_mindmap.py DeepfakeMindmap
"""

from manim import (
    BOLD,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Circle,
    Create,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    Line,
    RoundedRectangle,
    Scene,
    SurroundingRectangle,
    Text,
    VGroup,
    Write,
)

BACKGROUND = "#0d1117"
TEXT_DARK = "#0d1117"
HUB_COLOR = "#2743d6"
HIGHLIGHT_COLOR = "#ff5c5c"
FONT = "JetBrains Mono"


def label_box(text: str, color: str, font_size: int = 34) -> VGroup:
    """A rounded colored box with dark text, like the mind-map nodes."""
    txt = Text(text, font=FONT, weight=BOLD, color=TEXT_DARK, font_size=font_size)
    box = RoundedRectangle(
        corner_radius=0.12,
        width=txt.width + 0.55,
        height=txt.height + 0.42,
        fill_color=color,
        fill_opacity=1.0,
        stroke_width=0,
    )
    txt.move_to(box)
    group = VGroup(box, txt)
    group.set_z_index(2)
    return group


class DeepfakeMindmap(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND

        # Central hub
        hub_circle = Circle(
            radius=1.05, fill_color=HUB_COLOR, fill_opacity=1.0, stroke_width=0
        )
        hub_text = Text(
            "DEEPFAKE", font=FONT, weight=BOLD, color="#f0f2f5", font_size=34
        )
        hub = VGroup(hub_circle, hub_text)
        hub.set_z_index(3)

        # Branches: (color, vietnamese label, position, english label, position)
        branch_specs = [
            ("#f96fa8", "Tạo hình ảnh giả", UP * 2.35 + LEFT * 0.6,
             "AI-generated Image", UP * 3.45 + LEFT * 1.2),
            ("#f5e663", "Đổi mặt", UP * 1.3 + RIGHT * 3.0,
             "Face Swap", UP * 2.3 + RIGHT * 5.0),
            ("#b6e05f", "Giả giọng", DOWN * 1.1 + RIGHT * 3.1,
             "Voice Cloning", DOWN * 2.2 + RIGHT * 5.3),
            ("#ff6b6b", "Tạo video giả", DOWN * 2.4 + RIGHT * 0.35,
             "AI-generated Video", DOWN * 3.45 + RIGHT * 1.7),
            ("#f65ccb", "Thay đổi biểu cảm", DOWN * 1.3 + LEFT * 2.7,
             "Face Reenactment", DOWN * 2.7 + LEFT * 4.6),
            ("#b493ea", "Nhép môi", UP * 1.35 + LEFT * 2.9,
             "Lip Sync", UP * 2.5 + LEFT * 4.9),
        ]

        branches = VGroup()
        for color, vn, vn_pos, en, en_pos in branch_specs:
            vn_box = label_box(vn, color).move_to(vn_pos)
            en_box = label_box(en, color, font_size=30).move_to(en_pos)
            main_line = Line(hub.get_center(), vn_box.get_center(),
                             color=color, stroke_width=8)
            sub_line = Line(vn_box.get_center(), en_box.get_center(),
                            color=color, stroke_width=5)
            main_line.set_z_index(0)
            sub_line.set_z_index(0)
            branches.add(VGroup(main_line, vn_box, sub_line, en_box))

        # 1. Hub appears.
        self.play(GrowFromCenter(hub_circle), Write(hub_text), run_time=1.0)
        self.wait(0.3)

        # 2. Branches grow out one at a time, clockwise from the top.
        for main_line, vn_box, sub_line, en_box in branches:
            self.play(Create(main_line), FadeIn(vn_box, scale=0.85), run_time=0.55)
            self.play(Create(sub_line), FadeIn(en_box, scale=0.85), run_time=0.45)
        self.wait(0.8)

        # 3. Highlight the Face Swap branch (the topic of the series);
        #    every other branch dims.
        face_swap_branch = branches[1]
        others = VGroup(*[b for i, b in enumerate(branches) if i != 1])
        frame = SurroundingRectangle(
            VGroup(face_swap_branch[1], face_swap_branch[3]),
            color=HIGHLIGHT_COLOR,
            buff=0.35,
            stroke_width=5,
        )
        frame.set_z_index(4)
        self.play(
            Create(frame),
            others.animate.set_opacity(0.22),
            run_time=1.1,
        )
        self.wait(1.6)

        # 4. Clean exit so the clip cuts nicely in the editor.
        self.play(FadeOut(VGroup(hub, branches, frame)), run_time=0.8)
        self.wait(0.3)
