"""Animated graphic: the face swap problem — two inputs, one result.

Illustrates VIDEO_SCRIPTS_2.md Part 2, 0:00-0:35 (Bài toán):
- the model takes two inputs: source (identity to take) and target
  (the image/video whose face gets replaced)
- the result keeps the source identity, follows the target's pose,
  expression and lighting, and must blend naturally into the frame

Uses assets/images/{source,target,result}.* when present.

Render (from the video_graphics/ folder):

    manim -pqh scenes/part2_problem.py Part2Problem

Render with transparent background (for overlaying in a video editor):

    manim -qh -t scenes/part2_problem.py Part2Problem
"""

from pathlib import Path

from manim import (
    BOLD,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    FadeIn,
    FadeOut,
    Group,
    ImageMobject,
    Indicate,
    LaggedStart,
    RoundedRectangle,
    Scene,
    SVGMobject,
    Text,
    VGroup,
    VMobject,
    Write,
)

BACKGROUND = "#0d1117"
TEXT_COLOR = "#f0f2f5"
MUTED_TEXT = "#8b98a9"
ACCENT_BLUE = "#4da6ff"
ACCENT_YELLOW = "#f5e663"
ACCENT_RED = "#ff5c5c"
ACCENT_GREEN = "#b6e05f"
ACCENT_PURPLE = "#b493ea"
ACCENT_PINK = "#f96fa8"
CARD_FILL = "#1d2a3a"
FONT = "Arial"
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
ICONS_DIR = ASSETS_DIR / "icons"
IMAGES_DIR = ASSETS_DIR / "images"

PHOTO_W, PHOTO_H = 2.55, 2.15


def find_image(stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        path = IMAGES_DIR / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def photo_content(stem: str, color: str) -> Group:
    path = find_image(stem)
    if path is not None:
        img = ImageMobject(str(path))
        img.scale(min(PHOTO_W / img.width, PHOTO_H / img.height))
        return Group(img)
    icon = SVGMobject(ICONS_DIR / "image.svg")
    icon.set_color(color).set_height(0.8)
    note = Text(f"{stem}.png", font=FONT, color=MUTED_TEXT, font_size=20)
    return Group(VGroup(icon, note).arrange(DOWN, buff=0.22))


def photo_card(stem: str, title: str, color: str, height: float = 3.7) -> Group:
    frame = RoundedRectangle(
        corner_radius=0.18,
        width=3.15,
        height=height,
        fill_color=CARD_FILL,
        fill_opacity=1.0,
        stroke_color=color,
        stroke_width=3.5,
    )
    content = photo_content(stem, color)
    content.move_to(frame.get_center() + UP * 0.35)
    caption = Text(title, font=FONT, weight=BOLD, color=TEXT_COLOR, font_size=25)
    if caption.width > frame.width - 0.4:
        caption.scale_to_fit_width(frame.width - 0.4)
    caption.move_to(frame.get_center() + DOWN * (height / 2 - 0.42))
    return Group(frame, content, caption)


def chip(text: str, color: str, icon_file: str | None = None,
         font_size: int = 24) -> VGroup:
    txt = Text(text, font=FONT, weight=BOLD, color=TEXT_COLOR, font_size=font_size)
    content = txt
    if icon_file is not None:
        icon = SVGMobject(ICONS_DIR / icon_file)
        icon.set_color(color).set_height(0.38)
        content = VGroup(icon, txt).arrange(RIGHT, buff=0.2)
    box = RoundedRectangle(
        corner_radius=0.12,
        width=content.width + 0.5,
        height=content.height + 0.34,
        fill_color=CARD_FILL,
        fill_opacity=1.0,
        stroke_color=color,
        stroke_width=2.8,
    )
    content.move_to(box)
    return VGroup(box, content)


def check_line(text: str, color: str, font_size: int = 26) -> VGroup:
    # Arial lacks a checkmark glyph, so draw one from a polyline instead.
    mark = VMobject(stroke_color=color, stroke_width=6)
    mark.set_points_as_corners([
        LEFT * 0.14 + DOWN * 0.02,
        LEFT * 0.03 + DOWN * 0.14,
        RIGHT * 0.17 + UP * 0.14,
    ])
    body = Text(text, font=FONT, color=TEXT_COLOR, font_size=font_size)
    return VGroup(mark, body).arrange(RIGHT, buff=0.28)


class Part2Problem(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND

        # --- Phase 0: recap Part 1 and preview this episode ------------------
        recap_title = Text(
            "Ở phần trước",
            font=FONT,
            weight=BOLD,
            color=TEXT_COLOR,
            font_size=38,
        ).to_edge(UP, buff=0.55)
        recap = VGroup(
            chip("Deepfake", ACCENT_RED, font_size=24),
            chip("Face Swap", ACCENT_BLUE, "scan-face.svg", 24),
            chip("Bản chất mô hình AI", ACCENT_PURPLE, "layers.svg", 24),
        ).arrange(RIGHT, buff=0.38)
        recap.move_to(UP * 0.25)

        self.play(Write(recap_title), run_time=0.7)
        self.play(
            LaggedStart(
                *[FadeIn(item, shift=UP * 0.2) for item in recap],
                lag_ratio=0.28,
            ),
            run_time=1.25,
        )
        self.wait(0.7)
        self.play(FadeOut(VGroup(recap_title, recap)), run_time=0.55)

        episode_title = Text(
            "PHẦN 2  ·  CHUẨN BỊ XÂY DỰNG MÔ HÌNH",
            font=FONT,
            weight=BOLD,
            color=TEXT_COLOR,
            font_size=34,
        ).to_edge(UP, buff=0.5)

        roadmap = VGroup(
            chip("1  Bài toán", ACCENT_BLUE, font_size=24),
            Text("→", font=FONT, color=MUTED_TEXT, font_size=30),
            chip("2  Khung mô hình", ACCENT_PURPLE, font_size=24),
            Text("→", font=FONT, color=MUTED_TEXT, font_size=30),
            chip("3  Chuẩn bị dữ liệu", ACCENT_GREEN, "database.svg", 24),
        ).arrange(RIGHT, buff=0.24)
        roadmap.move_to(UP * 0.25)

        later = VGroup(
            Text(
                "HUẤN LUYỆN",
                font=FONT,
                weight=BOLD,
                color=MUTED_TEXT,
                font_size=25,
            ),
            Text(
                "→  các phần sau",
                font=FONT,
                color=MUTED_TEXT,
                font_size=25,
            ),
        ).arrange(RIGHT, buff=0.2)
        later.next_to(roadmap, DOWN, buff=0.7)

        self.play(Write(episode_title), run_time=0.75)
        self.play(
            LaggedStart(
                *[FadeIn(item, shift=RIGHT * 0.15) for item in roadmap],
                lag_ratio=0.18,
            ),
            run_time=1.4,
        )
        self.play(FadeIn(later, shift=UP * 0.2), run_time=0.6)
        self.play(
            Indicate(roadmap[0], color=ACCENT_BLUE, scale_factor=1.06),
            run_time=0.65,
        )
        self.wait(0.7)
        self.play(
            FadeOut(VGroup(episode_title, roadmap, later)),
            run_time=0.65,
        )

        # --- Phase 1: the two inputs -----------------------------------------
        title = Text(
            "Mô hình nhận hai đầu vào",
            font=FONT,
            weight=BOLD,
            color=TEXT_COLOR,
            font_size=38,
        ).to_edge(UP, buff=0.4)

        source = photo_card("source-1", "Source — nguồn", ACCENT_BLUE)
        target = photo_card("target-1", "Target — đích", ACCENT_YELLOW)
        source.move_to(LEFT * 3.55 + UP * 0.35)
        target.move_to(RIGHT * 3.55 + UP * 0.35)

        source_chip = chip(
            "Danh tính muốn lấy", ACCENT_BLUE, "scan-face.svg", 22
        )
        source_chip.next_to(source, DOWN, buff=0.3)
        target_chip = chip(
            "Khuôn mặt cần thay thế", ACCENT_YELLOW, "image.svg", 22
        )
        target_chip.next_to(target, DOWN, buff=0.3)

        self.play(Write(title), run_time=0.9)
        self.play(FadeIn(source, shift=RIGHT * 0.35), run_time=0.7)
        self.play(FadeIn(source_chip, shift=UP * 0.25), run_time=0.55)
        self.play(FadeIn(target, shift=LEFT * 0.35), run_time=0.7)
        self.play(FadeIn(target_chip, shift=UP * 0.25), run_time=0.55)
        self.wait(1.2)

        # --- Phase 2: pipeline into the model, out to the result -------------
        self.play(
            FadeOut(Group(title, source_chip, target_chip)),
            run_time=0.6,
        )
        self.play(
            source.animate.scale(0.62).move_to(LEFT * 5.15 + UP * 1.55),
            target.animate.scale(0.62).move_to(LEFT * 5.15 + DOWN * 1.55),
            run_time=0.9,
        )

        model_label = Text(
            "Mô hình\nFace Swap",
            font=FONT,
            weight=BOLD,
            color=TEXT_COLOR,
            font_size=30,
            line_spacing=0.9,
        )
        model_box = RoundedRectangle(
            corner_radius=0.18,
            width=model_label.width + 1.1,
            height=model_label.height + 0.9,
            fill_color=CARD_FILL,
            fill_opacity=1.0,
            stroke_color=ACCENT_PURPLE,
            stroke_width=3.5,
        )
        model_label.move_to(model_box)
        model = VGroup(model_box, model_label)
        model.move_to(LEFT * 0.9)

        result = photo_card("result-1", "Kết quả", ACCENT_GREEN, height=3.45)
        result.move_to(RIGHT * 4.35)

        arrow_source = Arrow(
            source.get_right(), model_box.get_corner(UP + LEFT) + DOWN * 0.45,
            buff=0.15, color=ACCENT_BLUE, stroke_width=5,
        )
        arrow_target = Arrow(
            target.get_right(), model_box.get_corner(DOWN + LEFT) + UP * 0.45,
            buff=0.15, color=ACCENT_YELLOW, stroke_width=5,
        )
        arrow_out = Arrow(
            model_box.get_right(), result.get_left(),
            buff=0.15, color=ACCENT_GREEN, stroke_width=5,
        )

        self.play(FadeIn(model, scale=0.9), run_time=0.6)
        self.play(
            LaggedStart(
                FadeIn(arrow_source), FadeIn(arrow_target), lag_ratio=0.3
            ),
            run_time=0.8,
        )
        self.play(FadeIn(arrow_out), FadeIn(result, shift=LEFT * 0.3),
                  run_time=0.8)
        self.wait(1.0)

        # --- Phase 3: what the result must satisfy ---------------------------
        requirements = VGroup(
            check_line("Giữ danh tính của nguồn", ACCENT_BLUE),
            check_line("Theo góc mặt, biểu cảm, ánh sáng của đích",
                       ACCENT_YELLOW),
            check_line("Hòa vào khung hình tự nhiên", ACCENT_GREEN),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        requirements.to_edge(DOWN, buff=0.35).set_x(0.9)

        self.play(
            LaggedStart(
                *[FadeIn(line, shift=UP * 0.2) for line in requirements],
                lag_ratio=0.35,
            ),
            run_time=1.6,
        )
        self.play(
            Indicate(result[0], color=ACCENT_GREEN, scale_factor=1.04),
            run_time=0.7,
        )
        self.wait(2.0)

        # --- Clean exit -------------------------------------------------------
        self.play(
            FadeOut(
                Group(
                    source, target, model, result,
                    arrow_source, arrow_target, arrow_out, requirements,
                )
            ),
            run_time=0.8,
        )
        self.wait(0.3)
