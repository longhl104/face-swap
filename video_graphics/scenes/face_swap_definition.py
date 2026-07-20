"""Animated graphic: what face swap is, and how it relates to deepfake.

Illustrates VIDEO_SCRIPTS.md Part 1, 0:20-1:30 (line 42): face swap transfers
the facial identity of one person onto another person's face, so it can be
seen as one application of deepfake — but deepfake is more than face swap.

Face photos are loaded from assets/images/ using the stems "source",
"target" and "result" (png/jpg/jpeg/webp). Until those files exist the scene
renders labeled placeholders, so it can be previewed before the photos are
ready — just re-render after adding the images.

Render (from the video_graphics/ folder):

    manim -pqh scenes/face_swap_definition.py FaceSwapDefinition

Render with transparent background (for overlaying in a video editor):

    manim -qh -t scenes/face_swap_definition.py FaceSwapDefinition
"""

from pathlib import Path

from manim import (
    BOLD,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    ArcBetweenPoints,
    Arrow,
    Circle,
    Create,
    Ellipse,
    FadeIn,
    FadeOut,
    FadeTransform,
    Flash,
    Group,
    GrowFromCenter,
    ImageMobject,
    LaggedStart,
    MoveAlongPath,
    RoundedRectangle,
    Scene,
    SVGMobject,
    Text,
    VGroup,
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
FONT = "JetBrains Mono"
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
ICONS_DIR = ASSETS_DIR / "icons"
IMAGES_DIR = ASSETS_DIR / "images"

PHOTO_W, PHOTO_H = 2.7, 2.3


def find_image(stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        path = IMAGES_DIR / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def photo_content(stem: str, color: str) -> Group:
    """The real photo if assets/images/<stem>.* exists, else a placeholder."""
    path = find_image(stem)
    if path is not None:
        img = ImageMobject(str(path))
        img.scale(min(PHOTO_W / img.width, PHOTO_H / img.height))
        return Group(img)
    icon = SVGMobject(ICONS_DIR / "image.svg")
    icon.set_color(color).set_height(0.85)
    note = Text(f"{stem}.png", font=FONT, color=MUTED_TEXT, font_size=22)
    return Group(VGroup(icon, note).arrange(DOWN, buff=0.28))


def photo_card(stem: str, label: str, color: str) -> Group:
    """A framed photo (or placeholder) with a name caption underneath."""
    frame = RoundedRectangle(
        corner_radius=0.18,
        width=3.2,
        height=3.5,
        fill_color=CARD_FILL,
        fill_opacity=1.0,
        stroke_color=color,
        stroke_width=3.5,
    )
    content = photo_content(stem, color)
    content.move_to(frame.get_center() + UP * 0.3)
    caption = Text(label, font=FONT, weight=BOLD, color=TEXT_COLOR, font_size=30)
    caption.move_to(frame.get_center() + DOWN * 1.35)
    return Group(frame, content, caption)


def tag(text: str, color: str, icon_file: str | None = None,
        font_size: int = 28) -> VGroup:
    """A small rounded chip with an optional icon and a label."""
    txt = Text(text, font=FONT, weight=BOLD, color=TEXT_COLOR, font_size=font_size)
    content = txt
    if icon_file is not None:
        icon = SVGMobject(ICONS_DIR / icon_file)
        icon.set_color(color).set_height(0.42)
        content = VGroup(icon, txt).arrange(RIGHT, buff=0.24)
    box = RoundedRectangle(
        corner_radius=0.14,
        width=content.width + 0.55,
        height=content.height + 0.4,
        fill_color=CARD_FILL,
        fill_opacity=1.0,
        stroke_color=color,
        stroke_width=3,
    )
    content.move_to(box)
    return VGroup(box, content)


class FaceSwapDefinition(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND

        # --- Phase 1: identity moves from person A onto person B's face -----
        title = Text(
            "Face Swap", font=FONT, weight=BOLD, color=ACCENT_BLUE, font_size=46
        )
        subtitle = Text(
            "Chuyển danh tính khuôn mặt từ người này sang người khác",
            font=FONT,
            color=TEXT_COLOR,
            font_size=30,
        )
        header = VGroup(title, subtitle).arrange(DOWN, buff=0.28)
        header.to_edge(UP, buff=0.45)

        source_card = photo_card("source", "Người A", ACCENT_BLUE)
        target_card = photo_card("target", "Người B", ACCENT_YELLOW)
        source_card.move_to(LEFT * 4.0 + DOWN * 0.55)
        target_card.move_to(RIGHT * 4.0 + DOWN * 0.55)
        source_photo_center = source_card[1].get_center()
        target_photo_center = target_card[1].get_center()

        arrow = Arrow(
            source_card[0].get_right() + RIGHT * 0.15,
            target_card[0].get_left() + LEFT * 0.15,
            color=TEXT_COLOR,
            stroke_width=4,
        )

        self.play(Write(header), run_time=1.1)
        self.play(
            FadeIn(source_card, shift=RIGHT * 0.4),
            FadeIn(target_card, shift=LEFT * 0.4),
            run_time=0.9,
        )
        self.wait(0.4)

        # The "identity" chip is extracted from A's face...
        chip_badge = Circle(
            radius=0.48,
            fill_color=CARD_FILL,
            fill_opacity=1.0,
            stroke_color=ACCENT_BLUE,
            stroke_width=3.5,
        )
        chip_icon = SVGMobject(ICONS_DIR / "scan-face.svg")
        chip_icon.set_color(ACCENT_BLUE).set_height(0.52)
        chip_icon.move_to(chip_badge)
        chip_label = Text(
            "Danh tính", font=FONT, weight=BOLD, color=ACCENT_BLUE, font_size=26
        ).next_to(chip_badge, DOWN, buff=0.18)
        # Dark outline keeps the label readable on top of the photo.
        chip_label.set_stroke(BACKGROUND, width=6, background=True)
        chip = VGroup(chip_badge, chip_icon, chip_label)
        chip.move_to(source_photo_center)
        chip.set_z_index(5)

        self.play(GrowFromCenter(chip), run_time=0.6)
        self.play(
            Flash(source_photo_center, color=ACCENT_BLUE, line_length=0.25,
                  flash_radius=1.05, run_time=0.6)
        )

        # ...travels across to B...
        flight_path = ArcBetweenPoints(
            source_photo_center, target_photo_center, angle=-0.9
        )
        self.play(FadeIn(arrow, shift=RIGHT * 0.3), run_time=0.5)
        self.play(MoveAlongPath(chip, flight_path), run_time=1.3)

        # ...and is absorbed: B's photo becomes the result.
        result_content = photo_content("result", ACCENT_GREEN)
        result_content.move_to(target_photo_center)
        old_content = target_card[1]

        self.play(chip.animate.scale(0.1).fade(1), run_time=0.5)
        self.remove(chip)
        self.play(
            FadeOut(old_content),
            FadeIn(result_content),
            target_card[0].animate.set_stroke(ACCENT_GREEN),
            Flash(target_photo_center, color=ACCENT_GREEN, line_length=0.25,
                  flash_radius=1.05),
            run_time=0.9,
        )
        target_card.remove(old_content)
        target_card.add(result_content)

        result_caption = Text(
            "Khuôn mặt của B — mang danh tính của A",
            font=FONT,
            weight=BOLD,
            color=TEXT_COLOR,
            font_size=32,
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(result_caption, shift=UP * 0.3), run_time=0.7)
        self.wait(1.5)

        # --- Phase 2: face swap sits inside the bigger deepfake circle ------
        self.play(
            FadeOut(Group(header, source_card, target_card, arrow, result_caption)),
            run_time=0.8,
        )

        boundary = Ellipse(
            width=10.8, height=5.7, stroke_color=ACCENT_RED, stroke_width=4
        )
        boundary.move_to(DOWN * 0.25)
        df_label = Text(
            "DEEPFAKE", font=FONT, weight=BOLD, color=ACCENT_RED, font_size=36
        )
        df_label.move_to(boundary.get_center() + UP * 2.05)

        face_swap_node = tag("Face Swap", ACCENT_BLUE, "scan-face.svg", font_size=30)
        face_swap_node.move_to(boundary.get_center() + UP * 0.85)

        others = VGroup(
            tag("Giả giọng", ACCENT_PURPLE, "mic.svg", font_size=24),
            tag("Nhép môi", ACCENT_PINK, "message-square-quote.svg", font_size=24),
            tag("Thay đổi biểu cảm", ACCENT_GREEN, "smile.svg", font_size=24),
            tag("Tạo video giả", ACCENT_YELLOW, "video.svg", font_size=24),
        )
        others[0].move_to(boundary.get_center() + LEFT * 2.9 + DOWN * 0.35)
        others[1].move_to(boundary.get_center() + RIGHT * 2.9 + DOWN * 0.35)
        others[2].move_to(boundary.get_center() + LEFT * 2.4 + DOWN * 1.45)
        others[3].move_to(boundary.get_center() + RIGHT * 2.7 + DOWN * 1.45)

        caption_1 = Text(
            "Face swap là một ứng dụng của deepfake",
            font=FONT,
            weight=BOLD,
            color=TEXT_COLOR,
            font_size=32,
        ).to_edge(DOWN, buff=0.45)
        caption_2 = Text(
            "...nhưng deepfake không chỉ có face swap",
            font=FONT,
            weight=BOLD,
            color=TEXT_COLOR,
            font_size=32,
        ).to_edge(DOWN, buff=0.45)

        self.play(Create(boundary), Write(df_label), run_time=1.2)
        self.play(FadeIn(face_swap_node, scale=0.8), run_time=0.6)
        self.play(FadeIn(caption_1, shift=UP * 0.3), run_time=0.7)
        self.play(
            Flash(face_swap_node, color=ACCENT_BLUE, line_length=0.2,
                  flash_radius=face_swap_node.width / 2 + 0.25, run_time=0.6)
        )
        self.wait(1.0)

        self.play(
            FadeTransform(caption_1, caption_2),
            LaggedStart(
                *[FadeIn(node, scale=0.85, shift=UP * 0.2) for node in others],
                lag_ratio=0.25,
            ),
            run_time=1.6,
        )
        self.wait(1.8)

        # --- Clean exit so the clip cuts nicely in the editor ----------------
        self.play(
            FadeOut(VGroup(boundary, df_label, face_swap_node, others, caption_2)),
            run_time=0.8,
        )
        self.wait(0.3)
