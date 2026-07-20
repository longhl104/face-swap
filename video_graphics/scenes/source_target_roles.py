"""Animated graphic: source vs target roles in face swap.

Illustrates VIDEO_SCRIPTS.md Part 1, 0:20-1:30 (lines 44-48):
- người nguồn provides facial identity
- người đích provides expression, pose, lighting and skin tone
- the model does not cut-and-paste; it transfers identity while
  preserving the target's expression and visual context
- summary: identity from source, expression/context from target

Uses assets/images/{source,target,result}.* when present.

Render (from the video_graphics/ folder):

    manim -pqh scenes/source_target_roles.py SourceTargetRoles

Render with transparent background (for overlaying in a video editor):

    manim -qh -t scenes/source_target_roles.py SourceTargetRoles
"""

from pathlib import Path

from manim import (
    BOLD,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Cross,
    FadeIn,
    FadeOut,
    Flash,
    Group,
    ImageMobject,
    Indicate,
    LaggedStart,
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
    caption = Text(title, font=FONT, weight=BOLD, color=TEXT_COLOR, font_size=28)
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


class SourceTargetRoles(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND

        # --- Phase 1: name the two roles ------------------------------------
        title = Text(
            "Người nguồn  ·  Người đích",
            font=FONT,
            weight=BOLD,
            color=TEXT_COLOR,
            font_size=38,
        ).to_edge(UP, buff=0.4)

        source = photo_card("source", "Người nguồn", ACCENT_BLUE)
        target = photo_card("target", "Người đích", ACCENT_YELLOW)
        source.move_to(LEFT * 3.55 + UP * 0.35)
        target.move_to(RIGHT * 3.55 + UP * 0.35)

        source_chip = chip("Danh tính khuôn mặt", ACCENT_BLUE, "scan-face.svg")
        source_chip.next_to(source, DOWN, buff=0.3)

        target_row1 = VGroup(
            chip("Biểu cảm", ACCENT_PINK, "smile.svg", font_size=22),
            chip("Góc mặt", ACCENT_PURPLE, "rotate-ccw.svg", font_size=22),
        ).arrange(RIGHT, buff=0.22)
        target_row2 = VGroup(
            chip("Ánh sáng", ACCENT_YELLOW, "sun.svg", font_size=22),
            chip("Màu da", ACCENT_GREEN, "palette.svg", font_size=22),
        ).arrange(RIGHT, buff=0.22)
        target_chips = VGroup(target_row1, target_row2).arrange(DOWN, buff=0.18)
        target_chips.next_to(target, DOWN, buff=0.3)

        self.play(Write(title), run_time=0.9)
        self.play(
            FadeIn(source, shift=RIGHT * 0.35),
            FadeIn(target, shift=LEFT * 0.35),
            run_time=0.9,
        )
        self.play(FadeIn(source_chip, shift=UP * 0.25), run_time=0.55)
        self.play(
            Indicate(source_chip, color=ACCENT_BLUE, scale_factor=1.08),
            run_time=0.55,
        )
        flat_chips = [*target_row1, *target_row2]
        self.play(
            LaggedStart(
                *[FadeIn(c, shift=UP * 0.2) for c in flat_chips],
                lag_ratio=0.22,
            ),
            run_time=1.4,
        )
        self.wait(1.4)

        # --- Phase 2: not cut-and-paste -------------------------------------
        self.play(
            FadeOut(Group(title, source, target, source_chip, target_chips)),
            run_time=0.7,
        )

        wrong_title = Text(
            "Không phải cắt & dán",
            font=FONT,
            weight=BOLD,
            color=TEXT_COLOR,
            font_size=40,
        ).to_edge(UP, buff=0.55)

        scissors = SVGMobject(ICONS_DIR / "scissors.svg")
        scissors.set_color(ACCENT_RED).set_height(0.85)
        cut_label = Text(
            "Cắt mặt nguồn → dán lên đích",
            font=FONT,
            color=MUTED_TEXT,
            font_size=28,
        )
        cut_group = VGroup(scissors, cut_label).arrange(RIGHT, buff=0.4)
        cut_box = RoundedRectangle(
            corner_radius=0.18,
            width=cut_group.width + 1.0,
            height=cut_group.height + 0.7,
            fill_color=CARD_FILL,
            fill_opacity=1.0,
            stroke_color=ACCENT_RED,
            stroke_width=3,
        )
        cut_group.move_to(cut_box)
        cut_card = VGroup(cut_box, cut_group)
        cut_card.move_to(UP * 0.55)

        cross = Cross(cut_card, stroke_color=ACCENT_RED, stroke_width=8)
        cross.set_z_index(4)

        right_caption = Text(
            "Mô hình chuyển danh tính — giữ biểu cảm & bối cảnh",
            font=FONT,
            weight=BOLD,
            color=ACCENT_GREEN,
            font_size=30,
        ).to_edge(DOWN, buff=0.7)

        self.play(Write(wrong_title), run_time=0.8)
        self.play(FadeIn(cut_card, scale=0.9), run_time=0.7)
        self.play(FadeIn(cross, scale=1.2), run_time=0.55)
        self.play(FadeIn(right_caption, shift=UP * 0.25), run_time=0.7)
        self.wait(1.5)

        # --- Phase 3: concrete example with the three photos ----------------
        self.play(
            FadeOut(VGroup(wrong_title, cut_card, cross, right_caption)),
            run_time=0.7,
        )

        example_title = Text(
            "Ví dụ",
            font=FONT,
            weight=BOLD,
            color=TEXT_COLOR,
            font_size=38,
        ).to_edge(UP, buff=0.35)

        ex_source = photo_card("source", "Nguồn", ACCENT_BLUE, height=3.45)
        ex_target = photo_card("target", "Đích", ACCENT_YELLOW, height=3.45)
        ex_result = photo_card("result", "Kết quả", ACCENT_GREEN, height=3.45)
        trio = Group(ex_source, ex_target, ex_result).arrange(RIGHT, buff=0.55)
        trio.move_to(UP * 0.15)

        keep_note = chip(
            "Giữ biểu cảm + góc quay của đích", ACCENT_YELLOW, "smile.svg", 22
        )
        keep_note.next_to(ex_target, DOWN, buff=0.28)

        id_note = chip(
            "Mang danh tính của nguồn", ACCENT_BLUE, "scan-face.svg", 22
        )
        id_note.next_to(ex_result, DOWN, buff=0.28)
        # Keep the two notes from overlapping when the middle/right cards are close.
        notes = VGroup(keep_note, id_note).arrange(RIGHT, buff=0.45)
        notes.next_to(trio, DOWN, buff=0.32)
        # Re-align each note under its card after the arrange.
        keep_note.set_x(ex_target.get_center()[0])
        id_note.set_x(ex_result.get_center()[0])

        self.play(Write(example_title), run_time=0.7)
        self.play(
            LaggedStart(
                FadeIn(ex_source, shift=UP * 0.3),
                FadeIn(ex_target, shift=UP * 0.3),
                FadeIn(ex_result, shift=UP * 0.3),
                lag_ratio=0.3,
            ),
            run_time=1.5,
        )
        self.play(FadeIn(keep_note, shift=UP * 0.2), run_time=0.55)
        self.play(
            Flash(ex_target[1].get_center(), color=ACCENT_YELLOW,
                  line_length=0.2, flash_radius=0.95, run_time=0.5)
        )
        self.play(FadeIn(id_note, shift=UP * 0.2), run_time=0.55)
        self.play(
            Flash(ex_result[1].get_center(), color=ACCENT_BLUE,
                  line_length=0.2, flash_radius=0.95, run_time=0.5)
        )
        self.wait(1.4)

        # --- Phase 4: one-line summary --------------------------------------
        summary_alt = VGroup(
            Text("Danh tính", font=FONT, weight=BOLD, color=ACCENT_BLUE, font_size=30),
            Text(" ← nguồn", font=FONT, weight=BOLD, color=TEXT_COLOR, font_size=30),
            Text("    ·    ", font=FONT, color=MUTED_TEXT, font_size=30),
            Text("Biểu cảm & bối cảnh", font=FONT, weight=BOLD,
                 color=ACCENT_YELLOW, font_size=30),
            Text(" ← đích", font=FONT, weight=BOLD, color=TEXT_COLOR, font_size=30),
        ).arrange(RIGHT, buff=0.08)
        summary_alt.to_edge(DOWN, buff=0.4)

        self.play(
            FadeOut(Group(keep_note, id_note)),
            FadeIn(summary_alt, shift=UP * 0.25),
            run_time=0.9,
        )
        self.wait(2.0)

        # --- Clean exit -----------------------------------------------------
        self.play(
            FadeOut(Group(example_title, trio, summary_alt)),
            run_time=0.8,
        )
        self.wait(0.3)
