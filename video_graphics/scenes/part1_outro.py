"""Animated Part 1 outro: flash face-swap results, then a Phần 2 title card.

Illustrates VIDEO_SCRIPTS.md Part 1, 2:15-2:45 (lines 66-74):
flash a few face-swap results, ask for comments, preview what Part 2
covers, then end on the "Phần 2" title card.

Result photos are loaded from assets/images/results/ (any png/jpg/jpeg/webp).
Until those files exist the scene renders labeled placeholders.

Render (from the video_graphics/ folder):

    manim -pqh scenes/part1_outro.py Part1Outro

Render with transparent background (for overlaying in a video editor):

    manim -qh -t scenes/part1_outro.py Part1Outro
"""

from pathlib import Path

from manim import (
    BOLD,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    FadeIn,
    FadeOut,
    Flash,
    Group,
    ImageMobject,
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
RESULTS_DIR = ASSETS_DIR / "images" / "results"

PHOTO_W, PHOTO_H = 3.15, 3.2
ACCENTS = [ACCENT_BLUE, ACCENT_YELLOW, ACCENT_GREEN, ACCENT_PINK, ACCENT_PURPLE]


def list_result_images() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        paths.extend(RESULTS_DIR.glob(ext))
    return sorted(paths)


def photo_from_path(path: Path | None, color: str, label: str) -> Group:
    frame = RoundedRectangle(
        corner_radius=0.18,
        width=3.4,
        height=3.55,
        fill_color=CARD_FILL,
        fill_opacity=1.0,
        stroke_color=color,
        stroke_width=3.5,
    )
    if path is not None:
        img = ImageMobject(str(path))
        img.scale(min((PHOTO_W) / img.width, (PHOTO_H) / img.height))
        content = Group(img)
    else:
        icon = SVGMobject(ICONS_DIR / "image.svg")
        icon.set_color(color).set_height(0.9)
        note = Text(label, font=FONT, color=MUTED_TEXT, font_size=22)
        content = Group(VGroup(icon, note).arrange(DOWN, buff=0.25))
    content.move_to(frame.get_center())
    return Group(frame, content)


def topic_card(icon_file: str, label: str, color: str) -> VGroup:
    icon = SVGMobject(ICONS_DIR / icon_file)
    icon.set_color(color).set_height(0.7)
    text = Text(label, font=FONT, weight=BOLD, color=TEXT_COLOR, font_size=26)
    content = VGroup(icon, text).arrange(DOWN, buff=0.3)
    box = RoundedRectangle(
        corner_radius=0.18,
        width=2.85,
        height=2.35,
        fill_color=CARD_FILL,
        fill_opacity=1.0,
        stroke_color=color,
        stroke_width=3.2,
    )
    content.move_to(box)
    return VGroup(box, content)


class Part1Outro(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND

        # --- Phase 1: flash a few face-swap results -------------------------
        flash_title = Text(
            "Kết quả Face Swap",
            font=FONT,
            weight=BOLD,
            color=TEXT_COLOR,
            font_size=40,
        ).to_edge(UP, buff=0.5)

        result_paths = list_result_images()
        n_slots = max(3, min(len(result_paths), 3))
        cards = Group()
        for i in range(n_slots):
            path = result_paths[i] if i < len(result_paths) else None
            label = path.name if path else f"result_{i + 1}.jpg"
            cards.add(photo_from_path(path, ACCENTS[i % len(ACCENTS)], label))
        cards.arrange(RIGHT, buff=0.55).move_to(DOWN * 0.35)

        self.play(Write(flash_title), run_time=0.8)
        for card in cards:
            self.play(FadeIn(card, scale=0.92), run_time=0.45)
            self.play(
                Flash(
                    card[1].get_center(),
                    color=card[0].get_stroke_color(),
                    line_length=0.22,
                    flash_radius=1.1,
                    run_time=0.45,
                )
            )
        self.wait(1.2)

        # --- Phase 2: soft ask for comments ---------------------------------
        self.play(FadeOut(Group(flash_title, cards)), run_time=0.7)

        comment_icon = SVGMobject(ICONS_DIR / "message-circle.svg")
        comment_icon.set_color(ACCENT_BLUE).set_height(0.85)
        comment_line = Text(
            "Góp ý ở phần bình luận nhé",
            font=FONT,
            weight=BOLD,
            color=TEXT_COLOR,
            font_size=38,
        )
        comment = VGroup(comment_icon, comment_line).arrange(RIGHT, buff=0.4)
        comment_sub = Text(
            "Lần đầu làm video dạng này — mọi góp ý đều hữu ích",
            font=FONT,
            color=MUTED_TEXT,
            font_size=26,
        )
        comment_block = VGroup(comment, comment_sub).arrange(DOWN, buff=0.45)

        self.play(FadeIn(comment_block, shift=UP * 0.3), run_time=0.9)
        self.wait(1.6)

        # --- Phase 3: preview of Part 2 topics ------------------------------
        self.play(FadeOut(comment_block), run_time=0.6)

        next_title = Text(
            "Trong phần tiếp theo",
            font=FONT,
            weight=BOLD,
            color=TEXT_COLOR,
            font_size=38,
        ).to_edge(UP, buff=0.55)

        topics = VGroup(
            topic_card("database.svg", "Dữ liệu", ACCENT_BLUE),
            topic_card("layers.svg", "Kiến trúc mô hình", ACCENT_YELLOW),
            topic_card("chart.svg", "Huấn luyện", ACCENT_PINK),
            topic_card("scan-face.svg", "Kết quả", ACCENT_GREEN),
        ).arrange(RIGHT, buff=0.4)
        topics.move_to(DOWN * 0.3)

        self.play(Write(next_title), run_time=0.8)
        self.play(
            LaggedStart(
                *[FadeIn(card, shift=UP * 0.35, scale=0.9) for card in topics],
                lag_ratio=0.28,
            ),
            run_time=1.8,
        )
        self.wait(1.6)

        # --- Phase 4: Phần 2 title card -------------------------------------
        self.play(FadeOut(VGroup(next_title, topics)), run_time=0.7)

        part_label = Text(
            "PHẦN 2",
            font=FONT,
            weight=BOLD,
            color=ACCENT_BLUE,
            font_size=96,
        )
        part_subtitle = Text(
            "Tôi đã xây dựng mô hình Face Swap như thế nào?",
            font=FONT,
            weight=BOLD,
            color=TEXT_COLOR,
            font_size=34,
        )
        goodbye = Text(
            "Hẹn gặp lại các bạn",
            font=FONT,
            color=MUTED_TEXT,
            font_size=28,
        )
        title_card = VGroup(part_label, part_subtitle, goodbye).arrange(
            DOWN, buff=0.45
        )

        self.play(FadeIn(part_label, scale=0.85), run_time=1.0)
        self.play(FadeIn(part_subtitle, shift=UP * 0.25), run_time=0.8)
        self.play(FadeIn(goodbye, shift=UP * 0.2), run_time=0.6)
        self.wait(2.2)

        self.play(FadeOut(title_card), run_time=0.9)
        self.wait(0.3)
