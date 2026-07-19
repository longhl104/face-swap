"""Animated graphic: the forms deepfake content can take, and what it can do.

Illustrates VIDEO_SCRIPTS.md Part 1, 0:20-1:30 (line 40): deepfake can appear
as images, video or audio; it can make someone appear to say things they never
said, mimic a voice, change expressions or put a face into another video.

Render (from the video_graphics/ folder):

    manim -pqh scenes/deepfake_forms.py DeepfakeForms

Render with transparent background (for overlaying in a video editor):

    manim -qh -t scenes/deepfake_forms.py DeepfakeForms
"""

from pathlib import Path

from manim import (
    BOLD,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Circle,
    FadeIn,
    FadeOut,
    Flash,
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
ACCENT_BLUE = "#4da6ff"
ACCENT_YELLOW = "#f5e663"
ACCENT_RED = "#ff5c5c"
ACCENT_GREEN = "#b6e05f"
ACCENT_PURPLE = "#b493ea"
ACCENT_PINK = "#f96fa8"
CARD_FILL = "#1d2a3a"
FONT = "Arial"
ICONS_DIR = Path(__file__).resolve().parents[1] / "assets" / "icons"


def form_card(icon_file: str, label: str, color: str) -> VGroup:
    """A big rounded card with an icon on top and a label underneath."""
    icon = SVGMobject(ICONS_DIR / icon_file)
    icon.set_color(color).set_height(1.05)
    text = Text(label, font=FONT, weight=BOLD, color=TEXT_COLOR, font_size=34)
    content = VGroup(icon, text).arrange(DOWN, buff=0.4)
    box = RoundedRectangle(
        corner_radius=0.22,
        width=3.3,
        height=2.7,
        fill_color=CARD_FILL,
        fill_opacity=1.0,
        stroke_color=color,
        stroke_width=3.5,
    )
    content.move_to(box)
    return VGroup(box, content)


def capability_item(icon_file: str, label: str, color: str) -> VGroup:
    """A round icon badge with a caption to its right."""
    badge = Circle(
        radius=0.46,
        fill_color=CARD_FILL,
        fill_opacity=1.0,
        stroke_color=color,
        stroke_width=3,
    )
    icon = SVGMobject(ICONS_DIR / icon_file)
    icon.set_color(color).set_height(0.5)
    icon.move_to(badge)
    text = Text(label, font=FONT, color=TEXT_COLOR, font_size=30)
    text.next_to(badge, RIGHT, buff=0.35)
    return VGroup(badge, icon, text)


class DeepfakeForms(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND

        # --- Phase 1: deepfake comes in many forms ---------------------------
        title = Text(
            "Deepfake có thể xuất hiện dưới nhiều dạng",
            font=FONT,
            weight=BOLD,
            color=TEXT_COLOR,
            font_size=40,
        ).to_edge(UP, buff=0.7)

        forms = VGroup(
            form_card("image.svg", "Hình ảnh", ACCENT_BLUE),
            form_card("video.svg", "Video", ACCENT_YELLOW),
            form_card("audio-lines.svg", "Âm thanh", ACCENT_RED),
        ).arrange(RIGHT, buff=0.85)
        forms.move_to(DOWN * 0.55)

        self.play(Write(title), run_time=1.0)
        self.play(
            LaggedStart(
                *[FadeIn(card, shift=UP * 0.45, scale=0.9) for card in forms],
                lag_ratio=0.35,
            ),
            run_time=1.8,
        )
        self.wait(1.0)

        # --- Phase 2: the form cards shrink into a top row, capabilities ----
        #     appear below one by one.
        question = Text(
            "Nó có thể làm gì?",
            font=FONT,
            weight=BOLD,
            color=TEXT_COLOR,
            font_size=40,
        ).to_edge(UP, buff=0.7)

        forms_target = forms.copy().scale(0.52)
        forms_target.arrange(RIGHT, buff=0.55).next_to(question, DOWN, buff=0.5)

        self.play(
            FadeOut(title, shift=UP * 0.3),
            forms.animate.become(forms_target),
            run_time=1.0,
        )
        self.play(FadeIn(question, shift=DOWN * 0.3), run_time=0.7)

        capabilities = VGroup(
            capability_item(
                "message-square-quote.svg", "Nói điều họ chưa từng nói", ACCENT_GREEN
            ),
            capability_item("mic.svg", "Bắt chước giọng nói", ACCENT_PURPLE),
            capability_item("smile.svg", "Thay đổi biểu cảm", ACCENT_PINK),
            capability_item(
                "scan-face.svg", "Ghép mặt vào video khác", ACCENT_BLUE
            ),
        )
        # Two columns, two rows; badges of each column left-aligned.
        for i, item in enumerate(capabilities):
            row, col = divmod(i, 2)
            item.move_to(
                LEFT * 6.1 + RIGHT * col * 6.6 + DOWN * (0.85 + row * 1.75),
                aligned_edge=LEFT,
            )

        for item in capabilities:
            badge = item[0]
            self.play(FadeIn(item, shift=RIGHT * 0.4), run_time=0.6)
            self.play(
                Flash(badge, color=badge.get_stroke_color(), line_length=0.18,
                      flash_radius=0.6, run_time=0.5)
            )

        self.wait(1.6)

        # --- Clean exit so the clip cuts nicely in the editor ----------------
        self.play(FadeOut(VGroup(question, forms, capabilities)), run_time=0.8)
        self.wait(0.3)
