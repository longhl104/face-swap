"""Animated title graphic: "Deep Learning + Fake = Deepfake".

Used in Part 1 of the video (0:20-1:30, "Deepfake va Face Swap" section).

Render (from the video_graphics/ folder):

    manim -pqh scenes/deepfake_equation.py DeepfakeEquation

Render with transparent background (for overlaying in a video editor):

    manim -qh -t scenes/deepfake_equation.py DeepfakeEquation
"""

from manim import (
    BOLD,
    DOWN,
    UP,
    Circumscribe,
    FadeIn,
    FadeOut,
    ReplacementTransform,
    Scene,
    Text,
    VGroup,
    Write,
)

BACKGROUND = "#0d1117"
DEEP_COLOR = "#4da6ff"  # blue for "Deep Learning"
FAKE_COLOR = "#ff5c5c"  # red for "Fake"
TEXT_COLOR = "#f0f2f5"
FONT = "Arial"


class DeepfakeEquation(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND

        deep_learning = Text(
            "Deep Learning", font=FONT, weight=BOLD, color=DEEP_COLOR, font_size=72
        )
        plus = Text("+", font=FONT, weight=BOLD, color=TEXT_COLOR, font_size=72)
        fake = Text("Fake", font=FONT, weight=BOLD, color=FAKE_COLOR, font_size=72)

        equation = VGroup(deep_learning, plus, fake).arrange(buff=0.55)
        equation.move_to(UP * 0.9)

        # "Deepfake" keeps the colors of the two words it was built from.
        deepfake = Text(
            "Deepfake",
            font=FONT,
            weight=BOLD,
            font_size=110,
            t2c={"Deep": DEEP_COLOR, "fake": FAKE_COLOR},
        )
        deepfake.move_to(DOWN * 1.0)

        equals = Text("=", font=FONT, weight=BOLD, color=TEXT_COLOR, font_size=72)
        equals.move_to((equation.get_bottom() + deepfake.get_top()) / 2)

        # 1. The two ingredients appear.
        self.play(Write(deep_learning), run_time=1.2)
        self.play(FadeIn(plus, scale=1.6), Write(fake), run_time=0.9)
        self.wait(0.6)

        # 2. The equals sign appears.
        self.play(FadeIn(equals, shift=DOWN * 0.3), run_time=0.5)

        # 3. "Deep" and "Fake" fly together to form "Deepfake";
        #    the leftover word "Learning" and the "+" fade away.
        deep_part = deep_learning[:4].copy()  # glyphs of "Deep"
        fake_part = fake.copy()
        self.add(deep_part, fake_part)

        self.play(
            ReplacementTransform(deep_part, deepfake[:4]),
            ReplacementTransform(fake_part, deepfake[4:]),
            deep_learning.animate.set_opacity(0.25),
            plus.animate.set_opacity(0.25),
            fake.animate.set_opacity(0.25),
            run_time=1.3,
        )

        # 4. Emphasize the result.
        self.play(Circumscribe(deepfake, color=FAKE_COLOR, buff=0.25), run_time=1.2)
        self.wait(1.2)

        # 5. Clean exit so the clip cuts nicely in the editor.
        self.play(
            FadeOut(VGroup(deep_learning, plus, fake, equals, deepfake)),
            run_time=0.8,
        )
        self.wait(0.3)
