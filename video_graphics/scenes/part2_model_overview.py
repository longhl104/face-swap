"""Animated graphic: how the face swap model works (FaceNet + Generator/U-Net).

Illustrates VIDEO_SCRIPTS_2.md Part 2, 0:35-1:55 (Mô hình hoạt động như thế nào?):
- face swap needs two jobs: know who the face is, and redraw it on the target
- FaceNet (pretrained) turns a face into a vector of numbers = identity
- Generator takes identity (source) + pose/expression/lighting (target)
- it compresses the image into a small summary, then redraws the face;
  naive compress-and-restore loses eye/mouth positions
- U-Net adds skip connections between the compress and redraw halves;
  originally proposed for biomedical (cell) segmentation (separate screen)
- Discriminator + loss functions are saved for later episodes

Uses assets/images/{source-1,target-1,result-1,unet-origin-*,unet-enc-*,
unet-dec-*,unet-bottleneck}.* when present.
Generate the U-Net stage images with:

    python scripts/generate_unet_presentation.py

Render (from the video_graphics/ folder):

    manim -pqh scenes/part2_model_overview.py Part2ModelOverview

Render with transparent background (for overlaying in a video editor):

    manim -qh -t scenes/part2_model_overview.py Part2ModelOverview
"""

from pathlib import Path

from manim import (
    BOLD,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    CurvedArrow,
    DashedVMobject,
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

PHOTO_W, PHOTO_H = 2.1, 1.8


def find_image(stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".JPG", ".webp"):
        path = IMAGES_DIR / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def photo(stem: str, color: str, width: float = PHOTO_W) -> Group:
    path = find_image(stem)
    if path is not None:
        img = ImageMobject(str(path))
        img.scale_to_fit_width(width)
        return Group(img)
    icon = SVGMobject(ICONS_DIR / "image.svg")
    icon.set_color(color).set_height(0.7)
    return Group(icon)


def framed_photo(stem: str, label: str, color: str,
                 width: float = PHOTO_W) -> Group:
    content = photo(stem, color, width)
    frame = RoundedRectangle(
        corner_radius=0.14,
        width=content.width + 0.3,
        height=content.height + 0.3,
        fill_color=CARD_FILL,
        fill_opacity=1.0,
        stroke_color=color,
        stroke_width=3,
    )
    content.move_to(frame)
    caption = Text(label, font=FONT, weight=BOLD, color=TEXT_COLOR,
                   font_size=21)
    if caption.width > frame.width + 0.6:
        caption.scale_to_fit_width(frame.width + 0.6)
    caption.next_to(frame, DOWN, buff=0.18)
    return Group(frame, content, caption)


def box(label: str, color: str, font_size: int = 27,
        pad_w: float = 0.9, pad_h: float = 0.65) -> VGroup:
    txt = Text(label, font=FONT, weight=BOLD, color=TEXT_COLOR,
               font_size=font_size, line_spacing=0.9)
    rect = RoundedRectangle(
        corner_radius=0.16,
        width=txt.width + pad_w,
        height=txt.height + pad_h,
        fill_color=CARD_FILL,
        fill_opacity=1.0,
        stroke_color=color,
        stroke_width=3.2,
    )
    txt.move_to(rect)
    return VGroup(rect, txt)


def chip(text: str, color: str, font_size: int = 23) -> VGroup:
    txt = Text(text, font=FONT, weight=BOLD, color=TEXT_COLOR,
               font_size=font_size)
    rect = RoundedRectangle(
        corner_radius=0.12,
        width=txt.width + 0.5,
        height=txt.height + 0.34,
        fill_color=CARD_FILL,
        fill_opacity=1.0,
        stroke_color=color,
        stroke_width=2.6,
    )
    txt.move_to(rect)
    return VGroup(rect, txt)


class Part2ModelOverview(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND

        # --- Phase 1: the two jobs -------------------------------------------
        title = Text(
            "Face swap cần làm hai việc",
            font=FONT,
            weight=BOLD,
            color=TEXT_COLOR,
            font_size=36,
        ).to_edge(UP, buff=0.5)

        job1 = box('1. Biết khuôn mặt "là ai"', ACCENT_BLUE, 26)
        job2 = box("2. Vẽ lại khuôn mặt lên người đích", ACCENT_PINK, 26)
        jobs = VGroup(job1, job2).arrange(DOWN, buff=0.5)
        jobs.move_to(DOWN * 0.2)

        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(job1, shift=UP * 0.25), run_time=0.7)
        self.play(FadeIn(job2, shift=UP * 0.25), run_time=0.7)
        self.wait(1.0)
        self.play(FadeOut(VGroup(title, jobs)), run_time=0.6)

        # --- Phase 2: FaceNet turns a face into identity numbers -------------
        fn_title = Text(
            "1. Nhận biết danh tính: FaceNet",
            font=FONT,
            weight=BOLD,
            color=ACCENT_BLUE,
            font_size=32,
        ).to_edge(UP, buff=0.5)

        src = framed_photo("source-1", "Source", ACCENT_BLUE)
        src.move_to(LEFT * 4.6 + DOWN * 0.1)

        facenet = box("FaceNet", ACCENT_BLUE, 30)
        facenet.move_to(LEFT * 0.9 + DOWN * 0.1)
        pretrained = Text(
            "huấn luyện sẵn",
            font=FONT,
            color=MUTED_TEXT,
            font_size=20,
        ).next_to(facenet, DOWN, buff=0.18)

        vector = Text(
            "[ 0.24, -0.81,  0.05,\n  0.63,  0.17, ... ]",
            font=FONT,
            color=ACCENT_GREEN,
            font_size=24,
            line_spacing=0.9,
        )
        vec_box = RoundedRectangle(
            corner_radius=0.14,
            width=vector.width + 0.6,
            height=vector.height + 0.5,
            fill_color=CARD_FILL,
            fill_opacity=1.0,
            stroke_color=ACCENT_GREEN,
            stroke_width=3,
        )
        vector.move_to(vec_box)
        vec_label = Text(
            "dãy số = danh tính",
            font=FONT,
            weight=BOLD,
            color=TEXT_COLOR,
            font_size=21,
        )
        vec_group = VGroup(vec_box, vector)
        vec_group.move_to(RIGHT * 3.9 + DOWN * 0.1)
        vec_label.next_to(vec_group, DOWN, buff=0.2)

        arr_in = Arrow(src.get_right() + RIGHT * 0.02, facenet.get_left(),
                       buff=0.12, color=ACCENT_BLUE, stroke_width=5)
        arr_out = Arrow(facenet.get_right(), vec_box.get_left(),
                        buff=0.12, color=ACCENT_GREEN, stroke_width=5)

        self.play(Write(fn_title), run_time=0.8)
        self.play(FadeIn(src, shift=RIGHT * 0.3), run_time=0.6)
        self.play(FadeIn(arr_in), FadeIn(facenet), FadeIn(pretrained),
                  run_time=0.7)
        self.play(FadeIn(arr_out), FadeIn(vec_group), run_time=0.7)
        self.play(FadeIn(vec_label, shift=UP * 0.2), run_time=0.5)
        self.wait(1.3)
        self.play(
            FadeOut(Group(fn_title, src, facenet, pretrained,
                          arr_in, arr_out, vec_group, vec_label)),
            run_time=0.6,
        )

        # --- Phase 3: Generator combines identity + target -------------------
        gen_title = Text(
            "2. Vẽ lại khuôn mặt: Generator",
            font=FONT,
            weight=BOLD,
            color=ACCENT_PINK,
            font_size=32,
        ).to_edge(UP, buff=0.5)

        id_chip = chip("Danh tính (source)", ACCENT_BLUE)
        attr_chip = chip("Góc mặt, biểu cảm,\nánh sáng (target)",
                         ACCENT_YELLOW)
        id_chip.move_to(LEFT * 3.9 + UP * 1.15)
        attr_chip.move_to(LEFT * 2.65 + DOWN * 1.35)

        generator = box("Generator", ACCENT_PINK, 30)
        generator.move_to(RIGHT * 1.45 + DOWN * 0.1)

        out = framed_photo("result-1", "Khuôn mặt mới", ACCENT_GREEN,
                           width=1.9)
        out.move_to(RIGHT * 5.0 + DOWN * 0.1)

        arr_id = Arrow(id_chip.get_bottom() + DOWN * 0.05,
                       generator.get_corner(UP + LEFT) + RIGHT * 0.35,
                       buff=0.12, color=ACCENT_BLUE, stroke_width=5)
        arr_attr = Arrow(attr_chip.get_right(),
                         generator.get_corner(DOWN + LEFT) + UP * 0.28,
                         buff=0.12, color=ACCENT_YELLOW, stroke_width=5)
        arr_gen = Arrow(generator.get_right(), out.get_left() + LEFT * 0.02,
                        buff=0.12, color=ACCENT_GREEN, stroke_width=5)

        self.play(Write(gen_title), run_time=0.8)
        self.play(FadeIn(generator, scale=0.92), run_time=0.6)
        self.play(FadeIn(id_chip, shift=DOWN * 0.2), FadeIn(arr_id),
                  run_time=0.7)
        self.play(FadeIn(attr_chip, shift=UP * 0.2), FadeIn(arr_attr),
                  run_time=0.7)
        self.play(FadeIn(arr_gen), FadeIn(out, shift=LEFT * 0.25),
                  run_time=0.7)
        self.wait(1.3)
        self.play(
            FadeOut(Group(gen_title, id_chip, attr_chip, generator, out,
                          arr_id, arr_attr, arr_gen)),
            run_time=0.6,
        )

        # --- Phase 4: compress -> summary -> redraw ---------------------------
        cp_title = Text(
            "Nén ảnh thành bản tóm tắt, rồi vẽ lại",
            font=FONT,
            weight=BOLD,
            color=TEXT_COLOR,
            font_size=32,
        ).to_edge(UP, buff=0.5)

        big_img = framed_photo("target-1", "Ảnh gốc", ACCENT_YELLOW,
                               width=1.9)
        big_img.move_to(LEFT * 4.9 + UP * 0.15)

        summary = VGroup(
            Text("cười", font=FONT, color=TEXT_COLOR, font_size=19),
            Text("quay trái", font=FONT, color=TEXT_COLOR, font_size=19),
            Text("sáng ấm", font=FONT, color=TEXT_COLOR, font_size=19),
        ).arrange(DOWN, buff=0.12)
        sum_box = RoundedRectangle(
            corner_radius=0.12,
            width=summary.width + 0.5,
            height=summary.height + 0.4,
            fill_color=CARD_FILL,
            fill_opacity=1.0,
            stroke_color=ACCENT_PURPLE,
            stroke_width=3,
        )
        summary.move_to(sum_box)
        sum_label = Text("bản tóm tắt nhỏ", font=FONT, weight=BOLD,
                         color=ACCENT_PURPLE, font_size=20)
        sum_group = VGroup(sum_box, summary)
        sum_group.move_to(UP * 0.15)
        sum_label.next_to(sum_group, DOWN, buff=0.2)

        redraw = framed_photo("result-1", "Vẽ lại", ACCENT_RED, width=1.9)
        redraw.move_to(RIGHT * 4.9 + UP * 0.15)
        # Blur the redrawn photo slightly by fading it — stands for lost detail.
        redraw[1].set_opacity(0.55)

        arr_c1 = Arrow(big_img.get_right(), sum_box.get_left(), buff=0.15,
                       color=ACCENT_PURPLE, stroke_width=5)
        c1_label = Text("nén", font=FONT, color=MUTED_TEXT, font_size=20)
        c1_label.next_to(arr_c1, UP, buff=0.12)
        arr_c2 = Arrow(sum_box.get_right(), redraw.get_left() + LEFT * 0.02,
                       buff=0.15, color=ACCENT_RED, stroke_width=5)
        c2_label = Text("vẽ lại", font=FONT, color=MUTED_TEXT, font_size=20)
        c2_label.next_to(arr_c2, UP, buff=0.12)

        self.play(Write(cp_title), run_time=0.8)
        self.play(FadeIn(big_img, shift=RIGHT * 0.3), run_time=0.6)
        self.play(FadeIn(arr_c1), FadeIn(c1_label),
                  FadeIn(sum_group, scale=0.9), FadeIn(sum_label),
                  run_time=0.8)
        self.play(FadeIn(arr_c2), FadeIn(c2_label),
                  FadeIn(redraw, shift=LEFT * 0.25), run_time=0.8)
        self.play(Indicate(redraw[0], color=ACCENT_RED, scale_factor=1.05),
                  run_time=0.7)
        self.wait(1.2)
        self.play(
            FadeOut(Group(cp_title, big_img, sum_group, sum_label, redraw,
                          arr_c1, arr_c2, c1_label, c2_label)),
            run_time=0.6,
        )

        # --- Phase 5: U-Net skip connections ----------------------------------
        un_title = Text(
            "U-Net: đường nối tắt (skip connections)\ngiữ lại chi tiết",
            font=FONT,
            weight=BOLD,
            color=ACCENT_GREEN,
            font_size=26,
            line_spacing=0.85,
        ).to_edge(UP, buff=0.25)

        def unet_stage(stem: str, width: float, color: str) -> Group:
            content = photo(stem, color, width)
            frame = RoundedRectangle(
                corner_radius=0.1,
                width=content.width + 0.14,
                height=content.height + 0.14,
                fill_color=CARD_FILL,
                fill_opacity=1.0,
                stroke_color=color,
                stroke_width=3,
            )
            content.move_to(frame)
            return Group(frame, content)

        stage_widths = (1.35, 1.0, 0.75)
        enc = Group(*[
            unet_stage(f"unet-enc-{i}", w, ACCENT_PURPLE)
            for i, w in enumerate(stage_widths, start=1)
        ])
        dec = Group(*[
            unet_stage(f"unet-dec-{i}", w, ACCENT_PINK)
            for i, w in enumerate(stage_widths, start=1)
        ])
        bottom = unet_stage("unet-bottleneck", 0.7, ACCENT_GREEN)

        # Explicit U layout: matching levels share Y; bottleneck sits below skips.
        col_x = 2.55
        level_y = (0.95, -0.85, -2.3)
        for i, (e, d) in enumerate(zip(enc, dec)):
            e.move_to(LEFT * col_x + UP * level_y[i])
            d.move_to(RIGHT * col_x + UP * level_y[i])
        bottom.move_to(DOWN * 3.35)

        enc_label = Text("nén", font=FONT, weight=BOLD, color=ACCENT_PURPLE,
                         font_size=22).next_to(enc[0], UP, buff=0.16)
        dec_label = Text("vẽ lại", font=FONT, weight=BOLD, color=ACCENT_PINK,
                         font_size=22).next_to(dec[0], UP, buff=0.16)

        def stage_arrow(start, end) -> Arrow:
            return Arrow(
                start, end, buff=0.1,
                color=MUTED_TEXT, stroke_width=4,
                max_tip_length_to_length_ratio=0.32,
                max_stroke_width_to_length_ratio=8,
            )

        down_arrows = VGroup(
            stage_arrow(enc[0].get_bottom(), enc[1].get_top()),
            stage_arrow(enc[1].get_bottom(), enc[2].get_top()),
            CurvedArrow(
                enc[2].get_bottom() + RIGHT * 0.15,
                bottom.get_left() + UP * 0.1,
                angle=0.75,
                color=MUTED_TEXT,
                stroke_width=4,
            ),
        )
        up_arrows = VGroup(
            CurvedArrow(
                bottom.get_right() + UP * 0.1,
                dec[2].get_bottom() + LEFT * 0.15,
                angle=0.75,
                color=MUTED_TEXT,
                stroke_width=4,
            ),
            stage_arrow(dec[2].get_top(), dec[1].get_bottom()),
            stage_arrow(dec[1].get_top(), dec[0].get_bottom()),
        )

        skips = VGroup(*[
            DashedVMobject(
                Arrow(
                    e.get_right(),
                    d.get_left(),
                    buff=0.12,
                    color=ACCENT_GREEN,
                    stroke_width=4,
                    max_tip_length_to_length_ratio=0.16,
                ),
                num_dashes=9,
                dashed_ratio=0.55,
            )
            for e, d in zip(enc, dec)
        ])
        self.play(Write(un_title), run_time=0.8)
        self.play(
            FadeIn(enc, shift=RIGHT * 0.2), FadeIn(enc_label),
            FadeIn(dec, shift=LEFT * 0.2), FadeIn(dec_label),
            FadeIn(bottom),
            run_time=0.9,
        )
        self.play(
            LaggedStart(*[FadeIn(a) for a in [*down_arrows, *up_arrows]],
                        lag_ratio=0.12),
            run_time=1.0,
        )
        self.play(
            LaggedStart(*[FadeIn(s) for s in skips], lag_ratio=0.25),
            run_time=1.1,
        )
        self.wait(1.5)
        self.play(
            FadeOut(Group(un_title, enc, dec, bottom, enc_label, dec_label,
                          down_arrows, up_arrows, skips)),
            run_time=0.6,
        )

        # --- Phase 6: U-Net biomedical origin ---------------------------------
        origin_title = Text(
            "U-Net: gốc từ ảnh y sinh",
            font=FONT,
            weight=BOLD,
            color=ACCENT_PURPLE,
            font_size=32,
        ).to_edge(UP, buff=0.5)

        bio_input = framed_photo(
            "unet-origin-input", "Ảnh kính hiển vi", ACCENT_PURPLE, width=2.5,
        )
        bio_output = framed_photo(
            "unet-origin-output", "Khoanh vùng tế bào", ACCENT_GREEN, width=2.5,
        )
        bio_pair = Group(bio_input, bio_output).arrange(RIGHT, buff=1.3)
        bio_pair.move_to(DOWN * 0.1)

        bio_arrow = Arrow(
            bio_input.get_right() + RIGHT * 0.02,
            bio_output.get_left(),
            buff=0.12,
            color=MUTED_TEXT,
            stroke_width=5,
        )

        origin_caption = VGroup(
            Text("Nguồn:", font=FONT, weight=BOLD, color=MUTED_TEXT,
                 font_size=17),
            Text(" U-Net: Convolutional Networks for Biomedical Image Segmentation",
                 font=FONT, color=MUTED_TEXT, font_size=17),
        ).arrange(RIGHT, buff=0.08, aligned_edge=DOWN).to_edge(DOWN, buff=0.3)
        if origin_caption.width > 12.5:
            origin_caption.scale_to_fit_width(12.5)

        self.play(Write(origin_title), run_time=0.8)
        self.play(FadeIn(bio_input, shift=RIGHT * 0.25), run_time=0.6)
        self.play(FadeIn(bio_arrow), FadeIn(bio_output, shift=LEFT * 0.25),
                  run_time=0.7)
        self.play(FadeIn(origin_caption), run_time=0.5)
        self.wait(1.5)
        self.play(
            FadeOut(Group(origin_title, bio_pair, bio_arrow, origin_caption)),
            run_time=0.6,
        )

        # --- Phase 7: summary + what comes later ------------------------------
        sm_title = Text(
            "Hai thành phần chính",
            font=FONT,
            weight=BOLD,
            color=TEXT_COLOR,
            font_size=34,
        ).to_edge(UP, buff=0.55)

        card_fn = box("FaceNet\ndanh tính", ACCENT_BLUE, 26)
        card_gen = box("Generator (U-Net)\nvẽ lại khuôn mặt", ACCENT_PINK, 26)
        cards = VGroup(card_fn, card_gen).arrange(RIGHT, buff=0.8)
        cards.move_to(UP * 0.35)

        later = VGroup(
            chip("Discriminator", MUTED_TEXT),
            chip("Loss functions", MUTED_TEXT),
            Text("→  các phần sau", font=FONT, color=MUTED_TEXT,
                 font_size=23),
        ).arrange(RIGHT, buff=0.35)
        later.next_to(cards, DOWN, buff=0.9)
        for item in later[:2]:
            item.set_opacity(0.7)

        self.play(Write(sm_title), run_time=0.7)
        self.play(
            LaggedStart(
                FadeIn(card_fn, shift=UP * 0.2),
                FadeIn(card_gen, shift=UP * 0.2),
                lag_ratio=0.3,
            ),
            run_time=1.0,
        )
        self.play(FadeIn(later, shift=UP * 0.2), run_time=0.7)
        self.wait(1.8)

        # --- Clean exit --------------------------------------------------------
        self.play(FadeOut(VGroup(sm_title, cards, later)), run_time=0.7)
        self.wait(0.3)
