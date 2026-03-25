"""
TurboQuant algorithm visualizations for README.

Prerequisites:
    brew install --cask basictex    # LaTeX for MathTex
    uv add manim

Render all GIFs:
    uv run manim -ql scenes.py RandomRotation --format gif
    uv run manim -ql scenes.py PolarQuant --format gif
    uv run manim -ql scenes.py TurboQuantPipeline --format gif

High quality (for README):
    uv run manim -qh scenes.py RandomRotation --format gif
"""

from manim import *
import numpy as np


# Use xelatex → .xdv → dvisvgm (avoids PostScript font issues with homebrew dvisvgm)
_TEMPLATE = TexTemplate()
_TEMPLATE.tex_compiler = "xelatex"
_TEMPLATE.output_format = ".xdv"
MathTex.set_default(tex_template=_TEMPLATE)
Tex.set_default(tex_template=_TEMPLATE)


PLANE_STYLE = dict(
    background_line_style={"stroke_color": BLUE_E, "stroke_opacity": 0.35},
    axis_config={"stroke_color": WHITE, "stroke_opacity": 0.5},
)


# ─── Scene 1: Random Rotation ─────────────────────────────────────────────────

class RandomRotation(Scene):
    """
    A biased vector (large x₁, tiny x₂) is hard to quantize uniformly.
    After multiplying by a random rotation matrix R, both components
    become roughly equal — quantization error drops significantly.
    """

    def construct(self):
        title    = Text("Random Rotation", font_size=38, weight=BOLD)
        subtitle = Text("Balancing energy across dimensions", font_size=22, color=GRAY_B)
        VGroup(title, subtitle).arrange(DOWN, buff=0.15).to_edge(UP, buff=0.35)
        self.play(Write(title), FadeIn(subtitle, shift=UP * 0.2))

        plane = NumberPlane(
            x_range=[-3.8, 3.8], y_range=[-3.8, 3.8],
            x_length=6.4, y_length=6.4,
            **PLANE_STYLE,
        ).shift(DOWN * 0.35)
        self.play(Create(plane), run_time=0.7)

        v = np.array([3.0, 0.5])

        def arrow(vec, color):
            return Arrow(
                plane.c2p(0, 0), plane.c2p(*vec), buff=0,
                color=color, stroke_width=4,
                max_tip_length_to_length_ratio=0.15,
            )

        def components(vec, color):
            px = DashedLine(plane.c2p(vec[0], 0), plane.c2p(*vec), color=color, stroke_width=2)
            py = DashedLine(plane.c2p(0, vec[1]), plane.c2p(*vec), color=color, stroke_width=2)
            lx = MathTex(f"x_1={vec[0]:.1f}", font_size=26, color=color).next_to(
                plane.c2p(vec[0] / 2, 0), DOWN, buff=0.12)
            ly = MathTex(f"x_2={vec[1]:.1f}", font_size=26, color=color).next_to(
                plane.c2p(vec[0], vec[1] / 2), RIGHT, buff=0.1)
            return VGroup(px, py, lx, ly)

        arr_orig  = arrow(v, BLUE)
        comp_orig = components(v, BLUE_B)
        note_bad  = Text("Imbalanced → wasteful quantization", font_size=20, color=RED_B).to_edge(DOWN, buff=0.3)

        self.play(GrowArrow(arr_orig))
        self.play(Create(comp_orig))
        self.play(FadeIn(note_bad))
        self.wait(0.6)

        # Rotate by 75°
        theta = 75 * DEGREES
        R     = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta),  np.cos(theta)]])
        v_rot = R @ v

        rot_arc = Arc(
            radius=0.85, start_angle=np.arctan2(v[1], v[0]), angle=theta,
            color=YELLOW, stroke_width=3, arc_center=plane.c2p(0, 0),
        )
        rot_label = MathTex(r"\mathbf{R}", font_size=32, color=YELLOW).next_to(rot_arc, UR, buff=0.05)

        arr_rot   = arrow(v_rot, GREEN)
        comp_rot  = components(v_rot, GREEN_B)
        note_good = Text("Balanced → uniform quantization", font_size=20, color=GREEN_B).to_edge(DOWN, buff=0.3)

        self.play(Create(rot_arc), Write(rot_label))
        self.play(
            Transform(arr_orig,  arr_rot),
            Transform(comp_orig, comp_rot),
            Transform(note_bad,  note_good),
            run_time=1.2,
        )
        self.wait(1.5)


# ─── Scene 2: PolarQuant ──────────────────────────────────────────────────────

class PolarQuant(Scene):
    """
    (x, y) → (r, θ) conversion, followed by angle quantization.
    Because post-rotation angles are near-uniform, a fixed angular grid
    works well and requires zero per-block overhead.
    """

    def construct(self):
        title    = Text("PolarQuant", font_size=38, weight=BOLD)
        subtitle = Text("Cartesian → Polar → Quantize angles", font_size=22, color=GRAY_B)
        VGroup(title, subtitle).arrange(DOWN, buff=0.15).to_edge(UP, buff=0.35)
        self.play(Write(title), FadeIn(subtitle, shift=UP * 0.2))

        plane = NumberPlane(
            x_range=[-0.4, 4.3], y_range=[-0.4, 3.9],
            x_length=6.0, y_length=5.0,
            **PLANE_STYLE,
        ).shift(LEFT * 0.9 + DOWN * 0.6)
        self.play(Create(plane), run_time=0.7)

        x, y  = 3.0, 2.0
        r     = np.hypot(x, y)       # ≈ 3.61
        theta = np.arctan2(y, x)     # ≈ 33.7°

        # ── Step 1: Cartesian ─────────────────────────────────────────────────
        point = Dot(plane.c2p(x, y), color=BLUE, radius=0.1)
        vec   = Arrow(plane.c2p(0, 0), plane.c2p(x, y), buff=0,
                      color=BLUE, stroke_width=4, max_tip_length_to_length_ratio=0.13)

        dx = DashedLine(plane.c2p(x, 0), plane.c2p(x, y), color=GRAY_B, stroke_width=2)
        dy = DashedLine(plane.c2p(0, y), plane.c2p(x, y), color=GRAY_B, stroke_width=2)
        lx = MathTex("x", font_size=28, color=YELLOW).next_to(plane.c2p(x, 0), DOWN, buff=0.12)
        ly = MathTex("y", font_size=28, color=YELLOW).next_to(plane.c2p(0, y), LEFT, buff=0.12)

        cart_eq = MathTex(r"(x,y)=(3.0,\;2.0)", font_size=26, color=BLUE).to_corner(UR, buff=0.45)

        self.play(GrowArrow(vec), FadeIn(point))
        self.play(Create(dx), Create(dy), Write(lx), Write(ly), Write(cart_eq))
        self.wait(0.5)

        # ── Step 2: Polar form ────────────────────────────────────────────────
        arc_theta = Arc(
            radius=0.85, start_angle=0, angle=theta,
            color=GREEN, stroke_width=3, arc_center=plane.c2p(0, 0),
        )
        lbl_theta = MathTex(r"\theta", font_size=28, color=GREEN).move_to(plane.c2p(0.75, 0.28))
        lbl_r     = MathTex(r"r",      font_size=28, color=ORANGE).move_to(
            plane.c2p(x / 2 - 0.2, y / 2 + 0.22))
        polar_eq  = MathTex(r"(r,\theta)=(3.6,\;33.7°)", font_size=26, color=GREEN).next_to(
            cart_eq, DOWN, buff=0.22)

        self.play(
            FadeOut(dx), FadeOut(dy), FadeOut(lx), FadeOut(ly),
            Create(arc_theta), Write(lbl_theta), Write(lbl_r),
        )
        self.play(Write(polar_eq))
        self.wait(0.5)

        # ── Step 3: Quantize angle ────────────────────────────────────────────
        n_levels = 8
        step     = 2 * PI / n_levels
        quant_lines = VGroup(*[
            Line(
                plane.c2p(0, 0),
                plane.c2p(3.9 * np.cos(i * step), 3.9 * np.sin(i * step)),
                stroke_width=1.2, color=RED_E,
            )
            for i in range(n_levels)
        ])
        quant_note = Text("8 angular levels  (3 bits)", font_size=20, color=RED_B).to_edge(DOWN, buff=0.3)

        self.play(Create(quant_lines), Write(quant_note))

        theta_q = round(theta / step) * step    # snaps to π/4 = 45°
        vec_q   = Arrow(
            plane.c2p(0, 0),
            plane.c2p(r * np.cos(theta_q), r * np.sin(theta_q)),
            buff=0, color=RED, stroke_width=4, max_tip_length_to_length_ratio=0.13,
        )
        quant_eq = MathTex(r"\hat{\theta}=45°\;\rightarrow\;3\;\text{bits}",
                           font_size=24, color=RED).next_to(polar_eq, DOWN, buff=0.2)

        self.play(Transform(vec, vec_q), run_time=0.9)
        self.play(Write(quant_eq))
        self.wait(1.5)


# ─── Scene 3: TurboQuant Pipeline ────────────────────────────────────────────

class TurboQuantPipeline(Scene):
    """
    Full TurboQuant pipeline: Rotation → PolarQuant → QJL.
    Memory shrinks from 32 bits → ~3 bits per KV cache element.
    """

    def construct(self):
        title = Text("TurboQuant  Pipeline", font_size=40, weight=BOLD).to_edge(UP, buff=0.35)
        self.play(Write(title))

        def make_step(num, name, detail, color):
            box     = RoundedRectangle(
                corner_radius=0.2, width=2.85, height=2.0,
                color=color, fill_color=color, fill_opacity=0.08, stroke_width=2,
            )
            n_label = Text(f"Step {num}", font_size=15, color=color)
            t_label = Text(name,   font_size=20, weight=BOLD, color=color)
            d_label = Text(detail, font_size=14, color=GRAY_B, line_spacing=1.4)
            VGroup(n_label, t_label, d_label).arrange(DOWN, buff=0.1)
            return VGroup(box, n_label, t_label, d_label)

        s1 = make_step(1, "Random Rotation", "Uniform distribution\nacross dimensions", BLUE)
        s2 = make_step(2, "PolarQuant",      "Cartesian → Polar\nQuantize angles",       GREEN)
        s3 = make_step(3, "QJL  (1 bit)",    "Residual error\ncorrection",               ORANGE)

        VGroup(s1, s2, s3).arrange(RIGHT, buff=0.72).shift(DOWN * 0.1)

        arr12 = Arrow(s1.get_right(), s2.get_left(), buff=0.05, color=WHITE, stroke_width=2)
        arr23 = Arrow(s2.get_right(), s3.get_left(), buff=0.05, color=WHITE, stroke_width=2)

        b0 = Text("32 bit",   font_size=18, color=RED   ).next_to(s1, UP, buff=0.18)
        b1 = Text("~3.5 bit", font_size=18, color=YELLOW).next_to(s2, UP, buff=0.18)
        b2 = Text("~3 bit",   font_size=18, color=GREEN ).next_to(s3, UP, buff=0.18)

        inp     = Text("KV Cache\nvector",  font_size=17, color=GRAY_B ).next_to(s1, LEFT,  buff=0.55)
        out     = Text("3-bit\nvector",     font_size=17, color=GREEN_B).next_to(s3, RIGHT, buff=0.55)
        arr_in  = Arrow(inp.get_right(), s1.get_left(), buff=0.08, color=GRAY_B,  stroke_width=2)
        arr_out = Arrow(s3.get_right(), out.get_left(), buff=0.08, color=GREEN_B, stroke_width=2)

        self.play(FadeIn(inp), GrowArrow(arr_in))
        self.play(FadeIn(s1), FadeIn(b0))
        self.wait(0.25)
        self.play(GrowArrow(arr12))
        self.play(FadeIn(s2), FadeIn(b1))
        self.wait(0.25)
        self.play(GrowArrow(arr23))
        self.play(FadeIn(s3), FadeIn(b2))
        self.play(GrowArrow(arr_out), FadeIn(out))

        summary = Text(
            "10× compression  ·  zero accuracy loss  ·  no fine-tuning",
            font_size=19, color=YELLOW, weight=BOLD,
        ).to_edge(DOWN, buff=0.35)
        self.play(Write(summary))
        self.wait(2)
