"""Streamlit entry point for the Microstructure Interface Area Calculator.

The actual generation and analysis lives in the ``microstructure`` package;
this module only wires inputs from the UI to those functions and displays
the results.
"""

import csv
from datetime import datetime
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from microstructure import distributions as dist
from microstructure import generators as gen
from microstructure.anisotropy import (
    anisotropy_index,
    directional_intercept_density,
    elongation_direction_deg,
)
from microstructure.metrics import (
    interface_to_area_ratio_per_um,
    interfacial_length_um,
    mean_free_path_um,
    mean_intercept_length_um,
    measured_volume_fraction,
    specific_surface_area_per_um,
)
from microstructure.placement import place_particles
from microstructure.rendering import add_scale_bar
from microstructure.validation import validate_monodisperse_circles

st.set_page_config(page_title="Microstructure Generator", layout="centered")
st.title("Microstructure Interface Area Calculator")
st.caption(
    "Simulate 2D microstructures and measure the interfacial length per unit "
    "area — useful for relating microstructure to bulk properties (diffusion, "
    "strength, conductivity)."
)

tab_gen, tab_val = st.tabs(["Generate", "Validation"])

# ===========================================================================
# Generate tab — the original generator + analyzer flow.
# ===========================================================================
with tab_gen:
    col1, col2 = st.columns(2)
    with col1:
        image_width_um = st.number_input(
            "Image Width (µm)",
            50.0,
            1000.0,
            200.0,
            help="Physical width of the simulated region in microns.",
        )
        particle_diameter_um = st.number_input(
            "Avg. Particle Diameter (µm)",
            min_value=0.1,
            max_value=100.0,
            value=10.0,
            step=0.1,
            format="%.1f",
            help="Nominal particle diameter. Actual sizes vary by ± the size "
            "variation set below.",
        )
        pixel_per_um = st.slider(
            "Pixels per Micron",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            help="Spatial resolution. Higher values make perimeter "
            "measurements more accurate but the canvas grows quadratically — "
            "expect slow generation above ~20 px/µm for large images.",
        )
    with col2:
        image_height_um = st.number_input(
            "Image Height (µm)",
            50.0,
            1000.0,
            200.0,
            help="Physical height of the simulated region in microns.",
        )
        volume_fraction = st.number_input(
            "Volume Fraction (%)",
            min_value=0.1,
            max_value=99.9,
            value=20.0,
            step=0.1,
            format="%.1f",
            help="Target fraction of the image area occupied by particles. "
            "The *achieved* fraction is reported with results and may differ "
            "when overlap is disallowed at high target values.",
        )
        shape = st.selectbox(
            "Particle Shape",
            gen.SHAPES,
            help="Particle geometry. Non-circular shapes use different "
            "drawing logic and may produce different interfacial-length "
            "numbers for the same nominal diameter.",
        )

    with st.expander("Advanced settings"):
        size_distribution = st.selectbox(
            "Size distribution",
            dist.DISTRIBUTIONS,
            help="How particle diameters are drawn. The *Avg. Particle "
            "Diameter* above is the central tendency: arithmetic mean for "
            "Uniform/Normal, median for Log-normal, characteristic size "
            "d₆₃ for Rosin-Rammler.",
        )
        size_variation = 5.0  # Uniform default; overridden by widget below
        lognormal_sigma_g = 1.5
        normal_sigma_pct = 10.0
        rr_shape_n = 2.5
        if size_distribution == dist.UNIFORM:
            size_variation = st.number_input(
                "Size variation ± (%)",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=0.1,
                help="Half-width of the uniform distribution around the "
                "nominal diameter (e.g., 10% means radii in 0.9× to 1.1×).",
            )
        elif size_distribution == dist.LOG_NORMAL:
            lognormal_sigma_g = st.number_input(
                "Geometric std σ_g (dimensionless)",
                min_value=1.01,
                max_value=4.0,
                value=1.5,
                step=0.05,
                help="Multiplicative spread. σ_g = 1 is monodisperse; "
                "typical ball-milled powders sit at 1.3–2.0; very broad "
                "distributions reach 2.5+. Note σ_ln = ln(σ_g).",
            )
        elif size_distribution == dist.NORMAL:
            normal_sigma_pct = st.number_input(
                "Standard deviation σ (% of mean)",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=1.0,
                help="Arithmetic std as a percentage of the mean diameter. "
                "Negative samples are clipped to ≥ 1 pixel.",
            )
        elif size_distribution == dist.ROSIN_RAMMLER:
            rr_shape_n = st.number_input(
                "Shape parameter n",
                min_value=1.0,
                max_value=10.0,
                value=2.5,
                step=0.1,
                help="Weibull shape exponent. Smaller n → broader, more "
                "skewed distribution. Crushed minerals typically n ≈ 1.5–3; "
                "tighter classifiers reach n ≈ 5+.",
            )
        rng_seed = st.number_input(
            "Random Seed",
            min_value=-1,
            max_value=9999,
            value=-1,
            step=1,
            help="-1 means a different random arrangement each run. Set to "
            "any non-negative integer to make the result reproducible.",
        )
        allow_overlap = st.checkbox(
            "Allow particle overlap",
            value=False,
            help="When off, the generator rejects placements within 1.8× the "
            "average radius of an existing particle. Required for very high "
            "volume fractions (≳ 70%).",
        )
        periodic_boundaries = st.checkbox(
            "Periodic boundaries (toroidal)",
            value=False,
            help="When on, the canvas behaves like a torus: particles can "
            "sit anywhere (no edge margin), wrap across the image edges, "
            "and overlap is checked via the minimum image convention. "
            "Removes the depletion-near-edges bias that makes achieved "
            "volume fraction undershoot target at large particle sizes.",
        )
        if shape == gen.ROUGH_SPHERES:
            bumpiness_pct = st.slider(
                "Bumpiness ± (%)",
                min_value=0.0,
                max_value=30.0,
                value=10.0,
                step=1.0,
                help="Standard deviation of the radial perturbation, as a "
                "fraction of radius.",
            )
        else:
            bumpiness_pct = 10.0
        if shape == gen.CRACKED_FLAKES:
            jitter_pct = st.slider(
                "Edge jitter ± (%)",
                min_value=0.0,
                max_value=30.0,
                value=15.0,
                step=1.0,
                help="Per-vertex displacement, as a fraction of diameter. "
                "Higher values produce more angular, fractured-looking edges.",
            )
        else:
            jitter_pct = 15.0
        mix_ratio = (
            st.slider(
                "% Circular in Mix",
                0,
                100,
                33,
                help="In Mixed mode, the percentage of particles drawn as "
                "circles. The remainder is split evenly between elliptical "
                "and irregular blobs.",
            )
            if shape == gen.MIXED
            else None
        )

    calculate = st.button("Calculate", type="primary", key="calculate_gen")
    if not calculate:
        st.info(
            "Set your parameters above and click **Calculate** to generate a "
            "microstructure. Increase **Pixels per Micron** for sharper "
            "edges; increase the **Image** dimensions for better statistics."
        )
    else:
        if rng_seed >= 0:
            np.random.seed(int(rng_seed))

        width_px = int(image_width_um * pixel_per_um)
        height_px = int(image_height_um * pixel_per_um)
        canvas = np.zeros((height_px, width_px), dtype=np.uint8)

        rad_um = particle_diameter_um / 2
        area_factor = gen.expected_area_factor(shape, mix_ratio)
        dist_factor = dist.expected_r2_factor(
            size_distribution,
            uniform_pct=size_variation,
            lognormal_sigma_g=lognormal_sigma_g,
            normal_sigma_pct=normal_sigma_pct,
            rr_shape_n=rr_shape_n,
        )
        shape_area_um2 = np.pi * rad_um**2 * area_factor * dist_factor
        area_target_um2 = image_width_um * image_height_um * volume_fraction / 100
        num_particles = max(1, int(area_target_um2 / shape_area_um2))

        avg_rad_px = max(1, int(rad_um * pixel_per_um))
        pil_img = Image.fromarray(canvas)

        size_sampler = dist.make_size_sampler(
            size_distribution,
            float(avg_rad_px),
            uniform_pct=size_variation,
            lognormal_sigma_g=lognormal_sigma_g,
            normal_sigma_pct=normal_sigma_pct,
            rr_shape_n=rr_shape_n,
        )

        progress = st.progress(0.0)
        particles = place_particles(
            pil_img=pil_img,
            shape=shape,
            num_particles=num_particles,
            avg_rad_px=avg_rad_px,
            size_variation=size_variation,
            allow_overlap=allow_overlap,
            bumpiness_pct=bumpiness_pct,
            jitter_pct=jitter_pct,
            mix_ratio=mix_ratio,
            volume_fraction=volume_fraction,
            progress_callback=progress.progress,
            size_sampler=size_sampler,
            periodic_boundaries=periodic_boundaries,
        )

        canvas = np.array(pil_img)
        final = add_scale_bar(
            pil_img, image_width_um, image_height_um, pixel_per_um
        )

        binary = canvas > 0
        interface_um = interfacial_length_um(binary, pixel_per_um)
        ratio = interface_to_area_ratio_per_um(
            interface_um, image_width_um, image_height_um
        )
        achieved_vf = measured_volume_fraction(binary)
        achieved_vf_pct = achieved_vf * 100.0

        s_v_per_um = specific_surface_area_per_um(ratio)
        mean_intercept_particles_um = mean_intercept_length_um(
            achieved_vf, s_v_per_um
        )
        mean_free_path_matrix_um = mean_free_path_um(achieved_vf, s_v_per_um)

        st.markdown("---")
        st.subheader("Results")

        if len(particles) < num_particles:
            st.warning(
                f"Placed only {len(particles)} of {num_particles} requested "
                "particles — the non-overlap budget was exhausted. Try "
                "enabling **Allow particle overlap**, lowering **Volume "
                "Fraction**, or increasing **Image** size."
            )

        col_a, col_b = st.columns(2)
        col_a.metric("Particles Placed", f"{len(particles)}")
        col_a.metric(
            "Interfacial Length",
            f"{interface_um:.2f} µm",
            help="Total perimeter of all connected components, in microns, "
            "measured by the Crofton formula on the rasterized image.",
        )
        col_b.metric(
            "Volume Fraction (achieved)",
            f"{achieved_vf_pct:.2f}%",
            delta=f"{achieved_vf_pct - volume_fraction:+.2f}% vs target",
            delta_color="off",
            help="Fraction of image pixels occupied by particles. Under "
            "standard stereological assumptions this equals the 3D volume "
            "fraction.",
        )
        col_b.metric(
            "Interface / Area (L_A)",
            f"{ratio:.5f} µm⁻¹",
            help="Interfacial length divided by image area. A direct, "
            "resolution-independent measure of how finely the microstructure "
            "is subdivided. Standard symbol in stereology: L_A.",
        )

        st.markdown("##### 3D estimates from the 2D section")
        st.caption(
            "Assumes the image is an isotropic uniform random (IUR) section "
            "through a 3D structure (Underwood, 1970). If the image *is* the "
            "structure, treat these as 2D analogues only."
        )
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric(
            "Specific Surface Area (S_V)",
            f"{s_v_per_um:.5f} µm⁻¹",
            help="S_V = (4/π) · L_A. Surface area per unit volume in 3D. "
            "Drives sintering kinetics, gas-solid reaction rates, and "
            "catalyst activity.",
        )
        col_s2.metric(
            "Mean Intercept ⟨L_α⟩",
            f"{mean_intercept_particles_um:.3f} µm",
            help="⟨L_α⟩ = 4·V_V / S_V. Mean chord length through a particle "
            "along a random ray — a characteristic particle size measure "
            "that's robust to shape.",
        )
        col_s3.metric(
            "Mean Free Path (λ)",
            f"{mean_free_path_matrix_um:.3f} µm",
            help="λ = 4·(1 - V_V) / S_V. Mean chord length through the "
            "matrix between particle encounters. Classical input to "
            "diffusion path estimates and dispersion-strengthening models.",
        )

        st.image(
            np.array(final),
            caption="Simulated Microstructure",
            channels="GRAY",
            width="stretch",
        )

        if particles:
            sizes_um = np.array([2 * p.r_px / pixel_per_um for p in particles])
            with st.expander("Particle size distribution"):
                bins = max(5, min(40, int(np.sqrt(len(sizes_um)))))
                counts, edges = np.histogram(sizes_um, bins=bins)
                centers_um = 0.5 * (edges[:-1] + edges[1:])
                hist_df = pd.DataFrame(
                    {"count": counts}, index=np.round(centers_um, 2)
                )
                hist_df.index.name = "diameter (µm)"
                st.bar_chart(hist_df, height=240)
                st.caption(
                    f"n = {len(sizes_um)} particles | "
                    f"mean = {sizes_um.mean():.2f} µm | "
                    f"std = {sizes_um.std():.2f} µm | "
                    f"nominal = {particle_diameter_um:.2f} µm"
                )

        # Anisotropy (directional intercept density). Computed lazily inside
        # the expander so a closed expander costs nothing.
        anisotropy_angles_deg = np.arange(0.0, 180.0, 10.0)
        with st.expander("Anisotropy (directional analysis)"):
            p_l = directional_intercept_density(
                binary, anisotropy_angles_deg, pixel_per_um
            )
            da = anisotropy_index(p_l)
            elongation_deg = elongation_direction_deg(
                anisotropy_angles_deg, p_l
            )

            rose_df = pd.DataFrame(
                {"P_L (µm⁻¹)": p_l}, index=anisotropy_angles_deg
            )
            rose_df.index.name = "angle θ (°)"
            st.line_chart(rose_df, height=260)
            st.caption(
                "Boundary intersections per unit length along scan lines at "
                "angle θ. Flat curve → isotropic; pronounced minimum at "
                "angle θ₀ → features elongated along θ₀. Computed "
                "inside a √2-inscribed crop to avoid rotation-padding bias."
            )

            ac1, ac2 = st.columns(2)
            ac1.metric(
                "Anisotropy index DA",
                f"{da:.3f}",
                help="(max − min) / (max + min) of P_L(θ). 0 = perfectly "
                "isotropic; values around 0.05–0.10 are noise floor for "
                "synthetic isotropic structures of this size. >0.2 starts "
                "to suggest real texture.",
            )
            ac2.metric(
                "Elongation direction",
                f"{elongation_deg:.0f}°",
                help="Angle (0°–180°) of the P_L minimum, i.e. the long "
                "axis of foreground features. P_L is *minimum* parallel to "
                "elongation (fewer intersections) and maximum perpendicular.",
            )

        png_buf = BytesIO()
        final.save(png_buf, format="PNG")

        csv_buf = StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["field", "value"])
        writer.writerow(
            ["timestamp", datetime.utcnow().isoformat(timespec="seconds") + "Z"]
        )
        writer.writerow(["image_width_um", image_width_um])
        writer.writerow(["image_height_um", image_height_um])
        writer.writerow(["pixel_per_um", pixel_per_um])
        writer.writerow(["particle_diameter_um", particle_diameter_um])
        writer.writerow(["shape", shape])
        writer.writerow(["volume_fraction_target_pct", volume_fraction])
        writer.writerow(["size_distribution", size_distribution])
        writer.writerow(["size_variation_pct", size_variation])
        writer.writerow(["lognormal_sigma_g", lognormal_sigma_g])
        writer.writerow(["normal_sigma_pct", normal_sigma_pct])
        writer.writerow(["rosin_rammler_shape_n", rr_shape_n])
        writer.writerow(["allow_overlap", allow_overlap])
        writer.writerow(["periodic_boundaries", periodic_boundaries])
        writer.writerow(["rng_seed", rng_seed])
        writer.writerow(
            ["mix_ratio_pct", mix_ratio if mix_ratio is not None else ""]
        )
        writer.writerow(["bumpiness_pct", bumpiness_pct])
        writer.writerow(["jitter_pct", jitter_pct])
        writer.writerow(["particles_requested", num_particles])
        writer.writerow(["particles_placed", len(particles)])
        writer.writerow(["volume_fraction_achieved_pct", round(achieved_vf_pct, 4)])
        writer.writerow(["interfacial_length_um", round(interface_um, 4)])
        writer.writerow(["interface_area_ratio_per_um_L_A", round(ratio, 6)])
        writer.writerow(["specific_surface_area_per_um_S_V", round(s_v_per_um, 6)])
        writer.writerow(
            [
                "mean_intercept_length_particles_um",
                round(mean_intercept_particles_um, 4),
            ]
        )
        writer.writerow(
            ["mean_free_path_matrix_um", round(mean_free_path_matrix_um, 4)]
        )
        writer.writerow(["anisotropy_index", round(da, 4)])
        writer.writerow(["elongation_direction_deg", round(elongation_deg, 1)])

        dl_col1, dl_col2 = st.columns(2)
        dl_col1.download_button(
            "Download PNG",
            data=png_buf.getvalue(),
            file_name="microstructure.png",
            mime="image/png",
        )
        dl_col2.download_button(
            "Download metrics (CSV)",
            data=csv_buf.getvalue(),
            file_name="microstructure_metrics.csv",
            mime="text/csv",
        )

# ===========================================================================
# Validation tab — sweep φ and compare measured L_A against 4·φ/d.
# ===========================================================================
with tab_val:
    st.markdown(
        "Compare the measured interfacial length per area against the "
        "closed-form result for **monodisperse, non-overlapping circles**: "
        "`L_A = 4·φ / d`. A close match across φ confirms the rasterization "
        "+ perimeter pipeline is unbiased; the discrepancy at high φ tells "
        "you the placement budget has been exhausted."
    )

    vcol1, vcol2 = st.columns(2)
    with vcol1:
        val_canvas_um = st.number_input(
            "Validation canvas (µm × µm)",
            min_value=100.0,
            max_value=1000.0,
            value=400.0,
            step=50.0,
            key="val_canvas_um",
            help="Square canvas side length in microns. Larger canvases give "
            "better statistics but slower runs.",
        )
        val_diameter_um = st.number_input(
            "Particle diameter (µm)",
            min_value=0.5,
            max_value=50.0,
            value=10.0,
            step=0.5,
            key="val_diameter_um",
            help="Monodisperse circle diameter used for every sweep point.",
        )
        val_pixel_per_um = st.slider(
            "Pixels per Micron",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            key="val_pixel_per_um",
            help="Higher resolution shrinks rasterization bias but slows the "
            "sweep.",
        )
    with vcol2:
        val_max_phi_pct = st.slider(
            "Max φ to sweep (%)",
            min_value=5,
            max_value=70,
            value=40,
            step=5,
            key="val_max_phi_pct",
            help="Cap on target volume fraction. Above ~50% the non-overlap "
            "budget gets exhausted and theory-vs-measurement disagreement "
            "starts dominating.",
        )
        val_n_points = st.slider(
            "Sweep points",
            min_value=4,
            max_value=20,
            value=10,
            step=1,
            key="val_n_points",
            help="Number of φ values sampled uniformly between 1% and the "
            "max.",
        )
        val_seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=9999,
            value=0,
            step=1,
            key="val_seed",
            help="Sweep is fully deterministic for reproducibility.",
        )
        val_pbc = st.checkbox(
            "Periodic boundaries",
            value=False,
            key="val_pbc",
            help="When on, runs the sweep with toroidal placement. Achieved "
            "φ tracks target closely and the residual error reflects only "
            "the perimeter estimator's bias — the sharpest test of the "
            "rasterization pipeline.",
        )

    if st.button("Run validation sweep", type="primary", key="val_run"):
        progress = st.progress(0.0)
        phi_targets = np.linspace(0.01, val_max_phi_pct / 100.0, val_n_points)
        points = validate_monodisperse_circles(
            canvas_um=val_canvas_um,
            diameter_um=val_diameter_um,
            pixel_per_um=val_pixel_per_um,
            phi_targets=phi_targets.tolist(),
            seed=int(val_seed),
            progress_callback=progress.progress,
            periodic_boundaries=val_pbc,
        )

        df = pd.DataFrame(
            {
                "phi_target": [p.phi_target for p in points],
                "phi_achieved": [p.phi_achieved for p in points],
                "L_A_measured (µm⁻¹)": [p.L_A_measured for p in points],
                "L_A_theory (µm⁻¹)": [p.L_A_theory for p in points],
                "rel_error (%)": [p.relative_error_pct for p in points],
                "particles_requested": [p.particles_requested for p in points],
                "particles_placed": [p.particles_placed for p in points],
            }
        )

        st.markdown("---")
        st.subheader("Measured vs theory")
        chart_df = df.set_index("phi_achieved")[
            ["L_A_measured (µm⁻¹)", "L_A_theory (µm⁻¹)"]
        ]
        st.line_chart(chart_df, x_label="φ (achieved)", y_label="L_A (µm⁻¹)")

        st.subheader("Relative error")
        err_df = df.set_index("phi_achieved")[["rel_error (%)"]]
        st.line_chart(err_df, x_label="φ (achieved)", y_label="rel. error (%)")

        st.subheader("Sweep data")
        st.dataframe(
            df.style.format(
                {
                    "phi_target": "{:.3f}",
                    "phi_achieved": "{:.4f}",
                    "L_A_measured (µm⁻¹)": "{:.5f}",
                    "L_A_theory (µm⁻¹)": "{:.5f}",
                    "rel_error (%)": "{:+.2f}",
                }
            ),
            width="content",
        )

        max_abs_err = float(np.max(np.abs(df["rel_error (%)"])))
        if max_abs_err < 10.0:
            st.success(
                f"Worst-case |error| over the sweep: {max_abs_err:.2f}% — "
                "well within typical rasterization tolerance."
            )
        else:
            st.warning(
                f"Worst-case |error| over the sweep: {max_abs_err:.2f}%. "
                "Bump up Pixels per Micron, lower the max φ, or enlarge the "
                "validation canvas to bring this down."
            )

        val_csv = StringIO()
        df.to_csv(val_csv, index=False)
        st.download_button(
            "Download sweep data (CSV)",
            data=val_csv.getvalue(),
            file_name="validation_sweep.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "Pick the sweep parameters above and click **Run validation "
            "sweep**. Each point generates a fresh microstructure and takes "
            "a few seconds at default settings."
        )
