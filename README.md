# microstructure-app
An interactive web app to simulate and analyze synthetic microstructures with customizable particle shapes, sizes, and volume fractions. Calculates the interfacial length-to-area ratio using real-world units (µm). Built with Python and Streamlit.
# Microstructure Interface Analyzer

This Streamlit app simulates 2D microstructures and calculates the interfacial length-to-area ratio based on:

- Real-world dimensions in microns
- User-selected particle size, shape, and volume fraction
- Instant image generation and quantitative analysis

## Features
- Circular, elliptical, or irregular (blob) particle shapes
- Particle generation based on volume fraction
- Live image preview and results
- Downloadable output image

## How to Run Locally
```bash
pip install streamlit numpy opencv-python matplotlib scikit-image Pillow
streamlit run micro_app.py
