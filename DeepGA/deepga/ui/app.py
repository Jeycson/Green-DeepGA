# -*- coding: utf-8 -*-
"""
Interfaz Gráfica Web para DeepGA V10 (Streamlit).
Permite configurar hiperparámetros, cargar datasets personalizados (carpetas o zip),
lanzar la optimización evolutiva y visualizar la convergencia y métricas en tiempo real.
"""

import os
import sys
import tempfile
import zipfile
from pathlib import Path

try:
    import streamlit as st
except ImportError:
    print("❌ Streamlit no está instalado. Instálalo con: pip install streamlit")
    sys.exit(1)

import pandas as pd
import numpy as np

# Asegurar que deepga esté en el PYTHONPATH
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deepga import DeepGAConfig, DeepGASearch


def main():
    st.set_page_config(
        page_title="DeepGA V10 Dashboard",
        page_icon="🧬",
        layout="wide"
    )

    st.title("🧬 DeepGA V10: Explorador Evolutivo de Arquitecturas")
    st.markdown("""
    Optimización de arquitecturas CNN mediante algoritmos genéticos mejorados con feromonas (**ACO-Enhanced DeepGA V10**).
    Carga tu propio dataset, ajusta los hiperparámetros e inicia la búsqueda de la red óptima.
    """)

    # ----------------------------------------------------
    # BARRA LATERAL: Configuración de Parámetros V10
    # ----------------------------------------------------
    st.sidebar.header("⚙️ Configuración del Algoritmo")

    st.sidebar.subheader("Población y Evolución")
    pop_size = st.sidebar.slider("Tamaño de Población (N)", min_value=4, max_value=50, value=10, step=2)
    generations = st.sidebar.slider("Generaciones (T)", min_value=1, max_value=30, value=5, step=1)
    crossover_rate = st.sidebar.slider("Probabilidad de Cruce (cr)", 0.1, 1.0, 0.8, 0.05)
    mutation_rate = st.sidebar.slider("Probabilidad de Mutación (mr)", 0.01, 0.5, 0.15, 0.01)
    weight_params = st.sidebar.slider("Penalización por Complejidad (w)", 0.0, 0.5, 0.10, 0.01)

    st.sidebar.subheader("Feromonas V10 (ACO)")
    alpha = st.sidebar.slider("Sensibilidad Feromonas (alpha)", 0.5, 3.0, 1.2, 0.1)
    rho = st.sidebar.slider("Evaporación Feromonas (rho)", 0.01, 0.5, 0.10, 0.01)
    top_k_ratio = st.sidebar.slider("Ratio Élite Depósito Feromona", 0.1, 0.8, 0.35, 0.05)

    st.sidebar.subheader("Entrenamiento y Hardware")
    train_epochs = st.sidebar.number_input("Épocas de búsqueda rápida", min_value=1, max_value=20, value=2)
    final_epochs = st.sidebar.number_input("Épocas de re-entrenamiento final", min_value=0, max_value=100, value=5)
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=2)
    lr = st.sidebar.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
    device = st.sidebar.selectbox("Dispositivo", ["auto", "cuda", "cpu"], index=0)
    seed = st.sidebar.number_input("Semilla Aleatoria", value=42)

    # ----------------------------------------------------
    # PANEL PRINCIPAL: Selección y Carga de Datos
    # ----------------------------------------------------
    st.subheader("📂 Fuente de Datos (Dataset)")
    data_mode = st.radio(
        "Selecciona cómo ingresar los datos:",
        ["CIFAR-10 (Predefinido)", "Ruta local a carpeta (ImageFolder)", "Subir archivo comprimido (.zip)"],
        horizontal=True
    )

    dataset_target = None

    if data_mode == "CIFAR-10 (Predefinido)":
        dataset_target = "cifar10"
        st.info("ℹ️ Se utilizará CIFAR-10 (10 clases, imágenes 32x32). Si no está descargado, se descargará automáticamente.")

    elif data_mode == "Ruta local a carpeta (ImageFolder)":
        folder_path = st.text_input("Ingresa la ruta absoluta o relativa a la carpeta del dataset:", value="./data/mi_dataset")
        if folder_path and os.path.exists(folder_path):
            st.success(f"Carpeta localizada: `{folder_path}`")
            dataset_target = folder_path
        else:
            st.warning("Especifica una ruta válida con subcarpetas para cada clase (e.g. `dataset/clase_a/`, `dataset/clase_b/`).")

    elif data_mode == "Subir archivo comprimido (.zip)":
        uploaded_zip = st.file_uploader("Sube un archivo .zip que contenga las subcarpetas de clases:", type=["zip"])
        if uploaded_zip is not None:
            temp_extract_dir = os.path.join(tempfile.gettempdir(), "deepga_uploaded_dataset")
            os.makedirs(temp_extract_dir, exist_ok=True)
            with zipfile.ZipFile(uploaded_zip, "r") as z:
                z.extractall(temp_extract_dir)
            st.success(f"Dataset descomprimido exitosamente en `{temp_extract_dir}`.")
            dataset_target = temp_extract_dir

    st.markdown("---")

    # ----------------------------------------------------
    # EJECUCIÓN Y RESULTADOS
    # ----------------------------------------------------
    col_btn, _ = st.columns([2, 8])
    with col_btn:
        start_btn = st.button("🚀 Iniciar Búsqueda Evolutiva V10", type="primary", use_container_width=True)

    if start_btn:
        if dataset_target is None:
            st.error("❌ Por favor selecciona o sube un dataset válido antes de iniciar.")
            return

        cfg = DeepGAConfig(
            pop_size=pop_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            weight_params=weight_params,
            alpha=alpha,
            rho=rho,
            top_k_ratio=top_k_ratio,
            train_epochs=train_epochs,
            final_epochs=final_epochs,
            batch_size=batch_size,
            learning_rate=lr,
            device=device,
            seed=seed,
            output_dir="./results_v10_ui"
        )

        st.info("Búsqueda en curso. Revisa la terminal para ver los logs detallados de cada generación...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Inicializando motor evolutivo DeepGA V10...")
        searcher = DeepGASearch(config=cfg)

        try:
            with st.spinner("Evolucionando arquitecturas neuronales..."):
                searcher.fit(dataset=dataset_target)
                progress_bar.progress(100)
                status_text.text("¡Búsqueda y re-entrenamiento completados!")

            st.success("🎉 ¡Optimización finalizada con éxito!")

            # Métricas del mejor modelo
            best_ind = searcher.best_individual
            if best_ind is not None:
                st.subheader("🏆 Mejor Arquitectura Encontrada")
                m1, m2, m3 = st.columns(3)
                m1.metric("Fitness Final", f"{best_ind[1]:.4f}")
                m2.metric("Accuracy Validación", f"{best_ind[2]:.2f}%")
                m3.metric("Parámetros Estimados", f"{int(best_ind[3]):,}")

            # Gráfica de evolución
            hist_df = searcher.summary()
            if not hist_df.empty:
                st.subheader("📈 Progreso de la Evolución")
                st.line_chart(hist_df.set_index("generation")[["best_fitness", "avg_fitness"]])

            # Guardar y permitir descarga
            save_path = os.path.join(cfg.output_dir, "best_model_v10.pt")
            searcher.save(save_path)
            st.success(f"Modelo guardado en `{save_path}`.")

        except Exception as e:
            st.error(f"Ocurrió un error durante la ejecución: {e}")
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
