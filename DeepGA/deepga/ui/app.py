# -*- coding: utf-8 -*-
"""
Interfaz Gráfica Web para DeepGA V10 (Streamlit).
Permite configurar hiperparámetros, cargar datasets personalizados (carpetas o zip),
lanzar la optimización evolutiva y visualizar la convergencia y métricas en tiempo real.

Por defecto utiliza la configuración óptima calibrada mediante irace (best_configuration_4.json).
La modificación manual de hiperparámetros está protegida bajo un panel de modo experto.
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
    Carga tu propio dataset y ejecuta la búsqueda con la **configuración óptima calibrada por irace**.
    """)

    # ----------------------------------------------------
    # BARRA LATERAL: Configuración de Parámetros V10
    # ----------------------------------------------------
    st.sidebar.header("⚙️ Configuración del Algoritmo")

    # Configuración óptima por defecto
    opt_cfg = DeepGAConfig.optimal()

    st.sidebar.success("✅ **Configuración Óptima Activa** (irace best_configuration_4)")
    st.sidebar.caption(
        f"• Población: {opt_cfg.pop_size}  \n"
        f"• Generaciones: {opt_cfg.generations}  \n"
        f"• Learning Rate: {opt_cfg.learning_rate}  \n"
        f"• Cruce: {opt_cfg.crossover_rate:.3f} | Mutación: {opt_cfg.mutation_rate:.3f}  \n"
        f"• Feromonas (Alpha: {opt_cfg.alpha:.3f}, Rho: {opt_cfg.rho:.3f})"
    )

    # Parámetros básicos de ejecución
    st.sidebar.subheader("Hardware y Dataset")
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=2)
    device = st.sidebar.selectbox("Dispositivo", ["auto", "cuda", "cpu"], index=0)
    seed = st.sidebar.number_input("Semilla Aleatoria", value=42)

    # PANEL PROTEGIDO / OCULTO: MODO EXPERTO
    with st.sidebar.expander("🔒 Ajustes Avanzados (Solo Expertos)", expanded=False):
        st.warning(
            "⚠️ **ADVERTENCIA:**  \n"
            "Los valores por defecto provienen de una calibración exhaustiva con **irace** "
            "(`best_configuration_4.json`). Modificar estos parámetros sin comprender su impacto "
            "en la convergencia estocástica puede degradar los resultados."
        )
        enable_expert = st.checkbox("Deseo modificar la configuración óptima manualmente", value=False)

        if enable_expert:
            st.subheader("Evolución y Población")
            pop_size = st.slider("Tamaño de Población (N)", min_value=4, max_value=50, value=opt_cfg.pop_size, step=1)
            generations = st.slider("Generaciones (T)", min_value=1, max_value=50, value=opt_cfg.generations, step=1)
            crossover_rate = st.slider("Probabilidad de Cruce (cr)", 0.1, 1.0, opt_cfg.crossover_rate, 0.01)
            mutation_rate = st.slider("Probabilidad de Mutación (mr)", 0.01, 0.5, opt_cfg.mutation_rate, 0.01)
            tournament_size = st.slider("Tamaño Torneo", 2, 10, opt_cfg.tournament_size, 1)

            st.subheader("Feromonas V10 (ACO)")
            alpha = st.slider("Sensibilidad Feromonas (alpha)", 0.1, 3.0, opt_cfg.alpha, 0.05)
            rho = st.slider("Evaporación Feromonas (rho)", 0.01, 0.5, opt_cfg.rho, 0.01)
            top_k_ratio = st.slider("Ratio Élite Depósito Feromona", 0.05, 0.8, opt_cfg.top_k_ratio, 0.01)

            st.subheader("Entrenamiento")
            lr = st.number_input("Learning Rate", value=opt_cfg.learning_rate, format="%.5f")
            train_epochs = st.number_input("Épocas de evaluación rápida", min_value=1, max_value=20, value=opt_cfg.train_epochs)
            final_epochs = st.number_input("Épocas de re-entrenamiento final", min_value=0, max_value=100, value=opt_cfg.final_epochs)
        else:
            pop_size = opt_cfg.pop_size
            generations = opt_cfg.generations
            crossover_rate = opt_cfg.crossover_rate
            mutation_rate = opt_cfg.mutation_rate
            tournament_size = opt_cfg.tournament_size
            alpha = opt_cfg.alpha
            rho = opt_cfg.rho
            top_k_ratio = opt_cfg.top_k_ratio
            lr = opt_cfg.learning_rate
            train_epochs = opt_cfg.train_epochs
            final_epochs = opt_cfg.final_epochs

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
            expert_mode=enable_expert,
            pop_size=pop_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            tournament_size=tournament_size,
            alpha=alpha,
            rho=rho,
            top_k_ratio=top_k_ratio,
            learning_rate=lr,
            train_epochs=train_epochs,
            final_epochs=final_epochs,
            batch_size=batch_size,
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
            with st.spinner("Evolucionando arquitecturas neuronales con V10..."):
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
