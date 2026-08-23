#!/usr/bin/env Rscript
# ==============================================================================
# Script de Análisis Post-Optimización para irace y DeepGA
# Carga 'irace.Rdata' e inspecciona las configuraciones ganadoras,
# el rendimiento sobre instancias de entrenamiento y test, y exporta parámetros.
# ==============================================================================

suppressPackageStartupMessages({
  if (!require("irace", quietly = TRUE)) {
    install.packages("irace", repos = "https://cloud.r-project.org")
    library("irace")
  }
})

args <- commandArgs(trailingOnly = TRUE)
rdata_file <- if (length(args) > 0) args[1] else "irace.Rdata"

if (!file.exists(rdata_file)) {
  stop(sprintf("Error: No se encontró el archivo de resultados '%s'. Asegúrese de haber ejecutado irace primero.", rdata_file))
}

cat("======================================================================\n")
cat("       ANÁLISIS DE RESULTADOS DE OPTIMIZACIÓN IRACE (DEEPGA)\n")
cat("======================================================================\n")
cat(sprintf("📂 Cargando archivo: %s\n\n", rdata_file))

# Cargar el archivo de resultados
load(rdata_file)

# Extraer el objeto de resultados
if (!exists("iraceResults")) {
  stop("El archivo no contiene el objeto 'iraceResults' esperado de irace.")
}

# 1. Obtener configuraciones de élite
elites <- getFinalElites(iraceResults)
cat(sprintf("🏆 Total de configuraciones élite finales: %d\n\n", nrow(elites)))

cat("--- TABLA RESUMEN DE CONFIGURACIONES ÉLITE ---\n")
print(elites)

# 2. Obtener la mejor configuración absoluta (Candidato #1)
best_id <- elites$.ID.[1]
best_config <- elites[1, , drop = FALSE]

cat("\n======================================================================\n")
cat(sprintf("🥇 CONFIGURACIÓN ÓPTIMA RECOMENDADA (ID: %s)\n", best_id))
cat("======================================================================\n")

param_names <- colnames(best_config)
param_names <- param_names[!param_names %in% c(".ID.", ".PARENT.", ".WEIGHT.")]

cli_args <- ""
json_entries <- c()

for (p in param_names) {
  val <- best_config[[p]]
  if (!is.na(val)) {
    cat(sprintf("  • %-26s : %s\n", p, as.character(val)))
    flag_name <- gsub("_", "-", p)
    cli_args <- paste0(cli_args, sprintf("--%s %s ", flag_name, as.character(val)))
    
    # Formatear para JSON
    if (is.numeric(val)) {
      json_entries <- c(json_entries, sprintf('  "%s": %s', p, as.character(val)))
    } else {
      json_entries <- c(json_entries, sprintf('  "%s": "%s"', p, as.character(val)))
    }
  }
}

# 3. Mostrar rendimiento en el conjunto de prueba (Test Instances)
if (!is.null(iraceResults$testing) && !is.null(iraceResults$testing$experiments)) {
  cat("\n======================================================================\n")
  cat("📊 EVALUACIÓN DE GENERALIZACIÓN EN TEST SET (DATASETS NO VISTOS):\n")
  cat("======================================================================\n")
  print(iraceResults$testing$experiments)
}

# 4. Exportar a JSON
json_content <- paste0("{\n", paste(json_entries, collapse = ",\n"), "\n}")
json_file <- "best_configuration.json"
writeLines(json_content, con = json_file)
cat(sprintf("\n💾 Parámetros óptimos exportados en formato JSON: %s\n", json_file))

# 5. Exportar comando para re-entrenar modelo final
cmd_file <- "best_configuration_cmd.txt"
writeLines(cli_args, con = cmd_file)
cat(sprintf("💾 Comando CLI exportado en: %s\n", cmd_file))

cat("\n----------------------------------------------------------------------\n")
cat("🚀 Para entrenar completamente la mejor red neuronal y obtener matrices de confusión:\n")
cat(sprintf("   python evaluate_best_config.py %s\n", cli_args))
cat("======================================================================\n")
