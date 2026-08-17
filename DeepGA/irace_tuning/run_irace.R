#!/usr/bin/env Rscript
# ==============================================================================
# Script de Ejecución Programática de irace para DeepGA
# Ejecuta la optimización, guarda checkpoints y exporta las mejores configuraciones
# ==============================================================================

suppressPackageStartupMessages({
  if (!require("irace", quietly = TRUE)) {
    install.packages("irace", repos = "https://cloud.r-project.org")
    library("irace")
  }
})

cat("======================================================================\n")
cat("    IRACE OPTIMIZATION PIPELINE FOR DEEPGA (V10, V11, V12)\n")
cat("    Datasets: Tumour, Tumour_3\n")
cat("======================================================================\n\n")

# 1. Leer escenario
scenario_file <- "scenario.txt"
if (!file.exists(scenario_file)) {
  stop("Error: No se encontró el archivo 'scenario.txt' en el directorio actual.")
}

scenario <- readScenario(filename = scenario_file)

# Verificar parámetros de entrada
checkIraceScenario(scenario = scenario)

cat(sprintf("📌 Presupuesto de experimentos (maxExperiments): %d\n", scenario$maxExperiments))
cat(sprintf("📌 Paralelismo activo: %d\n", scenario$parallel))
cat(sprintf("📌 Archivo de instancias: %s\n", scenario$trainInstancesFile))
cat(sprintf("📌 Target runner: %s\n\n", scenario$targetRunner))

# 2. Ejecutar la búsqueda de irace
start_time <- Sys.time()
cat("🚀 Iniciando proceso de búsqueda iterativa con irace...\n\n")

irace_results <- irace(scenario = scenario)

end_time <- Sys.time()
elapsed_mins <- as.numeric(difftime(end_time, start_time, units = "mins"))

cat("\n======================================================================\n")
cat(sprintf("✨ Optimización finalizada en %.2f minutos.\n", elapsed_mins))
cat("======================================================================\n\n")

# 3. Extraer y mostrar las mejores configuraciones encontradas
cat("🏆 CONFIGURACIONES ÉLITE ENCONTRADAS POR IRACE:\n")
cat("----------------------------------------------------------------------\n")

# Obtener configuraciones de élite finales
elite_configs <- get_elite_configurations(irace_results)
print(elite_configs)

# Extraer la mejor configuración individual (#1)
best_config <- elite_configs[1, , drop = FALSE]
best_id <- best_config$.ID.

cat("\n----------------------------------------------------------------------\n")
cat(sprintf("🥇 MEJOR CONFIGURACIÓN GANADORA (ID: %s):\n", best_id))
cat("----------------------------------------------------------------------\n")

# Mostrar cada hiperparámetro de forma legible
param_names <- colnames(best_config)
param_names <- param_names[!param_names %in% c(".ID.", ".PARENT.", ".WEIGHT.")]

for (p in param_names) {
  val <- best_config[[p]]
  if (!is.na(val)) {
    cat(sprintf("   %-25s = %s\n", p, as.character(val)))
  }
}

# 4. Guardar mejor configuración en formato de texto plano y comando CLI
cli_flags <- ""
for (p in param_names) {
  val <- best_config[[p]]
  if (!is.na(val)) {
    # Convertir nombre a flag de python (--param-name)
    flag_name <- gsub("_", "-", p)
    cli_flags <- paste0(cli_flags, sprintf("--%s %s ", flag_name, as.character(val)))
  }
}

output_cli_file <- "best_configuration_cmd.txt"
writeLines(cli_flags, con = output_cli_file)
cat(sprintf("\n📄 Comando CLI para re-entrenar guardado en: %s\n", output_cli_file))
cat(sprintf("   python evaluate_best_config.py %s\n", cli_flags))

cat("\n======================================================================\n")
cat("✅ Proceso completado exitosamente. Archivo Rdata: irace.Rdata\n")
