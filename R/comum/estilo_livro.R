# Estilo gráfico comum às figuras do livro em R (paleta e tema ggplot2)

raiz_codigo <- function() {
  # pasta codigo/, independentemente de onde o script é chamado
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(arg)) {
    return(normalizePath(file.path(dirname(sub("^--file=", "", arg)), "..", "..")))
  }
  normalizePath(".")
}

cores_livro <- c(primaria = "#1F4E79", secundaria = "#2A9D8F", destaque = "#E76F51",
                 ouro = "#E9A23B", violeta = "#6D597A", cinza = "#5C6770")

# paleta categórica (ordem fixa, validada para daltonismo)
categ_livro <- c("#3A6FB0", "#E06A4E", "#2A9D8F", "#C98A1A", "#9063B5")

tema_livro <- function(base_size = 9) {
  ggplot2::theme_minimal(base_size = base_size) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major = ggplot2::element_line(colour = "#DDE3EA", linewidth = 0.3),
      axis.line = ggplot2::element_line(colour = cores_livro[["cinza"]], linewidth = 0.3),
      plot.title = ggplot2::element_text(face = "bold", colour = cores_livro[["primaria"]])
    )
}

salvar_figura <- function(grafico, capitulo, nome, largura = 6, altura = 3.5) {
  destino <- file.path(raiz_codigo(), "figuras", capitulo)
  dir.create(destino, recursive = TRUE, showWarnings = FALSE)
  caminho <- file.path(destino, paste0(nome, ".pdf"))
  ggplot2::ggsave(caminho, grafico, width = largura, height = altura, device = grDevices::cairo_pdf)
  message("figura salva: ", caminho)
  invisible(caminho)
}
