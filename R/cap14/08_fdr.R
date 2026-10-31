# Cap. 14: Benjamini-Hochberg: FDR em simulação
source(file.path("R", "comum", "gerador.R"))
source(file.path("R", "comum", "multiplas.R"))
pr <- function(...) cat(sprintf(...), "\n", sep = "")
m <- 50; N <- 4000; al <- 0.10; delta <- 3
g <- gerador(2026)
sim <- function(m0, rho) {
  Z <- matriz_normal(g, N, m); u <- g$normal(N)
  W <- sqrt(rho) * u + sqrt(1 - rho) * Z
  if (m0 < m) W[, (m0 + 1):m] <- W[, (m0 + 1):m] + delta
  fdp <- fwb <- pdb <- pwb <- pwo <- numeric(N)
  for (i in seq_len(N)) {
    p <- 2 * pnorm(-abs(W[i, ]))
    R <- bh(p, al); lim <- al * R / m
    rej <- if (R > 0) p <= lim else rep(FALSE, m)
    V <- if (m0 > 0) sum(rej[seq_len(m0)]) else 0
    S <- R - V
    fdp[i] <- if (R > 0) V / R else 0
    bo <- p <= al / m
    fwb[i] <- if (m0 > 0) any(bo[seq_len(m0)]) else 0
    if (m0 < m) {
      pdb[i] <- S / (m - m0)
      pwb[i] <- sum(bo[-seq_len(m0)]) / (m - m0)
    }
  }
  c(fdr = mean(fdp), fwer_bonf = mean(fwb), poder_bh = mean(pdb),
    poder_bonf = mean(pwb))
}
pr("%4s %4s %7s %7s %7s %7s %7s", "m0", "rho", "FDR", "lim",
   "FWER_B", "pod_BH", "pod_B")
for (rho in c(0, 0.5)) for (m0 in c(50, 45, 25, 5)) {
  v <- sim(m0, rho)
  pr("%4d %4.1f %7.4f %7.4f %7.4f %7.4f %7.4f", m0, rho, v[1],
     al * m0 / m, v[2], v[3], v[4])
}
