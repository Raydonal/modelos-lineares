"""Gerador pseudoaleatório portátil (Park-Miller, a = 48271, m = 2^31 - 1) com Box-Muller.

Produz exatamente os mesmos números que R/comum/gerador.R, para exemplos sintéticos cujos
números são citados no texto.
"""
import numpy as np


class Gerador:
    def __init__(self, semente):
        self.estado = semente

    def unif(self, k):
        u = np.empty(k)
        for i in range(k):
            self.estado = (48271 * self.estado) % 2147483647
            u[i] = self.estado / 2147483647
        return u

    def normal(self, k):
        u = self.unif(2 * k).reshape(k, 2)
        return np.sqrt(-2 * np.log(u[:, 0])) * np.cos(2 * np.pi * u[:, 1])

    def matriz_normal(self, k1, k2, sd=1.0):
        """Matriz k1 x k2 preenchida por linhas (igual a matriz_normal do R)."""
        return sd * self.normal(k1 * k2).reshape(k1, k2)
