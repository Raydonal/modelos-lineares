"""Exporta para dados/ conjuntos de dados disponíveis apenas em Python."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comum.estilo_livro import DADOS  # noqa: E402

DADOS.mkdir(exist_ok=True)

# Cap. 1 -- dígitos manuscritos (8x8 pixels, 1797 imagens)
dig = load_digits()
df = pd.DataFrame(dig.data.astype(np.int64), columns=[f"px{i:02d}" for i in range(64)])
df.insert(0, "digito", dig.target)
df.to_csv(DADOS / "digitos.csv", index=False)
print(f"dados/digitos.csv  ({len(df)} linhas)")

# Cap. 3 -- diabetes (Efron et al., 2004): 442 pacientes, 10 covariáveis
from sklearn.datasets import load_diabetes  # noqa: E402

db = load_diabetes(scaled=False)
df = pd.DataFrame(db.data, columns=db.feature_names)
df["y"] = db.target
df.to_csv(DADOS / "diabetes.csv", index=False)
print(f"dados/diabetes.csv  ({len(df)} linhas)")
