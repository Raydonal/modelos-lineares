# Ambiente reprodutível completo (R + Python) para o código do livro.
#   docker build -t modelos-lineares .
#   docker run --rm -v "$PWD":/livro modelos-lineares make tudo
FROM rocker/r-ver:4.6.1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv make libcurl4-openssl-dev libssl-dev \
      libxml2-dev libfontconfig1-dev libfreetype6-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /livro
COPY requirements.txt renv.lock DESCRIPTION ./
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt
ENV PATH="/opt/venv/bin:${PATH}"
RUN R -q -e 'install.packages("renv", repos = "https://cloud.r-project.org"); renv::restore(prompt = FALSE, library = .libPaths()[1])'

COPY . .
CMD ["make", "tudo"]
