<h1 align="center" style="font-weight: bold;">Fusão de Representações & Adaptação de Domínio 💻</h1>

<p align="center">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python"/>
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch"/>
    <img src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white" alt="Keras"/>
    <img src="https://img.shields.io/badge/Albumentations-000000?style=for-the-badge" alt="Albumentations"/>
    <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
</p>

<p align="center">
  <a href="#projeto">Projeto</a> •
  <a href="#metodologia">Metodologia</a> • 
  <a href="#datasets">Bases de Dados</a> •
  <a href="#resultados">Protocolo Experimental</a> •
  <a href="#rodar">Como rodar</a> •
  <a href="#autor">Autor</a> 
</p>

<p align="center">
  <i>Projeto de Iniciação Científica focado em fusão de representações latentes para adaptação de domínio em modelos profundos aplicados ao monitoramento inteligente de vagas de estacionamento.</i>
</p>


<h2 id="projeto">📫 Projeto</h2>

Este repositório documenta o projeto PIBIC intitulado:

<b>"Fusão de Representações para Adaptação de Domínio de Modelos Profundos no Monitoramento Inteligente de Vagas de Estacionamento"</b>.

A pesquisa investiga como reduzir a dependência de dados rotulados e aumentar a capacidade de generalização de modelos de Deep Learning em cenários com mudança de domínio (domain shift).

<p align="center">
  <img src="https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow?style=for-the-badge"/>
</p>


<h2 id="metodologia">🔬 Metodologia</h2>

A abordagem segue o paradigma de <b>Self-Taught Learning (STL)</b>, utilizando aprendizado não supervisionado para geração de representações latentes robustas.

<h3>1️⃣ Extração de Representações</h3>

São avaliadas diferentes arquiteturas de autoencoders:

<ul>
  <li>Convolutional Autoencoder (CAE)</li>
  <li>Variational Autoencoder (VAE)</li>
  <li>Autoencoder com Skip Connections (AESc)</li>
</ul>

Os encoders aprendem representações compactas que são reutilizadas como extratores de características para classificação.

<h3>2️⃣ Classificação</h3>

Os encoders pré-treinados são conectados a uma CNN composta por:

<ul>
  <li>Encoder congelado</li>
  <li>Dropout (20%)</li>
  <li>Camadas densas (256 → 128)</li>
  <li>Saída Softmax (ocupada / livre)</li>
</ul>

<h3>3️⃣ Fusão de Modelos</h3>

São exploradas três estratégias de combinação de previsões:

<ul>
  <li><b>Soma</b> de probabilidades</li>
  <li><b>Produto</b> das probabilidades</li>
  <li><b>Votação majoritária</b></li>
  <li><b>Máximo</b> das probabilidades</li>
  
</ul>

O objetivo é combinar diferentes representações para aumentar robustez e generalização cross-domain.

---

<h2 id="datasets">📍 Bases de Dados</h2>

<h3>PKLot</h3>
<p>Dataset com imagens capturadas em três estacionamentos (PUCPR, UFPR04, UFPR05), sob diferentes condições climáticas e perspectivas de câmera.</p>

<h3>CNR-EXT</h3>
<p>Base composta por imagens capturadas por 9 câmeras distintas, com variações de iluminação, oclusão parcial e mudanças climáticas.</p>

<h3>Kyoto</h3>
<p>Base auxiliar utilizada para aprendizado não supervisionado e geração de representações genéricas.</p>

---

<h2 id="resultados">📊 Protocolo Experimental</h2>

<ul>
  <li>Treinamento progressivo (64 → 1024 amostras)</li>
  <li>Avaliação cross-domain (treino em um dataset, teste em outro)</li>
  <li>Análise de generalização</li>
  <li>Avaliação de reconstrução (SSIM, PSNR, MSE, RMSE, NCC, VIF, SCC)</li>
</ul>

O foco principal é medir robustez sob mudança de domínio e redução de dados rotulados.

---

<h2 id="rodar">🚀 Como Rodar o Projeto</h2>

Este projeto utiliza **uv** para gerenciamento de dependências e ambiente virtual.  
Certifique-se de ter o uv instalado: https://docs.astral.sh/uv/

---

### 1️⃣ Clone o repositório


Este projeto utiliza **uv** para gerenciamento de dependências e ambiente virtual.

### 🔧 Pré-requisito

Instale o uv:
https://docs.astral.sh/uv/

---

### 1️⃣ Clone o repositório

```bash
git clone https://github.com/Lucas-doc26/ML.git
cd ML
```

### 2️⃣ Sincronize o ambiente com o lockfile
```bash
uv sync
```

### 3️⃣ Ative o ambiente virtual
#### Linux
```bash
source .venv/bin/activate
```
#### Windows 
```bash
.venv\Scripts\activate
```

### 4️⃣ Execute o projeto

```bash
uv run python main.py
```