"""
CRA Satellite Analyzer — Dashboard Web
Análise satelital para Cotas de Reserva Ambiental

Deploy: streamlit run app.py
"""

import streamlit as st
import ee
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import zipfile, glob, os, shutil, tempfile, json

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

st.set_page_config(
    page_title="CRA Analyzer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONFIG = {
    "reserva_legal": {
        "mata_atlantica": 0.20,
        "cerrado_fora_amazonia": 0.20,
        "cerrado_amazonia": 0.35,
        "amazonia": 0.80,
    },
    "preco_cra_ha_ano": 500,
    "bioma_labels": {
        "mata_atlantica": "Mata Atlântica",
        "cerrado_fora_amazonia": "Cerrado (fora Amazônia Legal)",
        "cerrado_amazonia": "Cerrado (Amazônia Legal)",
        "amazonia": "Amazônia",
    },
}

CLASSE_CORES = {
    "floresta_nativa": "#1a6b3c",
    "vegetacao_secundaria": "#5cb85c",
    "pastagem_agricultura": "#e8c441",
    "solo_exposto": "#c9513e",
    "agua": "#3a7fc2",
}

CLASSE_LABELS = {
    "floresta_nativa": "Floresta nativa",
    "vegetacao_secundaria": "Vegetação secundária",
    "pastagem_agricultura": "Pastagem / Agricultura",
    "solo_exposto": "Solo exposto",
    "agua": "Água",
}


# ============================================================================
# CSS CUSTOMIZADO
# ============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main-title {
    font-size: 2rem;
    font-weight: 700;
    color: #1a1a1a;
    margin-bottom: 0.2rem;
    letter-spacing: -0.5px;
}

.sub-title {
    font-size: 1rem;
    color: #888;
    margin-bottom: 2rem;
}

.metric-card {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    border: 1px solid #eee;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1a1a1a;
    line-height: 1.2;
}

.metric-label {
    font-size: 0.8rem;
    color: #888;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.excedente-box {
    background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
    border: 2px solid #4caf50;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
}

.deficit-box {
    background: linear-gradient(135deg, #fce4ec, #fff3e0);
    border: 2px solid #ef5350;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
}

.status-label {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.5rem;
}

.status-value {
    font-size: 2.5rem;
    font-weight: 700;
    line-height: 1.2;
}

.status-sub {
    font-size: 0.9rem;
    color: #666;
    margin-top: 0.3rem;
}

.bar-container {
    background: #eee;
    border-radius: 6px;
    height: 10px;
    margin-top: 4px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s ease;
}

.section-divider {
    border: none;
    border-top: 1px solid #eee;
    margin: 2rem 0;
}

div[data-testid="stSidebar"] {
    background: #fafafa;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# INICIALIZAÇÃO GEE
# ============================================================================

@st.cache_resource
def init_gee():
    """Inicializa o Google Earth Engine (roda uma vez)."""
    try:
        ee.Initialize(project='satellite-cra')
        return True
    except:
        try:
            ee.Authenticate()
            ee.Initialize(project='satellite-cra')
            return True
        except Exception as e:
            return False


# ============================================================================
# FUNÇÕES DE ANÁLISE (mesmo pipeline do Colab)
# ============================================================================

def obter_sentinel2(aoi, meses=12, max_nuvens=15):
    fim = datetime.now()
    inicio = fim - timedelta(days=meses * 30)
    bandas = ["B2", "B3", "B4", "B8", "B11", "B12"]
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(aoi)
           .filterDate(inicio.strftime("%Y-%m-%d"), fim.strftime("%Y-%m-%d"))
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_nuvens))
           .select(bandas))
    n = col.size().getInfo()
    if n == 0:
        raise ValueError("Nenhuma imagem Sentinel-2 encontrada.")
    return col.median().clip(aoi), n

def adicionar_indices(img):
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    evi = img.expression(
        "2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))",
        {"NIR": img.select("B8"), "RED": img.select("B4"), "BLUE": img.select("B2")}
    ).rename("EVI")
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    nbr = img.normalizedDifference(["B8", "B12"]).rename("NBR")
    return img.addBands([ndvi, evi, ndwi, nbr])

def classificar_ml(img, aoi):
    mb = (ee.Image("projects/mapbiomas-public/assets/brazil/lulc/collection9/"
                    "mapbiomas_collection90_integration_v1")
          .select("classification_2023"))
    remap_from = [1,3,4,5,6,49,10,11,12,14,15,18,19,20,21,
                  36,39,40,41,46,47,48,9,22,23,24,25,29,30,26,33,31]
    remap_to   = [1,1,2,2,2,2,2,2,3,3,3,3,3,3,3,
                  3,3,3,3,3,3,3,2,4,4,4,4,4,4,5,5,5]
    ref = mb.remap(remap_from, remap_to).rename("classe_ref")
    bandas = ["B2","B3","B4","B8","B11","B12","NDVI","EVI","NDWI","NBR"]
    amostras = (img.select(bandas).addBands(ref)
                .stratifiedSample(numPoints=300, classBand="classe_ref",
                                  region=aoi, scale=10, seed=42, geometries=True))
    rf = (ee.Classifier.smileRandomForest(50)
          .train(features=amostras, classProperty="classe_ref",
                 inputProperties=bandas))
    acc = (amostras.classify(rf)
           .errorMatrix("classe_ref", "classification").accuracy().getInfo())
    return img.select(bandas).classify(rf).rename("classificacao"), acc

def calcular_areas(classif, aoi):
    px = ee.Image.pixelArea().divide(10000)
    total = round(list(px.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi,
        scale=10, maxPixels=1e10).getInfo().values())[0], 2)
    nomes = {1: "floresta_nativa", 2: "vegetacao_secundaria",
             3: "pastagem_agricultura", 4: "solo_exposto", 5: "agua"}
    areas = {}
    for cid, nome in nomes.items():
        val = list(px.updateMask(classif.eq(cid)).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi,
            scale=10, maxPixels=1e10).getInfo().values())[0]
        areas[nome] = round(val or 0, 2)
    return {"area_total_ha": total, "classes": areas}

def calcular_cra(areas, bioma="mata_atlantica"):
    pct = CONFIG["reserva_legal"][bioma]
    total = areas["area_total_ha"]
    veg = areas["classes"]["floresta_nativa"] + areas["classes"]["vegetacao_secundaria"]
    rl = total * pct
    exc = veg - rl
    cra = max(0, exc)
    valor = cra * CONFIG["preco_cra_ha_ano"]
    return {
        "bioma": bioma, "pct_rl": pct,
        "area_total_ha": total,
        "veg_nativa_ha": round(veg, 2),
        "rl_exigida_ha": round(rl, 2),
        "excedente_ha": round(exc, 2),
        "tem_excedente": exc > 0,
        "cra_ha": round(cra, 2),
        "valor_anual_brl": round(valor, 2),
        "classes": areas["classes"],
    }


# ============================================================================
# FUNÇÕES DE EXTRAÇÃO SICAR
# ============================================================================

def extrair_sicar(uploaded_file):
    """Extrai e lê shapefiles de um ZIP do SICAR."""
    tmpdir = tempfile.mkdtemp()

    # Salva arquivo
    zip_path = os.path.join(tmpdir, "sicar.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Descompacta
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(tmpdir)

    # Descompacta ZIPs internos
    for zf in glob.glob(os.path.join(tmpdir, "**/*.zip"), recursive=True):
        if zf != zip_path:
            with zipfile.ZipFile(zf, 'r') as z:
                z.extractall(tmpdir)

    # Encontra shapefiles
    shps = glob.glob(os.path.join(tmpdir, "**/*.shp"), recursive=True)

    def find_shp(keywords):
        for s in shps:
            name = os.path.basename(s).lower()
            if any(k in name for k in keywords):
                return gpd.read_file(s).to_crs(epsg=4326)
        return None

    data = {
        "imovel": find_shp(["imovel", "area_do"]),
        "reserva": find_shp(["reserva", "legal"]),
        "vegetacao": find_shp(["cobertura", "vegetacao", "veg_nativa"]),
        "app": find_shp(["preservacao", "app"]),
    }

    # Extrai metadados
    meta = {}
    if data["imovel"] is not None:
        row = data["imovel"].iloc[0]
        for col in data["imovel"].columns:
            cl = col.lower()
            if 'area' in cl and 'num' in cl: meta["area_ha"] = row[col]
            elif cl == 'num_area': meta["area_ha"] = row[col]
            elif 'cod_imovel' in cl: meta["cod_car"] = row[col]
            elif 'municipio' in cl or 'nom_munic' in cl: meta["municipio"] = row[col]
            elif 'mod_fiscal' in cl: meta["modulos"] = row[col]

        if "area_ha" not in meta:
            meta["area_ha"] = round(data["imovel"].to_crs(epsg=32723).area.sum() / 10000, 2)

    if data["reserva"] is not None:
        meta["rl_ha"] = round(data["reserva"].to_crs(epsg=32723).area.sum() / 10000, 2)
    if data["vegetacao"] is not None:
        meta["veg_ha"] = round(data["vegetacao"].to_crs(epsg=32723).area.sum() / 10000, 2)
    if data["app"] is not None:
        meta["app_ha"] = round(data["app"].to_crs(epsg=32723).area.sum() / 10000, 2)

    # Limpa
    shutil.rmtree(tmpdir, ignore_errors=True)

    return data, meta


def detectar_bioma(lon, lat):
    """Detecta bioma pela localização (simplificado)."""
    if lat < -22 or lon > -44:
        return "mata_atlantica"
    elif lon < -50 and lat > -15:
        return "amazonia"
    else:
        return "cerrado_fora_amazonia"


# ============================================================================
# GERAR MAPA FOLIUM
# ============================================================================

def gerar_mapa(geom, resultado, img_indices=None, classificacao=None, aoi=None):
    """Gera mapa interativo com Folium."""
    centroid = geom.centroid
    m = folium.Map(
        location=[centroid.y, centroid.x],
        zoom_start=14,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
    )

    # Contorno da propriedade
    cor = "#4caf50" if resultado["tem_excedente"] else "#ef5350"
    coords_folium = [[c[1], c[0]] for c in geom.exterior.coords]
    folium.Polygon(
        locations=coords_folium,
        color=cor,
        weight=3,
        fill=True,
        fill_opacity=0.05,
        popup=f"Área: {resultado['area_total_ha']:.1f} ha",
    ).add_to(m)

    # Camada de classificação via GEE (se disponível)
    if classificacao is not None and aoi is not None:
        try:
            class_params = {
                "min": 1, "max": 5,
                "palette": ["#1a6b3c", "#5cb85c", "#e8c441", "#c9513e", "#3a7fc2"],
            }
            url = classificacao.getMapId(class_params)["tile_fetcher"].url_format
            folium.TileLayer(
                tiles=url, attr="Classificação ML",
                name="Cobertura vegetal", overlay=True,
            ).add_to(m)
        except:
            pass

    # Camada NDVI (se disponível)
    if img_indices is not None:
        try:
            ndvi_params = {
                "bands": ["NDVI"], "min": -0.1, "max": 0.9,
                "palette": ["#d73027","#fc8d59","#fee08b","#d9ef8b","#91cf60","#1a9850"],
            }
            url = img_indices.getMapId(ndvi_params)["tile_fetcher"].url_format
            folium.TileLayer(
                tiles=url, attr="NDVI",
                name="NDVI", overlay=True, show=False,
            ).add_to(m)
        except:
            pass

    folium.LayerControl().add_to(m)
    return m


# ============================================================================
# INTERFACE
# ============================================================================

# Header
st.markdown('<p class="main-title">🛰️ CRA Satellite Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Análise satelital para Cotas de Reserva Ambiental</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Configuração")

    metodo = st.radio(
        "Método de entrada",
        ["Upload SICAR (ZIP)", "Coordenadas manuais"],
        help="Baixe o ZIP de uma propriedade em car.gov.br"
    )

    bioma_opcao = st.selectbox(
        "Bioma",
        list(CONFIG["bioma_labels"].keys()),
        format_func=lambda x: CONFIG["bioma_labels"][x],
        help="Auto-detectado no upload SICAR, manual nas coordenadas"
    )

    preco = st.number_input(
        "Preço CRA (R$/ha/ano)",
        min_value=100, max_value=2000, value=500, step=50,
        help="Referência SFB: R$ 500/ha/ano"
    )
    CONFIG["preco_cra_ha_ano"] = preco

    st.markdown("---")
    st.markdown("### Sobre")
    st.markdown(
        "Ferramenta que usa imagens **Sentinel-2** (10m) "
        "e **Machine Learning** treinado com MapBiomas "
        "para identificar excedentes e déficits de "
        "Reserva Legal em propriedades rurais."
    )
    st.markdown(
        "**Fontes:** Copernicus/ESA, MapBiomas, SICAR"
    )

# Init GEE
gee_ok = init_gee()
if not gee_ok:
    st.error("Não foi possível conectar ao Google Earth Engine. Verifique a autenticação.")
    st.stop()

# ============================================================================
# MÉTODO 1: UPLOAD SICAR
# ============================================================================

if metodo == "Upload SICAR (ZIP)":
    st.markdown("### Upload do arquivo SICAR")
    st.markdown("Baixe o ZIP da propriedade em [car.gov.br](https://www.car.gov.br/publico/imoveis/index) e faça upload aqui.")

    uploaded = st.file_uploader("Arquivo ZIP do SICAR", type=["zip"])

    if uploaded and st.button("🛰️ Analisar propriedade", type="primary", use_container_width=True):

        with st.spinner("Extraindo dados do SICAR..."):
            data, meta = extrair_sicar(uploaded)

        if data["imovel"] is None:
            st.error("Shapefile de Área do Imóvel não encontrado no ZIP.")
            st.stop()

        # Mostra dados do CAR
        st.markdown("---")
        st.markdown("### Dados do CAR (declarados)")
        cols = st.columns(4)
        with cols[0]:
            st.metric("Área do imóvel", f"{meta.get('area_ha', 'N/A')} ha")
        with cols[1]:
            st.metric("Reserva Legal", f"{meta.get('rl_ha', 'N/A')} ha")
        with cols[2]:
            st.metric("Vegetação nativa", f"{meta.get('veg_ha', 'N/A')} ha")
        with cols[3]:
            st.metric("APP", f"{meta.get('app_ha', 'N/A')} ha")

        if meta.get("cod_car"):
            st.caption(f"CAR: {meta['cod_car']} · {meta.get('municipio', '')}")

        # Prepara geometria
        geom = data["imovel"].geometry.iloc[0]
        if geom.geom_type == 'MultiPolygon':
            geom = max(geom.geoms, key=lambda g: g.area)
        coords = [list(c) for c in geom.exterior.coords]
        aoi = ee.Geometry.Polygon([coords])

        # Detecta bioma
        centroid = geom.centroid
        bioma = detectar_bioma(centroid.x, centroid.y)

        # Análise satelital
        st.markdown("---")
        st.markdown("### Análise satelital")
        progress = st.progress(0, text="Buscando imagens Sentinel-2...")

        img, n_imgs = obter_sentinel2(aoi)
        progress.progress(25, text=f"{n_imgs} imagens encontradas. Calculando índices...")

        img = adicionar_indices(img)
        progress.progress(40, text="Classificando cobertura com Machine Learning...")

        classif, acc = classificar_ml(img, aoi)
        progress.progress(70, text=f"Acurácia: {acc:.1%}. Calculando áreas...")

        areas = calcular_areas(classif, aoi)
        resultado = calcular_cra(areas, bioma)
        progress.progress(100, text="Análise concluída!")

        # Resultado principal
        st.markdown("---")
        if resultado["tem_excedente"]:
            st.markdown(f"""
            <div class="excedente-box">
                <div class="status-label" style="color: #2e7d32;">✓ Potencial vendedor de CRA</div>
                <div class="status-value" style="color: #2e7d32;">{resultado['excedente_ha']:.1f} ha excedentes</div>
                <div class="status-sub">Valor estimado: <strong>R$ {resultado['valor_anual_brl']:,.0f}/ano</strong> · Projeção 10 anos: R$ {resultado['valor_anual_brl']*10:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            custo = abs(resultado["excedente_ha"]) * CONFIG["preco_cra_ha_ano"]
            st.markdown(f"""
            <div class="deficit-box">
                <div class="status-label" style="color: #c62828;">✗ Potencial comprador de CRA</div>
                <div class="status-value" style="color: #c62828;">{abs(resultado['excedente_ha']):.1f} ha de déficit</div>
                <div class="status-sub">Custo de regularização: <strong>R$ {custo:,.0f}/ano</strong></div>
            </div>
            """, unsafe_allow_html=True)

        # Métricas
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{resultado['area_total_ha']:.1f}</div>
                <div class="metric-label">Área total (ha)</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{resultado['veg_nativa_ha']:.1f}</div>
                <div class="metric-label">Vegetação nativa (ha)</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{resultado['rl_exigida_ha']:.1f}</div>
                <div class="metric-label">RL exigida (ha)</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{acc:.1%}</div>
                <div class="metric-label">Acurácia ML</div>
            </div>""", unsafe_allow_html=True)

        # Cobertura vegetal
        st.markdown("---")
        st.markdown("### Cobertura vegetal")
        for cls, val in resultado["classes"].items():
            if val > 0:
                pct = val / resultado["area_total_ha"] * 100
                cor = CLASSE_CORES.get(cls, "#999")
                label = CLASSE_LABELS.get(cls, cls)
                st.markdown(f"""
                <div style="margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                        <span>{label}</span>
                        <span><strong>{val:.1f} ha</strong> ({pct:.1f}%)</span>
                    </div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: {pct}%; background: {cor};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Comparação CAR vs Satélite
        if meta.get("veg_ha"):
            st.markdown("---")
            st.markdown("### Validação: CAR vs Satélite")
            diff = resultado["veg_nativa_ha"] - meta["veg_ha"]
            pct_err = abs(diff) / meta["veg_ha"] * 100

            v1, v2, v3 = st.columns(3)
            with v1:
                st.metric("CAR declarado", f"{meta['veg_ha']:.1f} ha")
            with v2:
                st.metric("Satélite (10m)", f"{resultado['veg_nativa_ha']:.1f} ha", f"{diff:+.1f} ha")
            with v3:
                if pct_err < 10:
                    st.metric("Status", "✓ Consistente", f"{pct_err:.1f}% erro")
                elif pct_err < 20:
                    st.metric("Status", "⚠ Verificar", f"{pct_err:.1f}% erro")
                else:
                    st.metric("Status", "✗ Divergente", f"{pct_err:.1f}% erro")

        # Mapa
        st.markdown("---")
        st.markdown("### Mapa da propriedade")
        mapa = gerar_mapa(geom, resultado, img, classif, aoi)
        st_folium(mapa, use_container_width=True, height=500)

        # Salvar no session state para batch
        if "historico" not in st.session_state:
            st.session_state.historico = []
        st.session_state.historico.append({
            "cod_car": meta.get("cod_car", "N/A"),
            "municipio": meta.get("municipio", "N/A"),
            **resultado
        })


# ============================================================================
# MÉTODO 2: COORDENADAS MANUAIS
# ============================================================================

elif metodo == "Coordenadas manuais":
    st.markdown("### Coordenadas da propriedade")
    st.markdown("Cole as coordenadas do polígono (uma por linha, formato: `longitude, latitude`)")
    st.markdown("Pegue em [geojson.io](https://geojson.io) ou no Google Maps (clique direito).")

    coords_text = st.text_area(
        "Coordenadas (lon, lat)",
        value="-49.74, -21.91\n-49.74, -21.89\n-49.73, -21.89\n-49.73, -21.90\n-49.74, -21.91",
        height=150,
    )

    if st.button("🛰️ Analisar propriedade", type="primary", use_container_width=True):
        # Parse coordenadas
        try:
            coords = []
            for line in coords_text.strip().split("\n"):
                parts = line.strip().split(",")
                lon, lat = float(parts[0].strip()), float(parts[1].strip())
                coords.append([lon, lat])
            aoi = ee.Geometry.Polygon([coords])
        except:
            st.error("Formato inválido. Use: longitude, latitude (uma por linha)")
            st.stop()

        # Análise
        progress = st.progress(0, text="Buscando imagens Sentinel-2...")

        img, n_imgs = obter_sentinel2(aoi)
        progress.progress(25, text=f"{n_imgs} imagens. Calculando índices...")

        img = adicionar_indices(img)
        progress.progress(40, text="Classificando com ML...")

        classif, acc = classificar_ml(img, aoi)
        progress.progress(70, text=f"Acurácia: {acc:.1%}. Calculando áreas...")

        areas = calcular_areas(classif, aoi)
        resultado = calcular_cra(areas, bioma_opcao)
        progress.progress(100, text="Concluído!")

        # Resultado
        st.markdown("---")
        if resultado["tem_excedente"]:
            st.markdown(f"""
            <div class="excedente-box">
                <div class="status-label" style="color: #2e7d32;">✓ Potencial vendedor de CRA</div>
                <div class="status-value" style="color: #2e7d32;">{resultado['excedente_ha']:.1f} ha excedentes</div>
                <div class="status-sub">Valor estimado: <strong>R$ {resultado['valor_anual_brl']:,.0f}/ano</strong></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            custo = abs(resultado["excedente_ha"]) * CONFIG["preco_cra_ha_ano"]
            st.markdown(f"""
            <div class="deficit-box">
                <div class="status-label" style="color: #c62828;">✗ Potencial comprador de CRA</div>
                <div class="status-value" style="color: #c62828;">{abs(resultado['excedente_ha']):.1f} ha de déficit</div>
                <div class="status-sub">Custo de regularização: <strong>R$ {custo:,.0f}/ano</strong></div>
            </div>
            """, unsafe_allow_html=True)

        # Métricas e cobertura
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Área total", f"{resultado['area_total_ha']:.1f} ha")
        with c2:
            st.metric("Veg. nativa", f"{resultado['veg_nativa_ha']:.1f} ha")
        with c3:
            st.metric("RL exigida", f"{resultado['rl_exigida_ha']:.1f} ha")
        with c4:
            st.metric("Acurácia ML", f"{acc:.1%}")

        st.markdown("---")
        st.markdown("### Cobertura vegetal")
        for cls, val in resultado["classes"].items():
            if val > 0:
                pct = val / resultado["area_total_ha"] * 100
                cor = CLASSE_CORES.get(cls, "#999")
                label = CLASSE_LABELS.get(cls, cls)
                st.markdown(f"""
                <div style="margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                        <span>{label}</span>
                        <span><strong>{val:.1f} ha</strong> ({pct:.1f}%)</span>
                    </div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: {pct}%; background: {cor};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Mapa
        from shapely.geometry import Polygon as ShapelyPolygon
        geom = ShapelyPolygon(coords)
        st.markdown("---")
        st.markdown("### Mapa")
        mapa = gerar_mapa(geom, resultado, img, classif, aoi)
        st_folium(mapa, use_container_width=True, height=500)


# ============================================================================
# HISTÓRICO DE ANÁLISES
# ============================================================================

if "historico" in st.session_state and len(st.session_state.historico) > 1:
    st.markdown("---")
    st.markdown("### Histórico de análises")
    df_hist = pd.DataFrame(st.session_state.historico)
    st.dataframe(df_hist[["cod_car", "municipio", "area_total_ha", "veg_nativa_ha",
                           "excedente_ha", "tem_excedente", "cra_ha", "valor_anual_brl"]],
                 use_container_width=True)

    csv = df_hist.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 Baixar CSV", csv, "analises_cra.csv", "text/csv")
