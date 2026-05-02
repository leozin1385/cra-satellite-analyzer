[README.md](https://github.com/user-attachments/files/27309272/README.md)
# CRA Satellite Analyzer — Dashboard

Aplicação web para análise satelital de Cotas de Reserva Ambiental.

## Setup local (5 minutos)

```bash
pip install -r requirements.txt
earthengine authenticate
streamlit run app.py
```

O app abre em `localhost:8501`.

## Deploy gratuito no Streamlit Cloud

1. Suba este repositório no GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte seu repo e selecione `app.py`
4. Em **Secrets**, adicione suas credenciais do GEE:

```toml
[ee]
project = "satellite-cra"
service_account = "sua-service-account@projeto.iam.gserviceaccount.com"
private_key = "-----BEGIN PRIVATE KEY-----\n..."
```

Para gerar a service account:
- Google Cloud Console → IAM → Service Accounts → Create
- Baixe a chave JSON
- Cole o conteúdo no Secrets do Streamlit Cloud

## Como usar

1. Baixe o ZIP de uma propriedade em [car.gov.br](https://www.car.gov.br/publico/imoveis/index)
2. Faça upload no dashboard
3. Aguarde ~1 minuto pra análise
4. Veja o relatório com mapa, métricas e valor de CRA

## Estrutura

```
cra_dashboard/
├── app.py              # Aplicação Streamlit
├── requirements.txt    # Dependências
└── README.md           # Este arquivo
```
