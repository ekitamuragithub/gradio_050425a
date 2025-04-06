# define a imagem base
FROM python:3.11

# define o diretório onde os comando seguintes serão executados
WORKDIR /code

# copia os arquivos de requirements.txt para o diretório de trabalho
COPY requirements.txt /code/requirements.txt

# instala as dependências do sistema operacional necessárias para o Flask
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["tail", "-f", "/dev/null"]

# expõe a porta 8080 para acesso externo
EXPOSE 8080

